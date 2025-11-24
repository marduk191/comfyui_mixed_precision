"""
SmoothQuant Implementation
Based on: "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models"

SmoothQuant migrates quantization difficulty from activations to weights using
mathematically equivalent per-channel scaling transformations.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from .quantization_utils import QuantizationConfig, quantize_tensor, dequantize_tensor


class SmoothQuantizer:
    """
    SmoothQuant: Smooth activation outliers by migrating difficulty to weights

    The key insight is: Y = (X * diag(s)^-1) * (diag(s) * W) = X' * W'
    where s is a smoothing factor that balances quantization difficulty
    """

    def __init__(
        self,
        config: QuantizationConfig,
        alpha: float = 0.5,
        calibration_samples: int = 512,
        auto_alpha: bool = False,
        preserve_zero_point: bool = True
    ):
        """
        Args:
            config: Quantization configuration
            alpha: Smoothing strength [0, 1]. 0 = all difficulty to weights,
                   1 = all difficulty to activations, 0.5 = balanced
            calibration_samples: Number of samples for calibration
            auto_alpha: Automatically search for optimal alpha
            preserve_zero_point: Preserve zero points in smoothing
        """
        self.config = config
        self.alpha = alpha
        self.calibration_samples = calibration_samples
        self.auto_alpha = auto_alpha
        self.preserve_zero_point = preserve_zero_point

        # Store statistics
        self.activation_ranges: Dict[str, torch.Tensor] = {}
        self.weight_ranges: Dict[str, torch.Tensor] = {}
        self.smoothing_factors: Dict[str, torch.Tensor] = {}

    def collect_statistics(
        self,
        module: nn.Module,
        name: str,
        inp: torch.Tensor
    ) -> None:
        """Collect activation range statistics for a module"""
        if isinstance(inp, tuple):
            inp = inp[0]

        # Calculate per-channel activation ranges
        if inp.dim() >= 2:
            # Reshape to [*, channels]
            if isinstance(module, nn.Linear):
                inp_flat = inp.view(-1, inp.shape[-1])
            elif isinstance(module, nn.Conv2d):
                # [N, C, H, W] -> [*, C]
                inp_flat = inp.permute(0, 2, 3, 1).reshape(-1, inp.shape[1])
            else:
                inp_flat = inp.view(-1, inp.shape[-1])

            # Calculate range per channel
            channel_max = inp_flat.abs().max(dim=0)[0]

            if name in self.activation_ranges:
                # Running maximum
                self.activation_ranges[name] = torch.maximum(
                    self.activation_ranges[name],
                    channel_max
                )
            else:
                self.activation_ranges[name] = channel_max

    def compute_smoothing_factors(
        self,
        layer: nn.Module,
        name: str,
        alpha: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute smoothing factors for a layer

        Args:
            layer: Layer to compute factors for
            name: Layer name
            alpha: Override default alpha

        Returns:
            Per-channel smoothing factors
        """
        if alpha is None:
            alpha = self.alpha

        weights = layer.weight.data

        # Calculate weight ranges per input channel
        if isinstance(layer, nn.Linear):
            # [out_features, in_features]
            weight_ranges = weights.abs().max(dim=0)[0]
        elif isinstance(layer, nn.Conv2d):
            # [out_channels, in_channels, K, K]
            weight_ranges = weights.abs().amax(dim=[0, 2, 3])
        else:
            raise ValueError(f"Unsupported layer type: {type(layer)}")

        # Get activation ranges
        if name in self.activation_ranges:
            act_ranges = self.activation_ranges[name]
        else:
            # Default: assume uniform activations
            act_ranges = torch.ones_like(weight_ranges)

        # Compute smoothing factors
        # s = act_range^alpha / weight_range^(1-alpha)
        act_ranges = act_ranges.clamp(min=1e-5)
        weight_ranges = weight_ranges.clamp(min=1e-5)

        smoothing = (act_ranges.pow(alpha) / weight_ranges.pow(1 - alpha))

        # Normalize to prevent scale drift
        smoothing = smoothing / smoothing.mean()

        return smoothing

    def search_optimal_alpha(
        self,
        layer: nn.Module,
        name: str,
        alpha_range: Tuple[float, float] = (0.3, 0.7),
        n_steps: int = 10
    ) -> float:
        """
        Search for optimal alpha that minimizes quantization error

        Args:
            layer: Layer to optimize
            name: Layer name
            alpha_range: Range of alpha values to search
            n_steps: Number of search steps

        Returns:
            Optimal alpha value
        """
        alphas = torch.linspace(alpha_range[0], alpha_range[1], n_steps)
        best_alpha = self.alpha
        best_error = float('inf')

        original_weight = layer.weight.data.clone()

        for alpha in alphas:
            # Compute smoothing with this alpha
            smoothing = self.compute_smoothing_factors(layer, name, alpha.item())

            # Apply smoothing to weights
            if isinstance(layer, nn.Linear):
                smoothed_weight = original_weight * smoothing.unsqueeze(0)
            elif isinstance(layer, nn.Conv2d):
                smoothed_weight = original_weight * smoothing.view(1, -1, 1, 1)
            else:
                continue

            # Quantize and dequantize
            q_weight, scale, zero = quantize_tensor(
                smoothed_weight,
                self.config,
                return_scale_zero=True
            )
            dq_weight = dequantize_tensor(
                q_weight, scale, zero, self.config, original_weight.dtype
            )

            # Unsmooth
            if isinstance(layer, nn.Linear):
                dq_weight = dq_weight / smoothing.unsqueeze(0)
            elif isinstance(layer, nn.Conv2d):
                dq_weight = dq_weight / smoothing.view(1, -1, 1, 1)

            # Calculate error
            error = ((original_weight - dq_weight) ** 2).mean().item()

            if error < best_error:
                best_error = error
                best_alpha = alpha.item()

        return best_alpha

    def smooth_layer(
        self,
        layer: nn.Module,
        name: str,
        next_layer: Optional[nn.Module] = None
    ) -> torch.Tensor:
        """
        Apply smoothing to a layer

        Args:
            layer: Layer to smooth
            name: Layer name
            next_layer: Next layer (to apply inverse smoothing)

        Returns:
            Smoothing factors used
        """
        # Compute or retrieve smoothing factors
        if name not in self.smoothing_factors:
            if self.auto_alpha:
                alpha = self.search_optimal_alpha(layer, name)
            else:
                alpha = self.alpha

            smoothing = self.compute_smoothing_factors(layer, name, alpha)
            self.smoothing_factors[name] = smoothing
        else:
            smoothing = self.smoothing_factors[name]

        # Apply smoothing to current layer's weights
        # W' = diag(s) * W
        if isinstance(layer, nn.Linear):
            layer.weight.data = layer.weight.data * smoothing.unsqueeze(0)
            if layer.bias is not None and next_layer is None:
                # Only scale bias if there's no next layer
                layer.bias.data = layer.bias.data  # Keep bias unchanged
        elif isinstance(layer, nn.Conv2d):
            layer.weight.data = layer.weight.data * smoothing.view(1, -1, 1, 1)
            if layer.bias is not None and next_layer is None:
                layer.bias.data = layer.bias.data  # Keep bias unchanged
        else:
            raise ValueError(f"Unsupported layer type: {type(layer)}")

        # Apply inverse smoothing to next layer's weights (if exists)
        # W_next' = W_next * diag(s)^-1
        if next_layer is not None:
            inv_smoothing = 1.0 / smoothing.clamp(min=1e-5)

            if isinstance(next_layer, nn.Linear):
                next_layer.weight.data = next_layer.weight.data * inv_smoothing.unsqueeze(1)
            elif isinstance(next_layer, nn.Conv2d):
                # For conv, the inverse smoothing applies to output channels of previous layer
                # which correspond to input channels of next layer
                next_layer.weight.data = next_layer.weight.data * inv_smoothing.view(1, -1, 1, 1)

        return smoothing

    def apply_to_model(
        self,
        model: nn.Module,
        layer_pairs: Optional[List[Tuple[str, str]]] = None,
        inplace: bool = True
    ) -> nn.Module:
        """
        Apply SmoothQuant to entire model

        Args:
            model: Model to apply smoothing
            layer_pairs: List of (layer_name, next_layer_name) pairs
            inplace: Modify model in-place

        Returns:
            Smoothed model
        """
        if not inplace:
            import copy
            model = copy.deepcopy(model)

        # Auto-detect layer pairs if not provided
        if layer_pairs is None:
            layer_pairs = self._detect_layer_pairs(model)

        # Apply smoothing to each pair
        for layer_name, next_layer_name in layer_pairs:
            layer = self._get_module_by_name(model, layer_name)
            next_layer = self._get_module_by_name(model, next_layer_name) if next_layer_name else None

            if layer is not None:
                try:
                    smoothing = self.smooth_layer(layer, layer_name, next_layer)
                    print(f"Applied smoothing to {layer_name}, factors range: "
                          f"[{smoothing.min().item():.3f}, {smoothing.max().item():.3f}]")
                except Exception as e:
                    print(f"Warning: Failed to smooth layer {layer_name}: {e}")

        return model

    def _detect_layer_pairs(self, model: nn.Module) -> List[Tuple[str, Optional[str]]]:
        """Automatically detect layer pairs for smoothing"""
        layers = []
        layer_names = []

        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                layers.append(module)
                layer_names.append(name)

        # Create pairs of consecutive layers
        pairs = []
        for i in range(len(layers) - 1):
            pairs.append((layer_names[i], layer_names[i + 1]))

        # Add last layer without pair
        if layers:
            pairs.append((layer_names[-1], None))

        return pairs

    def _get_module_by_name(self, model: nn.Module, name: str) -> Optional[nn.Module]:
        """Get module by its name"""
        try:
            parts = name.split('.')
            module = model
            for part in parts:
                module = getattr(module, part)
            return module
        except AttributeError:
            return None

    def quantize_with_smoothing(
        self,
        model: nn.Module,
        layer_config: Optional[Dict[str, QuantizationConfig]] = None
    ) -> nn.Module:
        """
        Apply SmoothQuant and then quantize the model

        Args:
            model: Model to quantize
            layer_config: Optional per-layer quantization config

        Returns:
            Quantized model with smoothing applied
        """
        # First, apply smoothing
        model = self.apply_to_model(model, inplace=True)

        # Then quantize each layer
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                config = layer_config.get(name, self.config) if layer_config else self.config

                try:
                    # Quantize weights
                    quantized, scale, zero_point = quantize_tensor(
                        module.weight.data,
                        config,
                        return_scale_zero=True
                    )

                    # Store quantization parameters
                    module.register_buffer('quantized_weight', quantized)
                    module.register_buffer('weight_scale', scale)
                    if zero_point is not None:
                        module.register_buffer('weight_zero_point', zero_point)

                    # Store smoothing factors
                    if name in self.smoothing_factors:
                        module.register_buffer('smoothing_factors', self.smoothing_factors[name])

                except Exception as e:
                    print(f"Warning: Failed to quantize layer {name}: {e}")

        return model


def register_smoothquant_hooks(
    model: nn.Module,
    quantizer: SmoothQuantizer
) -> List:
    """
    Register forward hooks to collect statistics for SmoothQuant

    Args:
        model: Model to instrument
        quantizer: SmoothQuant quantizer instance

    Returns:
        List of hook handles
    """
    hooks = []

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            hook = module.register_forward_pre_hook(
                lambda m, inp, n=name: quantizer.collect_statistics(m, n, inp)
            )
            hooks.append(hook)

    return hooks
