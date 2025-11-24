"""
AWQ (Activation-aware Weight Quantization) Implementation
Based on: "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from .quantization_utils import QuantizationConfig, quantize_tensor, dequantize_tensor


class AWQQuantizer:
    """
    Activation-aware Weight Quantization

    AWQ protects salient weights by searching for optimal per-channel scaling
    that minimizes quantization error based on activation magnitudes.
    """

    def __init__(
        self,
        config: QuantizationConfig,
        n_samples: int = 128,
        seqlen: int = 512,
        auto_scale: bool = True,
        mse_range: bool = True,
        percentile: float = 0.01
    ):
        """
        Args:
            config: Quantization configuration
            n_samples: Number of calibration samples
            seqlen: Sequence length for calibration
            auto_scale: Enable automatic scaling
            mse_range: Use MSE for finding optimal scale range
            percentile: Percentile for salient weight detection
        """
        self.config = config
        self.n_samples = n_samples
        self.seqlen = seqlen
        self.auto_scale = auto_scale
        self.mse_range = mse_range
        self.percentile = percentile

        # Store activation statistics
        self.activation_stats: Dict[str, torch.Tensor] = {}
        self.weight_scales: Dict[str, torch.Tensor] = {}

    def collect_activations(
        self,
        module: nn.Module,
        name: str,
        inp: torch.Tensor
    ) -> None:
        """Collect activation statistics for a module"""
        if isinstance(inp, tuple):
            inp = inp[0]

        # Calculate activation magnitude
        if inp.dim() >= 2:
            # Average over batch and sequence dimensions
            act_magnitude = inp.abs().mean(dim=list(range(inp.dim() - 1)))
        else:
            act_magnitude = inp.abs()

        if name in self.activation_stats:
            # Running average
            self.activation_stats[name] = (
                0.9 * self.activation_stats[name] + 0.1 * act_magnitude
            )
        else:
            self.activation_stats[name] = act_magnitude

    def compute_scaling_factors(
        self,
        weights: torch.Tensor,
        activations: torch.Tensor,
        group_size: int = 128
    ) -> torch.Tensor:
        """
        Compute optimal per-channel scaling factors

        Args:
            weights: Weight tensor [out_features, in_features]
            activations: Activation magnitudes [in_features]
            group_size: Group size for quantization

        Returns:
            Scaling factors per channel
        """
        # Identify salient channels based on activation magnitude
        threshold = torch.quantile(activations, 1.0 - self.percentile)
        salient_mask = activations > threshold

        if self.auto_scale:
            # Search for optimal scaling that minimizes quantization error
            scales = self._search_optimal_scale(weights, activations, salient_mask)
        else:
            # Use activation magnitude as scale
            scales = activations.clamp(min=1e-5)
            scales = scales / scales.mean()

        return scales

    def _search_optimal_scale(
        self,
        weights: torch.Tensor,
        activations: torch.Tensor,
        salient_mask: torch.Tensor,
        n_grid: int = 20
    ) -> torch.Tensor:
        """
        Search for optimal scaling factors using grid search

        Args:
            weights: Weight tensor
            activations: Activation magnitudes
            salient_mask: Boolean mask for salient channels
            n_grid: Number of grid points to search

        Returns:
            Optimal scaling factors
        """
        out_features = weights.shape[0]
        in_features = weights.shape[1]

        # Initialize scales
        scales = torch.ones(in_features, device=weights.device, dtype=weights.dtype)

        if not salient_mask.any():
            return scales

        # Grid search for optimal scale
        if self.mse_range:
            # Search scale range based on MSE
            scale_range = torch.linspace(0.5, 2.0, n_grid, device=weights.device)
        else:
            # Use activation-based range
            act_scale = activations / activations.mean()
            scale_range = torch.linspace(
                act_scale.min().item(),
                act_scale.max().item(),
                n_grid,
                device=weights.device
            )

        best_error = float('inf')
        best_scales = scales.clone()

        for scale_val in scale_range:
            # Apply candidate scale to salient channels
            test_scales = scales.clone()
            test_scales[salient_mask] = scale_val

            # Scale weights
            scaled_weights = weights * test_scales.unsqueeze(0)

            # Quantize and dequantize
            q_weights, scale, zero = quantize_tensor(
                scaled_weights,
                self.config,
                return_scale_zero=True
            )
            dq_weights = dequantize_tensor(
                q_weights, scale, zero, self.config, weights.dtype
            )

            # Unscale
            dq_weights = dq_weights / test_scales.unsqueeze(0)

            # Calculate error on salient channels
            error = ((weights - dq_weights)[:, salient_mask] ** 2).mean()

            if error < best_error:
                best_error = error
                best_scales = test_scales.clone()

        return best_scales

    def quantize_layer(
        self,
        layer: nn.Module,
        name: str,
        use_cached_activations: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Quantize a layer using AWQ

        Args:
            layer: Layer to quantize (typically Linear or Conv)
            name: Layer name
            use_cached_activations: Use cached activation statistics

        Returns:
            Tuple of (quantized_weights, scale, zero_point, weight_scales)
        """
        if not hasattr(layer, 'weight'):
            raise ValueError(f"Layer {name} does not have weights")

        weights = layer.weight.data

        # Get activation statistics
        if use_cached_activations and name in self.activation_stats:
            activations = self.activation_stats[name]
        else:
            # Use uniform activation if no stats available
            activations = torch.ones(weights.shape[1], device=weights.device)

        # Compute scaling factors
        weight_scales = self.compute_scaling_factors(
            weights, activations, self.config.group_size
        )

        # Apply scales to weights
        scaled_weights = weights * weight_scales.unsqueeze(0)

        # Quantize scaled weights
        quantized, scale, zero_point = quantize_tensor(
            scaled_weights,
            self.config,
            return_scale_zero=True
        )

        # Store scales for later use
        self.weight_scales[name] = weight_scales

        return quantized, scale, zero_point, weight_scales

    def apply_to_model(
        self,
        model: nn.Module,
        layer_config: Optional[Dict[str, QuantizationConfig]] = None,
        inplace: bool = False
    ) -> nn.Module:
        """
        Apply AWQ quantization to entire model

        Args:
            model: Model to quantize
            layer_config: Optional per-layer quantization config
            inplace: Modify model in-place

        Returns:
            Quantized model
        """
        if not inplace:
            model = type(model)(**model.config.to_dict())
            model.load_state_dict(model.state_dict())

        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Get config for this layer
                config = layer_config.get(name, self.config) if layer_config else self.config

                # Quantize layer
                try:
                    quantized, scale, zero_point, weight_scales = self.quantize_layer(
                        module, name, use_cached_activations=True
                    )

                    # Store quantization parameters as buffer
                    module.register_buffer('quantized_weight', quantized)
                    module.register_buffer('weight_scale', scale)
                    if zero_point is not None:
                        module.register_buffer('weight_zero_point', zero_point)
                    module.register_buffer('awq_scales', weight_scales)

                except Exception as e:
                    print(f"Warning: Failed to quantize layer {name}: {e}")
                    continue

        return model


def prepare_calibration_data(
    data_loader,
    n_samples: int = 128,
    device: str = 'cpu'
) -> List[torch.Tensor]:
    """
    Prepare calibration data for AWQ

    Args:
        data_loader: DataLoader with calibration samples
        n_samples: Number of samples to collect
        device: Device to store samples

    Returns:
        List of calibration tensors
    """
    calibration_data = []

    for i, batch in enumerate(data_loader):
        if i >= n_samples:
            break

        if isinstance(batch, (list, tuple)):
            batch = batch[0]

        calibration_data.append(batch.to(device))

    return calibration_data


def register_activation_hooks(
    model: nn.Module,
    quantizer: AWQQuantizer
) -> List:
    """
    Register forward hooks to collect activation statistics

    Args:
        model: Model to instrument
        quantizer: AWQ quantizer instance

    Returns:
        List of hook handles
    """
    hooks = []

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            hook = module.register_forward_pre_hook(
                lambda m, inp, n=name: quantizer.collect_activations(m, n, inp)
            )
            hooks.append(hook)

    return hooks
