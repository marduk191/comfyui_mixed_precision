"""
SVDQ (Singular Value Decomposition Quantization) Implementation

SVDQ decomposes weight matrices using SVD and applies different quantization
levels to components based on their importance (singular values).
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from .quantization_utils import QuantizationConfig, PrecisionMode, quantize_tensor, dequantize_tensor


class SVDQConfig:
    """Configuration for SVDQ"""
    def __init__(
        self,
        rank_ratio: float = 0.9,
        high_precision_ratio: float = 0.1,
        high_precision_mode: PrecisionMode = PrecisionMode.FP16,
        low_precision_mode: PrecisionMode = PrecisionMode.INT4,
        use_adaptive_rank: bool = True,
        energy_threshold: float = 0.95
    ):
        """
        Args:
            rank_ratio: Ratio of singular values to keep (0-1)
            high_precision_ratio: Ratio of components to keep in high precision
            high_precision_mode: Precision for important components
            low_precision_mode: Precision for less important components
            use_adaptive_rank: Automatically determine rank based on energy
            energy_threshold: Energy threshold for adaptive rank (0-1)
        """
        self.rank_ratio = rank_ratio
        self.high_precision_ratio = high_precision_ratio
        self.high_precision_mode = high_precision_mode
        self.low_precision_mode = low_precision_mode
        self.use_adaptive_rank = use_adaptive_rank
        self.energy_threshold = energy_threshold


class SVDQuantizer:
    """
    Singular Value Decomposition Quantization

    Decomposes weight matrices as W = U @ S @ V^T and applies mixed precision
    quantization based on singular value importance.
    """

    def __init__(self, config: SVDQConfig):
        """
        Args:
            config: SVDQ configuration
        """
        self.config = config
        self.svd_cache: Dict[str, Tuple] = {}

    def compute_adaptive_rank(
        self,
        singular_values: torch.Tensor,
        energy_threshold: float
    ) -> int:
        """
        Compute adaptive rank based on energy preservation

        Args:
            singular_values: Singular values from SVD
            energy_threshold: Minimum energy to preserve (0-1)

        Returns:
            Optimal rank
        """
        # Calculate cumulative energy
        energy = singular_values ** 2
        total_energy = energy.sum()
        cumulative_energy = torch.cumsum(energy, dim=0)
        energy_ratio = cumulative_energy / total_energy

        # Find minimum rank that preserves threshold energy
        rank = torch.searchsorted(energy_ratio, energy_threshold).item() + 1
        rank = min(rank, len(singular_values))

        return rank

    def decompose_weight(
        self,
        weight: torch.Tensor,
        compute_rank: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Decompose weight matrix using SVD

        Args:
            weight: Weight tensor [out_features, in_features]
            compute_rank: Whether to compute adaptive rank

        Returns:
            Tuple of (U, S, V, rank)
        """
        # Reshape weight for SVD if needed
        original_shape = weight.shape
        if weight.dim() > 2:
            # For Conv layers, reshape to 2D
            weight_2d = weight.view(weight.shape[0], -1)
        else:
            weight_2d = weight

        # Perform SVD
        U, S, Vh = torch.linalg.svd(weight_2d, full_matrices=False)

        # Determine rank
        if compute_rank and self.config.use_adaptive_rank:
            rank = self.compute_adaptive_rank(S, self.config.energy_threshold)
        else:
            rank = int(len(S) * self.config.rank_ratio)
            rank = max(1, min(rank, len(S)))

        return U, S, Vh, rank

    def quantize_components(
        self,
        U: torch.Tensor,
        S: torch.Tensor,
        Vh: torch.Tensor,
        rank: int
    ) -> Dict[str, any]:
        """
        Quantize SVD components with mixed precision

        Args:
            U: Left singular vectors
            S: Singular values
            Vh: Right singular vectors
            rank: Rank to use

        Returns:
            Dictionary with quantized components and metadata
        """
        # Truncate to rank
        U_trunc = U[:, :rank]
        S_trunc = S[:rank]
        Vh_trunc = Vh[:rank, :]

        # Determine split point for high/low precision
        high_rank = max(1, int(rank * self.config.high_precision_ratio))

        # Split components
        U_high = U_trunc[:, :high_rank]
        U_low = U_trunc[:, high_rank:]
        S_high = S_trunc[:high_rank]
        S_low = S_trunc[high_rank:]
        Vh_high = Vh_trunc[:high_rank, :]
        Vh_low = Vh_trunc[high_rank:, :]

        # Create configs for each precision
        high_config = QuantizationConfig(precision=self.config.high_precision_mode)
        low_config = QuantizationConfig(precision=self.config.low_precision_mode)

        # Quantize high-precision components
        U_high_q, U_high_scale, U_high_zero = quantize_tensor(
            U_high, high_config, return_scale_zero=True
        )
        Vh_high_q, Vh_high_scale, Vh_high_zero = quantize_tensor(
            Vh_high, high_config, return_scale_zero=True
        )

        # Quantize low-precision components if they exist
        if U_low.numel() > 0:
            U_low_q, U_low_scale, U_low_zero = quantize_tensor(
                U_low, low_config, return_scale_zero=True
            )
            Vh_low_q, Vh_low_scale, Vh_low_zero = quantize_tensor(
                Vh_low, low_config, return_scale_zero=True
            )
        else:
            U_low_q = U_low_scale = U_low_zero = None
            Vh_low_q = Vh_low_scale = Vh_low_zero = None

        return {
            # High precision components
            'U_high': U_high_q,
            'U_high_scale': U_high_scale,
            'U_high_zero': U_high_zero,
            'S_high': S_high,  # Keep singular values in full precision
            'Vh_high': Vh_high_q,
            'Vh_high_scale': Vh_high_scale,
            'Vh_high_zero': Vh_high_zero,
            # Low precision components
            'U_low': U_low_q,
            'U_low_scale': U_low_scale,
            'U_low_zero': U_low_zero,
            'S_low': S_low,
            'Vh_low': Vh_low_q,
            'Vh_low_scale': Vh_low_scale,
            'Vh_low_zero': Vh_low_zero,
            # Metadata
            'rank': rank,
            'high_rank': high_rank,
            'original_shape': None  # Will be set by caller
        }

    def reconstruct_weight(
        self,
        quantized_components: Dict[str, any],
        target_dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        Reconstruct weight from quantized SVD components

        Args:
            quantized_components: Dictionary with quantized components
            target_dtype: Target dtype for reconstruction

        Returns:
            Reconstructed weight tensor
        """
        high_config = QuantizationConfig(precision=self.config.high_precision_mode)
        low_config = QuantizationConfig(precision=self.config.low_precision_mode)

        # Dequantize high precision components
        U_high = dequantize_tensor(
            quantized_components['U_high'],
            quantized_components['U_high_scale'],
            quantized_components['U_high_zero'],
            high_config,
            target_dtype
        )
        Vh_high = dequantize_tensor(
            quantized_components['Vh_high'],
            quantized_components['Vh_high_scale'],
            quantized_components['Vh_high_zero'],
            high_config,
            target_dtype
        )
        S_high = quantized_components['S_high'].to(target_dtype)

        # Reconstruct high precision part: U_high @ diag(S_high) @ Vh_high
        weight_high = U_high @ torch.diag(S_high) @ Vh_high

        # Reconstruct low precision part if exists
        if quantized_components['U_low'] is not None:
            U_low = dequantize_tensor(
                quantized_components['U_low'],
                quantized_components['U_low_scale'],
                quantized_components['U_low_zero'],
                low_config,
                target_dtype
            )
            Vh_low = dequantize_tensor(
                quantized_components['Vh_low'],
                quantized_components['Vh_low_scale'],
                quantized_components['Vh_low_zero'],
                low_config,
                target_dtype
            )
            S_low = quantized_components['S_low'].to(target_dtype)

            weight_low = U_low @ torch.diag(S_low) @ Vh_low
            weight = weight_high + weight_low
        else:
            weight = weight_high

        # Reshape if needed
        if quantized_components['original_shape'] is not None:
            weight = weight.view(quantized_components['original_shape'])

        return weight

    def quantize_layer(
        self,
        layer: nn.Module,
        name: str
    ) -> Dict[str, any]:
        """
        Quantize a layer using SVDQ

        Args:
            layer: Layer to quantize
            name: Layer name

        Returns:
            Dictionary with quantized components
        """
        if not hasattr(layer, 'weight'):
            raise ValueError(f"Layer {name} does not have weights")

        weight = layer.weight.data
        original_shape = weight.shape

        # Decompose weight
        U, S, Vh, rank = self.decompose_weight(weight)

        # Cache SVD for analysis
        self.svd_cache[name] = (U, S, Vh, rank)

        # Quantize components
        quantized = self.quantize_components(U, S, Vh, rank)
        quantized['original_shape'] = original_shape

        return quantized

    def apply_to_model(
        self,
        model: nn.Module,
        layer_names: Optional[List[str]] = None,
        inplace: bool = False
    ) -> nn.Module:
        """
        Apply SVDQ to model layers

        Args:
            model: Model to quantize
            layer_names: Specific layers to quantize (None = all Linear/Conv)
            inplace: Modify model in-place

        Returns:
            Model with SVDQ applied
        """
        if not inplace:
            import copy
            model = copy.deepcopy(model)

        for name, module in model.named_modules():
            # Skip if layer_names specified and this layer not in list
            if layer_names is not None and name not in layer_names:
                continue

            if isinstance(module, (nn.Linear, nn.Conv2d)):
                try:
                    # Quantize layer
                    quantized = self.quantize_layer(module, name)

                    # Store quantized components as buffers
                    for key, value in quantized.items():
                        if isinstance(value, torch.Tensor):
                            module.register_buffer(f'svdq_{key}', value)
                        elif key == 'original_shape':
                            # Store shape as a list (can't register non-tensor)
                            module.svdq_original_shape = value
                        else:
                            setattr(module, f'svdq_{key}', value)

                    print(f"Applied SVDQ to {name}: rank={quantized['rank']}, "
                          f"high_rank={quantized['high_rank']}")

                except Exception as e:
                    print(f"Warning: Failed to apply SVDQ to layer {name}: {e}")

        return model

    def analyze_compression(self, layer_name: str) -> Dict[str, float]:
        """
        Analyze compression statistics for a layer

        Args:
            layer_name: Name of layer to analyze

        Returns:
            Dictionary with compression metrics
        """
        if layer_name not in self.svd_cache:
            raise ValueError(f"Layer {layer_name} not found in SVD cache")

        U, S, Vh, rank = self.svd_cache[layer_name]

        # Calculate compression metrics
        original_params = U.shape[0] * Vh.shape[1]
        compressed_params = (U.shape[0] + Vh.shape[1]) * rank + rank  # U + Vh + S

        compression_ratio = original_params / compressed_params

        # Energy preservation
        total_energy = (S ** 2).sum().item()
        kept_energy = (S[:rank] ** 2).sum().item()
        energy_ratio = kept_energy / total_energy

        # Singular value analysis
        high_rank = max(1, int(rank * self.config.high_precision_ratio))
        high_energy = (S[:high_rank] ** 2).sum().item()
        high_energy_ratio = high_energy / total_energy

        return {
            'compression_ratio': compression_ratio,
            'energy_preserved': energy_ratio,
            'high_precision_energy': high_energy_ratio,
            'rank': rank,
            'high_rank': high_rank,
            'original_params': original_params,
            'compressed_params': compressed_params
        }


def create_svdq_layer(quantized_components: Dict[str, any], layer_type: str = 'linear') -> nn.Module:
    """
    Create a specialized layer that uses SVDQ components

    Args:
        quantized_components: Dictionary with quantized SVD components
        layer_type: Type of layer ('linear' or 'conv2d')

    Returns:
        Custom layer with SVDQ forward pass
    """
    if layer_type == 'linear':
        return SVDQLinear(quantized_components)
    else:
        raise NotImplementedError(f"SVDQ layer type {layer_type} not implemented")


class SVDQLinear(nn.Module):
    """Custom Linear layer using SVDQ components"""

    def __init__(self, quantized_components: Dict[str, any]):
        super().__init__()

        # Register all components as buffers
        for key, value in quantized_components.items():
            if isinstance(value, torch.Tensor):
                self.register_buffer(key, value)
            else:
                setattr(self, key, value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using quantized SVD components"""
        # This is a placeholder - in practice, you'd want to optimize this
        # by keeping components in quantized form and dequantizing on-the-fly
        weight = self._reconstruct_weight()

        return torch.nn.functional.linear(x, weight)

    def _reconstruct_weight(self) -> torch.Tensor:
        """Reconstruct weight from quantized components"""
        # This should be optimized in practice
        quantizer = SVDQuantizer(SVDQConfig())
        components = {k: getattr(self, k) for k in dir(self) if k.startswith('svdq_') or k in [
            'U_high', 'U_high_scale', 'U_high_zero', 'S_high', 'Vh_high', 'Vh_high_scale', 'Vh_high_zero',
            'U_low', 'U_low_scale', 'U_low_zero', 'S_low', 'Vh_low', 'Vh_low_scale', 'Vh_low_zero',
            'rank', 'high_rank', 'original_shape'
        ]}
        return quantizer.reconstruct_weight(components)
