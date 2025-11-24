"""
Core quantization utilities for mixed precision models
Supports 4-bit to 32-bit quantization with various methods
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum


class PrecisionMode(Enum):
    """Supported precision modes"""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"
    NF4 = "nf4"  # 4-bit NormalFloat
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"


class QuantizationConfig:
    """Configuration for quantization operations"""
    def __init__(
        self,
        precision: PrecisionMode = PrecisionMode.FP32,
        group_size: int = 128,
        symmetric: bool = True,
        clip_ratio: float = 1.0,
        use_double_quant: bool = False,
    ):
        self.precision = precision
        self.group_size = group_size
        self.symmetric = symmetric
        self.clip_ratio = clip_ratio
        self.use_double_quant = use_double_quant


def get_nf4_quantization_levels():
    """
    Get NormalFloat 4-bit quantization levels
    Based on the distribution of normal distribution quantiles
    """
    return torch.tensor([
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
    ])


def quantize_tensor(
    tensor: torch.Tensor,
    config: QuantizationConfig,
    return_scale_zero: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
    """
    Quantize a tensor to specified precision

    Args:
        tensor: Input tensor to quantize
        config: Quantization configuration
        return_scale_zero: If True, return (quantized, scale, zero_point)

    Returns:
        Quantized tensor or tuple of (quantized, scale, zero_point)
    """
    if config.precision == PrecisionMode.FP32:
        return (tensor, torch.ones(1), None) if return_scale_zero else tensor

    elif config.precision == PrecisionMode.FP16:
        result = tensor.half()
        return (result, torch.ones(1), None) if return_scale_zero else result

    elif config.precision == PrecisionMode.BF16:
        result = tensor.bfloat16()
        return (result, torch.ones(1), None) if return_scale_zero else result

    elif config.precision in [PrecisionMode.INT8, PrecisionMode.INT4]:
        return quantize_int(tensor, config, return_scale_zero)

    elif config.precision == PrecisionMode.NF4:
        return quantize_nf4(tensor, config, return_scale_zero)

    elif config.precision in [PrecisionMode.FP8_E4M3, PrecisionMode.FP8_E5M2]:
        return quantize_fp8(tensor, config, return_scale_zero)

    else:
        raise ValueError(f"Unsupported precision mode: {config.precision}")


def quantize_int(
    tensor: torch.Tensor,
    config: QuantizationConfig,
    return_scale_zero: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Quantize tensor to integer format (INT4 or INT8)"""
    n_bits = 4 if config.precision == PrecisionMode.INT4 else 8

    # Determine quantization range
    if config.symmetric:
        qmax = 2 ** (n_bits - 1) - 1
        qmin = -qmax
    else:
        qmax = 2 ** n_bits - 1
        qmin = 0

    # Group-wise quantization
    original_shape = tensor.shape
    if config.group_size > 0 and tensor.numel() > config.group_size:
        tensor_flat = tensor.flatten()
        n_groups = (tensor_flat.numel() + config.group_size - 1) // config.group_size

        # Pad to group size
        pad_size = n_groups * config.group_size - tensor_flat.numel()
        if pad_size > 0:
            tensor_flat = torch.cat([tensor_flat, torch.zeros(pad_size, dtype=tensor.dtype, device=tensor.device)])

        tensor_grouped = tensor_flat.view(n_groups, config.group_size)

        # Calculate scales per group
        if config.symmetric:
            abs_max = tensor_grouped.abs().max(dim=1, keepdim=True)[0]
            abs_max = abs_max * config.clip_ratio
            abs_max = torch.clamp(abs_max, min=1e-8)
            scale = abs_max / qmax
            zero_point = torch.zeros_like(scale, dtype=torch.int8)
        else:
            min_val = tensor_grouped.min(dim=1, keepdim=True)[0]
            max_val = tensor_grouped.max(dim=1, keepdim=True)[0]
            scale = (max_val - min_val) / (qmax - qmin)
            scale = torch.clamp(scale, min=1e-8)
            zero_point = qmin - (min_val / scale).round()

        # Quantize
        quantized = torch.clamp(
            (tensor_grouped / scale + zero_point).round(),
            qmin, qmax
        ).to(torch.int8)

        # Remove padding and reshape
        quantized = quantized.flatten()[:original_shape.numel()].view(original_shape)

    else:
        # Full tensor quantization
        if config.symmetric:
            abs_max = tensor.abs().max() * config.clip_ratio
            abs_max = torch.clamp(abs_max, min=1e-8)
            scale = abs_max / qmax
            zero_point = torch.zeros(1, dtype=torch.int8, device=tensor.device)
        else:
            min_val = tensor.min()
            max_val = tensor.max()
            scale = (max_val - min_val) / (qmax - qmin)
            scale = torch.clamp(scale, min=1e-8)
            zero_point = (qmin - (min_val / scale).round()).to(torch.int8)

        quantized = torch.clamp(
            (tensor / scale + zero_point).round(),
            qmin, qmax
        ).to(torch.int8)

    if return_scale_zero:
        return quantized, scale, zero_point
    return quantized


def quantize_nf4(
    tensor: torch.Tensor,
    config: QuantizationConfig,
    return_scale_zero: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, None]]:
    """Quantize tensor to NF4 (4-bit NormalFloat)"""
    nf4_levels = get_nf4_quantization_levels().to(tensor.device)

    # Normalize to [-1, 1] range
    abs_max = tensor.abs().max()
    abs_max = torch.clamp(abs_max, min=1e-8)
    normalized = tensor / abs_max

    # Find closest NF4 level for each value
    distances = torch.abs(normalized.unsqueeze(-1) - nf4_levels)
    quantized_indices = torch.argmin(distances, dim=-1)

    # Convert to int4 representation (stored as int8)
    quantized = quantized_indices.to(torch.int8)
    scale = abs_max

    if return_scale_zero:
        return quantized, scale.unsqueeze(0), None
    return quantized


def quantize_fp8(
    tensor: torch.Tensor,
    config: QuantizationConfig,
    return_scale_zero: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, None]]:
    """Quantize tensor to FP8 format"""
    # FP8 quantization using torch's native support or manual implementation
    # E4M3: 1 sign, 4 exponent, 3 mantissa bits
    # E5M2: 1 sign, 5 exponent, 2 mantissa bits

    if config.precision == PrecisionMode.FP8_E4M3:
        # E4M3 range: ~[-448, 448]
        max_val = 448.0
    else:  # E5M2
        # E5M2 range: ~[-57344, 57344]
        max_val = 57344.0

    # Scale to FP8 range
    abs_max = tensor.abs().max()
    abs_max = torch.clamp(abs_max, min=1e-8)
    scale = abs_max / max_val

    scaled = tensor / scale

    # Simulate FP8 by quantizing to reduced precision
    # (In practice, you'd use hardware FP8 or a proper FP8 library)
    quantized = scaled.half()  # Placeholder - use proper FP8 when available

    if return_scale_zero:
        return quantized, scale.unsqueeze(0), None
    return quantized


def dequantize_tensor(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    config: QuantizationConfig,
    target_dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Dequantize a tensor back to floating point

    Args:
        quantized: Quantized tensor
        scale: Quantization scale
        zero_point: Zero point (None for symmetric quantization)
        config: Quantization configuration
        target_dtype: Target dtype for dequantization

    Returns:
        Dequantized tensor
    """
    if config.precision in [PrecisionMode.FP32, PrecisionMode.FP16, PrecisionMode.BF16]:
        return quantized.to(target_dtype)

    elif config.precision in [PrecisionMode.INT8, PrecisionMode.INT4]:
        if zero_point is not None:
            dequantized = (quantized.float() - zero_point.float()) * scale.float()
        else:
            dequantized = quantized.float() * scale.float()
        return dequantized.to(target_dtype)

    elif config.precision == PrecisionMode.NF4:
        nf4_levels = get_nf4_quantization_levels().to(quantized.device)
        dequantized = nf4_levels[quantized.long()] * scale.float()
        return dequantized.to(target_dtype)

    elif config.precision in [PrecisionMode.FP8_E4M3, PrecisionMode.FP8_E5M2]:
        dequantized = quantized.float() * scale.float()
        return dequantized.to(target_dtype)

    else:
        raise ValueError(f"Unsupported precision mode: {config.precision}")


def estimate_quantization_error(
    original: torch.Tensor,
    quantized: torch.Tensor,
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    config: QuantizationConfig
) -> Dict[str, float]:
    """
    Estimate quantization error metrics

    Returns:
        Dictionary with error metrics (MSE, MAE, SQNR)
    """
    dequantized = dequantize_tensor(quantized, scale, zero_point, config, original.dtype)

    mse = torch.mean((original - dequantized) ** 2).item()
    mae = torch.mean(torch.abs(original - dequantized)).item()

    # Signal to Quantization Noise Ratio (SQNR)
    signal_power = torch.mean(original ** 2).item()
    noise_power = mse
    sqnr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))

    return {
        "mse": mse,
        "mae": mae,
        "sqnr_db": sqnr_db
    }
