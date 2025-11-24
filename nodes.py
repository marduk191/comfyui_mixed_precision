"""
ComfyUI Nodes for Mixed Precision Model Operations
"""

import torch
import torch.nn as nn
import folder_paths
import comfy.model_management as mm
import comfy.utils
import os
from typing import Dict, List, Tuple, Optional

from .quantization_utils import (
    PrecisionMode,
    QuantizationConfig,
    quantize_tensor,
    dequantize_tensor,
    estimate_quantization_error
)
from .awq import AWQQuantizer, register_activation_hooks as register_awq_hooks
from .smoothquant import SmoothQuantizer, register_smoothquant_hooks
from .svdq import SVDQuantizer, SVDQConfig


class MixedPrecisionConfig:
    """Node for configuring mixed precision quantization"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "precision": (["fp32", "fp16", "bf16", "int8", "int4", "nf4", "fp8_e4m3", "fp8_e5m2"],),
                "group_size": ("INT", {"default": 128, "min": 1, "max": 1024}),
                "symmetric": ("BOOLEAN", {"default": True}),
                "clip_ratio": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.01}),
                "use_double_quant": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("QUANT_CONFIG",)
    FUNCTION = "create_config"
    CATEGORY = "mixed_precision"

    def create_config(self, precision, group_size, symmetric, clip_ratio, use_double_quant):
        precision_mode = PrecisionMode(precision)
        config = QuantizationConfig(
            precision=precision_mode,
            group_size=group_size,
            symmetric=symmetric,
            clip_ratio=clip_ratio,
            use_double_quant=use_double_quant
        )
        return (config,)


class LayerPrecisionSelector:
    """Node for selecting which layers should use which precision"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layer_pattern": ("STRING", {"default": "*.weight", "multiline": False}),
                "precision_config": ("QUANT_CONFIG",),
                "apply_to_attention": ("BOOLEAN", {"default": True}),
                "apply_to_mlp": ("BOOLEAN", {"default": True}),
                "apply_to_embeddings": ("BOOLEAN", {"default": False}),
                "apply_to_output": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "exclude_patterns": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("LAYER_CONFIG",)
    FUNCTION = "create_layer_config"
    CATEGORY = "mixed_precision"

    def create_layer_config(self, layer_pattern, precision_config, apply_to_attention,
                           apply_to_mlp, apply_to_embeddings, apply_to_output,
                           exclude_patterns=""):
        config = {
            "pattern": layer_pattern,
            "quant_config": precision_config,
            "apply_to_attention": apply_to_attention,
            "apply_to_mlp": apply_to_mlp,
            "apply_to_embeddings": apply_to_embeddings,
            "apply_to_output": apply_to_output,
            "exclude_patterns": [p.strip() for p in exclude_patterns.split("\n") if p.strip()]
        }
        return (config,)


class AWQQuantize:
    """Node for applying AWQ quantization"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "quant_config": ("QUANT_CONFIG",),
                "n_samples": ("INT", {"default": 128, "min": 1, "max": 1024}),
                "auto_scale": ("BOOLEAN", {"default": True}),
                "percentile": ("FLOAT", {"default": 0.01, "min": 0.001, "max": 0.1, "step": 0.001}),
            },
            "optional": {
                "layer_config": ("LAYER_CONFIG",),
            }
        }

    RETURN_TYPES = ("MODEL", "AWQ_INFO")
    FUNCTION = "quantize"
    CATEGORY = "mixed_precision/quantization"

    def quantize(self, model, quant_config, n_samples, auto_scale, percentile, layer_config=None):
        # Clone model to avoid modifying original
        model_clone = model.clone()

        # Get the actual PyTorch model
        pytorch_model = model_clone.model.diffusion_model

        # Create AWQ quantizer
        quantizer = AWQQuantizer(
            config=quant_config,
            n_samples=n_samples,
            auto_scale=auto_scale,
            percentile=percentile
        )

        # Parse layer config if provided
        layer_configs = {}
        if layer_config is not None:
            # Build per-layer config based on patterns
            for name, module in pytorch_model.named_modules():
                if self._matches_layer_config(name, layer_config):
                    layer_configs[name] = layer_config["quant_config"]

        # Apply AWQ quantization
        pytorch_model = quantizer.apply_to_model(
            pytorch_model,
            layer_config=layer_configs if layer_configs else None,
            inplace=True
        )

        # Create info dict
        info = {
            "method": "AWQ",
            "n_samples": n_samples,
            "quantizer": quantizer,
            "weight_scales": quantizer.weight_scales
        }

        return (model_clone, info)

    def _matches_layer_config(self, layer_name: str, layer_config: dict) -> bool:
        """Check if layer matches the configuration patterns"""
        import fnmatch

        # Check exclusions first
        for exclude_pattern in layer_config.get("exclude_patterns", []):
            if fnmatch.fnmatch(layer_name, exclude_pattern):
                return False

        # Check layer type filters
        if not layer_config.get("apply_to_attention", True):
            if "attn" in layer_name.lower() or "attention" in layer_name.lower():
                return False

        if not layer_config.get("apply_to_mlp", True):
            if "mlp" in layer_name.lower() or "ffn" in layer_name.lower():
                return False

        if not layer_config.get("apply_to_embeddings", False):
            if "embed" in layer_name.lower():
                return True
        else:
            if "embed" in layer_name.lower():
                return False

        if not layer_config.get("apply_to_output", False):
            if "output" in layer_name.lower() or "head" in layer_name.lower():
                return True
        else:
            if "output" in layer_name.lower() or "head" in layer_name.lower():
                return False

        # Check main pattern
        pattern = layer_config.get("pattern", "*")
        return fnmatch.fnmatch(layer_name, pattern)


class SmoothQuantize:
    """Node for applying SmoothQuant"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "quant_config": ("QUANT_CONFIG",),
                "alpha": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "auto_alpha": ("BOOLEAN", {"default": False}),
                "calibration_samples": ("INT", {"default": 512, "min": 1, "max": 2048}),
            },
            "optional": {
                "layer_config": ("LAYER_CONFIG",),
            }
        }

    RETURN_TYPES = ("MODEL", "SMOOTH_INFO")
    FUNCTION = "quantize"
    CATEGORY = "mixed_precision/quantization"

    def quantize(self, model, quant_config, alpha, auto_alpha, calibration_samples, layer_config=None):
        # Clone model
        model_clone = model.clone()
        pytorch_model = model_clone.model.diffusion_model

        # Create SmoothQuant quantizer
        quantizer = SmoothQuantizer(
            config=quant_config,
            alpha=alpha,
            auto_alpha=auto_alpha,
            calibration_samples=calibration_samples
        )

        # Parse layer config if provided
        layer_configs = {}
        if layer_config is not None:
            for name, module in pytorch_model.named_modules():
                if self._matches_layer_config(name, layer_config):
                    layer_configs[name] = layer_config["quant_config"]

        # Apply SmoothQuant with quantization
        pytorch_model = quantizer.quantize_with_smoothing(
            pytorch_model,
            layer_config=layer_configs if layer_configs else None
        )

        info = {
            "method": "SmoothQuant",
            "alpha": alpha,
            "quantizer": quantizer,
            "smoothing_factors": quantizer.smoothing_factors
        }

        return (model_clone, info)

    def _matches_layer_config(self, layer_name: str, layer_config: dict) -> bool:
        """Check if layer matches the configuration patterns"""
        import fnmatch
        for exclude_pattern in layer_config.get("exclude_patterns", []):
            if fnmatch.fnmatch(layer_name, exclude_pattern):
                return False
        pattern = layer_config.get("pattern", "*")
        return fnmatch.fnmatch(layer_name, pattern)


class SVDQuantize:
    """Node for applying SVDQ (SVD-based quantization)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "rank_ratio": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 1.0, "step": 0.05}),
                "high_precision_ratio": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.5, "step": 0.05}),
                "high_precision": (["fp32", "fp16", "bf16"],),
                "low_precision": (["int8", "int4", "nf4"],),
                "use_adaptive_rank": ("BOOLEAN", {"default": True}),
                "energy_threshold": ("FLOAT", {"default": 0.95, "min": 0.5, "max": 0.999, "step": 0.01}),
            },
            "optional": {
                "layer_names": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("MODEL", "SVD_INFO")
    FUNCTION = "quantize"
    CATEGORY = "mixed_precision/quantization"

    def quantize(self, model, rank_ratio, high_precision_ratio, high_precision,
                 low_precision, use_adaptive_rank, energy_threshold, layer_names=""):
        # Clone model
        model_clone = model.clone()
        pytorch_model = model_clone.model.diffusion_model

        # Create SVDQ config
        svdq_config = SVDQConfig(
            rank_ratio=rank_ratio,
            high_precision_ratio=high_precision_ratio,
            high_precision_mode=PrecisionMode(high_precision),
            low_precision_mode=PrecisionMode(low_precision),
            use_adaptive_rank=use_adaptive_rank,
            energy_threshold=energy_threshold
        )

        # Create quantizer
        quantizer = SVDQuantizer(svdq_config)

        # Parse layer names if provided
        target_layers = None
        if layer_names.strip():
            target_layers = [name.strip() for name in layer_names.split("\n") if name.strip()]

        # Apply SVDQ
        pytorch_model = quantizer.apply_to_model(
            pytorch_model,
            layer_names=target_layers,
            inplace=True
        )

        # Collect compression statistics
        compression_stats = {}
        for name in quantizer.svd_cache.keys():
            try:
                compression_stats[name] = quantizer.analyze_compression(name)
            except:
                pass

        info = {
            "method": "SVDQ",
            "quantizer": quantizer,
            "compression_stats": compression_stats
        }

        return (model_clone, info)


class MixedPrecisionModelLoader:
    """Load a model with mixed precision configuration"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("checkpoints"),),
                "default_precision": (["fp32", "fp16", "bf16"],),
            },
            "optional": {
                "quantization_method": (["none", "awq", "smoothquant", "svdq"],),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "mixed_precision"

    def load_model(self, model_name, default_precision, quantization_method="none"):
        model_path = folder_paths.get_full_path("checkpoints", model_name)

        # Load model using ComfyUI's loader
        model = comfy.sd.load_checkpoint_guess_config(
            model_path,
            output_vae=False,
            output_clip=False,
            embedding_directory=folder_paths.get_folder_paths("embeddings")
        )[0]

        # Convert to specified precision
        if default_precision == "fp16":
            model.model.half()
        elif default_precision == "bf16":
            model.model.bfloat16()

        return (model,)


class MixedPrecisionModelSaver:
    """Save a model with mixed precision"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "filename": ("STRING", {"default": "mixed_precision_model"}),
                "format": (["safetensors", "ckpt"],),
            },
            "optional": {
                "quantization_info": ("AWQ_INFO,SMOOTH_INFO,SVD_INFO",),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save_model"
    CATEGORY = "mixed_precision"

    def save_model(self, model, filename, format, quantization_info=None):
        output_dir = folder_paths.get_output_directory()
        full_output_folder = os.path.join(output_dir, "mixed_precision")

        if not os.path.exists(full_output_folder):
            os.makedirs(full_output_folder)

        # Prepare filename
        if not filename.endswith(f".{format}"):
            filename = f"{filename}.{format}"

        filepath = os.path.join(full_output_folder, filename)

        # Extract state dict
        state_dict = model.model.state_dict()

        # Add quantization metadata if available
        metadata = {}
        if quantization_info is not None:
            metadata["quantization_method"] = quantization_info.get("method", "unknown")

        # Save based on format
        if format == "safetensors":
            from safetensors.torch import save_file
            save_file(state_dict, filepath, metadata=metadata)
        else:
            torch.save({
                "state_dict": state_dict,
                "metadata": metadata
            }, filepath)

        return {"ui": {"saved_path": [filepath]}}


class QuantizationAnalyzer:
    """Analyze quantization error and statistics"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_model": ("MODEL",),
                "quantized_model": ("MODEL",),
            },
            "optional": {
                "quant_info": ("AWQ_INFO,SMOOTH_INFO,SVD_INFO",),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "analyze"
    CATEGORY = "mixed_precision/analysis"

    def analyze(self, original_model, quantized_model, quant_info=None):
        report = ["=" * 50]
        report.append("QUANTIZATION ANALYSIS REPORT")
        report.append("=" * 50)

        if quant_info:
            method = quant_info.get("method", "Unknown")
            report.append(f"\nMethod: {method}")
            report.append("-" * 50)

        # Compare state dicts
        orig_sd = original_model.model.state_dict()
        quant_sd = quantized_model.model.state_dict()

        total_params = 0
        total_mse = 0
        layer_count = 0

        for key in orig_sd.keys():
            if "weight" in key and key in quant_sd:
                orig_weight = orig_sd[key].float()
                quant_weight = quant_sd[key].float()

                if orig_weight.shape == quant_weight.shape:
                    mse = torch.mean((orig_weight - quant_weight) ** 2).item()
                    total_mse += mse
                    layer_count += 1
                    total_params += orig_weight.numel()

                    report.append(f"\n{key}:")
                    report.append(f"  MSE: {mse:.6e}")
                    report.append(f"  Params: {orig_weight.numel():,}")

        if layer_count > 0:
            avg_mse = total_mse / layer_count
            report.append("\n" + "=" * 50)
            report.append(f"Average MSE: {avg_mse:.6e}")
            report.append(f"Total Parameters: {total_params:,}")
            report.append(f"Layers Analyzed: {layer_count}")

        # Add compression stats for SVDQ
        if quant_info and quant_info.get("method") == "SVDQ":
            stats = quant_info.get("compression_stats", {})
            if stats:
                report.append("\n" + "=" * 50)
                report.append("SVDQ COMPRESSION STATISTICS")
                report.append("-" * 50)
                for layer_name, layer_stats in list(stats.items())[:10]:  # Show first 10
                    report.append(f"\n{layer_name}:")
                    report.append(f"  Compression Ratio: {layer_stats['compression_ratio']:.2f}x")
                    report.append(f"  Energy Preserved: {layer_stats['energy_preserved']*100:.2f}%")
                    report.append(f"  Rank: {layer_stats['rank']}")

        report.append("\n" + "=" * 50)

        return ("\n".join(report),)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "MixedPrecisionConfig": MixedPrecisionConfig,
    "LayerPrecisionSelector": LayerPrecisionSelector,
    "AWQQuantize": AWQQuantize,
    "SmoothQuantize": SmoothQuantize,
    "SVDQuantize": SVDQuantize,
    "MixedPrecisionModelLoader": MixedPrecisionModelLoader,
    "MixedPrecisionModelSaver": MixedPrecisionModelSaver,
    "QuantizationAnalyzer": QuantizationAnalyzer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MixedPrecisionConfig": "Mixed Precision Config",
    "LayerPrecisionSelector": "Layer Precision Selector",
    "AWQQuantize": "AWQ Quantize",
    "SmoothQuantize": "SmoothQuant",
    "SVDQuantize": "SVD Quantize",
    "MixedPrecisionModelLoader": "Load Mixed Precision Model",
    "MixedPrecisionModelSaver": "Save Mixed Precision Model",
    "QuantizationAnalyzer": "Analyze Quantization",
}
