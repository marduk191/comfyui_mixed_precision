# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-11-24

### Added
- Initial release of ComfyUI Mixed Precision Nodes
- Support for multiple precision formats (4-bit to 32-bit)
  - FP32, FP16, BF16
  - INT8, INT4
  - NF4 (NormalFloat 4-bit)
  - FP8 (E4M3 and E5M2 formats)
- AWQ (Activation-aware Weight Quantization) implementation
  - Automatic per-channel scaling
  - Salient weight protection
  - Configurable calibration samples
- SmoothQuant implementation
  - Per-channel smoothing with alpha parameter
  - Automatic alpha search
  - Migration of quantization difficulty from activations to weights
- SVDQ (Singular Value Decomposition Quantization) implementation
  - SVD-based weight decomposition
  - Mixed precision on components by importance
  - Adaptive rank selection
  - Energy-based preservation threshold
- Layer-wise precision control
  - Pattern-based layer selection
  - Layer type filtering (attention, MLP, embeddings, output)
  - Exclude patterns support
- ComfyUI nodes:
  - MixedPrecisionConfig: Quantization configuration
  - LayerPrecisionSelector: Layer selection and filtering
  - AWQQuantize: AWQ quantization node
  - SmoothQuantize: SmoothQuant quantization node
  - SVDQuantize: SVDQ quantization node
  - MixedPrecisionModelLoader: Load models with precision control
  - MixedPrecisionModelSaver: Save quantized models
  - QuantizationAnalyzer: Analyze quantization error and statistics
- Comprehensive documentation
  - Installation guide
  - Node reference
  - Example workflows
  - Performance tips
  - Troubleshooting guide
- Example workflows
  - Basic INT8 quantization
  - Mixed precision by layer type

### Technical Details
- Group-wise quantization support
- Symmetric and asymmetric quantization
- Double quantization option
- Quantization error metrics (MSE, MAE, SQNR)
- SafeTensors and checkpoint format support
- Metadata preservation in saved models

## [Unreleased]

### Planned Features
- GPTQ (Generative Pre-trained Transformer Quantization)
- GGUF format export support
- Activation quantization (W8A8, W4A8)
- Dynamic quantization support
- Quantization-aware training hooks
- Integration with ComfyUI model caching
- Real-time quantization preview
- Batch quantization for multiple models
- Custom quantization schemes
- Hardware-specific optimizations (CUDA, ROCm, Metal)
