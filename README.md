# ComfyUI Mixed Precision Nodes

Advanced quantization and mixed precision support for ComfyUI models. This custom node pack enables you to work with models using various precision formats (4-bit to 32-bit) and apply state-of-the-art quantization techniques.

## Features

### 🎯 Supported Precision Formats
- **32-bit**: FP32 (Full precision)
- **16-bit**: FP16, BF16
- **8-bit**: INT8, FP8 (E4M3, E5M2)
- **4-bit**: INT4, NF4 (NormalFloat4)

### 🔬 Advanced Quantization Methods

#### AWQ (Activation-aware Weight Quantization)
- Protects salient weights based on activation magnitudes
- Automatic per-channel scaling optimization
- Minimal accuracy loss with aggressive quantization
- Ideal for large language models and transformers

#### SmoothQuant
- Migrates quantization difficulty from activations to weights
- Per-channel smoothing with configurable alpha parameter
- Excellent for models with activation outliers
- Automatic alpha search for optimal results

#### SVDQ (Singular Value Decomposition Quantization)
- Decomposes weights using SVD
- Mixed precision on components based on importance
- Adaptive rank selection based on energy preservation
- Superior compression with maintained quality

### 🎛️ Layer-wise Precision Control
- Select specific layers for quantization
- Apply different precision to different layer types
- Pattern-based layer selection
- Exclude sensitive layers (embeddings, output heads)

## Installation

1. Clone this repository into your ComfyUI custom_nodes folder:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/comfyui_mixed_precision.git
```

2. Install dependencies:
```bash
cd comfyui_mixed_precision
pip install -r requirements.txt
```

3. Restart ComfyUI

## Node Reference

### Configuration Nodes

#### Mixed Precision Config
Creates a quantization configuration that can be reused across multiple quantization nodes.

**Parameters:**
- `precision`: Target precision format (fp32, fp16, bf16, int8, int4, nf4, fp8_e4m3, fp8_e5m2)
- `group_size`: Group size for grouped quantization (default: 128)
- `symmetric`: Use symmetric quantization (default: True)
- `clip_ratio`: Clip ratio for outlier handling (default: 1.0)
- `use_double_quant`: Apply double quantization (default: False)

**Output:** `QUANT_CONFIG`

#### Layer Precision Selector
Selects which layers should be quantized based on patterns and layer types.

**Parameters:**
- `layer_pattern`: Glob pattern for layer matching (e.g., "*.weight")
- `precision_config`: Quantization configuration to apply
- `apply_to_attention`: Include attention layers (default: True)
- `apply_to_mlp`: Include MLP/FFN layers (default: True)
- `apply_to_embeddings`: Include embedding layers (default: False)
- `apply_to_output`: Include output layers (default: False)
- `exclude_patterns`: Newline-separated patterns to exclude (optional)

**Output:** `LAYER_CONFIG`

### Quantization Nodes

#### AWQ Quantize
Applies Activation-aware Weight Quantization to a model.

**Parameters:**
- `model`: Input model
- `quant_config`: Quantization configuration
- `n_samples`: Number of calibration samples (default: 128)
- `auto_scale`: Enable automatic scaling (default: True)
- `percentile`: Percentile for salient weight detection (default: 0.01)
- `layer_config`: Optional layer-specific configuration

**Outputs:** `MODEL`, `AWQ_INFO`

#### SmoothQuant
Applies SmoothQuant quantization to a model.

**Parameters:**
- `model`: Input model
- `quant_config`: Quantization configuration
- `alpha`: Smoothing strength 0-1 (default: 0.5)
  - 0 = all difficulty to weights
  - 1 = all difficulty to activations
  - 0.5 = balanced
- `auto_alpha`: Automatically search for optimal alpha (default: False)
- `calibration_samples`: Number of calibration samples (default: 512)
- `layer_config`: Optional layer-specific configuration

**Outputs:** `MODEL`, `SMOOTH_INFO`

#### SVD Quantize
Applies SVD-based mixed precision quantization.

**Parameters:**
- `model`: Input model
- `rank_ratio`: Ratio of singular values to keep (default: 0.9)
- `high_precision_ratio`: Ratio of components in high precision (default: 0.1)
- `high_precision`: Precision for important components (fp32, fp16, bf16)
- `low_precision`: Precision for less important components (int8, int4, nf4)
- `use_adaptive_rank`: Auto-determine rank based on energy (default: True)
- `energy_threshold`: Energy preservation threshold (default: 0.95)
- `layer_names`: Specific layers to quantize (optional, newline-separated)

**Outputs:** `MODEL`, `SVD_INFO`

### Model I/O Nodes

#### Load Mixed Precision Model
Loads a model with specified default precision.

**Parameters:**
- `model_name`: Model checkpoint to load
- `default_precision`: Base precision (fp32, fp16, bf16)
- `quantization_method`: Optional quantization method (none, awq, smoothquant, svdq)

**Output:** `MODEL`

#### Save Mixed Precision Model
Saves a quantized model to disk.

**Parameters:**
- `model`: Model to save
- `filename`: Output filename
- `format`: File format (safetensors, ckpt)
- `quantization_info`: Optional quantization metadata

**Output:** None (saves to disk)

### Analysis Nodes

#### Quantization Analyzer
Analyzes quantization error and provides detailed statistics.

**Parameters:**
- `original_model`: Original unquantized model
- `quantized_model`: Quantized model
- `quant_info`: Optional quantization info from quantization nodes

**Output:** `STRING` (analysis report)

## Example Workflows

### Basic INT8 Quantization

```
Load Model → Mixed Precision Config (int8) → AWQ Quantize → Save Model
```

1. Load your model with "Load Mixed Precision Model"
2. Create a config with "Mixed Precision Config" (set to int8)
3. Connect to "AWQ Quantize"
4. Save with "Save Mixed Precision Model"

### Advanced Mixed Precision with Layer Selection

```
Load Model →
  ├─ Config (int4) → Layer Selector (MLP) → AWQ Quantize →
  ├─ Config (fp16) → Layer Selector (Attention) → AWQ Quantize →
  └─ Merge → Save Model
```

1. Create separate configs for different precisions
2. Use Layer Selectors to target specific layer types
3. Apply different quantization to each group
4. Combine and save

### SVDQ with Analysis

```
Load Model → SVD Quantize →
  ├─ Save Model
  └─ Quantization Analyzer (with original) → Display
```

1. Load model
2. Apply SVD Quantize with desired parameters
3. Save quantized model
4. Compare with original using Quantization Analyzer

### SmoothQuant for Transformers

```
Load Model →
  Mixed Precision Config (int8) →
  SmoothQuant (alpha=0.5, auto_alpha=True) →
  Quantization Analyzer →
  Save Model
```

## Technical Details

### Quantization Methods Explained

#### AWQ (Activation-aware Weight Quantization)
AWQ identifies and protects "salient" weights that are critical for model accuracy. It does this by:
1. Collecting activation statistics during calibration
2. Computing per-channel scaling factors based on activation magnitudes
3. Applying optimal scaling that minimizes quantization error
4. Quantizing scaled weights to target precision

Best for: Large models, aggressive quantization (4-bit), maintaining accuracy

#### SmoothQuant
SmoothQuant addresses the challenge of quantizing both weights and activations by:
1. Analyzing activation and weight ranges per channel
2. Computing smoothing factors: `s = act_range^α / weight_range^(1-α)`
3. Migrating difficulty: `Y = (X / s) * (s * W)`
4. Quantizing smoothed tensors

Best for: Models with activation outliers, balanced W8A8 quantization

#### SVDQ (Singular Value Decomposition Quantization)
SVDQ decomposes weight matrices and applies mixed precision:
1. Decompose: `W = U @ diag(S) @ V^T`
2. Select rank based on singular value energy
3. Apply high precision to top components (large singular values)
4. Apply low precision to remaining components
5. Reconstruct on-demand or store compressed

Best for: Maximum compression, structured sparsity, memory-constrained deployment

### Precision Modes

| Mode | Bits | Range | Use Case |
|------|------|-------|----------|
| FP32 | 32 | ±3.4e38 | Full precision baseline |
| FP16 | 16 | ±65,504 | Standard mixed precision |
| BF16 | 16 | ±3.4e38 | Training, better range than FP16 |
| INT8 | 8 | -128 to 127 | Good accuracy/speed tradeoff |
| INT4 | 4 | -8 to 7 | Aggressive compression |
| NF4 | 4 | Normalized | Better than INT4 for normal distributions |
| FP8_E4M3 | 8 | ±448 | ML training, more precision |
| FP8_E5M2 | 8 | ±57,344 | ML inference, more range |

## Performance Tips

1. **Start Conservative**: Begin with INT8 or FP16 before trying 4-bit
2. **Use Layer Selection**: Keep embeddings and output layers at higher precision
3. **Calibration Data**: More calibration samples = better quantization quality
4. **AWQ for Aggressive**: Use AWQ when going to INT4/NF4
5. **SmoothQuant for Transformers**: Best results with transformer architectures
6. **SVDQ for Memory**: Use when memory is more critical than speed
7. **Analyze First**: Use Quantization Analyzer to verify quality before deployment

## Troubleshooting

### Out of Memory
- Reduce `group_size` in config
- Use SVDQ with lower `rank_ratio`
- Process layers sequentially instead of in parallel

### Poor Quality
- Increase `n_samples` for calibration
- Use less aggressive precision (INT8 instead of INT4)
- Enable `auto_scale` or `auto_alpha`
- Exclude sensitive layers with Layer Selector

### Slow Quantization
- Reduce `n_samples`
- Disable `auto_alpha` or `auto_scale`
- Use simpler quantization method

## Citation

If you use this in your research, please cite the original papers:

```bibtex
@article{lin2023awq,
  title={AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration},
  author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and Dang, Xingyu and Han, Song},
  journal={arXiv preprint arXiv:2306.00978},
  year={2023}
}

@article{xiao2022smoothquant,
  title={SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models},
  author={Xiao, Guangxuan and Lin, Ji and Seznec, Mickael and Demuynck, Julien and Han, Song},
  journal={arXiv preprint arXiv:2211.10438},
  year={2022}
}
```

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

MIT License - See LICENSE file for details

## Support

For issues, questions, or feature requests, please open an issue on GitHub.