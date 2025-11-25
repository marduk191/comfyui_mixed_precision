# Example Workflows

This directory contains ready-to-use ComfyUI workflow files demonstrating various quantization techniques.

## How to Use

1. Open ComfyUI
2. Drag and drop any `.json` file from this folder onto the ComfyUI canvas
3. Update the model path in the "MixedPrecisionModelLoader" node
4. Queue the workflow

## Available Workflows

### 1. AWQ INT8 Quantization
**File:** `awq_int8_workflow.json`

**Description:** Basic INT8 quantization using AWQ (Activation-aware Weight Quantization)

**Features:**
- Loads model at FP32
- Quantizes to INT8 using AWQ
- Analyzes quantization quality
- Saves quantized model

**Expected Results:**
- Size reduction: ~75%
- Quality loss: Minimal (<1% typically)
- Speed: Faster inference on INT8-capable hardware

**Best For:** General-purpose quantization with good quality/compression balance

---

### 2. SmoothQuant Workflow
**File:** `smoothquant_workflow.json`

**Description:** INT8 quantization using SmoothQuant with automatic alpha tuning

**Features:**
- SmoothQuant with auto-alpha enabled
- Migrates quantization difficulty from activations to weights
- Optimal for models with activation outliers
- Includes quality analysis

**Expected Results:**
- Size reduction: ~75%
- Quality: Better than plain INT8 for transformer models
- Best alpha parameter automatically selected

**Best For:** Transformer models, models with large activation ranges

---

### 3. SVDQ Compression
**File:** `svdq_workflow.json`

**Description:** Aggressive compression using SVD-based quantization

**Features:**
- SVD decomposition of weight matrices
- FP16 for high-importance components (top 10%)
- INT4 for low-importance components (bottom 90%)
- Adaptive rank selection based on 95% energy preservation
- Compression statistics included

**Expected Results:**
- Size reduction: ~85-90%
- Quality: Better than uniform INT4
- Compression stats showing energy preservation

**Best For:** Maximum compression, memory-constrained deployment

---

### 4. Mixed Precision Layer-wise
**File:** `mixed_precision_layerwise_workflow.json`

**Description:** Advanced workflow applying different precision to different layer types

**Features:**
- **Attention layers:** FP16 (high quality)
- **MLP layers:** INT4 (high compression)
- **Embeddings/Output:** Excluded (keep original precision)
- Separate layer selectors for fine control
- Comprehensive analysis comparing to original

**Expected Results:**
- Size reduction: ~85%
- Quality: Better than uniform INT4 (attention preserved)
- Compression: Better than uniform FP16 (MLP aggressively quantized)

**Best For:** Optimal quality/compression balance, production deployments

**Workflow Breakdown:**
```
1. Load Model (FP32)
2. Config: FP16 → Select Attention Layers → AWQ Quantize
3. Config: INT4 → Select MLP Layers → AWQ Quantize
4. Save & Analyze
```

---

### 5. NF4 Aggressive Compression
**File:** `nf4_aggressive_workflow.json`

**Description:** Maximum compression using 4-bit NormalFloat quantization

**Features:**
- NF4 (4-bit NormalFloat) quantization
- Double quantization enabled for extra compression
- AWQ protection for salient weights
- 512 calibration samples for quality
- Excludes embeddings and output layers

**Expected Results:**
- Size reduction: ~87.5%
- Quality: Better than INT4 for normally distributed weights
- Uses more calibration samples (512) for better 4-bit quality

**Best For:** Extreme compression scenarios, deployment on very limited hardware

**Configuration Highlights:**
- `precision`: nf4
- `use_double_quant`: true
- `n_samples`: 512 (high quality calibration)
- `percentile`: 0.005 (protect more salient weights)

---

## Workflow Comparison Table

| Workflow | Precision | Method | Size Reduction | Quality | Complexity | Use Case |
|----------|-----------|--------|----------------|---------|------------|----------|
| AWQ INT8 | INT8 | AWQ | ~75% | High | Low | General purpose |
| SmoothQuant | INT8 | SmoothQuant | ~75% | High | Low | Transformers |
| SVDQ | FP16+INT4 | SVDQ | ~85-90% | Medium-High | Medium | Max compression |
| Layer-wise | FP16+INT4 | AWQ | ~85% | High | High | Production |
| NF4 Aggressive | NF4 | AWQ | ~87.5% | Medium | Medium | Extreme compression |

## Customization Tips

### Adjusting Quality vs Size

**Higher Quality (less compression):**
- Use INT8 instead of INT4
- Use FP16 instead of INT8
- Increase `n_samples` (more calibration)
- Enable `auto_scale` or `auto_alpha`
- Include fewer layer types

**Higher Compression (smaller size):**
- Use INT4 or NF4 instead of INT8
- Enable `use_double_quant`
- Apply to more layer types
- Use SVDQ with lower `rank_ratio`

### Layer Selection Examples

**Conservative (protect quality-critical layers):**
```
Exclude patterns:
*embed*
*output*
*head*
*norm*
```

**Aggressive (quantize everything except embeddings):**
```
Exclude patterns:
*embed*
```

**Custom pattern matching:**
```
Layer pattern: *.attn.*.weight
Exclude patterns:
*.bias
*LayerNorm*
```

## Troubleshooting Workflows

### "Model not found" error
- Update the model path in `MixedPrecisionModelLoader`
- Ensure model is in ComfyUI's checkpoints folder

### Out of memory during quantization
- Reduce `n_samples` in AWQ/SmoothQuant nodes
- Reduce `calibration_samples` in SmoothQuant
- Close other applications

### Poor quality results
- Increase `n_samples` for better calibration
- Use less aggressive precision (INT8 vs INT4)
- Enable `auto_scale` or `auto_alpha`
- Exclude more layers with Layer Selector

### Slow quantization
- Reduce `n_samples`
- Disable `auto_alpha` in SmoothQuant
- Use simpler method (AWQ INT8 instead of SVDQ)

## Creating Your Own Workflows

Start with one of these workflows and modify:

1. **Change precision:** Edit the `MixedPrecisionConfig` node
2. **Target different layers:** Modify `LayerPrecisionSelector` patterns
3. **Mix methods:** Combine AWQ, SmoothQuant, and SVDQ on different layers
4. **Add analysis:** Connect `QuantizationAnalyzer` to compare results

## Further Reading

- Main README: `../README.md`
- Technical details on quantization methods
- Performance tips and best practices
- API documentation

## Questions?

If you have questions about these workflows or need help customizing them, please open an issue on GitHub.
