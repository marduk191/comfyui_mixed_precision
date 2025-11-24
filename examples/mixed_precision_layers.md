# Mixed Precision by Layer Type

This example shows how to apply different precision levels to different layer types.

## Use Case

Keep attention layers at higher precision (FP16) while aggressively quantizing MLP layers (INT4) for optimal quality/performance balance.

## Workflow Steps

### 1. Load Model
- Node: `MixedPrecisionModelLoader`
- Settings:
  - model_name: Your checkpoint
  - default_precision: fp32

### 2. Create High Precision Config (for Attention)
- Node: `MixedPrecisionConfig`
- Settings:
  - precision: fp16
  - group_size: 128

### 3. Select Attention Layers
- Node: `LayerPrecisionSelector`
- Settings:
  - layer_pattern: "*"
  - apply_to_attention: True
  - apply_to_mlp: False
  - apply_to_embeddings: False
  - apply_to_output: False
- Connect: precision_config from step 2

### 4. Quantize Attention Layers
- Node: `AWQQuantize`
- Settings:
  - n_samples: 128
  - auto_scale: True
- Connect:
  - model from step 1
  - quant_config from step 2
  - layer_config from step 3

### 5. Create Low Precision Config (for MLP)
- Node: `MixedPrecisionConfig`
- Settings:
  - precision: int4
  - group_size: 128

### 6. Select MLP Layers
- Node: `LayerPrecisionSelector`
- Settings:
  - layer_pattern: "*"
  - apply_to_attention: False
  - apply_to_mlp: True
- Connect: precision_config from step 5

### 7. Quantize MLP Layers
- Node: `AWQQuantize`
- Settings:
  - n_samples: 128
  - auto_scale: True
  - percentile: 0.01
- Connect:
  - model from step 4
  - quant_config from step 5
  - layer_config from step 6

### 8. Analyze Results
- Node: `QuantizationAnalyzer`
- Connect:
  - original_model from step 1
  - quantized_model from step 7

### 9. Save Final Model
- Node: `MixedPrecisionModelSaver`
- Settings:
  - filename: model_mixed_precision
  - format: safetensors
- Connect: model from step 7

## Expected Results

- Size reduction: ~85% (better than uniform INT8)
- Quality: Better than uniform INT4 (attention preserved)
- Performance: Good balance between speed and accuracy

## Why This Works

- **Attention layers** are critical for maintaining model quality but are a smaller portion of parameters
- **MLP layers** are more robust to quantization and make up the majority of parameters
- This approach maximizes compression while protecting quality-critical components

## Variations

### Even More Aggressive
- Attention: INT8
- MLP: INT4
- Embeddings/Output: FP16

### Conservative
- Attention: FP16
- MLP: INT8
- Embeddings/Output: FP16

### Ultra-Compressed (SVDQ)
- Use SVDQuantize on MLP layers with:
  - high_precision: fp16
  - low_precision: int4
  - rank_ratio: 0.8
