# Basic INT8 Quantization Workflow

This example demonstrates how to quantize a model to INT8 using AWQ.

## Workflow Steps

1. **Load Model**
   - Node: `MixedPrecisionModelLoader`
   - Settings:
     - model_name: Your checkpoint file
     - default_precision: fp32

2. **Create Quantization Config**
   - Node: `MixedPrecisionConfig`
   - Settings:
     - precision: int8
     - group_size: 128
     - symmetric: True
     - clip_ratio: 1.0

3. **Apply AWQ Quantization**
   - Node: `AWQQuantize`
   - Settings:
     - n_samples: 128
     - auto_scale: True
     - percentile: 0.01
   - Connect: model from step 1, quant_config from step 2

4. **Save Quantized Model**
   - Node: `MixedPrecisionModelSaver`
   - Settings:
     - filename: model_int8_awq
     - format: safetensors
   - Connect: model from step 3

## Expected Results

- Model size reduction: ~75% (32-bit to 8-bit)
- Minimal accuracy loss (<1% typically)
- Faster inference on compatible hardware

## Tips

- Increase n_samples if you notice quality issues
- Try SmoothQuant instead of AWQ for transformer models with outliers
- Use the QuantizationAnalyzer to compare with original model
