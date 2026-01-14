# Models Directory

This directory contains trained models for Bean Leaf Lesions Classification.

## Expected files after training:

- `simple_cnn_best.keras` - Best checkpoint from Simple CNN training
- `simple_cnn_final.keras` - Final Simple CNN model
- `mobilenetv2_best.keras` - Best checkpoint from MobileNetV2 training
- `mobilenetv2_final.keras` - Final MobileNetV2 model
- `logs/` - TensorBoard logs

## How to train a model:

```bash
cd src
python main.py train --model simple_cnn --epochs 30
```
