# Output Directory

This directory contains output files from the Bean Leaf Lesions Classification system.

## Expected files after running EDA and evaluation:

### EDA outputs:
- `class_distribution.png` - Distribution of classes in train/val sets
- `sample_images.png` - Sample images from each class
- `image_size_distribution.png` - Distribution of image sizes

### Evaluation outputs:
- `confusion_matrix.png` - Confusion matrix
- `confusion_matrix_normalized.png` - Normalized confusion matrix
- `roc_curves.png` - ROC curves for each class
- `per_class_metrics.png` - Per-class precision, recall, F1 scores
- `training_history.png` - Training accuracy and loss curves

## How to generate outputs:

```bash
cd src

# Generate EDA outputs
python main.py eda

# Generate evaluation outputs (requires trained model)
python main.py train --model simple_cnn --epochs 30
```
