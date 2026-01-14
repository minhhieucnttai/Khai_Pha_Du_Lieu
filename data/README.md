# Data Directory

This directory contains the Bean Leaf Lesions dataset for classification.

## Dataset Structure:

```
data/
├── train/                  # Training data (1035 images)
│   ├── angular_leaf_spot/  # Angular leaf spot disease
│   ├── bean_rust/          # Bean rust disease
│   └── healthy/            # Healthy leaves
└── val/                    # Validation data (133 images)
    ├── angular_leaf_spot/
    ├── bean_rust/
    └── healthy/
```

## Classes:

1. **Angular Leaf Spot** - Đốm góc lá: Bệnh do vi khuẩn gây ra, tạo các đốm góc cạnh trên lá
2. **Bean Rust** - Gỉ sắt đậu: Bệnh nấm tạo các đốm màu nâu đỏ giống gỉ sắt
3. **Healthy** - Khỏe mạnh: Lá đậu không có dấu hiệu bệnh

## Image Format:
- Format: JPEG (.jpg)
- Resolution: Variable (will be resized to 224x224 during preprocessing)
