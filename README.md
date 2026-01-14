# Phân Loại Vết Bệnh Trên Lá Đậu (Bean Leaf Lesions Classification)

## Giới thiệu (Introduction)

Dự án này sử dụng mô hình Deep Learning (CNN) để phân loại các loại bệnh trên lá đậu. Hệ thống có thể nhận diện 3 loại:

1. **Angular Leaf Spot** (Đốm góc lá) - Bệnh đốm góc trên lá đậu
2. **Bean Rust** (Gỉ sắt đậu) - Bệnh gỉ sắt trên lá đậu
3. **Healthy** (Khỏe mạnh) - Lá đậu khỏe mạnh, không bệnh

This project uses Deep Learning (CNN) models to classify bean leaf diseases. The system can identify 3 classes: Angular Leaf Spot, Bean Rust, and Healthy leaves.

## Cấu trúc Project (Project Structure)

```
MinhHieu/
├── README.md                    # Hướng dẫn cài đặt và chạy project
├── N5_report.pdf               # Báo cáo project (file PDF)
├── data/                       # Thư mục chứa dữ liệu
│   ├── train/                  # Dữ liệu huấn luyện
│   │   ├── angular_leaf_spot/
│   │   ├── bean_rust/
│   │   └── healthy/
│   └── val/                    # Dữ liệu validation
│       ├── angular_leaf_spot/
│       ├── bean_rust/
│       └── healthy/
├── src/                        # Mã nguồn Python
│   ├── __init__.py
│   ├── preprocessing.py        # Xử lý dữ liệu
│   ├── eda.py                  # Phân tích dữ liệu khám phá (EDA)
│   ├── feature_engineering.py  # Tạo và chọn đặc trưng
│   ├── model_minhhieu.py       # Huấn luyện mô hình CNN
│   ├── evaluation.py           # Đánh giá mô hình
│   └── main.py                 # Script chính để chạy demo
├── requirements.txt            # Danh sách thư viện cần cài
├── models/                     # Thư mục lưu mô hình đã train
└── output/                     # Thư mục lưu kết quả
```

## Yêu cầu hệ thống (System Requirements)

- Python 3.8 hoặc cao hơn
- GPU (khuyến nghị) với CUDA support cho TensorFlow
- RAM: tối thiểu 8GB
- Disk: tối thiểu 2GB trống

## Cài đặt (Installation)

### 1. Clone repository

```bash
git clone https://github.com/minhhieucnttai/Khai_Pha_Du_Lieu.git
cd Khai_Pha_Du_Lieu
```

### 2. Tạo virtual environment (khuyến nghị)

```bash
# Sử dụng venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows

# Hoặc sử dụng conda
conda create -n bean_leaf python=3.9
conda activate bean_leaf
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

## Sử dụng (Usage)

### 1. Chạy Exploratory Data Analysis (EDA)

```bash
cd src
python main.py eda
```

### 2. Huấn luyện mô hình (Train Model)

```bash
cd src

# Train với Simple CNN
python main.py train --model simple_cnn --epochs 30

# Train với Transfer Learning (MobileNetV2)
python main.py train --model mobilenetv2 --epochs 30

# Train với Transfer Learning (ResNet50V2)
python main.py train --model resnet50v2 --epochs 30
```

**Tham số:**
- `--model`: Loại mô hình (simple_cnn, mobilenetv2, resnet50v2)
- `--epochs`: Số epoch huấn luyện (mặc định: 30)
- `--batch-size`: Kích thước batch (mặc định: 32)
- `--data`: Đường dẫn đến thư mục dữ liệu (mặc định: ../data)
- `--output`: Thư mục lưu mô hình (mặc định: ../models)

### 3. Dự đoán ảnh đơn (Single Prediction)

```bash
cd src
python main.py predict ../models/simple_cnn_final.keras path/to/image.jpg
```

### 4. Dự đoán nhiều ảnh (Batch Prediction)

```bash
cd src
python main.py batch ../models/simple_cnn_final.keras path/to/images_folder/
```

### 5. Chạy Demo

```bash
cd src
python main.py demo
```

## Mô tả các module (Module Description)

### preprocessing.py
- Xử lý và chuẩn hóa dữ liệu hình ảnh
- Data augmentation cho training
- Tạo data generators cho TensorFlow

### eda.py
- Phân tích phân bố dữ liệu
- Trực quan hóa mẫu ảnh
- Phân tích kích thước và pixel intensity

### feature_engineering.py
- Trích xuất đặc trưng màu sắc (Color Histogram)
- Trích xuất HOG features
- Trích xuất LBP features
- Các đặc trưng texture và shape

### model_minhhieu.py
- Xây dựng mô hình CNN đơn giản
- Transfer Learning với MobileNetV2, ResNet50V2, EfficientNetB0
- Các callbacks cho training (EarlyStopping, ModelCheckpoint, etc.)

### evaluation.py
- Tính toán các metrics: Accuracy, Precision, Recall, F1-Score
- Vẽ Confusion Matrix
- Vẽ ROC Curves
- Xuất báo cáo đánh giá

### main.py
- Entry point chính của ứng dụng
- Command-line interface
- Tích hợp tất cả các module

## Dataset

Dataset Bean Leaf Lesions gồm:
- **Training set**: 1035 ảnh
- **Validation set**: 133 ảnh
- **Classes**: 3 (Angular Leaf Spot, Bean Rust, Healthy)
- **Format**: JPEG images

## Kết quả mong đợi (Expected Results)

Với Simple CNN model, kỳ vọng đạt:
- Accuracy: ~85-95%
- F1-Score: ~0.85-0.95

Với Transfer Learning (MobileNetV2), kỳ vọng đạt:
- Accuracy: ~90-98%
- F1-Score: ~0.90-0.98

## Tác giả (Author)

- **Tên**: Minh Hieu
- **Project**: Khai Phá Dữ Liệu - Bean Leaf Lesions Classification

## License

Dự án này được sử dụng cho mục đích học tập và nghiên cứu.