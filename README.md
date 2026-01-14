# Phân Loại Bệnh Lá Đậu (Bean Leaf Disease Classification)

## Giới thiệu

Đây là đồ án môn học **Khai Phá Dữ Liệu**, xây dựng hệ thống nhận diện và phân loại bệnh trên lá đậu từ hình ảnh sử dụng mô hình CNN (Convolutional Neural Network).

### Các loại bệnh được phân loại:
- **Healthy** - Lá khỏe mạnh
- **Angular Leaf Spot** - Bệnh đốm góc lá
- **Bean Rust** - Bệnh gỉ sắt đậu

## Cấu trúc thư mục

```
Khai_Pha_Du_Lieu/
├── README.md               # File hướng dẫn
├── requirements.txt        # Thư viện cần cài
├── train.csv              # Dữ liệu training (đường dẫn + nhãn)
├── val.csv                # Dữ liệu validation
├── train/                 # Thư mục ảnh training
│   ├── healthy/
│   ├── angular_leaf_spot/
│   └── bean_rust/
├── val/                   # Thư mục ảnh validation  
├── data/                  # Symlink đến dữ liệu
├── src/                   # Mã nguồn
│   ├── preprocessing.py   # Tiền xử lý dữ liệu
│   ├── eda.py            # Phân tích dữ liệu
│   ├── feature_engineering.py  # Trích xuất đặc trưng
│   ├── model_minhhieu.py # Xây dựng và huấn luyện mô hình
│   ├── evaluation.py     # Đánh giá mô hình
│   └── main.py           # Script chính
├── models/               # Lưu model đã train
└── output/               # Kết quả (biểu đồ, báo cáo)
```

## Cài đặt

### 1. Clone repo
```bash
git clone https://github.com/minhhieucnttai/Khai_Pha_Du_Lieu.git
cd Khai_Pha_Du_Lieu
```

### 2. Cài thư viện
```bash
pip install -r requirements.txt
```

## Sử dụng

### Xem thông tin dữ liệu
```bash
cd src
python main.py info
```

### Chạy phân tích dữ liệu (EDA)
```bash
python main.py eda
```

### Huấn luyện mô hình
```bash
# Train với CNN
python main.py train --model cnn --epochs 30

# Train với MobileNetV2 (Transfer Learning)
python main.py train --model mobilenet --epochs 30
```

### Dự đoán ảnh
```bash
python main.py predict ../models/cnn_final.keras path/to/image.jpg
```

## Thông tin dữ liệu

- **Training**: ~1034 ảnh
- **Validation**: ~133 ảnh  
- **Số class**: 3
- **Định dạng**: JPEG

### Mapping nhãn trong CSV:
- 0 = healthy
- 1 = angular_leaf_spot
- 2 = bean_rust

## Tác giả

- **Tên**: Minh Hiếu
- **Môn học**: Khai Phá Dữ Liệu