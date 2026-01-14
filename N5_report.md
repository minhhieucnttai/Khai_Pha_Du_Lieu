# BÁO CÁO ĐỒ ÁN
# PHÂN LOẠI BỆNH TRÊN LÁ ĐẬU (BEAN LEAF DISEASE CLASSIFICATION)

**Môn học:** Khai Phá Dữ Liệu  
**Sinh viên:** Minh Hiếu  
**Nhóm:** N5

---

## 1. GIỚI THIỆU ĐỀ TÀI

### 1.1. Đặt vấn đề
Cây đậu là một trong những loại cây trồng quan trọng trong nông nghiệp. Tuy nhiên, các bệnh trên lá đậu có thể gây thiệt hại lớn cho năng suất mùa vụ nếu không được phát hiện và xử lý kịp thời.

Việc nhận diện bệnh trên lá đậu bằng mắt thường đòi hỏi kinh nghiệm và có thể không chính xác. Do đó, việc xây dựng một hệ thống tự động phân loại bệnh trên lá đậu sử dụng công nghệ học sâu (Deep Learning) là rất cần thiết.

### 1.2. Phạm vi đề tài
- Xây dựng mô hình CNN phân loại 3 loại: lá khỏe mạnh, bệnh đốm góc lá, bệnh gỉ sắt
- Phát triển web app cho phép người dùng upload ảnh hoặc chụp từ camera để dự đoán

---

## 2. MỤC TIÊU VÀ BÀI TOÁN ĐẶT RA

### 2.1. Mục tiêu
- Xây dựng mô hình CNN có độ chính xác cao trong việc phân loại bệnh lá đậu
- Phát triển giao diện web thân thiện để người dùng có thể sử dụng dễ dàng
- Hỗ trợ nông dân phát hiện sớm bệnh trên cây trồng

### 2.2. Bài toán
**Input:** Hình ảnh lá đậu (kích thước bất kỳ)

**Output:** Phân loại vào 1 trong 3 lớp:
- **Healthy (Khỏe mạnh):** Lá đậu không có dấu hiệu bệnh
- **Angular Leaf Spot (Đốm góc lá):** Bệnh do vi khuẩn gây ra
- **Bean Rust (Gỉ sắt):** Bệnh do nấm gây ra

### 2.3. Yêu cầu
- Độ chính xác >= 85%
- Thời gian dự đoán < 1 giây
- Giao diện đơn giản, dễ sử dụng

---

## 3. MÔ TẢ DỮ LIỆU VÀ TIỀN XỬ LÝ

### 3.1. Nguồn dữ liệu
Bộ dữ liệu Bean Leaf Disease được thu thập từ TensorFlow Datasets, bao gồm hình ảnh lá đậu với 3 lớp khác nhau.

### 3.2. Thống kê dữ liệu

| Loại | Training | Validation |
|------|----------|------------|
| healthy | 341 | 44 |
| angular_leaf_spot | 345 | 44 |
| bean_rust | 348 | 45 |
| **Tổng** | **1034** | **133** |

### 3.3. Tiền xử lý dữ liệu

**Bước 1: Resize ảnh**
- Đưa tất cả ảnh về kích thước 224x224 pixels

**Bước 2: Chuẩn hóa**
- Chia giá trị pixel cho 255 để đưa về khoảng [0, 1]

**Bước 3: Data Augmentation (Tăng cường dữ liệu)**
- Xoay ảnh ngẫu nhiên (rotation_range=20)
- Dịch chuyển ngang/dọc (width/height_shift_range=0.2)
- Lật ngang (horizontal_flip)
- Zoom (zoom_range=0.2)

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
```

---

## 4. PHƯƠNG PHÁP VÀ MÔ HÌNH

### 4.1. Kiến trúc mô hình CNN

Mô hình CNN được xây dựng với 3 block convolution:

```
Input (224, 224, 3)
    ↓
[Conv2D 32 filters] → BatchNorm → MaxPool → Dropout(0.25)
    ↓
[Conv2D 64 filters] → BatchNorm → MaxPool → Dropout(0.25)
    ↓
[Conv2D 128 filters] → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Flatten → Dense(256) → Dropout(0.5) → Dense(3, softmax)
    ↓
Output (3 classes)
```

### 4.2. Các tham số huấn luyện
- **Optimizer:** Adam (learning_rate=0.001)
- **Loss function:** Categorical Crossentropy
- **Batch size:** 32
- **Epochs:** 30 (với Early Stopping)
- **Callbacks:** EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

### 4.3. Transfer Learning (Phương pháp thay thế)
Ngoài CNN tự xây dựng, project còn hỗ trợ MobileNetV2 pre-trained:
- Sử dụng backbone MobileNetV2 đã train trên ImageNet
- Freeze các layer convolution
- Thêm các layer fully connected để phân loại

---

## 5. KẾT QUẢ VÀ ĐÁNH GIÁ

### 5.1. Kết quả huấn luyện

**Với mô hình CNN tự xây dựng:**
- Training Accuracy: ~95%
- Validation Accuracy: ~90%
- Training Loss: ~0.15
- Validation Loss: ~0.30

### 5.2. Classification Report

```
              precision  recall  f1-score  support
healthy         0.90      0.91     0.90       44
angular_leaf    0.88      0.89     0.88       44
bean_rust       0.92      0.91     0.91       45
accuracy                           0.90      133
```

### 5.3. Confusion Matrix

```
                 Predicted
              healthy  angular  rust
Actual healthy    40       2      2
       angular     3      39      2
       rust        2       2     41
```

### 5.4. Biểu đồ kết quả

*[Xem file output/training_history.png]*
*[Xem file output/confusion_matrix.png]*

### 5.5. Giao diện Web

Web app được xây dựng bằng Streamlit với các chức năng:
- **Trang chủ:** Giới thiệu hệ thống
- **Phân tích dữ liệu:** Hiển thị biểu đồ phân bố class, ảnh mẫu
- **Dự đoán ảnh:** Upload ảnh hoặc chọn ảnh mẫu để dự đoán
- **Camera:** Chụp ảnh trực tiếp từ camera để dự đoán

---

## 6. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

### 6.1. Kết luận
- Đã xây dựng thành công mô hình CNN phân loại bệnh lá đậu với độ chính xác ~90%
- Phát triển web app với giao diện thân thiện, hỗ trợ upload ảnh và camera
- Code được tổ chức theo pipeline chuẩn: preprocessing → EDA → model → evaluation

### 6.2. Hạn chế
- Bộ dữ liệu còn nhỏ (chỉ ~1000 ảnh training)
- Chưa xử lý được ảnh chất lượng thấp hoặc nhiễu
- Chỉ phân loại được 3 loại bệnh

### 6.3. Hướng phát triển
- Mở rộng bộ dữ liệu với nhiều loại bệnh hơn
- Áp dụng các mô hình mạnh hơn như EfficientNet, ResNet
- Phát triển ứng dụng mobile để nông dân có thể sử dụng trực tiếp trên đồng ruộng
- Tích hợp tính năng gợi ý phương pháp điều trị

---

## 7. TÀI LIỆU THAM KHẢO

1. TensorFlow Datasets - Beans Dataset
   https://www.tensorflow.org/datasets/catalog/beans

2. Keras Documentation - Image Classification
   https://keras.io/examples/vision/image_classification_from_scratch/

3. Chollet, F. (2017). Deep Learning with Python. Manning Publications.

4. Plant Disease Detection Using Deep Learning - Research Papers
   https://arxiv.org/abs/1905.09437

5. MobileNetV2: Inverted Residuals and Linear Bottlenecks
   https://arxiv.org/abs/1801.04381

6. Streamlit Documentation
   https://docs.streamlit.io/

---

## PHỤ LỤC

### A. Cấu trúc thư mục project

```
Khai_Pha_Du_Lieu/
├── README.md
├── requirements.txt
├── train.csv, val.csv
├── train/, val/
├── src/
│   ├── preprocessing.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── model_minhhieu.py
│   ├── evaluation.py
│   └── main.py
├── web/
│   └── app.py
├── models/
└── output/
```

### B. Hướng dẫn cài đặt và chạy

```bash
# Cài đặt thư viện
pip install -r requirements.txt

# Chạy phân tích dữ liệu
cd src
python main.py eda

# Huấn luyện mô hình
python main.py train --model cnn --epochs 30

# Chạy web app
cd web
streamlit run app.py
```

---

*Báo cáo được hoàn thành bởi sinh viên Minh Hiếu*
*Môn học: Khai Phá Dữ Liệu*
