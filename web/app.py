# app.py
# Web app phan loai benh la dau
# Su dung Streamlit
# Tac gia: Minh Hieu

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys

# Them duong dan src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing import CLASSES, CATEGORY_MAP, IMG_SIZE, load_image, count_images, get_data_paths
from preprocessing import get_csv_paths, get_csv_distribution

# Cau hinh trang
st.set_page_config(
    page_title="Phan Loai Benh La Dau",
    page_icon="🌿",
    layout="wide"
)

# Tieu de
st.title("🌿 Phân Loại Bệnh Lá Đậu")
st.markdown("**Bean Leaf Disease Classification**")
st.markdown("---")

# Sidebar - Menu
st.sidebar.title("📋 Menu")
menu = st.sidebar.selectbox(
    "Chọn chức năng:",
    ["🏠 Trang chủ", "📊 Phân tích dữ liệu", "🔍 Dự đoán ảnh", "📷 Camera"]
)

# Mo ta cac loai benh
DISEASE_INFO = {
    'healthy': {
        'name': 'Lá khỏe mạnh',
        'desc': 'Lá đậu không có dấu hiệu bệnh, màu xanh tươi.',
        'color': 'green'
    },
    'angular_leaf_spot': {
        'name': 'Bệnh đốm góc lá',
        'desc': 'Bệnh do vi khuẩn gây ra, tạo các đốm góc cạnh màu nâu trên lá.',
        'color': 'orange'
    },
    'bean_rust': {
        'name': 'Bệnh gỉ sắt',
        'desc': 'Bệnh do nấm gây ra, tạo các đốm màu nâu đỏ giống gỉ sắt.',
        'color': 'brown'
    }
}


def load_model_cached():
    """Load model da train (neu co)"""
    import tensorflow as tf
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'cnn_model_best.keras')
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    
    # Thu tim model khac
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    if os.path.exists(model_dir):
        for f in os.listdir(model_dir):
            if f.endswith('.keras') or f.endswith('.h5'):
                return tf.keras.models.load_model(os.path.join(model_dir, f))
    return None


def predict_image(model, img):
    """Du doan 1 anh"""
    # Resize va normalize
    img = img.resize(IMG_SIZE)
    img_arr = np.array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    
    # Du doan
    pred = model.predict(img_arr, verbose=0)
    class_idx = np.argmax(pred[0])
    confidence = pred[0][class_idx]
    class_name = CLASSES[class_idx]
    
    return class_name, confidence, pred[0]


def show_home():
    """Trang chu"""
    st.header("👋 Chào mừng!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📌 Giới thiệu")
        st.write("""
        Đây là ứng dụng phân loại bệnh trên lá đậu sử dụng mô hình CNN (Convolutional Neural Network).
        
        **Các loại bệnh được phân loại:**
        - 🟢 **Healthy** - Lá khỏe mạnh
        - 🟠 **Angular Leaf Spot** - Bệnh đốm góc lá
        - 🟤 **Bean Rust** - Bệnh gỉ sắt đậu
        """)
        
        st.subheader("📁 Thông tin dữ liệu")
        try:
            train_path, val_path = get_data_paths()
            train_counts = count_images(train_path)
            val_counts = count_images(val_path)
            
            st.write(f"**Training:** {sum(train_counts.values())} ảnh")
            st.write(f"**Validation:** {sum(val_counts.values())} ảnh")
        except:
            st.write("Không tìm thấy dữ liệu")
    
    with col2:
        st.subheader("🔬 Các loại bệnh")
        for cls, info in DISEASE_INFO.items():
            st.markdown(f"**{info['name']}**")
            st.write(info['desc'])
            st.write("")


def show_eda():
    """Phan tich du lieu"""
    st.header("📊 Phân Tích Dữ Liệu (EDA)")
    
    try:
        train_path, val_path = get_data_paths()
        train_counts = count_images(train_path)
        val_counts = count_images(val_path)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Phân bố dữ liệu Training")
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            colors = ['green', 'orange', 'brown']
            bars = ax1.bar(train_counts.keys(), train_counts.values(), color=colors)
            ax1.set_xlabel('Loại')
            ax1.set_ylabel('Số lượng')
            ax1.set_title('Phân bố class - Training')
            plt.xticks(rotation=45)
            for bar, val in zip(bars, train_counts.values()):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                        str(val), ha='center')
            plt.tight_layout()
            st.pyplot(fig1)
        
        with col2:
            st.subheader("📈 Phân bố dữ liệu Validation")
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            bars = ax2.bar(val_counts.keys(), val_counts.values(), color=colors)
            ax2.set_xlabel('Loại')
            ax2.set_ylabel('Số lượng')
            ax2.set_title('Phân bố class - Validation')
            plt.xticks(rotation=45)
            for bar, val in zip(bars, val_counts.values()):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        str(val), ha='center')
            plt.tight_layout()
            st.pyplot(fig2)
        
        # Thong ke tong hop
        st.subheader("📋 Thống kê tổng hợp")
        data = {
            'Loại': list(train_counts.keys()),
            'Training': list(train_counts.values()),
            'Validation': list(val_counts.values())
        }
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
        
        # Hien thi anh mau
        st.subheader("🖼️ Ảnh mẫu từ mỗi class")
        cols = st.columns(3)
        for i, cls in enumerate(CLASSES):
            with cols[i]:
                st.write(f"**{DISEASE_INFO[cls]['name']}**")
                cls_path = os.path.join(train_path, cls)
                if os.path.exists(cls_path):
                    imgs = [f for f in os.listdir(cls_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
                    if imgs:
                        img_path = os.path.join(cls_path, imgs[0])
                        img = Image.open(img_path)
                        st.image(img, caption=cls, use_container_width=True)
                        
    except Exception as e:
        st.error(f"Lỗi: {e}")


def show_predict():
    """Du doan anh"""
    st.header("🔍 Dự Đoán Ảnh")
    
    # Chon nguon anh
    source = st.radio("Chọn nguồn ảnh:", ["📤 Upload ảnh", "📁 Chọn ảnh mẫu"])
    
    img = None
    
    if source == "📤 Upload ảnh":
        uploaded = st.file_uploader("Tải ảnh lên", type=['jpg', 'jpeg', 'png'])
        if uploaded:
            img = Image.open(uploaded).convert('RGB')
            st.image(img, caption="Ảnh đã tải lên", width=300)
    
    else:
        # Chon anh mau
        try:
            train_path, _ = get_data_paths()
            cls = st.selectbox("Chọn loại:", CLASSES)
            cls_path = os.path.join(train_path, cls)
            
            if os.path.exists(cls_path):
                imgs = [f for f in os.listdir(cls_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
                if imgs:
                    selected = st.selectbox("Chọn ảnh:", imgs[:20])
                    img_path = os.path.join(cls_path, selected)
                    img = Image.open(img_path).convert('RGB')
                    st.image(img, caption=f"Ảnh mẫu: {selected}", width=300)
        except Exception as e:
            st.error(f"Lỗi: {e}")
    
    # Nut du doan
    if img and st.button("🚀 Dự đoán", type="primary"):
        with st.spinner("Đang xử lý..."):
            model = load_model_cached()
            
            if model:
                cls_name, conf, probs = predict_image(model, img)
                
                st.success("✅ Dự đoán thành công!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Kết quả")
                    info = DISEASE_INFO[cls_name]
                    st.markdown(f"### {info['name']}")
                    st.write(f"**Độ tin cậy:** {conf*100:.2f}%")
                    st.write(info['desc'])
                
                with col2:
                    st.subheader("Xác suất các class")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    colors = ['green', 'orange', 'brown']
                    bars = ax.barh(CLASSES, probs, color=colors)
                    ax.set_xlabel('Xác suất')
                    ax.set_xlim(0, 1)
                    for bar, p in zip(bars, probs):
                        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                               f'{p*100:.1f}%', va='center')
                    plt.tight_layout()
                    st.pyplot(fig)
            else:
                st.warning("⚠️ Chưa có model. Hãy train model trước!")
                st.info("Chạy lệnh: `python main.py train` trong thư mục src/")


def show_camera():
    """Du doan tu camera"""
    st.header("📷 Dự Đoán Từ Camera")
    
    st.info("📸 Chụp ảnh từ camera để dự đoán")
    
    # Camera input
    camera_img = st.camera_input("Chụp ảnh lá đậu")
    
    if camera_img:
        img = Image.open(camera_img).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(img, caption="Ảnh từ camera", use_container_width=True)
        
        with col2:
            if st.button("🚀 Dự đoán", type="primary"):
                with st.spinner("Đang xử lý..."):
                    model = load_model_cached()
                    
                    if model:
                        cls_name, conf, probs = predict_image(model, img)
                        
                        info = DISEASE_INFO[cls_name]
                        st.success(f"**Kết quả:** {info['name']}")
                        st.write(f"**Độ tin cậy:** {conf*100:.2f}%")
                        st.write(info['desc'])
                        
                        # Bieu do xac suat
                        fig, ax = plt.subplots(figsize=(5, 3))
                        colors = ['green', 'orange', 'brown']
                        ax.barh(CLASSES, probs, color=colors)
                        ax.set_xlabel('Xác suất')
                        ax.set_xlim(0, 1)
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.warning("⚠️ Chưa có model!")


# Main
if menu == "🏠 Trang chủ":
    show_home()
elif menu == "📊 Phân tích dữ liệu":
    show_eda()
elif menu == "🔍 Dự đoán ảnh":
    show_predict()
elif menu == "📷 Camera":
    show_camera()

# Footer
st.markdown("---")
st.markdown("**Đồ án Khai Phá Dữ Liệu** - Phân loại bệnh lá đậu | Tác giả: Minh Hiếu")
