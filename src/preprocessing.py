# preprocessing.py
# Tien xu ly du lieu anh la dau
# Tac gia: Minh Hieu

import os
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Kich thuoc anh va batch
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Mapping nhan tu file CSV
# 0 = healthy, 1 = angular_leaf_spot, 2 = bean_rust  
CATEGORY_MAP = {0: 'healthy', 1: 'angular_leaf_spot', 2: 'bean_rust'}
CLASSES = list(CATEGORY_MAP.values())
NUM_CLASSES = len(CLASSES)


def get_data_paths(base_path=None):
    """Lay duong dan train/val"""
    if base_path is None:
        base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    train_path = os.path.join(base_path, 'train')
    val_path = os.path.join(base_path, 'val')
    return train_path, val_path


def create_data_generators(train_path, val_path, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """Tao data generator cho training va validation"""
    
    # Augmentation cho training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Chi rescale cho validation
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_gen = train_datagen.flow_from_directory(
        train_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=CLASSES,
        shuffle=True
    )
    
    val_gen = val_datagen.flow_from_directory(
        val_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=CLASSES,
        shuffle=False
    )
    
    return train_gen, val_gen


def load_image(img_path, img_size=IMG_SIZE):
    """Doc va xu ly 1 anh"""
    img = Image.open(img_path)
    img = img.convert('RGB')
    img = img.resize(img_size)
    img_arr = np.array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_arr


def count_images(data_path):
    """Dem so anh trong moi class"""
    counts = {}
    for cls in CLASSES:
        cls_path = os.path.join(data_path, cls)
        if os.path.exists(cls_path):
            imgs = [f for f in os.listdir(cls_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            counts[cls] = len(imgs)
    return counts


def get_sample_images(data_path, n=3):
    """Lay mau anh tu moi class"""
    samples = {}
    for cls in CLASSES:
        cls_path = os.path.join(data_path, cls)
        if os.path.exists(cls_path):
            imgs = [f for f in os.listdir(cls_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            samples[cls] = [os.path.join(cls_path, img) for img in imgs[:n]]
    return samples


# === Cac ham doc CSV ===

def get_csv_paths(base_path=None):
    """Lay duong dan file CSV"""
    if base_path is None:
        base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    train_csv = os.path.join(base_path, 'train.csv')
    val_csv = os.path.join(base_path, 'val.csv')
    return train_csv, val_csv


def load_csv(csv_path, base_path=None):
    """Doc file CSV"""
    df = pd.read_csv(csv_path)
    df.columns = ['image_path', 'category']
    df['class_name'] = df['category'].map(CATEGORY_MAP)
    if base_path:
        df['image_path'] = df['image_path'].apply(lambda x: os.path.join(base_path, x))
    return df


def get_csv_distribution(csv_path):
    """Lay phan bo class tu CSV"""
    df = load_csv(csv_path)
    return df['class_name'].value_counts().to_dict()


# Test
if __name__ == '__main__':
    train_path, val_path = get_data_paths()
    print("Train path:", train_path)
    print("Val path:", val_path)
    
    print("\nSo luong anh:")
    counts = count_images(train_path)
    for cls, cnt in counts.items():
        print(f"  {cls}: {cnt}")
