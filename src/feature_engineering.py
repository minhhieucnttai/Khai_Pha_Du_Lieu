# feature_engineering.py
# Trich xuat dac trung (feature engineering)
# Tac gia: Minh Hieu
# Ghi chu: Module nay cung cap cac ham trich xuat dac trung truyen thong
#          Voi CNN thi khong can dung vi CNN tu dong hoc dac trung

import os
import numpy as np
import cv2
from PIL import Image

from preprocessing import CLASSES, IMG_SIZE


def extract_color_histogram(img, bins=32):
    """Trich xuat histogram mau"""
    if isinstance(img, Image.Image):
        img = np.array(img)
    
    hist_features = []
    for i in range(3):  # RGB
        hist, _ = np.histogram(img[:, :, i], bins=bins, range=(0, 256))
        hist = hist / hist.sum()  # Normalize
        hist_features.extend(hist)
    
    return np.array(hist_features)


def extract_texture_features(img):
    """Trich xuat dac trung texture"""
    if isinstance(img, Image.Image):
        img = np.array(img)
    
    # Chuyen sang grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    features = []
    
    # Mean, std
    features.append(np.mean(gray))
    features.append(np.std(gray))
    
    # Contrast
    features.append(gray.max() - gray.min())
    
    return np.array(features)


def extract_color_stats(img):
    """Trich xuat thong ke mau"""
    if isinstance(img, Image.Image):
        img = np.array(img)
    
    features = []
    for i in range(3):
        channel = img[:, :, i]
        features.extend([
            np.mean(channel),
            np.std(channel),
            np.min(channel),
            np.max(channel)
        ])
    
    return np.array(features)


def extract_all_features(img):
    """Trich xuat tat ca dac trung"""
    features = []
    features.extend(extract_color_histogram(img))
    features.extend(extract_texture_features(img))
    features.extend(extract_color_stats(img))
    return np.array(features)


# Test
if __name__ == '__main__':
    print("Module feature_engineering.py")
    print("Cac ham co san:")
    print("  - extract_color_histogram(img)")
    print("  - extract_texture_features(img)")
    print("  - extract_color_stats(img)")
    print("  - extract_all_features(img)")
