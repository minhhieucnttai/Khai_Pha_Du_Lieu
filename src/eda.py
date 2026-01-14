# eda.py
# Phan tich du lieu kham pha (EDA)
# Tac gia: Minh Hieu

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from preprocessing import CLASSES, CATEGORY_MAP, get_data_paths, count_images, get_sample_images
from preprocessing import get_csv_paths, get_csv_distribution


def plot_class_distribution(train_path, val_path, save_path=None):
    """Ve bieu do phan bo class"""
    train_counts = count_images(train_path)
    val_counts = count_images(val_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bieu do train
    classes = list(train_counts.keys())
    train_vals = list(train_counts.values())
    axes[0].bar(classes, train_vals, color=['green', 'orange', 'brown'])
    axes[0].set_title('Phan bo du lieu Training')
    axes[0].set_xlabel('Loai benh')
    axes[0].set_ylabel('So luong anh')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Bieu do val
    val_vals = list(val_counts.values())
    axes[1].bar(classes, val_vals, color=['green', 'orange', 'brown'])
    axes[1].set_title('Phan bo du lieu Validation')
    axes[1].set_xlabel('Loai benh')
    axes[1].set_ylabel('So luong anh')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Da luu: {save_path}")
    plt.show()


def show_sample_images(data_path, n=3, save_path=None):
    """Hien thi anh mau tu moi class"""
    samples = get_sample_images(data_path, n)
    
    fig, axes = plt.subplots(len(CLASSES), n, figsize=(12, 12))
    
    for i, cls in enumerate(CLASSES):
        for j, img_path in enumerate(samples.get(cls, [])[:n]):
            if os.path.exists(img_path):
                img = Image.open(img_path)
                axes[i, j].imshow(img)
                if j == 0:
                    axes[i, j].set_ylabel(cls, fontsize=10)
            axes[i, j].axis('off')
    
    plt.suptitle('Anh mau tu moi loai', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def analyze_image_sizes(data_path):
    """Phan tich kich thuoc anh"""
    widths = []
    heights = []
    
    for cls in CLASSES:
        cls_path = os.path.join(data_path, cls)
        if os.path.exists(cls_path):
            for f in os.listdir(cls_path):
                if f.endswith(('.jpg', '.jpeg', '.png')):
                    try:
                        img = Image.open(os.path.join(cls_path, f))
                        widths.append(img.width)
                        heights.append(img.height)
                    except:
                        pass
    
    print(f"Tong so anh: {len(widths)}")
    print(f"Width: min={min(widths)}, max={max(widths)}, trung binh={np.mean(widths):.1f}")
    print(f"Height: min={min(heights)}, max={max(heights)}, trung binh={np.mean(heights):.1f}")
    
    return widths, heights


def run_eda(train_path, val_path, output_dir='../output'):
    """Chay EDA day du"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*50)
    print("PHAN TICH DU LIEU - BEAN LEAF DISEASE")
    print("="*50)
    
    # 1. Thong ke co ban
    print("\n1. THONG KE DU LIEU")
    print("-"*30)
    train_counts = count_images(train_path)
    val_counts = count_images(val_path)
    
    print(f"Tong anh training: {sum(train_counts.values())}")
    print(f"Tong anh validation: {sum(val_counts.values())}")
    print(f"So class: {len(CLASSES)}")
    
    print("\nPhan bo training:")
    for cls, cnt in train_counts.items():
        print(f"  - {cls}: {cnt} anh")
    
    print("\nPhan bo validation:")
    for cls, cnt in val_counts.items():
        print(f"  - {cls}: {cnt} anh")
    
    # 2. Kich thuoc anh
    print("\n2. KICH THUOC ANH")
    print("-"*30)
    analyze_image_sizes(train_path)
    
    # 3. Ve bieu do
    print("\n3. VE BIEU DO")
    print("-"*30)
    plot_class_distribution(train_path, val_path, 
                           os.path.join(output_dir, 'class_distribution.png'))
    
    show_sample_images(train_path, 3,
                      os.path.join(output_dir, 'sample_images.png'))
    
    # 4. Kiem tra CSV
    print("\n4. KIEM TRA FILE CSV")
    print("-"*30)
    train_csv, val_csv = get_csv_paths()
    if os.path.exists(train_csv):
        csv_dist = get_csv_distribution(train_csv)
        print("Phan bo tu CSV:")
        for cls, cnt in csv_dist.items():
            print(f"  - {cls}: {cnt} anh")
    
    print("\n" + "="*50)
    print("HOAN THANH EDA!")
    print("="*50)


if __name__ == '__main__':
    train_path, val_path = get_data_paths()
    run_eda(train_path, val_path)
