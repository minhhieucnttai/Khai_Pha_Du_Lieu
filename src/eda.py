"""
Exploratory Data Analysis (EDA) module for Bean Leaf Lesions Classification
Phân tích dữ liệu khám phá
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import Counter

from preprocessing import (
    CLASSES, CATEGORY_MAP, 
    get_data_paths, count_images_per_class, get_sample_images,
    get_csv_paths, get_csv_class_distribution, load_csv_data
)


def plot_class_distribution(train_path, val_path, save_path=None):
    """
    Plot the distribution of classes in training and validation sets.
    
    Args:
        train_path: Path to training data
        val_path: Path to validation data
        save_path: Path to save the plot (optional)
    """
    train_counts = count_images_per_class(train_path)
    val_counts = count_images_per_class(val_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training set distribution
    ax1 = axes[0]
    classes = list(train_counts.keys())
    train_values = list(train_counts.values())
    bars1 = ax1.bar(classes, train_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax1.set_title('Training Set Class Distribution', fontsize=14)
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Number of Images')
    ax1.tick_params(axis='x', rotation=45)
    for bar, value in zip(bars1, train_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                str(value), ha='center', va='bottom', fontsize=10)
    
    # Validation set distribution
    ax2 = axes[1]
    val_values = list(val_counts.values())
    bars2 = ax2.bar(classes, val_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax2.set_title('Validation Set Class Distribution', fontsize=14)
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Number of Images')
    ax2.tick_params(axis='x', rotation=45)
    for bar, value in zip(bars2, val_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                str(value), ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved class distribution plot to {save_path}")
    
    plt.show()


def plot_sample_images(data_path, num_samples=3, save_path=None):
    """
    Display sample images from each class.
    
    Args:
        data_path: Path to the data directory
        num_samples: Number of samples per class
        save_path: Path to save the plot (optional)
    """
    samples = get_sample_images(data_path, num_samples)
    
    fig, axes = plt.subplots(len(CLASSES), num_samples, figsize=(4*num_samples, 4*len(CLASSES)))
    
    for i, class_name in enumerate(CLASSES):
        for j, img_path in enumerate(samples.get(class_name, [])[:num_samples]):
            if os.path.exists(img_path):
                img = Image.open(img_path)
                axes[i, j].imshow(img)
                if j == 0:
                    axes[i, j].set_ylabel(class_name, fontsize=12)
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(f'Sample {j+1}', fontsize=10)
    
    plt.suptitle('Sample Images from Each Class', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved sample images to {save_path}")
    
    plt.show()


def analyze_image_sizes(data_path):
    """
    Analyze the sizes of images in the dataset.
    
    Args:
        data_path: Path to the data directory
        
    Returns:
        dict: Statistics about image sizes
    """
    widths = []
    heights = []
    
    for class_name in CLASSES:
        class_path = os.path.join(data_path, class_name)
        if os.path.exists(class_path):
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_path, img_file)
                    try:
                        with Image.open(img_path) as img:
                            widths.append(img.width)
                            heights.append(img.height)
                    except Exception as e:
                        print(f"Error reading {img_path}: {e}")
    
    stats = {
        'num_images': len(widths),
        'width_min': min(widths) if widths else 0,
        'width_max': max(widths) if widths else 0,
        'width_mean': np.mean(widths) if widths else 0,
        'height_min': min(heights) if heights else 0,
        'height_max': max(heights) if heights else 0,
        'height_mean': np.mean(heights) if heights else 0,
        'unique_sizes': len(set(zip(widths, heights)))
    }
    
    return stats


def plot_image_size_distribution(data_path, save_path=None):
    """
    Plot the distribution of image sizes.
    
    Args:
        data_path: Path to the data directory
        save_path: Path to save the plot (optional)
    """
    widths = []
    heights = []
    
    for class_name in CLASSES:
        class_path = os.path.join(data_path, class_name)
        if os.path.exists(class_path):
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_path, img_file)
                    try:
                        with Image.open(img_path) as img:
                            widths.append(img.width)
                            heights.append(img.height)
                    except Exception:
                        pass
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Width distribution
    ax1 = axes[0]
    ax1.hist(widths, bins=30, color='#4ECDC4', edgecolor='black', alpha=0.7)
    ax1.set_title('Image Width Distribution')
    ax1.set_xlabel('Width (pixels)')
    ax1.set_ylabel('Frequency')
    
    # Height distribution
    ax2 = axes[1]
    ax2.hist(heights, bins=30, color='#FF6B6B', edgecolor='black', alpha=0.7)
    ax2.set_title('Image Height Distribution')
    ax2.set_xlabel('Height (pixels)')
    ax2.set_ylabel('Frequency')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved image size distribution to {save_path}")
    
    plt.show()


def analyze_pixel_intensity(data_path, num_samples=100):
    """
    Analyze pixel intensity distribution across classes.
    
    Args:
        data_path: Path to the data directory
        num_samples: Number of samples to analyze per class
        
    Returns:
        dict: Pixel intensity statistics per class
    """
    intensity_stats = {}
    
    for class_name in CLASSES:
        class_path = os.path.join(data_path, class_name)
        if not os.path.exists(class_path):
            continue
            
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:num_samples]
        
        all_pixels = []
        for img_file in images:
            img_path = os.path.join(class_path, img_file)
            try:
                with Image.open(img_path) as img:
                    img_array = np.array(img.convert('RGB'))
                    all_pixels.extend(img_array.mean(axis=(0, 1)).tolist())
            except Exception:
                pass
        
        if all_pixels:
            intensity_stats[class_name] = {
                'mean': np.mean(all_pixels),
                'std': np.std(all_pixels),
                'min': np.min(all_pixels),
                'max': np.max(all_pixels)
            }
    
    return intensity_stats


def generate_eda_report(train_path, val_path, output_dir='../output'):
    """
    Generate a complete EDA report with all visualizations.
    
    Args:
        train_path: Path to training data
        val_path: Path to validation data
        output_dir: Directory to save output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("BEAN LEAF LESIONS CLASSIFICATION - EDA REPORT")
    print("=" * 60)
    
    # Dataset overview
    print("\n1. DATASET OVERVIEW")
    print("-" * 40)
    
    train_counts = count_images_per_class(train_path)
    val_counts = count_images_per_class(val_path)
    
    total_train = sum(train_counts.values())
    total_val = sum(val_counts.values())
    
    print(f"Total training images: {total_train}")
    print(f"Total validation images: {total_val}")
    print(f"Number of classes: {len(CLASSES)}")
    print(f"Classes: {', '.join(CLASSES)}")
    
    # Class distribution
    print("\n2. CLASS DISTRIBUTION")
    print("-" * 40)
    
    print("\nTraining Set:")
    for class_name, count in train_counts.items():
        percentage = (count / total_train) * 100 if total_train > 0 else 0
        print(f"  {class_name}: {count} images ({percentage:.1f}%)")
    
    print("\nValidation Set:")
    for class_name, count in val_counts.items():
        percentage = (count / total_val) * 100 if total_val > 0 else 0
        print(f"  {class_name}: {count} images ({percentage:.1f}%)")
    
    # Image size analysis
    print("\n3. IMAGE SIZE ANALYSIS")
    print("-" * 40)
    
    train_stats = analyze_image_sizes(train_path)
    print(f"Number of images analyzed: {train_stats['num_images']}")
    print(f"Width range: {train_stats['width_min']} - {train_stats['width_max']} pixels")
    print(f"Height range: {train_stats['height_min']} - {train_stats['height_max']} pixels")
    print(f"Mean width: {train_stats['width_mean']:.1f} pixels")
    print(f"Mean height: {train_stats['height_mean']:.1f} pixels")
    print(f"Unique sizes: {train_stats['unique_sizes']}")
    
    # Generate plots
    print("\n4. GENERATING VISUALIZATIONS...")
    print("-" * 40)
    
    # Class distribution plot
    plot_class_distribution(
        train_path, val_path, 
        save_path=os.path.join(output_dir, 'class_distribution.png')
    )
    
    # Sample images plot
    plot_sample_images(
        train_path, 
        num_samples=3,
        save_path=os.path.join(output_dir, 'sample_images.png')
    )
    
    # Image size distribution
    plot_image_size_distribution(
        train_path,
        save_path=os.path.join(output_dir, 'image_size_distribution.png')
    )
    
    # CSV Data Analysis
    print("\n5. CSV DATA ANALYSIS")
    print("-" * 40)
    
    train_csv, val_csv = get_csv_paths()
    
    if os.path.exists(train_csv) and os.path.exists(val_csv):
        print(f"Train CSV: {train_csv}")
        print(f"Val CSV: {val_csv}")
        
        train_csv_counts = get_csv_class_distribution(train_csv)
        val_csv_counts = get_csv_class_distribution(val_csv)
        
        print("\nTraining Set (from CSV):")
        total_csv_train = sum(train_csv_counts.values())
        for class_name, count in train_csv_counts.items():
            percentage = (count / total_csv_train) * 100 if total_csv_train > 0 else 0
            print(f"  {class_name}: {count} images ({percentage:.1f}%)")
        
        print("\nValidation Set (from CSV):")
        total_csv_val = sum(val_csv_counts.values())
        for class_name, count in val_csv_counts.items():
            percentage = (count / total_csv_val) * 100 if total_csv_val > 0 else 0
            print(f"  {class_name}: {count} images ({percentage:.1f}%)")
        
        # Category mapping info
        print("\nCategory Mapping (from CSV):")
        for cat_id, class_name in CATEGORY_MAP.items():
            print(f"  {cat_id}: {class_name}")
    else:
        print("CSV files not found. Skipping CSV analysis.")
    
    print("\n" + "=" * 60)
    print("EDA REPORT COMPLETED!")
    print("=" * 60)


if __name__ == '__main__':
    train_path, val_path = get_data_paths('../data')
    generate_eda_report(train_path, val_path)
