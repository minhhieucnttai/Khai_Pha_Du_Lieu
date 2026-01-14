"""
Preprocessing module for Bean Leaf Lesions Classification
Xử lý dữ liệu hình ảnh lá đậu
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Category mapping from CSV (numeric to class name)
# CSV uses: 0 = healthy, 1 = angular_leaf_spot, 2 = bean_rust
CATEGORY_MAP = {0: 'healthy', 1: 'angular_leaf_spot', 2: 'bean_rust'}

# Classes list ordered by category ID (for model training)
CLASSES = list(CATEGORY_MAP.values())  # ['healthy', 'angular_leaf_spot', 'bean_rust']
NUM_CLASSES = len(CLASSES)


def get_data_paths(base_path=None):
    """
    Get paths to train and validation data directories.
    
    Args:
        base_path: Base path to the data folder. If None, defaults to '../data'
                   relative to this file's directory.
        
    Returns:
        tuple: (train_path, val_path)
    """
    if base_path is None:
        # Use path relative to this file's location
        base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    train_path = os.path.join(base_path, 'train')
    val_path = os.path.join(base_path, 'val')
    return train_path, val_path


def create_data_generators(train_path, val_path, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """
    Create data generators for training and validation with data augmentation.
    
    Args:
        train_path: Path to training data
        val_path: Path to validation data
        img_size: Target image size (width, height)
        batch_size: Batch size for training
        
    Returns:
        tuple: (train_generator, val_generator)
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=CLASSES,
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=CLASSES,
        shuffle=False
    )
    
    return train_generator, val_generator


def load_and_preprocess_image(image_path, img_size=IMG_SIZE):
    """
    Load and preprocess a single image for prediction.
    
    Args:
        image_path: Path to the image file
        img_size: Target image size
        
    Returns:
        np.array: Preprocessed image array
    """
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = img.resize(img_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def count_images_per_class(data_path):
    """
    Count the number of images in each class.
    
    Args:
        data_path: Path to the data directory
        
    Returns:
        dict: Dictionary with class names and image counts
    """
    class_counts = {}
    for class_name in CLASSES:
        class_path = os.path.join(data_path, class_name)
        if os.path.exists(class_path):
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            class_counts[class_name] = len(images)
    return class_counts


def get_sample_images(data_path, num_samples=3):
    """
    Get sample image paths from each class.
    
    Args:
        data_path: Path to the data directory
        num_samples: Number of samples per class
        
    Returns:
        dict: Dictionary with class names and sample image paths
    """
    samples = {}
    for class_name in CLASSES:
        class_path = os.path.join(data_path, class_name)
        if os.path.exists(class_path):
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            samples[class_name] = [
                os.path.join(class_path, img) 
                for img in images[:num_samples]
            ]
    return samples


def get_csv_paths(base_path=None):
    """
    Get paths to train and validation CSV files.
    
    Args:
        base_path: Base path to the data folder. If None, defaults to repository root.
        
    Returns:
        tuple: (train_csv_path, val_csv_path)
    """
    if base_path is None:
        base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    train_csv = os.path.join(base_path, 'train.csv')
    val_csv = os.path.join(base_path, 'val.csv')
    return train_csv, val_csv


def load_csv_data(csv_path, base_path=None):
    """
    Load data from CSV file.
    
    Args:
        csv_path: Path to the CSV file
        base_path: Base path to prepend to image paths
        
    Returns:
        pd.DataFrame: DataFrame with image paths and labels
    """
    df = pd.read_csv(csv_path)
    
    # Validate and normalize column names
    # Expected columns: image:FILE (or image_path) and category
    expected_cols = ['image_path', 'category']
    if len(df.columns) >= 2:
        # Rename first two columns to standard names
        df.columns = expected_cols[:len(df.columns)]
    else:
        raise ValueError(f"CSV must have at least 2 columns (image path and category). Found: {df.columns.tolist()}")
    
    # Convert category to class name
    df['class_name'] = df['category'].map(CATEGORY_MAP)
    
    # If base_path provided, prepend to image paths
    if base_path:
        df['image_path'] = df['image_path'].apply(lambda x: os.path.join(base_path, x))
    
    return df


def create_data_generators_from_csv(train_csv, val_csv, base_path=None, 
                                     img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """
    Create data generators from CSV files.
    
    Args:
        train_csv: Path to training CSV file
        val_csv: Path to validation CSV file
        base_path: Base path for image paths in CSV
        img_size: Target image size
        batch_size: Batch size for training
        
    Returns:
        tuple: (train_generator, val_generator)
    """
    # Load CSV data
    train_df = load_csv_data(train_csv, base_path)
    val_df = load_csv_data(val_csv, base_path)
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators from dataframes
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='image_path',
        y_col='class_name',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=CLASSES,
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='image_path',
        y_col='class_name',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=CLASSES,
        shuffle=False
    )
    
    return train_generator, val_generator


def get_csv_class_distribution(csv_path):
    """
    Get class distribution from CSV file.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        dict: Dictionary with class names and counts
    """
    # Use load_csv_data which handles column normalization
    df = load_csv_data(csv_path)
    distribution = df['class_name'].value_counts().to_dict()
    return distribution


if __name__ == '__main__':
    # Test preprocessing functions
    train_path, val_path = get_data_paths('../data')
    
    print("Train path:", train_path)
    print("Val path:", val_path)
    
    # Count images from directories
    print("\nTraining data class distribution (from directories):")
    train_counts = count_images_per_class(train_path)
    for class_name, count in train_counts.items():
        print(f"  {class_name}: {count} images")
    
    print("\nValidation data class distribution (from directories):")
    val_counts = count_images_per_class(val_path)
    for class_name, count in val_counts.items():
        print(f"  {class_name}: {count} images")
    
    # Test CSV functions
    train_csv, val_csv = get_csv_paths()
    print(f"\nTrain CSV: {train_csv}")
    print(f"Val CSV: {val_csv}")
    
    if os.path.exists(train_csv):
        print("\nTraining data class distribution (from CSV):")
        train_csv_counts = get_csv_class_distribution(train_csv)
        for class_name, count in train_csv_counts.items():
            print(f"  {class_name}: {count} images")
    
    if os.path.exists(val_csv):
        print("\nValidation data class distribution (from CSV):")
        val_csv_counts = get_csv_class_distribution(val_csv)
        for class_name, count in val_csv_counts.items():
            print(f"  {class_name}: {count} images")
