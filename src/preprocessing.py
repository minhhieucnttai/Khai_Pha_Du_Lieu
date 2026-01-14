"""
Preprocessing module for Bean Leaf Lesions Classification
Xử lý dữ liệu hình ảnh lá đậu
"""

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CLASSES = ['angular_leaf_spot', 'bean_rust', 'healthy']
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


if __name__ == '__main__':
    # Test preprocessing functions
    train_path, val_path = get_data_paths('../data')
    
    print("Train path:", train_path)
    print("Val path:", val_path)
    
    # Count images
    print("\nTraining data class distribution:")
    train_counts = count_images_per_class(train_path)
    for class_name, count in train_counts.items():
        print(f"  {class_name}: {count} images")
    
    print("\nValidation data class distribution:")
    val_counts = count_images_per_class(val_path)
    for class_name, count in val_counts.items():
        print(f"  {class_name}: {count} images")
