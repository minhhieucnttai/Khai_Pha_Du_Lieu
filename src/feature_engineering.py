"""
Feature Engineering module for Bean Leaf Lesions Classification
Tạo và chọn đặc trưng

Note: For deep learning (CNN) models, feature engineering is done automatically
by the model through convolutional layers. This module provides additional
techniques for traditional machine learning approaches if needed.
"""

import os
import numpy as np
import cv2
from PIL import Image
from skimage.feature import hog, local_binary_pattern
from skimage import color, filters

from preprocessing import CLASSES, IMG_SIZE


def extract_color_histogram(image, bins=32):
    """
    Extract color histogram features from an image.
    
    Args:
        image: Input image (numpy array or PIL Image)
        bins: Number of bins for histogram
        
    Returns:
        np.array: Flattened histogram features
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert BGR to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Calculate histogram for each channel
        hist_features = []
        for channel in range(3):
            hist, _ = np.histogram(image[:, :, channel], bins=bins, range=(0, 256))
            hist = hist / hist.sum()  # Normalize
            hist_features.extend(hist)
        return np.array(hist_features)
    else:
        hist, _ = np.histogram(image.flatten(), bins=bins, range=(0, 256))
        return hist / hist.sum()


def extract_hog_features(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    """
    Extract Histogram of Oriented Gradients (HOG) features.
    
    Args:
        image: Input image (numpy array or PIL Image)
        orientations: Number of orientation bins
        pixels_per_cell: Size of a cell
        cells_per_block: Number of cells in each block
        
    Returns:
        np.array: HOG feature vector
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = color.rgb2gray(image)
    else:
        gray = image
    
    # Resize to standard size
    gray = cv2.resize(gray, IMG_SIZE)
    
    # Extract HOG features
    features = hog(
        gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm='L2-Hys',
        feature_vector=True
    )
    
    return features


def extract_lbp_features(image, radius=1, n_points=8, method='uniform'):
    """
    Extract Local Binary Pattern (LBP) features.
    
    Args:
        image: Input image (numpy array or PIL Image)
        radius: Radius of circle
        n_points: Number of circularly symmetric neighbor points
        method: LBP method
        
    Returns:
        np.array: LBP histogram features
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Calculate LBP
    lbp = local_binary_pattern(gray, n_points, radius, method=method)
    
    # Calculate histogram
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype('float')
    hist /= (hist.sum() + 1e-7)  # Normalize
    
    return hist


def extract_texture_features(image):
    """
    Extract various texture features from an image.
    
    Args:
        image: Input image (numpy array or PIL Image)
        
    Returns:
        np.array: Texture feature vector
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    features = []
    
    # Mean and std of pixel intensity
    features.append(np.mean(gray))
    features.append(np.std(gray))
    
    # Edge detection using Sobel
    sobel_x = filters.sobel_h(gray)
    sobel_y = filters.sobel_v(gray)
    features.append(np.mean(np.abs(sobel_x)))
    features.append(np.mean(np.abs(sobel_y)))
    
    # Contrast
    features.append(gray.max() - gray.min())
    
    # Entropy (measure of randomness) - using simple entropy approximation
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    hist = hist / hist.sum()
    entropy_val = -np.sum(hist * np.log2(hist + 1e-7))
    features.append(entropy_val)
    
    return np.array(features)


def extract_color_statistics(image):
    """
    Extract statistical features from color channels.
    
    Args:
        image: Input image (numpy array or PIL Image)
        
    Returns:
        np.array: Color statistics feature vector
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    features = []
    
    if len(image.shape) == 3 and image.shape[2] >= 3:
        for channel in range(3):
            channel_data = image[:, :, channel]
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.min(channel_data),
                np.max(channel_data),
                np.median(channel_data)
            ])
    else:
        features.extend([
            np.mean(image),
            np.std(image),
            np.min(image),
            np.max(image),
            np.median(image)
        ])
    
    return np.array(features)


def extract_shape_features(image, threshold=127):
    """
    Extract shape-based features using contours.
    
    Args:
        image: Input image (numpy array or PIL Image)
        threshold: Threshold for binary conversion
        
    Returns:
        np.array: Shape feature vector
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Binary threshold
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    features = []
    
    if contours:
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Area
        area = cv2.contourArea(largest_contour)
        features.append(area)
        
        # Perimeter
        perimeter = cv2.arcLength(largest_contour, True)
        features.append(perimeter)
        
        # Circularity
        circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-7)
        features.append(circularity)
        
        # Bounding box aspect ratio
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = float(w) / (h + 1e-7)
        features.append(aspect_ratio)
        
        # Extent (ratio of contour area to bounding box area)
        extent = area / (w * h + 1e-7)
        features.append(extent)
    else:
        features = [0, 0, 0, 0, 0]
    
    return np.array(features)


def extract_all_features(image):
    """
    Extract all features from an image.
    
    Args:
        image: Input image (numpy array or PIL Image)
        
    Returns:
        np.array: Combined feature vector
    """
    features = []
    
    # Color histogram
    color_hist = extract_color_histogram(image)
    features.extend(color_hist)
    
    # Color statistics
    color_stats = extract_color_statistics(image)
    features.extend(color_stats)
    
    # Texture features
    texture = extract_texture_features(image)
    features.extend(texture)
    
    # LBP features
    lbp = extract_lbp_features(image)
    features.extend(lbp)
    
    # Shape features
    shape = extract_shape_features(image)
    features.extend(shape)
    
    return np.array(features)


def extract_features_from_directory(data_path, feature_extractor=extract_all_features):
    """
    Extract features from all images in a directory.
    
    Args:
        data_path: Path to the data directory
        feature_extractor: Function to extract features from an image
        
    Returns:
        tuple: (features array, labels array)
    """
    all_features = []
    all_labels = []
    
    for class_idx, class_name in enumerate(CLASSES):
        class_path = os.path.join(data_path, class_name)
        if not os.path.exists(class_path):
            continue
        
        for img_file in os.listdir(class_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_path, img_file)
                try:
                    image = Image.open(img_path).convert('RGB')
                    image = image.resize(IMG_SIZE)
                    features = feature_extractor(image)
                    all_features.append(features)
                    all_labels.append(class_idx)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
    
    return np.array(all_features), np.array(all_labels)


if __name__ == '__main__':
    from preprocessing import get_data_paths, get_sample_images
    
    train_path, val_path = get_data_paths('../data')
    
    # Test feature extraction on sample images
    samples = get_sample_images(train_path, num_samples=1)
    
    print("Feature Extraction Test")
    print("=" * 50)
    
    for class_name, img_paths in samples.items():
        if img_paths:
            img_path = img_paths[0]
            image = Image.open(img_path).convert('RGB')
            image = image.resize(IMG_SIZE)
            
            print(f"\nClass: {class_name}")
            print(f"Image: {os.path.basename(img_path)}")
            
            # Test individual feature extractors
            print(f"  Color histogram features: {len(extract_color_histogram(image))}")
            print(f"  Color statistics features: {len(extract_color_statistics(image))}")
            print(f"  Texture features: {len(extract_texture_features(image))}")
            print(f"  LBP features: {len(extract_lbp_features(image))}")
            print(f"  Shape features: {len(extract_shape_features(image))}")
            
            all_features = extract_all_features(image)
            print(f"  Total features: {len(all_features)}")
