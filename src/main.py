"""
Main script for Bean Leaf Lesions Classification
Script chính để chạy demo phân loại vết bệnh trên lá đậu
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import (
    CLASSES, IMG_SIZE, CATEGORY_MAP,
    get_data_paths, create_data_generators,
    load_and_preprocess_image, count_images_per_class,
    get_csv_paths, get_csv_class_distribution
)
from model_minhhieu import (
    build_simple_cnn, build_transfer_learning_model,
    get_callbacks, train_model, save_model, load_model, predict_image
)
from evaluation import generate_evaluation_report


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def run_eda(data_path='../data'):
    """Run exploratory data analysis."""
    from eda import generate_eda_report
    train_path = os.path.join(data_path, 'train')
    val_path = os.path.join(data_path, 'val')
    generate_eda_report(train_path, val_path)


def train(model_type='simple_cnn', epochs=30, batch_size=32, data_path='../data', 
          output_dir='../models'):
    """
    Train a model for bean leaf classification.
    
    Args:
        model_type: Type of model ('simple_cnn', 'mobilenetv2', 'resnet50v2')
        epochs: Number of training epochs
        batch_size: Batch size for training
        data_path: Path to data directory
        output_dir: Directory to save models
    """
    print("=" * 60)
    print("BEAN LEAF LESIONS CLASSIFICATION - TRAINING")
    print("=" * 60)
    
    # Set seed
    set_seed(42)
    
    # Get data paths
    train_path = os.path.join(data_path, 'train')
    val_path = os.path.join(data_path, 'val')
    
    # Display dataset info
    print("\nDataset Information:")
    print("-" * 40)
    train_counts = count_images_per_class(train_path)
    val_counts = count_images_per_class(val_path)
    print(f"Training samples: {sum(train_counts.values())}")
    print(f"Validation samples: {sum(val_counts.values())}")
    print(f"Classes: {', '.join(CLASSES)}")
    
    # Create data generators
    print("\nCreating data generators...")
    train_gen, val_gen = create_data_generators(train_path, val_path, batch_size=batch_size)
    
    # Build model
    print(f"\nBuilding {model_type} model...")
    if model_type == 'simple_cnn':
        model = build_simple_cnn()
    else:
        model = build_transfer_learning_model(base_model_name=model_type)
    
    model.summary()
    
    # Get callbacks
    callbacks_list = get_callbacks(model_name=model_type, output_dir=output_dir)
    
    # Train model
    print("\nTraining model...")
    history = train_model(
        model=model,
        train_generator=train_gen,
        val_generator=val_gen,
        epochs=epochs,
        callbacks_list=callbacks_list
    )
    
    # Save final model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f'{model_type}_final.keras')
    save_model(model, model_path)
    
    # Evaluate model
    print("\nEvaluating model...")
    metrics = generate_evaluation_report(model, val_gen, history, output_dir='../output')
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Final model saved to: {model_path}")
    print("=" * 60)
    
    return model, history


def predict(model_path, image_path):
    """
    Predict the class of an image using a trained model.
    
    Args:
        model_path: Path to the trained model
        image_path: Path to the image to classify
    """
    print("=" * 60)
    print("BEAN LEAF LESIONS CLASSIFICATION - PREDICTION")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    model = load_model(model_path)
    
    # Make prediction
    print(f"\nAnalyzing image: {image_path}")
    predicted_class, confidence = predict_image(model, image_path)
    
    # Display result
    print("\n" + "-" * 40)
    print("PREDICTION RESULT")
    print("-" * 40)
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")
    
    # Map class to Vietnamese
    class_descriptions = {
        'angular_leaf_spot': 'Đốm góc lá (Angular Leaf Spot)',
        'bean_rust': 'Gỉ sắt đậu (Bean Rust)',
        'healthy': 'Khỏe mạnh (Healthy)'
    }
    print(f"Description: {class_descriptions.get(predicted_class, predicted_class)}")
    
    print("=" * 60)
    
    return predicted_class, confidence


def batch_predict(model_path, image_dir):
    """
    Predict classes for all images in a directory.
    
    Args:
        model_path: Path to the trained model
        image_dir: Directory containing images
    """
    print("=" * 60)
    print("BEAN LEAF LESIONS CLASSIFICATION - BATCH PREDICTION")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    model = load_model(model_path)
    
    # Get all images
    image_extensions = ('.png', '.jpg', '.jpeg')
    images = [f for f in os.listdir(image_dir) if f.lower().endswith(image_extensions)]
    
    print(f"\nFound {len(images)} images in {image_dir}")
    print("\n" + "-" * 60)
    
    results = []
    for img_name in images:
        img_path = os.path.join(image_dir, img_name)
        predicted_class, confidence = predict_image(model, img_path)
        results.append((img_name, predicted_class, confidence))
        print(f"{img_name:40s} | {predicted_class:20s} | {confidence:.2%}")
    
    print("-" * 60)
    print(f"Processed {len(results)} images")
    print("=" * 60)
    
    return results


def demo():
    """Run a quick demo of the classification system."""
    print("=" * 60)
    print("BEAN LEAF LESIONS CLASSIFICATION - DEMO")
    print("=" * 60)
    
    print("\nThis demo shows the bean leaf disease classification system.")
    print("\nClasses:")
    print("  1. Angular Leaf Spot - Đốm góc lá")
    print("  2. Bean Rust - Gỉ sắt đậu")
    print("  3. Healthy - Lá khỏe mạnh")
    
    print("\nCategory Mapping (from CSV):")
    for cat_id, class_name in CATEGORY_MAP.items():
        print(f"  {cat_id}: {class_name}")
    
    print("\nAvailable commands:")
    print("  python main.py eda               - Run exploratory data analysis")
    print("  python main.py train             - Train a new model")
    print("  python main.py predict <model> <image> - Classify an image")
    print("  python main.py batch <model> <dir>     - Classify all images in directory")
    
    # Show dataset info from directories
    data_path = '../data'
    if os.path.exists(data_path):
        train_path = os.path.join(data_path, 'train')
        val_path = os.path.join(data_path, 'val')
        
        if os.path.exists(train_path) and os.path.exists(val_path):
            train_counts = count_images_per_class(train_path)
            val_counts = count_images_per_class(val_path)
            
            print("\nDataset Summary (from directories):")
            print("-" * 40)
            print(f"Training images: {sum(train_counts.values())}")
            for class_name, count in train_counts.items():
                print(f"  - {class_name}: {count}")
            print(f"\nValidation images: {sum(val_counts.values())}")
            for class_name, count in val_counts.items():
                print(f"  - {class_name}: {count}")
    
    # Show dataset info from CSV files
    train_csv, val_csv = get_csv_paths()
    if os.path.exists(train_csv) and os.path.exists(val_csv):
        train_csv_counts = get_csv_class_distribution(train_csv)
        val_csv_counts = get_csv_class_distribution(val_csv)
        
        print("\nDataset Summary (from CSV files):")
        print("-" * 40)
        print(f"Train CSV: {os.path.basename(train_csv)}")
        print(f"  Total: {sum(train_csv_counts.values())} images")
        for class_name, count in train_csv_counts.items():
            print(f"  - {class_name}: {count}")
        
        print(f"\nVal CSV: {os.path.basename(val_csv)}")
        print(f"  Total: {sum(val_csv_counts.values())} images")
        for class_name, count in val_csv_counts.items():
            print(f"  - {class_name}: {count}")
    
    print("\n" + "=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Bean Leaf Lesions Classification System'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # EDA command
    eda_parser = subparsers.add_parser('eda', help='Run exploratory data analysis')
    eda_parser.add_argument('--data', default='../data', help='Path to data directory')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--model', default='simple_cnn', 
                             choices=['simple_cnn', 'mobilenetv2', 'resnet50v2'],
                             help='Model architecture')
    train_parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    train_parser.add_argument('--data', default='../data', help='Path to data directory')
    train_parser.add_argument('--output', default='../models', help='Output directory')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict image class')
    predict_parser.add_argument('model', help='Path to trained model')
    predict_parser.add_argument('image', help='Path to image')
    
    # Batch predict command
    batch_parser = subparsers.add_parser('batch', help='Batch predict images')
    batch_parser.add_argument('model', help='Path to trained model')
    batch_parser.add_argument('directory', help='Directory containing images')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run demo')
    
    args = parser.parse_args()
    
    if args.command == 'eda':
        run_eda(args.data)
    elif args.command == 'train':
        train(
            model_type=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            data_path=args.data,
            output_dir=args.output
        )
    elif args.command == 'predict':
        predict(args.model, args.image)
    elif args.command == 'batch':
        batch_predict(args.model, args.directory)
    elif args.command == 'demo':
        demo()
    else:
        demo()


if __name__ == '__main__':
    main()
