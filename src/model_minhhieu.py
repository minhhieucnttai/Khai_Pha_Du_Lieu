"""
CNN Model for Bean Leaf Lesions Classification
Model: Convolutional Neural Network
Author: Minh Hieu

Mô hình CNN để phân loại vết bệnh trên lá đậu
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.applications import MobileNetV2, ResNet50V2, EfficientNetB0

from preprocessing import (
    IMG_SIZE, BATCH_SIZE, NUM_CLASSES, CLASSES,
    get_data_paths, create_data_generators
)


def build_simple_cnn(input_shape=(224, 224, 3), num_classes=NUM_CLASSES):
    """
    Build a simple CNN model from scratch.
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # First Conv Block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Conv Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Conv Block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fourth Conv Block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_transfer_learning_model(base_model_name='mobilenetv2', input_shape=(224, 224, 3), 
                                  num_classes=NUM_CLASSES, freeze_base=True):
    """
    Build a transfer learning model using a pre-trained base.
    
    Args:
        base_model_name: Name of the base model ('mobilenetv2', 'resnet50v2', 'efficientnetb0')
        input_shape: Input image shape
        num_classes: Number of output classes
        freeze_base: Whether to freeze base model weights
        
    Returns:
        Compiled Keras model
    """
    # Select base model
    if base_model_name.lower() == 'mobilenetv2':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name.lower() == 'resnet50v2':
        base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name.lower() == 'efficientnetb0':
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Unknown base model: {base_model_name}")
    
    # Freeze base model
    base_model.trainable = not freeze_base
    
    # Build full model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_callbacks(model_name='bean_leaf_model', patience=10, output_dir='../models'):
    """
    Create callbacks for training.
    
    Args:
        model_name: Name for saved model
        patience: Patience for early stopping
        output_dir: Directory to save model checkpoints
        
    Returns:
        list: List of callbacks
    """
    os.makedirs(output_dir, exist_ok=True)
    
    callback_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, f'{model_name}_best.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.TensorBoard(
            log_dir=os.path.join(output_dir, 'logs'),
            histogram_freq=1
        )
    ]
    
    return callback_list


def train_model(model, train_generator, val_generator, epochs=50, callbacks_list=None):
    """
    Train the model.
    
    Args:
        model: Compiled Keras model
        train_generator: Training data generator
        val_generator: Validation data generator
        epochs: Number of training epochs
        callbacks_list: List of callbacks
        
    Returns:
        History object from training
    """
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks_list,
        verbose=1
    )
    
    return history


def predict_image(model, image_path):
    """
    Predict the class of a single image.
    
    Args:
        model: Trained Keras model
        image_path: Path to the image
        
    Returns:
        tuple: (predicted_class, confidence)
    """
    from preprocessing import load_and_preprocess_image
    
    img_array = load_and_preprocess_image(image_path)
    predictions = model.predict(img_array, verbose=0)
    
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx]
    predicted_class = CLASSES[predicted_idx]
    
    return predicted_class, confidence


def save_model(model, model_path):
    """
    Save the trained model.
    
    Args:
        model: Trained Keras model
        model_path: Path to save the model
    """
    model.save(model_path)
    print(f"Model saved to {model_path}")


def load_model(model_path):
    """
    Load a saved model.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded Keras model
    """
    return tf.keras.models.load_model(model_path)


def print_model_summary(model):
    """
    Print a summary of the model architecture.
    
    Args:
        model: Keras model
    """
    model.summary()
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")


if __name__ == '__main__':
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Get data paths
    train_path, val_path = get_data_paths('../data')
    
    print("=" * 60)
    print("BEAN LEAF LESIONS CLASSIFICATION - CNN MODEL TRAINING")
    print("=" * 60)
    
    # Create data generators
    print("\nCreating data generators...")
    train_gen, val_gen = create_data_generators(train_path, val_path)
    
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Classes: {train_gen.class_indices}")
    
    # Build model
    print("\n" + "-" * 40)
    print("Building Simple CNN Model...")
    print("-" * 40)
    
    model = build_simple_cnn()
    print_model_summary(model)
    
    # Get callbacks
    callbacks_list = get_callbacks(model_name='simple_cnn', patience=10)
    
    # Train model
    print("\n" + "-" * 40)
    print("Training model...")
    print("-" * 40)
    
    history = train_model(
        model=model,
        train_generator=train_gen,
        val_generator=val_gen,
        epochs=30,
        callbacks_list=callbacks_list
    )
    
    # Save final model
    save_model(model, '../models/simple_cnn_final.keras')
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
