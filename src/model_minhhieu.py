# model_minhhieu.py
# Mo hinh CNN phan loai benh la dau
# Tac gia: Minh Hieu
# Mo ta: Xay dung va huan luyen mo hinh CNN

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2

from preprocessing import IMG_SIZE, NUM_CLASSES, CLASSES, get_data_paths, create_data_generators


def build_cnn_model(input_shape=(224, 224, 3), num_classes=NUM_CLASSES):
    """
    Xay dung mo hinh CNN don gian
    Gom 3 block Conv + 2 Dense layer
    """
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Block 2  
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Block 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Fully connected
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_mobilenet_model(input_shape=(224, 224, 3), num_classes=NUM_CLASSES):
    """
    Xay dung mo hinh dung MobileNetV2 (Transfer Learning)
    """
    # Load MobileNetV2 pre-trained
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Dong bang cac layer
    
    model = Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_callbacks(model_name='model', save_dir='../models'):
    """Tao cac callback cho training"""
    os.makedirs(save_dir, exist_ok=True)
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(save_dir, f'{model_name}_best.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    return callbacks


def train_model(model, train_gen, val_gen, epochs=30, callbacks=None):
    """Huan luyen mo hinh"""
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    return history


def predict_image(model, img_path):
    """Du doan 1 anh"""
    from preprocessing import load_image
    
    img = load_image(img_path)
    pred = model.predict(img, verbose=0)
    
    class_idx = np.argmax(pred[0])
    confidence = pred[0][class_idx]
    class_name = CLASSES[class_idx]
    
    return class_name, confidence


def save_model(model, path):
    """Luu mo hinh"""
    model.save(path)
    print(f"Da luu model: {path}")


def load_model(path):
    """Load mo hinh da luu"""
    return tf.keras.models.load_model(path)


# Main - chay thu
if __name__ == '__main__':
    print("="*50)
    print("HUAN LUYEN MO HINH CNN - BEAN LEAF DISEASE")
    print("="*50)
    
    # Lay data
    train_path, val_path = get_data_paths()
    train_gen, val_gen = create_data_generators(train_path, val_path)
    
    print(f"\nSo anh training: {train_gen.samples}")
    print(f"So anh validation: {val_gen.samples}")
    print(f"Classes: {train_gen.class_indices}")
    
    # Xay dung model
    print("\nXay dung mo hinh CNN...")
    model = build_cnn_model()
    model.summary()
    
    # Huan luyen
    print("\nBat dau huan luyen...")
    callbacks = get_callbacks('cnn_model')
    history = train_model(model, train_gen, val_gen, epochs=30, callbacks=callbacks)
    
    # Luu model
    save_model(model, '../models/cnn_model_final.keras')
    
    print("\nHoan thanh!")
