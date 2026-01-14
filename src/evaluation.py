# evaluation.py
# Danh gia mo hinh
# Tac gia: Minh Hieu

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from preprocessing import CLASSES, NUM_CLASSES, get_data_paths, create_data_generators


def evaluate_model(model, val_gen):
    """Danh gia model tren tap validation"""
    val_gen.reset()
    
    # Du doan
    predictions = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_gen.classes
    
    # Tinh cac chi so
    acc = accuracy_score(y_true, y_pred)
    
    return y_true, y_pred, predictions, acc


def print_report(y_true, y_pred):
    """In bao cao danh gia"""
    print("\n" + "="*50)
    print("BAO CAO DANH GIA MO HINH")
    print("="*50)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASSES))


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Ve ma tran nham lan"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Ma tran nham lan (Confusion Matrix)')
    plt.xlabel('Du doan')
    plt.ylabel('Thuc te')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Da luu: {save_path}")
    plt.show()


def plot_training_history(history, save_path=None):
    """Ve bieu do qua trinh training"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Training')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_title('Do chinh xac (Accuracy)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    
    # Loss
    axes[1].plot(history.history['loss'], label='Training')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title('Ham mat mat (Loss)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def full_evaluation(model, val_gen, history=None, output_dir='../output'):
    """Danh gia day du mo hinh"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Danh gia
    y_true, y_pred, _, acc = evaluate_model(model, val_gen)
    
    print(f"\nDo chinh xac tong: {acc:.4f} ({acc*100:.2f}%)")
    
    # In bao cao
    print_report(y_true, y_pred)
    
    # Ve confusion matrix
    plot_confusion_matrix(y_true, y_pred, 
                         os.path.join(output_dir, 'confusion_matrix.png'))
    
    # Ve training history neu co
    if history:
        plot_training_history(history,
                             os.path.join(output_dir, 'training_history.png'))
    
    return acc


# Test
if __name__ == '__main__':
    print("Module evaluation.py")
    print("Su dung: full_evaluation(model, val_gen, history)")
