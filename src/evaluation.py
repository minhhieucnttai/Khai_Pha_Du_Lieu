"""
Model Evaluation module for Bean Leaf Lesions Classification
Đánh giá mô hình
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import tensorflow as tf

from preprocessing import CLASSES, NUM_CLASSES, get_data_paths, create_data_generators


def evaluate_model(model, val_generator):
    """
    Evaluate model performance on validation data.
    
    Args:
        model: Trained Keras model
        val_generator: Validation data generator
        
    Returns:
        dict: Evaluation metrics
    """
    # Get predictions
    val_generator.reset()
    predictions = model.predict(val_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_generator.classes
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted')
    }
    
    return metrics, y_true, y_pred, predictions


def print_evaluation_report(metrics, y_true, y_pred):
    """
    Print a comprehensive evaluation report.
    
    Args:
        metrics: Dictionary of metrics
        y_true: True labels
        y_pred: Predicted labels
    """
    print("\n" + "=" * 60)
    print("MODEL EVALUATION REPORT")
    print("=" * 60)
    
    print("\n1. OVERALL METRICS")
    print("-" * 40)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (macro): {metrics['recall_macro']:.4f}")
    print(f"F1-Score (macro): {metrics['f1_macro']:.4f}")
    
    print("\n2. CLASSIFICATION REPORT")
    print("-" * 40)
    print(classification_report(y_true, y_pred, target_names=CLASSES))


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the plot (optional)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=CLASSES,
        yticklabels=CLASSES
    )
    plt.title('Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    
    plt.show()


def plot_normalized_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot normalized confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the plot (optional)
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized, 
        annot=True, 
        fmt='.2%', 
        cmap='Blues',
        xticklabels=CLASSES,
        yticklabels=CLASSES
    )
    plt.title('Normalized Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved normalized confusion matrix to {save_path}")
    
    plt.show()


def plot_training_history(history, save_path=None):
    """
    Plot training history (accuracy and loss curves).
    
    Args:
        history: Training history object
        save_path: Path to save the plot (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    ax1 = axes[0]
    ax1.plot(history.history['accuracy'], label='Training Accuracy', color='#4ECDC4')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', color='#FF6B6B')
    ax1.set_title('Model Accuracy', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Loss plot
    ax2 = axes[1]
    ax2.plot(history.history['loss'], label='Training Loss', color='#4ECDC4')
    ax2.plot(history.history['val_loss'], label='Validation Loss', color='#FF6B6B')
    ax2.set_title('Model Loss', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training history to {save_path}")
    
    plt.show()


def plot_roc_curves(y_true, predictions, save_path=None):
    """
    Plot ROC curves for each class.
    
    Args:
        y_true: True labels
        predictions: Model predictions (probabilities)
        save_path: Path to save the plot (optional)
    """
    # Binarize true labels
    y_true_bin = label_binarize(y_true, classes=range(NUM_CLASSES))
    
    plt.figure(figsize=(10, 8))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, (class_name, color) in enumerate(zip(CLASSES, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], predictions[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, linewidth=2,
                label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Class', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC curves to {save_path}")
    
    plt.show()


def plot_per_class_metrics(y_true, y_pred, save_path=None):
    """
    Plot per-class precision, recall, and F1 scores.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the plot (optional)
    """
    precision_per_class = precision_score(y_true, y_pred, average=None)
    recall_per_class = recall_score(y_true, y_pred, average=None)
    f1_per_class = f1_score(y_true, y_pred, average=None)
    
    x = np.arange(len(CLASSES))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width, precision_per_class, width, label='Precision', color='#FF6B6B')
    bars2 = ax.bar(x, recall_per_class, width, label='Recall', color='#4ECDC4')
    bars3 = ax.bar(x + width, f1_per_class, width, label='F1-Score', color='#45B7D1')
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Metrics', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved per-class metrics to {save_path}")
    
    plt.show()


def generate_evaluation_report(model, val_generator, history=None, output_dir='../output'):
    """
    Generate a comprehensive evaluation report with all visualizations.
    
    Args:
        model: Trained Keras model
        val_generator: Validation data generator
        history: Training history object (optional)
        output_dir: Directory to save output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate model
    metrics, y_true, y_pred, predictions = evaluate_model(model, val_generator)
    
    # Print report
    print_evaluation_report(metrics, y_true, y_pred)
    
    # Generate plots
    print("\nGenerating evaluation plots...")
    
    # Confusion matrix
    plot_confusion_matrix(
        y_true, y_pred,
        save_path=os.path.join(output_dir, 'confusion_matrix.png')
    )
    
    # Normalized confusion matrix
    plot_normalized_confusion_matrix(
        y_true, y_pred,
        save_path=os.path.join(output_dir, 'confusion_matrix_normalized.png')
    )
    
    # ROC curves
    plot_roc_curves(
        y_true, predictions,
        save_path=os.path.join(output_dir, 'roc_curves.png')
    )
    
    # Per-class metrics
    plot_per_class_metrics(
        y_true, y_pred,
        save_path=os.path.join(output_dir, 'per_class_metrics.png')
    )
    
    # Training history (if provided)
    if history is not None:
        plot_training_history(
            history,
            save_path=os.path.join(output_dir, 'training_history.png')
        )
    
    print("\n" + "=" * 60)
    print("Evaluation report completed!")
    print(f"Plots saved to: {output_dir}")
    print("=" * 60)
    
    return metrics


if __name__ == '__main__':
    # Example usage
    train_path, val_path = get_data_paths('../data')
    
    print("Creating validation data generator...")
    _, val_gen = create_data_generators(train_path, val_path)
    
    print("\nTo evaluate a trained model, use:")
    print("  from evaluation import generate_evaluation_report")
    print("  from model_minhhieu import load_model")
    print("  model = load_model('path/to/model.keras')")
    print("  metrics = generate_evaluation_report(model, val_gen)")
