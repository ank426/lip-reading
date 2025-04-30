import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import json
from src import config

def save_class_mappings(class_list):
    """Saves the index-to-word mapping for inference later."""
    with open(config.CLASS_MAPPING_PATH, 'w') as f:
        json.dump(class_list, f)

def load_class_mappings():
    """Loads class mappings."""
    if not os.path.exists(config.CLASS_MAPPING_PATH):
        raise FileNotFoundError("Class mapping file not found. Train the model first.")
    with open(config.CLASS_MAPPING_PATH, 'r') as f:
        return json.load(f)

def extract_mouth_region(frame):
    """
    Simplified mouth region extraction.
    Note: For production, consider using dlib or mediapipe for accurate face landmarks.
    """
    h, w = frame.shape[:2]
    mouth_region = frame[h//2:, w//4:3*w//4]
    return mouth_region if mouth_region.size > 0 else frame

def plot_history(history):
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss History')

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy History')

    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, 'training_history.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # Normalize

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Normalized)')
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, 'confusion_matrix.png'))
    plt.close()
