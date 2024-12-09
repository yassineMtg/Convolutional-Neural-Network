import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

def make_confusion_matrix(y_true, y_pred, classes):
    """
    Plots a confusion matrix with labels.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        classes (list): List of class names.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, cbar=True, linewidths=0.5)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def plot_loss_curves(csv_path):
    """
    Plots training and validation loss and accuracy curves from a CSV file.

    Args:
        csv_path (str): Path to the CSV file containing training metrics.
    """
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Plot training and validation loss
    plt.figure(figsize=(12, 6))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(df['loss'], label='Training Loss', color='blue')
    plt.plot(df['val_loss'], label='Validation Loss', color='orange')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(df['accuracy'], label='Training Accuracy', color='blue')
    plt.plot(df['val_acc'], label='Validation Accuracy', color='orange')  # Adjust the column name if necessary
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Show plots
    plt.tight_layout()
    plt.show()

