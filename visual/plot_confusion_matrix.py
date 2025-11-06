import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import os

# Sample data for demonstration
y_true = ['-1', '0', '1', '2', '3', '-1', '0', '1', '2', '3', '-1', '0', '1', '2', '3']
y_pred = ['-1', '0', '1', '2', '2', '-1', '1', '1', '2', '3', '-1', '0', '2', '2', '3']

dataset_name = "xhs-SearchCorrelation"
model_name = "sample_model"

def plot_confusion_matrix(y_true, y_pred, classes):
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Get unique classes
    classes = unique_labels(y_true, y_pred)

    # Create confusion matrix visualization
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlGnBu',
                xticklabels=classes, yticklabels=classes)

    # Add title and labels
    plt.title(f'{dataset_name}-{model_name} Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    # Create temp directory if it doesn't exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(script_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    # Save the figure using absolute path
    output_path = os.path.join(temp_dir, f'{dataset_name}_{model_name}_confusion_matrix.png')
    plt.savefig(output_path, dpi=200)

    # Show the plot
    plt.show()
    plt.close()

