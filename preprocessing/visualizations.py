import matplotlib.pyplot as plt
import numpy as np
import os


# The function handle the structure of the metrics dictionary.
# It extracts precision, recall, and f1 for both classes, as well as accuracy, and plots them as a bar chart.
# Only to be called in the luke_siamese.py code
def plot_metrics_histogram(metrics, positive_pair_count, save_folder="experiments", filename=None):
    """
    Plots histograms for model metrics for a single `positive_pair_count` and saves the plot to a given folder.
    
    Args:
        metrics: Dictionary containing metrics for a specific positive pair count.
        positive_pair_count: The number of positive pairs per person used in the experiment.
        save_folder: Folder where the JPEG file will be saved. Default is 'experiments'.
        filename: Optional; name of the JPEG file. If None, a default name is generated.
    """
    # Extract values for plotting
    precision_0 = metrics['precision'][0]
    precision_1 = metrics['precision'][1]
    recall_0 = metrics['recall'][0]
    recall_1 = metrics['recall'][1]
    f1_0 = metrics['f1'][0]
    f1_1 = metrics['f1'][1]
    accuracy = metrics['accuracy']
    
    # Prepare the metric names and their corresponding values
    metric_names = ['accuracy', 'precision_0', 'recall_0', 'f1_0', 'precision_1', 'recall_1', 'f1_1']
    values = [accuracy, precision_0, recall_0, f1_0, precision_1, recall_1, f1_1]
    
    # Plotting
    plt.figure(figsize=(15, 8))
    plt.bar(metric_names, values, color='lightblue')
    plt.title(f'Model Performance Metrics for Positive Pair Count = {positive_pair_count}')
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Ensure the filename is set if not provided
    if not filename:
        filename = f'metrics_histogram_{positive_pair_count}.jpeg'
    
    # Save the plot
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, filename)
    plt.savefig(save_path, format='jpeg')
    print(f"Metrics histogram saved as {save_path}")
    
    plt.show()


# Generates performance chart only for a single model at a time
# Only to be called in the luke_siamese.py code
def plot_model_performance1(history, model_name, test_accuracy=None, save_folder='experiments', filename=None):
    """
    Plots the training and validation accuracy curves for a single model and marks the test accuracy.

    Args:
        history: Training history object from model training.
        model_name: Name of the model.
        test_accuracy (float, optional): Test accuracy for the model.
        save_folder: Folder where the JPEG file will be saved. Default is 'experiments'
        filename: Optional; name of the JPEG file. If None, a default name is generated.
    """

    plt.figure(figsize=(14, 7))

    # Plot training and validation accuracy
    plt.plot(history.history['accuracy'], label=f'{model_name} - Training Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label=f'{model_name} - Validation Accuracy')

    # Plot test accuracy as a point if provided
    if test_accuracy is not None:
        plt.scatter(len(history.history['accuracy']) - 1, test_accuracy, color='red', label=f'{model_name} - Test Accuracy', marker='o')

    # Customize the plot
    plt.title(f'{model_name} Performance: Training, Validation, and Testing Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)

     # Ensure the filename is set if not provided
    if not filename:
        filename = f'model_performance_{model_name}.jpeg'
    
    # Save the plot
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, filename)
    plt.savefig(save_path, format='jpeg')
    print(f"Model Plot saved as {save_path}")
    
    plt.show()

# Generates performance chart only for both ResNet50 and VGG19
# Only to be called in the luke_siamese_model_performance_exp.py code
def plot_model_performance2(histories, model_names, test_accuracies=None, save_folder='experiments', filename=None):
    """
    Plots the training and validation accuracy curves for multiple models and marks their test accuracies.

    Args:
        histories: List of training history objects from model training.
        model_names: List of model names corresponding to the histories.
        test_accuracies (list, optional): List of test accuracies for each model.
        save_folder: Folder where the JPEG file will be saved. Default is 'experiments'.
        filename: Optional; name of the JPEG file. If None, a default name is generated.
    """

    plt.figure(figsize=(14, 7))

    for i, history in enumerate(histories):
        model_name = model_names[i]
        plt.plot(history.history['accuracy'], label=f'{model_name} - Training Accuracy')
        if 'val_accuracy' in history.history:
            plt.plot(history.history['val_accuracy'], label=f'{model_name} - Validation Accuracy')
        
        if test_accuracies and i < len(test_accuracies):
            plt.scatter(len(history.history['accuracy']) - 1, test_accuracies[i], color='red', label=f'{model_name} - Test Accuracy', marker='o')

    plt.title('Model Performance: Training, Validation, and Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)

    if not filename:
        filename = f'model_performance_comparison.jpeg'

    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, filename)
    plt.savefig(save_path, format='jpeg')
    print(f"Model plot saved as {save_path}")

    plt.show()


