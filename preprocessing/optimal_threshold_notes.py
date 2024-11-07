import tensorflow as tf
import numpy as np
import csv
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import cv2
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report

def find_optimal_threshold(model, validation_pairs, validation_labels, thresholds=None, metrics='all'):
    """
    Find the optimal threshold for face verification using multiple metrics.
    
    Args:
        model: Trained siamese network
        validation_pairs: Tuple of (pairs_0, pairs_1) validation image pairs
        validation_labels: Ground truth labels (0: different, 1: same person)
        thresholds: List of thresholds to try (default: np.arange(0.1, 2.0, 0.1))
        metrics: 'all' or list of metrics to compute ('accuracy', 'precision', 'recall', 'f1')
    
    Returns:
        dict: Best thresholds for each metric and their scores
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 2.0, 0.1)
    
    # Get distances for all pairs
    distances = model.predict(validation_pairs)
    
    # Initialize results storage
    results = {
        'accuracy': {'threshold': 0, 'score': 0},
        'precision': {'threshold': 0, 'score': 0},
        'recall': {'threshold': 0, 'score': 0},
        'f1': {'threshold': 0, 'score': 0},
        'balanced_accuracy': {'threshold': 0, 'score': 0}
    }
    
    # Store all metrics for plotting
    threshold_metrics = {
        'thresholds': thresholds,
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'balanced_accuracy': []
    }
    
    print("Testing thresholds...")
    for threshold in thresholds:
        # Convert distances to predictions based on threshold
        predictions = (distances < threshold).astype(int)
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(validation_labels, predictions).ravel()
        
        # Calculate all metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        balanced_accuracy = ((tp / (tp + fn) if (tp + fn) > 0 else 0) + 
                           (tn / (tn + fp) if (tn + fp) > 0 else 0)) / 2
        
        # Store metrics for plotting
        threshold_metrics['accuracy'].append(accuracy)
        threshold_metrics['precision'].append(precision)
        threshold_metrics['recall'].append(recall)
        threshold_metrics['f1'].append(f1)
        threshold_metrics['balanced_accuracy'].append(balanced_accuracy)
        
        # Update best results
        if accuracy > results['accuracy']['score']:
            results['accuracy'] = {'threshold': threshold, 'score': accuracy}
        if precision > results['precision']['score']:
            results['precision'] = {'threshold': threshold, 'score': precision}
        if recall > results['recall']['score']:
            results['recall'] = {'threshold': threshold, 'score': recall}
        if f1 > results['f1']['score']:
            results['f1'] = {'threshold': threshold, 'score': f1}
        if balanced_accuracy > results['balanced_accuracy']['score']:
            results['balanced_accuracy'] = {'threshold': threshold, 'score': balanced_accuracy}
    
    # Print results
    print("\nOptimal thresholds for different metrics:")
    for metric, result in results.items():
        print(f"{metric.capitalize()}:")
        print(f"  Threshold: {result['threshold']:.2f}")
        print(f"  Score: {result['score']:.4f}")
    
    return results, threshold_metrics

def analyze_threshold_selection(threshold_metrics, save_plot=False):
    """
    Analyze and visualize threshold selection results.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot all metrics
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'balanced_accuracy']:
        plt.plot(threshold_metrics['thresholds'], 
                threshold_metrics[metric], 
                label=metric.capitalize())
    
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Metric Scores vs. Threshold')
    plt.legend()
    plt.grid(True)
    
    if save_plot:
        plt.savefig('threshold_analysis.png')
    plt.show()

def test_threshold_examples(model, test_pairs, threshold):
    """
    Test specific examples with the chosen threshold to understand its behavior.
    
    Args:
        model: Trained siamese network
        test_pairs: Tuple of (pairs_0, pairs_1) test image pairs
        threshold: Chosen threshold for verification
    """
    # Get distances for test pairs
    distances = model.predict(test_pairs)
    
    # Print some example results
    print("\nExample predictions with threshold {:.2f}:".format(threshold))
    for i in range(min(5, len(distances))):
        print(f"Pair {i+1}:")
        print(f"  Distance: {distances[i][0]:.4f}")
        print(f"  Prediction: {'Same person' if distances[i][0] < threshold else 'Different people'}")
        print()

# Example usage:
"""
# Find optimal thresholds
results, threshold_metrics = find_optimal_threshold(
    model,
    (validation_pairs_0, validation_pairs_1),
    validation_labels
)

# Visualize results
analyze_threshold_selection(threshold_metrics)

# Test with specific examples
test_threshold_examples(
    model,
    (test_pairs_0, test_pairs_1),
    results['balanced_accuracy']['threshold']
)
"""