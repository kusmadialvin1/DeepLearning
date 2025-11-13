import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import learning_curve
import tensorflow as tf
import cv2
import os

def plot_confusion_matrix(y_true, y_pred, class_names, model_name='Model'):
    """
    Plot confusion matrix for model predictions.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        model_name: Name of the model for plot title
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'reports/{model_name.lower()}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_confusion_matrix(y_true, y_pred, class_names, top_k=5):
    """
    Analyze the confusion matrix to find most confused classes.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        top_k: Number of top confused pairs to return

    Returns:
        confused_pairs: List of tuples (true_class, pred_class, count)
    """
    cm = confusion_matrix(y_true, y_pred)

    # Find off-diagonal elements (misclassifications)
    confused_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                confused_pairs.append((class_names[i], class_names[j], cm[i, j]))

    # Sort by count (most confused first)
    confused_pairs.sort(key=lambda x: x[2], reverse=True)

    print(f"Top {min(top_k, len(confused_pairs))} most confused class pairs:")
    for true_class, pred_class, count in confused_pairs[:top_k]:
        print(f"{true_class} -> {pred_class}: {count} times")

    return confused_pairs[:top_k]

def plot_feature_maps(model, img, layer_name, model_name='Model'):
    """
    Visualize feature maps from a convolutional layer.

    Args:
        model: Trained Keras model
        img: Input image (preprocessed)
        layer_name: Name of the layer to visualize
        model_name: Name of the model for plot title
    """
    # Get the layer output
    layer_output = model.get_layer(layer_name).output

    # Create a model that outputs the feature maps
    feature_map_model = tf.keras.Model(inputs=model.input, outputs=layer_output)

    # Get feature maps
    feature_maps = feature_map_model.predict(np.expand_dims(img, axis=0))

    # Plot first 16 feature maps
    num_maps = min(16, feature_maps.shape[-1])
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()

    for i in range(num_maps):
        axes[i].imshow(feature_maps[0, :, :, i], cmap='viridis')
        axes[i].set_title(f'Feature Map {i+1}')
        axes[i].axis('off')

    plt.suptitle(f'{model_name} Feature Maps from {layer_name}')
    plt.tight_layout()
    plt.savefig(f'reports/{model_name.lower()}_feature_maps_{layer_name}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_hyperparameter_sensitivity(results, param_name, model_name='Model'):
    """
    Plot hyperparameter sensitivity analysis.

    Args:
        results: Dictionary with parameter values as keys and metrics as values
        param_name: Name of the hyperparameter
        model_name: Name of the model
    """
    param_values = list(results.keys())
    accuracies = [results[val]['accuracy'] for val in param_values]
    losses = [results[val]['loss'] for val in param_values]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(param_values, accuracies, 'o-', color='blue')
    ax1.set_title(f'{model_name} Accuracy vs {param_name}')
    ax1.set_xlabel(param_name)
    ax1.set_ylabel('Accuracy')
    ax1.grid(True)

    ax2.plot(param_values, losses, 'o-', color='red')
    ax2.set_title(f'{model_name} Loss vs {param_name}')
    ax2.set_xlabel(param_name)
    ax2.set_ylabel('Loss')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f'reports/{model_name.lower()}_{param_name.lower()}_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_model_comparison(models_results, metric='accuracy'):
    """
    Plot comparison of different models.

    Args:
        models_results: Dictionary with model names as keys and results as values
        metric: Metric to compare ('accuracy' or 'loss')
    """
    model_names = list(models_results.keys())
    values = [models_results[name][metric] for name in model_names]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    plt.title(f'Model Comparison - {metric.capitalize()}')
    plt.xlabel('Model')
    plt.ylabel(metric.capitalize())
    plt.ylim(0, 1 if metric == 'accuracy' else max(values) * 1.1)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'reports/model_comparison_{metric}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_data_augmentation_comparison(results_no_aug, results_aug, model_name='Model'):
    """
    Plot comparison of model performance with and without data augmentation.

    Args:
        results_no_aug: Results without augmentation
        results_aug: Results with augmentation
        model_name: Name of the model
    """
    labels = ['No Augmentation', 'With Augmentation']
    accuracies = [results_no_aug['accuracy'], results_aug['accuracy']]
    losses = [results_no_aug['loss'], results_aug['loss']]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    bars1 = ax1.bar(labels, accuracies, color=['lightcoral', 'lightgreen'])
    ax1.set_title(f'{model_name} Accuracy Comparison')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)

    bars2 = ax2.bar(labels, losses, color=['lightcoral', 'lightgreen'])
    ax2.set_title(f'{model_name} Loss Comparison')
    ax2.set_ylabel('Loss')

    # Add value labels
    for ax, bars, values in [(ax1, bars1, accuracies), (ax2, bars2, losses)]:
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f'reports/{model_name.lower()}_augmentation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_classification_report(y_true, y_pred, class_names, model_name='Model'):
    """
    Generate and save detailed classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        model_name: Name of the model
    """
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    # Save as text file
    with open(f'reports/{model_name.lower()}_classification_report.txt', 'w') as f:
        f.write(f"{model_name} Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(classification_report(y_true, y_pred, target_names=class_names))

    return report

if __name__ == "__main__":
    print("Analysis module loaded. Use functions to analyze model performance and generate visualizations.")
