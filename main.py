import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Add src to Python path to import your custom modules
sys.path.append('src')

# Import your custom modules
from preprocess import load_train_data, load_test_data, create_data_generators
from deep_learning import create_resnet_model, create_vgg_model, compile_model, train_model, plot_training_history, save_model
from traditional_ml import extract_hog_features, train_svm_classifier, evaluate_model, save_model as save_traditional_model

def run_deep_learning_experiment():
    """Run deep learning experiments with ResNet and VGG"""
    print("=" * 60)
    print("DEEP LEARNING EXPERIMENT")
    print("=" * 60)
    
    # Load data - using correct paths for your structure
    print("Loading data...")
    X_train_full, y_train_full, class_names = load_train_data(train_path='TRAIN')
    X_test, y_test, _ = load_test_data(test_path='TEST')
    
    # Split data for deep learning
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42
    )
    
    print(f"Classes: {class_names}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create data generators
    train_gen, val_gen = create_data_generators(X_train, y_train, X_val, y_val, batch_size=32, augmentation=True)
    
    # Experiment with different models
    models = {
        'resnet': create_resnet_model(len(class_names)),
        'vgg': create_vgg_model(len(class_names))
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name.upper()}...")
        
        # Compile model
        model = compile_model(model, learning_rate=1e-4)
        
        # Train model
        history = train_model(model, train_gen, val_gen, epochs=50, model_name=model_name)
        
        # Plot training history
        plot_training_history(history, model_name.upper())
        
        # Evaluate on test set
        X_test_normalized = X_test / 255.0
        test_loss, test_accuracy = model.evaluate(X_test_normalized, y_test, verbose=0)
        
        # Predictions
        y_pred = model.predict(X_test_normalized)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        results[model_name] = {
            'model': model,
            'history': history,
            'test_accuracy': test_accuracy,
            'y_pred': y_pred_classes,
            'y_true': y_test
        }
        
        print(f"{model_name.upper()} Test Accuracy: {test_accuracy:.4f}")
        
        # Save model
        save_model(model, model_name)
    
    return results, class_names

def run_traditional_ml_experiment():
    """Run traditional machine learning experiment with HOG + SVM"""
    print("\n" + "=" * 60)
    print("TRADITIONAL ML EXPERIMENT (HOG + SVM)")
    print("=" * 60)
    
    # Load data - using smaller images for HOG
    X_train_full, y_train_full, class_names = load_train_data(train_path='TRAIN', img_size=(128, 128))
    X_test, y_test, _ = load_test_data(test_path='TEST', img_size=(128, 128))
    
    # Split data for traditional ML
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42
    )
    
    print("Extracting HOG features...")
    # Extract HOG features
    X_train_features = extract_hog_features(X_train)
    X_test_features = extract_hog_features(X_test)
    
    print(f"HOG feature dimensions: {X_train_features.shape}")
    
    # Train SVM classifier
    print("Training SVM classifier...")
    svm_classifier, scaler = train_svm_classifier(
        X_train_features, y_train,
        C=1.0, kernel='rbf', gamma='scale'
    )
    
    # Evaluate
    accuracy, report, y_pred = evaluate_model(
        svm_classifier, scaler, X_test_features, y_test, class_names
    )
    
    print(f"SVM Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    # Save traditional model
    save_traditional_model(svm_classifier, scaler)
    
    return {
        'classifier': svm_classifier,
        'scaler': scaler,
        'test_accuracy': accuracy,
        'y_pred': y_pred,
        'y_true': y_test,
        'report': report
    }, class_names

def plot_comparison_results(dl_results, trad_results, class_names):
    """Plot comparison between deep learning and traditional ML results"""
    # Create reports directory if it doesn't exist
    os.makedirs('reports', exist_ok=True)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy comparison
    models = list(dl_results.keys()) + ['SVM (HOG)']
    accuracies = [dl_results[model]['test_accuracy'] for model in dl_results] + [trad_results['test_accuracy']]
    
    bars = ax1.bar(models, accuracies, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax1.set_title('Model Comparison: Test Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # Confusion matrix for best DL model
    best_dl_model = max(dl_results.keys(), key=lambda x: dl_results[x]['test_accuracy'])
    y_pred_dl = dl_results[best_dl_model]['y_pred']
    y_true_dl = dl_results[best_dl_model]['y_true']
    
    cm_dl = confusion_matrix(y_true_dl, y_pred_dl)
    sns.heatmap(cm_dl, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, 
                yticklabels=class_names, ax=ax2)
    ax2.set_title(f'Confusion Matrix: {best_dl_model.upper()}')
    ax2.tick_params(axis='x', rotation=45)
    ax2.tick_params(axis='y', rotation=0)
    
    # Confusion matrix for traditional ML
    cm_trad = confusion_matrix(trad_results['y_true'], trad_results['y_pred'])
    sns.heatmap(cm_trad, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, 
                yticklabels=class_names, ax=ax3)
    ax3.set_title('Confusion Matrix: SVM (HOG)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.tick_params(axis='y', rotation=0)
    
    # Training history for best DL model
    history = dl_results[best_dl_model]['history']
    ax4.plot(history.history['accuracy'], label='Train Accuracy')
    ax4.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax4.set_title(f'{best_dl_model.upper()} Training History')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('reports/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run all experiments"""
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    print("PLANTDOC DISEASE CLASSIFICATION EXPERIMENT")
    print("This will run both Deep Learning and Traditional ML experiments")
    print("It may take a while depending on your hardware...")
    
    try:
        # Run deep learning experiments
        dl_results, dl_class_names = run_deep_learning_experiment()
        
        # Run traditional ML experiment
        trad_results, trad_class_names = run_traditional_ml_experiment()
        
        # Plot comparison results
        plot_comparison_results(dl_results, trad_results, dl_class_names)
        
        # Print final comparison
        print("\n" + "=" * 60)
        print("FINAL RESULTS COMPARISON")
        print("=" * 60)
        for model_name, result in dl_results.items():
            print(f"{model_name.upper()}: {result['test_accuracy']:.4f}")
        print(f"SVM (HOG): {trad_results['test_accuracy']:.4f}")
        
        # Save final results summary
        with open('reports/final_results_summary.txt', 'w') as f:
            f.write("PLANTDOC DISEASE CLASSIFICATION - FINAL RESULTS\n")
            f.write("=" * 50 + "\n\n")
            for model_name, result in dl_results.items():
                f.write(f"{model_name.upper()}: {result['test_accuracy']:.4f}\n")
            f.write(f"SVM (HOG): {trad_results['test_accuracy']:.4f}\n")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()