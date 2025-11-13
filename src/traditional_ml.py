import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import joblib
import os

def extract_enhanced_features(images, orientations=12, pixels_per_cell=(16, 16), cells_per_block=(2, 2)):
    """
    Extract enhanced features (HOG + Color Histogram + LBP)
    """
    features_list = []
    
    for img in images:
        feature_vector = []
        
        # Convert to grayscale for HOG and LBP
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        # 1. HOG Features (improved parameters)
        hog_features = hog(gray, 
                          orientations=orientations,
                          pixels_per_cell=pixels_per_cell,
                          cells_per_block=cells_per_block,
                          block_norm='L2-Hys',
                          feature_vector=True)
        feature_vector.extend(hog_features)
        
        # 2. Color Histogram Features
        color_features = []
        for channel in range(3):
            hist = cv2.calcHist([img], [channel], None, [16], [0, 256])
            color_features.extend(hist.flatten())
        feature_vector.extend(color_features)
        
        # 3. LBP Texture Features
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
        feature_vector.extend(lbp_hist)
        
        features_list.append(feature_vector)
    
    return np.array(features_list)

def extract_hog_features(images, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    """
    Extract HOG features from a list of images.
    """
    features = []
    for img in images:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img

        hog_features = hog(gray, orientations=orientations,
                          pixels_per_cell=pixels_per_cell,
                          cells_per_block=cells_per_block,
                          block_norm='L2-Hys',
                          feature_vector=True)
        features.append(hog_features)

    return np.array(features)

def train_svm_classifier(X_train, y_train, C=10.0, kernel='rbf', gamma='scale', use_grid_search=False):
    """
    Train SVM classifier with improved parameters.
    """
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    if use_grid_search:
        # Hyperparameter tuning
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'kernel': ['rbf', 'linear']
        }
        svm = SVC(probability=True, random_state=42)
        grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, scaler
    else:
        # Use improved default parameters
        svm = SVC(C=C, kernel=kernel, gamma=gamma, probability=True, random_state=42)
        svm.fit(X_train_scaled, y_train)
        return svm, scaler

def evaluate_model(classifier, scaler, X_test, y_test, class_names):
    """
    Evaluate the trained model on test data.
    """
    X_test_scaled = scaler.transform(X_test)
    y_pred = classifier.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)

    return accuracy, report, y_pred

def save_model(classifier, scaler, model_path='models/traditional_ml'):
    """
    Save the trained model and scaler.
    """
    os.makedirs(model_path, exist_ok=True)

    joblib.dump(classifier, os.path.join(model_path, 'svm_classifier.pkl'))
    joblib.dump(scaler, os.path.join(model_path, 'scaler.pkl'))

    print(f"Model saved to {model_path}")

def load_model(model_path='models/traditional_ml'):
    """
    Load a saved model and scaler.
    """
    classifier = joblib.load(os.path.join(model_path, 'svm_classifier.pkl'))
    scaler = joblib.load(os.path.join(model_path, 'scaler.pkl'))

    return classifier, scaler

# ==============================
# 实验分析功能 (添加到文件末尾)
# ==============================

def analyze_hyperparameter_sensitivity(X_train, y_train, X_test, y_test, param_name='C', param_range=[0.1, 1, 10, 100]):
    """
    超参数敏感性分析
    """
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    import matplotlib.pyplot as plt
    
    train_scores = []
    test_scores = []
    
    for param_value in param_range:
        print(f"Testing {param_name} = {param_value}")
        
        # 训练模型
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if param_name == 'C':
            svm = SVC(C=param_value, kernel='rbf', gamma='scale', random_state=42)
        else:
            svm = SVC(C=1.0, kernel=param_value, gamma='scale', random_state=42)
        
        svm.fit(X_train_scaled, y_train)
        
        # 计算准确率
        train_accuracy = svm.score(X_train_scaled, y_train)
        test_accuracy = svm.score(X_test_scaled, y_test)
        
        train_scores.append(train_accuracy)
        test_scores.append(test_accuracy)
    
    # 绘制图表
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, train_scores, 'o-', label='Training Accuracy', linewidth=2)
    plt.plot(param_range, test_scores, 's-', label='Test Accuracy', linewidth=2)
    plt.xlabel(param_name)
    plt.ylabel('Accuracy')
    plt.title(f'Hyperparameter Sensitivity: {param_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图表
    import os
    os.makedirs('reports', exist_ok=True)
    plt.savefig(f'reports/hyperparameter_{param_name}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 返回最佳参数
    best_idx = np.argmax(test_scores)
    best_param = param_range[best_idx]
    best_score = test_scores[best_idx]
    
    print(f"🎯 Best {param_name}: {best_param}, Accuracy: {best_score:.4f}")
    return best_param, best_score, train_scores, test_scores

def feature_ablation_analysis(X_train, y_train, X_test, y_test, class_names):
    """
    特征消融实验分析
    """
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    import matplotlib.pyplot as plt
    
    # 不同的特征组合
    feature_sets = {
        'HOG Only': (X_train[:, :8100], X_test[:, :8100]),  # 假设HOG是前8100个特征
        'Color Only': (X_train[:, 8100:8192], X_test[:, 8100:8192]),  # 假设颜色特征是接下来的
        'All Features': (X_train, X_test)
    }
    
    accuracies = []
    names = []
    
    for name, (X_train_sub, X_test_sub) in feature_sets.items():
        print(f"Testing {name}...")
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sub)
        X_test_scaled = scaler.transform(X_test_sub)
        
        svm = SVC(C=10.0, kernel='rbf', random_state=42)
        svm.fit(X_train_scaled, y_train)
        
        y_pred = svm.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        accuracies.append(accuracy)
        names.append(name)
        print(f"  {name}: {accuracy:.4f}")
    
    # 绘制柱状图
    plt.figure(figsize=(10, 6))
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    bars = plt.bar(names, accuracies, color=colors, alpha=0.8)
    
    plt.title('Feature Ablation Study', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    # 添加数值标签
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('reports/feature_ablation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return dict(zip(names, accuracies))

def detailed_failure_analysis(classifier, scaler, X_test, y_test, class_names):
    """
    详细的失败案例分析
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    X_test_scaled = scaler.transform(X_test)
    y_pred = classifier.predict(X_test_scaled)
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names)
    plt.title('Detailed Confusion Matrix - Failure Analysis', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('reports/detailed_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 找出最常混淆的类别对
    confusion_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append((i, j, cm[i, j]))
    
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)
    
    print("\n🔍 Top Confusing Class Pairs:")
    for i, j, count in confusion_pairs[:5]:
        print(f"  {class_names[i]} → {class_names[j]}: {count} cases")
    
    return confusion_pairs

# 保持原有的 if __name__ == "__main__" 部分不变