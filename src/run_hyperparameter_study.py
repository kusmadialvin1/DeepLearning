import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

sys.path.append('src')
from traditional_ml import extract_all_features
from experiment_analysis import TraditionalMLAnalyzer

def run_hyperparameter_experiment():
    """专门运行超参数实验"""
    print("🎯 HYPERPARAMETER SENSITIVITY EXPERIMENT")
    
    # 加载数据
    def load_data():
        images, labels, class_names = [], [], []
        folder = Path('TRAIN')
        class_dirs = [d for d in folder.iterdir() if d.is_dir()]
        
        for class_idx, class_dir in enumerate(sorted(class_dirs)):
            class_names.append(class_dir.name)
            for img_path in class_dir.glob('*.[jJ][pP][gG]'):
                img = cv2.imread(str(img_path))
                if img is not None:
                    img = cv2.resize(img, (128, 128))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    labels.append(class_idx)
        
        return np.array(images), np.array(labels), class_names
    
    X_train, y_train, class_names = load_data()
    X_test, y_test, _ = load_data()  # 简化版本
    
    # 提取特征
    X_train_features = extract_all_features(X_train)
    X_test_features = extract_all_features(X_test)
    
    # 超参数分析
    analyzer = TraditionalMLAnalyzer(X_train_features, y_train, X_test_features, y_test, class_names)
    
    def create_classifier(C_value):
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_features)
        
        svm = SVC(C=C_value, kernel='rbf', random_state=42)
        svm.fit(X_train_scaled, y_train)
        return svm, scaler
    
    # 测试不同的C值
    C_range = [0.01, 0.1, 1, 10, 100, 1000]
    best_C, best_score = analyzer.hyperparameter_sensitivity_analysis(create_classifier, 'C', C_range)
    
    print(f"\n🏆 Best C parameter: {best_C}")
    print(f"🎯 Best accuracy: {best_score:.4f}")

if __name__ == "__main__":
    run_hyperparameter_experiment()