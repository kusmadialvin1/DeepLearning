import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

sys.path.append('src')
from traditional_ml import extract_all_features, extract_hog_features, extract_color_histogram, train_classifiers
from experiment_analysis import TraditionalMLAnalyzer, simple_augmentation

def load_data():
    """
    加载数据
    """
    print("📁 Loading data...")
    
    def load_images(folder_path, img_size=(128, 128), allowed_classes=None):
        images = []
        labels = []
        class_names = []
        
        folder = Path(folder_path)
        class_dirs = [d for d in folder.iterdir() if d.is_dir()]
        
        for class_dir in sorted(class_dirs):
            class_name = class_dir.name
            
            if allowed_classes and class_name not in allowed_classes:
                continue
                
            if class_name not in class_names:
                class_names.append(class_name)
            
            class_idx = class_names.index(class_name)
            
            for img_path in class_dir.glob('*.[jJ][pP][gG]'):
                img = cv2.imread(str(img_path))
                if img is not None:
                    img = cv2.resize(img, img_size)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    labels.append(class_idx)
        
        return np.array(images), np.array(labels), class_names
    
    # 加载数据
    X_train, y_train, train_classes = load_images('TRAIN')
    X_test, y_test, test_classes = load_images('TEST', allowed_classes=train_classes)
    
    print(f"✅ Training: {len(X_train)} images, {len(train_classes)} classes")
    print(f"✅ Test: {len(X_test)} images")
    
    return X_train, y_train, X_test, y_test, train_classes

def main():
    """
    运行完整的实验分析
    """
    print("🎯 COMPREHENSIVE TRADITIONAL ML EXPERIMENTS")
    print("=" * 60)
    print("Including: Hyperparameter analysis, Feature ablation,")
    print("           Data augmentation, Failure analysis, Visualization")
    print("=" * 60)
    
    try:
        # 1. 加载数据
        X_train, y_train, X_test, y_test, class_names = load_data()
        
        # 2. 提取特征
        print("\n🔍 Extracting features for experiments...")
        X_train_features = extract_all_features(X_train)
        X_test_features = extract_all_features(X_test)
        
        # 3. 初始化分析器
        analyzer = TraditionalMLAnalyzer(X_train_features, y_train, X_test_features, y_test, class_names)
        
        # 4. 超参数敏感性分析
        print("\n" + "="*50)
        print("1. HYPERPARAMETER SENSITIVITY ANALYSIS")
        print("="*50)
        
        def create_svm_classifier(C_value):
            from sklearn.svm import SVC
            from sklearn.preprocessing import StandardScaler
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_features)
            
            svm = SVC(C=C_value, kernel='rbf', gamma='scale', random_state=42)
            svm.fit(X_train_scaled, y_train)
            
            return svm, scaler
        
        C_range = [0.1, 1, 10, 100, 1000]
        best_C, best_score = analyzer.hyperparameter_sensitivity_analysis(
            create_svm_classifier, 'C', C_range
        )
        
        # 5. 特征消融实验
        print("\n" + "="*50)
        print("2. FEATURE ABLATION STUDY")
        print("="*50)
        
        def extract_hog_only(images):
            return extract_hog_features(images)
        
        def extract_color_only(images):
            return extract_color_histogram(images)
        
        def extract_hog_color(images):
            hog_feat = extract_hog_features(images)
            color_feat = extract_color_histogram(images)
            return np.hstack([hog_feat, color_feat])
        
        feature_funcs = {
            'HOG Only': extract_hog_only,
            'Color Only': extract_color_only,
            'HOG + Color': extract_hog_color,
            'All Features': extract_all_features
        }
        
        analyzer.feature_ablation_study(feature_funcs)
        
        # 6. 数据增强影响
        print("\n" + "="*50)
        print("3. DATA AUGMENTATION IMPACT")
        print("="*50)
        
        analyzer.data_augmentation_impact(simple_augmentation)
        
        # 7. 训练最佳模型进行失败分析
        print("\n" + "="*50)
        print("4. FAILURE CASE ANALYSIS")
        print("="*50)
        
        # 使用最佳参数训练模型
        best_classifier, best_scaler = create_svm_classifier(best_C)
        analyzer.failure_case_analysis(best_classifier, best_scaler, X_test_features)
        
        # 8. 特征可视化
        print("\n" + "="*50)
        print("5. FEATURE VISUALIZATION")
        print("="*50)
        
        analyzer.feature_visualization(X_train, extract_all_features, n_samples=3)
        
        # 9. 生成综合报告
        print("\n" + "="*50)
        print("6. GENERATING COMPREHENSIVE REPORT")
        print("="*50)
        
        analyzer.generate_comprehensive_report()
        
        print("\n🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("📊 Check the 'reports/' folder for all analysis results")
        print("📈 Multiple visualization files have been generated")
        print("📄 Comprehensive report: 'reports/comprehensive_experiment_report.txt'")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()