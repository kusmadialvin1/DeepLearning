import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import learning_curve, validation_curve
import cv2
import os
from pathlib import Path

class TraditionalMLAnalyzer:
    """
    传统机器学习实验分析类
    """
    def __init__(self, X_train, y_train, X_test, y_test, class_names):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.class_names = class_names
        self.results = {}
        
    def hyperparameter_sensitivity_analysis(self, classifier_func, param_name, param_range):
        """
        超参数敏感性分析
        """
        print(f"🔧 Analyzing {param_name} sensitivity...")
        
        train_scores = []
        test_scores = []
        
        for param_value in param_range:
            print(f"  Testing {param_name} = {param_value}")
            
            # 训练模型
            classifier, scaler = classifier_func(param_value)
            X_test_scaled = scaler.transform(self.X_test)
            
            # 计算训练和测试准确率
            train_accuracy = classifier.score(scaler.transform(self.X_train), self.y_train)
            test_accuracy = classifier.score(X_test_scaled, self.y_test)
            
            train_scores.append(train_accuracy)
            test_scores.append(test_accuracy)
        
        # 绘制敏感性曲线
        plt.figure(figsize=(10, 6))
        plt.plot(param_range, train_scores, 'o-', label='Training Accuracy', linewidth=2)
        plt.plot(param_range, test_scores, 's-', label='Test Accuracy', linewidth=2)
        plt.xlabel(param_name)
        plt.ylabel('Accuracy')
        plt.title(f'Hyperparameter Sensitivity: {param_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'reports/hyperparameter_sensitivity_{param_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 找到最佳参数
        best_idx = np.argmax(test_scores)
        best_param = param_range[best_idx]
        best_score = test_scores[best_idx]
        
        print(f"🎯 Best {param_name}: {best_param}, Test Accuracy: {best_score:.4f}")
        
        self.results[f'hyperparameter_{param_name}'] = {
            'param_range': param_range,
            'train_scores': train_scores,
            'test_scores': test_scores,
            'best_param': best_param,
            'best_score': best_score
        }
        
        return best_param, best_score
    
    def feature_ablation_study(self, feature_extraction_funcs):
        """
        特征消融实验 - 分析不同特征组合的影响
        """
        print("🔍 Performing feature ablation study...")
        
        feature_combinations = []
        accuracies = []
        
        # 测试不同的特征组合
        for i, (combo_name, extract_func) in enumerate(feature_extraction_funcs.items()):
            print(f"  Testing feature combination: {combo_name}")
            
            # 提取特征
            X_train_features = extract_func(self.X_train)
            X_test_features = extract_func(self.X_test)
            
            # 训练SVM分类器
            from sklearn.svm import SVC
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import accuracy_score
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_features)
            X_test_scaled = scaler.transform(X_test_features)
            
            svm = SVC(C=10.0, kernel='rbf', random_state=42)
            svm.fit(X_train_scaled, self.y_train)
            
            y_pred = svm.predict(X_test_scaled)
            accuracy = accuracy_score(self.y_test, y_pred)
            
            feature_combinations.append(combo_name)
            accuracies.append(accuracy)
            
            print(f"    Accuracy: {accuracy:.4f}")
        
        # 绘制消融实验结果
        plt.figure(figsize=(12, 6))
        colors = plt.cm.Set3(np.linspace(0, 1, len(feature_combinations)))
        bars = plt.bar(feature_combinations, accuracies, color=colors, alpha=0.8)
        
        plt.title('Feature Ablation Study', fontsize=14, fontweight='bold')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # 添加数值标签
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('reports/feature_ablation_study.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        self.results['feature_ablation'] = {
            'combinations': feature_combinations,
            'accuracies': accuracies
        }
        
        return feature_combinations, accuracies
    
    def data_augmentation_impact(self, augmentation_func):
        """
        数据增强影响分析
        """
        print("🔄 Analyzing data augmentation impact...")
        
        # 原始数据（无增强）
        from traditional_ml import extract_all_features, train_classifiers
        
        X_train_original = self.X_train.copy()
        X_test_original = self.X_test.copy()
        
        # 提取特征
        X_train_features_original = extract_all_features(X_train_original)
        X_test_features_original = extract_all_features(X_test_original)
        
        # 训练原始模型
        classifier_original, scaler_original = train_classifiers(X_train_features_original, self.y_train, 'svm')
        accuracy_original, _, _ = self.evaluate_classifier(classifier_original, scaler_original, 
                                                         X_test_features_original, self.y_test)
        
        # 增强数据
        X_train_augmented, y_train_augmented = augmentation_func(self.X_train, self.y_train)
        
        # 提取增强数据的特征
        X_train_features_augmented = extract_all_features(X_train_augmented)
        
        # 训练增强模型
        classifier_augmented, scaler_augmented = train_classifiers(X_train_features_augmented, y_train_augmented, 'svm')
        accuracy_augmented, _, _ = self.evaluate_classifier(classifier_augmented, scaler_augmented,
                                                          X_test_features_original, self.y_test)
        
        # 绘制对比结果
        methods = ['Without Augmentation', 'With Augmentation']
        accuracies = [accuracy_original, accuracy_augmented]
        improvements = [0, accuracy_augmented - accuracy_original]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 准确率对比
        bars1 = ax1.bar(methods, accuracies, color=['lightcoral', 'lightgreen'], alpha=0.8)
        ax1.set_title('Data Augmentation Impact on Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 改进程度
        bars2 = ax2.bar(['Improvement'], improvements[1:], color='skyblue', alpha=0.8)
        ax2.set_title('Accuracy Improvement')
        ax2.set_ylabel('Accuracy Gain')
        
        for bar, imp in zip(bars2, improvements[1:]):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'+{imp:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('reports/data_augmentation_impact.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Results:")
        print(f"  Without augmentation: {accuracy_original:.4f}")
        print(f"  With augmentation: {accuracy_augmented:.4f}")
        print(f"  Improvement: +{improvements[1]:.4f}")
        
        self.results['data_augmentation'] = {
            'original_accuracy': accuracy_original,
            'augmented_accuracy': accuracy_augmented,
            'improvement': improvements[1]
        }
        
        return accuracy_original, accuracy_augmented
    
    def failure_case_analysis(self, classifier, scaler, X_test_features):
        """
        失败案例分析
        """
        print("🔎 Analyzing failure cases...")
        
        X_test_scaled = scaler.transform(X_test_features)
        y_pred = classifier.predict(X_test_scaled)
        y_proba = classifier.predict_proba(X_test_scaled)
        
        # 计算混淆矩阵
        cm = confusion_matrix(self.y_test, y_pred)
        
        # 找出最容易被混淆的类别对
        confusion_pairs = []
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                if i != j and cm[i, j] > 0:
                    confusion_pairs.append((i, j, cm[i, j]))
        
        # 按混淆次数排序
        confusion_pairs.sort(key=lambda x: x[2], reverse=True)
        
        print("\n🔍 Top Confusion Pairs:")
        for i, j, count in confusion_pairs[:5]:
            print(f"  {self.class_names[i]} → {self.class_names[j]}: {count} cases")
        
        # 绘制详细的混淆矩阵
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix - Failure Analysis', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('reports/failure_analysis_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 分析置信度分布
        max_probs = np.max(y_proba, axis=1)
        correct_mask = (y_pred == self.y_test)
        
        plt.figure(figsize=(10, 6))
        plt.hist([max_probs[correct_mask], max_probs[~correct_mask]],
                bins=20, alpha=0.7, label=['Correct', 'Incorrect'], color=['green', 'red'])
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Frequency')
        plt.title('Prediction Confidence Distribution')
        plt.legend()
        plt.savefig('reports/confidence_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 保存失败案例
        failure_cases = []
        for idx in range(len(self.y_test)):
            if y_pred[idx] != self.y_test[idx]:
                true_class = self.class_names[self.y_test[idx]]
                pred_class = self.class_names[y_pred[idx]]
                confidence = max_probs[idx]
                
                failure_cases.append({
                    'index': idx,
                    'true_class': true_class,
                    'predicted_class': pred_class,
                    'confidence': confidence,
                    'probabilities': y_proba[idx]
                })
        
        # 输出最不确定的预测
        failure_cases.sort(key=lambda x: x['confidence'])
        print(f"\n❌ Most Uncertain Predictions (Low Confidence):")
        for case in failure_cases[:5]:
            print(f"  True: {case['true_class']}, Predicted: {case['predicted_class']}, "
                  f"Confidence: {case['confidence']:.4f}")
        
        self.results['failure_analysis'] = {
            'confusion_pairs': confusion_pairs[:10],
            'failure_cases': failure_cases[:20],
            'avg_confidence_correct': np.mean(max_probs[correct_mask]),
            'avg_confidence_incorrect': np.mean(max_probs[~correct_mask])
        }
        
        return confusion_pairs, failure_cases
    
    def feature_visualization(self, images, feature_extraction_func, n_samples=5):
        """
        特征可视化 - 展示不同特征提取方法的结果
        """
        print("👁️ Visualizing feature extraction...")
        
        # 选择一些样本图像
        sample_indices = np.random.choice(len(images), n_samples, replace=False)
        sample_images = images[sample_indices]
        
        # 创建可视化
        fig, axes = plt.subplots(n_samples, 4, figsize=(15, 3*n_samples))
        
        for i, idx in enumerate(sample_indices):
            img = sample_images[i]
            
            # 原始图像
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f'Sample {i+1}\nOriginal')
            axes[i, 0].axis('off')
            
            # 灰度图像 (用于HOG等)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            axes[i, 1].imshow(gray, cmap='gray')
            axes[i, 1].set_title('Grayscale')
            axes[i, 1].axis('off')
            
            # HOG特征可视化
            from skimage import exposure
            from skimage.feature import hog
            
            fd, hog_image = hog(gray, orientations=8, pixels_per_cell=(16, 16),
                               cells_per_block=(1, 1), visualize=True)
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
            axes[i, 2].imshow(hog_image_rescaled, cmap='gray')
            axes[i, 2].set_title('HOG Features')
            axes[i, 2].axis('off')
            
            # LBP特征可视化
            from skimage.feature import local_binary_pattern
            lbp = local_binary_pattern(gray, 8, 1, method='uniform')
            axes[i, 3].imshow(lbp, cmap='gray')
            axes[i, 3].set_title('LBP Features')
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig('reports/feature_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Feature visualization completed!")
    
    def generate_comprehensive_report(self):
        """
        生成综合实验报告
        """
        print("📄 Generating comprehensive experiment report...")
        
        report_content = []
        report_content.append("COMPREHENSIVE TRADITIONAL ML EXPERIMENT REPORT")
        report_content.append("=" * 70)
        report_content.append("")
        
        # 超参数分析结果
        if 'hyperparameter_C' in self.results:
            hp_result = self.results['hyperparameter_C']
            report_content.append("HYPERPARAMETER SENSITIVITY ANALYSIS")
            report_content.append("-" * 40)
            report_content.append(f"Best C parameter: {hp_result['best_param']}")
            report_content.append(f"Best test accuracy: {hp_result['best_score']:.4f}")
            report_content.append("")
        
        # 特征消融结果
        if 'feature_ablation' in self.results:
            ablation_result = self.results['feature_ablation']
            report_content.append("FEATURE ABLATION STUDY")
            report_content.append("-" * 25)
            for combo, acc in zip(ablation_result['combinations'], ablation_result['accuracies']):
                report_content.append(f"{combo:<30}: {acc:.4f}")
            report_content.append("")
        
        # 数据增强结果
        if 'data_augmentation' in self.results:
            aug_result = self.results['data_augmentation']
            report_content.append("DATA AUGMENTATION IMPACT")
            report_content.append("-" * 28)
            report_content.append(f"Without augmentation: {aug_result['original_accuracy']:.4f}")
            report_content.append(f"With augmentation: {aug_result['augmented_accuracy']:.4f}")
            report_content.append(f"Improvement: +{aug_result['improvement']:.4f}")
            report_content.append("")
        
        # 失败分析结果
        if 'failure_analysis' in self.results:
            fail_result = self.results['failure_analysis']
            report_content.append("FAILURE CASE ANALYSIS")
            report_content.append("-" * 25)
            report_content.append("Top Confusion Pairs:")
            for i, j, count in fail_result['confusion_pairs'][:5]:
                report_content.append(f"  {self.class_names[i]} → {self.class_names[j]}: {count} cases")
            report_content.append("")
            report_content.append(f"Average confidence (correct): {fail_result['avg_confidence_correct']:.4f}")
            report_content.append(f"Average confidence (incorrect): {fail_result['avg_confidence_incorrect']:.4f}")
        
        # 保存报告
        with open('reports/comprehensive_experiment_report.txt', 'w') as f:
            f.write('\n'.join(report_content))
        
        print("✅ Comprehensive report generated!")
    
    def evaluate_classifier(self, classifier, scaler, X_test_features, y_test):
        """
        评估分类器性能
        """
        from sklearn.metrics import accuracy_score, classification_report
        
        X_test_scaled = scaler.transform(X_test_features)
        y_pred = classifier.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=self.class_names, zero_division=0)
        
        return accuracy, report, y_pred

# 数据增强函数
def simple_augmentation(images, labels):
    """
    简单的数据增强实现
    """
    augmented_images = []
    augmented_labels = []
    
    for img, label in zip(images, labels):
        # 原始图像
        augmented_images.append(img)
        augmented_labels.append(label)
        
        # 水平翻转
        flipped = np.fliplr(img)
        augmented_images.append(flipped)
        augmented_labels.append(label)
        
        # 轻微旋转 (±10度)
        rows, cols = img.shape[:2]
        angle = np.random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        rotated = cv2.warpAffine(img, M, (cols, rows))
        augmented_images.append(rotated)
        augmented_labels.append(label)
    
    return np.array(augmented_images), np.array(augmented_labels)