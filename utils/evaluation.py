#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型评估工具
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    silhouette_score, calinski_harabasz_score,
    adjusted_rand_score, homogeneity_score
)
from typing import Dict, Any, List, Tuple
import json
import logging
from sklearn.manifold import TSNE

class ModelEvaluator:
    """
    模型评估器，负责评估分类和聚类模型的性能。
    """
    def __init__(self, config: Dict[str, Any]):
        """
        初始化评估器。
        
        Args:
            config (Dict[str, Any]): 包含评估相关参数的配置字典。
        """
        self.config = config
        self.evaluation_config = self.config.get('evaluation_config', {})
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, Any]:
        """
        评估分类性能。
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_proba: 预测概率（可选）
            
        Returns:
            评估指标字典
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted')
        }
        
        # 如果提供了概率，计算AUC
        if y_prob is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
            except:
                metrics['roc_auc'] = 0.0
        
        return metrics
    
    def evaluate_clustering(self, features: np.ndarray, true_labels: np.ndarray, cluster_labels: np.ndarray) -> Dict[str, float]:
        """
        评估聚类性能。
        
        Args:
            features (np.ndarray): 用于聚类的特征。
            true_labels (np.ndarray): 真实的标签（用于计算对齐指标）。
            cluster_labels (np.ndarray): 聚类算法生成的标签。
            
        Returns:
            Dict[str, float]: 包含各种聚类评估指标的字典。
        """
        self.logger.info("评估聚类性能...")
        metrics = {}
        
        # 过滤掉噪声点（标签为-1），DBSCAN可能会产生
        valid_indices = cluster_labels != -1
        if np.sum(valid_indices) == 0:
            self.logger.warning("没有有效的聚类标签可供评估。")
            return {
                'silhouette': -1.0, 
                'calinski_harabasz': -1.0,
                'adjusted_rand_index': -1.0,
                'homogeneity': -1.0
            }
            
        features = features[valid_indices]
        true_labels_filtered = true_labels[valid_indices]
        cluster_labels_filtered = cluster_labels[valid_indices]
        
        # 内部指标
        try:
            if len(np.unique(cluster_labels_filtered)) > 1:
                metrics['silhouette'] = silhouette_score(features, cluster_labels_filtered)
                metrics['calinski_harabasz'] = calinski_harabasz_score(features, cluster_labels_filtered)
            else:
                metrics['silhouette'] = 0.0
                metrics['calinski_harabasz'] = 0.0
        except Exception as e:
            self.logger.error(f"计算内部聚类指标时出错: {e}")
            metrics['silhouette'] = -1.0
            metrics['calinski_harabasz'] = -1.0
        
        # 外部指标 (对齐指标)
        if true_labels is not None:
            metrics['adjusted_rand_index'] = adjusted_rand_score(true_labels_filtered, cluster_labels_filtered)
            metrics['homogeneity'] = homogeneity_score(true_labels_filtered, cluster_labels_filtered)
            
        return metrics

    def plot_clusters(self, features: np.ndarray, labels: np.ndarray, title: str = 'Cluster Visualization', save_path: str = None):
        """
        使用 t-SNE 将高维特征降维并绘制聚类结果。

        Args:
            features (np.ndarray): 高维特征数据。
            labels (np.ndarray): 每个特征点的簇标签。
            title (str, optional): 图表标题。
            save_path (str, optional): 如果提供，则将图表保存到此路径。
        """
        self.logger.info("使用 t-SNE 生成聚类可视化...")
        
        # 使用 t-SNE 进行降维
        # 注意: 'n_iter' 在新版 scikit-learn 中已弃用，使用 'max_iter'
        tsne = TSNE(n_components=2, 
                    random_state=self.config.get('clustering_config', {}).get('random_state', 42), 
                    perplexity=30, 
                    max_iter=1000)
        features_2d = tsne.fit_transform(features)
        
        # 绘图
        plt.figure(figsize=(12, 10))
        palette = sns.color_palette("bright", len(np.unique(labels)))
        sns.scatterplot(x=features_2d[:, 0], y=features_2d[:, 1], hue=labels, palette=palette, legend='full')
        
        plt.title(title, fontsize=16)
        plt.xlabel("T-SNE Component 1")
        plt.ylabel("T-SNE Component 2")
        plt.legend(title='Cluster')
        
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"聚类图已保存到: {save_path}")
        else:
            plt.show()
    
    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     target_names: List[str] = None) -> str:
        """
        生成分类报告
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            target_names: 目标类别名称
            
        Returns:
            分类报告字符串
        """
        report_str = classification_report(y_true, y_pred, target_names=target_names)
        self.logger.info(f"分类报告:\n{report_str}")
        return report_str
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              class_names: List[str] = None, save_path: str = None):
        """
        绘制混淆矩阵
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            target_names: 目标类别名称
            save_path: 保存路径
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_history(self, history: Dict[str, List[float]], save_path: str = None):
        """
        绘制训练历史
        
        Args:
            history: 训练历史字典
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        axes[0, 0].plot(history.get('sequence_loss', []), label='Sequence Loss')
        axes[0, 0].plot(history.get('anomaly_loss', []), label='Anomaly Loss')
        axes[0, 0].plot(history.get('total_loss', []), label='Total Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 准确率曲线
        axes[0, 1].plot(history.get('sequence_accuracy', []), label='Sequence Accuracy')
        axes[0, 1].plot(history.get('anomaly_accuracy', []), label='Anomaly Accuracy')
        axes[0, 1].set_title('Training Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 预测分布
        if 'anomaly_predictions' in history:
            axes[1, 0].hist(history['anomaly_predictions'], bins=20, alpha=0.7)
            axes[1, 0].set_title('Anomaly Prediction Distribution')
            axes[1, 0].set_xlabel('Predicted Class')
            axes[1, 0].set_ylabel('Count')
        
        # 置信度分布
        if 'anomaly_confidences' in history:
            axes[1, 1].hist(history['anomaly_confidences'], bins=20, alpha=0.7)
            axes[1, 1].set_title('Anomaly Confidence Distribution')
            axes[1, 1].set_xlabel('Confidence')
            axes[1, 1].set_ylabel('Count')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, feature_names: List[str], 
                              feature_importance: np.ndarray, 
                              top_k: int = 20, save_path: str = None):
        """
        绘制特征重要性
        
        Args:
            feature_names: 特征名称列表
            feature_importance: 特征重要性数组
            top_k: 显示前k个重要特征
            save_path: 保存路径
        """
        # 获取前k个重要特征
        indices = np.argsort(feature_importance)[::-1][:top_k]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(indices)), feature_importance[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_k} Feature Importance')
        plt.gca().invert_yaxis()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_evaluation_results(self, results: Dict[str, Any], file_path: str):
        """
        保存评估结果
        
        Args:
            results: 评估结果字典
            file_path: 保存路径
        """
        # 转换为可序列化的格式
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, np.integer):
                serializable_results[key] = int(value)
            elif isinstance(value, np.floating):
                serializable_results[key] = float(value)
            else:
                serializable_results[key] = value
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"评估结果已保存到: {file_path}")
    
    def load_evaluation_results(self, file_path: str) -> Dict[str, Any]:
        """
        加载评估结果
        
        Args:
            file_path: 文件路径
            
        Returns:
            评估结果字典
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # 转换回numpy数组
        for key, value in results.items():
            if isinstance(value, list):
                results[key] = np.array(value)
        
        return results
    
    def print_evaluation_summary(self, metrics: Dict[str, float]):
        """
        打印评估摘要
        
        Args:
            metrics: 评估指标字典
        """
        print("\n" + "="*50)
        print("模型评估摘要")
        print("="*50)
        
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                print(f"{metric_name}: {value:.4f}")
            else:
                print(f"{metric_name}: {value}")
        
        print("="*50) 