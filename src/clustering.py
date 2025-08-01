#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
聚类分析模块
负责自动发现和区分异常模式
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import logging

class AnomalyClusterer:
    """
    异常聚类器，用于将异常特征分组。
    """
    def __init__(self, config: Dict[str, Any]):
        """
        初始化聚类器。
        
        Args:
            config (Dict[str, Any]): 聚类相关的配置，
                                     例如 `{'algorithm': 'kmeans', 'n_clusters': 3}`。
        """
        self.clustering_config = config
        self.algorithm = self.clustering_config.get('algorithm', 'kmeans')
        self.n_clusters = self.clustering_config.get('n_clusters', 3)
        self.random_state = self.clustering_config.get('random_state', 42)
        
        # 根据配置选择模型
        self.model = None
        if self.algorithm == 'kmeans':
            self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        elif self.algorithm == 'dbscan':
            # DBSCAN的参数可以从config中获取，这里使用默认值
            eps = self.clustering_config.get('eps', 0.5)
            min_samples = self.clustering_config.get('min_samples', 5)
            self.model = DBSCAN(eps=eps, min_samples=min_samples)
        else:
            raise ValueError(f"不支持的聚类算法: {self.algorithm}")

        # 聚类结果
        self.cluster_labels = None
        self.cluster_centers = None
        self.cluster_metrics = {}
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def fit(self, features: np.ndarray) -> np.ndarray:
        """
        执行聚类分析
        
        Args:
            features: 特征数组
            
        Returns:
            聚类标签
        """
        self.logger.info(f"执行聚类分析，算法: {self.algorithm}, 聚类数: {self.n_clusters}")
        
        if self.algorithm == 'kmeans':
            self.cluster_labels, self.cluster_centers = self._kmeans_clustering(features)
        elif self.algorithm == 'dbscan':
            self.cluster_labels = self._dbscan_clustering(features)
        else:
            raise ValueError(f"不支持的聚类算法: {self.algorithm}")
        
        # 计算聚类质量指标
        self._calculate_cluster_metrics(features)
        
        return self.cluster_labels
    
    def _kmeans_clustering(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """K-means聚类"""
        if not isinstance(self.model, KMeans):
             raise TypeError("模型不是KMeans实例。")
        labels = self.model.fit_predict(features)
        centers = self.model.cluster_centers_
        return labels, centers
    
    def _dbscan_clustering(self, features: np.ndarray) -> np.ndarray:
        """DBSCAN聚类"""
        if not isinstance(self.model, DBSCAN):
            raise TypeError("模型不是DBSCAN实例。")
        labels = self.model.fit_predict(features)
        return labels
    
    def _calculate_cluster_metrics(self, features: np.ndarray):
        """计算聚类质量指标"""
        if len(np.unique(self.cluster_labels)) > 1:
            self.cluster_metrics['silhouette_score'] = silhouette_score(features, self.cluster_labels)
            self.cluster_metrics['calinski_harabasz_score'] = calinski_harabasz_score(features, self.cluster_labels)
        else:
            self.cluster_metrics['silhouette_score'] = 0
            self.cluster_metrics['calinski_harabasz_score'] = 0
        
        self.logger.info(f"聚类指标: {self.cluster_metrics}")
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """获取聚类信息"""
        unique_labels, counts = np.unique(self.cluster_labels, return_counts=True)
        
        cluster_info = {
            'n_clusters': len(unique_labels),
            'cluster_sizes': dict(zip(unique_labels, counts)),
            'metrics': self.cluster_metrics,
            'algorithm': self.algorithm
        }
        
        return cluster_info 

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        使用训练好的模型预测簇标签。
        
        Args:
            features (np.ndarray): 用于预测的特征。
            
        Returns:
            np.ndarray: 预测的簇标签。
        """
        if self.model is None:
            raise RuntimeError("模型尚未训练，请先调用 fit()。")
        return self.model.predict(features)

    def fit_predict(self, features: np.ndarray) -> np.ndarray:
        """
        训练模型并返回簇标签。
        
        Args:
            features (np.ndarray): 用于训练和预测的特征。
            
        Returns:
            np.ndarray: 预测的簇标签。
        """
        self.fit(features)
        return self.predict(features) 