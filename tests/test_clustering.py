#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试聚类分析模块
"""

import unittest
import numpy as np
import os
import sys

# 添加项目路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.clustering import AnomalyClusterer
from utils.data_utils import DataUtils

class TestAnomalyClusterer(unittest.TestCase):
    """测试异常聚类器"""
    
    def setUp(self):
        """测试准备"""
        # 创建模拟配置文件
        self.config = {
            "clustering_config": {
                "n_clusters": 3,
                "algorithm": "kmeans",
                "random_state": 42
            }
        }
        
        # 创建模拟特征数据
        self.features = np.random.rand(100, 60)
        # 添加三个明显的聚类
        self.features[:30, :] += 3
        self.features[30:60, :] -= 3
        
        self.clusterer = AnomalyClusterer(self.config)
    
    def test_kmeans_clustering(self):
        """测试K-means聚类"""
        labels = self.clusterer.fit(self.features)
        
        # 验证标签数量
        self.assertEqual(len(labels), 100)
        # 验证聚类数量
        self.assertEqual(len(np.unique(labels)), 3)
    
    def test_dbscan_clustering(self):
        """测试DBSCAN聚类"""
        self.config['clustering_config']['algorithm'] = 'dbscan'
        clusterer = AnomalyClusterer(self.config)
        labels = clusterer.fit(self.features)
        
        # 验证标签数量
        self.assertEqual(len(labels), 100)
    
    def test_get_cluster_info(self):
        """测试获取聚类信息"""
        self.clusterer.fit(self.features)
        info = self.clusterer.get_cluster_info()
        
        # 验证聚类数量
        self.assertEqual(info['n_clusters'], 3)
        # 验证指标
        self.assertIn('silhouette_score', info['metrics'])
        self.assertGreater(info['metrics']['silhouette_score'], 0.5)

if __name__ == '__main__':
    unittest.main() 