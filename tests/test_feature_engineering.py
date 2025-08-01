#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试特征工程模块
"""

import unittest
import numpy as np
import os
import sys

# 添加项目路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.feature_engineering import FeatureEngineer
from utils.data_utils import DataUtils

class TestFeatureEngineer(unittest.TestCase):
    """测试特征工程师"""
    
    def setUp(self):
        """测试准备"""
        # 创建模拟配置文件
        self.config = {
            "data_config": {},
            "feature_config": {
                "use_statistical_features": True,
                "use_prediction_features": True,
                "use_pattern_features": True,
                "feature_dimension": 60 # 20+15+25
            },
            "clustering_config": {},
            "model_config": {},
            "training_config": {},
            "evaluation_config": {}
        }
        
        # 创建模拟序列数据
        self.sequences = np.array([
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0, 0, 0, 0],
            [2, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 0],
            [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 10, 9, 8, 7, 6]
        ])
        
        # 创建模拟预测误差数据
        self.prediction_errors = np.random.random(len(self.sequences) * 15).reshape(len(self.sequences), 15)
        
        self.feature_engineer = FeatureEngineer(self.config)
    
    def test_extract_statistical_features(self):
        """测试提取统计特征"""
        features = self.feature_engineer.extract_statistical_features(self.sequences)
        
        # 验证特征形状
        self.assertEqual(features.shape, (3, 20))
        # 验证第一个序列的长度特征
        self.assertAlmostEqual(features[0][0], 10)
    
    def test_extract_prediction_features(self):
        """测试提取预测特征"""
        features = self.feature_engineer.extract_prediction_features(self.sequences, self.prediction_errors)
        
        # 验证特征形状
        self.assertEqual(features.shape, (3, 15))
    
    def test_extract_pattern_features(self):
        """测试提取模式特征"""
        features = self.feature_engineer.extract_pattern_features(self.sequences)
        
        # 验证特征形状
        self.assertEqual(features.shape, (3, 25))
        # 验证第二个序列的重复事件数
        self.assertEqual(features[1][0], 4) # 2,3,4,5
    
    def test_combine_features(self):
        """测试组合特征"""
        self.feature_engineer.extract_statistical_features(self.sequences)
        self.feature_engineer.extract_prediction_features(self.sequences, self.prediction_errors)
        self.feature_engineer.extract_pattern_features(self.sequences)
        
        combined_features = self.feature_engineer.combine_features()
        
        # 验证组合后的形状
        self.assertEqual(combined_features.shape, (3, 60))
    
    def test_extract_all_features_pipeline(self):
        """测试完整的特征提取流程"""
        features = self.feature_engineer.extract_all_features(self.sequences, self.prediction_errors)
        
        # 验证输出形状
        self.assertEqual(features.shape, (3, 60))

if __name__ == '__main__':
    unittest.main() 