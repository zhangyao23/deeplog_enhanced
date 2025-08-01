#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试数据预处理模块
"""

import unittest
import numpy as np
import os
import json
import sys

# 添加项目路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.data_preprocessing import DataPreprocessor
from utils.data_utils import DataUtils

class TestDataPreprocessor(unittest.TestCase):
    """测试数据预处理器"""
    
    def setUp(self):
        """测试准备"""
        # 创建模拟配置文件
        self.config = {
            "data_config": {
                "max_sequence_length": 20,
                "min_sequence_length": 5,
                "num_event_types": 30,
                "padding_value": 0,
                "window_size": 10
            },
            "feature_config": {},
            "clustering_config": {},
            "model_config": {},
            "training_config": {},
            "evaluation_config": {}
        }
        
        # 创建模拟数据文件
        self.test_data_path = 'test_data.txt'
        with open(self.test_data_path, 'w') as f:
            f.write("1 2 3 4 5 6 7 8 9 10 11 12\n")
            f.write("2 3 4 5 6\n")
            f.write("3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23\n") # 超长
            f.write("4 5 6\n") # 过短
        
        self.preprocessor = DataPreprocessor(self.config)
    
    def tearDown(self):
        """测试清理"""
        if os.path.exists(self.test_data_path):
            os.remove(self.test_data_path)
    
    def test_load_data(self):
        """测试加载数据"""
        sequences = self.preprocessor.load_data(self.test_data_path)
        
        # 验证加载的序列数量（过滤掉过短的序列）
        self.assertEqual(len(sequences), 3)
        # 验证第一个序列的内容
        self.assertEqual(sequences[0], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        # 验证第二个序列的内容
        self.assertEqual(sequences[1], [1, 2, 3, 4, 5])
    
    def test_pad_sequences(self):
        """测试序列补齐"""
        sequences = self.preprocessor.load_data(self.test_data_path)
        padded_sequences = self.preprocessor.pad_sequences(sequences)
        
        # 验证形状
        self.assertEqual(padded_sequences.shape, (3, 20))
        # 验证补齐值
        self.assertEqual(padded_sequences[1][-1], 0)
        # 验证截断
        self.assertEqual(len(padded_sequences[2]), 20)
        self.assertEqual(padded_sequences[2][-1], 21) # 截断后的最后一个元素 (22-1)
    
    def test_create_event_mapping(self):
        """测试事件映射"""
        sequences = self.preprocessor.load_data(self.test_data_path)
        event_mapping = self.preprocessor.create_event_mapping(sequences)
        
        # 验证映射数量
        self.assertEqual(len(event_mapping), 23)
        # 验证映射的连续性
        self.assertEqual(min(event_mapping.values()), 0)
        self.assertEqual(max(event_mapping.values()), 22)
    
    def test_split_sequences(self):
        """测试序列分割"""
        sequences = self.preprocessor.load_data(self.test_data_path)
        window_size = self.config['data_config']['window_size']
        
        # 使用第一个序列进行测试
        input_seqs, target_seqs = self.preprocessor.split_sequences([sequences[0]], window_size)
        
        # 验证样本数量
        self.assertEqual(len(input_seqs), 2)
        # 验证第一个样本
        self.assertEqual(input_seqs[0], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(target_seqs[0], 10)
    
    def test_preprocess_pipeline(self):
        """测试完整的预处理流程"""
        inputs, targets, stats = self.preprocessor.preprocess(self.test_data_path)
        
        # 验证输出形状
        self.assertIsNotNone(inputs)
        self.assertIsNotNone(targets)
        # 验证统计信息
        self.assertEqual(stats['num_sequences'], 3)
        self.assertEqual(stats['max_length'], 21)

if __name__ == '__main__':
    unittest.main() 