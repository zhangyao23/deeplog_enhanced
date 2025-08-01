#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试改进的LSTM模型
"""

import unittest
import torch
import os
import sys

# 添加项目路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.enhanced_model import EnhancedDeepLogModel

class TestEnhancedDeepLogModel(unittest.TestCase):
    """测试改进的DeepLog模型"""
    
    def setUp(self):
        """测试准备"""
        self.config = {
            "model_config": {
                "lstm_hidden_size": 128,
                "lstm_num_layers": 2,
                "lstm_dropout": 0.2,
                "attention_heads": 8,
                "attention_dropout": 0.1,
                "sequence_head_dropout": 0.3,
                "anomaly_head_dropout": 0.4,
                "use_attention": True
            }
        }
        self.num_event_types = 30
        self.num_anomaly_types = 3
        
        self.model = EnhancedDeepLogModel(
            self.config, self.num_event_types, self.num_anomaly_types
        )
        
        # 模拟输入数据
        self.batch_size = 64
        self.seq_len = 20
        self.input_tensor = torch.randn(self.batch_size, self.seq_len, 1)
    
    def test_forward_pass(self):
        """测试前向传播"""
        seq_pred, anomaly_pred = self.model(self.input_tensor)
        
        # 验证输出形状
        self.assertEqual(seq_pred.shape, (self.batch_size, self.num_event_types))
        self.assertEqual(anomaly_pred.shape, (self.batch_size, self.num_anomaly_types))
    
    def test_without_attention(self):
        """测试不使用注意力机制"""
        self.config['model_config']['use_attention'] = False
        model_no_attention = EnhancedDeepLogModel(
            self.config, self.num_event_types, self.num_anomaly_types
        )
        
        seq_pred, anomaly_pred = model_no_attention(self.input_tensor)
        
        # 验证输出形状
        self.assertEqual(seq_pred.shape, (self.batch_size, self.num_event_types))
        self.assertEqual(anomaly_pred.shape, (self.batch_size, self.num_anomaly_types))
    
    def test_get_model_info(self):
        """测试获取模型信息"""
        info = self.model.get_model_info()
        
        # 验证信息完整性
        self.assertIn('total_parameters', info)
        self.assertIn('trainable_parameters', info)
        self.assertEqual(info['num_event_types'], self.num_event_types)
        self.assertEqual(info['num_anomaly_types'], self.num_anomaly_types)

if __name__ == '__main__':
    unittest.main() 