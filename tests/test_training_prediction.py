#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试训练和预测模块
"""

import unittest
import torch
import numpy as np
import os
import sys
import json

# 添加项目路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.enhanced_model import EnhancedDeepLogModel
from src.training import ModelTrainer
from src.prediction import AnomalyPredictor
from utils.data_utils import DataUtils

class TestTrainingPrediction(unittest.TestCase):
    """测试训练和预测流程"""
    
    def setUp(self):
        """测试准备"""
        self.config = {
            "data_config": { "num_event_types": 20 },
            "model_config": {
                "lstm_hidden_size": 32,
                "lstm_num_layers": 1,
                "lstm_dropout": 0,
                "attention_heads": 2,
                "attention_dropout": 0,
                "sequence_head_dropout": 0,
                "anomaly_head_dropout": 0,
                "use_attention": True
            },
            "training_config": {
                "batch_size": 16,
                "learning_rate": 0.01,
                "num_epochs": 2,
                "sequence_loss_weight": 0.5,
                "anomaly_loss_weight": 0.5,
                "weight_decay": 0,
                "gradient_clip": 1.0,
                "early_stopping_patience": 5
            },
            "evaluation_config": {}
        }
        self.num_event_types = self.config['data_config']['num_event_types']
        self.num_anomaly_types = 3
        
        # 创建模型
        self.model = EnhancedDeepLogModel(
            self.config, self.num_event_types, self.num_anomaly_types
        )
        
        # 创建模拟数据
        self.num_samples = 100
        self.seq_len = 15
        self.inputs = np.random.randint(1, self.num_event_types, size=(self.num_samples, self.seq_len))
        self.sequence_targets = np.random.randint(0, self.num_event_types, size=(self.num_samples,))
        self.anomaly_targets = np.random.randint(0, self.num_anomaly_types, size=(self.num_samples,))
        
        # 创建目录
        DataUtils.ensure_directory('models')
        DataUtils.ensure_directory('results')

    def tearDown(self):
        """测试清理"""
        if os.path.exists('models/best_model.pth'):
            os.remove('models/best_model.pth')
        if os.path.exists('results/training_history.json'):
            os.remove('results/training_history.json')
        if os.path.exists('results/test_predictions.json'):
            os.remove('results/test_predictions.json')

    def test_training_pipeline(self):
        """测试训练流程"""
        trainer = ModelTrainer(self.config, self.model)
        
        # 准备数据
        train_loader, val_loader = trainer.prepare_data(
            self.inputs, self.sequence_targets, self.anomaly_targets
        )
        
        # 训练模型
        device = torch.device("cpu")
        training_results = trainer.train(train_loader, val_loader, device)
        
        # 验证结果
        self.assertIn('train_history', training_results)
        self.assertEqual(len(training_results['train_history']['total_loss']), self.config['training_config']['num_epochs'])
        self.assertTrue(os.path.exists('models/best_model.pth'))
        
        # 保存训练历史
        trainer.save_training_history('results/training_history.json')
        self.assertTrue(os.path.exists('results/training_history.json'))

    def test_prediction_pipeline(self):
        """测试预测流程"""
        # 先进行一次简短的训练
        trainer = ModelTrainer(self.config, self.model)
        train_loader, val_loader = trainer.prepare_data(
            self.inputs, self.sequence_targets, self.anomaly_targets
        )
        device = torch.device("cpu")
        trainer.train(train_loader, val_loader, device)
        
        # 加载训练好的模型
        trained_model = EnhancedDeepLogModel(
            self.config, self.num_event_types, self.num_anomaly_types
        )
        trained_model.load_state_dict(torch.load('models/best_model.pth'))
        
        # 创建预测器
        predictor = AnomalyPredictor(trained_model, self.config)
        
        # 进行预测
        test_inputs = np.random.randint(1, self.num_event_types, size=(10, self.seq_len))
        predictions = predictor.predict(test_inputs, device)
        
        # 验证预测结果
        self.assertEqual(len(predictions['sequence_predictions']), 10)
        self.assertEqual(len(predictions['anomaly_predictions']), 10)
        self.assertEqual(predictions['sequence_probs'].shape, (10, self.num_event_types))
        self.assertEqual(predictions['anomaly_probs'].shape, (10, self.num_anomaly_types))
        
        # 保存预测结果
        predictor.save_predictions(predictions, 'results/test_predictions.json')
        self.assertTrue(os.path.exists('results/test_predictions.json'))

if __name__ == '__main__':
    unittest.main() 