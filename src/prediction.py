#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预测模块
负责模型预测和结果输出
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple
import json
import logging

class AnomalyPredictor:
    """异常预测器"""
    
    def __init__(self, model: torch.nn.Module, config: Dict[str, Any]):
        """
        初始化预测器
        
        Args:
            model: 训练好的模型
            config: 配置字典
        """
        self.model = model
        self.config = config
        self.evaluation_config = config['evaluation_config']
        
        # 异常阈值
        self.anomaly_threshold = None
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def set_anomaly_threshold(self, threshold: float):
        """
        设置异常阈值
        
        Args:
            threshold: 异常阈值
        """
        self.anomaly_threshold = threshold
        self.logger.info(f"设置异常阈值: {threshold}")
    
    def predict(self, sequences: np.ndarray, device: torch.device) -> Dict[str, Any]:
        """
        进行预测
        
        Args:
            sequences: 输入序列
            device: 设备
            
        Returns:
            预测结果
        """
        self.model.eval()
        self.model = self.model.to(device)
        
        # 转换为张量
        inputs = torch.FloatTensor(sequences).to(device)
        
        with torch.no_grad():
            # 前向传播
            sequence_pred, anomaly_pred = self.model(inputs)
            
            # 获取预测结果
            sequence_probs = torch.softmax(sequence_pred, dim=1)
            anomaly_probs = torch.softmax(anomaly_pred, dim=1)
            
            # 预测类别
            sequence_predictions = torch.argmax(sequence_pred, dim=1).cpu().numpy()
            anomaly_predictions = torch.argmax(anomaly_pred, dim=1).cpu().numpy()
            
            # 预测置信度
            sequence_confidences = torch.max(sequence_probs, dim=1)[0].cpu().numpy()
            anomaly_confidences = torch.max(anomaly_probs, dim=1)[0].cpu().numpy()
        
        return {
            'sequence_predictions': sequence_predictions,
            'anomaly_predictions': anomaly_predictions,
            'sequence_confidences': sequence_confidences,
            'anomaly_confidences': anomaly_confidences,
            'sequence_probs': sequence_probs.cpu().numpy(),
            'anomaly_probs': anomaly_probs.cpu().numpy()
        }
    
    def predict_with_features(self, sequences: np.ndarray, features: np.ndarray,
                            device: torch.device) -> Dict[str, Any]:
        """
        结合特征进行预测
        
        Args:
            sequences: 输入序列
            features: 提取的特征
            device: 设备
            
        Returns:
            预测结果
        """
        # 基础预测
        basic_results = self.predict(sequences, device)
        
        # 结合特征进行后处理
        enhanced_results = self._enhance_predictions_with_features(
            basic_results, features
        )
        
        return enhanced_results
    
    def _enhance_predictions_with_features(self, basic_results: Dict[str, Any],
                                         features: np.ndarray) -> Dict[str, Any]:
        """
        使用特征增强预测结果
        
        Args:
            basic_results: 基础预测结果
            features: 特征数组
            
        Returns:
            增强的预测结果
        """
        # 这里可以实现基于特征的预测增强逻辑
        # 例如：使用特征调整置信度、重新分类等
        
        enhanced_results = basic_results.copy()
        
        # 示例：基于特征调整异常置信度
        if features is not None and len(features) > 0:
            # 使用特征中的异常分数调整置信度
            anomaly_scores = features[:, 0] if features.shape[1] > 0 else np.zeros(len(features))
            enhanced_results['anomaly_confidences'] = np.clip(
                basic_results['anomaly_confidences'] * (1 + anomaly_scores), 0, 1
            )
        
        return enhanced_results
    
    def classify_anomalies(self, prediction_errors: np.ndarray,
                          anomaly_predictions: np.ndarray) -> List[Dict[str, Any]]:
        """
        分类异常
        
        Args:
            prediction_errors: 预测误差
            anomaly_predictions: 异常预测
            
        Returns:
            异常分类结果
        """
        results = []
        
        for i, (error, pred) in enumerate(zip(prediction_errors, anomaly_predictions)):
            # 判断是否为异常
            is_anomaly = error > self.anomaly_threshold if self.anomaly_threshold else True
            
            # 异常类型映射
            anomaly_types = ['normal', 'type_1', 'type_2', 'type_3']
            anomaly_type = anomaly_types[pred] if pred < len(anomaly_types) else 'unknown'
            
            result = {
                'sequence_id': i,
                'is_anomaly': bool(is_anomaly),
                'anomaly_type': anomaly_type,
                'prediction_error': float(error),
                'anomaly_confidence': float(anomaly_predictions[i])
            }
            
            results.append(result)
        
        return results
    
    def save_predictions(self, predictions: Dict[str, Any], file_path: str):
        """
        保存预测结果
        
        Args:
            predictions: 预测结果
            file_path: 保存路径
        """
        # 转换为可序列化的格式
        serializable_predictions = {}
        for key, value in predictions.items():
            if isinstance(value, np.ndarray):
                serializable_predictions[key] = value.tolist()
            else:
                serializable_predictions[key] = value
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_predictions, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"预测结果已保存到: {file_path}")
    
    def load_predictions(self, file_path: str) -> Dict[str, Any]:
        """
        加载预测结果
        
        Args:
            file_path: 文件路径
            
        Returns:
            预测结果
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        
        # 转换回numpy数组
        for key, value in predictions.items():
            if isinstance(value, list):
                predictions[key] = np.array(value)
        
        return predictions
    
    def get_prediction_summary(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取预测摘要
        
        Args:
            predictions: 预测结果
            
        Returns:
            预测摘要
        """
        anomaly_predictions = predictions['anomaly_predictions']
        anomaly_confidences = predictions['anomaly_confidences']
        
        # 统计信息
        unique_predictions, counts = np.unique(anomaly_predictions, return_counts=True)
        prediction_distribution = dict(zip(unique_predictions, counts))
        
        # 置信度统计
        confidence_stats = {
            'mean': float(np.mean(anomaly_confidences)),
            'std': float(np.std(anomaly_confidences)),
            'min': float(np.min(anomaly_confidences)),
            'max': float(np.max(anomaly_confidences))
        }
        
        summary = {
            'total_predictions': len(anomaly_predictions),
            'prediction_distribution': prediction_distribution,
            'confidence_stats': confidence_stats,
            'anomaly_threshold': self.anomaly_threshold
        }
        
        return summary 