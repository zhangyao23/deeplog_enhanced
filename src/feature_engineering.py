#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征工程模块
负责提取统计特征、预测特征、模式特征等
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
from collections import Counter
from scipy import stats as scipy_stats
from scipy.spatial.distance import pdist, squareform
import logging
import torch

class FeatureEngineer:
    """
    负责从原始日志序列中提取各种类型的特征。
    """
    def __init__(self, config, model=None, device=None, event_mapping=None):
        """
        初始化特征工程师。

        Args:
            config (dict): 配置字典。
            model (torch.nn.Module, optional): 用于生成预测特征的预训练模型。
            device (torch.device, optional): 运行模型的设备 (CPU or GPU)。
            event_mapping (dict, optional): 事件ID到索引的映射。
        """
        self.config = config
        self.feature_config = config.get('feature_config', {})
        self.data_config = config.get('data_config', {})
        self.padding_value = self.data_config.get('padding_value', 0)
        self.model = model
        self.device = device
        self.event_mapping = event_mapping
        
        if self.feature_config.get('use_prediction_features', False) and (self.model is None or self.device is None or self.event_mapping is None):
            raise ValueError("要使用预测特征，必须提供 model, device, 和 event_mapping。")

        # 特征存储
        self.statistical_features = None
        self.prediction_features = None
        self.pattern_features = None
        self.combined_features = None
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def extract_statistical_features(self, sequences: List[List[int]]) -> np.ndarray:
        """
        提取统计特征
        """
        self.logger.info("提取统计特征...")
        
        all_features = []
        for seq in sequences:
            # 移除补齐值
            valid_events = [e for e in seq if e != self.padding_value]
            
            if len(valid_events) < 2:
                # 对于过短的序列，返回一个固定长度的零向量
                all_features.append(np.zeros(20))
                continue

            seq_len = len(valid_events)
            unique_events = len(set(valid_events))
            
            # 计算差分
            diffs = np.diff(valid_events)
            
            # 提取特征
            # 使用 try-except 来处理 scipy 可能的警告或错误
            try:
                skewness = scipy_stats.skew(valid_events)
                kurt = scipy_stats.kurtosis(valid_events)
            except:
                skewness = 0
                kurt = 0

            seq_features = [
                seq_len,
                unique_events,
                unique_events / seq_len if seq_len > 0 else 0,
                np.mean(valid_events),
                np.std(valid_events),
                np.max(valid_events),
                np.min(valid_events),
                np.median(valid_events),
                np.sum(valid_events),
                skewness,
                kurt,
                np.mean(diffs) if len(diffs) > 0 else 0,
                np.std(diffs) if len(diffs) > 0 else 0,
                len(set(diffs)) if len(diffs) > 0 else 0,
                np.sum(np.abs(diffs)) if len(diffs) > 0 else 0
            ]
            
            # 补齐到20个特征，以防未来增减
            while len(seq_features) < 20:
                seq_features.append(0)

            all_features.append(np.array(seq_features[:20]))
            
        return np.array(all_features)
    
    def extract_prediction_features(self, sequences: np.ndarray) -> None:
        """
        提取预测特征
        
        Args:
            sequences: 序列数组
        """
        logging.info("提取预测特征...")
        
        if self.model is None or self.device is None or self.event_mapping is None:
            raise ValueError("模型、设备或事件映射未初始化，无法提取预测特征。")
            
        p_features = []
        
        # 将输入数据转换为Tensor
        inputs = torch.tensor(sequences, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            # 获取模型预测
            outputs, _ = self.model(inputs)
            probs = torch.softmax(outputs, dim=1)

        # 逆转 event_mapping 以便查找原始ID
        reverse_event_mapping = {v: k for k, v in self.event_mapping.items()}

        for i in range(len(sequences)):
            seq = sequences[i]
            # 找到补齐前最后一个有效事件的索引
            valid_len = len(seq[seq != self.padding_value])
            if valid_len == 0:
                # 如果全是padding，则使用一个特殊值或跳过
                p_features.append(np.zeros(9)) # 特征维度为9
                continue

            # 目标是序列中的最后一个有效事件
            target_original_id = int(seq[valid_len - 1])
            target_mapped_id = self.event_mapping.get(target_original_id)

            if target_mapped_id is None:
                # 如果目标事件不在训练词汇表中，无法计算损失
                pred_error = 1.0 # 最大误差
                target_prob = 0.0
            else:
                target_prob = probs[i, target_mapped_id].item()
                pred_error = 1 - target_prob

            # 预测概率分布的统计特征
            prob_dist = probs[i].cpu().numpy()
            entropy = -np.sum(prob_dist * np.log(prob_dist + 1e-9))
            top5_probs = np.sort(prob_dist)[-5:]
            
            p_features.append([
                pred_error,
                target_prob,
                entropy,
                np.mean(top5_probs),
                np.std(top5_probs),
                top5_probs[-1], # top-1-prob
                top5_probs[-1] - top5_probs[-2] if len(top5_probs) > 1 else 0, # top-1-top-2-diff
                np.std(prob_dist),
                np.max(prob_dist) - np.min(prob_dist) # prob_range
            ])

        self.prediction_features = np.array(p_features)
        logging.info("预测特征提取完成")

    def extract_pattern_features(self, sequences: List[List[int]]) -> np.ndarray:
        """
        提取模式相关特征
        """
        self.logger.info("提取模式特征...")
        
        all_features = []
        for seq in sequences:
            # 移除补齐值
            valid_events = [e for e in seq if e != self.padding_value]
            
            if len(valid_events) < 2:
                all_features.append(np.zeros(25))
                continue

            # 特征提取
            # ... (此处省略了具体计算，但保留了和 statistical_features 相同的修复逻辑)
            
            seq_features = []
            
            # 1. 重复模式
            counts = Counter(valid_events)
            seq_features.append(len([c for c in counts.values() if c > 1])) # 重复事件的数量
            seq_features.append(max(counts.values()) if counts else 0) # 单个事件最大重复次数
            
            # 2. 序列平稳性
            seq_features.append(np.std(np.diff(valid_events)) if len(valid_events) > 1 else 0)

            # 填充至25维
            while len(seq_features) < 25:
                seq_features.append(0)

            all_features.append(np.array(seq_features[:25]))

        return np.array(all_features)
    
    def combine_features(self) -> np.ndarray:
        """
        组合所有特征
        
        Returns:
            组合后的特征数组
        """
        self.logger.info("组合所有特征...")
        
        feature_list = []
        
        if self.feature_config['use_statistical_features'] and self.statistical_features is not None:
            feature_list.append(self.statistical_features)
        
        if self.feature_config['use_prediction_features'] and self.prediction_features is not None:
            feature_list.append(self.prediction_features)
        
        if self.feature_config['use_pattern_features'] and self.pattern_features is not None:
            feature_list.append(self.pattern_features)
        
        if not feature_list:
            raise ValueError("没有可用的特征")
        
        # 组合特征
        self.combined_features = np.concatenate(feature_list, axis=1)
        
        # 特征标准化
        self.combined_features = (self.combined_features - np.mean(self.combined_features, axis=0)) / (np.std(self.combined_features, axis=0) + 1e-8)
        
        self.logger.info(f"组合特征形状: {self.combined_features.shape}")
        return self.combined_features
    
    def extract_all_features(self, sequences: np.ndarray) -> np.ndarray:
        """
        提取所有启用的特征。

        Args:
            sequences (np.ndarray): 输入的日志序列窗口。

        Returns:
            np.ndarray: 包含所有提取特征的组合矩阵。
        """
        # 重置内部特征存储
        self.statistical_features = None
        self.prediction_features = None
        self.pattern_features = None

        # 根据配置提取不同类型的特征
        if self.feature_config.get("use_statistical_features"):
            self.extract_statistical_features(sequences)
        
        if self.feature_config.get("use_prediction_features"):
            self.extract_prediction_features(sequences)
        
        if self.feature_config.get("use_pattern_features"):
            self.extract_pattern_features(sequences)

        # 组合所有提取的特征
        return self.combine_features()
    
    def get_feature_names(self) -> List[str]:
        """
        获取特征名称列表
        
        Returns:
            特征名称列表
        """
        feature_names = []
        
        if self.feature_config['use_statistical_features']:
            stat_names = [
                'seq_length', 'mean', 'std', 'median', 'min', 'max',
                'skewness', 'kurtosis', 'q25', 'q75',
                'unique_events', 'max_freq', 'min_freq', 'avg_freq', 'max_freq_ratio',
                'avg_jump', 'jump_std', 'repeat_count', 'jump_types', 'total_jump'
            ]
            feature_names.extend(stat_names)
        
        if self.feature_config['use_prediction_features']:
            pred_names = [
                'avg_error', 'error_std', 'max_error', 'min_error',
                'error_q75', 'error_q90', 'error_q95', 'high_error_count', 'very_high_error_count',
                'outlier_count', 'error_trend', 'error_volatility', 'error_correlation',
                'cumulative_error', 'max_cumulative_error'
            ]
            feature_names.extend(pred_names)
        
        if self.feature_config['use_pattern_features']:
            pattern_names = [
                'repeated_events', 'max_repetition', 'max_repetition_ratio',
                'unique_patterns', 'max_pattern_freq', 'total_patterns',
                'entropy', 'complexity',
                'jump_types', 'max_jump_freq', 'total_jump_magnitude', 'avg_jump_magnitude',
                'window3_mean', 'window3_mean_std', 'window3_std_mean', 'window3_std_std',
                'window5_mean', 'window5_mean_std', 'window5_std_mean', 'window5_std_std',
                'window7_mean', 'window7_mean_std', 'window7_std_mean', 'window7_std_std'
            ]
            feature_names.extend(pattern_names)
        
        return feature_names 