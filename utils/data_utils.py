#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据处理工具
"""

import numpy as np
import json
import os
from typing import Dict, Any, List, Tuple
import logging

class DataUtils:
    """数据处理工具类"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str):
        """
        保存配置文件
        
        Args:
            config: 配置字典
            config_path: 配置文件路径
        """
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def ensure_directory(directory: str):
        """
        确保目录存在
        
        Args:
            directory: 目录路径
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def load_sequences(file_path: str) -> List[List[int]]:
        """
        从文本文件加载序列数据。
        
        Args:
            file_path: 数据文件路径。
            
        Returns:
            包含整数序列的列表。
        """
        sequences = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                sequences.append([int(e) for e in line.strip().split()])
        return sequences

    @staticmethod
    def load_labels(file_path: str) -> List[int]:
        """
        从文本文件加载标签数据。
        
        Args:
            file_path: 标签文件路径。
            
        Returns:
            包含整数标签的列表。
        """
        labels = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                labels.append(int(line.strip()))
        return labels
    
    @staticmethod
    def split_data(data: np.ndarray, labels: np.ndarray, 
                  train_ratio: float = 0.8, random_state: int = 42) -> Tuple:
        """
        分割数据为训练集和测试集
        
        Args:
            data: 数据数组
            labels: 标签数组
            train_ratio: 训练集比例
            random_state: 随机种子
            
        Returns:
            训练数据、测试数据、训练标签、测试标签
        """
        np.random.seed(random_state)
        indices = np.random.permutation(len(data))
        
        train_size = int(len(data) * train_ratio)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        train_data = data[train_indices]
        test_data = data[test_indices]
        train_labels = labels[train_indices]
        test_labels = labels[test_indices]
        
        return train_data, test_data, train_labels, test_labels
    
    @staticmethod
    def normalize_features(features: np.ndarray, method: str = 'standard') -> np.ndarray:
        """
        特征标准化
        
        Args:
            features: 特征数组
            method: 标准化方法 ('standard', 'minmax', 'robust')
            
        Returns:
            标准化后的特征
        """
        if method == 'standard':
            # Z-score标准化
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0)
            normalized = (features - mean) / (std + 1e-8)
        elif method == 'minmax':
            # Min-Max标准化
            min_val = np.min(features, axis=0)
            max_val = np.max(features, axis=0)
            normalized = (features - min_val) / (max_val - min_val + 1e-8)
        elif method == 'robust':
            # 鲁棒标准化
            median = np.median(features, axis=0)
            mad = np.median(np.abs(features - median), axis=0)
            normalized = (features - median) / (mad + 1e-8)
        else:
            raise ValueError(f"不支持的标准化方法: {method}")
        
        return normalized
    
    @staticmethod
    def remove_outliers(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """
        移除异常值
        
        Args:
            data: 数据数组
            threshold: 异常值阈值（标准差的倍数）
            
        Returns:
            清理后的数据
        """
        z_scores = np.abs((data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8))
        mask = np.all(z_scores < threshold, axis=1)
        return data[mask]
    
    @staticmethod
    def balance_dataset(data: np.ndarray, labels: np.ndarray, 
                       method: str = 'undersample') -> Tuple[np.ndarray, np.ndarray]:
        """
        平衡数据集
        
        Args:
            data: 数据数组
            labels: 标签数组
            method: 平衡方法 ('undersample', 'oversample')
            
        Returns:
            平衡后的数据和标签
        """
        unique_labels, counts = np.unique(labels, return_counts=True)
        min_count = np.min(counts)
        
        if method == 'undersample':
            # 欠采样
            balanced_data = []
            balanced_labels = []
            
            for label in unique_labels:
                label_indices = np.where(labels == label)[0]
                selected_indices = np.random.choice(label_indices, min_count, replace=False)
                balanced_data.append(data[selected_indices])
                balanced_labels.extend([label] * min_count)
            
            return np.vstack(balanced_data), np.array(balanced_labels)
        
        elif method == 'oversample':
            # 过采样
            balanced_data = []
            balanced_labels = []
            
            for label in unique_labels:
                label_indices = np.where(labels == label)[0]
                if len(label_indices) < min_count:
                    # 需要过采样
                    selected_indices = np.random.choice(label_indices, min_count, replace=True)
                else:
                    # 不需要过采样
                    selected_indices = np.random.choice(label_indices, min_count, replace=False)
                
                balanced_data.append(data[selected_indices])
                balanced_labels.extend([label] * min_count)
            
            return np.vstack(balanced_data), np.array(balanced_labels)
        
        else:
            raise ValueError(f"不支持的平衡方法: {method}")
    
    @staticmethod
    def save_numpy_data(data: np.ndarray, file_path: str):
        """
        保存numpy数据
        
        Args:
            data: 数据数组
            file_path: 文件路径
        """
        np.save(file_path, data)
    
    @staticmethod
    def load_numpy_data(file_path: str) -> np.ndarray:
        """
        加载numpy数据
        
        Args:
            file_path: 文件路径
            
        Returns:
            数据数组
        """
        return np.load(file_path) 