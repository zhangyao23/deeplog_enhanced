#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理模块
负责序列补齐、数据清洗、格式转换等
"""

import numpy as np
import json
from typing import List, Tuple, Dict, Any
from collections import defaultdict, Counter
import logging

class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据预处理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.data_config = config['data_config']
        self.max_length = self.data_config['max_sequence_length']
        self.min_length = self.data_config['min_sequence_length']
        self.padding_value = self.data_config['padding_value']
        
        # 统计信息
        self.sequence_stats = {}
        self.event_mapping = {}
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_data(self, file_path: str) -> List[List[int]]:
        """
        加载原始数据
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            序列列表
        """
        self.logger.info(f"加载数据: {file_path}")
        
        sequences = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # 解析序列，我们的数据已经是0-based，无需-1
                    sequence = [int(x) for x in line.split()]
                    if len(sequence) >= self.min_length:
                        sequences.append(sequence)
        
        self.logger.info(f"加载了 {len(sequences)} 个序列")
        return sequences
    
    def analyze_sequences(self, sequences: List[List[int]]) -> Dict[str, Any]:
        """
        分析序列统计信息
        
        Args:
            sequences: 序列列表
            
        Returns:
            统计信息字典
        """
        self.logger.info("分析序列统计信息...")
        
        lengths = [len(seq) for seq in sequences]
        all_events = [event for seq in sequences for event in seq]
        
        stats = {
            'num_sequences': len(sequences),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'avg_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'num_unique_events': len(set(all_events)),
            'event_frequency': Counter(all_events),
            'length_distribution': Counter(lengths)
        }
        
        self.sequence_stats = stats
        self.logger.info(f"序列统计: {stats}")
        return stats
    
    def pad_sequences(self, sequences: List[List[int]], pad_to_length: int = None) -> np.ndarray:
        """
        补齐序列到相同长度
        
        Args:
            sequences: 序列列表
            pad_to_length: (可选) 指定补齐的目标长度。如果为None，则使用配置中的max_sequence_length。
            
        Returns:
            补齐后的序列数组
        """
        target_length = pad_to_length if pad_to_length is not None else self.max_length
        self.logger.info(f"补齐序列到长度 {target_length}...")
        
        padded_sequences = []
        for sequence in sequences:
            if len(sequence) > target_length:
                # 截断长序列
                padded_seq = sequence[:target_length]
            else:
                # 补齐短序列
                padded_seq = sequence + [self.padding_value] * (target_length - len(sequence))
            padded_sequences.append(padded_seq)
        
        return np.array(padded_sequences)
    
    def create_event_mapping(self, sequences: List[List[int]]) -> Dict[int, int]:
        """
        创建事件ID映射
        
        Args:
            sequences: 序列列表
            
        Returns:
            事件映射字典
        """
        all_events = set()
        for sequence in sequences:
            all_events.update(sequence)
        
        # 创建0-based的连续映射
        event_mapping = {event: idx for idx, event in enumerate(sorted(all_events))}
        self.event_mapping = event_mapping
        
        self.logger.info(f"创建了 {len(event_mapping)} 个事件的映射")
        return event_mapping
    
    def apply_event_mapping(self, sequences: List[List[int]]) -> List[List[int]]:
        """
        应用事件映射
        
        Args:
            sequences: 原始序列列表
            
        Returns:
            映射后的序列列表
        """
        if not self.event_mapping:
            self.create_event_mapping(sequences)
        
        mapped_sequences = []
        for sequence in sequences:
            mapped_seq = [self.event_mapping.get(event, 0) for event in sequence]
            mapped_sequences.append(mapped_seq)
        
        return mapped_sequences
    
    def split_sequences(self, sequences: List[List[int]], window_size: int) -> Tuple[List[List[int]], List[int]]:
        """
        将序列分割为滑动窗口
        
        Args:
            sequences: 序列列表
            window_size: 窗口大小
            
        Returns:
            输入序列和目标序列
        """
        self.logger.info(f"使用窗口大小 {window_size} 分割序列...")
        
        input_sequences = []
        target_sequences = []
        
        for sequence in sequences:
            for i in range(len(sequence) - window_size):
                input_seq = sequence[i:i + window_size]
                target_seq = sequence[i + window_size]
                input_sequences.append(input_seq)
                target_sequences.append(target_seq)
        
        self.logger.info(f"生成了 {len(input_sequences)} 个训练样本")
        return input_sequences, target_sequences
    
    def split_and_pad_sequence(self, sequence: List[int], window_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        将单个序列切分成窗口，并进行填充。
        
        Args:
            sequence (List[int]): 单个日志序列。
            window_size (int): 滑动窗口的大小。
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 返回处理好的输入窗口和对应的目标事件。
        """
        # 1. 切分序列
        inputs, targets = self.split_sequences([sequence], window_size)
        
        if not inputs:
            return np.array([]), np.array([])

        # 2. 填充
        # pad_sequences 期望处理多个序列，所以我们直接传入已经切分好的窗口列表
        padded_inputs = self.pad_sequences(inputs, self.max_length)
        
        return padded_inputs, np.array(targets)

    def preprocess(self, file_path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        完整的数据预处理流程
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            处理后的数据、统计信息
        """
        # 1. 加载数据
        sequences = self.load_data(file_path)
        
        # 2. 分析统计信息
        stats = self.analyze_sequences(sequences)
        
        # 3. 应用事件映射
        mapped_sequences = self.apply_event_mapping(sequences)
        
        # 4. 【修正流程】先分割为训练样本
        window_size = self.data_config['window_size']
        input_sequences, target_sequences = self.split_sequences(mapped_sequences, window_size)
        
        # 5. 【修正流程】再对分割后的输入序列进行补齐
        # 注意：在此设计中，所有input_sequences长度都等于window_size，所以补齐是无效的，
        # 但为了流程的通用性，我们保留这一步，并使用window_size作为补齐长度。
        padded_inputs = self.pad_sequences(input_sequences, pad_to_length=window_size)
        targets = np.array(target_sequences)
        
        self.logger.info(f"预处理完成: 输入形状 {padded_inputs.shape}, 目标形状 {targets.shape}")
        
        return padded_inputs, targets, stats
    
    def save_preprocessing_info(self, file_path: str):
        """
        保存预处理信息
        
        Args:
            file_path: 保存路径
        """
        info = {
            'sequence_stats': self.sequence_stats,
            'event_mapping': self.event_mapping,
            'config': self.config
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"预处理信息已保存到: {file_path}")
    
    def load_preprocessing_info(self, file_path: str):
        """
        加载预处理信息
        
        Args:
            file_path: 文件路径
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
        
        self.sequence_stats = info['sequence_stats']
        self.event_mapping = info['event_mapping']
        
        self.logger.info(f"预处理信息已从 {file_path} 加载") 