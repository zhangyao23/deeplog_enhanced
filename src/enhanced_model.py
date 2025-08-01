#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的LSTM模型
结合上下文进行全局分析
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any
import logging

class EnhancedDeepLogModel(nn.Module):
    """改进的DeepLog模型"""
    
    def __init__(self, config: Dict[str, Any], num_event_types: int, num_anomaly_types: int, feature_dim: int):
        """
        初始化模型
        
        Args:
            config: 配置字典
            num_event_types: 事件类型数量
            num_anomaly_types: 异常类型数量
            feature_dim: 特征维度
        """
        super(EnhancedDeepLogModel, self).__init__()
        
        self.config = config
        self.model_config = config['model_config']
        self.feature_dim = feature_dim
        
        # 模型参数
        self.hidden_size = self.model_config['lstm_hidden_size']
        self.num_layers = self.model_config['lstm_num_layers']
        self.lstm_dropout = self.model_config['lstm_dropout']
        self.use_attention = self.model_config['use_attention']
        self.attention_heads = self.model_config['attention_heads']
        self.attention_dropout = self.model_config['attention_dropout']
        
        # 输出维度
        self.num_event_types = num_event_types
        self.num_anomaly_types = num_anomaly_types
        
        # LSTM编码器
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.lstm_dropout
        )
        
        # 注意力机制
        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=self.hidden_size,
                num_heads=self.attention_heads,
                batch_first=True,
                dropout=self.attention_dropout
            )
        
        # 序列预测头（原始DeepLog功能）
        self.sequence_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.model_config['sequence_head_dropout']),
            nn.Linear(self.hidden_size // 2, num_event_types)
        )
        
        # 异常分类头（新增功能）
        # 输入维度为 LSTM 输出的 hidden_size + 外部特征的 feature_dim
        self.anomaly_head = nn.Sequential(
            nn.Linear(self.hidden_size + self.feature_dim, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.model_config['anomaly_head_dropout']),
            nn.Linear(self.hidden_size // 2, self.hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(self.model_config['anomaly_head_dropout']),
            nn.Linear(self.hidden_size // 4, num_anomaly_types)
        )
        
        # 全局上下文分析层
        self.context_analyzer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_size // 2, self.hidden_size // 4)
        )
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def forward(self, x_seq: torch.Tensor, x_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x_seq: 输入序列张量 (batch_size, sequence_length)
            x_feat: 输入特征张量 (batch_size, feature_dim)
            
        Returns:
            序列预测和异常分类结果
        """
        # LSTM只处理序列
        x_seq = x_seq.unsqueeze(-1)
        
        # LSTM编码
        h0 = torch.zeros(self.num_layers, x_seq.size(0), self.hidden_size).to(x_seq.device)
        c0 = torch.zeros(self.num_layers, x_seq.size(0), self.hidden_size).to(x_seq.device)
        
        lstm_out, _ = self.lstm(x_seq, (h0, c0))
        
        # 注意力机制
        if self.use_attention:
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            sequence_features = attn_out[:, -1, :]
        else:
            sequence_features = lstm_out[:, -1, :]
        
        # 序列预测
        sequence_pred = self.sequence_head(sequence_features)
        
        # 异常分类（结合LSTM输出和外部特征）
        combined_features = torch.cat([sequence_features, x_feat], dim=1)
        anomaly_pred = self.anomaly_head(combined_features)
        
        return sequence_pred, anomaly_pred
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取注意力权重（用于可视化）
        
        Args:
            x: 输入张量
            
        Returns:
            注意力权重
        """
        if not self.use_attention:
            return None
        
        x = x.unsqueeze(-1) # 增加一个维度以匹配LSTM输入
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        _, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        return attn_weights
    
    def predict_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """
        仅进行序列预测
        
        Args:
            x: 输入张量
            
        Returns:
            序列预测结果
        """
        sequence_pred, _ = self.forward(x)
        return sequence_pred
    
    def predict_anomaly(self, x: torch.Tensor) -> torch.Tensor:
        """
        仅进行异常分类
        
        Args:
            x: 输入张量
            
        Returns:
            异常分类结果
        """
        _, anomaly_pred = self.forward(x)
        return anomaly_pred
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        model_info = {
            'model_type': 'EnhancedDeepLogModel',
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'use_attention': self.use_attention,
            'attention_heads': self.attention_heads if self.use_attention else 0,
            'num_event_types': self.num_event_types,
            'num_anomaly_types': self.num_anomaly_types,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }
        
        return model_info 