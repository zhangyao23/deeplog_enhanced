#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
定义基础的序列预测模型
"""
import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    """
    一个简化的LSTM模型，专用于序列预测任务，作为“正常模式专家”。
    """
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, x):
        """
        前向传播。
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len)
            
        Returns:
            序列预测 logits, None (为了与双头模型输出格式保持一致)
        """
        # LSTM期望的输入形状是 (batch_size, seq_len, input_size)
        x = x.unsqueeze(-1)
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))
        
        # 我们只取序列最后一个时间步的输出来预测下一个事件
        out = self.fc(out[:, -1, :])
        
        # 返回None是为了与EnhancedDeepLogModel的(seq_pred, anomaly_pred)输出格式保持一致
        return out, None 