#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练模块
负责模型训练和验证
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Any, Tuple, List
import logging
import json
from tqdm import tqdm
import os
import torch.nn.functional as F

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, config: Dict[str, Any], model: nn.Module):
        """
        初始化训练器
        
        Args:
            config: 配置字典
            model: 模型实例
        """
        self.config = config
        self.training_config = config['training_config']
        self.model = model
        
        # 训练参数
        self.batch_size = self.training_config['batch_size']
        self.learning_rate = self.training_config['learning_rate']
        self.num_epochs = self.training_config['num_epochs']
        self.sequence_loss_weight = self.training_config['sequence_loss_weight']
        self.anomaly_loss_weight = self.training_config['anomaly_loss_weight']
        self.weight_decay = self.training_config['weight_decay']
        self.gradient_clip = self.training_config['gradient_clip']
        
        # 训练历史
        self.train_history = {
            'sequence_loss': [],
            'anomaly_loss': [],
            'total_loss': [],
            'sequence_accuracy': [],
            'anomaly_accuracy': []
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def train(self, train_data: Tuple, val_data: Tuple, model_save_path: str) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            train_data: 包含训练序列、特征和标签的元组 (X_train_seq, X_train_feat, y_train)
            val_data: 包含验证序列、特征和标签的元组 (X_val_seq, X_val_feat, y_val)
            model_save_path: 最佳模型的保存路径
            
        Returns:
            训练结果字典
        """
        X_train_seq, X_train_feat, y_train_anomaly = train_data
        X_val_seq, X_val_feat, y_val_anomaly = val_data

        # 确保目录存在
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

        # 获取用于序列预测的目标 (下一个事件)
        # 这里的目标是输入窗口中的最后一个事件
        y_train_seq = np.array([seq[-1] for seq in X_train_seq])
        y_val_seq = np.array([seq[-1] for seq in X_val_seq])

        # 创建数据加载器
        train_dataset = TensorDataset(
            torch.from_numpy(X_train_seq).float(),
            torch.from_numpy(X_train_feat).float(),
            torch.from_numpy(y_train_seq).long(),
            torch.from_numpy(y_train_anomaly).long()
        )
        val_dataset = TensorDataset(
            torch.from_numpy(X_val_seq).float(),
            torch.from_numpy(X_val_feat).float(),
            torch.from_numpy(y_val_seq).long(),
            torch.from_numpy(y_val_anomaly).long()
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        self.logger.info(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
        
        self.logger.info("开始训练模型...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        # 损失函数
        sequence_criterion = nn.CrossEntropyLoss()
        anomaly_criterion = nn.CrossEntropyLoss()
        
        # 优化器
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=3
        )
        
        # 早停
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = self.training_config['early_stopping_patience']
        
        for epoch in range(self.num_epochs):
            # 训练阶段
            train_metrics = self._train_epoch(
                train_loader, optimizer, sequence_criterion, anomaly_criterion, device
            )
            
            # 验证阶段
            val_metrics = self._validate_epoch(
                val_loader, sequence_criterion, anomaly_criterion, device
            )
            
            # 更新学习率
            scheduler.step(val_metrics['total_loss'])
            
            # 记录历史
            for key in self.train_history:
                if key in train_metrics:
                    self.train_history[key].append(train_metrics[key])
            
            # 早停检查
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                patience_counter = 0
                # 保存最佳模型
                torch.save({'model_state_dict': self.model.state_dict(), 'config': self.config}, model_save_path)
                self.logger.info(f"🎉 Val loss improved to {best_val_loss:.4f}. Saving best model.")
            else:
                patience_counter += 1
                self.logger.info(f"Val loss did not improve. Patience: {patience_counter}/{early_stopping_patience}")
            
            # 打印进度
            self.logger.info(
                f"Epoch [{epoch+1}/{self.num_epochs}] | "
                f"Train Loss: {train_metrics['total_loss']:.4f} | "
                f"Val Loss: {val_metrics['total_loss']:.4f} | "
                f"Train Anomaly Acc: {train_metrics['anomaly_accuracy']:.4f}"
            )

            # 早停
            if patience_counter >= early_stopping_patience:
                self.logger.info(f"🚫 Early stopping triggered at epoch {epoch+1}")
                break
        
        # 加载最佳模型
        self.logger.info(f"Loading best model from {model_save_path}")
        self.model.load_state_dict(torch.load(model_save_path)['model_state_dict'])
        
        return {
            'train_history': self.train_history,
            'best_val_loss': best_val_loss,
            'final_epoch': epoch + 1
        }
    
    def _train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer,
                    sequence_criterion: nn.Module, anomaly_criterion: nn.Module,
                    device: torch.device) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_sequence_loss = 0
        total_anomaly_loss = 0
        total_loss = 0
        sequence_correct = 0
        anomaly_correct = 0
        total_samples = 0
        
        for batch_seq, batch_feat, batch_seq_targets, batch_anomaly_targets in tqdm(train_loader, desc=f"Epoch {self.train_history.get('total_loss', [0]).__len__()} Training"):
            batch_seq = batch_seq.to(device)
            batch_feat = batch_feat.to(device)
            batch_seq_targets = batch_seq_targets.to(device)
            batch_anomaly_targets = batch_anomaly_targets.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            sequence_pred, anomaly_pred = self.model(batch_seq, batch_feat)
            
            # 计算损失
            sequence_loss = sequence_criterion(sequence_pred, batch_seq_targets)
            anomaly_loss = anomaly_criterion(anomaly_pred, batch_anomaly_targets)
            total_batch_loss = (self.sequence_loss_weight * sequence_loss + 
                              self.anomaly_loss_weight * anomaly_loss)
            
            # 反向传播
            total_batch_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            optimizer.step()
            
            # 统计
            total_sequence_loss += sequence_loss.item()
            total_anomaly_loss += anomaly_loss.item()
            total_loss += total_batch_loss.item()
            
            # 计算准确率
            sequence_correct += (torch.argmax(sequence_pred, dim=1) == batch_seq_targets).sum().item()
            anomaly_correct += (torch.argmax(anomaly_pred, dim=1) == batch_anomaly_targets).sum().item()
            total_samples += batch_seq.size(0)
        
        return {
            'sequence_loss': total_sequence_loss / len(train_loader),
            'anomaly_loss': total_anomaly_loss / len(train_loader),
            'total_loss': total_loss / len(train_loader),
            'sequence_accuracy': sequence_correct / total_samples,
            'anomaly_accuracy': anomaly_correct / total_samples
        }
    
    def _validate_epoch(self, val_loader: DataLoader, sequence_criterion: nn.Module,
                       anomaly_criterion: nn.Module, device: torch.device) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        
        total_sequence_loss = 0
        total_anomaly_loss = 0
        total_loss = 0
        
        with torch.no_grad():
            for batch_seq, batch_feat, batch_seq_targets, batch_anomaly_targets in tqdm(val_loader, desc="Validating"):
                batch_seq = batch_seq.to(device)
                batch_feat = batch_feat.to(device)
                batch_seq_targets = batch_seq_targets.to(device)
                batch_anomaly_targets = batch_anomaly_targets.to(device)
                
                # 前向传播
                sequence_pred, anomaly_pred = self.model(batch_seq, batch_feat)
                
                # 计算损失
                sequence_loss = sequence_criterion(sequence_pred, batch_seq_targets)
                anomaly_loss = anomaly_criterion(anomaly_pred, batch_anomaly_targets)
                total_batch_loss = (self.sequence_loss_weight * sequence_loss + 
                                  self.anomaly_loss_weight * anomaly_loss)
                
                # 统计
                total_sequence_loss += sequence_loss.item()
                total_anomaly_loss += anomaly_loss.item()
                total_loss += total_batch_loss.item()
        
        return {
            'sequence_loss': total_sequence_loss / len(val_loader),
            'anomaly_loss': total_anomaly_loss / len(val_loader),
            'total_loss': total_loss / len(val_loader)
        }
    
    def predict(self, test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        在测试集上进行预测。

        Args:
            test_loader (DataLoader): 测试数据加载器。

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (真实标签, 预测索引, 预测概率)
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        all_true_labels = []
        all_pred_indices = []
        all_pred_probs = []

        with torch.no_grad():
            for batch_seq, batch_feat, batch_anomaly_targets in tqdm(test_loader, desc="Predicting"):
                batch_seq = batch_seq.to(device)
                batch_feat = batch_feat.to(device)
                
                # 前向传播
                _, anomaly_pred = self.model(batch_seq, batch_feat)
                
                # 获取预测结果
                pred_probs = F.softmax(anomaly_pred, dim=1)
                pred_indices = torch.argmax(pred_probs, dim=1)

                all_true_labels.extend(batch_anomaly_targets.cpu().numpy())
                all_pred_indices.extend(pred_indices.cpu().numpy())
                all_pred_probs.extend(pred_probs.cpu().numpy())

        return np.array(all_true_labels), np.array(all_pred_indices), np.array(all_pred_probs)

    def save_training_history(self, file_path: str):
        """保存训练历史"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.train_history, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"训练历史已保存到: {file_path}") 