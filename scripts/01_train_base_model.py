#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段一：训练“正常模式专家”模型
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import logging
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# 添加项目路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import DataPreprocessor
from src.base_model import SimpleLSTM
from utils.data_utils import DataUtils

# --- 日志和设备配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_prepare_data(config, data_path, label_path):
    """加载所有数据以构建完整的事件映射，然后筛选出正常数据进行处理。"""
    logging.info("加载所有数据以构建全局事件映射...")
    
    # 加载所有序列来构建映射
    with open(data_path, 'r') as f:
        all_sequences = [list(map(int, line.strip().split())) for line in f]
    
    # 使用DataPreprocessor来创建全局映射
    preprocessor = DataPreprocessor(config)
    preprocessor.create_event_mapping(all_sequences)
    num_total_events = len(preprocessor.event_mapping)
    logging.info(f"全局事件映射创建完成，共 {num_total_events} 个唯一事件。")
    
    # 更新配置
    config['data_config']['num_event_types'] = num_total_events

    # 现在，加载并筛选正常数据
    logging.info("加载并筛选正常数据用于训练...")
    with open(label_path, 'r') as f:
        all_labels = [int(line.strip()) for line in f]

    normal_sequences_raw = [seq for seq, label in zip(all_sequences, all_labels) if label == 0]
    
    # 应用已创建的全局映射
    mapped_sequences = preprocessor.apply_event_mapping(normal_sequences_raw)
    
    # 滑窗
    window_size = config['data_config']['window_size']
    inputs, targets = preprocessor.split_sequences(mapped_sequences, window_size)

    # 补齐
    padded_inputs = preprocessor.pad_sequences(inputs, pad_to_length=window_size)
    
    # 划分训练集和验证集
    train_inputs, val_inputs, train_targets, val_targets = train_test_split(
        padded_inputs, np.array(targets), test_size=0.2, random_state=42
    )
    
    logging.info(f"数据划分完成: 训练集 {len(train_inputs)}条, 验证集 {len(val_inputs)}条")
    
    return train_inputs, val_inputs, train_targets, val_targets, config, preprocessor

def train_base_model(config, train_inputs, val_inputs, train_targets, val_targets, model_save_path, preprocessor):
    """使用早停机制训练基础模型"""
    logging.info("开始使用早停机制训练基础模型...")
    
    # 准备数据加载器
    train_dataset = TensorDataset(torch.FloatTensor(train_inputs), torch.LongTensor(train_targets))
    train_loader = DataLoader(train_dataset, batch_size=config['training_config']['batch_size'], shuffle=True)
    
    val_dataset = TensorDataset(torch.FloatTensor(val_inputs), torch.LongTensor(val_targets))
    val_loader = DataLoader(val_dataset, batch_size=config['training_config']['batch_size'], shuffle=False)
    
    # 创建简化的基础模型
    num_event_types = config['data_config']['num_event_types']
    model = SimpleLSTM(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        num_keys=num_event_types
    )
    model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # 早停机制参数
    patience = 5
    best_val_loss = float('inf')
    patience_counter = 0
    
    num_epochs = 80
    
    for epoch in range(num_epochs):
        # --- 训练 ---
        model.train()
        epoch_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [训练]")
        for batch_inputs, batch_targets in pbar:
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
            optimizer.zero_grad()
            seq_pred, _ = model(batch_inputs)
            loss = criterion(seq_pred, batch_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training_config']['gradient_clip'])
            optimizer.step()
            epoch_train_loss += loss.item()
            pbar.set_postfix({'train_loss': loss.item()})
        avg_train_loss = epoch_train_loss / len(train_loader)

        # --- 验证 ---
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for batch_inputs, batch_targets in val_loader:
                batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
                seq_pred, _ = model(batch_inputs)
                loss = criterion(seq_pred, batch_targets)
                epoch_val_loss += loss.item()
        avg_val_loss = epoch_val_loss / len(val_loader)
        
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, 训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}")

        # --- 早停判断 ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # 保存时一并保存event_mapping
            config['event_mapping'] = preprocessor.event_mapping
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'model_class': 'SimpleLSTM'
            }, model_save_path)
            logging.info(f"🎉 验证损失降低，保存最佳模型到: {model_save_path}")
        else:
            patience_counter += 1
            logging.info(f"验证损失未降低，耐心计数: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            logging.info("🚫 触发早停机制，训练结束。")
            break
            
    return model

def main():
    """主函数"""
    print("🚀 Enhanced DeepLog - 阶段一：训练“正常模式专家”模型 🚀")
    print("="*60)
    
    # 加载配置
    config_path = 'config/model_config.json'
    if not os.path.exists(os.path.join('enhanced_deeplog', config_path)):
        logging.error(f"配置文件未找到: {config_path}")
        return
    config = DataUtils.load_config(os.path.join('enhanced_deeplog', config_path))

    # 数据路径
    data_path = 'enhanced_deeplog/data/raw/synthetic_hdfs_train.txt'
    label_path = 'enhanced_deeplog/data/raw/synthetic_hdfs_train_labels.txt'

    if not os.path.exists(data_path):
        logging.error("请先运行 00_generate_synthetic_data.py 生成数据。")
        return
        
    # 1. 加载和准备数据
    train_inputs, val_inputs, train_targets, val_targets, updated_config, preprocessor = load_and_prepare_data(
        config, 
        data_path, 
        label_path
    )
    
    # 2. 训练基础模型
    model_dir = 'enhanced_deeplog/models'
    DataUtils.ensure_directory(model_dir)
    model_save_path = os.path.join(model_dir, 'base_model.pth')
    
    base_model = train_base_model(updated_config, train_inputs, val_inputs, train_targets, val_targets, model_save_path, preprocessor)
    
    logging.info(f"✅ 训练完成。最佳模型已保存在: {model_save_path}")

if __name__ == "__main__":
    main() 