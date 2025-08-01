#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断脚本：评估基础模型的效果
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import DataPreprocessor
from src.base_model import SimpleLSTM
from utils.data_utils import DataUtils

# --- 日志和设备配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_base_model(model_path):
    """根据保存的信息加载正确的模型"""
    logging.info(f"从 {model_path} 加载基础模型...")
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    model_class = checkpoint.get('model_class')
    if model_class != 'SimpleLSTM':
        raise TypeError(f"此评估脚本专为SimpleLSTM模型设计。")
        
    num_event_types = config['data_config']['num_event_types']
    
    model = SimpleLSTM(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        num_keys=num_event_types
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, config

def calculate_sequence_prediction_errors(model, preprocessor, config, sequences):
    """计算整个序列的平均预测误差"""
    all_avg_errors = []
    
    preprocessor.logger.setLevel(logging.WARNING)
    
    pbar = tqdm(sequences, desc="计算序列预测误差")
    for seq in pbar:
        if not seq:
            all_avg_errors.append(0.0)
            continue
            
        mapped_seqs = preprocessor.apply_event_mapping([seq])
        inputs, targets = preprocessor.split_sequences(mapped_seqs, config['data_config']['window_size'])

        if len(inputs) == 0:
            all_avg_errors.append(0.0)
            continue

        inputs_tensor = torch.FloatTensor(inputs).to(device)
        targets_tensor = torch.LongTensor(targets).to(device)

        with torch.no_grad():
            seq_pred, _ = model(inputs_tensor)
            probs = torch.softmax(seq_pred, dim=1)
            
            # 确保targets在合法范围内
            valid_targets_mask = targets_tensor < seq_pred.shape[1]
            if not valid_targets_mask.all():
                logging.warning(f"发现越界目标索引，将跳过。序列: {seq[:10]}...")
                targets_tensor = targets_tensor[valid_targets_mask]
                if targets_tensor.numel() == 0:
                    all_avg_errors.append(1.0) # 如果所有目标都无效，则视为最大误差
                    continue

            target_probs = probs.gather(1, targets_tensor.unsqueeze(1)).squeeze()
            prediction_errors = (1 - target_probs).cpu().numpy()
        
        avg_error = np.mean(prediction_errors) if prediction_errors.size > 0 else 0.0
        all_avg_errors.append(avg_error)
        
    return np.array(all_avg_errors)

def plot_error_distribution(normal_errors, abnormal_errors, output_path):
    """绘制并保存误差分布图"""
    logging.info(f"绘制误差分布图并保存到 {output_path}")
    plt.figure(figsize=(12, 7))
    sns.histplot(normal_errors, color="blue", label='Normal Sequences', kde=True, stat="density", element="step")
    sns.histplot(abnormal_errors, color="red", label='Abnormal Sequences', kde=True, stat="density", element="step")
    plt.title('Prediction Error Distribution (Base Model)', fontsize=16)
    plt.xlabel('Average Prediction Error', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(output_path)
    plt.close()
    logging.info("✅ 图表保存成功。")

def main():
    """主函数"""
    print("🚀 Enhanced DeepLog - 诊断：评估基础模型 🚀")
    print("="*60)
    
    # --- 加载模型和配置 ---
    model_path = 'enhanced_deeplog/models/base_model.pth'
    if not os.path.exists(model_path):
        logging.error("基础模型 'base_model.pth' 未找到. 请先运行 01_train_base_model.py。")
        return
        
    model, config = load_base_model(model_path)
    
    # --- 加载数据 ---
    data_path = 'enhanced_deeplog/data/raw/synthetic_hdfs_train.txt'
    label_path = 'enhanced_deeplog/data/raw/synthetic_hdfs_train_labels.txt'
    with open(data_path, 'r') as f:
        all_sequences = [list(map(int, line.strip().split())) for line in f]
    with open(label_path, 'r') as f:
        all_labels = np.array([int(line.strip()) for line in f])

    # --- 计算误差 ---
    preprocessor = DataPreprocessor(config)
    preprocessor.event_mapping = config['event_mapping']
    
    all_errors = calculate_sequence_prediction_errors(model, preprocessor, config, all_sequences)
    
    normal_errors = all_errors[all_labels == 0]
    abnormal_errors = all_errors[all_labels != 0]

    # --- 分析和报告 ---
    print("\n--- 预测误差统计分析 ---")
    print(f"正常序列 (共 {len(normal_errors)} 条):")
    print(f"  - 平均误差: {np.mean(normal_errors):.4f}")
    print(f"  - 中位数误差: {np.median(normal_errors):.4f}")
    print(f"  - 误差标准差: {np.std(normal_errors):.4f}")
    print(f"  - 最大误差: {np.max(normal_errors):.4f}\n")
    
    print(f"异常序列 (共 {len(abnormal_errors)} 条):")
    print(f"  - 平均误差: {np.mean(abnormal_errors):.4f}")
    print(f"  - 中位数误差: {np.median(abnormal_errors):.4f}")
    print(f"  - 误差标准差: {np.std(abnormal_errors):.4f}")
    print(f"  - 最小误差: {np.min(abnormal_errors):.4f}\n")
    
    # --- 可视化 ---
    output_dir = 'enhanced_deeplog/results'
    DataUtils.ensure_directory(output_dir)
    plot_path = os.path.join(output_dir, 'base_model_evaluation.png')
    plot_error_distribution(normal_errors, abnormal_errors, plot_path)
    
    print(f"📈 评估图表已保存到: {plot_path}")
    print("="*60)

if __name__ == "__main__":
    main() 