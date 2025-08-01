#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
脚本 05: 分析聚类结果

此脚本用于深入理解在阶段三中通过聚类生成的不同类别（Class 0-3）代表了何种日志行为。
它会执行以下操作：
1.  加载配置文件、最终的聚类标签以及原始的、经过处理的序列数据。
2.  加载基础模型以获取事件ID到模板的映射关系。
3.  为每个类别（特别是我们感兴趣的类别）随机抽取几个日志窗口样本。
4.  将这些数字序列解码回人类可读的日志模板。
5.  打印结果，以便分析每个类别所代表的具体日志模式。
"""

import os
import sys
import logging
import torch
import numpy as np
import json

# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from enhanced_deeplog.utils.data_utils import DataUtils
from enhanced_deeplog.src.data_preprocessing import DataPreprocessor

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_clusters(config_path: str, num_samples_per_class: int = 5):
    """
    主分析函数，加载数据并打印每个类别的日志样本。

    Args:
        config_path (str): 配置文件的路径。
        num_samples_per_class (int): 每个类别要抽取的样本数量。
    """
    print("🚀 Enhanced DeepLog - 聚类结果分析 🚀")
    print("="*60)

    # 1. 加载配置
    if not os.path.exists(config_path):
        logging.error(f"配置文件未找到: {config_path}")
        return
    config = DataUtils.load_config(config_path)

    # 2. 加载数据
    # 加载最终标签
    labels_path = os.path.join(config['data_config']['processed_dir'], 'final_labels.npy')
    if not os.path.exists(labels_path):
        logging.error("最终标签文件未找到。请先运行 03_anomaly_clustering.py。")
        return
    
    logging.info("加载聚类标签...")
    labels = DataUtils.load_numpy_data(labels_path)

    # 加载原始序列数据并进行处理（与训练脚本保持一致）
    logging.info("加载并处理原始序列数据...")
    raw_data_path = os.path.join(config['data_config']['raw_data_dir'], config['data_config']['data_file'])
    raw_sequences = DataUtils.load_sequences(raw_data_path)

    # 加载基础模型以获取 event_mapping
    base_model_path = config['model_config']['base_model_path']
    if not os.path.exists(base_model_path):
        logging.error(f"基础模型文件未找到: {base_model_path}")
        return
    checkpoint = torch.load(base_model_path)
    event_mapping = checkpoint['config']['event_mapping']
    
    # 创建反向映射 (ID -> Template)
    id_to_event_mapping = {v: k for k, v in event_mapping.items()}
    # 添加对 <PAD> 和 <UNK> 的处理
    id_to_event_mapping[0] = "<PAD>"
    id_to_event_mapping.setdefault(1, "<UNK>") # 假定 1 是 <UNK>

    # 使用与特征提取相同的逻辑来生成窗口
    preprocessor = DataPreprocessor(config)
    preprocessor.event_mapping = event_mapping
    
    all_windows = []
    for seq in raw_sequences:
        windows, _ = preprocessor.split_and_pad_sequence(seq, window_size=config['data_config']['window_size'])
        if windows.shape[0] > 0:
            all_windows.append(windows)
    
    sequences_padded = np.vstack(all_windows)

    # 确保数据对齐
    if len(sequences_padded) != len(labels):
        logging.error(f"数据不对齐！序列: {len(sequences_padded)}, 标签: {len(labels)}")
        return

    # 3. 分析每个类别
    unique_labels = sorted(np.unique(labels))
    logging.info(f"发现类别: {unique_labels}")

    for label in unique_labels:
        print("\n" + "="*60)
        print(f"🔬 分析 Class_{label}...")
        print("="*60)

        # 找到属于当前类别的所有序列的索引
        indices = np.where(labels == label)[0]
        
        # 随机选择几个样本
        num_to_sample = min(num_samples_per_class, len(indices))
        if num_to_sample == 0:
            print(f"Class_{label} 中没有样本。")
            continue
        
        sample_indices = np.random.choice(indices, num_to_sample, replace=False)
        
        print(f"从 {len(indices)} 个总样本中随机抽取 {num_to_sample} 个进行展示:\n")

        for i, idx in enumerate(sample_indices):
            sequence = sequences_padded[idx]
            decoded_sequence = [id_to_event_mapping.get(event_id, f"<ID_{event_id}?>") for event_id in sequence if event_id != 0] # 过滤掉 PAD
            
            print(f"--- 样本 {i+1} (原始索引: {idx}) ---")
            for step, template in enumerate(decoded_sequence):
                print(f"  Step {step}: {template}")
            print("-"*(len(str(i+1))+13))


if __name__ == "__main__":
    analyze_clusters(config_path='enhanced_deeplog/config/model_config.json')
