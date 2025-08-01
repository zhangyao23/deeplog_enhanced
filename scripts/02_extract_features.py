#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段二：提取和量化“偏差模式”
"""

import sys
import os
import torch
import numpy as np
import json
import logging
from tqdm import tqdm
from scipy import stats as scipy_stats
import torch.nn as nn

# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# 导入本地模块
from enhanced_deeplog.src.data_preprocessing import DataPreprocessor
from enhanced_deeplog.src.feature_engineering import FeatureEngineer
from enhanced_deeplog.src.base_model import SimpleLSTM
from enhanced_deeplog.utils.data_utils import DataUtils

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_config():
    """
    定义脚本所需的配置。
    """
    config = {
        "data_config": {
            "raw_data_dir": "enhanced_deeplog/data/raw",
            "processed_dir": "enhanced_deeplog/data/processed",
            "data_file": "synthetic_hdfs_train.txt",
            "labels_file": "synthetic_hdfs_train_labels.txt",
            "padding_value": 0,
            "window_size": 10,
            "max_sequence_length": 128,
            "min_sequence_length": 2
        },
        "feature_config": {
            "use_statistical_features": True,
            "use_prediction_features": True,
            "use_pattern_features": True
        },
        "model_config": {
            "base_model_path": "enhanced_deeplog/models/base_model.pth"
        }
    }
    return config

def load_base_model(model_path, device):
    """
    加载训练好的SimpleLSTM基础模型。
    """
    logging.info(f"从 {model_path} 加载基础模型...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # 从checkpoint中恢复config，确保一致性
    config = checkpoint['config']
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
    return model, config, config.get('event_mapping', {})

def extract_features_for_all_sequences(config, model, preprocessor, feature_engineer, data_path, label_path):
    """为所有序列提取特征"""
    logging.info("为所有序列提取特征...")

    # 加载所有原始数据
    with open(data_path, 'r') as f:
        all_sequences = [list(map(int, line.strip().split())) for line in f]
    with open(label_path, 'r') as f:
        all_labels = [int(line.strip()) for line in f]

    final_features = []
    final_labels = []

    pbar = tqdm(zip(all_sequences, all_labels), total=len(all_sequences), desc="提取特征")
    for seq, label in pbar:
        if not seq: continue

        # 1. 对单个序列进行滑窗处理
        # 预处理器现在包含了全局的event_mapping
        mapped_seqs = preprocessor.apply_event_mapping([seq])
        inputs, targets = preprocessor.split_sequences(mapped_seqs, config['data_config']['window_size'])

        if len(inputs) == 0:
            continue

        # 2. 获取基础模型的预测误差
        inputs_tensor = torch.FloatTensor(inputs).to(device)
        targets_tensor = torch.LongTensor(targets).to(device)

        with torch.no_grad():
            seq_pred, _ = model(inputs_tensor)
            probs = torch.softmax(seq_pred, dim=1)
            target_probs = probs.gather(1, targets_tensor.unsqueeze(1)).squeeze()
            prediction_errors = (1 - target_probs).cpu().numpy()

        # 3. 聚合预测误差特征
        if prediction_errors.size == 0: continue
        
        # 处理可能的nan/inf值
        prediction_errors = prediction_errors[np.isfinite(prediction_errors)]
        if prediction_errors.size == 0: continue
        
        skewness = scipy_stats.skew(prediction_errors)
        kurtosis = scipy_stats.kurtosis(prediction_errors)

        pred_feats = np.array([
            np.mean(prediction_errors),
            np.std(prediction_errors),
            np.max(prediction_errors),
            np.min(prediction_errors),
            np.median(prediction_errors),
            skewness if np.isfinite(skewness) else 0,
            kurtosis if np.isfinite(kurtosis) else 0
        ])

        # 4. 提取统计和模式特征
        stat_feats = feature_engineer.extract_statistical_features([seq])[0]
        patt_feats = feature_engineer.extract_pattern_features([seq])[0]
        
        # 5. 合并所有特征
        combined_features = np.concatenate([pred_feats, stat_feats, patt_feats])
        
        final_features.append(combined_features)
        final_labels.append(label)

    return np.array(final_features), np.array(final_labels)

def main():
    """
    主函数，执行特征提取流程。
    """
    print("🚀 Enhanced DeepLog - 阶段二：提取和量化“偏差模式” 🚀")
    print("="*60)
    
    # 获取配置
    config = get_config()

    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")

    # 加载基础模型
    base_model_path = config['model_config']['base_model_path']
    if not os.path.exists(base_model_path):
        logging.error(f"基础模型文件未找到: {base_model_path}")
        logging.error("请先运行 01_train_base_model.py 脚本来训练和保存基础模型。")
        return
    
    model, model_config, event_mapping = load_base_model(base_model_path, device)
    model.eval()

    # 初始化数据预处理器和特征工程师
    preprocessor = DataPreprocessor(config)
    feature_engineer = FeatureEngineer(config, model, device, event_mapping)

    # 加载原始数据和标签
    raw_data_path = os.path.join(config['data_config']['raw_data_dir'], config['data_config']['data_file'])
    labels_path = os.path.join(config['data_config']['raw_data_dir'], config['data_config']['labels_file'])
    
    if not os.path.exists(raw_data_path) or not os.path.exists(labels_path):
        logging.error(f"数据文件或标签文件未找到。请确保 {raw_data_path} 和 {labels_path} 存在。")
        logging.error("如果文件不存在，请运行 00_generate_synthetic_data.py 来生成它们。")
        return

    sequences = DataUtils.load_sequences(raw_data_path)
    labels = DataUtils.load_labels(labels_path)

    # 提取特征
    all_features = []
    all_labels = []

    for i in tqdm(range(len(sequences)), desc="提取特征"):
        seq = sequences[i]
        label = labels[i]
        
        # 预处理单个序列（切片和填充）
        windows, _ = preprocessor.split_and_pad_sequence(seq, window_size=config['data_config']['window_size'])
        
        if windows.shape[0] == 0:
            continue
            
        # 提取特征
        features = feature_engineer.extract_all_features(windows)
        
        all_features.append(features)
        # 为每个窗口分配相同的序列标签
        all_labels.extend([label] * features.shape[0])

    if not all_features:
        logging.error("未能从数据中提取任何特征。请检查输入数据和配置。")
        return

    features = np.vstack(all_features)
    labels = np.array(all_labels)
    
    logging.info(f"特征提取完成。特征矩阵形状: {features.shape}, 标签数量: {len(labels)}")

    # 保存特征和标签
    processed_dir = config['data_config']['processed_dir']
    DataUtils.ensure_directory(processed_dir)
    features_path = os.path.join(processed_dir, 'features.npy')
    labels_path = os.path.join(processed_dir, 'labels.npy')
    
    DataUtils.save_numpy_data(features, features_path)
    DataUtils.save_numpy_data(labels, labels_path)
    
    logging.info(f"✅ 特征已保存到 {features_path}")
    logging.info(f"✅ 标签已保存到 {labels_path}")

if __name__ == "__main__":
    main() 