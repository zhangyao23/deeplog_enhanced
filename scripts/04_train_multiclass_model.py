#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
脚本 04: 训练多分类模型

此脚本执行 Enhanced DeepLog 模型的第四个也是最后一个阶段：
1. 加载阶段二生成的特征矩阵和阶段三生成的最终聚类标签。
2. 加载原始的、经过填充的序列数据，因为增强模型需要它作为输入。
3. 将数据（序列、特征、标签）分割为训练集、验证集和测试集。
4. 定义并训练 `EnhancedDeepLogModel`，这是一个多任务模型，同时进行序列预测和异常分类。
5. 在训练过程中使用早停和学习率调度等策略来优化性能。
6. 在测试集上评估最终模型的性能，并保存分类报告和混淆矩阵。
7. 保存训练好的多分类模型以供将来预测使用。
"""

import os
import sys
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from enhanced_deeplog.src.enhanced_model import EnhancedDeepLogModel
from enhanced_deeplog.src.training import ModelTrainer
from enhanced_deeplog.utils.data_utils import DataUtils
from enhanced_deeplog.utils.evaluation import ModelEvaluator
from enhanced_deeplog.src.data_preprocessing import DataPreprocessor # 需要用它来加载原始序列

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    """
    主函数，执行多分类模型的训练、评估和保存流程。
    """
    print("🚀 Enhanced DeepLog - 阶段四：训练多分类注意力模型 🚀")
    print("="*60)

    # 1. 加载配置
    config_path = 'enhanced_deeplog/config/model_config.json'
    if not os.path.exists(config_path):
        logging.error(f"配置文件未找到: {config_path}")
        return
    config = DataUtils.load_config(config_path)
    
    # 2. 加载数据
    # 加载特征和最终标签
    features_path = os.path.join(config['data_config']['processed_dir'], 'features.npy')
    labels_path = os.path.join(config['data_config']['processed_dir'], 'final_labels.npy')
    
    if not os.path.exists(features_path) or not os.path.exists(labels_path):
        logging.error("特征或最终标签文件未找到。请先运行 02_extract_features.py 和 03_anomaly_clustering.py。")
        return

    logging.info("加载特征和聚类标签...")
    features = DataUtils.load_numpy_data(features_path)
    labels = DataUtils.load_numpy_data(labels_path)

    # 加载原始序列数据并进行处理
    logging.info("加载并处理原始序列数据以用于模型输入...")
    raw_data_path = os.path.join(config['data_config']['raw_data_dir'], config['data_config']['data_file'])
    raw_sequences = DataUtils.load_sequences(raw_data_path)
    
    # 加载基础模型以获取 event_mapping
    base_model_path = config['model_config']['base_model_path']
    if not os.path.exists(base_model_path):
        logging.error(f"基础模型文件未找到: {base_model_path}")
        return
    checkpoint = torch.load(base_model_path)
    event_mapping = checkpoint['config']['event_mapping']
    
    # 使用与特征提取相同的逻辑来生成窗口
    preprocessor = DataPreprocessor(config)
    preprocessor.event_mapping = event_mapping
    
    all_windows = []
    for seq in raw_sequences:
        windows, _ = preprocessor.split_and_pad_sequence(seq, window_size=config['data_config']['window_size'])
        if windows.shape[0] > 0:
            all_windows.append(windows)
    
    sequences_padded = np.vstack(all_windows)
    
    # 确保所有数据源的样本数量一致
    if not (len(sequences_padded) == len(features) == len(labels)):
        logging.error(f"数据不对齐！序列: {len(sequences_padded)}, 特征: {len(features)}, 标签: {len(labels)}")
        return

    # 3. 数据集分割
    logging.info(f"将数据集分割为训练集、验证集和测试集...")
    
    # 首先，分割出测试集 (e.g., 20%)
    indices = np.arange(len(sequences_padded))
    X_train_val_idx, X_test_idx = train_test_split(
        indices, 
        test_size=0.2, 
        random_state=42, 
        stratify=labels
    )
    
    # 然后，从剩余数据中分割出训练集和验证集 (e.g., 25% of the rest, which is 20% of original)
    y_train_val = labels[X_train_val_idx]
    X_train_idx, X_val_idx = train_test_split(
        X_train_val_idx,
        test_size=0.25, 
        random_state=42,
        stratify=y_train_val
    )

    # 创建数据集
    X_train_seq, X_val_seq, X_test_seq = sequences_padded[X_train_idx], sequences_padded[X_val_idx], sequences_padded[X_test_idx]
    X_train_feat, X_val_feat, X_test_feat = features[X_train_idx], features[X_val_idx], features[X_test_idx]
    y_train, y_val, y_test = labels[X_train_idx], labels[X_val_idx], labels[X_test_idx]
    
    logging.info(f"训练集大小: {len(X_train_seq)}")
    logging.info(f"验证集大小: {len(X_val_seq)}")
    logging.info(f"测试集大小: {len(X_test_seq)}")

    # 4. 初始化模型
    num_keys = len(event_mapping)
    num_classes = len(np.unique(labels))

    model = EnhancedDeepLogModel(
        config=config,
        num_event_types=num_keys,
        num_anomaly_types=num_classes,
        feature_dim=features.shape[1]
    )
    
    # 5. 训练模型
    trainer = ModelTrainer(config, model)
    
    model_save_path = 'enhanced_deeplog/models/multiclass_model.pth'
    
    logging.info("🚀 开始训练多分类模型...")
    trainer.train(
        (X_train_seq, X_train_feat, y_train),
        (X_val_seq, X_val_feat, y_val),
        model_save_path
    )
    logging.info(f"✅ 模型训练完成，最佳模型已保存到: {model_save_path}")
    
    # 6. 评估模型
    logging.info("在测试集上评估最终模型...")
    # trainer内部已经加载了最佳模型
    evaluator = ModelEvaluator(config)
    
    # 准备测试数据加载器
    test_dataset = TensorDataset(
        torch.from_numpy(X_test_seq).float(), 
        torch.from_numpy(X_test_feat).float(),
        torch.from_numpy(y_test).long()
    )
    test_loader = DataLoader(test_dataset, batch_size=config['training_config']['batch_size'], shuffle=False)

    # 获取预测结果
    y_true, y_pred_indices, y_pred_probs = trainer.predict(test_loader)
    
    # 生成和打印分类报告
    class_names = [f"Class_{i}" for i in range(num_classes)]
    report_str = evaluator.generate_classification_report(y_true, y_pred_indices, class_names)
    print("\n📊 分类性能报告 (测试集) 📊")
    print("="*60)
    print(report_str)
    print("="*60)
    
    # 保存报告
    report_path = 'enhanced_deeplog/diagnostics/classification_report.txt'
    DataUtils.ensure_directory(os.path.dirname(report_path))
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_str)
    logging.info(f"分类报告已保存到: {report_path}")

    # 绘制和保存混淆矩阵
    cm_path = 'enhanced_deeplog/diagnostics/confusion_matrix.png'
    evaluator.plot_confusion_matrix(y_true, y_pred_indices, class_names=class_names, save_path=cm_path)
    logging.info(f"混淆矩阵图已保存到: {cm_path}")
    
    logging.info("🎉 阶段四完成！整个流程结束。🎉")

if __name__ == "__main__":
    main() 