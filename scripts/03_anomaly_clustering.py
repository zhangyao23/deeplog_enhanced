#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
脚本 03: 异常聚类

此脚本执行 Enhanced DeepLog 模型的第三阶段：
1. 加载从阶段二提取的特征。
2. 仅对异常样本的特征进行聚类，以发现异常模式的内在结构。
3. 使用 K-means 或 DBSCAN 等算法将异常划分为不同的簇。
4. 评估聚类质量，并将聚类结果（模型和标签）保存到磁盘，以供阶段四使用。
"""

import os
import sys
import logging
import numpy as np
import joblib

# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from enhanced_deeplog.src.clustering import AnomalyClusterer
from enhanced_deeplog.utils.data_utils import DataUtils
from enhanced_deeplog.utils.evaluation import ModelEvaluator

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    主函数，执行异常聚类流程。
    """
    print("🚀 Enhanced DeepLog - 阶段三：异常模式聚类 🚀")
    print("="*60)

    # 1. 加载配置
    config_path = 'enhanced_deeplog/config/model_config.json'
    if not os.path.exists(config_path):
        logging.error(f"配置文件未找到: {config_path}")
        return
    config = DataUtils.load_config(config_path)
    
    # 确保目录存在
    model_dir = os.path.dirname(config['model_config'].get('clustering_model_path', 'enhanced_deeplog/models/clustering_model.pkl'))
    DataUtils.ensure_directory(model_dir)

    # 2. 加载处理好的特征和标签
    features_path = os.path.join(config['data_config']['processed_dir'], 'features.npy')
    labels_path = os.path.join(config['data_config']['processed_dir'], 'labels.npy')

    if not os.path.exists(features_path) or not os.path.exists(labels_path):
        logging.error("特征或标签文件未找到。请先运行 02_extract_features.py。")
        return
    
    logging.info("加载特征和标签...")
    features = DataUtils.load_numpy_data(features_path)
    labels = DataUtils.load_numpy_data(labels_path)

    # 3. 筛选异常特征
    # 标签 0 通常是正常，大于 0 是不同类型的异常
    anomaly_indices = np.where(labels > 0)[0]
    anomaly_features = features[anomaly_indices]
    original_anomaly_labels = labels[anomaly_indices]

    if len(anomaly_features) == 0:
        logging.warning("数据中未发现异常特征，无法进行聚类。")
        return

    logging.info(f"找到了 {len(anomaly_features)} 个异常特征样本用于聚类。")

    # 4. 初始化并执行聚类
    clusterer = AnomalyClusterer(config['clustering_config'])
    logging.info(f"使用 {config['clustering_config']['algorithm']} 算法进行聚类...")
    
    cluster_labels = clusterer.fit_predict(anomaly_features)
    
    # K-means 聚类标签从0开始，而我们的异常标签从1开始。
    # 为了保持一致性，我们将聚类标签加1，使其从1开始编号。
    cluster_labels += 1

    logging.info("聚类完成。")

    # 5. 评估聚类结果
    evaluator = ModelEvaluator(config)
    clustering_report = evaluator.evaluate_clustering(anomaly_features, original_anomaly_labels, cluster_labels)
    
    print("\n📊 聚类评估报告 📊")
    print("="*60)
    for metric, value in clustering_report.items():
        print(f"{metric}: {value:.4f}")
    print("="*60)
    
    # 可视化 (可选，但推荐)
    try:
        plot_path = os.path.join(os.path.dirname(config['data_config']['processed_dir']), 'diagnostics', 'clustering_visualization.png')
        DataUtils.ensure_directory(os.path.dirname(plot_path))
        evaluator.plot_clusters(anomaly_features, cluster_labels, title="异常特征 T-SNE 可视化", save_path=plot_path)
        logging.info(f"聚类可视化图已保存到: {plot_path}")
    except Exception as e:
        logging.warning(f"无法生成聚类可视化图: {e}")


    # 6. 保存聚类模型和新的标签
    # 创建一个新的标签数组，其中正常样本标签为0，异常样本标签为聚类结果
    final_labels = np.zeros_like(labels)
    final_labels[anomaly_indices] = cluster_labels
    
    # 保存聚类模型
    clustering_model_path = config['model_config'].get('clustering_model_path', 'enhanced_deeplog/models/clustering_model.pkl')
    joblib.dump(clusterer.model, clustering_model_path)
    logging.info(f"✅ 聚类模型已保存到: {clustering_model_path}")

    # 保存最终的标签
    final_labels_path = os.path.join(config['data_config']['processed_dir'], 'final_labels.npy')
    DataUtils.save_numpy_data(final_labels, final_labels_path)
    logging.info(f"✅ 最终标签 (聚类后) 已保存到: {final_labels_path}")

    logging.info("🎉 阶段三完成！🎉")


if __name__ == "__main__":
    main() 