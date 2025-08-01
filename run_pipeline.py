#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced DeepLog 主运行脚本
"""

import sys
import os
import torch
import numpy as np
import json
import logging
from typing import Dict, Any

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.clustering import AnomalyClusterer
from src.enhanced_model import EnhancedDeepLogModel
from src.training import ModelTrainer
from src.prediction import AnomalyPredictor
from utils.data_utils import DataUtils
from utils.evaluation import ModelEvaluator

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('enhanced_deeplog.log'),
            logging.StreamHandler()
        ]
    )

def load_config(config_path: str = 'config/model_config.json') -> Dict[str, Any]:
    """加载配置"""
    return DataUtils.load_config(config_path)

def main():
    """主函数"""
    print("🚀 Enhanced DeepLog: 基于改进架构的多分类异常检测")
    print("=" * 60)
    
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    try:
        # 1. 加载配置
        logger.info("1️⃣ 加载配置...")
        config = load_config()
        
        # 2. 数据预处理
        logger.info("2️⃣ 数据预处理...")
        preprocessor = DataPreprocessor(config)
        
        # 加载原始数据（这里需要根据实际数据路径调整）
        data_file = '../data/hdfs_train'  # 调整路径
        if not os.path.exists(data_file):
            logger.error(f"数据文件不存在: {data_file}")
            return
        
        # 预处理数据
        inputs, targets, stats = preprocessor.preprocess(data_file)
        
        # 保存预处理信息
        preprocessor.save_preprocessing_info('data/processed/preprocessing_info.json')
        
        # 3. 特征工程
        logger.info("3️⃣ 特征工程...")
        feature_engineer = FeatureEngineer(config)
        
        # 提取特征（这里需要预测误差，暂时使用模拟数据）
        prediction_errors = np.random.random(len(inputs))  # 模拟预测误差
        features = feature_engineer.extract_all_features(inputs, prediction_errors)
        
        # 保存特征
        DataUtils.save_numpy_data(features, 'data/features/extracted_features.npy')
        
        # 4. 聚类分析
        logger.info("4️⃣ 聚类分析...")
        clusterer = AnomalyClusterer(config)
        cluster_labels = clusterer.fit(features)
        
        # 获取聚类信息
        cluster_info = clusterer.get_cluster_info()
        logger.info(f"聚类结果: {cluster_info}")
        
        # 5. 创建模型
        logger.info("5️⃣ 创建模型...")
        num_event_types = config['data_config']['num_event_types']
        num_anomaly_types = config['clustering_config']['n_clusters']
        
        model = EnhancedDeepLogModel(config, num_event_types, num_anomaly_types)
        model_info = model.get_model_info()
        logger.info(f"模型信息: {model_info}")
        
        # 6. 准备训练数据
        logger.info("6️⃣ 准备训练数据...")
        trainer = ModelTrainer(config, model)
        
        # 使用聚类标签作为异常分类目标
        train_loader, val_loader = trainer.prepare_data(
            inputs, targets, cluster_labels
        )
        
        # 7. 训练模型
        logger.info("7️⃣ 训练模型...")
        training_results = trainer.train(train_loader, val_loader, device)
        
        # 保存训练历史
        trainer.save_training_history('results/training_history.json')
        
        # 8. 模型评估
        logger.info("8️⃣ 模型评估...")
        evaluator = ModelEvaluator()
        
        # 在验证集上进行预测
        predictor = AnomalyPredictor(model, config)
        predictions = predictor.predict(inputs, device)
        
        # 评估分类性能
        classification_metrics = evaluator.evaluate_classification(
            cluster_labels, predictions['anomaly_predictions']
        )
        
        # 打印评估结果
        evaluator.print_evaluation_summary(classification_metrics)
        
        # 保存评估结果
        evaluation_results = {
            'classification_metrics': classification_metrics,
            'cluster_info': cluster_info,
            'model_info': model_info,
            'training_results': training_results
        }
        evaluator.save_evaluation_results(evaluation_results, 'results/evaluation_results.json')
        
        # 9. 保存模型
        logger.info("9️⃣ 保存模型...")
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'model_info': model_info,
            'cluster_info': cluster_info
        }, 'models/enhanced_deeplog_model.pth')
        
        logger.info("✅ Enhanced DeepLog 训练完成！")
        logger.info("📊 结果文件:")
        logger.info("  - 模型: models/enhanced_deeplog_model.pth")
        logger.info("  - 训练历史: results/training_history.json")
        logger.info("  - 评估结果: results/evaluation_results.json")
        logger.info("  - 预处理信息: data/processed/preprocessing_info.json")
        logger.info("  - 特征: data/features/extracted_features.npy")
        
    except Exception as e:
        logger.error(f"❌ 训练过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 