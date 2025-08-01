# Enhanced DeepLog: Multi-Class Anomaly Classification

## 1. 项目概述

本项目是对经典日志异常检测模型DeepLog的增强实现，旨在不仅能识别异常，还能对异常进行**多分类**。我们采用了一种新颖的、分阶段的半监督方法，该方法更符合DeepLog的核心思想，即**异常是“对正常模式的偏离”**。

### 核心改进
- **分阶段训练**: 将任务分解为：1) 训练“正常模式专家”；2) 提取偏差特征；3) 聚类发现异常类别；4) 训练最终分类器。
- **避免过拟合**: 引入验证集和早停机制，确保基础模型具有最佳的泛化能力。
- **数据驱动的异常定义**: 使用无监督聚类来自动发现和定义异常类别，而不是依赖于预先设定的人工规则。
- **模块化代码结构**: 将数据处理、模型定义、训练、评估等功能解耦，提高了代码的可维护性和可扩展性。

## 2. 技术架构

我们的Pipeline分为四个主要阶段，每个阶段由一个独立的脚本负责：

1.  **`scripts/01_train_base_model.py`**: **训练“正常模式专家”**
    *   **输入**: 正常日志序列。
    *   **过程**: 训练一个专门学习正常行为模式的LSTM模型。
    *   **输出**: `models/base_model.pth`，一个能够精确预测正常序列后续事件的模型。

2.  **`scripts/02_extract_features.py`**: **提取偏差特征**
    *   **输入**: `base_model.pth` 和所有日志序列（正常+异常）。
    *   **过程**: 利用基础模型对所有序列进行预测，量化其“预测偏差”，并结合统计和模式特征，为每个序列生成一个高维特征向量。
    *   **输出**: `data/processed/features.npy` 和 `labels.npy`。

3.  **`scripts/03_cluster_anomalies.py`**: **自动发现异常类别**
    *   **输入**: `features.npy`（仅异常部分的特征）。
    *   **过程**: 使用K-Means等聚类算法对异常特征进行分组，自动划分出不同的异常模式。
    *   **输出**: `models/cluster_model.pkl` 和 `data/processed/cluster_labels.npy`。

4.  **`scripts/04_train_multiclass_model.py`**: **训练多分类器**
    *   **输入**: 原始序列数据 和 `cluster_labels.npy`。
    *   **过程**: 训练一个端到端的 `EnhancedDeepLogModel`，该模型以原始序列为输入，直接输出具体的异常类别。
    *   **输出**: `models/multiclass_model.pth`，最终的应用模型。

## 3. 项目结构

```
enhanced_deeplog/
├── config/
│   └── model_config.json       # 主配置文件
├── data/
│   ├── raw/                    # 原始数据
│   └── processed/              # 处理后的特征数据
├── diagnostics/
│   └── evaluate_base_model.py  # 用于评估基础模型的诊断脚本
├── models/                     # 存放训练好的模型
├── results/                    # 存放图表、报告等结果
├── scripts/
│   ├── 00_generate_synthetic_data.py
│   ├── 01_train_base_model.py
│   ├── 02_extract_features.py
│   ├── 03_cluster_anomalies.py
│   └── 04_train_multiclass_model.py
├── src/
│   ├── __init__.py
│   ├── base_model.py           # SimpleLSTM基础模型
│   ├── clustering.py
│   ├── data_preprocessing.py
│   ├── enhanced_model.py
│   ├── feature_engineering.py
│   ├── prediction.py
│   └── training.py
├── utils/
│   ├── __init__.py
│   ├── data_utils.py
│   └── evaluation.py
├── run_pipeline.py             # 串联所有脚本的主运行文件 (待实现)
└── README.md
```

## 4. 快速开始

1.  **安装依赖**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **生成数据**:
    ```bash
    python3 scripts/00_generate_synthetic_data.py
    ```

3.  **运行Pipeline (按顺序执行)**:
    ```bash
    python3 scripts/01_train_base_model.py
    python3 scripts/02_extract_features.py
    python3 scripts/03_cluster_anomalies.py
    python3 scripts/04_train_multiclass_model.py
    ``` 