# Enhanced DeepLog: Multi-Class Anomaly Classification

## 🚀 快速导航

| 用户类型 | 推荐阅读顺序 | 主要文档 |
|---------|-------------|----------|
| **新手用户** | 快速开始 → 项目原理 → 技术概览 | [快速开始](guide/quick_start.md) |
| **研究人员** | 技术论文 → 项目原理 → 召回率策略 | [技术论文](guide/technical_paper.md) |
| **工程师** | 快速开始 → API参考 → 滑动窗口算法 | [API参考](guide/api_reference.md) |
| **运维人员** | 项目原理 → 快速开始 → 文本处理指南 | [项目原理](guide/project_principles.md) |
| **算法研究者** | 滑动窗口算法 → 召回率策略 → 技术论文 | [滑动窗口算法](guide/sliding_window_explanation.md) |

## 1. 项目概述

本项目是对经典日志异常检测模型DeepLog的增强实现，旨在不仅能识别异常，还能对异常进行**多分类**。我们采用了一种新颖的、分阶段的半监督方法，该方法更符合DeepLog的核心思想，即**异常是"对正常模式的偏离"**。

### 核心改进
- **分阶段训练**: 将任务分解为：1) 训练"正常模式专家"；2) 提取偏差特征；3) 聚类发现异常类别；4) 训练最终分类器。
- **避免过拟合**: 引入验证集和早停机制，确保基础模型具有最佳的泛化能力。
- **数据驱动的异常定义**: 使用无监督聚类来自动发现和定义异常类别，而不是依赖于预先设定的人工规则。
- **模块化代码结构**: 将数据处理、模型定义、训练、评估等功能解耦，提高了代码的可维护性和可扩展性。

### 📖 详细原理说明
- **技术论文**: [`guide/technical_paper.md`](guide/technical_paper.md) - 完整的学术论文，包含数学公式和实验分析
- **项目原理**: [`guide/project_principles.md`](guide/project_principles.md) - 详细的技术原理和工作流程说明

## 📚 完整文档体系

本项目提供了完整的文档体系，帮助您从不同角度理解和使用增强DeepLog：

### 🎯 核心文档
- **[技术论文](guide/technical_paper.md)** - 完整的学术论文，包含数学公式、实验设计和结果分析
- **[项目原理](guide/project_principles.md)** - 详细的技术原理、工作流程和架构说明
- **[技术概览](guide/technical_overview.md)** - 技术架构和核心概念的全面介绍

### 🚀 使用指南
- **[快速开始](guide/quick_start.md)** - 5分钟快速上手指南
- **[API参考](guide/api_reference.md)** - 完整的API文档和代码示例
- **[文本处理指南](guide/text_processing_guide.md)** - 日志文本处理的最佳实践

### 🔧 算法详解
- **[滑动窗口算法](guide/sliding_window_explanation.md)** - 滑动窗口分割算法的详细解释和实现
- **[召回率提升策略](guide/recall_improvement_strategies.md)** - 提高模型召回率的完整策略和代码实现

### 📋 文档管理
- **[文档总结](guide/documentation_summary.md)** - 所有文档的索引和导航指南

### 📊 文档统计
- **总文档数**: 9个专业文档
- **总字数**: 约100,000字
- **代码示例**: 200+个代码片段
- **数学公式**: 20+个核心公式
- **图表**: 10+个架构图和流程图

## 2. 技术架构

我们的Pipeline分为四个主要阶段，每个阶段由一个独立的脚本负责：

1.  **`scripts/01_train_base_model.py`**: **训练"正常模式专家"**
    *   **输入**: 正常日志序列。
    *   **过程**: 训练一个专门学习正常行为模式的LSTM模型。
    *   **输出**: `models/base_model.pth`，一个能够精确预测正常序列后续事件的模型。

2.  **`scripts/02_extract_features.py`**: **提取偏差特征**
    *   **输入**: `base_model.pth` 和所有日志序列（正常+异常）。
    *   **过程**: 利用基础模型对所有序列进行预测，量化其"预测偏差"，并结合统计和模式特征，为每个序列生成一个高维特征向量。
    *   **输出**: `data/processed/features.npy` 和 `labels.npy`。

3.  **`scripts/03_cluster_anomalies.py`**: **自动发现异常类别**
    *   **输入**: `features.npy`（仅异常部分的特征）。
    *   **过程**: 使用K-Means等聚类算法对异常特征进行分组，自动划分出不同的异常模式。
    *   **输出**: `models/cluster_model.pkl` 和 `data/processed/cluster_labels.npy`。

4.  **`scripts/04_train_multiclass_model.py`**: **训练多分类器**
    *   **输入**: 原始序列数据 和 `cluster_labels.npy`。
    *   **过程**: 训练一个端到端的 `EnhancedDeepLogModel`，该模型以原始序列为输入，直接输出具体的异常类别。
    *   **输出**: `models/multiclass_model.pth`，最终的应用模型。

## 3. 与原始DeepLog的关系

### 3.1 核心原理的完全保留

本增强模型**完全保留了原始DeepLog的核心原理**，主要体现在以下几个方面：

#### **阶段一：正常模式学习（完全沿用DeepLog）**
```python
# 在 scripts/01_train_base_model.py 中
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        # 完全按照DeepLog的架构设计
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_keys)  # 预测下一个事件类型
```

**关键点**：
- **训练数据**：只使用正常日志序列进行训练
- **学习目标**：学习正常操作下日志序列的语法和语义规则
- **预测机制**：给定序列前缀，预测下一个可能的日志事件
- **异常检测原理**：当预测失败时，认为检测到了异常

#### **阶段二：深度特征提取（DeepLog隐状态的再利用）**
```python
# 在 scripts/02_extract_features.py 中
def extract_features_for_all_sequences(config, model, preprocessor, feature_engineer, data_path, label_path):
    # 使用训练好的DeepLog模型提取LSTM隐状态
    with torch.no_grad():
        for batch in data_loader:
            lstm_out, (hidden, cell) = model.lstm(batch.unsqueeze(-1).float())
            # 取最后一层的隐状态作为特征
            batch_features = hidden[-1].cpu().numpy()
```

**创新点**：
- **隐状态利用**：将DeepLog模型在处理异常序列时的LSTM隐状态作为"深度特征"
- **信息保留**：这些隐状态包含了LSTM对序列模式的"理解"和"困惑"
- **异常模式捕捉**：不同类型的异常会在隐状态中产生不同的"扰动模式"

### 3.2 DeepLog原理的扩展应用

#### **双头架构设计**
```python
# 在 enhanced_model.py 中
class EnhancedDeepLogModel(nn.Module):
    def __init__(self, config, num_event_types, num_anomaly_types, feature_dim):
        # 序列预测头（原始DeepLog功能）
        self.sequence_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, num_event_types)  # 预测下一个事件
        )
        
        # 异常分类头（新增功能）
        self.anomaly_head = nn.Sequential(
            nn.Linear(self.hidden_size + self.feature_dim, self.hidden_size // 2),
            nn.Linear(self.hidden_size // 2, num_anomaly_types)  # 分类异常类型
        )
```

**设计理念**：
- **输入1（原始序列）**：让模型像标准DeepLog一样学习序列的底层模式
- **输入2（特征向量）**：提供关于序列模式的先验信息，加速收敛

### 3.3 具体的技术实现对比

| 方面 | 原始DeepLog | 增强DeepLog | 利用关系 |
|------|-------------|-------------|----------|
| **训练数据** | 仅正常日志 | 仅正常日志（阶段一） | ✅ 完全保留 |
| **模型架构** | LSTM + 全连接 | LSTM + 双头输出 | ✅ 扩展保留 |
| **学习目标** | 预测下一个事件 | 预测下一个事件 + 异常分类 | ✅ 功能扩展 |
| **异常检测** | 预测失败即异常 | 预测失败 + 隐状态分析 | ✅ 原理增强 |
| **输出结果** | 二分类（正常/异常） | 多分类（正常/异常类型1/2/3...） | ✅ 能力提升 |

### 3.4 关键创新点的DeepLog基础

#### **隐状态特征提取**
```python
# 原始DeepLog只关注预测结果
if next_event not in top_k_predictions:
    anomaly_detected = True

# 增强DeepLog还利用隐状态
lstm_out, (hidden, cell) = model.lstm(sequence)
deep_features = hidden[-1]  # 提取隐状态作为特征
```

#### **多任务学习**
```python
def forward(self, x_seq, x_feat):
    # 任务1：序列预测（DeepLog原始功能）
    sequence_pred = self.sequence_head(sequence_features)
    
    # 任务2：异常分类（新增功能）
    combined_features = torch.cat([sequence_features, x_feat], dim=1)
    anomaly_pred = self.anomaly_head(combined_features)
    
    return sequence_pred, anomaly_pred
```

### 3.5 总结

这个增强模型**不是替代**DeepLog，而是**在DeepLog基础上的扩展**：

1. **完全保留**了DeepLog的核心原理（正常模式学习、序列预测、异常检测机制）
2. **创新利用**了DeepLog的中间产物（LSTM隐状态）作为新的特征
3. **功能扩展**了DeepLog的输出能力（从二分类到多分类）
4. **架构增强**了DeepLog的模型设计（双头输出、注意力机制）

这种设计确保了增强模型既保持了DeepLog的有效性，又实现了更细粒度的异常分类能力。

## 4. 项目结构

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

## 5. 快速开始

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

## 📖 文档使用指南

### 🎯 根据您的需求选择合适的文档

#### 如果您是**研究人员**：
- 从 **[技术论文](guide/technical_paper.md)** 开始，了解理论基础和实验设计
- 查看 **[项目原理](guide/project_principles.md)** 了解技术实现细节
- 参考 **[召回率提升策略](guide/recall_improvement_strategies.md)** 进行模型优化

#### 如果您是**工程师**：
- 从 **[快速开始](guide/quick_start.md)** 开始，快速上手项目
- 查看 **[API参考](guide/api_reference.md)** 了解具体接口
- 参考 **[技术概览](guide/technical_overview.md)** 理解整体架构

#### 如果您是**运维人员**：
- 从 **[项目原理](guide/project_principles.md)** 开始，了解系统工作原理
- 查看 **[快速开始](guide/quick_start.md)** 学习如何部署
- 参考 **[文本处理指南](guide/text_processing_guide.md)** 了解日志处理

#### 如果您想**深入算法**：
- 查看 **[滑动窗口算法](guide/sliding_window_explanation.md)** 了解核心算法
- 参考 **[召回率提升策略](guide/recall_improvement_strategies.md)** 学习优化方法
- 查看 **[技术论文](guide/technical_paper.md)** 了解数学原理

### 🔍 文档导航建议

1. **首次接触项目**：`快速开始` → `项目原理` → `技术概览`
2. **学术研究**：`技术论文` → `项目原理` → `召回率提升策略`
3. **工程实现**：`快速开始` → `API参考` → `滑动窗口算法`
4. **性能优化**：`召回率提升策略` → `技术论文` → `项目原理`

### 📝 文档反馈

如果您发现文档中的问题或有改进建议，欢迎：
- 提交Issue到项目仓库
- 发送邮件到项目维护者
- 参与文档贡献

### 🔗 相关资源

- **原始DeepLog论文**：[DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning](https://dl.acm.org/doi/10.1145/3132747.3132785)
- **HDFS数据集**：[HDFS Dataset](https://github.com/logpai/loghub)
- **相关工具**：[LogPai](https://github.com/logpai/loglizer) 