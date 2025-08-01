# Enhanced DeepLog API 参考文档

## 概述

本文档详细介绍了 Enhanced DeepLog 系统中各个模块的 API 接口，包括类定义、方法说明、参数描述和返回值。

## 核心模块 API

### 1. 数据预处理模块 (DataPreprocessor)

#### 类定义
```python
class DataPreprocessor:
    def __init__(self, config: Dict[str, Any])
```

#### 主要方法

##### load_data(file_path: str) -> List[List[int]]
**功能**: 加载原始数据文件

**参数**:
- `file_path`: 数据文件路径

**返回值**: 序列列表，每个序列为整数列表

**示例**:
```python
preprocessor = DataPreprocessor(config)
sequences = preprocessor.load_data("data/raw/logs.txt")
```

##### preprocess(file_path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]
**功能**: 完整的数据预处理流程

**参数**:
- `file_path`: 数据文件路径

**返回值**: 
- 补齐后的输入序列
- 目标序列
- 统计信息字典

##### save_preprocessing_info(file_path: str)
**功能**: 保存预处理信息

**参数**:
- `file_path`: 保存路径

### 2. 特征工程模块 (FeatureEngineer)

#### 类定义
```python
class FeatureEngineer:
    def __init__(self, config: Dict[str, Any])
```

#### 主要方法

##### extract_statistical_features(sequences: np.ndarray) -> np.ndarray
**功能**: 提取统计特征

**参数**:
- `sequences`: 序列数组

**返回值**: 统计特征数组 (20维)

##### extract_prediction_features(sequences: np.ndarray, prediction_errors: np.ndarray) -> np.ndarray
**功能**: 提取预测相关特征

**参数**:
- `sequences`: 序列数组
- `prediction_errors`: 预测误差数组

**返回值**: 预测特征数组 (15维)

##### extract_pattern_features(sequences: np.ndarray) -> np.ndarray
**功能**: 提取模式特征

**参数**:
- `sequences`: 序列数组

**返回值**: 模式特征数组 (25维)

##### extract_all_features(sequences: np.ndarray, prediction_errors: np.ndarray = None) -> np.ndarray
**功能**: 提取所有特征

**参数**:
- `sequences`: 序列数组
- `prediction_errors`: 预测误差数组（可选）

**返回值**: 组合后的特征数组 (60维)

### 3. 聚类分析模块 (AnomalyClusterer)

#### 类定义
```python
class AnomalyClusterer:
    def __init__(self, config: Dict[str, Any])
```

#### 主要方法

##### fit(features: np.ndarray) -> np.ndarray
**功能**: 执行聚类分析

**参数**:
- `features`: 特征数组

**返回值**: 聚类标签数组

##### get_cluster_info() -> Dict[str, Any]
**功能**: 获取聚类信息

**返回值**: 包含聚类统计信息的字典

### 4. 改进LSTM模型 (EnhancedDeepLogModel)

#### 类定义
```python
class EnhancedDeepLogModel(nn.Module):
    def __init__(self, config: Dict[str, Any], num_event_types: int, num_anomaly_types: int)
```

#### 主要方法

##### forward(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
**功能**: 前向传播

**参数**:
- `x`: 输入张量

**返回值**: 
- 序列预测结果
- 异常分类结果

##### predict_sequence(x: torch.Tensor) -> torch.Tensor
**功能**: 仅进行序列预测

**参数**:
- `x`: 输入张量

**返回值**: 序列预测结果

##### predict_anomaly(x: torch.Tensor) -> torch.Tensor
**功能**: 仅进行异常分类

**参数**:
- `x`: 输入张量

**返回值**: 异常分类结果

##### get_model_info() -> Dict[str, Any]
**功能**: 获取模型信息

**返回值**: 模型信息字典

### 5. 训练模块 (ModelTrainer)

#### 类定义
```python
class ModelTrainer:
    def __init__(self, config: Dict[str, Any], model: nn.Module)
```

#### 主要方法

##### prepare_data(inputs: np.ndarray, sequence_targets: np.ndarray, anomaly_targets: np.ndarray) -> Tuple[DataLoader, DataLoader]
**功能**: 准备训练和验证数据

**参数**:
- `inputs`: 输入数据
- `sequence_targets`: 序列预测目标
- `anomaly_targets`: 异常分类目标

**返回值**: 训练和验证数据加载器

##### train(train_loader: DataLoader, val_loader: DataLoader, device: torch.device) -> Dict[str, Any]
**功能**: 训练模型

**参数**:
- `train_loader`: 训练数据加载器
- `val_loader`: 验证数据加载器
- `device`: 计算设备

**返回值**: 训练结果字典

##### save_training_history(file_path: str)
**功能**: 保存训练历史

**参数**:
- `file_path`: 保存路径

### 6. 预测模块 (AnomalyPredictor)

#### 类定义
```python
class AnomalyPredictor:
    def __init__(self, model: nn.Module, config: Dict[str, Any])
```

#### 主要方法

##### predict(sequences: np.ndarray, device: torch.device) -> Dict[str, Any]
**功能**: 进行预测

**参数**:
- `sequences`: 输入序列
- `device`: 计算设备

**返回值**: 预测结果字典

##### predict_with_features(sequences: np.ndarray, features: np.ndarray, device: torch.device) -> Dict[str, Any]
**功能**: 结合特征进行预测

**参数**:
- `sequences`: 输入序列
- `features`: 特征数组
- `device`: 计算设备

**返回值**: 增强的预测结果

##### classify_anomalies(prediction_errors: np.ndarray, anomaly_predictions: np.ndarray) -> List[Dict[str, Any]]
**功能**: 分类异常

**参数**:
- `prediction_errors`: 预测误差
- `anomaly_predictions`: 异常预测

**返回值**: 异常分类结果列表

##### save_predictions(predictions: Dict[str, Any], file_path: str)
**功能**: 保存预测结果

**参数**:
- `predictions`: 预测结果
- `file_path`: 保存路径

## 工具模块 API

### 1. 数据处理工具 (DataUtils)

#### 静态方法

##### load_config(config_path: str) -> Dict[str, Any]
**功能**: 加载配置文件

**参数**:
- `config_path`: 配置文件路径

**返回值**: 配置字典

##### save_config(config: Dict[str, Any], config_path: str)
**功能**: 保存配置文件

**参数**:
- `config`: 配置字典
- `config_path`: 配置文件路径

##### split_data(data: np.ndarray, labels: np.ndarray, train_ratio: float = 0.8, random_state: int = 42) -> Tuple
**功能**: 分割数据为训练集和测试集

**参数**:
- `data`: 数据数组
- `labels`: 标签数组
- `train_ratio`: 训练集比例
- `random_state`: 随机种子

**返回值**: 训练数据、测试数据、训练标签、测试标签

##### normalize_features(features: np.ndarray, method: str = 'standard') -> np.ndarray
**功能**: 特征标准化

**参数**:
- `features`: 特征数组
- `method`: 标准化方法

**返回值**: 标准化后的特征

### 2. 模型评估工具 (ModelEvaluator)

#### 类定义
```python
class ModelEvaluator:
    def __init__(self)
```

#### 主要方法

##### evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict[str, float]
**功能**: 评估分类性能

**参数**:
- `y_true`: 真实标签
- `y_pred`: 预测标签
- `y_proba`: 预测概率（可选）

**返回值**: 评估指标字典

##### evaluate_clustering(features: np.ndarray, labels: np.ndarray) -> Dict[str, float]
**功能**: 评估聚类性能

**参数**:
- `features`: 特征数组
- `labels`: 聚类标签

**返回值**: 聚类评估指标

##### generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray, target_names: List[str] = None) -> str
**功能**: 生成分类报告

**参数**:
- `y_true`: 真实标签
- `y_pred`: 预测标签
- `target_names`: 目标类别名称

**返回值**: 分类报告字符串

##### plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, target_names: List[str] = None, save_path: str = None)
**功能**: 绘制混淆矩阵

**参数**:
- `y_true`: 真实标签
- `y_pred`: 预测标签
- `target_names`: 目标类别名称
- `save_path`: 保存路径

##### save_evaluation_results(results: Dict[str, Any], file_path: str)
**功能**: 保存评估结果

**参数**:
- `results`: 评估结果
- `file_path`: 保存路径

## 配置参数说明

### 数据配置 (data_config)
- `max_sequence_length`: 最大序列长度
- `min_sequence_length`: 最小序列长度
- `num_event_types`: 事件类型数量
- `padding_value`: 补齐值
- `window_size`: 滑动窗口大小

### 特征配置 (feature_config)
- `use_statistical_features`: 是否使用统计特征
- `use_prediction_features`: 是否使用预测特征
- `use_pattern_features`: 是否使用模式特征
- `feature_dimension`: 特征维度

### 聚类配置 (clustering_config)
- `n_clusters`: 聚类数量
- `algorithm`: 聚类算法
- `random_state`: 随机种子

### 模型配置 (model_config)
- `lstm_hidden_size`: LSTM隐藏层大小
- `lstm_num_layers`: LSTM层数
- `lstm_dropout`: LSTM dropout率
- `attention_heads`: 注意力头数
- `attention_dropout`: 注意力dropout率
- `use_attention`: 是否使用注意力机制

### 训练配置 (training_config)
- `batch_size`: 批次大小
- `learning_rate`: 学习率
- `num_epochs`: 训练轮数
- `sequence_loss_weight`: 序列损失权重
- `anomaly_loss_weight`: 异常损失权重
- `weight_decay`: 权重衰减
- `gradient_clip`: 梯度裁剪
- `early_stopping_patience`: 早停耐心值

## 错误处理

### 常见异常

1. **ValueError**: 参数值无效
2. **FileNotFoundError**: 文件不存在
3. **MemoryError**: 内存不足
4. **RuntimeError**: 运行时错误

### 错误处理示例

```python
try:
    preprocessor = DataPreprocessor(config)
    sequences = preprocessor.load_data("data/raw/logs.txt")
except FileNotFoundError:
    print("数据文件不存在")
except ValueError as e:
    print(f"参数错误: {e}")
except Exception as e:
    print(f"未知错误: {e}")
```

## 性能优化建议

### 1. 内存优化
- 使用数据生成器减少内存占用
- 分批处理大数据集
- 及时释放不需要的变量

### 2. 计算优化
- 使用GPU加速训练
- 并行处理特征提取
- 缓存中间结果

### 3. 存储优化
- 压缩存储特征数据
- 使用高效的数据格式
- 定期清理临时文件

## 版本兼容性

### Python版本
- Python 3.8+
- 推荐使用 Python 3.9 或 3.10

### 依赖包版本
- PyTorch >= 1.8.0
- NumPy >= 1.19.0
- scikit-learn >= 0.24.0

### 向后兼容性
- 主要版本更新可能包含破坏性变更
- 次要版本更新保持向后兼容
- 补丁版本更新完全向后兼容 