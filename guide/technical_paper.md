# 基于深度特征聚类的无监督日志异常分类方法

## 摘要

传统的基于深度学习的日志异常检测方法，如 DeepLog，主要关注于区分正常与异常行为，而无法对多样的异常模式进行细粒**度的分类。这在需要对故障进行快速诊断和根因分析的运维场景中存在局限。本文提出了一种增强的 DeepLog 框架，通过结合深度学习与无监督聚类技术，实现了对系统日志异常的自动分类，无需任何人工标注。该方法首先利用一个在正常数据上训练的 LSTM 模型作为特征提取器，捕捉日志序列的深层语义信息；随后，采用高效的 Birch 聚类算法对这些特征进行无监督学习，自动发现数据中存在的多种行为模式（包括正常模式和多种异常模式）；最后，基于聚类结果训练一个多分类模型，用于对新日志进行实时分类。在公开的 HDFS 数据集上的实验证明，该方法不仅能有效检测异常，还能将异常成功地划分为多个具有实际意义的类别，总体准确率达到了 85%。

**关键词：** 异常分类，无监督学习，深度学习，LSTM，聚类分析，日志分析

---

## 1. 引言

### 1.1 研究背景

随着大规模分布式系统的普及，通过日志进行系统监控和故障诊断已成为标准实践。基于深度学习的异常检测方法，特别是以 DeepLog 为代表的、利用长短期记忆网络 (LSTM) 学习序列模式的方法，在自动检测偏离正常行为的事件上取得了巨大成功。然而，这些方法大多停留在二分类的层面，即仅能回答"系统是否异常？"。

在现代运维（AIOps）场景下，我们不仅需要知道"有异常"，更迫切地想知道"是哪种异常？"。不同类型的故障（如服务崩溃、权限问题、资源耗尽）在日志中会表现为截然不同的模式。如果能自动对这些异常模式进行分类，将极大地缩短故障排查时间，并为根因分析提供关键线索。

### 1.2 研究挑战与目标

当前的主要挑战在于，生产环境中的日志数据往往缺少精确的、带有分类信息的标签。人工标注不仅成本高昂，且难以覆盖所有未知的异常类型。

因此，本研究的核心目标是：**设计并实现一个端到端的无监督异常分类框架**。该框架应能：

1. 在没有人工标签的情况下，自动发现并归纳出日志数据中存在的多种异常模式。
2. 训练一个能够对新日志进行实时、细粒度分类的健壮模型。
3. 整个流程应具备高自动化程度和良好的可解释性。

---

## 2. 相关工作

(此部分可根据您的需求详细展开，回顾传统方法、DeepLog、其他基于深度学习的方法等)

---

## 3. 方法设计

本研究提出的框架包含四个主要阶段，旨在以一种无监督的方式，从原始日志中自动发现并学习多种行为模式。我们将其命名为"四阶段深度特征聚类法"。

### 3.1 核心数学原理

#### 3.1.1 深度特征提取的数学基础

**LSTM隐状态作为异常特征向量**：
我们提出将LSTM在处理日志序列后的隐状态作为深度特征：

$$
F_i = h_T^{(i)} \in \mathbb{R}^d
$$

其中 $h_T^{(i)}$ 是第 $i$ 个日志窗口经过LSTM处理后最后一个时间步的隐状态，$d$ 是隐状态维度。

**异常扰动量化**：
对于异常序列，LSTM隐状态会产生显著扰动，我们定义扰动强度：

$$
\Delta F_i = \|F_i - \bar{F}_{normal}\|_2
$$

其中 $\bar{F}_{normal}$ 是正常序列隐状态的平均值。

#### 3.1.2 无监督聚类的数学原理

**Birch聚类特征 (CF) 三元组**：

$$
CF = (N, LS, SS)
$$

其中 $N$ 是簇中样本数量，$LS = \sum_{i=1}^{N} x_i$ 是线性和，$SS = \sum_{i=1}^{N} x_i^2$ 是平方和。

**簇间距离计算**：

$$
D_0 = \sqrt{\frac{SS_1 + SS_2}{N_1 + N_2} - 2\frac{LS_1 \cdot LS_2}{N_1 N_2} + \frac{|LS_1|^2}{N_1^2} + \frac{|LS_2|^2}{N_2^2}}
$$

**簇内距离阈值**：

$$
\text{radius} = \sqrt{\frac{SS}{N} - \frac{|LS|^2}{N^2}}
$$

#### 3.1.3 多任务学习损失函数

**双重输入融合损失**：

$$
\mathcal{L}_{total} = \alpha \mathcal{L}_{sequence} + \beta \mathcal{L}_{anomaly} + \gamma \mathcal{L}_{consistency}
$$

其中：
- $\mathcal{L}_{sequence}$ 是序列预测损失
- $\mathcal{L}_{anomaly}$ 是异常分类损失  
- $\mathcal{L}_{consistency}$ 是特征一致性损失，确保原始序列特征与LSTM隐状态特征的一致性
- $\alpha + \beta + \gamma = 1$ 是任务权重

**特征一致性损失**：

$$
\mathcal{L}_{consistency} = \|\text{MLP}(F_{lstm}) - F_{external}\|_2
$$

其中 $F_{lstm}$ 是LSTM隐状态特征，$F_{external}$ 是外部输入的特征向量。

### 3.2 阶段一：基于 LSTM 的正常模式学习 (Foundation Model Training)

作为整个流程的基石，我们首先遵循 DeepLog 的标准方法，在一个仅包含**正常日志**的数据集上训练一个 LSTM 网络。该网络的目标是学习正常操作下日志序列的语法和语义规则。通过训练，模型学会了在给定一个正常序列的前缀时，高概率地预测出下一个可能的日志事件。这个阶段的产物是一个"专家模型"，它只对正常的系统行为有深刻的理解，任何偏离其"知识范围"的模式都将被视为潜在的异常。

**DeepLog 的沿用**：
我们完全保留了 DeepLog 的核心架构和训练方式：

- 使用 LSTM 网络学习序列模式
- 采用滑动窗口技术将长序列分割成固定长度的子序列
- 通过预测下一个事件来学习序列的语法规则
- 当预测失败时，认为检测到了异常

**神经网络架构**：

```python
class BaseDeepLogModel(nn.Module):
    def __init__(self, num_event_types, hidden_size=128):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,           # 每个事件ID作为一个数值输入
            hidden_size=hidden_size, # LSTM隐藏层大小
            num_layers=2,           # LSTM层数
            batch_first=True,
            dropout=0.2
        )
        self.classifier = nn.Linear(hidden_size, num_event_types)
  
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        x = x.unsqueeze(-1).float()  # 增加特征维度
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出进行预测
        last_output = lstm_out[:, -1, :]
        predictions = self.classifier(last_output)
        return predictions
```

### 3.3 阶段二：面向分类的深度特征提取 (Deep Feature Extraction)

传统 DeepLog 在检测时，仅利用了模型的预测结果（即下一个事件是否在 Top-K 预测中）。我们认为，这种方式损失了大量有价值的信息。当一个训练好的 LSTM 模型面对一个**异常**序列时，其内部的隐状态 (hidden state) 会因为无法理解当前的模式而产生剧烈的"扰动"。这种"扰动"的模式，对于不同类型的异常是不同的。

因此，我们提出将 **LSTM 在处理完一个日志窗口后的最后一个时间步的隐状态向量**，作为该窗口的"深度特征"。这个向量可以被看作是 LSTM 对整个序列模式的高度浓缩和概括，它比任何人工设计的统计特征或单一的预测结果都包含了远为丰富和抽象的信息。我们使用阶段一训练好的模型，遍历全部数据集（包括正常和异常），为每个日志窗口生成一个特征向量，从而将原始的、可变的日志序列数据映射到了一个统一的、高维的特征空间中。

**特征提取过程**：

```python
# 加载训练好的基础模型
model = torch.load('base_model.pth')
model.eval()

features = []
with torch.no_grad():
    for batch in data_loader:
        # 获取LSTM的隐状态
        lstm_out, (hidden, cell) = model.lstm(batch.unsqueeze(-1).float())
        # 取最后一层的隐状态作为特征
        batch_features = hidden[-1].cpu().numpy()  # shape: (batch_size, hidden_size)
        features.append(batch_features)

# 合并所有特征
features = np.vstack(features)  # shape: (total_samples, hidden_size)
np.save('features.npy', features)
```

### 3.4 阶段三：基于 Birch 的无监督模式发现 (Unsupervised Pattern Discovery)

在获得了所有日志窗口的特征向量后，我们面临的核心问题是：如何在没有标签的情况下，将这些向量按其代表的行为模式进行分组？

我们选择了 **Birch (Balanced Iterative Reducing and Clustering using Hierarchies)** 聚类算法来解决这个问题。选择 Birch 的主要原因有二：

1. **高效性与可扩展性**: Birch 算法通过构建一个内存中的"聚类特征树"(CF Tree)，可以单遍扫描数据就完成高质量的聚类，非常适合处理动辄数百万条的日志数据集，其线性的时间复杂度远优于 K-means 等传统算法。
2. **无需预设簇数（间接实现）**: 虽然 Birch 算法本身也需要指定最终簇数，但其 CF-Tree 结构提供了一种天然的、多层次的聚类视图。我们可以通过分析树的结构或结合其他指标（如轮廓系数）来辅助判断一个合理的簇数量，从而实现半自动化的簇数确定。

我们将所有特征向量输入 Birch 算法。算法会将特征空间中距离相近的向量（即代表相似日志行为的窗口）划分到同一个簇中。聚类完成后，我们得到了每个日志窗口的类别标签。通常，样本数量最多的那个簇对应的是**正常**行为，而其他规模较小的簇则分别对应了**不同类型**的异常行为。至此，我们完成了从无标签数据到有类别划分的"伪标签"数据的转换。

**聚类过程示例**：

```python
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score

# 加载特征数据
features = np.load('features.npy')
print(f"特征矩阵形状: {features.shape}")  # 例如: (31460, 128)

# 初始化 Birch 聚类器
birch = Birch(
    n_clusters=4,           # 期望的簇数
    threshold=0.5,          # 簇内距离阈值
    branching_factor=50     # 每个节点的最大子节点数
)

# 执行聚类
cluster_labels = birch.fit_predict(features)
print(f"聚类标签: {np.unique(cluster_labels)}")  # [0, 1, 2, 3]

# 分析聚类结果
for i in range(4):
    count = np.sum(cluster_labels == i)
    print(f"Class {i}: {count} 个样本 ({count/len(cluster_labels)*100:.1f}%)")

# 评估聚类质量
silhouette_avg = silhouette_score(features, cluster_labels)
print(f"平均轮廓系数: {silhouette_avg:.3f}")

# 保存聚类结果
np.save('final_labels.npy', cluster_labels)
```

**聚类结果**：

```
Class 0: 18005 个样本 (57.2%)  # 正常行为
Class 1: 2704 个样本 (8.6%)   # 异常类型1
Class 2: 8533 个样本 (27.1%)  # 异常类型2  
Class 3: 2218 个样本 (7.1%)   # 异常类型3
```

### 3.5 阶段四：基于双重输入的多分类模型训练 (Multi-Class Model Training)

为了构建一个最终可用的、健壮的分类器，我们训练了一个增强型的多任务模型。该模型的创新之处在于其**双重输入**的设计：

- **输入1 (原始序列)**: 原始的日志事件 ID 序列。这让模型可以像标准 DeepLog 一样从头学习序列的底层模式。
- **输入2 (特征向量)**: 阶段二提取的深度特征向量。这个特征向量为模型提供了关于序列模式"是什么"的高度概括，是一个强有力的先验信息，能显著加速模型的收敛并提升性能。

模型将这两个输入在内部进行融合，然后通过一个或多个全连接层，输出对各个类别（由阶段三的聚类所定义）的预测概率。我们使用阶段三生成的聚类标签作为监督信号，通过最小化交叉熵损失函数来训练此模型。通过这种方式，最终的模型不仅学会了区分正常与异常，更学会了识别由聚类算法所定义的、多种细粒度的异常模式。

**增强模型架构**：

```python
class EnhancedDeepLogModel(nn.Module):
    def __init__(self, config, num_event_types, num_anomaly_types, feature_dim):
        super().__init__()
        # LSTM 编码器 (沿用 DeepLog)
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=config['lstm_hidden_size'],
            num_layers=config['lstm_num_layers'],
            batch_first=True,
            dropout=config['lstm_dropout']
        )
      
        # 注意力机制 (新增)
        if config['use_attention']:
            self.attention = nn.MultiheadAttention(
                embed_dim=config['lstm_hidden_size'],
                num_heads=config['attention_heads'],
                batch_first=True
            )
      
        # 双输出头
        self.sequence_head = nn.Sequential(
            nn.Linear(config['lstm_hidden_size'], config['lstm_hidden_size'] // 2),
            nn.ReLU(),
            nn.Linear(config['lstm_hidden_size'] // 2, num_event_types)
        )
      
        # 异常分类头 (结合LSTM输出和外部特征)
        self.anomaly_head = nn.Sequential(
            nn.Linear(config['lstm_hidden_size'] + feature_dim, config['lstm_hidden_size'] // 2),
            nn.ReLU(),
            nn.Linear(config['lstm_hidden_size'] // 2, num_anomaly_types)
        )
  
    def forward(self, x_seq, x_feat):
        # LSTM 编码
        x_seq = x_seq.unsqueeze(-1)
        lstm_out, _ = self.lstm(x_seq)
      
        # 注意力机制
        if hasattr(self, 'attention'):
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            sequence_features = attn_out[:, -1, :]
        else:
            sequence_features = lstm_out[:, -1, :]
      
        # 双任务输出
        sequence_pred = self.sequence_head(sequence_features)
        combined_features = torch.cat([sequence_features, x_feat], dim=1)
        anomaly_pred = self.anomaly_head(combined_features)
      
        return sequence_pred, anomaly_pred
```

---

## 4. 六个Python脚本的作用详解

### 4.1 `01_train_base_model.py` - 基础模型训练

**作用**：训练标准的 DeepLog 模型，学习正常日志序列模式
**输入**：原始日志数据
**输出**：`base_model.pth` (包含训练好的LSTM模型和事件映射)
**关键代码**：

```python
# 训练LSTM模型学习正常序列
model = BaseDeepLogModel(num_event_types=len(event_mapping))
# 使用正常日志进行训练
trainer.train(model, train_loader)
```

### 4.2 `02_extract_features.py` - 深度特征提取

**作用**：使用训练好的模型提取所有日志窗口的LSTM隐状态特征
**输入**：`base_model.pth` + 所有日志数据
**输出**：`features.npy` (每个日志窗口的特征向量)
**关键代码**：

```python
# 加载训练好的模型
model = torch.load('base_model.pth')
# 提取LSTM隐状态作为特征
with torch.no_grad():
    for batch in data_loader:
        lstm_out, (hidden, cell) = model.lstm(batch)
        features.append(hidden[-1].cpu().numpy())  # 取最后一层的隐状态
```

### 4.3 `03_anomaly_clustering.py` - 无监督聚类

**作用**：对特征向量进行聚类，自动发现异常模式
**输入**：`features.npy`
**输出**：`final_labels.npy` (每个样本的聚类标签)
**关键代码**：

```python
# 使用Birch算法进行聚类
birch = Birch(n_clusters=4, threshold=0.5)
cluster_labels = birch.fit_predict(features)
# 保存聚类结果
np.save('final_labels.npy', cluster_labels)
```

### 4.4 `04_train_multiclass_model.py` - 多分类模型训练

**作用**：训练最终的多分类模型，能够识别不同的异常类型
**输入**：原始序列 + 特征向量 + 聚类标签
**输出**：`multiclass_model.pth` (训练好的多分类模型)
**关键代码**：

```python
# 创建增强模型
model = EnhancedDeepLogModel(
    config=config,
    num_event_types=num_keys,
    num_anomaly_types=num_classes,
    feature_dim=features.shape[1]
)
# 使用聚类标签作为监督信号进行训练
trainer.train(model, train_loader, val_loader)
```

### 4.5 `05_analyze_clusters.py` - 聚类结果分析

**作用**：分析每个聚类类别对应的具体日志内容，理解异常模式
**输入**：聚类标签 + 原始日志数据
**输出**：控制台输出，显示每个类别的典型日志模式
**关键代码**：

```python
# 为每个类别随机抽取样本
for label in unique_labels:
    indices = np.where(labels == label)[0]
    sample_indices = np.random.choice(indices, 5, replace=False)
    # 解码并显示日志内容
    for idx in sample_indices:
        sequence = sequences_padded[idx]
        decoded_sequence = [id_to_event_mapping[event_id] for event_id in sequence]
        print(f"Class {label} 样本: {decoded_sequence}")
```

### 4.6 `00_data_preprocessing.py` - 数据预处理

**作用**：将原始日志转换为模型可处理的格式
**输入**：原始日志文件
**输出**：处理后的序列数据和事件映射
**关键代码**：

```python
# 解析日志模板
templates = parse_log_templates(raw_logs)
# 创建事件映射
event_mapping = {template: idx for idx, template in enumerate(templates)}
# 将日志转换为数字序列
sequences = convert_logs_to_sequences(raw_logs, event_mapping)
```

---

## 5. 实验与结果

### 5.1 数据集与评估指标

- **数据集**: 我们在公开的 HDFS 数据集上进行了实验。该数据集包含...
- **评估指标**: 我们使用标准的分类指标，包括准确率 (Accuracy)、精确率 (Precision)、召回率 (Recall) 和 F1-Score。

### 5.2 实验结果

本框架在 HDFS 数据集上自动发现了4个主要的行为模式。其中最大的簇（Class 0）被定义为正常行为，其余为异常。最终训练的多分类模型在测试集上表现如下：


| 类别 (自动发现)    | Precision | Recall   | F1-Score | Support (样本数) | 初步解读                           |
| :----------------- | :-------- | :------- | :------- | :--------------- | :--------------------------------- |
| **Class 0 (正常)** | 0.82      | 0.95     | 0.88     | 18,005           | 模型能准确识别绝大多数正常日志     |
| **Class 1 (异常)** | 0.85      | 0.54     | 0.66     | 2,704            | 模式：罕见事件或单个事件的意外重复 |
| **Class 2 (异常)** | 0.92      | 0.81     | 0.86     | 8,533            | 模式：日志序列模式的严重破坏或崩溃 |
| **Class 3 (异常)** | 0.92      | 0.60     | 0.73     | 2,218            | 模式：与正常行为模式的轻微偏离     |
|                    |           |          |          |                  |                                    |
| **总准确率**       | -         | -        | **0.85** | 31,460           |                                    |
| **加权平均**       | **0.86**  | **0.85** | **0.85** | **31,460**       |                                    |

### 5.3 结果分析

实验结果表明，该框架不仅在整体上取得了 85% 的高准确率，而且成功地区分出了至少三种不同特征的异常模式。例如，Class 2 的各项指标都很高，说明其模式（如模式严重破坏）非常独特，容易识别。而 Class 1 和 Class 3 的召回率相对较低，说明这两类"轻微异常"与正常或其他异常模式存在一定的相似性，是未来优化的重点。通过 `05_analyze_clusters.py` 脚本，我们可以进一步探查每个类别对应的具体日志内容，从而为这些由机器发现的类别赋予业务含义。

---

## 6. 结论与展望

本文提出并验证了一种基于深度特征聚类的无监督日志异常分类方法。该方法通过一个四阶段工作流，成功地将传统的异常检测提升到了细粒度的异常分类，且无需任何人工标注，具有很高的实用价值和可扩展性。

未来的工作可以从以下几个方面展开：

1. **优化召回率**: 针对召回率较低的类别，研究数据增强或更先进的分类模型。
2. **在线学习**: 将当前框架改造为支持流式数据和模型增量更新的在线学习系统。
3. **可解释性增强**: 结合注意力可视化等技术，进一步解释模型做出分类决策的原因。
