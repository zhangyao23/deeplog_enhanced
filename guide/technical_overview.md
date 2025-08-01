# Enhanced DeepLog 技术原理总览

## 概述

Enhanced DeepLog 是对原始 DeepLog 方法论的重大升级，其核心目标是从**异常检测**演进到**无监督的异常分类**。本框架旨在解决以下核心问题：

1.  **自动发现异常模式**：无需人工预定义，自动从数据中归纳出不同类型的异常行为。
2.  **细粒度异常分类**：不仅判断"是否异常"，更能指出"是哪种异常"，为故障诊断提供更深度的洞察。
3.  **端到端自动化**：建立一套从原始日志到最终分类模型的完整、自动化的工作流。

---

## 数学原理

### 数据预处理数学原理

#### 日志模板解析
给定原始日志集合 $L = \{l_1, l_2, ..., l_N\}$，其中每条日志 $l_i$ 包含时间戳、事件类型和参数。

**模板提取过程**：
对于日志 $l_i$，我们通过以下步骤提取模板：
1. **参数识别**：使用正则表达式识别动态参数
2. **模板生成**：将动态参数替换为占位符

**模板相似度计算**：
$$sim(t_1, t_2) = \frac{|t_1 \cap t_2|}{|t_1 \cup t_2|}$$

其中 $t_1, t_2$ 是两个模板，$|t_1 \cap t_2|$ 是公共部分长度。

#### 事件映射
建立从模板到数字ID的映射函数：

$$f: T \rightarrow \{0, 1, 2, ..., |T|-1\}$$

其中 $T$ 是模板集合，$|T|$ 是模板总数。

**映射过程**：
$$e_i = f(t_i)$$

其中 $e_i$ 是事件ID，$t_i$ 是对应的模板。

#### 序列分割与补齐
**滑动窗口分割**：
给定序列 $S = [e_1, e_2, ..., e_L]$，使用窗口大小 $w$ 进行分割：

$$W_i = [e_i, e_{i+1}, ..., e_{i+w-1}]$$

**序列补齐**：
对于长度不足 $w$ 的窗口，使用零填充：

$$\hat{W}_i = [e_i, e_{i+1}, ..., e_{i+k-1}, 0, 0, ..., 0]$$

其中 $k$ 是实际长度，补齐后长度为 $w$。

**批量处理**：
将多个窗口组织成批次：

$$B = [W_1, W_2, ..., W_b] \in \mathbb{R}^{b \times w}$$

其中 $b$ 是批次大小。

### LSTM 网络数学原理

LSTM 网络通过以下数学公式处理序列数据：

**输入门 (Input Gate)**：
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

**遗忘门 (Forget Gate)**：
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

**输出门 (Output Gate)**：
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

**候选记忆单元**：
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

**记忆单元更新**：
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

**隐藏状态输出**：
$$h_t = o_t \odot \tanh(C_t)$$

其中：
- $x_t$ 是时间步 $t$ 的输入
- $h_t$ 是时间步 $t$ 的隐藏状态
- $C_t$ 是时间步 $t$ 的记忆单元
- $W_i, W_f, W_o, W_C$ 是权重矩阵
- $b_i, b_f, b_o, b_C$ 是偏置向量
- $\sigma$ 是 sigmoid 激活函数
- $\odot$ 表示逐元素乘法

### 注意力机制数学原理

多头注意力机制的计算公式：

**查询、键、值矩阵**：
$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

**注意力权重**：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**多头注意力**：
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

其中 $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$

### Birch 聚类算法数学原理

Birch 算法基于聚类特征 (Clustering Feature, CF) 的概念：

**聚类特征三元组**：
$$CF = (N, LS, SS)$$

其中：
- $N$ 是簇中样本数量
- $LS$ 是线性和 (Linear Sum)
- $SS$ 是平方和 (Square Sum)

**簇间距离计算**：
$$D_0 = \sqrt{\frac{SS_1 + SS_2}{N_1 + N_2} - 2\frac{LS_1 \cdot LS_2}{N_1 N_2} + \frac{|LS_1|^2}{N_1^2} + \frac{|LS_2|^2}{N_2^2}}$$

**簇内距离阈值**：
$$\text{radius} = \sqrt{\frac{SS}{N} - \frac{|LS|^2}{N^2}}$$

### 损失函数数学原理

**交叉熵损失**：
$$\mathcal{L}_{CE} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$

其中 $y_i$ 是真实标签，$\hat{y}_i$ 是预测概率，$C$ 是类别数。

**多任务学习损失**：
$$\mathcal{L}_{total} = \alpha \mathcal{L}_{sequence} + \beta \mathcal{L}_{anomaly}$$

其中 $\alpha$ 和 $\beta$ 是任务权重。

**轮廓系数 (Silhouette Score)**：
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

其中：
- $a(i)$ 是样本 $i$ 到同簇其他样本的平均距离
- $b(i)$ 是样本 $i$ 到最近其他簇的最小平均距离

---

## 核心原理：四阶段工作流

我们通过一个分为四个关键阶段的流程，来实现无监督的异常分类。其判断原理是：**不同类型的系统异常，会在正常模式的 LSTM 模型中留下不同模式的"印记"，我们可以通过捕捉和聚类这些"印记"来自动区分它们。**

### 阶段一：基础模型训练 - 学习"正常"

- **目标**：训练一个标准的 DeepLog (LSTM) 模型，使其深度学习**正常**的日志序列模式。
- **产出**：一个只认识"正常"行为的 `base_model.pth`，它将作为后续阶段的"判断基准"。

### 阶段二：深度特征提取 - 捕捉"意外"

- **问题**：仅靠"预测是否正确"来判断异常，信息维度太低。
- **解决方案**：我们不再手动设计特征，而是将阶段一训练好的模型作为特征提取器。当模型遇到它无法理解的异常序列时，其内部的 **LSTM 隐状态 (hidden state)** 会产生独特的"扰动"。我们捕捉这个高维向量作为该序列的"深度特征"。
- **效果**：这个特征向量是对序列模式的高度浓缩，比任何手动设计的特征都更具代表性。

### 阶段三：无监督聚类 - 归类"模式"

- **目标**：在没有任何标签的情况下，自动发现数据中存在哪些主要的日志行为模式。
- **方法**：使用高效的 **Birch 聚类算法**对第二阶段提取的所有深度特征进行聚类。
- **结果**：算法会自动将相似的日志行为（即特征向量在高维空间中相近的样本）划分到同一个簇。我们由此得到每个日志窗口的"伪标签"，其中最大的簇被视为**正常 (Class 0)**，其余的则代表了不同类型的**异常 (Class 1, Class 2...)**。

### 阶段四：多分类模型训练 - 学会"区分"

- **目标**：训练一个最终的、能够将新日志精确分类到我们所发现类别中的分类器。
- **架构**：我们设计了一个**双输入**的增强模型，它同时接收：
    1.  **原始日志序列**：从原始数据中学习模式。
    2.  **深度特征向量**：从阶段二获取的、关于模式的强先验信息。
- **训练**：以阶段三产生的聚类标签作为监督信号，训练模型将特定的日志模式映射到对应的类别。

---

## 技术架构

```
                                      +-------------------------+
                                      |      原始日志数据        |
                                      +-------------------------+
                                                  |
                                                  ▼
+---------------------------+       +---------------------------+       +---------------------------+       +-------------------------------+
|         阶段一            |------>|         阶段二            |------>|         阶段三            |------>|            阶段四             |
|      训练基础模型         |       |      提取深度特征         |       |      无监督聚类           |       |         训练多分类模型          |
| (只用正常日志)            |       | (使用基础模型处理所有日志)  |       | (使用Birch算法)         |       | (使用双输入增强模型)          |
+---------------------------+       +---------------------------+       +---------------------------+       +-------------------------------+
| 产出: base_model.pth      |       | 产出: features.npy        |       | 产出: final_labels.npy      |       | 产出: multiclass_model.pth    |
+---------------------------+       +---------------------------+       +---------------------------+       +-------------------------------+
```

---

## 神经网络的使用

### 1. LSTM 网络 (阶段一和阶段二)
我们沿用了 DeepLog 的核心思想，使用 LSTM 网络来学习日志序列的模式：

```python
# LSTM 网络结构示例
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

**DeepLog 的沿用**：
- 我们完全保留了 DeepLog 的 LSTM 架构和训练方式
- 使用滑动窗口将长序列分割成固定长度的子序列
- 通过预测下一个事件来学习序列模式
- 当预测失败时，认为检测到了异常

### 2. 增强的多分类模型 (阶段四)
在最终的多分类模型中，我们使用了更复杂的神经网络架构：

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

## 聚类过程详解

### Birch 聚类算法原理
我们选择 Birch 算法是因为它具有以下优势：
1. **高效性**：单遍扫描数据，时间复杂度为 O(n)
2. **内存友好**：通过 CF-Tree 结构有效管理内存
3. **自动确定簇数**：可以通过调整阈值来控制簇的数量

### 聚类过程示例

```python
# 聚类过程示例
from sklearn.cluster import Birch

# 1. 准备特征数据 (来自阶段二)
features = np.load('enhanced_deeplog/processed_data/features.npy')
print(f"特征矩阵形状: {features.shape}")  # 例如: (31460, 128)

# 2. 初始化 Birch 聚类器
birch = Birch(
    n_clusters=4,           # 期望的簇数
    threshold=0.5,          # 簇内距离阈值
    branching_factor=50     # 每个节点的最大子节点数
)

# 3. 执行聚类
cluster_labels = birch.fit_predict(features)
print(f"聚类标签: {np.unique(cluster_labels)}")  # [0, 1, 2, 3]

# 4. 分析聚类结果
for i in range(4):
    count = np.sum(cluster_labels == i)
    print(f"Class {i}: {count} 个样本 ({count/len(cluster_labels)*100:.1f}%)")
```

**聚类结果示例**：
```
Class 0: 18005 个样本 (57.2%)  # 正常行为
Class 1: 2704 个样本 (8.6%)   # 异常类型1
Class 2: 8533 个样本 (27.1%)  # 异常类型2  
Class 3: 2218 个样本 (7.1%)   # 异常类型3
```

### 聚类质量评估
我们使用轮廓系数 (Silhouette Score) 来评估聚类质量：

```python
from sklearn.metrics import silhouette_score

# 计算轮廓系数
silhouette_avg = silhouette_score(features, cluster_labels)
print(f"平均轮廓系数: {silhouette_avg:.3f}")  # 值越接近1，聚类质量越好
```

---

## 六个Python脚本的作用

### 1. `01_train_base_model.py` - 基础模型训练
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

### 2. `02_extract_features.py` - 深度特征提取
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

### 3. `03_anomaly_clustering.py` - 无监督聚类
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

### 4. `04_train_multiclass_model.py` - 多分类模型训练
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

### 5. `05_analyze_clusters.py` - 聚类结果分析
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

### 6. `00_data_preprocessing.py` - 数据预处理
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

## 性能表现

在 HDFS 数据集上，本框架自动发现了1个正常类别和3个异常类别，最终的多分类模型取得了以下性能：

| 类别 (自动发现) | Precision | Recall | F1-Score | Support (样本数) | 初步解读 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Class 0 (正常)** | 0.82 | 0.95 | 0.88 | 18,005 | 能够准确识别绝大多数正常日志 |
| **Class 1 (异常)** | 0.85 | 0.54 | 0.66 | 2,704 | 模式：罕见事件/意外重复 |
| **Class 2 (异常)** | 0.92 | 0.81 | 0.86 | 8,533 | 模式：模式严重破坏 |
| **Class 3 (异常)** | 0.92 | 0.60 | 0.73 | 2,218 | 模式：轻微模式偏离 |
| | | | | |
| **总准确率** | - | - | **0.85** | 31,460 | |

---

## 部署与应用

1.  **数据准备**：提供高质量的、具有代表性的历史日志。
2.  **执行工作流**：按 `enhanced_deeplog/scripts/` 目录下的 `01` 到 `04` 脚本顺序执行。
3.  **分析与迭代**：使用 `05_analyze_clusters.py` 理解异常类别含义，并根据新的数据定期重新运行流程以优化模型。
