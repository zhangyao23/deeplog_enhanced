# 文本日志处理指南

## 概述

本指南详细说明如何将原始文本日志转换为DeepLog所需的数字序列格式。DeepLog项目本身只支持数字序列输入，因此需要先将文本日志进行预处理。

## 转换流程

### 1. 文本日志 → 日志模板 → 数字序列

```
原始文本日志 → [Log Parser] → 日志模板 → [模板映射] → 数字序列
```

**示例转换过程：**
```
# 原始日志
"2023-01-01 10:00:01 INFO User john login successful from 192.168.1.100"
"2023-01-01 10:01:15 INFO User mary login successful from 192.168.1.101"
"2023-01-01 10:02:30 ERROR Database connection failed for user admin"

# 解析后模板
"User <*> login successful from <ip>"     → ID: 15
"Database connection failed for user <*>"  → ID: 22

# 最终序列
15 22 15 15 22 15
```

## 安装依赖

### 推荐方案：使用Log Parser（Drain3）

```bash
# 安装Drain3（推荐）
pip install drain3

# 或者安装完整的Log Parser工具包
pip install logparser
```

### 备选方案：使用基础文本处理

如果无法安装Log Parser，项目提供了基础文本处理功能，无需额外依赖。

## 使用方法

### 方法一：使用示例脚本（推荐）

```bash
# 运行示例脚本
python3 enhanced_deeplog/scripts/00_text_to_sequences.py
```

这个脚本会：
1. 创建示例日志文件
2. 使用Log Parser解析日志
3. 生成数字序列文件
4. 保存模板映射

### 方法二：自定义处理

```python
from enhanced_deeplog.src.log_parser_processor import LogParserProcessor

# 配置
config = {
    "parser_config": {
        "drain_depth": 4,
        "drain_max_children": 100,
        "drain_sim_threshold": 0.4,
        "drain_max_clusters": 1000
    },
    "data_config": {
        "session_size": 50,
        "min_sequence_length": 10
    }
}

# 初始化处理器
processor = LogParserProcessor(config)

# 处理文本文件
sequences = processor.process_text_file_advanced(
    input_file="your_logs.txt",
    output_file="converted_sequences.txt",
    session_size=50
)

# 保存模板映射
processor.save_template_mapping_advanced("template_mapping.json")
```

## 配置说明

### Log Parser配置

```python
parser_config = {
    "drain_depth": 4,              # 解析树深度
    "drain_max_children": 100,     # 最大子节点数
    "drain_sim_threshold": 0.4,    # 相似度阈值
    "drain_max_clusters": 1000     # 最大聚类数
}
```

### 数据处理配置

```python
data_config = {
    "session_size": 50,            # 会话大小（序列长度）
    "min_sequence_length": 10      # 最小序列长度
}
```

## 支持的日志格式

### 自动识别的模式

- **时间戳**：`2023-01-01 10:00:01` → `<timestamp>`
- **IP地址**：`192.168.1.100` → `<ip>`
- **数字ID**：`12345` → `<id>`
- **路径**：`/var/log/app.log` → `<path>`
- **邮箱**：`user@example.com` → `<email>`
- **URL**：`https://example.com` → `<url>`
- **UUID**：`550e8400-e29b-41d4-a716-446655440000` → `<uuid>`
- **引号内容**：`"some text"` → `<quoted>`

### 日志类型支持

- **系统日志**：操作系统事件、服务启停
- **应用日志**：Web服务器访问日志、数据库操作日志
- **网络日志**：防火墙日志、路由器日志
- **分布式系统日志**：Hadoop HDFS、微服务调用链

## 输出文件说明

### 1. 序列文件 (`converted_sequences.txt`)

每行一个数字序列，空格分隔：
```
15 22 15 15 22 15 8 12 15 22
22 15 8 12 15 22 15 8 12 15
```

### 2. 模板映射文件 (`template_mapping.json`)

包含模板到ID的映射关系：
```json
{
  "template_to_id": {
    "User <*> login successful from <ip>": 15,
    "Database connection failed for user <*>": 22
  },
  "id_to_template": {
    "15": "User <*> login successful from <ip>",
    "22": "Database connection failed for user <*>"
  },
  "stats": {
    "total_logs": 1000,
    "parsed_logs": 998,
    "parse_rate": 0.998
  }
}
```

## 性能优化建议

### 1. 日志预处理

- **清理无效日志**：移除空行、格式错误的日志
- **时间排序**：确保日志按时间顺序排列
- **去重处理**：移除重复的日志条目

### 2. 参数调优

- **相似度阈值**：`drain_sim_threshold` 影响模板聚合程度
- **会话大小**：`session_size` 影响序列长度
- **最大聚类数**：`drain_max_clusters` 影响模板数量

### 3. 内存优化

- **分批处理**：对于大文件，分批读取和处理
- **流式处理**：使用生成器避免一次性加载全部数据

## 常见问题

### Q1: Log Parser安装失败怎么办？

**A**: 使用基础文本处理器，无需额外依赖：
```python
from enhanced_deeplog.src.text_processor import TextProcessor
```

### Q2: 解析率很低怎么办？

**A**: 
1. 检查日志格式是否规范
2. 调整相似度阈值
3. 增加日志预处理步骤

### Q3: 生成的序列太短怎么办？

**A**: 
1. 增加 `session_size` 参数
2. 合并多个日志文件
3. 调整序列分割策略

### Q4: 模板数量太多怎么办？

**A**: 
1. 降低 `drain_sim_threshold`
2. 增加 `drain_max_clusters`
3. 优化日志预处理规则

## 最佳实践

### 1. 日志质量

- 确保日志格式一致
- 移除无关的调试信息
- 标准化时间戳格式

### 2. 参数选择

- 根据日志特点调整参数
- 进行小规模测试验证
- 监控解析率和模板质量

### 3. 数据管理

- 定期更新模板映射
- 备份重要的映射文件
- 记录处理配置和参数

## 下一步

转换完成后，您可以使用生成的序列文件进行DeepLog训练：

```bash
# 使用转换后的序列文件进行训练
python3 enhanced_deeplog/scripts/01_train_base_model.py
```

确保在配置文件中指定正确的数据文件路径。 