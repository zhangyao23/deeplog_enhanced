#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本日志转换为DeepLog序列格式
使用Log Parser将原始文本日志转换为DeepLog所需的数字序列格式
"""

import sys
import os
import json
import logging
from pathlib import Path

# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# 导入本地模块
from enhanced_deeplog.src.log_parser_processor import LogParserProcessor

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_config():
    """
    定义脚本所需的配置
    """
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
    return config

def create_sample_log_file():
    """
    创建示例日志文件用于测试
    """
    sample_logs = [
        "2023-01-01 10:00:01 INFO User john login successful from 192.168.1.100",
        "2023-01-01 10:00:15 INFO User mary login successful from 192.168.1.101",
        "2023-01-01 10:01:30 ERROR Database connection failed for user admin",
        "2023-01-01 10:02:45 INFO File /var/log/app.log uploaded successfully",
        "2023-01-01 10:03:12 WARN High memory usage detected: 85%",
        "2023-01-01 10:04:20 INFO User john logout successful",
        "2023-01-01 10:05:33 ERROR Authentication failed for user unknown",
        "2023-01-01 10:06:45 INFO Backup completed successfully to /backup/data",
        "2023-01-01 10:07:18 WARN Disk space low: 10% remaining",
        "2023-01-01 10:08:30 INFO User mary logout successful",
        "2023-01-01 10:09:42 ERROR Network timeout occurred",
        "2023-01-01 10:10:55 INFO System maintenance completed",
        "2023-01-01 10:11:08 WARN CPU usage high: 90%",
        "2023-01-01 10:12:20 INFO User admin login successful from 192.168.1.102",
        "2023-01-01 10:13:35 ERROR File access denied for user guest",
        "2023-01-01 10:14:48 INFO Database backup started",
        "2023-01-01 10:15:12 WARN Slow query detected: 5.2 seconds",
        "2023-01-01 10:16:25 INFO User admin logout successful",
        "2023-01-01 10:17:38 ERROR Service unavailable: database connection",
        "2023-01-01 10:18:50 INFO System reboot initiated",
        "2023-01-01 10:19:03 WARN Low disk space: 5% remaining",
        "2023-01-01 10:20:15 INFO User john login successful from 192.168.1.103",
        "2023-01-01 10:21:28 ERROR Permission denied for file /etc/config",
        "2023-01-01 10:22:41 INFO Log rotation completed",
        "2023-01-01 10:23:54 WARN Memory usage critical: 95%",
        "2023-01-01 10:24:07 INFO User mary login successful from 192.168.1.104",
        "2023-01-01 10:25:20 ERROR Network interface down: eth0",
        "2023-01-01 10:26:33 INFO Security scan completed",
        "2023-01-01 10:27:46 WARN High temperature detected: 75°C",
        "2023-01-01 10:28:59 INFO User john logout successful",
        "2023-01-01 10:29:12 ERROR Database corruption detected",
        "2023-01-01 10:30:25 INFO System update completed",
        "2023-01-01 10:31:38 WARN Battery level low: 15%",
        "2023-01-01 10:32:51 INFO User admin login successful from 192.168.1.105",
        "2023-01-01 10:33:04 ERROR File not found: /var/data/missing.txt",
        "2023-01-01 10:34:17 INFO Cache cleared successfully",
        "2023-01-01 10:35:30 WARN Disk I/O high: 80%",
        "2023-01-01 10:36:43 INFO User mary logout successful",
        "2023-01-01 10:37:56 ERROR Service timeout: web server",
        "2023-01-01 10:38:09 INFO Database optimization completed",
        "2023-01-01 10:39:22 WARN Network latency high: 200ms",
        "2023-01-01 10:40:35 INFO User john login successful from 192.168.1.106",
        "2023-01-01 10:41:48 ERROR Memory allocation failed",
        "2023-01-01 10:42:01 INFO Backup verification completed",
        "2023-01-01 10:43:14 WARN CPU temperature high: 80°C",
        "2023-01-01 10:44:27 INFO User admin logout successful",
        "2023-01-01 10:45:40 ERROR Disk space exhausted",
        "2023-01-01 10:46:53 INFO System health check completed",
        "2023-01-01 10:47:06 WARN Low memory: 8% available",
        "2023-01-01 10:48:19 INFO User mary login successful from 192.168.1.107",
        "2023-01-01 10:49:32 ERROR Network packet loss: 5%",
        "2023-01-01 10:50:45 INFO Log compression completed",
        "2023-01-01 10:51:58 WARN High disk usage: 92%",
        "2023-01-01 10:52:11 INFO User john logout successful",
        "2023-01-01 10:53:24 ERROR Service crash: application server",
        "2023-01-01 10:54:37 INFO Database recovery completed",
        "2023-01-01 10:55:50 WARN Battery critical: 5%",
        "2023-01-01 10:56:03 INFO User admin login successful from 192.168.1.108",
        "2023-01-01 10:57:16 ERROR File system corruption detected",
        "2023-01-01 10:58:29 INFO Security update installed",
        "2023-01-01 10:59:42 WARN Network bandwidth high: 85%",
        "2023-01-01 11:00:55 INFO User mary logout successful"
    ]
    
    # 创建数据目录
    data_dir = Path("enhanced_deeplog/data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 写入示例日志文件
    sample_file = data_dir / "sample_logs.txt"
    with open(sample_file, 'w', encoding='utf-8') as f:
        for log in sample_logs:
            f.write(log + '\n')
    
    print(f"示例日志文件已创建: {sample_file}")
    return str(sample_file)

def main():
    """
    主函数：演示文本到序列的转换过程
    """
    print("=== 文本日志转换为DeepLog序列格式 ===")
    
    # 1. 获取配置
    config = get_config()
    print("✓ 配置加载完成")
    
    # 2. 创建示例日志文件
    input_file = create_sample_log_file()
    print("✓ 示例日志文件创建完成")
    
    # 3. 初始化Log Parser处理器
    processor = LogParserProcessor(config)
    print("✓ Log Parser处理器初始化完成")
    
    # 4. 创建Drain3配置文件
    processor.create_drain3_config()
    print("✓ Drain3配置文件创建完成")
    
    # 5. 处理文本文件
    output_file = "enhanced_deeplog/data/raw/converted_sequences.txt"
    session_size = config['data_config']['session_size']
    
    print(f"开始处理文本文件: {input_file}")
    sequences = processor.process_text_file_advanced(
        input_file=input_file,
        output_file=output_file,
        session_size=session_size
    )
    print(f"✓ 序列转换完成，输出文件: {output_file}")
    
    # 6. 保存模板映射
    mapping_file = "enhanced_deeplog/data/processed/template_mapping.json"
    Path(mapping_file).parent.mkdir(parents=True, exist_ok=True)
    processor.save_template_mapping_advanced(mapping_file)
    print(f"✓ 模板映射已保存: {mapping_file}")
    
    # 7. 显示处理结果
    template_info = processor.get_template_info_advanced()
    print("\n=== 处理结果统计 ===")
    print(f"总日志条数: {template_info['stats']['total_logs']}")
    print(f"解析成功条数: {template_info['stats']['parsed_logs']}")
    print(f"解析率: {template_info['stats']['parse_rate']:.2%}")
    print(f"唯一模板数: {template_info['total_templates']}")
    print(f"生成序列数: {len(sequences)}")
    print(f"Log Parser可用: {template_info['parser_available']}")
    
    # 8. 显示示例模板
    print("\n=== 示例模板映射 ===")
    for template, template_id in list(template_info['sample_templates'].items())[:5]:
        print(f"ID {template_id}: {template}")
    
    # 9. 显示示例序列
    print("\n=== 示例序列 ===")
    for i, seq in enumerate(sequences[:3]):
        print(f"序列 {i+1}: {' '.join(map(str, seq))}")
    
    print("\n=== 转换完成 ===")
    print("现在您可以使用转换后的序列文件进行DeepLog训练了！")
    print(f"序列文件路径: {output_file}")
    print(f"模板映射路径: {mapping_file}")

if __name__ == "__main__":
    main() 