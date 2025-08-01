#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成用于Enhanced DeepLog训练的合成数据
- 1类正常序列
- 3类不同模式的异常序列
"""

import numpy as np
import os
import random
from tqdm import tqdm

# --- 配置参数 ---
OUTPUT_DIR = 'enhanced_deeplog/data/raw'
TRAIN_DATA_FILE = os.path.join(OUTPUT_DIR, 'synthetic_hdfs_train.txt')
TRAIN_LABEL_FILE = os.path.join(OUTPUT_DIR, 'synthetic_hdfs_train_labels.txt')

NUM_NORMAL_SEQUENCES = 4000
NUM_ANOMALY_SEQUENCES_PER_TYPE = 1000

MIN_SEQ_LEN = 15
MAX_SEQ_LEN = 50

# 事件ID范围
NORMAL_EVENT_POOL = list(range(1, 21))  # 正常事件ID池
RARE_EVENT_POOL = list(range(80, 100)) # 稀有/侵入事件ID池
CHAOTIC_EVENT_POOL = list(range(21, 80)) # 混乱事件ID池

# --- 正常序列模式 ---
NORMAL_PATTERN_A = [10, 11, 12, 13, 14, 15]
NORMAL_PATTERN_B = [1, 2, 3, 1, 4, 5, 1, 6]

def generate_normal_sequence():
    """生成一条正常序列"""
    seq_len = random.randint(MIN_SEQ_LEN, MAX_SEQ_LEN)
    base_pattern = random.choice([NORMAL_PATTERN_A, NORMAL_PATTERN_B])
    
    sequence = []
    while len(sequence) < seq_len:
        sequence.extend(base_pattern)
        
    # 添加少量噪声
    for _ in range(int(0.05 * seq_len)): # 5%的噪声
        idx_to_replace = random.randint(0, len(sequence) - 1)
        sequence[idx_to_replace] = random.choice(NORMAL_EVENT_POOL)
        
    return sequence[:seq_len]

# --- 异常序列模式 ---

def generate_spike_anomaly():
    """异常类型1: 事件侵入"""
    sequence = generate_normal_sequence()
    seq_len = len(sequence)
    
    # 插入1-3个稀有事件
    num_spikes = random.randint(1, 3)
    for _ in range(num_spikes):
        spike_event = random.choice(RARE_EVENT_POOL)
        insert_pos = random.randint(int(seq_len * 0.2), int(seq_len * 0.8))
        sequence.insert(insert_pos, spike_event)
        
    return sequence[:seq_len]

def generate_repetition_anomaly():
    """异常类型2: 事件重复"""
    sequence = generate_normal_sequence()
    seq_len = len(sequence)
    
    # 选择一个正常事件进行重复
    event_to_repeat = random.choice(NORMAL_EVENT_POOL)
    repetition_count = random.randint(5, 10)
    
    insert_pos = random.randint(int(seq_len * 0.2), int(seq_len * 0.8))
    
    # 替换或插入
    for i in range(repetition_count):
        if insert_pos + i < len(sequence):
            sequence[insert_pos + i] = event_to_repeat
        else:
            sequence.append(event_to_repeat)
            
    return sequence[:seq_len]

def generate_chaotic_anomaly():
    """异常类型3: 序列中断"""
    sequence = generate_normal_sequence()
    seq_len = len(sequence)
    
    # 在序列的后半部分引入混乱
    break_point = random.randint(int(seq_len * 0.4), int(seq_len * 0.7))
    chaotic_part_len = seq_len - break_point
    
    chaotic_part = [random.choice(CHAOTIC_EVENT_POOL) for _ in range(chaotic_part_len)]
    
    return sequence[:break_point] + chaotic_part

def main():
    """主函数"""
    print("🚀 开始生成合成日志序列数据...")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    all_sequences = []
    all_labels = []

    # 生成正常序列 (标签 0)
    print(f"   - 生成 {NUM_NORMAL_SEQUENCES} 条正常序列 (Class 0)...")
    for _ in tqdm(range(NUM_NORMAL_SEQUENCES)):
        all_sequences.append(generate_normal_sequence())
        all_labels.append(0)

    # 生成异常类型1序列 (标签 1)
    print(f"   - 生成 {NUM_ANOMALY_SEQUENCES_PER_TYPE} 条'事件侵入'异常序列 (Class 1)...")
    for _ in tqdm(range(NUM_ANOMALY_SEQUENCES_PER_TYPE)):
        all_sequences.append(generate_spike_anomaly())
        all_labels.append(1)

    # 生成异常类型2序列 (标签 2)
    print(f"   - 生成 {NUM_ANOMALY_SEQUENCES_PER_TYPE} 条'事件重复'异常序列 (Class 2)...")
    for _ in tqdm(range(NUM_ANOMALY_SEQUENCES_PER_TYPE)):
        all_sequences.append(generate_repetition_anomaly())
        all_labels.append(2)

    # 生成异常类型3序列 (标签 3)
    print(f"   - 生成 {NUM_ANOMALY_SEQUENCES_PER_TYPE} 条'序列中断'异常序列 (Class 3)...")
    for _ in tqdm(range(NUM_ANOMALY_SEQUENCES_PER_TYPE)):
        all_sequences.append(generate_chaotic_anomaly())
        all_labels.append(3)

    # 打乱数据
    print("   - 打乱数据...")
    combined = list(zip(all_sequences, all_labels))
    random.shuffle(combined)
    all_sequences, all_labels = zip(*combined)

    # 写入数据文件
    print(f"   - 写入序列数据到 {TRAIN_DATA_FILE}...")
    with open(TRAIN_DATA_FILE, 'w') as f:
        for seq in all_sequences:
            f.write(' '.join(map(str, seq)) + '\n')

    # 写入标签文件
    print(f"   - 写入标签数据到 {TRAIN_LABEL_FILE}...")
    with open(TRAIN_LABEL_FILE, 'w') as f:
        for label in all_labels:
            f.write(str(label) + '\n')
            
    total_sequences = len(all_sequences)
    print(f"\n✅ 数据生成完成！共生成 {total_sequences} 条序列。")
    print(f"   - 正常: {NUM_NORMAL_SEQUENCES}")
    print(f"   - 异常类型1: {NUM_ANOMALY_SEQUENCES_PER_TYPE}")
    print(f"   - 异常类型2: {NUM_ANOMALY_SEQUENCES_PER_TYPE}")
    print(f"   - 异常类型3: {NUM_ANOMALY_SEQUENCES_PER_TYPE}")

if __name__ == "__main__":
    main() 