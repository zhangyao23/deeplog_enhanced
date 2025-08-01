#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本处理模块
负责将原始文本日志转换为DeepLog所需的数字序列格式
"""

import re
import json
import logging
from typing import List, Dict, Tuple, Any
from collections import defaultdict, Counter
import numpy as np

class TextProcessor:
    """文本日志处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化文本处理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.text_config = config.get('text_config', {})
        
        # 模板存储
        self.templates = {}
        self.template_to_id = {}
        self.id_to_template = {}
        self.next_template_id = 0
        
        # 统计信息
        self.stats = {
            'total_logs': 0,
            'parsed_logs': 0,
            'unique_templates': 0,
            'parse_rate': 0.0
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def parse_log_line(self, log_line: str) -> str:
        """
        解析单行日志，提取模板
        
        Args:
            log_line: 原始日志行
            
        Returns:
            解析后的模板
        """
        # 移除时间戳
        timestamp_pattern = r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}'
        log_line = re.sub(timestamp_pattern, '<timestamp>', log_line)
        
        # 移除IP地址
        ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        log_line = re.sub(ip_pattern, '<ip>', log_line)
        
        # 移除数字ID
        id_pattern = r'\b\d{5,}\b'
        log_line = re.sub(id_pattern, '<id>', log_line)
        
        # 移除引号内的内容
        quote_pattern = r'"[^"]*"'
        log_line = re.sub(quote_pattern, '<quoted>', log_line)
        
        # 移除路径
        path_pattern = r'/[^\s]*'
        log_line = re.sub(path_pattern, '<path>', log_line)
        
        # 移除邮箱
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        log_line = re.sub(email_pattern, '<email>', log_line)
        
        # 移除URL
        url_pattern = r'https?://[^\s]+'
        log_line = re.sub(url_pattern, '<url>', log_line)
        
        # 移除UUID
        uuid_pattern = r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b'
        log_line = re.sub(uuid_pattern, '<uuid>', log_line)
        
        # 清理多余空格
        log_line = re.sub(r'\s+', ' ', log_line).strip()
        
        return log_line
    
    def extract_templates(self, log_lines: List[str]) -> Dict[str, int]:
        """
        从日志行中提取模板并分配ID
        
        Args:
            log_lines: 日志行列表
            
        Returns:
            模板到ID的映射字典
        """
        self.logger.info("开始提取日志模板...")
        
        template_counts = Counter()
        
        for line in log_lines:
            self.stats['total_logs'] += 1
            
            # 解析日志行
            template = self.parse_log_line(line)
            template_counts[template] += 1
            self.stats['parsed_logs'] += 1
        
        # 为模板分配ID
        for template, count in template_counts.most_common():
            if template not in self.template_to_id:
                self.template_to_id[template] = self.next_template_id
                self.id_to_template[self.next_template_id] = template
                self.next_template_id += 1
        
        self.stats['unique_templates'] = len(self.template_to_id)
        self.stats['parse_rate'] = self.stats['parsed_logs'] / self.stats['total_logs']
        
        self.logger.info(f"提取了 {self.stats['unique_templates']} 个唯一模板")
        self.logger.info(f"解析率: {self.stats['parse_rate']:.2%}")
        
        return self.template_to_id
    
    def convert_to_sequences(self, log_lines: List[str], session_size: int = 50) -> List[List[int]]:
        """
        将日志行转换为数字序列
        
        Args:
            log_lines: 日志行列表
            session_size: 会话大小（每个序列的最大长度）
            
        Returns:
            数字序列列表
        """
        self.logger.info("开始转换为数字序列...")
        
        # 首先提取模板
        if not self.template_to_id:
            self.extract_templates(log_lines)
        
        # 转换为数字序列
        sequences = []
        current_sequence = []
        
        for line in log_lines:
            template = self.parse_log_line(line)
            template_id = self.template_to_id.get(template, 0)
            
            current_sequence.append(template_id)
            
            # 当序列达到指定长度时，保存并开始新序列
            if len(current_sequence) >= session_size:
                sequences.append(current_sequence[:session_size])
                current_sequence = []
        
        # 保存最后一个不完整的序列
        if current_sequence:
            sequences.append(current_sequence)
        
        self.logger.info(f"生成了 {len(sequences)} 个序列")
        return sequences
    
    def save_template_mapping(self, file_path: str):
        """
        保存模板映射
        
        Args:
            file_path: 保存路径
        """
        mapping_data = {
            'template_to_id': self.template_to_id,
            'id_to_template': self.id_to_template,
            'stats': self.stats
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"模板映射已保存到: {file_path}")
    
    def load_template_mapping(self, file_path: str):
        """
        加载模板映射
        
        Args:
            file_path: 加载路径
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
        
        self.template_to_id = mapping_data['template_to_id']
        self.id_to_template = mapping_data['id_to_template']
        self.stats = mapping_data['stats']
        self.next_template_id = len(self.template_to_id)
        
        self.logger.info(f"模板映射已从 {file_path} 加载")
    
    def process_text_file(self, input_file: str, output_file: str, session_size: int = 50) -> List[List[int]]:
        """
        处理文本文件并生成序列文件
        
        Args:
            input_file: 输入文本文件路径
            output_file: 输出序列文件路径
            session_size: 会话大小
            
        Returns:
            生成的序列列表
        """
        self.logger.info(f"处理文本文件: {input_file}")
        
        # 读取文本文件
        with open(input_file, 'r', encoding='utf-8') as f:
            log_lines = [line.strip() for line in f if line.strip()]
        
        # 转换为序列
        sequences = self.convert_to_sequences(log_lines, session_size)
        
        # 保存序列文件
        with open(output_file, 'w', encoding='utf-8') as f:
            for sequence in sequences:
                f.write(' '.join(map(str, sequence)) + '\n')
        
        self.logger.info(f"序列文件已保存到: {output_file}")
        return sequences
    
    def get_template_info(self) -> Dict[str, Any]:
        """
        获取模板信息
        
        Returns:
            模板信息字典
        """
        return {
            'total_templates': len(self.template_to_id),
            'stats': self.stats,
            'sample_templates': dict(list(self.template_to_id.items())[:10])
        } 