#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于Log Parser的文本处理模块
使用专业的日志解析工具将原始文本日志转换为DeepLog所需的数字序列格式
"""

import os
import sys
import json
import logging
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict, Counter
import numpy as np

# 尝试导入Log Parser相关库
try:
    from drain3 import TemplateMiner
    from drain3.template_miner_config import TemplateMinerConfig
    LOGPARSER_AVAILABLE = True
except ImportError:
    LOGPARSER_AVAILABLE = False
    print("警告: Log Parser库未安装，将使用基础文本处理")

class LogParserProcessor:
    """基于Log Parser的专业日志处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化日志解析器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.parser_config = config.get('parser_config', {})
        
        # 模板存储
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
        
        # 初始化Log Parser
        self.template_miner = None
        self._init_parser()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _init_parser(self):
        """初始化Log Parser"""
        if not LOGPARSER_AVAILABLE:
            self.logger.warning("Log Parser不可用，将使用基础文本处理")
            return
        
        try:
            # 配置Drain3模板挖掘器
            config = TemplateMinerConfig()
            config.load('drain3.ini')
            
            # 自定义配置
            config.drain_depth = self.parser_config.get('drain_depth', 4)
            config.drain_max_children = self.parser_config.get('drain_max_children', 100)
            config.drain_sim_threshold = self.parser_config.get('drain_sim_threshold', 0.4)
            config.drain_max_clusters = self.parser_config.get('drain_max_clusters', 1000)
            
            self.template_miner = TemplateMiner(config)
            self.logger.info("Log Parser初始化成功")
            
        except Exception as e:
            self.logger.error(f"Log Parser初始化失败: {e}")
            self.template_miner = None
    
    def parse_log_line_advanced(self, log_line: str) -> Tuple[str, Dict[str, Any]]:
        """
        使用Log Parser解析单行日志
        
        Args:
            log_line: 原始日志行
            
        Returns:
            (模板, 参数信息)
        """
        if self.template_miner is None:
            # 回退到基础解析
            return self._basic_parse(log_line), {}
        
        try:
            # 使用Drain3解析
            result = self.template_miner.add_log_message(log_line)
            
            if result:
                template = result['template_mined']
                parameters = result.get('parameter_list', [])
                
                # 提取参数信息
                param_info = {
                    'count': len(parameters),
                    'types': [type(p).__name__ for p in parameters]
                }
                
                return template, param_info
            else:
                # 解析失败，使用原始行作为模板
                return log_line, {}
                
        except Exception as e:
            self.logger.warning(f"Log Parser解析失败: {e}")
            return self._basic_parse(log_line), {}
    
    def _basic_parse(self, log_line: str) -> str:
        """
        基础文本解析（回退方案）
        
        Args:
            log_line: 原始日志行
            
        Returns:
            解析后的模板
        """
        import re
        
        # 移除时间戳
        timestamp_patterns = [
            r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}',
            r'\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}',
            r'\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2}:\d{2}'
        ]
        
        for pattern in timestamp_patterns:
            log_line = re.sub(pattern, '<timestamp>', log_line)
        
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
    
    def extract_templates_advanced(self, log_lines: List[str]) -> Dict[str, int]:
        """
        使用Log Parser提取模板并分配ID
        
        Args:
            log_lines: 日志行列表
            
        Returns:
            模板到ID的映射字典
        """
        self.logger.info("开始使用Log Parser提取日志模板...")
        
        template_counts = Counter()
        template_params = defaultdict(list)
        
        for line in log_lines:
            self.stats['total_logs'] += 1
            
            # 解析日志行
            template, param_info = self.parse_log_line_advanced(line)
            template_counts[template] += 1
            template_params[template].append(param_info)
            self.stats['parsed_logs'] += 1
        
        # 为模板分配ID（按频率排序）
        for template, count in template_counts.most_common():
            if template not in self.template_to_id:
                self.template_to_id[template] = self.next_template_id
                self.id_to_template[self.next_template_id] = template
                self.next_template_id += 1
        
        self.stats['unique_templates'] = len(self.template_to_id)
        self.stats['parse_rate'] = self.stats['parsed_logs'] / self.stats['total_logs']
        
        self.logger.info(f"提取了 {self.stats['unique_templates']} 个唯一模板")
        self.logger.info(f"解析率: {self.stats['parse_rate']:.2%}")
        
        # 保存模板参数统计
        self.template_params = dict(template_params)
        
        return self.template_to_id
    
    def convert_to_sequences_advanced(self, log_lines: List[str], session_size: int = 50) -> List[List[int]]:
        """
        将日志行转换为数字序列（使用Log Parser）
        
        Args:
            log_lines: 日志行列表
            session_size: 会话大小（每个序列的最大长度）
            
        Returns:
            数字序列列表
        """
        self.logger.info("开始转换为数字序列...")
        
        # 首先提取模板
        if not self.template_to_id:
            self.extract_templates_advanced(log_lines)
        
        # 转换为数字序列
        sequences = []
        current_sequence = []
        
        for line in log_lines:
            template, _ = self.parse_log_line_advanced(line)
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
    
    def process_text_file_advanced(self, input_file: str, output_file: str, session_size: int = 50) -> List[List[int]]:
        """
        使用Log Parser处理文本文件并生成序列文件
        
        Args:
            input_file: 输入文本文件路径
            output_file: 输出序列文件路径
            session_size: 会话大小
            
        Returns:
            生成的序列列表
        """
        self.logger.info(f"使用Log Parser处理文本文件: {input_file}")
        
        # 读取文本文件
        with open(input_file, 'r', encoding='utf-8') as f:
            log_lines = [line.strip() for line in f if line.strip()]
        
        # 转换为序列
        sequences = self.convert_to_sequences_advanced(log_lines, session_size)
        
        # 保存序列文件
        with open(output_file, 'w', encoding='utf-8') as f:
            for sequence in sequences:
                f.write(' '.join(map(str, sequence)) + '\n')
        
        self.logger.info(f"序列文件已保存到: {output_file}")
        return sequences
    
    def save_template_mapping_advanced(self, file_path: str):
        """
        保存模板映射（包含Log Parser信息）
        
        Args:
            file_path: 保存路径
        """
        mapping_data = {
            'template_to_id': self.template_to_id,
            'id_to_template': self.id_to_template,
            'stats': self.stats,
            'parser_info': {
                'logparser_available': LOGPARSER_AVAILABLE,
                'template_miner_initialized': self.template_miner is not None,
                'parser_config': self.parser_config
            }
        }
        
        # 添加模板参数统计（如果可用）
        if hasattr(self, 'template_params'):
            mapping_data['template_params'] = {
                template: {
                    'count': len(params),
                    'avg_param_count': np.mean([p.get('count', 0) for p in params]) if params else 0
                }
                for template, params in self.template_params.items()
            }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"模板映射已保存到: {file_path}")
    
    def get_template_info_advanced(self) -> Dict[str, Any]:
        """
        获取模板信息（包含Log Parser统计）
        
        Returns:
            模板信息字典
        """
        info = {
            'total_templates': len(self.template_to_id),
            'stats': self.stats,
            'parser_available': LOGPARSER_AVAILABLE,
            'sample_templates': dict(list(self.template_to_id.items())[:10])
        }
        
        # 添加模板质量统计
        if hasattr(self, 'template_params'):
            template_quality = {}
            for template, params in self.template_params.items():
                if params:
                    avg_params = np.mean([p.get('count', 0) for p in params])
                    template_quality[template] = {
                        'frequency': len(params),
                        'avg_parameters': avg_params
                    }
            
            info['template_quality'] = dict(list(template_quality.items())[:10])
        
        return info
    
    def create_drain3_config(self, config_path: str = 'drain3.ini'):
        """
        创建Drain3配置文件
        
        Args:
            config_path: 配置文件路径
        """
        config_content = """[DRAIN]
drain_depth = 4
drain_max_children = 100
drain_sim_threshold = 0.4
drain_max_clusters = 1000
drain_extra_delimiters = []
drain_remove_colon = true
drain_remove_punctuation = false
drain_remove_numbers = false
drain_remove_hex = false
drain_remove_ip = false
drain_remove_uuid = false
drain_remove_email = false
drain_remove_url = false
drain_remove_path = false
drain_remove_quotes = false
drain_remove_brackets = false
drain_remove_parentheses = false
drain_remove_braces = false
drain_remove_angle_brackets = false
drain_remove_square_brackets = false
drain_remove_curly_braces = false
drain_remove_round_brackets = false
drain_remove_hex_prefix = false
drain_remove_uuid_prefix = false
drain_remove_email_prefix = false
drain_remove_url_prefix = false
drain_remove_path_prefix = false
drain_remove_quotes_prefix = false
drain_remove_brackets_prefix = false
drain_remove_parentheses_prefix = false
drain_remove_braces_prefix = false
drain_remove_angle_brackets_prefix = false
drain_remove_square_brackets_prefix = false
drain_remove_curly_braces_prefix = false
drain_remove_round_brackets_prefix = false
drain_remove_hex_suffix = false
drain_remove_uuid_suffix = false
drain_remove_email_suffix = false
drain_remove_url_suffix = false
drain_remove_path_suffix = false
drain_remove_quotes_suffix = false
drain_remove_brackets_suffix = false
drain_remove_parentheses_suffix = false
drain_remove_braces_suffix = false
drain_remove_angle_brackets_suffix = false
drain_remove_square_brackets_suffix = false
drain_remove_curly_braces_suffix = false
drain_remove_round_brackets_suffix = false
"""
        
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        self.logger.info(f"Drain3配置文件已创建: {config_path}") 