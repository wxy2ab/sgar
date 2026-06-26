#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
错误分类辅助模块 - 统一的错误处理和分类

提供错误分类、重试决策、错误恢复等功能
用于整个项目的错误处理标准化
"""

import re
from typing import List, Dict, Tuple, Optional
from enum import Enum
from dataclasses import dataclass
from core.utils.log import logger


class ErrorType(Enum):
    """错误类型枚举"""
    IGNORABLE = "ignorable"  # 可忽略的错误
    RETRYABLE = "retryable"  # 需要重试的错误
    CRITICAL = "critical"  # 严重错误，无法恢复
    UNKNOWN = "unknown"  # 未知错误


@dataclass
class ErrorClassification:
    """错误分类结果"""
    error_type: ErrorType
    error_message: str
    should_retry: bool
    retry_strategy: Optional[str] = None  # 重试策略: 'immediate', 'exponential_backoff', 'llm_fix'
    confidence: float = 1.0  # 分类置信度 0-1


class UnifiedErrorClassifier:
    """统一的错误分类器"""
    
    # 可忽略的错误模式
    IGNORABLE_PATTERNS = [
        # 格式和风格问题
        (r"行号.*超出范围", 0.8),
        (r"代码长度变化", 0.9),
        (r"缩进风格", 0.7),
        (r"注释.*缺失", 0.9),
        (r"空行.*过多", 0.9),
        (r"变量名.*不规范", 0.6),
        
        # 轻微的结构问题
        (r"方法顺序.*变化", 0.7),
        (r"导入语句.*顺序", 0.8),
    ]
    
    # 需要重试的错误模式
    RETRYABLE_PATTERNS = [
        # 语法错误
        (r"SyntaxError", 0.9, 'llm_fix'),
        (r"语法错误", 0.9, 'llm_fix'),
        (r"IndentationError", 0.9, 'llm_fix'),
        (r"缩进错误", 0.9, 'llm_fix'),
        
        # 运行时错误
        (r"NameError", 0.8, 'llm_fix'),
        (r"未定义.*变量", 0.8, 'llm_fix'),
        (r"ImportError", 0.8, 'llm_fix'),
        (r"导入.*失败", 0.8, 'llm_fix'),
        (r"ModuleNotFoundError", 0.8, 'llm_fix'),
        
        # 网络和超时错误
        (r"timeout", 0.9, 'exponential_backoff'),
        (r"超时", 0.9, 'exponential_backoff'),
        (r"连接.*失败", 0.9, 'exponential_backoff'),
        (r"connection.*error", 0.9, 'exponential_backoff'),
        (r"网络.*错误", 0.9, 'exponential_backoff'),
        
        # LLM相关错误
        (r"LLM.*返回.*空", 0.8, 'immediate'),
        (r"LLM.*响应.*无效", 0.8, 'immediate'),
        (r"LLM.*格式.*错误", 0.7, 'immediate'),
    ]
    
    # 严重错误模式（无法恢复）
    CRITICAL_PATTERNS = [
        (r"循环引用", 0.9),
        (r"内存.*不足", 0.95),
        (r"MemoryError", 0.95),
        (r"核心.*类.*缺失", 0.9),
        (r"必要.*函数.*删除", 0.9),
        (r"文件.*不存在.*无法创建", 0.85),
    ]
    
    @classmethod
    def classify_error(cls, error_message: str) -> ErrorClassification:
        """
        分类单个错误
        
        Args:
            error_message: 错误消息
            
        Returns:
            ErrorClassification对象
        """
        # 检查是否是严重错误
        for pattern, confidence in cls.CRITICAL_PATTERNS:
            if re.search(pattern, error_message, re.IGNORECASE):
                return ErrorClassification(
                    error_type=ErrorType.CRITICAL,
                    error_message=error_message,
                    should_retry=False,
                    confidence=confidence
                )
        
        # 检查是否需要重试
        for item in cls.RETRYABLE_PATTERNS:
            pattern = item[0]
            confidence = item[1]
            strategy = item[2] if len(item) > 2 else 'immediate'
            
            if re.search(pattern, error_message, re.IGNORECASE):
                return ErrorClassification(
                    error_type=ErrorType.RETRYABLE,
                    error_message=error_message,
                    should_retry=True,
                    retry_strategy=strategy,
                    confidence=confidence
                )
        
        # 检查是否可忽略
        for pattern, confidence in cls.IGNORABLE_PATTERNS:
            if re.search(pattern, error_message, re.IGNORECASE):
                return ErrorClassification(
                    error_type=ErrorType.IGNORABLE,
                    error_message=error_message,
                    should_retry=False,
                    confidence=confidence
                )
        
        # 未知错误，保守起见标记为需要重试
        return ErrorClassification(
            error_type=ErrorType.UNKNOWN,
            error_message=error_message,
            should_retry=True,
            retry_strategy='immediate',
            confidence=0.5
        )
    
    @classmethod
    def classify_errors(cls, errors: List[str]) -> Dict[str, List[ErrorClassification]]:
        """
        批量分类错误
        
        Args:
            errors: 错误列表
            
        Returns:
            按类型分组的错误分类结果
        """
        classified = {
            'ignorable': [],
            'retryable': [],
            'critical': [],
            'unknown': []
        }
        
        for error in errors:
            classification = cls.classify_error(error)
            
            if classification.error_type == ErrorType.IGNORABLE:
                classified['ignorable'].append(classification)
            elif classification.error_type == ErrorType.RETRYABLE:
                classified['retryable'].append(classification)
            elif classification.error_type == ErrorType.CRITICAL:
                classified['critical'].append(classification)
            else:
                classified['unknown'].append(classification)
        
        return classified
    
    @classmethod
    def should_retry(cls, errors: List[str]) -> Tuple[bool, Optional[str]]:
        """
        判断是否应该重试
        
        Args:
            errors: 错误列表
            
        Returns:
            (是否应该重试, 推荐的重试策略)
        """
        classified = cls.classify_errors(errors)
        
        # 如果有严重错误，不重试
        if classified['critical']:
            logger.error(f"检测到 {len(classified['critical'])} 个严重错误，无法重试")
            return False, None
        
        # 如果有需要重试的错误，使用最适合的策略
        if classified['retryable']:
            # 统计每种策略的数量
            strategies = {}
            for classification in classified['retryable']:
                strategy = classification.retry_strategy
                strategies[strategy] = strategies.get(strategy, 0) + 1
            
            # 选择最常见的策略
            best_strategy = max(strategies.items(), key=lambda x: x[1])[0]
            
            logger.info(f"检测到 {len(classified['retryable'])} 个可重试错误，策略: {best_strategy}")
            return True, best_strategy
        
        # 如果只有可忽略的错误，不重试
        if classified['ignorable'] and not classified['retryable'] and not classified['unknown']:
            logger.info(f"只有 {len(classified['ignorable'])} 个可忽略错误，无需重试")
            return False, None
        
        # 如果有未知错误，保守起见重试
        if classified['unknown']:
            logger.warning(f"检测到 {len(classified['unknown'])} 个未知错误，保守重试")
            return True, 'immediate'
        
        return False, None
    
    @classmethod
    def get_retry_delay(cls, attempt: int, strategy: str) -> float:
        """
        获取重试延迟时间
        
        Args:
            attempt: 当前尝试次数（从0开始）
            strategy: 重试策略
            
        Returns:
            延迟秒数
        """
        if strategy == 'immediate':
            return 0.5
        elif strategy == 'exponential_backoff':
            return min(2 ** attempt, 30)  # 最多30秒
        elif strategy == 'llm_fix':
            return 1.0  # LLM修复需要一点时间
        else:
            return 1.0
    
    @classmethod
    def format_error_report(cls, errors: List[str]) -> str:
        """
        格式化错误报告
        
        Args:
            errors: 错误列表
            
        Returns:
            格式化的错误报告
        """
        if not errors:
            return "无错误"
        
        classified = cls.classify_errors(errors)
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("错误分类报告")
        report_lines.append("=" * 60)
        
        # 严重错误
        if classified['critical']:
            report_lines.append(f"\n❌ 严重错误 ({len(classified['critical'])} 个):")
            for i, classification in enumerate(classified['critical'], 1):
                report_lines.append(f"  {i}. {classification.error_message}")
                report_lines.append(f"     置信度: {classification.confidence:.1%}")
        
        # 可重试错误
        if classified['retryable']:
            report_lines.append(f"\n🔄 可重试错误 ({len(classified['retryable'])} 个):")
            for i, classification in enumerate(classified['retryable'], 1):
                report_lines.append(f"  {i}. {classification.error_message}")
                report_lines.append(f"     策略: {classification.retry_strategy}, 置信度: {classification.confidence:.1%}")
        
        # 可忽略错误
        if classified['ignorable']:
            report_lines.append(f"\n✓ 可忽略错误 ({len(classified['ignorable'])} 个):")
            for i, classification in enumerate(classified['ignorable'], 1):
                report_lines.append(f"  {i}. {classification.error_message}")
        
        # 未知错误
        if classified['unknown']:
            report_lines.append(f"\n❓ 未知错误 ({len(classified['unknown'])} 个):")
            for i, classification in enumerate(classified['unknown'], 1):
                report_lines.append(f"  {i}. {classification.error_message}")
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)


def classify_and_handle_errors(errors: List[str]) -> Dict[str, any]:
    """
    分类并处理错误的便捷函数
    
    Args:
        errors: 错误列表
        
    Returns:
        处理结果字典
    """
    should_retry, strategy = UnifiedErrorClassifier.should_retry(errors)
    classified = UnifiedErrorClassifier.classify_errors(errors)
    
    return {
        'should_retry': should_retry,
        'retry_strategy': strategy,
        'classified': classified,
        'report': UnifiedErrorClassifier.format_error_report(errors),
        'has_critical': len(classified['critical']) > 0,
        'has_retryable': len(classified['retryable']) > 0,
        'has_ignorable': len(classified['ignorable']) > 0,
    }


if __name__ == '__main__':
    """测试错误分类器"""
    
    test_errors = [
        "SyntaxError: invalid syntax on line 42",
        "行号 150 超出范围 (1-100)",
        "NameError: name 'undefined_var' is not defined",
        "timeout: connection timed out after 30s",
        "缩进风格不一致",
        "循环引用检测到",
        "注释缺失: 函数 calculate_alpha",
    ]
    
    print("测试错误分类器\n")
    
    for error in test_errors:
        classification = UnifiedErrorClassifier.classify_error(error)
        print(f"错误: {error}")
        print(f"  类型: {classification.error_type.value}")
        print(f"  重试: {classification.should_retry}")
        print(f"  策略: {classification.retry_strategy}")
        print(f"  置信度: {classification.confidence:.1%}")
        print()
    
    # 批量处理
    print("\n" + "=" * 60)
    print("批量处理测试")
    print("=" * 60)
    
    result = classify_and_handle_errors(test_errors)
    print(result['report'])
    print(f"\n应该重试: {result['should_retry']}")
    print(f"重试策略: {result['retry_strategy']}")

