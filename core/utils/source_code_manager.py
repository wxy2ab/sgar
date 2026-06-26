#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
源代码管理器 - 维护行号映射的代码编辑工具

核心设计理念：
1. 每一行都维护原始行号(source_line_number)
2. 无论如何编辑，始终能通过原始行号定位
3. 支持乱序应用任意编辑指令
4. 应用成功的指令不需要重复应用

适用场景：
- LLM代码编辑器（行级编辑和块级编辑）
- 代码重构工具
- 版本控制系统
- 任何需要精确行号定位的代码处理场景
"""

from typing import List, Optional


class LineCode:
    """表示代码中的一行，维护原始行号和当前内容"""
    
    def __init__(self, source_line_number: int, content: str):
        """
        初始化一行代码
        
        Args:
            source_line_number: 原始行号（从1开始）
            content: 代码内容
        """
        self.source_line_number = source_line_number
        self.content = content
    
    def __repr__(self):
        """返回可读的字符串表示"""
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Line({self.source_line_number}: {content_preview})"
    
    def __eq__(self, other):
        """比较两个LineCode对象是否相等"""
        if not isinstance(other, LineCode):
            return False
        return (self.source_line_number == other.source_line_number and 
                self.content == other.content)


class SourceCode:
    """
    源代码管理类 - 维护行号映射，支持乱序应用编辑指令
    
    核心设计：
    1. 每一行都维护原始行号(source_line_number)
    2. 无论如何编辑，始终能通过原始行号定位
    3. 支持乱序应用任意编辑指令
    4. 应用成功的指令不需要重复应用
    
    示例用法：
        >>> source = SourceCode("line1\\nline2\\nline3")
        >>> source.replace_line(2, "modified line2")
        >>> source.insert_after(1, "new line after line1")
        >>> source.delete_line(3)
        >>> code = source.get_code()
    """
    
    def __init__(self, code: str):
        """
        初始化源代码管理器
        
        Args:
            code: 原始代码字符串
        """
        self.lines: List[LineCode] = []
        
        # 解析代码行
        for i, line in enumerate(code.split('\n'), 1):
            self.lines.append(LineCode(source_line_number=i, content=line))
    
    def get_code(self) -> str:
        """
        获取当前代码
        
        Returns:
            代码字符串
        """
        return '\n'.join(line.content for line in self.lines)
    
    def get_original_line_count(self) -> int:
        """
        获取原始代码的行数
        
        Returns:
            原始行数
        """
        if not self.lines:
            return 0
        return max(line.source_line_number for line in self.lines)
    
    def get_current_line_count(self) -> int:
        """
        获取当前代码的行数
        
        Returns:
            当前行数
        """
        return len(self.lines)
    
    def find_line_by_source(self, source_line: int) -> Optional[int]:
        """
        根据原始行号查找当前索引
        
        Args:
            source_line: 原始行号
            
        Returns:
            当前索引，如果未找到返回None
        """
        for idx, line in enumerate(self.lines):
            if line.source_line_number == source_line:
                return idx
        return None
    
    def find_lines_by_source_range(self, start: int, end: int) -> List[int]:
        """
        根据原始行号范围查找当前索引列表
        
        Args:
            start: 起始原始行号
            end: 结束原始行号
            
        Returns:
            当前索引列表
        """
        indices = []
        for idx, line in enumerate(self.lines):
            if start <= line.source_line_number <= end:
                indices.append(idx)
        return indices
    
    # ==================== 单行操作 ====================
    
    def replace_line(self, source_line: int, new_content: str):
        """
        替换指定原始行号的内容（单行）
        
        Args:
            source_line: 原始行号
            new_content: 新内容
            
        Raises:
            ValueError: 如果未找到指定行号
        """
        idx = self.find_line_by_source(source_line)
        if idx is None:
            raise ValueError(f"未找到原始行号 {source_line}")
        
        # 替换内容，保持原始行号
        self.lines[idx].content = new_content
    
    def delete_line(self, source_line: int):
        """
        删除指定原始行号的行（单行）
        
        Args:
            source_line: 原始行号
            
        Raises:
            ValueError: 如果未找到指定行号
        """
        idx = self.find_line_by_source(source_line)
        if idx is None:
            raise ValueError(f"未找到原始行号 {source_line}")
        
        self.lines.pop(idx)
    
    def insert_after(self, source_line: int, content: str):
        """
        在指定原始行号后插入内容
        
        Args:
            source_line: 原始行号
            content: 要插入的内容
            
        Raises:
            ValueError: 如果未找到指定行号
        """
        idx = self.find_line_by_source(source_line)
        if idx is None:
            raise ValueError(f"未找到原始行号 {source_line}")
        
        # 插入新行（使用相同的原始行号）
        new_lines = content.split('\n')
        for i, line_content in enumerate(new_lines, 1):
            self.lines.insert(idx + i, LineCode(source_line, line_content))
    
    def insert_before(self, source_line: int, content: str):
        """
        在指定原始行号前插入内容
        
        Args:
            source_line: 原始行号
            content: 要插入的内容
            
        Raises:
            ValueError: 如果未找到指定行号
        """
        idx = self.find_line_by_source(source_line)
        if idx is None:
            raise ValueError(f"未找到原始行号 {source_line}")
        
        # 插入新行（使用相同的原始行号）
        new_lines = content.split('\n')
        for i, line_content in enumerate(new_lines):
            self.lines.insert(idx + i, LineCode(source_line, line_content))
    
    # ==================== 范围操作（用于块级编辑） ====================
    
    def replace_lines(self, start: int, end: int, new_content: str):
        """
        替换指定原始行号范围的内容（多行）
        
        Args:
            start: 起始原始行号
            end: 结束原始行号
            new_content: 新内容
            
        Raises:
            ValueError: 如果未找到指定行号范围
        """
        # 找到要替换的行索引
        indices = self.find_lines_by_source_range(start, end)
        if not indices:
            raise ValueError(f"未找到原始行号范围 {start}:{end}")
        
        # 使用第一行的原始行号作为新内容的行号
        source_line_number = self.lines[indices[0]].source_line_number
        
        # 删除旧行（从后往前删除以保持索引有效）
        for idx in reversed(indices):
            self.lines.pop(idx)
        
        # 插入新行（所有新行使用相同的原始行号）
        new_lines = new_content.split('\n')
        for i, content in enumerate(new_lines):
            self.lines.insert(indices[0] + i, LineCode(source_line_number, content))
    
    def delete_lines(self, start: int, end: int):
        """
        删除指定原始行号范围的行（多行）
        
        Args:
            start: 起始原始行号
            end: 结束原始行号
            
        Raises:
            ValueError: 如果未找到指定行号范围
        """
        indices = self.find_lines_by_source_range(start, end)
        if not indices:
            raise ValueError(f"未找到原始行号范围 {start}:{end}")
        
        # 从后往前删除
        for idx in reversed(indices):
            self.lines.pop(idx)
    
    # ==================== 通用操作 ====================
    
    def apply_edit_instruction(self, instruction) -> List[str]:
        """
        应用单个编辑指令（行级编辑）
        
        Args:
            instruction: 编辑指令对象，需要有 line_number, modifier, content 属性
            
        Returns:
            错误列表（如果有）
        """
        try:
            if instruction.modifier == 'r':  # replace
                self.replace_line(instruction.line_number, instruction.content)
            elif instruction.modifier == '-':  # delete
                self.delete_line(instruction.line_number)
            elif instruction.modifier == '+':  # add after
                self.insert_after(instruction.line_number, instruction.content)
            else:
                return [f"未知的指令类型: {instruction.modifier}"]
            
            return []
        except Exception as e:
            return [f"应用指令失败: {e}"]
    
    def apply_block_instruction(self, instruction) -> List[str]:
        """
        应用单个块编辑指令（块级编辑）
        
        Args:
            instruction: 块编辑指令对象，需要有 type, start_line, end_line, content 属性
            
        Returns:
            错误列表（如果有）
        """
        try:
            if instruction.type == 'replace':
                self.replace_lines(instruction.start_line, instruction.end_line, instruction.content)
            elif instruction.type == 'delete':
                self.delete_lines(instruction.start_line, instruction.end_line)
            elif instruction.type == 'insert_after':
                self.insert_after(instruction.start_line, instruction.content)
            elif instruction.type == 'insert_before':
                self.insert_before(instruction.start_line, instruction.content)
            else:
                return [f"未知的指令类型: {instruction.type}"]
            
            return []
        except Exception as e:
            return [f"应用指令失败: {e}"]
    
    def apply_instruction(self, instruction) -> List[str]:
        """
        应用指令（通用接口，自动判断指令类型）
        
        兼容性方法：根据指令对象的属性自动判断是行级编辑还是块级编辑
        
        Args:
            instruction: 编辑指令对象
            
        Returns:
            错误列表（如果有）
        """
        # 判断指令类型
        if hasattr(instruction, 'type'):
            # 块级编辑指令（有 type 属性）
            return self.apply_block_instruction(instruction)
        elif hasattr(instruction, 'modifier'):
            # 行级编辑指令（有 modifier 属性）
            return self.apply_edit_instruction(instruction)
        else:
            return [f"无法识别的指令类型: {type(instruction)}"]
    
    def clone(self) -> 'SourceCode':
        """
        克隆一份副本
        
        Returns:
            新的SourceCode对象
        """
        new_source = SourceCode("")
        new_source.lines = [LineCode(line.source_line_number, line.content) for line in self.lines]
        return new_source
    
    def __len__(self):
        """返回当前行数"""
        return len(self.lines)
    
    def __repr__(self):
        """返回可读的字符串表示"""
        return f"SourceCode({len(self.lines)} lines, original: {self.get_original_line_count()} lines)"


__all__ = ['LineCode', 'SourceCode']

