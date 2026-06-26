#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于LLM的块编辑器（带行号）

.. deprecated::
    Line-number-based block editing was found to be unreliable in practice.
    Prefer :class:`core.utils.llm_block_editor_lnfree.LineNumberFreeLLMBlockEditor`
    (content-based locator) or :class:`core.utils.smart_llm_editor_v2.SmartLLMEditorV2`
    (SEARCH/REPLACE blocks). See ``core/utils/code_editor.md``.
"""

import re
import ast
import tempfile
import time
import warnings
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

# 导入logger，处理可能的导入错误
try:
    from core.utils.log import logger
except ImportError:
    # 如果无法导入，创建一个简单的logger
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

# 导入统一的源代码管理类
from core.utils.source_code_manager import LineCode, SourceCode


@dataclass
class BlockEditInstruction:
    """块编辑指令"""
    type: str  # 'replace', 'insert_after', 'insert_before', 'delete'
    start_line: int
    end_line: Optional[int]  # 对于insert/delete单行，end_line == start_line
    content: str  # 新代码块（replace/insert时使用）
    raw_text: str = ""  # 原始指令文本，用于调试
    
    def __str__(self):
        if self.type == 'replace':
            return f"REPLACE {self.start_line}:{self.end_line} ({len(self.content.split(chr(10)))}行)"
        elif self.type == 'insert_after':
            return f"INSERT AFTER {self.start_line} ({len(self.content.split(chr(10)))}行)"
        elif self.type == 'insert_before':
            return f"INSERT BEFORE {self.start_line} ({len(self.content.split(chr(10)))}行)"
        elif self.type == 'delete':
            return f"DELETE {self.start_line}:{self.end_line}"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，用于序列化"""
        return {
            'type': self.type,
            'start_line': self.start_line,
            'end_line': self.end_line,
            'content': self.content,
            'raw_text': self.raw_text
        }


@dataclass
class BlockEditResult:
    """块编辑结果"""
    success: bool
    new_code: str
    applied_instructions: List[BlockEditInstruction]
    errors: List[str]
    diff: str
    warnings: List[str]
    original_code: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，用于序列化"""
        return {
            'success': self.success,
            'new_code': self.new_code,
            'applied_instructions': [inst.to_dict() for inst in self.applied_instructions],
            'errors': self.errors,
            'warnings': self.warnings,
            'diff': self.diff,
            'original_code': self.original_code
        }


class ErrorClassifier:
    """错误分类器 - 区分可忽略错误和需要重试的错误"""
    
    # 可忽略的错误模式（轻微问题，不影响代码执行）
    IGNORABLE_PATTERNS = [
        r"行号.*超出范围",  # 行号错误（如果AST验证通过）
        r"代码长度变化",  # 代码长度轻微变化
        r"缩进风格",  # 缩进风格不一致
        r"注释.*缺失",  # 注释相关问题
    ]
    
    # 需要重试的错误模式（严重问题，必须修复）
    RETRY_PATTERNS = [
        r"语法错误",  # 语法错误
        r"SyntaxError",  # Python语法错误
        r"IndentationError",  # 缩进错误
        r"未找到函数",  # 缺少必要的函数
        r"导入.*失败",  # 导入错误
        r"循环引用",  # 循环引用
    ]
    
    @classmethod
    def classify_error(cls, error_msg: str) -> str:
        """
        分类错误类型
        
        Args:
            error_msg: 错误消息
            
        Returns:
            'ignorable' | 'retryable' | 'unknown'
        """
        # 检查是否是可忽略错误
        for pattern in cls.IGNORABLE_PATTERNS:
            if re.search(pattern, error_msg, re.IGNORECASE):
                return 'ignorable'
        
        # 检查是否需要重试
        for pattern in cls.RETRY_PATTERNS:
            if re.search(pattern, error_msg, re.IGNORECASE):
                return 'retryable'
        
        # 未知类型，默认需要重试
        return 'unknown'
    
    @classmethod
    def classify_errors(cls, errors: List[str]) -> Dict[str, List[str]]:
        """
        批量分类错误
        
        Args:
            errors: 错误列表
            
        Returns:
            {'ignorable': [...], 'retryable': [...], 'unknown': [...]}
        """
        classified = {
            'ignorable': [],
            'retryable': [],
            'unknown': []
        }
        
        for error in errors:
            error_type = cls.classify_error(error)
            classified[error_type].append(error)
        
        return classified
    
    @classmethod
    def should_retry(cls, errors: List[str]) -> bool:
        """
        判断是否应该重试
        
        Args:
            errors: 错误列表
            
        Returns:
            是否应该重试
        """
        classified = cls.classify_errors(errors)
        
        # 如果有需要重试的错误，则重试
        if classified['retryable']:
            return True
        
        # 如果只有可忽略的错误，不重试
        if classified['ignorable'] and not classified['retryable'] and not classified['unknown']:
            return False
        
        # 如果有未知错误，保守起见，重试
        if classified['unknown']:
            return True
        
        return False


class CodeStructureValidator:
    """代码结构验证器"""
    
    @staticmethod
    def validate_python_syntax(code: str) -> Tuple[bool, List[str]]:
        """
        验证Python语法
        
        Args:
            code: 要验证的代码
            
        Returns:
            (是否有效, 错误列表)
        """
        errors = []
        try:
            ast.parse(code)
            return True, errors
        except SyntaxError as e:
            error_msg = f"语法错误 (行{e.lineno}:{e.offset}): {e.msg}"
            errors.append(error_msg)
            return False, errors
    
    @staticmethod
    def check_structure_integrity(original_code: str, new_code: str) -> Tuple[bool, List[str]]:
        """
        检查代码结构完整性
        
        Args:
            original_code: 原始代码
            new_code: 新代码
            
        Returns:
            (是否完整, 警告列表)
        """
        warnings = []
        
        # 检查括号平衡
        def check_brackets_balance(code):
            brackets = {'(': 0, '[': 0, '{': 0}
            brackets_close = {')': '(', ']': '[', '}': '{'}
            stack = []
            
            for i, char in enumerate(code):
                if char in brackets:
                    brackets[char] += 1
                    stack.append(char)
                elif char in brackets_close:
                    if not stack or stack[-1] != brackets_close[char]:
                        return False, f"括号不匹配，位置 {i}"
                    stack.pop()
                    brackets[brackets_close[char]] -= 1
            
            if stack:
                return False, f"未闭合的括号: {stack}"
            return True, ""
        
        # 检查原始代码
        orig_balanced, orig_msg = check_brackets_balance(original_code)
        new_balanced, new_msg = check_brackets_balance(new_code)
        
        if not orig_balanced:
            warnings.append(f"原始代码括号不平衡: {orig_msg}")
        
        if not new_balanced:
            warnings.append(f"新代码括号不平衡: {new_msg}")
            return False, warnings
        
        # 检查关键结构（简单的启发式检查）
        orig_lines = original_code.split('\n')
        new_lines = new_code.split('\n')
        
        # 检查类和方法定义
        orig_classes = [line for line in orig_lines if re.match(r'^\s*class\s+\w+', line)]
        new_classes = [line for line in new_lines if re.match(r'^\s*class\s+\w+', line)]
        
        orig_methods = [line for line in orig_lines if re.match(r'^\s*def\s+\w+', line)]
        new_methods = [line for line in new_lines if re.match(r'^\s*def\s+\w+', line)]
        
        if len(orig_classes) != len(new_classes):
            warnings.append(f"类定义数量变化: {len(orig_classes)} → {len(new_classes)}")
        
        if len(orig_methods) != len(new_methods):
            warnings.append(f"方法定义数量变化: {len(orig_methods)} → {len(new_methods)}")
        
        return True, warnings


class AdvancedCodeStructureAnalyzer:
    """增强的代码结构分析器"""
    
    def analyze_code_blocks(self, code: str) -> Dict[str, Any]:
        """分析代码块结构，识别函数、类、控制流等"""
        try:
            tree = ast.parse(code)
            blocks = {
                'functions': [],
                'classes': [],
                'imports': [],
                'module_level_code': []
            }
            
            # 收集模块级语句
            module_level_nodes = []
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                module_level_nodes.append(node)
            
            if module_level_nodes:
                first = getattr(module_level_nodes[0], 'lineno', 1)
                last_node = module_level_nodes[-1]
                last = getattr(last_node, 'end_lineno', max((getattr(n, 'lineno', 0) for n in ast.walk(last_node)), default=getattr(last_node, 'lineno', 0)))
                blocks['module_level_code'].append((first, last))
            
            # 分析函数
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_info = {
                        'name': node.name,
                        'start': node.lineno,
                        'end': getattr(node, 'end_lineno', max((getattr(n, 'lineno', 0) for n in ast.walk(node)), default=node.lineno)),
                        'args': [arg.arg for arg in node.args.args],
                        'decorators': [self._ast_to_code(dec) for dec in node.decorator_list]
                    }
                    blocks['functions'].append(func_info)
                
                elif isinstance(node, ast.ClassDef):
                    class_info = {
                        'name': node.name,
                        'start': node.lineno,
                        'end': getattr(node, 'end_lineno', max((getattr(n, 'lineno', 0) for n in ast.walk(node)), default=node.lineno)),
                        'methods': [],
                        'bases': [self._ast_to_code(base) for base in node.bases]
                    }
                    
                    # 分析类中的方法
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            method_info = {
                                'name': item.name,
                                'start': item.lineno,
                                'end': getattr(item, 'end_lineno', max((getattr(n, 'lineno', 0) for n in ast.walk(item)), default=item.lineno)),
                                'args': [arg.arg for arg in item.args.args]
                            }
                            class_info['methods'].append(method_info)
                    
                    blocks['classes'].append(class_info)
                
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_info = {
                        'start': node.lineno,
                        'end': getattr(node, 'end_lineno', node.lineno),
                        'names': []
                    }
                    
                    if isinstance(node, ast.Import):
                        import_info['names'] = [alias.name for alias in node.names]
                    else:  # ImportFrom
                        import_info['module'] = node.module
                        import_info['names'] = [alias.name for alias in node.names]
                    
                    blocks['imports'].append(import_info)
            
            return blocks
            
        except Exception as e:
            logger.error(f"代码结构分析失败: {e}")
            return {}
    
    def _ast_to_code(self, node):
        """将AST节点转换为代码字符串"""
        try:
            return ast.unparse(node)
        except:
            return str(node)


class LLMBlockEditor:
    """
    基于LLM的块编辑器 - 优化版
    
    主要改进：
    1. 更可靠的LLM输出解析
    2. 增强的指令验证
    3. 改进的代码结构检查
    4. 更好的错误恢复机制
    """
    
    def __init__(self, llm_client=None, max_retries: int = 2, simple_retry_mode: bool = True):
        """
        初始化编辑器
        
        Args:
            llm_client: LLM客户端
            max_retries: 最大重试次数
            simple_retry_mode: 简单重试模式（默认True）
                - True: 一次性应用所有Block，如果有错误直接让LLM重新生成整套指令（最多6次）
                - False: 复杂模式 - 两阶段应用 + 逐个修复 + LLM修复失败指令
        """
        warnings.warn(
            "core.utils.llm_block_editor.LLMBlockEditor (line-number based) "
            "is deprecated; prefer LineNumberFreeLLMBlockEditor or "
            "SmartLLMEditorV2 (see core/utils/code_editor.md).",
            DeprecationWarning,
            stacklevel=2,
        )
        self.llm_client = llm_client
        self.max_retries = max_retries
        self.simple_retry_mode = simple_retry_mode
        self.validator = CodeStructureValidator()
    
    def edit_with_llm(
        self, 
        original_code: str, 
        instruction: str,
        context: str = "",
        file_path: str = ""
    ) -> BlockEditResult:
        """
        使用LLM生成块编辑指令并应用 - 集成多层级回退策略
        
        Args:
            original_code: 原始代码
            instruction: 修改指令（自然语言）
            context: 额外上下文
            file_path: 文件路径（用于调试）
            
        Returns:
            BlockEditResult
        """
        logger.info("=" * 80)
        logger.info("LLM块编辑器 - 多层级回退策略版")
        logger.info("=" * 80)
        
        # 参数类型验证和安全转换，避免循环引用导致递归错误
        try:
            if not isinstance(original_code, str):
                logger.warning(f"⚠️ [BlockEditor] original_code 不是字符串类型: {type(original_code)}")
                original_code = str(original_code)
            
            if not isinstance(instruction, str):
                logger.warning(f"⚠️ [BlockEditor] instruction 不是字符串类型: {type(instruction)}")
                instruction = str(instruction)
            
            if not isinstance(context, str):
                logger.warning(f"⚠️ [BlockEditor] context 不是字符串类型: {type(context)}")
                # 使用安全转换，防止循环引用
                try:
                    context = str(context)
                except RecursionError:
                    logger.error("❌ [BlockEditor] context 包含循环引用，无法转换")
                    context = "[错误: context 包含循环引用]"
                except Exception as e:
                    logger.error(f"❌ [BlockEditor] context 转换失败: {e}")
                    context = f"[错误: 无法转换context - {type(context).__name__}]"
            
            if not isinstance(file_path, str):
                logger.warning(f"⚠️ [BlockEditor] file_path 不是字符串类型: {type(file_path)}")
                file_path = str(file_path)
                
        except RecursionError as e:
            logger.error(f"❌ [BlockEditor] 参数验证时遇到递归错误: {e}")
            return BlockEditResult(
                success=False,
                new_code=original_code if isinstance(original_code, str) else "",
                applied_instructions=[],
                errors=["参数包含循环引用，导致递归错误"],
                warnings=[],
                diff="",
                original_code=original_code if isinstance(original_code, str) else ""
            )
        except Exception as e:
            logger.error(f"❌ [BlockEditor] 参数验证失败: {e}")
            return BlockEditResult(
                success=False,
                new_code=original_code if isinstance(original_code, str) else "",
                applied_instructions=[],
                errors=[f"参数验证失败: {str(e)}"],
                warnings=[],
                diff="",
                original_code=original_code if isinstance(original_code, str) else ""
            )
        
        if file_path:
            logger.info(f"📁 目标文件: {file_path}")
        
        if not self.llm_client:
            return BlockEditResult(
                success=False,
                new_code=original_code,
                applied_instructions=[],
                errors=["LLM客户端不可用"],
                warnings=[],
                diff="",
                original_code=original_code
            )
        
        # 预处理代码
        processed_code, preprocess_warnings = self._preprocess_code(original_code)
        if preprocess_warnings:
            logger.warning(f"代码预处理警告: {preprocess_warnings}")
        
        # 尝试多次生成指令
        best_instructions = []
        best_error_count = float('inf')
        
        for attempt in range(self.max_retries):
            logger.info(f"🔄 尝试 {attempt + 1}/{self.max_retries}")
            
            # 步骤1: 生成块编辑指令
            edit_instructions = self._generate_block_instructions(
                processed_code, instruction, context, attempt
            )
            
            if not edit_instructions:
                logger.warning(f"❌ 第{attempt + 1}次尝试未能生成指令")
                continue
            
            # 🚨 检查是否有行号问题
            total_lines = len(processed_code.split('\n'))
            line_number_issues = self._detect_line_number_issues(edit_instructions, total_lines)
            
            if line_number_issues:
                logger.warning(f"⚠️ 第{attempt+1}次尝试检测到行号问题: {line_number_issues}")
                
                # 如果是最后一次尝试，启用强制修正
                if attempt == self.max_retries - 1:
                    logger.info("🔧 最后一次尝试，启用强制行号修正")
                    edit_instructions = self._force_correct_line_numbers(edit_instructions, processed_code)
            
            # 步骤2: 验证块编辑指令
            valid, errors = self._validate_instructions(edit_instructions, processed_code)
            
            if not valid:
                error_count = len(errors)
                logger.warning(f"❌ 第{attempt + 1}次尝试指令验证失败: {error_count}个错误")
                
                # 🔍 分类错误
                classified = ErrorClassifier.classify_errors(errors)
                
                # 输出错误分类信息
                if classified['ignorable']:
                    logger.info(f"  可忽略错误 ({len(classified['ignorable'])}个):")
                    for error in classified['ignorable']:
                        logger.info(f"    - {error}")
                
                if classified['retryable']:
                    logger.warning(f"  需重试错误 ({len(classified['retryable'])}个):")
                    for error in classified['retryable']:
                        logger.warning(f"    - {error}")
                
                if classified['unknown']:
                    logger.warning(f"  未知错误 ({len(classified['unknown'])}个):")
                    for error in classified['unknown']:
                        logger.warning(f"    - {error}")
                
                # 判断是否应该重试
                should_retry = ErrorClassifier.should_retry(errors)
                
                if not should_retry and classified['ignorable']:
                    logger.info(f"✅ 只有可忽略错误，接受此次结果")
                    # 虽然有错误，但都是可忽略的，接受这个结果
                    best_instructions = edit_instructions
                    break
                
                # 🔧 步骤2.5: 尝试使用LLM修复指令（block editor优先策略 - 只修复失败的部分）
                # 优先保证block editor成功，只有修复失败才重新生成
                logger.info("🔧 优先尝试修复失败的指令（block editor优先策略）")
                
                # 详细验证：区分成功和失败的指令
                valid_insts, failed_insts = self._validate_instructions_detailed(edit_instructions, processed_code)
                
                if not failed_insts:
                    # 如果详细验证后没有失败的指令，接受结果
                    logger.info("✅ 详细验证后所有指令都成功")
                    best_instructions = edit_instructions
                    break
                
                # 构造失败指令列表和错误列表
                failed_inst_list = [inst for inst, _ in failed_insts]
                failed_errors = []
                for inst, errs in failed_insts:
                    failed_errors.extend([f"{inst}: {e}" for e in errs])
                
                fixed_instructions = self._fix_block_instructions_with_llm(
                    original_code=processed_code,
                    instruction=instruction,
                    failed_instructions=failed_inst_list,
                    errors=failed_errors,
                    valid_instructions=valid_insts  # 传递成功的指令
                )
                
                if fixed_instructions:
                    # 验证修复后的指令
                    fixed_valid, fixed_errors = self._validate_instructions(fixed_instructions, processed_code)
                    
                    if fixed_valid:
                        logger.info("✅ LLM修复成功，使用修复后的指令")
                        best_instructions = fixed_instructions
                        break
                    else:
                        logger.warning(f"⚠️ LLM修复后仍有{len(fixed_errors)}个错误")
                        # 如果修复后的错误更少，使用修复后的
                        if len(fixed_errors) < error_count:
                            logger.info(f"✓ 修复减少了错误数量 ({error_count} → {len(fixed_errors)})，保存修复结果")
                            if len(fixed_errors) < best_error_count:
                                best_error_count = len(fixed_errors)
                                best_instructions = fixed_instructions
                else:
                    logger.warning("⚠️ LLM修复失败，将重新生成指令")
                
                if error_count < best_error_count:
                    best_error_count = error_count
                    best_instructions = edit_instructions
                
                continue
            
            logger.info(f"✅ 第{attempt + 1}次尝试成功生成有效指令")
            best_instructions = edit_instructions
            break
        
        if not best_instructions:
            logger.error("❌ 所有尝试都未能生成有效的块编辑指令")
            try:
                from core.utils.editor_fallback import FallbackLLMEditor
                fb = FallbackLLMEditor(self.llm_client, prefer="lnfree")
                fr = fb.edit_with_llm(original_code, instruction, context, file_path)
                if fr.success:
                    return fr
            except Exception:
                pass
            return BlockEditResult(
                success=False,
                new_code=original_code,
                applied_instructions=[],
                errors=["无法生成有效的块编辑指令"],
                warnings=preprocess_warnings,
                diff="",
                original_code=original_code
            )
        
        # 步骤3: 安全检查
        safety_warnings = self._apply_safety_checks(best_instructions, processed_code)
        if safety_warnings:
            logger.warning(f"⚠️ 安全检查发现警告: {safety_warnings}")
        
        # 步骤4: 应用块编辑指令（使用增量验证与回滚机制）
        new_code, applied_instructions, apply_errors = self._apply_instructions_with_rollback(
            processed_code, best_instructions
        )
        
        # 如果没有成功应用任何指令，回退到传统方法
        if not applied_instructions:
            logger.warning("增量验证失败，回退到传统批量应用")
            new_code, apply_errors, apply_warnings = self._apply_instructions(
                processed_code, best_instructions
            )
            applied_instructions = best_instructions if not apply_errors else []
        else:
            apply_warnings = []
        
        # 🔧 步骤4.5: 如果应用指令失败，根据模式选择重试策略
        if apply_errors or not applied_instructions:
            if self.simple_retry_mode:
                # 🚀 简单重试模式：直接让LLM重新生成整套指令
                logger.info("🚀 使用简单重试模式：让LLM重新生成完整指令集")
                result = self._simple_retry_with_regenerate(
                    original_code=processed_code,
                    instruction=instruction,
                    context=context,
                    initial_errors=apply_errors,
                    preprocess_warnings=preprocess_warnings,
                    max_attempts=6,
                    file_path=file_path
                )
                return result
            
            # 🔧 复杂模式：尝试LLM修复失败的指令（block editor优先策略 - 至少重试2次）
            logger.error(f"❌ 应用块编辑指令时出错: {apply_errors}")
            
            # 🔍 分类错误，判断是否值得修复
            from core.utils.error_classifier_helper import classify_and_handle_errors
            error_analysis = classify_and_handle_errors(apply_errors)
            
            logger.info("\n" + error_analysis['report'])
            
            # 🎯 策略：至少重试2次，最多5次，如果有进展则动态增加
            min_retry_attempts = 2  # 最少重试次数
            max_retry_attempts = 5  # 最多重试次数（如果持续有进展）
            base_retry_attempts = 3  # 基础重试次数
            
            retry_attempts = base_retry_attempts if error_analysis['should_retry'] else min_retry_attempts
            
            logger.info(f"🔧 尝试让LLM修复应用失败的指令（至少{min_retry_attempts}次，最多{max_retry_attempts}次）...")
            
            best_fixed_code = new_code
            best_fixed_applied = applied_instructions
            best_fixed_errors = list(apply_errors) if apply_errors else []
            
            # 🚀 使用SourceCode增量修复策略
            # 先创建SourceCode对象，应用所有成功的指令
            best_source = SourceCode(processed_code)
            if best_fixed_applied:
                logger.info(f"📦 先应用 {len(best_fixed_applied)} 个成功的指令")
                for inst in best_fixed_applied:
                    errors = best_source.apply_instruction(inst)
                    if errors:
                        logger.warning(f"  警告：成功指令应用时出错: {errors}")
            
            # 🎯 动态调整重试次数
            has_progress = False  # 标记是否有进展
            consecutive_no_progress = 0  # 连续无进展次数
            
            for fix_attempt in range(max_retry_attempts):
                    # 🎯 动态显示当前重试限制
                    if fix_attempt >= retry_attempts:
                        # 如果已经达到当前限制，检查是否应该继续
                        if not has_progress or consecutive_no_progress >= 2:
                            logger.info(f"⏹️ 已达到重试限制 ({retry_attempts} 次)，停止修复")
                            break
                    
                    logger.info(f"🔄 LLM修复尝试 {fix_attempt + 1} (当前限制: {retry_attempts}, 最大: {max_retry_attempts})")
                    
                    # 使用实际应用的结果
                    valid_insts = best_fixed_applied if best_fixed_applied else []
                    
                    # 计算实际失败的指令
                    if best_fixed_applied:
                        # 找出未成功应用的指令
                        applied_set = set(id(inst) for inst in best_fixed_applied)
                        failed_inst_list = [inst for inst in best_instructions if id(inst) not in applied_set]
                    else:
                        # 如果没有成功应用的指令，则全部失败
                        failed_inst_list = best_instructions
                    
                    failed_errors = list(best_fixed_errors)
                    
                    # 🔑 关键改进：只返回修复后的指令，不合并
                    fixed_only_instructions = self._fix_block_instructions_with_llm(
                        original_code=processed_code,
                        instruction=instruction,
                        failed_instructions=failed_inst_list,
                        errors=failed_errors,
                        valid_instructions=valid_insts,
                        return_merged=False  # 只返回修复后的指令
                    )
                    
                    if not fixed_only_instructions:
                        logger.warning("⚠️ LLM修复返回空结果")
                        if fix_attempt < min_retry_attempts - 1:
                            logger.info(f"↻ 未达到最小重试次数({min_retry_attempts})，继续重试")
                            continue
                        break
                    
                    # 验证修复后的指令（基于原始代码）
                    fixed_valid, fixed_validation_errors = self._validate_instructions(fixed_only_instructions, processed_code)
                    
                    if not fixed_valid:
                        logger.warning(f"⚠️ 修复后的指令验证失败: {fixed_validation_errors}")
                        if fix_attempt < min_retry_attempts - 1:
                            logger.info(f"↻ 未达到最小重试次数({min_retry_attempts})，继续重试")
                        continue
                    
                    # 🚀 增量应用：在已应用成功指令的基础上，只应用修复后的指令
                    logger.info(f"🔧 在已应用 {len(valid_insts)} 个成功指令的基础上，增量应用 {len(fixed_only_instructions)} 个修复后的指令")
                    test_source = best_source.clone()
                    
                    # 应用修复后的指令
                    temp_errors = []
                    temp_fixed_applied = []
                    for inst in fixed_only_instructions:
                        apply_errors = test_source.apply_instruction(inst)
                        if apply_errors:
                            temp_errors.extend(apply_errors)
                        else:
                            temp_fixed_applied.append(inst)
                    
                    # 验证最终代码
                    temp_code = test_source.get_code()
                    if not temp_errors:
                        syntax_ok, syntax_errors = self.validator.validate_python_syntax(temp_code)
                        if not syntax_ok:
                            temp_errors.extend(syntax_errors)
                    
                    # 判断修复效果
                    if not temp_errors:
                        logger.info("✅ LLM修复成功，增量应用修复后的指令，无语法错误")
                        # 合并所有成功的指令
                        all_applied = valid_insts + temp_fixed_applied
                        best_instructions = all_applied
                        best_fixed_code = temp_code
                        best_fixed_applied = all_applied
                        best_fixed_errors = []
                        best_source = test_source  # 更新成功的源代码对象
                        break
                    elif len(temp_errors) < len(best_fixed_errors):
                        # 🎯 有进展！
                        logger.info(f"✓ 修复减少了错误: {len(best_fixed_errors)} → {len(temp_errors)}")
                        all_applied = valid_insts + temp_fixed_applied
                        best_instructions = all_applied
                        best_fixed_code = temp_code
                        best_fixed_applied = all_applied
                        best_fixed_errors = list(temp_errors)
                        best_source = test_source  # 更新源代码对象
                        
                        # 🚀 有进展，增加重试次数（如果还没到最大值）
                        has_progress = True
                        consecutive_no_progress = 0  # 重置连续无进展计数
                        if retry_attempts < max_retry_attempts:
                            retry_attempts = min(retry_attempts + 1, max_retry_attempts)
                            logger.info(f"📈 检测到修复有进展，增加重试次数到 {retry_attempts}")
                        
                        # 继续尝试完全修复
                        if fix_attempt < min_retry_attempts - 1:
                            logger.info(f"↻ 未达到最小重试次数({min_retry_attempts})，继续尝试完全修复")
                        else:
                            logger.info(f"↻ 有进展，继续尝试完全修复")
                    else:
                        logger.warning(f"⚠️ 修复后错误数量未减少，仍有{len(temp_errors)}个错误")
                        consecutive_no_progress += 1
                        
                        if fix_attempt < min_retry_attempts - 1:
                            logger.info(f"↻ 未达到最小重试次数({min_retry_attempts})，继续重试")
                        elif consecutive_no_progress >= 2:
                            logger.warning(f"⚠️ 连续{consecutive_no_progress}次无进展，可能需要停止")
                        else:
                            logger.info(f"↻ 继续重试 (连续无进展: {consecutive_no_progress}次)")
            
            # 使用最佳修复结果
            new_code = best_fixed_code
            applied_instructions = best_fixed_applied
            apply_errors = best_fixed_errors
            
            # 判断是否接受部分成功的结果
            if applied_instructions and apply_errors:
                logger.warning(f"⚠️ 经过{retry_attempts}次修复，仍有{len(apply_errors)}个错误")
                logger.info(f"✓ 但有{len(applied_instructions)}个指令成功应用，尝试接受部分结果")
                # 将错误降级为警告
                apply_warnings.extend([f"部分应用后的错误: {e}" for e in apply_errors])
                apply_errors = []  # 清空错误，继续验证
            
            # 如果修复后仍然完全失败，才回退
            if not applied_instructions:
                logger.warning("⚠️ 精确编辑完全失败: " + str(best_fixed_errors))
                logger.info("↩️ 回退到标准编辑流程")
                try:
                    from core.utils.editor_fallback import FallbackLLMEditor
                    fb = FallbackLLMEditor(self.llm_client, prefer="lnfree")
                    fr = fb.edit_with_llm(original_code, instruction, context, file_path)
                    if fr.success:
                        return fr
                except Exception:
                    pass
                return BlockEditResult(
                    success=False,
                    new_code=original_code,
                    applied_instructions=[],
                    errors=best_fixed_errors,
                    warnings=preprocess_warnings + apply_warnings,
                    diff="",
                    original_code=original_code
                )
        
        # 步骤5: 验证代码结构
        structure_ok, structure_warnings = self.validator.check_structure_integrity(
            processed_code, new_code
        )
        
        all_warnings = preprocess_warnings + safety_warnings + apply_warnings + structure_warnings
        
        if not structure_ok:
            logger.warning("⚠️ 代码结构完整性检查发现问题")
        
        # 步骤6: 验证Python语法
        syntax_ok, syntax_errors = self.validator.validate_python_syntax(new_code)
        if not syntax_ok:
            logger.error(f"❌ 新代码存在语法错误: {syntax_errors}")
            return BlockEditResult(
                success=False,
                new_code=new_code,
                applied_instructions=applied_instructions,
                errors=syntax_errors,
                warnings=all_warnings,
                diff="",
                original_code=original_code
            )
        
        # 步骤7: 保持代码风格
        final_code = self._preserve_code_style(new_code, original_code)
        
        # 步骤8: 生成diff
        diff = self._generate_diff(original_code, final_code)
        
        logger.info(f"✅ 编辑完成，执行了 {len(applied_instructions)} 个块操作")
        
        return BlockEditResult(
            success=True,
            new_code=final_code,
            applied_instructions=applied_instructions,
            errors=[],
            warnings=all_warnings,
            diff=diff,
            original_code=original_code
        )
    
    def _preprocess_code(self, code: str) -> Tuple[str, List[str]]:
        """
        预处理代码
        
        Args:
            code: 原始代码
            
        Returns:
            (处理后的代码, 警告列表)
        """
        warnings = []
        # 输入清理：移除控制字符与超长裁剪
        try:
            if len(code) > 100000:
                warnings.append("输入代码过大，已截断为100000字符")
                code = code[:100000]
            code = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', code)
        except Exception:
            pass
        lines = code.split('\n')
        original_line_count = len(lines)
        
        # 移除代码开头/结尾的空行
        while lines and lines[0].strip() == '':
            removed_line = lines.pop(0)
            warnings.append(f"移除了开头的空行: '{removed_line}'")
        
        while lines and lines[-1].strip() == '':
            removed_line = lines.pop()
            warnings.append(f"移除了结尾的空行: '{removed_line}'")
        
        if len(lines) != original_line_count:
            warnings.append(f"修正代码格式: 行数 {original_line_count} → {len(lines)}")
        
        processed_code = '\n'.join(lines)
        
        # 验证原始代码语法
        syntax_ok, syntax_errors = self.validator.validate_python_syntax(processed_code)
        if not syntax_ok:
            warnings.extend([f"原始代码语法问题: {err}" for err in syntax_errors])
        
        return processed_code, warnings
    
    def _simple_retry_with_regenerate(
        self,
        original_code: str,
        instruction: str,
        context: str,
        initial_errors: List[str],
        preprocess_warnings: List[str],
        max_attempts: int = 6,
        file_path: str = ""
    ) -> BlockEditResult:
        """
        简单重试模式：如果有错误，直接让LLM重新生成整套指令
        
        Args:
            original_code: 原始代码
            instruction: 修改指令
            context: 上下文
            initial_errors: 初始错误列表
            preprocess_warnings: 预处理警告
            max_attempts: 最大尝试次数
            
        Returns:
            BlockEditResult
        """
        logger.info(f"🔄 开始简单重试模式（最多{max_attempts}次尝试）")
        logger.info(f"📋 初始错误: {initial_errors}")
        
        previous_errors = initial_errors
        
        for attempt in range(max_attempts):
            logger.info("=" * 60)
            logger.info(f"🔄 简单重试 - 第 {attempt + 1}/{max_attempts} 次尝试")
            logger.info("=" * 60)
            
            # 构建包含错误信息的增强指令
            if attempt == 0:
                enhanced_instruction = instruction
            else:
                enhanced_instruction = f"""
{instruction}

⚠️ 上一次尝试失败，出现以下错误：
{chr(10).join(f"  - {err}" for err in previous_errors)}

请重新生成完整的编辑指令，确保：
1. 行号准确无误
2. 缩进完全正确
3. 代码语法完整
4. 避免上述错误
"""
            
            # 步骤1: 让LLM重新生成完整的指令集
            logger.info("📝 请求LLM重新生成完整指令集...")
            edit_instructions = self._generate_block_instructions(
                original_code, 
                enhanced_instruction, 
                context, 
                attempt
            )
            
            if not edit_instructions:
                logger.warning(f"❌ 第{attempt + 1}次尝试未能生成指令")
                previous_errors = ["LLM未能生成有效指令"]
                continue
            
            # 步骤2: 验证指令
            valid, validation_errors = self._validate_instructions(edit_instructions, original_code)
            
            if not valid:
                logger.warning(f"⚠️ 指令验证失败: {validation_errors}")
                previous_errors = validation_errors
                continue
            
            # 步骤3: 一次性应用所有指令
            logger.info(f"🔨 一次性应用所有 {len(edit_instructions)} 条指令")
            
            # 使用SourceCode统一管理
            try:
                source = SourceCode(original_code)
                apply_errors = []
                
                for inst in edit_instructions:
                    inst_errors = source.apply_instruction(inst)
                    if inst_errors:
                        apply_errors.extend(inst_errors)
                
                if apply_errors:
                    logger.warning(f"⚠️ 应用指令时出错: {apply_errors}")
                    previous_errors = apply_errors
                    continue
                
                new_code = source.get_code()
                
            except Exception as e:
                logger.error(f"❌ 应用指令异常: {e}")
                previous_errors = [f"应用指令异常: {e}"]
                continue
            
            # 步骤4: 验证生成的代码
            syntax_ok, syntax_errors = self.validator.validate_python_syntax(new_code)
            
            if not syntax_ok:
                logger.warning(f"⚠️ 生成的代码有语法错误: {syntax_errors}")
                previous_errors = syntax_errors
                continue
            
            # 步骤5: 检查代码结构完整性
            structure_ok, structure_warnings = self.validator.check_structure_integrity(
                original_code, new_code
            )
            
            all_warnings = preprocess_warnings + structure_warnings
            
            if not structure_ok:
                logger.warning("⚠️ 代码结构完整性检查发现问题")
            
            # 步骤6: 成功！生成diff并返回
            diff = self._generate_diff(original_code, new_code)
            final_code = self._preserve_code_style(new_code, original_code)
            
            logger.info(f"✅ 简单重试模式成功！（第{attempt + 1}次尝试）")
            logger.info(f"✓ 应用了 {len(edit_instructions)} 条指令")
            
            return BlockEditResult(
                success=True,
                new_code=final_code,
                applied_instructions=edit_instructions,
                errors=[],
                warnings=all_warnings,
                diff=diff,
                original_code=original_code
            )
        
        # 所有尝试都失败了
        logger.error(f"❌ 简单重试模式失败：{max_attempts}次尝试都未能成功")
        logger.error(f"最后一次错误: {previous_errors}")
        try:
            from core.utils.editor_fallback import FallbackLLMEditor
            fb = FallbackLLMEditor(self.llm_client, prefer="lnfree")
            fr = fb.edit_with_llm(original_code, instruction, context, file_path)
            if fr.success:
                return fr
        except Exception:
            pass
        
        return BlockEditResult(
            success=False,
            new_code=original_code,
            applied_instructions=[],
            errors=previous_errors,
            warnings=preprocess_warnings,
            diff="",
            original_code=original_code
        )
    
    def _fix_block_instructions_with_llm(
        self,
        original_code: str,
        instruction: str,
        failed_instructions: List[BlockEditInstruction],
        errors: List[str],
        valid_instructions: List[BlockEditInstruction] = None,
        return_merged: bool = True
    ) -> Optional[List[BlockEditInstruction]]:
        """
        使用LLM修复失败的块编辑指令（优化版：只修复有错误的指令）
        
        Args:
            original_code: 原始代码
            instruction: 原始修改指令
            failed_instructions: 失败的指令列表
            errors: 错误列表
            valid_instructions: 已经验证成功的指令列表
            return_merged: 是否返回合并后的指令（成功+修复），False则只返回修复后的指令
            
        Returns:
            修复后的指令列表，如果修复失败返回None
        """
        if valid_instructions is None:
            valid_instructions = []
        logger.info("🔧 尝试使用LLM修复块编辑指令...")
        
        # 准备带行号的代码
        lines = original_code.split('\n')
        numbered_code = '\n'.join([
            f"|{i+1:04d}| {line}" for i, line in enumerate(lines)
        ])
        total_lines = len(lines)
        
        # 构建成功指令的描述
        valid_instructions_text = ""
        if valid_instructions:
            valid_list = []
            for i, inst in enumerate(valid_instructions, 1):
                valid_list.append(f"  {i}. {inst}")
            valid_instructions_text = f"""
✅ **已经验证成功的指令**（这些指令不需要修复，请保留）：
{chr(10).join(valid_list)}
"""
            logger.info(f"✓ 保留 {len(valid_instructions)} 个成功的指令")
        
        # 构建失败指令的描述
        failed_instructions_text = []
        for i, inst in enumerate(failed_instructions, 1):
            failed_instructions_text.append(f"  {i}. {inst}")
        
        logger.info(f"🔧 需要修复 {len(failed_instructions)} 个失败的指令")
        
        # 错误描述
        errors_text = '\n'.join([f"  - {error}" for error in errors])
        
        fix_prompt = f"""你之前生成的块编辑指令中，部分指令有错误需要修复。

**原始代码**（共{total_lines}行）：
```python
{numbered_code}
```

**原始修改需求**：
{instruction}

{valid_instructions_text}

❌ **需要修复的指令**（只修复这些！）：
{chr(10).join(failed_instructions_text)}

**错误信息**：
{errors_text}

**问题分析**：
请分析上述错误，可能的原因包括：
1. 行号超出范围（代码只有{total_lines}行）
2. 起始行大于结束行
3. 代码块语法不完整
4. 缩进不正确
5. **使用了省略号（...）或省略注释** ← 这是最常见的错误！

**修复要求**：
1. 仔细检查所有行号，确保在1-{total_lines}范围内
2. 确保起始行 ≤ 结束行
3. 确保替换/插入的代码语法完整
4. 保持正确的缩进

## 🚨 代码完整性要求（重要！）

**系统会对代码进行AST完整性检查，必须输出完整代码！**

❌ **严禁以下做法**：
- ❌ 不要使用省略号（`...`, `# ...`）
- ❌ 不要使用省略注释（如 `# 其他代码保持不变`）
- ❌ 不要截断函数、类或代码块

✅ **必须做到**：
- ✅ 输出的每个代码块必须是**完整的、可独立解析的**
- ✅ 函数必须包含完整的定义和所有代码行
- ✅ 类必须包含完整的定义和所有方法
- ✅ 代码必须能通过Python AST解析

## 📤 输出要求

⚠️ **重要**：你只需要输出**修复后的指令**（对应上面标记为❌的那些失败指令）。
✅ 标记为已验证成功的指令会自动保留，**不需要**重新输出。

格式示例：
```
>> REPLACE <start>:<end>
[修复后的完整代码，不要用省略号！]
<< END

>> INSERT AFTER <line>
[修复后的完整代码，不要用省略号！]
<< END
```

现在请输出**修复后的**块编辑指令（只输出修复的部分）："""
        
        try:
            # 调用LLM修复
            llm_output = self.llm_client.one_chat(fix_prompt)
            
            if not llm_output:
                logger.error("LLM修复返回为空")
                return None
            
            logger.debug(f"LLM修复输出 ({len(llm_output)} 字符)")
            
            # 解析修复后的指令
            fixed_instructions = self._parse_block_instructions(llm_output, original_code)
            
            if not fixed_instructions:
                logger.error("LLM修复未能生成有效指令")
                return None
            
            logger.info(f"✓ LLM修复生成了 {len(fixed_instructions)} 条指令")
            
            # 根据参数决定是否合并
            if return_merged:
                # 🔄 合并成功的指令和修复后的指令
                # 按行号排序，确保执行顺序正确
                merged_instructions = valid_instructions + fixed_instructions
                merged_instructions.sort(key=lambda x: x.start_line)
                
                logger.info(f"✅ 合并结果: {len(valid_instructions)} 个成功指令 + {len(fixed_instructions)} 个修复指令 = {len(merged_instructions)} 个总指令")
                
                return merged_instructions
            else:
                # 只返回修复后的指令
                logger.info(f"✅ 返回修复后的指令: {len(fixed_instructions)} 个")
                return fixed_instructions
            
        except Exception as e:
            logger.error(f"LLM修复失败: {e}")
            return None
    
    def _generate_block_instructions(
        self, 
        original_code: str, 
        instruction: str,
        context: str,
        attempt: int = 0
    ) -> List[BlockEditInstruction]:
        """
        向LLM请求块编辑指令
        
        Args:
            original_code: 原始代码
            instruction: 修改指令
            context: 上下文
            attempt: 尝试次数
            
        Returns:
            块编辑指令列表
        """
        # 安全转换参数为字符串，避免循环引用导致的递归错误
        def safe_str(obj, max_length=10000):
            """安全地将对象转换为字符串，避免递归错误"""
            try:
                if obj is None:
                    return "无"
                # 如果已经是字符串，直接返回
                if isinstance(obj, str):
                    result = obj
                else:
                    # 尝试转换为字符串
                    result = str(obj)
                # 限制长度，防止prompt过长
                if len(result) > max_length:
                    result = result[:max_length] + "...(内容过长已截断)"
                return result
            except RecursionError:
                logger.error("❌ [BlockEditor] 参数转换时遇到递归错误，可能包含循环引用")
                return "[错误: 参数包含循环引用，无法转换为字符串]"
            except Exception as e:
                logger.error(f"❌ [BlockEditor] 参数转换失败: {e}")
                return f"[错误: 无法转换参数 - {type(obj).__name__}]"
        
        # 安全转换参数
        instruction_str = safe_str(instruction)
        context_str = safe_str(context)
        
        # 准备带行号的代码
        lines = original_code.split('\n')
        numbered_code = '\n'.join([
            f"|{i+1:04d}| {line}" for i, line in enumerate(lines)
        ])
        
        total_lines = len(lines)
        
        # 🚨 行号范围警告（在提示词开头强调）
        header_warning = f"🚨 重要提醒：原始代码只有 {total_lines} 行！所有行号必须在 1-{total_lines} 范围内！\n\n"
        
        # 根据尝试次数调整提示词
        retry_note = f"\n⚠️ 注意：这是第 {attempt + 1} 次尝试，请确保输出格式完全正确，特别是行号范围！" if attempt > 0 else ""
        
        # 添加行号保护措施
        line_safeguards = self._add_line_number_safeguards(total_lines)
        
        prompt = header_warning + f"""你是专业的代码编辑专家。根据原始代码和修改需求，生成精确的块编辑指令。

**原始代码**（共{total_lines}行）：
```python
{numbered_code}
```

**修改需求**：{instruction_str}

**上下文信息**：{context_str}

**代码基本信息**：
- 文件总行数：{total_lines} 行
- 有效行号范围：1 到 {total_lines}
- ⚠️ 绝对不要使用超出这个范围的行号！

---

## 🧠 思考过程（Chain of Thought）

在生成指令之前，请先进行简短的思考：
1. **定位**：修改涉及哪些行？（查看带行号的原始代码）
2. **范围**：确定精确的起始行和结束行。
3. **完整性**：检查是否包含了完整的函数/类定义（如果需要替换整个函数/类）。
4. **缩进**：确认新代码的缩进层级。
5. **验证**：检查是否使用了省略号（严禁使用！）

请以 `>> THINKING` 开始，简述你的思考过程，然后以 `<< END` 结束。

---

## 📝 块编辑指令格式

### 四种操作类型

#### 1. 替换代码块
```
>> REPLACE <start_line>:<end_line>
[新代码内容]
<< END
```

**示例**：替换第5-10行
```
>> REPLACE 5:10
def new_function(self, data):
    result = process(data)
    return result
<< END
```

#### 2. 在某行后插入代码
```
>> INSERT AFTER <line_number>
[要插入的代码]
<< END
```

**示例**：在第20行后插入
```
>> INSERT AFTER 20
    # 新增的逻辑
    logger.info("处理开始")
<< END
```

#### 3. 在某行前插入代码
```
>> INSERT BEFORE <line_number>
[要插入的代码]
<< END
```

**示例**：在第15行前插入
```
>> INSERT BEFORE 15
    # 前置检查
    if not data:
        return None
<< END
```

#### 4. 删除代码块
```
>> DELETE <start_line>:<end_line>
<< END
```

**示例**：删除第30-35行
```
>> DELETE 30:35
<< END
```

---

## 🎯 核心规则

1. **精确的行号**：所有行号必须基于原始代码的带行号格式（如 |0021|）
2. **完整代码块**：替换/插入的内容必须是语法完整的代码块
3. **保持缩进**：新代码的缩进必须与上下文一致
4. **明确边界**：每个操作必须以 `<< END` 结束
5. **保持结构完整性**：
   - 不要删除方法/类的定义行（`def xxx():`, `class Xxx:`）
   - 不要分离装饰器和其方法（如 `@property` + `def method()`）
   - 替换方法时，必须包含完整的 `def` 定义行
   - 确保括号、引号等符号的完整性

## 🚨 代码完整性要求（重要！）

**系统会对代码进行AST完整性检查，必须输出完整代码！**

❌ **严禁以下做法**：
- ❌ 不要使用省略号（`...`, `# ...`）表示省略的代码
- ❌ 不要使用注释说明（如 `# 其他代码保持不变`, `# 这里保持原样`）
- ❌ 不要只输出修改部分而省略其他代码
- ❌ 不要截断函数、类或代码块

✅ **必须做到**：
- ✅ 输出的每个代码块都必须是**完整的、可独立解析的**
- ✅ 函数必须包含完整的定义和所有代码行
- ✅ 类必须包含完整的定义和所有方法
- ✅ 所有的括号、引号、缩进必须完整匹配
- ✅ 代码必须能通过Python AST解析

---

## 🚫 常见错误及避免方法

**错误1**：破坏代码结构
```python
# 原始代码：
|0021|    @property
|0022|    def signal_type(self):
|0023|        return 'value'

# ❌ 错误：只替换方法体，破坏了结构
>> REPLACE 22:23
    return 'new_value'
<< END

# ✅ 正确：包含完整的方法定义
>> REPLACE 22:23
    def signal_type(self):
        return 'new_value'
<< END
```

**错误2**：不匹配的缩进
```python
# ❌ 错误：缩进不正确
>> INSERT AFTER 10
# 新代码
def new_func():
print("hello")  # 缺少缩进
<< END

# ✅ 正确：保持正确缩进
>> INSERT AFTER 10
    # 新代码
    def new_func():
        print("hello")
<< END
```

**错误3**：不完整的代码块
```python
# ❌ 错误：缺少冒号
>> REPLACE 5:5
def incomplete_function
    return value
<< END

# ✅ 正确：完整的函数定义
>> REPLACE 5:5
def complete_function():
    return value
<< END
```

**错误4**：使用省略号（最常见！最危险！）
```python
# ❌ 严重错误：使用省略号会导致AST解析失败
>> REPLACE 10:50
def process_data(self, data):
    # 预处理
    data = self.preprocess(data)
    
    ...  # ❌ 不要这样！会导致语法错误！
    
    # 其他代码保持不变  ❌ 不要用注释说明！
    
    return result
<< END

# ✅ 正确：输出完整代码
>> REPLACE 10:50
def process_data(self, data):
    # 预处理
    data = self.preprocess(data)
    
    # 数据验证
    if not self.validate(data):
        return None
    
    # 处理逻辑
    result = self.transform(data)
    result = self.enrich(result)
    
    # 后处理
    result = self.postprocess(result)
    
    return result
<< END
```

**错误5**：截断长函数/类
```python
# ❌ 错误：只输出部分方法
>> REPLACE 100:200
class DataProcessor:
    def __init__(self):
        self.cache = {{}}
    
    def process(self, data):
        # ... 省略其他方法  ❌ 这会破坏类结构！
<< END

# ✅ 正确方案1：如果只修改部分，用多个小范围的REPLACE
>> REPLACE 102:105
    def __init__(self):
        self.cache = {{}}
        self.stats = {{}}  # 新增统计
<< END

# ✅ 正确方案2：如果修改较多，输出完整的类定义
>> REPLACE 100:200
class DataProcessor:
    def __init__(self):
        self.cache = {{}}
        self.stats = {{}}
    
    def process(self, data):
        # 完整的方法实现
        ...（完整代码）
    
    def validate(self, data):
        # 完整的方法实现
        ...（完整代码）
<< END
```

---

## 📤 输出要求

**严格遵守：**

1. ✅ **先输出思考过程**（`>> THINKING ... << END`）
2. ✅ **只输出块编辑指令**（不要输出其他解释）
3. ✅ **每个操作以 `>> 操作类型` 开始，以 `<< END` 结束**
4. ✅ **行号必须在 1-{total_lines} 范围内**
5. ✅ **代码内容保持正确的缩进和语法**
6. ✅ **确保代码结构完整性**
7. ✅ **绝对不要使用省略号（...）或省略注释**
8. ✅ **每个代码块必须能独立通过AST解析**
9. ❌ **不要输出原始代码的副本**

{retry_note}

{line_safeguards}

现在请生成块编辑指令：
"""
        
        try:
            # 调用LLM
            logger.info(f"📤 [BlockEditor] 调用LLM生成块编辑指令（尝试 {attempt + 1}）")
            logger.debug(f"📝 [BlockEditor] Prompt长度: {len(prompt)} 字符")
            
            llm_output = self.llm_client.one_chat(prompt)
            
            if not llm_output:
                logger.error("❌ [BlockEditor] LLM返回为空")
                return []
            
            logger.info(f"📥 [BlockEditor] 收到LLM响应: {len(llm_output)} 字符")
            logger.debug(f"📄 [BlockEditor] LLM原始输出:\n{llm_output[:1000]}..." if len(llm_output) > 1000 else f"📄 [BlockEditor] LLM原始输出:\n{llm_output}")
            
            # 保存LLM输出到临时文件（用于调试）
            self._save_llm_output(llm_output, f"attempt_{attempt}")
            
            # 解析指令（带行号修正）
            logger.info(f"🔍 [BlockEditor] 解析块编辑指令...")
            instructions = self._parse_block_instructions(llm_output, original_code)
            
            if instructions:
                logger.info(f"✓ [BlockEditor] 成功解析出 {len(instructions)} 条块编辑指令")
                for i, inst in enumerate(instructions, 1):
                    if inst.type == 'replace':
                        logger.debug(f"  {i}. REPLACE {inst.start_line}:{inst.end_line} ({len(inst.content.splitlines())}行新代码)")
                    elif inst.type == 'insert_after':
                        logger.debug(f"  {i}. INSERT AFTER {inst.start_line} ({len(inst.content.splitlines())}行)")
                    elif inst.type == 'insert_before':
                        logger.debug(f"  {i}. INSERT BEFORE {inst.start_line} ({len(inst.content.splitlines())}行)")
                    elif inst.type == 'delete':
                        logger.debug(f"  {i}. DELETE {inst.start_line}:{inst.end_line}")
            else:
                logger.warning(f"⚠️  [BlockEditor] 未能解析出任何有效指令")
            
            return instructions
            
        except Exception as e:
            logger.error(f"❌ [BlockEditor] LLM调用或解析失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return []
    
    def _apply_single_instruction(self, original_code: str, instruction: BlockEditInstruction) -> Tuple[str, List[str]]:
        """应用单个指令并返回结果"""
        try:
            # 创建临时指令列表
            temp_instructions = [instruction]
            
            # 应用指令
            new_code, errors, _ = self._apply_instructions(original_code, temp_instructions)
            
            return new_code, errors
            
        except Exception as e:
            return original_code, [f"单条指令应用失败: {e}"]
    
    def _detect_indent_size(self, code: str) -> int:
        """检测代码缩进大小"""
        lines = code.split('\n')
        for line in lines:
            if line.strip() and line.startswith(' '):
                # 计算前导空格数
                leading_spaces = len(line) - len(line.lstrip())
                if leading_spaces > 0:
                    return leading_spaces
        return 4  # 默认4空格
    
    def _detect_tabs_usage(self, code: str) -> bool:
        """检测是否使用制表符"""
        return '\t' in code
    
    def _detect_quote_style(self, code: str) -> str:
        """检测引号风格"""
        single_quotes = code.count("'")
        double_quotes = code.count('"')
        return "single" if single_quotes > double_quotes else "double"
    
    def _reindent_code(self, code: str, indent_size: int, use_tabs: bool) -> str:
        """重新缩进代码"""
        if use_tabs:
            # 转换为制表符
            return code.replace(' ' * 4, '\t')
        else:
            # 转换为空格
            # 将行首制表符统一为目标空格数，保留相对缩进
            def _convert_line(line: str) -> str:
                if not line.strip():
                    return ''
                # 替换行首的制表符为目标空格
                leading_tabs = len(line) - len(line.lstrip('\t'))
                if leading_tabs > 0:
                    line = (' ' * (indent_size * leading_tabs)) + line.lstrip('\t')
                return line
            converted_lines = [_convert_line(l) for l in code.split('\n')]
            return '\n'.join(converted_lines)
        return code
    
    def _convert_quotes(self, code: str, to_single: bool = True) -> str:
        """转换引号风格"""
        if to_single:
            # 双引号转单引号
            return code.replace('"', "'")
        else:
            # 单引号转双引号
            return code.replace("'", '"')
    
    def _apply_safety_checks(self, instructions: List[BlockEditInstruction], original_code: str) -> List[str]:
        """在应用指令前进行安全检查"""
        safety_warnings = []
        
        # 限制单次修改的最大行数比例
        total_lines = len(original_code.splitlines())
        max_allowed_change = total_lines * 0.7  # 70%上限
        
        modified_lines = 0
        for inst in instructions:
            if inst.type in ['replace', 'delete']:
                modified_lines += (inst.end_line - inst.start_line + 1)
            elif inst.type in ['insert_before', 'insert_after']:
                modified_lines += len(inst.content.splitlines())
        
        if modified_lines > max_allowed_change:
            safety_warnings.append(
                f"警告：修改范围过大 ({modified_lines}/{total_lines} 行)，可能引入意外变更"
            )
        
        # 检查危险操作
        for inst in instructions:
            if inst.type == 'delete' and (inst.end_line - inst.start_line + 1) > 20:
                safety_warnings.append(
                    f"警告：删除大段代码 ({inst.start_line}-{inst.end_line})，请确认必要性"
                )
        
        return safety_warnings
    
    def _preserve_code_style(self, new_code: str, original_code: str) -> str:
        """保持原始代码风格"""
        # 检测原始代码的缩进风格
        indent_size = self._detect_indent_size(original_code)
        uses_tabs = self._detect_tabs_usage(original_code)
        
        # 检测引号风格
        quote_style = self._detect_quote_style(original_code)
        
        # 应用风格到新代码
        if indent_size != 4 or uses_tabs:  # 默认是4空格
            new_code = self._reindent_code(new_code, indent_size, uses_tabs)
        
        if quote_style == "single":
            new_code = self._convert_quotes(new_code, to_single=True)
        
        return new_code
    
    def _apply_instructions_with_rollback(self, original_code: str, instructions: List[BlockEditInstruction]) -> Tuple[str, List[BlockEditInstruction], List[str]]:
        """应用指令 - 使用SourceCode类管理行号映射
        
        改进策略：
        1. 使用SourceCode类维护原始行号映射
        2. 第一阶段：一次性应用所有block，然后验证
        3. 如果验证通过，直接返回成功（避免block A制造的错误在block B中修复的情况）
        4. 如果验证失败，进入第二阶段：逐个应用定位问题block
        5. 关键优势：应用成功的block不需要重复应用，可以乱序应用
        
        Args:
            original_code: 原始代码
            instructions: 块编辑指令列表
            
        Returns:
            (最终代码, 成功应用的指令列表, 错误列表)
        """
        logger.info(f"🔨 [BlockEditor] 开始应用 {len(instructions)} 条块编辑指令（使用SourceCode管理）")
        
        # ==================== 第一阶段：批量应用所有block ====================
        logger.info(f"📦 第一阶段：批量应用所有 {len(instructions)} 个block")
        
        # 创建SourceCode对象
        source = SourceCode(original_code)
        
        # 应用所有指令
        batch_errors = []
        for inst in instructions:
            errors = source.apply_instruction(inst)
            if errors:
                batch_errors.extend(errors)
        
        if not batch_errors:
            # 验证批量应用后的代码
            logger.debug("  🔍 验证批量应用后的代码...")
            batch_code = source.get_code()
            
            syntax_ok, syntax_errors = self.validator.validate_python_syntax(batch_code)
            structure_ok, structure_warnings = self.validator.check_structure_integrity(original_code, batch_code)
            
            if syntax_ok and structure_ok:
                logger.info(f"✅ 批量应用成功！所有 {len(instructions)} 个block都已应用")
                return batch_code, instructions, []
            else:
                # 记录验证错误
                batch_errors = []
                if syntax_errors:
                    batch_errors.extend(syntax_errors)
                if structure_warnings:
                    batch_errors.extend(structure_warnings)
                logger.warning(f"⚠️ 批量应用后验证失败: {len(batch_errors)} 个错误")
                for err in batch_errors[:3]:  # 只显示前3个错误
                    logger.warning(f"   - {err}")
        else:
            logger.warning(f"⚠️ 批量应用失败: {len(batch_errors)} 个错误")
        
        # ==================== 第二阶段：逐个应用定位问题block ====================
        logger.info(f"🔍 第二阶段：逐个应用定位问题block")
        
        # 重新创建SourceCode对象
        source = SourceCode(original_code)
        applied_instructions = []
        errors = []
        
        for idx, inst in enumerate(instructions, 1):
            # 显示指令摘要
            if inst.type == 'replace':
                inst_desc = f"REPLACE {inst.start_line}:{inst.end_line}"
            elif inst.type == 'insert_after':
                inst_desc = f"INSERT AFTER {inst.start_line}"
            elif inst.type == 'insert_before':
                inst_desc = f"INSERT BEFORE {inst.start_line}"
            elif inst.type == 'delete':
                inst_desc = f"DELETE {inst.start_line}:{inst.end_line}"
            else:
                inst_desc = inst.type
            
            logger.info(f"  [{idx}/{len(instructions)}] 尝试: {inst_desc}")
            
            # 克隆当前状态以测试应用
            test_source = source.clone()
            apply_errors = test_source.apply_instruction(inst)
            
            if apply_errors:
                logger.warning(f"  ❌ 应用失败: {apply_errors[0] if apply_errors else '未知错误'}")
                errors.extend(apply_errors)
                continue
            
            # 验证临时代码
            logger.debug(f"  🔍 验证修改后的代码...")
            test_code = test_source.get_code()
            current_code = source.get_code()
            
            syntax_ok, syntax_errors = self.validator.validate_python_syntax(test_code)
            structure_ok, structure_warnings = self.validator.check_structure_integrity(current_code, test_code)
            
            if syntax_ok and structure_ok:
                # 应用成功，更新source
                source = test_source
                applied_instructions.append(inst)
                logger.info(f"  ✅ 应用成功")
            else:
                logger.warning(f"  ❌ 验证失败，回滚此指令")
                if syntax_errors:
                    logger.warning(f"     语法错误: {syntax_errors[0] if syntax_errors else '未知'}")
                    errors.extend(syntax_errors)
                if structure_warnings:
                    logger.warning(f"     结构错误: {structure_warnings[0] if structure_warnings else '未知'}")
                    errors.extend(structure_warnings)
        
        # 获取最终代码
        final_code = source.get_code()
        
        logger.info(f"📊 [BlockEditor] 应用结果: {len(applied_instructions)}/{len(instructions)} 成功, {len(errors)} 个错误")
        
        return final_code, applied_instructions, errors
    
    def _remove_line_markers(self, code: str) -> str:
        """移除所有行号标记"""
        lines = code.split('\n')
        clean_lines = []
        for line in lines:
            match = re.match(r'#__ORIG_LINE_\d+__#(.*)', line)
            if match:
                clean_lines.append(match.group(1))
            else:
                clean_lines.append(line)
        return '\n'.join(clean_lines)
    
    def _get_semantic_position(self, code: str, function_name: str) -> Optional[Tuple[int, int]]:
        """通过函数名获取其在代码中的位置范围"""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if node.name == function_name:
                        end_ln = getattr(node, 'end_lineno', max((getattr(n, 'lineno', 0) for n in ast.walk(node)), default=node.lineno))
                        return node.lineno, end_ln
        except Exception as e:
            logger.debug(f"AST解析失败: {e}")
        return None
    
    def _generate_context_aware_prompt(self, original_code: str, instruction: str, context: str = "") -> str:
        """生成包含代码上下文感知的提示词"""
        # 先分析代码结构
        analyzer = AdvancedCodeStructureAnalyzer()
        structure = analyzer.analyze_code_blocks(original_code)
        
        # 生成上下文摘要
        context_summary = f"""
代码结构摘要:
- 函数数量: {len(structure.get('functions', []))}
- 类数量: {len(structure.get('classes', []))}
- 模块级代码段: {len(structure.get('module_level_code', []))}

关键代码区域:
"""
        for func in structure.get('functions', [])[:3]:
            context_summary += f"- 函数 '{func['name']}' (行 {func['start']}-{func['end']})\n"
        
        for cls in structure.get('classes', [])[:2]:
            context_summary += f"- 类 '{cls['name']}' (行 {cls['start']}-{cls['end']})\n"
        
        # 将上下文摘要加入提示词
        return f"{context_summary}\n\n原始提示词内容: {instruction}\n\n额外上下文: {context if context else '无'}"
    
    def _apply_instructions_with_fuzzy_matching(self, original_code: str, instructions: List[BlockEditInstruction]) -> List[BlockEditInstruction]:
        """使用模糊匹配调整行号并返回修正后的指令列表"""
        lines = original_code.split('\n')
        total_lines = len(lines)
        
        # 创建代码指纹 (行内容哈希)
        code_fingerprints = {}
        for i, line in enumerate(lines, 1):
            # 忽略空格和注释的指纹
            clean_line = re.sub(r'#.*$', '', line).strip()
            if clean_line:
                code_fingerprints[hash(clean_line)] = i
        
        # 处理每条指令
        for inst in instructions:
            # 尝试通过内容匹配定位行
            if hasattr(inst, 'context_lines') and inst.context_lines:
                best_match = self._find_best_line_match(inst.context_lines, code_fingerprints)
                if best_match:
                    original_start = inst.start_line
                    original_end = inst.end_line
                    inst.start_line = max(1, min(best_match, total_lines))
                    if original_end:
                        span = original_end - original_start
                        inst.end_line = inst.start_line + span
        
        return instructions
    
    def _find_best_line_match(self, context_lines: List[str], code_fingerprints: Dict[int, int]) -> Optional[int]:
        """找到最佳行匹配"""
        best_match = None
        best_score = 0
        
        for line in context_lines:
            clean_line = re.sub(r'#.*$', '', line).strip()
            if clean_line:
                line_hash = hash(clean_line)
                if line_hash in code_fingerprints:
                    # 简单的匹配评分（可以改进为更复杂的算法）
                    score = len(clean_line)  # 行越长，匹配度越高
                    if score > best_score:
                        best_score = score
                        best_match = code_fingerprints[line_hash]
        
        return best_match
    
    def _try_block_edit(self, original_code: str, instruction: str, context: str = "") -> BlockEditResult:
        """尝试完整块编辑"""
        try:
            # 使用标准流程尝试块编辑
            return self.edit_with_llm(original_code, instruction, context)
        except Exception as e:
            logger.warning(f"块编辑失败: {e}")
            return BlockEditResult(
                success=False,
                new_code=original_code,
                applied_instructions=[],
                errors=[f"块编辑失败: {e}"],
                warnings=[],
                diff="",
                original_code=original_code
            )
    
    def _try_function_level_edit(self, original_code: str, instruction: str, context: str = "") -> BlockEditResult:
        """尝试函数级编辑"""
        try:
            # 分析代码结构，找到相关函数
            analyzer = AdvancedCodeStructureAnalyzer()
            structure = analyzer.analyze_code_blocks(original_code)
            
            # 从指令中提取关键词
            keywords = self._extract_keywords_from_instruction(instruction)
            
            # 找到可能相关的函数
            target_functions = []
            for func in structure.get('functions', []):
                if any(keyword in func['name'] for keyword in keywords):
                    target_functions.append(func)
            
            if not target_functions:
                logger.warning("未找到相关的函数进行函数级编辑")
                return BlockEditResult(
                    success=False,
                    new_code=original_code,
                    applied_instructions=[],
                    errors=["未找到相关的函数"],
                    warnings=[],
                    diff="",
                    original_code=original_code
                )
            
            # 针对第一个相关函数进行编辑
            target_func = target_functions[0]
            func_start = target_func['start'] - 1  # 转换为0基索引
            func_end = target_func['end']
            
            # 提取函数代码
            lines = original_code.split('\n')
            func_code = '\n'.join(lines[func_start:func_end])
            
            # 尝试编辑函数代码
            func_result = self.edit_with_llm(func_code, instruction, context)
            
            if func_result.success:
                # 将编辑后的函数代码替换回原代码
                new_lines = lines[:func_start] + func_result.new_code.split('\n') + lines[func_end:]
                new_code = '\n'.join(new_lines)
                
                return BlockEditResult(
                    success=True,
                    new_code=new_code,
                    applied_instructions=func_result.applied_instructions,
                    errors=[],
                    warnings=["使用函数级编辑完成"],
                    diff=self._generate_diff(original_code, new_code),
                    original_code=original_code
                )
            else:
                return func_result
                
        except Exception as e:
            logger.warning(f"函数级编辑失败: {e}")
            return BlockEditResult(
                success=False,
                new_code=original_code,
                applied_instructions=[],
                errors=[f"函数级编辑失败: {e}"],
                warnings=[],
                diff="",
                original_code=original_code
            )
    
    def _try_line_level_edit(self, original_code: str, instruction: str, context: str = "") -> BlockEditResult:
        """尝试行级编辑"""
        try:
            # 使用简单的行级编辑策略
            lines = original_code.split('\n')
            
            # 这里可以实现更智能的行级编辑逻辑
            # 目前作为最后的回退策略，返回原始代码
            logger.warning("行级编辑未实现，返回原始代码")
            
            return BlockEditResult(
                success=False,
                new_code=original_code,
                applied_instructions=[],
                errors=["行级编辑未实现"],
                warnings=["回退到行级编辑失败"],
                diff="",
                original_code=original_code
            )
            
        except Exception as e:
            logger.warning(f"行级编辑失败: {e}")
            return BlockEditResult(
                success=False,
                new_code=original_code,
                applied_instructions=[],
                errors=[f"行级编辑失败: {e}"],
                warnings=[],
                diff="",
                original_code=original_code
            )
    
    def _try_block_edit(self, original_code: str, instruction: str, context: str = "") -> BlockEditResult:
        """尝试完整块编辑（委托 edit_with_llm）"""
        try:
            return self.edit_with_llm(original_code, instruction, context)
        except Exception as e:
            logger.warning(f"块编辑失败: {e}")
            return BlockEditResult(
                success=False,
                new_code=original_code,
                applied_instructions=[],
                errors=[f"块编辑失败: {e}"],
                warnings=[],
                diff="",
                original_code=original_code
            )
    
    def _save_llm_output(self, llm_output: str, suffix: str = ""):
        """保存LLM输出到临时文件"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            suffix_str = f"_{suffix}" if suffix else ""
            temp_file = tempfile.NamedTemporaryFile(
                mode='w', 
                suffix=f'_block_llm_output{suffix_str}_{timestamp}.txt', 
                delete=False, 
                encoding='utf-8'
            )
            temp_file.write(llm_output)
            temp_file.close()
            logger.info(f"💾 LLM输出已保存到: {temp_file.name}")
        except Exception as e:
            logger.debug(f"保存LLM输出失败: {e}")
    
    def _parse_block_instructions(self, text: str, original_code: str = "") -> List[BlockEditInstruction]:
        """
        解析块编辑指令 - 增强版（带行号修正）
        
        Args:
            text: LLM返回的文本
            original_code: 原始代码（用于行号验证和修正）
            
        Returns:
            块编辑指令列表
        """
        instructions = []
        
        # 获取总行数
        total_lines = len(original_code.split('\n')) if original_code else float('inf')
        
        # 1. 提取并记录思考过程
        thinking_match = re.search(r'>>\s*THINKING\s*\n(.*?)\n<<\s*END', text, re.DOTALL | re.IGNORECASE)
        if thinking_match:
            thinking_content = thinking_match.group(1).strip()
            logger.info(f"🧠 [BlockEditor] LLM思考过程:\n{thinking_content}")
            # 从文本中移除思考过程，避免干扰后续解析
            text = text.replace(thinking_match.group(0), '')
        
        # 清理文本：移除可能的代码块标记
        cleaned_text = re.sub(r'```(?:\w+)?\s*\n', '', text)
        cleaned_text = re.sub(r'\n\s*```\s*$', '', cleaned_text)
        
        # 多种模式匹配，提高兼容性
        # 允许更宽松的格式：
        # - 忽略大小写
        # - 允许指令周围有更多空格
        # - 允许 : 周围有空格
        
        patterns = [
            # REPLACE
            (r'>>\s*REPLACE\s+(\d+)\s*:\s*(\d+)\s*\n(.*?)\n<<\s*END', 
             'replace', lambda m: (int(m.group(1)), int(m.group(2)), m.group(3))),
            
            # INSERT AFTER
            (r'>>\s*INSERT\s+AFTER\s+(\d+)\s*\n(.*?)\n<<\s*END', 
             'insert_after', lambda m: (int(m.group(1)), int(m.group(1)), m.group(2))),
            
            # INSERT BEFORE
            (r'>>\s*INSERT\s+BEFORE\s+(\d+)\s*\n(.*?)\n<<\s*END', 
             'insert_before', lambda m: (int(m.group(1)), int(m.group(1)), m.group(2))),
            
            # DELETE
            (r'>>\s*DELETE\s+(\d+)\s*:\s*(\d+)\s*\n<<\s*END', 
             'delete', lambda m: (int(m.group(1)), int(m.group(2)), '')),
        ]
        
        # 记录已处理的文本范围，避免重复匹配
        processed_ranges = []
        
        for pattern, inst_type, extractor in patterns:
            for match in re.finditer(pattern, cleaned_text, re.DOTALL | re.IGNORECASE):
                # 检查是否与已处理范围重叠
                is_overlap = False
                for start, end in processed_ranges:
                    if not (match.end() <= start or match.start() >= end):
                        is_overlap = True
                        break
                if is_overlap:
                    continue
                
                processed_ranges.append((match.start(), match.end()))
                
                try:
                    start_line, end_line, content = extractor(match)
                    original_start, original_end = start_line, end_line
                    
                    # 🚨 立即修正行号
                    if total_lines != float('inf'):
                        start_line = min(start_line, total_lines)
                        if end_line:
                            end_line = min(end_line, total_lines)
                            # 确保 start_line <= end_line
                            if start_line > end_line:
                                start_line, end_line = end_line, start_line
                        
                        # 记录修正
                        if start_line != original_start or (end_line and end_line != original_end):
                            logger.warning(f"🔧 行号自动修正: {inst_type} {original_start}:{original_end} → {start_line}:{end_line}")
                    
                    # 清理内容：移除前后的空行（保留中间的空行）
                    content_lines = content.split('\n')
                    # 移除开头的空行
                    while content_lines and content_lines[0].strip() == '':
                        content_lines.pop(0)
                    # 移除结尾的空行
                    while content_lines and content_lines[-1].strip() == '':
                        content_lines.pop(-1)
                    cleaned_content = '\n'.join(content_lines)
                    
                    instruction = BlockEditInstruction(
                        type=inst_type,
                        start_line=start_line,
                        end_line=end_line,
                        content=cleaned_content,
                        raw_text=match.group(0)
                    )
                    
                    instructions.append(instruction)
                    logger.debug(f"  解析: {instruction}")
                    
                except Exception as e:
                    logger.warning(f"解析指令失败: {e}")
                    continue
        
        # 按在文本中出现的顺序排序指令
        # (虽然正则匹配顺序可能打乱了，但我们可以根据match.start()重新排序)
        # 这里简化处理：假设LLM按顺序输出，或者我们信任正则的顺序
        # 为了更稳健，我们可以给instructions添加一个index属性，或者在上面收集时带上位置信息
        
        logger.info(f"解析出 {len(instructions)} 条块编辑指令")
        
        # 生成调试信息
        if instructions and original_code:
            debug_info = self._generate_debug_info(instructions, original_code)
            logger.debug(f"指令调试信息:\n{debug_info}")
        
        return instructions
    
    def _is_complete_code_replacement(
        self,
        instructions: List[BlockEditInstruction],
        original_code: str
    ) -> bool:
        """
        检测是否是完整代码替换
        
        Args:
            instructions: 块编辑指令列表
            original_code: 原始代码
            
        Returns:
            是否是完整代码替换
        """
        # 如果只有一个REPLACE指令，且替换范围覆盖了大部分代码，则认为是完整替换
        if len(instructions) == 1:
            inst = instructions[0]
            if inst.type == 'replace':
                total_lines = len(original_code.split('\n'))
                replaced_lines = inst.end_line - inst.start_line + 1 if inst.end_line else 1
                coverage = replaced_lines / total_lines if total_lines > 0 else 0
                
                # 如果替换了80%以上的代码，认为是完整替换
                if coverage >= 0.8:
                    logger.info(f"检测到完整代码替换：覆盖率 {coverage:.1%}")
                    return True
        
        return False
    
    def _validate_code_replacement_with_ast(
        self,
        original_code: str,
        new_code_content: str
    ) -> Tuple[bool, List[str]]:
        """
        使用AST验证代码替换的完整性
        
        Args:
            original_code: 原始代码
            new_code_content: 新代码内容
            
        Returns:
            (是否有效, 错误列表)
        """
        errors = []
        
        try:
            # 解析原代码
            orig_tree = ast.parse(original_code)
            orig_funcs = {node.name: node for node in ast.walk(orig_tree) 
                         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
            orig_classes = {node.name: node for node in ast.walk(orig_tree) 
                           if isinstance(node, ast.ClassDef)}
            
            logger.info(f"原代码: {len(orig_funcs)} 个函数, {len(orig_classes)} 个类")
            
        except SyntaxError as e:
            errors.append(f"原代码语法错误: {e}")
            return False, errors
        
        try:
            # 解析新代码
            new_tree = ast.parse(new_code_content)
            new_funcs = {node.name: node for node in ast.walk(new_tree) 
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
            new_classes = {node.name: node for node in ast.walk(new_tree) 
                          if isinstance(node, ast.ClassDef)}
            
            logger.info(f"新代码: {len(new_funcs)} 个函数, {len(new_classes)} 个类")
            
            # 检查是否有合理数量的函数和类
            if len(new_funcs) == 0 and len(new_classes) == 0:
                errors.append("新代码中没有任何函数或类定义")
                return False, errors
            
            # 如果新代码的函数/类数量与原代码接近，认为是完整替换
            func_ratio = len(new_funcs) / len(orig_funcs) if len(orig_funcs) > 0 else 1.0
            class_ratio = len(new_classes) / len(orig_classes) if len(orig_classes) > 0 else 1.0
            
            # 允许函数和类数量有一定变化（0.5-2.0倍）
            if 0.5 <= func_ratio <= 2.0 and 0.5 <= class_ratio <= 2.0:
                logger.info(f"✅ AST验证通过：函数比率 {func_ratio:.2f}, 类比率 {class_ratio:.2f}")
                return True, []
            else:
                logger.warning(f"⚠️ 代码结构变化较大：函数比率 {func_ratio:.2f}, 类比率 {class_ratio:.2f}")
                # 即使结构变化较大，只要语法正确也认为有效
                return True, []
            
        except SyntaxError as e:
            errors.append(f"新代码语法错误: {e}")
            return False, errors
    
    def _validate_instructions_detailed(
        self,
        instructions: List[BlockEditInstruction],
        original_code: str
    ) -> Tuple[List[BlockEditInstruction], List[Tuple[BlockEditInstruction, List[str]]]]:
        """
        详细验证块编辑指令，区分成功和失败的指令
        
        Args:
            instructions: 块编辑指令列表
            original_code: 原始代码
            
        Returns:
            (成功的指令列表, [(失败的指令, 错误列表)])
        """
        valid_instructions = []
        failed_instructions = []
        lines = original_code.split('\n')
        total_lines = len(lines)
        
        for i, inst in enumerate(instructions):
            inst_errors = []
            
            # 检查行号范围
            if inst.start_line < 1 or inst.start_line > total_lines:
                inst_errors.append(
                    f"起始行号 {inst.start_line} 超出范围 (1-{total_lines})"
                )
            
            if inst.end_line and (inst.end_line < 1 or inst.end_line > total_lines):
                inst_errors.append(
                    f"结束行号 {inst.end_line} 超出范围 (1-{total_lines})"
                )
            
            # 检查起始行必须 <= 结束行（带容错）
            if inst.end_line and inst.start_line > inst.end_line:
                # 🔧 容错处理：检查是否是LLM的"200:35"类型错误
                possible_correct_end = int(str(inst.start_line) + str(inst.end_line))
                
                if possible_correct_end <= total_lines:
                    try:
                        target_lines = lines[inst.start_line-1:possible_correct_end]
                        target_code = '\n'.join(target_lines)
                        
                        is_complete_block = False
                        if target_code.strip():
                            first_line = target_code.strip().split('\n')[0]
                            if any(first_line.strip().startswith(kw) for kw in ['def ', 'class ', 'async def ']):
                                lines_list = target_code.split('\n')
                                if len(lines_list) > 1:
                                    first_indent = len(first_line) - len(first_line.lstrip())
                                    last_line = lines_list[-1]
                                    if last_line.strip():
                                        last_indent = len(last_line) - len(last_line.lstrip())
                                        if last_indent <= first_indent or last_line.strip() in ['pass', 'return', 'break', 'continue']:
                                            is_complete_block = True
                        
                        if is_complete_block:
                            logger.warning(f"  🔧 自动修正指令#{i+1}: {inst.start_line}:{inst.end_line} → {inst.start_line}:{possible_correct_end}")
                            inst.end_line = possible_correct_end
                        else:
                            inst_errors.append(
                                f"起始行 {inst.start_line} 大于结束行 {inst.end_line} (可能是 {inst.start_line}:{possible_correct_end}?)"
                            )
                    except Exception as e:
                        logger.debug(f"行号容错检查失败: {e}")
                        inst_errors.append(
                            f"起始行 {inst.start_line} 大于结束行 {inst.end_line}"
                        )
                else:
                    inst_errors.append(
                        f"起始行 {inst.start_line} 大于结束行 {inst.end_line}"
                    )
            
            # 检查内容完整性（如果有内容）
            if inst.content and inst.type in ['replace', 'insert_before', 'insert_after']:
                # 简单的括号平衡检查
                if inst.content.count('(') != inst.content.count(')'):
                    inst_errors.append("代码括号不平衡")
                if inst.content.count('[') != inst.content.count(']'):
                    inst_errors.append("代码方括号不平衡")
                if inst.content.count('{') != inst.content.count('}'):
                    inst_errors.append("代码花括号不平衡")
            
            if inst_errors:
                failed_instructions.append((inst, inst_errors))
                logger.debug(f"指令#{i+1}验证失败: {inst_errors}")
            else:
                valid_instructions.append(inst)
                logger.debug(f"指令#{i+1}验证成功")
        
        return valid_instructions, failed_instructions
    
    def _check_indentation_consistency(self, inst: BlockEditInstruction, lines: List[str]) -> Optional[str]:
        """
        检查缩进一致性
        
        Args:
            inst: 指令
            lines: 原始代码行列表
            
        Returns:
            警告信息，如果一致则返回None
        """
        try:
            # 获取上下文行的缩进
            context_indent = None
            
            if inst.type == 'replace':
                # 尝试获取被替换块之前的行的缩进
                prev_line_idx = inst.start_line - 2
                while prev_line_idx >= 0:
                    line = lines[prev_line_idx]
                    if line.strip() and not line.strip().startswith('#'):
                        context_indent = len(line) - len(line.lstrip())
                        break
                    prev_line_idx -= 1
            
            elif inst.type == 'insert_after':
                # 获取插入点行的缩进
                if inst.start_line <= len(lines):
                    line = lines[inst.start_line - 1]
                    context_indent = len(line) - len(line.lstrip())
            
            elif inst.type == 'insert_before':
                # 获取插入点行的缩进
                if inst.start_line <= len(lines):
                    line = lines[inst.start_line - 1]
                    context_indent = len(line) - len(line.lstrip())
            
            if context_indent is None:
                return None
            
            # 检查新代码的第一行缩进
            content_lines = inst.content.split('\n')
            first_code_line = next((l for l in content_lines if l.strip()), None)
            
            if first_code_line:
                new_indent = len(first_code_line) - len(first_code_line.lstrip())
                
                # 如果新代码缩进比上下文少（且不是顶层），可能是错误的
                # 注意：这只是一个启发式检查，不一定完全准确
                if new_indent < context_indent and context_indent > 0:
                    # 特殊情况：else, elif, except, finally 等可能会减少缩进
                    first_word = first_code_line.strip().split()[0]
                    if first_word not in ['else:', 'elif', 'except', 'finally:', 'return', 'break', 'continue']:
                        return f"新代码缩进 ({new_indent}) 小于上下文缩进 ({context_indent})，可能不正确"
            
            return None
            
        except Exception:
            return None

    def _validate_instructions(
        self, 
        instructions: List[BlockEditInstruction],
        original_code: str
    ) -> Tuple[bool, List[str]]:
        """
        验证块编辑指令的有效性 - 增强版（支持AST兜底验证）
        
        Args:
            instructions: 块编辑指令列表
            original_code: 原始代码
            
        Returns:
            (是否有效, 错误列表)
        """
        errors = []
        warnings = []
        lines = original_code.split('\n')
        total_lines = len(lines)
        
        for i, inst in enumerate(instructions):
            # 检查行号范围
            if inst.start_line < 1 or inst.start_line > total_lines:
                errors.append(
                    f"指令#{i+1} {inst.type}: 起始行号 {inst.start_line} 超出范围 (1-{total_lines})"
                )
            
            if inst.end_line and (inst.end_line < 1 or inst.end_line > total_lines):
                errors.append(
                    f"指令#{i+1} {inst.type}: 结束行号 {inst.end_line} 超出范围 (1-{total_lines})"
                )
            
            # 检查起始行必须 <= 结束行（带容错）
            if inst.end_line and inst.start_line > inst.end_line:
                # 🔧 容错处理：检查是否是LLM的"200:35"类型错误（应该是200:235）
                # 尝试自动修正：如果end_line可以组合成合理的行号
                possible_correct_end = int(str(inst.start_line) + str(inst.end_line))
                
                if possible_correct_end <= total_lines:
                    # 进一步检查：这个范围是否覆盖完整的代码块（函数/类）
                    try:
                        target_lines = lines[inst.start_line-1:possible_correct_end]
                        target_code = '\n'.join(target_lines)
                        
                        # 简单检查：是否看起来是完整的代码块
                        is_complete_block = False
                        if target_code.strip():
                            # 检查是否以函数/类定义开始
                            first_line = target_code.strip().split('\n')[0]
                            if any(first_line.strip().startswith(kw) for kw in ['def ', 'class ', 'async def ']):
                                # 检查缩进是否回到同级或更外层（表示块结束）
                                lines_list = target_code.split('\n')
                                if len(lines_list) > 1:
                                    first_indent = len(first_line) - len(first_line.lstrip())
                                    last_line = lines_list[-1]
                                    if last_line.strip():  # 最后一行不是空行
                                        last_indent = len(last_line) - len(last_line.lstrip())
                                        # 最后一行缩进<=第一行，说明可能是完整块
                                        if last_indent <= first_indent or last_line.strip() in ['pass', 'return', 'break', 'continue']:
                                            is_complete_block = True
                        
                        if is_complete_block:
                            # 自动修正行号
                            logger.warning(f"  🔧 自动修正指令#{i+1}: {inst.start_line}:{inst.end_line} → {inst.start_line}:{possible_correct_end}")
                            inst.end_line = possible_correct_end
                        else:
                            errors.append(
                                f"指令#{i+1} {inst.type}: 起始行 {inst.start_line} 大于结束行 {inst.end_line} (可能是 {inst.start_line}:{possible_correct_end}?)"
                            )
                    except Exception as e:
                        logger.debug(f"行号容错检查失败: {e}")
                        errors.append(
                            f"指令#{i+1} {inst.type}: 起始行 {inst.start_line} 大于结束行 {inst.end_line}"
                        )
                else:
                    errors.append(
                        f"指令#{i+1} {inst.type}: 起始行 {inst.start_line} 大于结束行 {inst.end_line}"
                    )
            
            # 检查行号跨度是否合理
            if inst.end_line and (inst.end_line - inst.start_line) > 100:
                warnings.append(
                    f"指令#{i+1} {inst.type}: 修改范围较大 ({inst.end_line - inst.start_line + 1} 行)"
                )
            
            # 检查缩进一致性
            if inst.content and inst.type in ['replace', 'insert_after', 'insert_before']:
                indent_warning = self._check_indentation_consistency(inst, lines)
                if indent_warning:
                    warnings.append(f"指令#{i+1} {inst.type}: {indent_warning}")
        
        # 输出警告信息
        if warnings:
            for warning in warnings:
                logger.warning(f"⚠️ {warning}")
        
        # 如果有行号错误，但是是完整代码替换，尝试使用AST验证
        if errors and self._is_complete_code_replacement(instructions, original_code):
            logger.info("🔍 检测到完整代码替换，尝试使用AST验证来忽略行号错误")
            
            # 获取新代码内容
            replace_inst = instructions[0]
            if replace_inst.type == 'replace' and replace_inst.content:
                # 使用AST验证
                ast_valid, ast_errors = self._validate_code_replacement_with_ast(
                    original_code,
                    replace_inst.content
                )
                
                if ast_valid:
                    logger.info("✅ AST验证成功，忽略行号错误")
                    # 清空行号相关的错误
                    errors = [e for e in errors if '行号' not in e]
                    
                    if not errors:
                        logger.info("✅ 所有行号错误已被忽略，指令验证通过")
                        return True, []
                else:
                    logger.warning("❌ AST验证失败，保留行号错误")
                    errors.extend(ast_errors)
        
        return len(errors) == 0, errors
    
    def _apply_instructions(
        self, 
        original_code: str,
        instructions: List[BlockEditInstruction]
    ) -> Tuple[str, List[str], List[str]]:
        """
        应用块编辑指令
        
        核心思路：
        1. 给每一行添加real_line_number标记
        2. 按顺序应用所有操作（基于original line_number）
        3. 删除所有辅助标记，返回最终代码
        
        Args:
            original_code: 原始代码
            instructions: 块编辑指令列表
            
        Returns:
            (新代码, 错误列表, 警告列表)
        """
        errors = []
        warnings = []
        lines = original_code.split('\n')
        
        # 给每一行添加原始行号标记（用于后续操作定位）
        # 格式: #__ORIG_LINE_<number>__# actual_code
        # 
        # 关键改进：如果行已经有标记，保留原标记；否则添加新标记
        marked_lines = []
        for i, line in enumerate(lines):
            # 检查是否已经有标记
            if line.startswith('#__ORIG_LINE_'):
                # 保留原有标记
                marked_lines.append(line)
            else:
                # 添加新标记（使用当前行号）
                marked_line = f"#__ORIG_LINE_{i+1:04d}__#{line}"
                marked_lines.append(marked_line)
        
        logger.info(f"开始应用块编辑指令：原始代码 {len(lines)} 行")
        
        # 按操作类型分组并排序
        # 为了避免行号偏移问题，我们按照从后往前的顺序执行
        sorted_instructions = sorted(
            instructions, 
            key=lambda x: (x.start_line, x.type),
            reverse=True  # 从后往前
        )
        
        for inst in sorted_instructions:
            logger.debug(f"  执行: {inst}")
            
            # 找到对应的行
            target_indices = []
            for idx, marked_line in enumerate(marked_lines):
                match = re.match(r'#__ORIG_LINE_(\d+)__#', marked_line)
                if match:
                    orig_line_num = int(match.group(1))
                    
                    if inst.type in ['replace', 'delete']:
                        if inst.start_line <= orig_line_num <= inst.end_line:
                            target_indices.append(idx)
                    elif inst.type == 'insert_after':
                        if orig_line_num == inst.start_line:
                            target_indices.append(idx)
                    elif inst.type == 'insert_before':
                        if orig_line_num == inst.start_line:
                            target_indices.append(idx)
            
            if not target_indices:
                errors.append(f"{inst}: 未找到对应的行")
                continue
            
            # 执行操作
            if inst.type == 'replace':
                # 替换：删除原有行，插入新内容
                # 给新内容也添加标记（使用第一行的原始行号）
                first_idx = target_indices[0]
                orig_line_marker = re.match(r'#__ORIG_LINE_(\d+)__#', marked_lines[first_idx]).group(0)
                
                new_lines = inst.content.split('\n')
                marked_new_lines = [f"{orig_line_marker}{line}" for line in new_lines]
                
                # 删除旧行，插入新行
                for idx in sorted(target_indices, reverse=True):
                    marked_lines.pop(idx)
                
                # 在第一个位置插入新内容
                for new_line in reversed(marked_new_lines):
                    marked_lines.insert(target_indices[0], new_line)
                
                logger.debug(f"    替换了 {len(target_indices)} 行 → {len(new_lines)} 行")
            
            elif inst.type == 'delete':
                # 删除
                for idx in sorted(target_indices, reverse=True):
                    marked_lines.pop(idx)
                
                logger.debug(f"    删除了 {len(target_indices)} 行")
            
            elif inst.type == 'insert_after':
                # 在目标行后插入
                target_idx = target_indices[0]
                orig_line_marker = re.match(r'#__ORIG_LINE_(\d+)__#', marked_lines[target_idx]).group(0)
                
                new_lines = inst.content.split('\n')
                marked_new_lines = [f"{orig_line_marker}{line}" for line in new_lines]
                
                # 在目标行后插入
                for i, new_line in enumerate(marked_new_lines):
                    marked_lines.insert(target_idx + 1 + i, new_line)
                
                logger.debug(f"    在行后插入了 {len(new_lines)} 行")
            
            elif inst.type == 'insert_before':
                # 在目标行前插入
                target_idx = target_indices[0]
                orig_line_marker = re.match(r'#__ORIG_LINE_(\d+)__#', marked_lines[target_idx]).group(0)
                
                new_lines = inst.content.split('\n')
                marked_new_lines = [f"{orig_line_marker}{line}" for line in new_lines]
                
                # 在目标行前插入
                for i, new_line in enumerate(marked_new_lines):
                    marked_lines.insert(target_idx + i, new_line)
                
                logger.debug(f"    在行前插入了 {len(new_lines)} 行")
        
        # 保留标记，不移除！
        # 标记会在 _apply_instructions_with_rollback 的最后统一移除
        new_code = '\n'.join(marked_lines)
        
        # 计算实际行数（用于日志）
        actual_lines = len([m for m in marked_lines if m.strip()])
        
        logger.info(f"应用完成：新代码 {actual_lines} 行（原始 {len(lines)} 行）")
        
        return new_code, errors, warnings
    
    def _generate_diff(self, old_code: str, new_code: str) -> str:
        """生成diff"""
        import difflib
        
        old_lines = old_code.split('\n')
        new_lines = new_code.split('\n')
        
        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            lineterm='',
            n=3
        )
        
        return '\n'.join(diff)
    
    def _add_line_number_safeguards(self, total_lines: int) -> str:
        """
        添加行号保护措施到提示词
        
        Args:
            total_lines: 代码总行数
            
        Returns:
            行号保护措施文本
        """
        safeguards = f"""
## 行号保护措施

为了避免行号错误，请遵循以下规则：

1. **行号验证**：在输出前，确认每个行号都在 1-{total_lines} 范围内
2. **保守策略**：如果不确定具体行号，选择较小的范围而不是猜测
3. **分段修改**：如果需要修改大段代码，分成多个小范围操作
4. **位置参考**：使用函数名、类名等作为位置参考，而不是纯行号

**行号检查清单**：
- [ ] 所有起始行号 ≤ {total_lines}
- [ ] 所有结束行号 ≤ {total_lines}
- [ ] 起始行 ≤ 结束行
- [ ] 修改范围合理（通常不超过50行）
"""
        return safeguards
    
    def _generate_debug_info(self, instructions: List[BlockEditInstruction], original_code: str) -> str:
        """
        生成调试信息
        
        Args:
            instructions: 块编辑指令列表
            original_code: 原始代码
            
        Returns:
            调试信息文本
        """
        total_lines = len(original_code.split('\n'))
        
        debug_info = [
            f"代码总行数: {total_lines}",
            f"生成指令数: {len(instructions)}",
            "指令详情:"
        ]
        
        for i, inst in enumerate(instructions):
            # 检查行号是否有效
            start_valid = 1 <= inst.start_line <= total_lines
            end_valid = inst.end_line is None or (1 <= inst.end_line <= total_lines)
            status = "[有效]" if (start_valid and end_valid) else "[超出范围]"
            
            debug_info.append(
                f"  {i+1}. {inst.type} {inst.start_line}:{inst.end_line} - {status}"
            )
        
        return "\n".join(debug_info)
    
    def _detect_line_number_issues(self, instructions: List[BlockEditInstruction], total_lines: int) -> List[str]:
        """
        检测行号问题
        
        Args:
            instructions: 块编辑指令列表
            total_lines: 代码总行数
            
        Returns:
            问题列表
        """
        issues = []
        
        for i, inst in enumerate(instructions):
            if inst.start_line > total_lines:
                issues.append(f"指令#{i+1}: 起始行号 {inst.start_line} > 总行数 {total_lines}")
            
            if inst.end_line and inst.end_line > total_lines:
                issues.append(f"指令#{i+1}: 结束行号 {inst.end_line} > 总行数 {total_lines}")
            
            if inst.end_line and inst.start_line > inst.end_line:
                issues.append(f"指令#{i+1}: 起始行 {inst.start_line} > 结束行 {inst.end_line}")
        
        return issues
    
    def _force_correct_line_numbers(
        self, 
        instructions: List[BlockEditInstruction],
        original_code: str
    ) -> List[BlockEditInstruction]:
        """
        强制修正行号
        
        Args:
            instructions: 原始指令列表
            original_code: 原始代码
            
        Returns:
            修正后的指令列表
        """
        total_lines = len(original_code.split('\n'))
        corrected = []
        
        for inst in instructions:
            # 修正起始行号
            start_line = max(1, min(inst.start_line, total_lines))
            
            # 修正结束行号
            if inst.end_line:
                end_line = max(1, min(inst.end_line, total_lines))
                # 确保 start <= end
                if start_line > end_line:
                    start_line, end_line = end_line, start_line
            else:
                end_line = inst.end_line
            
            # 如果行号有变化，记录日志
            if start_line != inst.start_line or end_line != inst.end_line:
                logger.info(
                    f"🔧 强制修正行号: {inst.type} "
                    f"{inst.start_line}:{inst.end_line} → {start_line}:{end_line}"
                )
            
            # 创建修正后的指令
            corrected_inst = BlockEditInstruction(
                type=inst.type,
                start_line=start_line,
                end_line=end_line,
                content=inst.content,
                raw_text=inst.raw_text
            )
            corrected.append(corrected_inst)
        
        return corrected
    
    def _extract_keywords_from_instruction(self, instruction: str) -> List[str]:
        """
        从指令中提取关键词
        
        Args:
            instruction: 修改指令
            
        Returns:
            关键词列表
        """
        # 简单的关键词提取：提取可能是函数名、类名的单词
        keywords = []
        
        # 匹配函数名/类名模式（通常是标识符）
        pattern = r'\b[A-Za-z_][A-Za-z0-9_]{2,}\b'
        matches = re.findall(pattern, instruction)
        
        # 去重并过滤常见词
        common_words = {'the', 'and', 'for', 'from', 'import', 'return', 'def', 'class', 
                       'if', 'else', 'elif', 'while', 'try', 'except', 'with', 'as'}
        keywords = [w for w in set(matches) if w.lower() not in common_words]
        
        return keywords[:5]  # 最多返回5个关键词
    
    def _analyze_change_impact(self, original_code: str, new_code: str) -> Dict[str, Any]:
        """
        分析代码变更的影响
        
        Args:
            original_code: 原始代码
            new_code: 新代码
            
        Returns:
            影响分析结果
        """
        impact = {
            'potential_breaking_changes': [],
            'affected_functions': [],
            'affected_classes': [],
            'import_changes': [],
            'api_changes': []
        }
        
        # 使用AST分析变更影响
        try:
            orig_tree = ast.parse(original_code)
            new_tree = ast.parse(new_code)
            
            # 比较函数签名变化
            orig_funcs = {node.name: node for node in ast.walk(orig_tree) 
                         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
            
            new_funcs = {node.name: node for node in ast.walk(new_tree) 
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
            
            logger.debug(f"orig_funcs = {list(orig_funcs.keys())}")
            logger.debug(f"new_funcs = {list(new_funcs.keys())}")
            
            for name, node in orig_funcs.items():
                if name in new_funcs:
                    new_node = new_funcs[name]
                    # 检查参数变化
                    if len(node.args.args) != len(new_node.args.args):
                        impact['potential_breaking_changes'].append(
                            f"函数 '{name}' 参数数量变化: {len(node.args.args)} -> {len(new_node.args.args)}"
                        )
                        impact['affected_functions'].append(name)
                    
                    # 检查函数名变化（重命名）
                    if hasattr(node, 'name') and hasattr(new_node, 'name') and node.name != new_node.name:
                        impact['potential_breaking_changes'].append(
                            f"函数重命名: '{node.name}' -> '{new_node.name}'"
                        )
                else:
                    impact['potential_breaking_changes'].append(f"函数 '{name}' 被删除")
                    impact['affected_functions'].append(f"删除函数: {name}")
                    logger.debug(f"Added deleted function: {name}")
            
            # 检查新增函数
            for name in new_funcs:
                if name not in orig_funcs:
                    impact['affected_functions'].append(f"新增函数: {name}")
            
            # 检查删除的函数
            for name in orig_funcs:
                if name not in new_funcs:
                    impact['affected_functions'].append(f"删除函数: {name}")
            
            # 比较类定义变化
            orig_classes = {node.name: node for node in ast.walk(orig_tree) 
                           if isinstance(node, ast.ClassDef)}
            new_classes = {node.name: node for node in ast.walk(new_tree) 
                          if isinstance(node, ast.ClassDef)}
            
            for name in orig_classes:
                if name not in new_classes:
                    impact['potential_breaking_changes'].append(f"类 '{name}' 被删除")
                    impact['affected_classes'].append(name)
            
            for name in new_classes:
                if name not in orig_classes:
                    impact['affected_classes'].append(f"新增类: {name}")
            
            # 检查导入语句变化
            orig_imports = []
            new_imports = []
            
            for node in ast.walk(orig_tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        orig_imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        for alias in node.names:
                            orig_imports.append(f"{node.module}.{alias.name}")
            
            for node in ast.walk(new_tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        new_imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        for alias in node.names:
                            new_imports.append(f"{node.module}.{alias.name}")
            
            # 检查导入变化
            for imp in orig_imports:
                if imp not in new_imports:
                    impact['import_changes'].append(f"删除导入: {imp}")
            
            for imp in new_imports:
                if imp not in orig_imports:
                    impact['import_changes'].append(f"新增导入: {imp}")
            
        except Exception as e:
            logger.debug(f"变更影响分析失败: {e}")
        
        return impact
    
    def _generate_unit_tests_for_changes(self, original_code: str, new_code: str, instructions: List[BlockEditInstruction]) -> Optional[str]:
        """
        为代码变更生成单元测试建议
        
        Args:
            original_code: 原始代码
            new_code: 新代码
            instructions: 应用的指令列表
            
        Returns:
            测试建议或None
        """
        # 使用LLM生成针对变更的测试建议
        changes_summary = "\n".join([str(inst) for inst in instructions])
        
        prompt = f"""
原始代码:
```python
{original_code[:500]}  # 限制长度
```

新代码:
```python
{new_code[:500]}  # 限制长度
```

变更摘要:
{changes_summary}

请为这些变更生成2-3个单元测试建议，确保覆盖新功能和回归测试。
只输出测试代码，不要解释。
"""
        
        try:
            test_suggestions = self.llm_client.one_chat(prompt)
            return test_suggestions
        except Exception as e:
            logger.debug(f"生成测试建议失败: {e}")
            return None
    
    def apply_instruction_string(
        self, 
        original_code: str,
        instruction_string: str
    ) -> BlockEditResult:
        """
        直接应用块编辑指令字符串（不通过LLM）
        
        Args:
            original_code: 原始代码
            instruction_string: 块编辑指令字符串
            
        Returns:
            BlockEditResult
        """
        logger.info("应用预定义的块编辑指令")
        
        # 解析指令（带行号修正）
        instructions = self._parse_block_instructions(instruction_string, original_code)
        
        if not instructions:
            return BlockEditResult(
                success=False,
                new_code=original_code,
                applied_instructions=[],
                errors=["未能解析块编辑指令"],
                warnings=[],
                diff="",
                original_code=original_code
            )
        
        # 🚨 检查行号问题并修正
        total_lines = len(original_code.split('\n'))
        line_number_issues = self._detect_line_number_issues(instructions, total_lines)
        
        if line_number_issues:
            logger.warning(f"⚠️ 检测到行号问题: {line_number_issues}")
            logger.info("🔧 启用强制行号修正")
            instructions = self._force_correct_line_numbers(instructions, original_code)
        
        # 验证指令
        valid, errors = self._validate_instructions(instructions, original_code)
        if not valid:
            return BlockEditResult(
                success=False,
                new_code=original_code,
                applied_instructions=[],
                errors=errors,
                warnings=[],
                diff="",
                original_code=original_code
            )
        
        # 应用指令
        new_code, apply_errors, apply_warnings = self._apply_instructions(original_code, instructions)
        
        if apply_errors:
            return BlockEditResult(
                success=False,
                new_code=original_code,
                applied_instructions=[],
                errors=apply_errors,
                warnings=apply_warnings,
                original_code=original_code,
                diff=""
            )
        
        # 生成diff
        diff = self._generate_diff(original_code, new_code)
        
        return BlockEditResult(
            success=True,
            new_code=new_code,
            applied_instructions=instructions,
            errors=[],
            warnings=apply_warnings,
            original_code=original_code,
            diff=diff
        )


if __name__ == '__main__':
    """测试"""
    
    editor = LLMBlockEditor()
    
    # 测试代码
    original = """def calculate(x):
    result = x * 2
    return result

def other():
    pass

def third():
    value = 100
    return value
"""
    
    # 测试指令
    instruction_text = """
>> REPLACE 1:3
def calculate(x, y):
    '''新的计算函数'''
    result = x * y * 3
    return result
<< END

>> INSERT AFTER 7
    # 新增逻辑
    logger.info("third function called")
<< END

>> DELETE 5:6
<< END
"""
    
    result = editor.apply_instruction_string(original, instruction_text)
    
    if result.success:
        print("\n[OK] 块编辑成功")
        print("\n新代码:")
        print(result.new_code)
        print("\nDiff:")
        print(result.diff)
        if result.warnings:
            print(f"\n⚠️  警告: {result.warnings}")
    else:
        print(f"\n[ERROR] 块编辑失败: {result.errors}")
        if result.warnings:
            print(f"\n⚠️  警告: {result.warnings}")
    
    # 打印结果详情
    print(f"\n📊 结果统计:")
    print(f"  - 应用了 {len(result.applied_instructions)} 条指令")
    print(f"  - 原始代码: {len(result.original_code.split('\\n'))} 行")
    print(f"  - 新代码: {len(result.new_code.split('\\n'))} 行")

