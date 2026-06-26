#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于LLM的精确代码编辑器

使用行级编辑指令实现精确的代码修改
"""

import re
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from core.utils.log import logger
from core.utils.source_code_manager import LineCode, SourceCode


class LineNumberHandler:
    """动态行号处理器
    
    解决硬编码4位行号限制问题，支持动态行号宽度计算和格式化
    """
    
    def __init__(self, total_lines: int):
        """初始化行号处理器
        
        Args:
            total_lines: 总行数
        """
        self.total_lines = total_lines
        self.line_width = max(4, len(str(total_lines)))  # 最小4位，根据总行数动态调整
        self.max_line_number = 10 ** self.line_width - 1  # 最大支持的行号
        
    def format_line_number(self, line_num: int) -> str:
        """格式化行号
        
        Args:
            line_num: 行号（1-based）
            
        Returns:
            格式化后的行号字符串
            
        Raises:
            ValueError: 行号超出范围
        """
        if line_num < 0:
            raise ValueError(f"行号不能为负数: {line_num}")
        if line_num > self.max_line_number:
            raise ValueError(f"行号 {line_num} 超出最大支持范围 {self.max_line_number}")
        return f"{line_num:0{self.line_width}d}"
    
    def parse_line_number(self, line_str: str) -> int:
        """从字符串中解析行号
        
        Args:
            line_str: 包含行号的字符串（如"0005"）
            
        Returns:
            解析出的行号
            
        Raises:
            ValueError: 格式无效或行号超出范围
        """
        try:
            line_num = int(line_str)
            if line_num < 0:
                raise ValueError(f"行号不能为负数: {line_num}")
            if line_num > self.max_line_number:
                raise ValueError(f"行号 {line_num} 超出最大支持范围 {self.max_line_number}")
            return line_num
        except ValueError as e:
            raise ValueError(f"无效的行号格式 '{line_str}': {e}")
    
    def create_numbered_code(self, code_lines: List[str]) -> str:
        """创建带行号的代码字符串
        
        Args:
            code_lines: 代码行列表
            
        Returns:
            带行号的代码字符串
        """
        numbered_lines = []
        for i, line in enumerate(code_lines):
            line_num = i + 1
            formatted_num = self.format_line_number(line_num)
            numbered_lines.append(f"|{formatted_num}{line}")
        return '\n'.join(numbered_lines)
    
    def get_line_width(self) -> int:
        """获取当前行号宽度
        
        Returns:
            行号宽度
        """
        return self.line_width
    
    def get_max_supported_lines(self) -> int:
        """获取支持的最大行数
        
        Returns:
            最大支持的行数
        """
        return self.max_line_number


class CodeStructureValidator:
    """代码结构验证器
    
    验证代码的语法和结构完整性，提供详细的错误报告
    """
    
    def __init__(self):
        """初始化验证器"""
        self.validation_errors = []
        self.validation_warnings = []
        self.validation_recommendations = []
    
    def validate_syntax(self, code: str, filename: str = "<string>") -> Tuple[bool, List[str]]:
        """验证Python代码语法
        
        Args:
            code: 要验证的代码
            filename: 文件名（用于错误报告）
            
        Returns:
            (是否有效, 错误列表)
        """
        self.validation_errors = []
        self.validation_warnings = []
        self.validation_recommendations = []
        self.validation_recommendations = []
        
        try:
            # 尝试编译代码
            compile(code, filename, 'exec')
            return True, []
        except SyntaxError as e:
            error_msg = f"语法错误 (行 {e.lineno or '未知'}): {e.msg}"
            if e.text:
                error_msg += f"\n    {e.text.strip()}"
                if e.offset:
                    error_msg += f"\n    {' ' * (e.offset - 1)}^"
            self.validation_errors.append(error_msg)
            return False, self.validation_errors
        except Exception as e:
            error_msg = f"编译错误: {str(e)}"
            self.validation_errors.append(error_msg)
            return False, self.validation_errors
    
    def validate_structure(self, code: str) -> Tuple[bool, List[str], List[str]]:
        """验证代码结构完整性
        
        Args:
            code: 要验证的代码
            
        Returns:
            (是否有效, 错误列表, 警告列表)
        """
        self.validation_errors = []
        self.validation_warnings = []
        
        try:
            import ast
            tree = ast.parse(code)
            
            # 检查常见的结构问题
            self._check_indentation_consistency(code)
            self._check_balanced_brackets(code)
            self._check_common_issues(code)
            
            # AST结构验证
            self._validate_ast_structure(code)
            
            return len(self.validation_errors) == 0, self.validation_errors, self.validation_warnings
            
        except SyntaxError as e:
            return False, [f"结构验证失败: {str(e)}"], []
        except Exception as e:
            return False, [f"结构验证异常: {str(e)}"], []
    
    def _check_indentation_consistency(self, code: str):
        """检查缩进一致性"""
        lines = code.split('\n')
        
        # 检测混合使用空格和制表符
        has_spaces = False
        has_tabs = False
        
        for i, line in enumerate(lines, 1):
            stripped = line.lstrip()
            if not stripped or stripped.startswith('#'):
                continue
                
            if '\t' in line[:len(line) - len(stripped)]:
                has_tabs = True
            if ' ' in line[:len(line) - len(stripped)]:
                has_spaces = True
        
        if has_tabs and has_spaces:
            self.validation_warnings.append(
                "检测到混合使用空格和制表符进行缩进，建议统一使用空格"
            )
    
    def _check_balanced_brackets(self, code: str):
        """检查括号是否平衡"""
        brackets = {'(': ')', '[': ']', '{': '}'}
        stack = []
        lines = code.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # 跳过字符串和注释
            in_string = False
            string_char = None
            i = 0
            
            while i < len(line):
                char = line[i]
                
                # 处理字符串
                if not in_string and char in ['"', "'"]:
                    in_string = True
                    string_char = char
                elif in_string and char == string_char and line[i-1] != '\\':
                    in_string = False
                    string_char = None
                elif not in_string:
                    # 检查括号
                    if char in brackets:
                        stack.append((char, line_num, i))
                    elif char in brackets.values():
                        if not stack:
                            self.validation_errors.append(
                                f"多余的闭括号 '{char}' (行 {line_num}, 列 {i+1})"
                            )
                        else:
                            last_open, _, _ = stack[-1]
                            if brackets[last_open] != char:
                                self.validation_errors.append(
                                    f"括号不匹配: 期望 '{brackets[last_open]}' 但得到 '{char}' (行 {line_num}, 列 {i+1})"
                                )
                            else:
                                stack.pop()
                
                i += 1
        
        # 检查未闭合的括号
        for bracket, line_num, col in stack:
            self.validation_errors.append(
                f"未闭合的括号 '{bracket}' (行 {line_num}, 列 {col+1})"
            )
    
    def _validate_ast_structure(self, code: str):
        """使用AST进行深度代码结构验证"""
        try:
            import ast
            tree = ast.parse(code)
            
            # 收集AST节点信息
            function_defs = []
            class_defs = []
            import_statements = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    function_defs.append({
                        'name': node.name,
                        'line': node.lineno,
                        'args': len(node.args.args),
                        'has_return': any(isinstance(n, ast.Return) for n in ast.walk(node))
                    })
                elif isinstance(node, ast.ClassDef):
                    class_defs.append({
                        'name': node.name,
                        'line': node.lineno,
                        'methods': len([n for n in node.body if isinstance(n, ast.FunctionDef)])
                    })
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_statements.append({
                        'line': node.lineno,
                        'names': [alias.name for alias in node.names]
                    })
            
            # AST结构验证
            self._check_ast_issues(function_defs, class_defs, import_statements)
            
        except SyntaxError as e:
            # 语法错误已在validate_syntax中处理
            pass
        except Exception as e:
            self.validation_errors.append(f"AST解析错误: {str(e)}")
    
    def _check_ast_issues(self, functions: List[Dict], classes: List[Dict], imports: List[Dict]):
        """基于AST检查代码结构问题"""
        
        # 检查函数定义
        for func in functions:
            # 检查函数命名规范
            if not re.match(r'^[a-z_][a-z0-9_]*$', func['name']):
                self.validation_warnings.append(
                    f"函数 '{func['name']}' (行 {func['line']}) 命名不符合snake_case规范"
                )
            
            # 检查无返回值的函数
            if not func['has_return'] and func['name'] != '__init__':
                self.validation_recommendations.append(
                    f"函数 '{func['name']}' (行 {func['line']}) 没有return语句，建议添加显式返回"
                )
        
        # 检查类定义
        for cls in classes:
            # 检查类命名规范
            if not re.match(r'^[A-Z][a-zA-Z0-9]*$', cls['name']):
                self.validation_warnings.append(
                    f"类 '{cls['name']}' (行 {cls['line']}) 命名不符合PascalCase规范"
                )
            
            # 检查空类
            if cls['methods'] == 0:
                self.validation_recommendations.append(
                    f"类 '{cls['name']}' (行 {cls['line']}) 没有方法，考虑是否必要"
                )
        
        # 检查导入语句
        if len(imports) > 10:
            self.validation_recommendations.append(
                f"导入语句较多 ({len(imports)})，考虑按需导入或使用__init__.py组织"
            )
        
        # 检查重复导入
        import_names = []
        for imp in imports:
            import_names.extend(imp['names'])
        
        from collections import Counter
        import_counts = Counter(import_names)
        duplicates = {name: count for name, count in import_counts.items() if count > 1}
        
        for name, count in duplicates.items():
            self.validation_warnings.append(
                f"模块 '{name}' 被重复导入了 {count} 次"
            )

    def _check_common_issues(self, code: str):
        """检查常见的代码问题"""
        lines = code.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # 检查行尾反斜杠（续行符）
            if line.rstrip().endswith('\\'):
                self.validation_warnings.append(
                    f"行 {line_num}: 使用反斜杠续行，建议使用括号包裹长表达式"
                )
            
            # 检查可能的缩进错误
            if stripped and not line.startswith((' ', '\t')) and line_num > 1:
                prev_line = lines[line_num - 2].rstrip() if line_num > 1 else ""
                if prev_line.endswith(':'):
                    self.validation_warnings.append(
                        f"行 {line_num}: 可能缺少缩进（上一行以冒号结束）"
                    )
    
    def validate_imports(self, code: str) -> Tuple[bool, List[str]]:
        """验证导入语句的有效性
        
        Args:
            code: 要验证的代码
            
        Returns:
            (是否有效, 错误列表)
        """
        errors = []
        
        try:
            import ast
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name
                        try:
                            # 尝试导入模块
                            __import__(module_name)
                        except ImportError:
                            errors.append(f"无法导入模块: {module_name}")
                        except Exception as e:
                            errors.append(f"导入模块 {module_name} 时出错: {str(e)}")
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        try:
                            # 尝试从模块导入
                            module = __import__(node.module, fromlist=[alias.name for alias in node.names])
                            for alias in node.names:
                                if not hasattr(module, alias.name):
                                    errors.append(f"模块 {node.module} 中没有属性: {alias.name}")
                        except ImportError:
                            errors.append(f"无法从模块导入: {node.module}")
                        except Exception as e:
                            errors.append(f"从模块 {node.module} 导入时出错: {str(e)}")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            return False, [f"导入验证失败: {str(e)}"]
    
    def get_comprehensive_report(self, code: str, filename: str = "<string>") -> Dict:
        """获取综合验证报告
        
        Args:
            code: 要验证的代码
            filename: 文件名
            
        Returns:
            综合报告字典
        """
        report = {
            'filename': filename,
            'total_lines': len(code.split('\n')),
            'syntax_valid': False,
            'structure_valid': False,
            'import_valid': False,
            'errors': [],
            'warnings': [],
            'recommendations': []
        }
        
        # 语法验证
        syntax_valid, syntax_errors = self.validate_syntax(code, filename)
        report['syntax_valid'] = syntax_valid
        report['errors'].extend(syntax_errors)
        
        if syntax_valid:
            # 结构验证（只在语法有效时进行）
            structure_valid, structure_errors, structure_warnings = self.validate_structure(code)
            report['structure_valid'] = structure_valid
            report['errors'].extend(structure_errors)
            report['warnings'].extend(structure_warnings)
            
            # 添加AST验证的警告和建议
            report['warnings'].extend(self.validation_warnings)
            report['recommendations'].extend(self.validation_recommendations)
            
            # 导入验证（只在语法有效时进行）
            import_valid, import_errors = self.validate_imports(code)
            report['import_valid'] = import_valid
            report['errors'].extend(import_errors)
        
        # 生成建议
        if not syntax_valid:
            report['recommendations'].append("修复语法错误后再进行结构验证")
        
        if report['warnings']:
            report['recommendations'].append("处理警告信息以提高代码质量")
        
        if len(report['errors']) == 0 and len(report['warnings']) == 0:
            report['recommendations'].append("代码质量良好，无需改进")
        
        return report


@dataclass
class EditInstruction:
    """编辑指令"""
    line_number: int
    modifier: str  # '+' (add), '-' (delete), 'r' (replace)
    content: str
    
    def __str__(self):
        # 使用默认的4位行号格式保持向后兼容
        return f"{self.modifier}{self.line_number:04d}{self.content}"
    
    def format_with_handler(self, line_handler: LineNumberHandler) -> str:
        """使用LineNumberHandler格式化行号
        
        Args:
            line_handler: 行号处理器
            
        Returns:
            格式化后的指令字符串
        """
        formatted_line_num = line_handler.format_line_number(self.line_number)
        return f"{self.modifier}{formatted_line_num}{self.content}"


@dataclass
class EditResult:
    """编辑结果"""
    success: bool
    new_code: str
    applied_instructions: List[EditInstruction]
    errors: List[str]
    diff: str


class LLMCodeEditor:
    """
    基于LLM的代码编辑器
    
    工作流程：
    1. 向LLM提供原始代码和修改意图
    2. LLM返回编辑指令列表
    3. 执行编辑指令
    4. 验证结果
    """
    
    def __init__(self, llm_client=None):
        """
        初始化编辑器
        
        Args:
            llm_client: LLM客户端
        """
        self.llm_client = llm_client
    
    def edit_with_llm(
        self, 
        original_code: str, 
        instruction: str,
        context: str = ""
    ) -> EditResult:
        """
        使用LLM生成编辑指令并应用
        
        Args:
            original_code: 原始代码
            instruction: 修改指令（自然语言）
            context: 额外上下文
            
        Returns:
            EditResult
        """
        logger.info("=" * 80)
        logger.info("LLM辅助代码编辑")
        logger.info("=" * 80)
        
        # 参数类型验证和安全转换，避免循环引用导致递归错误
        try:
            if not isinstance(instruction, str):
                logger.warning(f"⚠️ [LLMCodeEditor] instruction 不是字符串类型: {type(instruction)}")
                instruction = str(instruction)
            
            if not isinstance(context, str):
                logger.warning(f"⚠️ [LLMCodeEditor] context 不是字符串类型: {type(context)}")
                try:
                    context = str(context)
                except RecursionError:
                    logger.error("❌ [LLMCodeEditor] context 包含循环引用")
                    context = "[错误: context 包含循环引用]"
                except Exception as e:
                    logger.error(f"❌ [LLMCodeEditor] context 转换失败: {e}")
                    context = ""
        except RecursionError as e:
            logger.error(f"❌ [LLMCodeEditor] 参数验证时遇到递归错误: {e}")
            return EditResult(
                success=False,
                new_code=original_code,
                applied_instructions=[],
                errors=["参数包含循环引用，导致递归错误"],
                diff=""
            )
        except Exception as e:
            logger.error(f"❌ [LLMCodeEditor] 参数验证失败: {e}")
            return EditResult(
                success=False,
                new_code=original_code,
                applied_instructions=[],
                errors=[f"参数验证失败: {str(e)}"],
                diff=""
            )
        
        if not self.llm_client:
            return EditResult(
                success=False,
                new_code=original_code,
                applied_instructions=[],
                errors=["LLM客户端不可用"],
                diff=""
            )
        
        # ⚠️ 修复：移除代码开头/结尾的空行（防止行号错位）
        lines_temp = original_code.split('\n')
        original_line_count = len(lines_temp)
        
        # 移除开头的空行
        while lines_temp and lines_temp[0] == '':
            lines_temp.pop(0)
            logger.warning("⚠️ 移除了代码开头的空行（防止行号错位）")
        
        # 移除结尾的空行
        while lines_temp and lines_temp[-1] == '':
            lines_temp.pop()
        
        if len(lines_temp) != original_line_count:
            logger.warning(f"⚠️ 修正代码格式: 行数 {original_line_count} → {len(lines_temp)} (移除 {original_line_count - len(lines_temp)} 个空行)")
            original_code = '\n'.join(lines_temp)  # 使用修正后的代码
        
        # 步骤1: 生成编辑指令
        logger.info("步骤1: 向LLM请求编辑指令")
        edit_instructions = self._generate_edit_instructions(
            original_code, instruction, context
        )
        
        if not edit_instructions:
            logger.error("❌ 未能生成有效的编辑指令")
            return EditResult(
                success=False,
                new_code=original_code,
                applied_instructions=[],
                errors=["未能生成编辑指令"],
                diff=""
            )
        
        logger.info(f"✓ 生成了 {len(edit_instructions)} 条编辑指令")
        
        # 步骤2: 验证编辑指令
        logger.info("步骤2: 验证编辑指令")
        valid, errors = self._validate_instructions(edit_instructions, original_code)
        
        if not valid:
            logger.error(f"❌ 编辑指令验证失败: {errors}")
            return EditResult(
                success=False,
                new_code=original_code,
                applied_instructions=[],
                errors=errors,
                diff=""
            )
        
        logger.info("✓ 编辑指令验证通过")
        
        # 步骤3: 应用编辑指令（启用事务性模式）
        logger.info("步骤3: 应用编辑指令")
        new_code, apply_errors = self._apply_instructions(
            original_code, edit_instructions, transactional=True
        )
        
        if apply_errors:
            logger.error(f"❌ 应用编辑指令时出错: {apply_errors}")
            return EditResult(
                success=False,
                new_code=original_code,
                applied_instructions=[],
                errors=apply_errors,
                diff=""
            )
        
        logger.info("✓ 编辑指令应用成功")
        
        # 步骤4: 验证结果代码的结构完整性
        logger.info("步骤4: 验证结果代码的结构完整性")
        validator = CodeStructureValidator()
        validation_report = validator.get_comprehensive_report(new_code, "<edited>")
        
        if not validation_report['syntax_valid']:
            syntax_errors = [err for err in validation_report['errors'] if '语法错误' in err]
            logger.error(f"❌ 结果代码语法验证失败: {syntax_errors}")
            return EditResult(
                success=False,
                new_code=original_code,
                applied_instructions=[],
                errors=[f"结果代码语法错误: {'; '.join(syntax_errors)}"],
                diff=""
            )
        
        if validation_report['warnings']:
            logger.warning(f"⚠️  结果代码存在警告: {validation_report['warnings']}")
        
        logger.info("✓ 结果代码验证通过")
        
        # 步骤5: 生成diff
        diff = self._generate_diff(original_code, new_code)
        
        logger.info(f"✅ 编辑完成，修改了 {len(edit_instructions)} 处")
        
        return EditResult(
            success=True,
            new_code=new_code,
            applied_instructions=edit_instructions,
            errors=[],
            diff=diff
        )
    
    def _generate_edit_instructions(
        self, 
        original_code: str, 
        instruction: str,
        context: str
    ) -> List[EditInstruction]:
        """
        向LLM请求编辑指令
        
        Args:
            original_code: 原始代码
            instruction: 修改指令
            context: 上下文
            
        Returns:
            编辑指令列表
        """
        # 安全转换参数，避免循环引用
        def safe_str(obj, default=""):
            try:
                if obj is None:
                    return default
                if isinstance(obj, str):
                    return obj
                return str(obj)
            except RecursionError:
                logger.error("❌ [LLMCodeEditor] 参数包含循环引用")
                return "[错误: 参数包含循环引用]"
            except Exception as e:
                logger.error(f"❌ [LLMCodeEditor] 参数转换失败: {e}")
                return default
        
        instruction_str = safe_str(instruction, "无修改需求")
        context_str = safe_str(context, "")
        
        # 准备prompt
        lines = original_code.split('\n')
        
        # 使用LineNumberHandler处理行号
        line_handler = LineNumberHandler(len(lines))
        numbered_code = line_handler.create_numbered_code(lines)
        
        # 计算原始代码行数
        total_lines = len(lines)
        max_line_num = line_handler.format_line_number(total_lines)
        
        prompt = f"""你是代码编辑专家。根据原始代码和修改需求，生成精确的编辑指令。

**原始代码**（共{total_lines}行）：
```python
{numbered_code}
```

**修改需求**：{instruction_str}

---

## 📝 编辑指令格式

### 唯一正确的格式：[命令符][行号][代码]

**格式规则：**
1. 命令符(`r`, `+`, `-`)必须在第一个字符
2. 命令符后紧跟{line_handler.get_line_width()}位行号（0001-{max_line_num}）
3. 行号后直接跟代码内容（保留原始缩进）

**三种命令：**
- `r` = 替换指定行
- `+` = 在指定行后插入新行
- `-` = 删除指定行

**示例：**
```
r{line_handler.format_line_number(50)}    def calculate(self, x):
+{line_handler.format_line_number(50)}        '''计算函数'''
-{line_handler.format_line_number(100)}
```

---

## 🎯 核心规则

1. **行号固定**：所有行号基于原始代码，不因操作而改变
2. **完整代码**：替换行必须是完整的代码行（包含缩进）
3. **仅一次操作**：每行只能有一个替换操作（不要重复输出同一行号的r指令）

---

## 📤 输出要求

**严格遵守以下规则：**

1. ✅ **只输出编辑指令**（不要输出任何解释、说明或原始代码）
2. ✅ **每行一条指令**（格式：[命令符][行号][代码]）
3. ✅ **每个行号只出现一次r指令**（同一行不要重复替换）
4. ❌ **不要输出原始代码的副本**
5. ❌ **不要添加注释或说明文字**
6. ❌ **不要使用代码块包裹指令**（直接输出指令本身）

**正确的输出示例：**
```
r{line_handler.format_line_number(32)}    def calculate(self, data: pd.DataFrame) -> pd.Series:
r{line_handler.format_line_number(50)}        result = self.process(data)
+{line_handler.format_line_number(50)}        logger.info("处理完成")
```

**错误的输出示例（不要这样做）：**
```
# 不要添加说明文字
r{line_handler.format_line_number(32)}    def calculate(self, data: pd.DataFrame) -> pd.Series:

# 不要重复输出同一行
r{line_handler.format_line_number(32)}    def calculate(self, data: pd.DataFrame) -> pd.Series:
r{line_handler.format_line_number(32)}    def calculate(self, data: pd.DataFrame) -> pd.Series:

# 不要输出原始代码
|{line_handler.format_line_number(33)}    ema_short = talib.EMA(...)
```

现在请生成编辑指令（只输出指令，不要任何其他内容）：
"""
        
        try:
            # 调用LLM（使用one_chat，无历史记录）
            # 注意：LLM Client的统一接口是 one_chat/text_chat/tool_chat
            llm_output = self.llm_client.one_chat(prompt)
            
            if not llm_output:
                logger.error("LLM返回为空")
                return []
            
            logger.debug(f"LLM输出:\n{llm_output}")
            
            # 保存LLM输出到临时文件，方便诊断
            try:
                import tempfile
                import time
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                temp_file = tempfile.NamedTemporaryFile(
                    mode='w', 
                    suffix=f'_llm_output_{timestamp}.txt', 
                    delete=False, 
                    encoding='utf-8'
                )
                temp_file.write(llm_output)
                temp_file.close()
                logger.info(f"💾 LLM输出已保存到: {temp_file.name}")
            except Exception as e:
                logger.debug(f"保存LLM输出失败: {e}")
            
            instructions = self._parse_edit_instructions(llm_output)
            
            return instructions
            
        except Exception as e:
            logger.error(f"生成编辑指令时出错: {e}")
            return []
    
    def _detect_line_width_from_text(self, text: str) -> int:
        """从文本中检测行号宽度
        
        通过扫描文本中的行号模式来确定合理的行号宽度
        
        Args:
            text: 要分析的文本
            
        Returns:
            检测到的行号宽度（默认返回4）
        """
        # 尝试不同的行号宽度（从4到8位）
        for width in range(4, 9):
            # 匹配格式: [cmd][width位数字] 或 [width位数字][cmd]
            pattern1 = rf'[+\-r]\d{{{width}}}'  # 命令符在前
            pattern2 = rf'\d{{{width}}}[+\-r]'  # 行号在前
            
            # 统计匹配的数量
            matches1 = len(re.findall(pattern1, text))
            matches2 = len(re.findall(pattern2, text))
            
            # 如果找到合理的匹配数量，返回这个宽度
            if matches1 + matches2 >= 3:  # 至少找到3个匹配
                logger.debug(f"检测到行号宽度 {width}（找到 {matches1 + matches2} 个匹配）")
                return width
        
        # 默认返回4位宽度
        return 4
    
    def _parse_edit_instructions(self, text: str) -> List[EditInstruction]:
        """
        解析编辑指令（增强版 - 支持动态行号宽度和多模式匹配）
        
        Args:
            text: LLM返回的文本
            
        Returns:
            编辑指令列表
        """
        instructions = []
        parse_errors = []
        
        # 提取代码块（如果有）
        code_blocks = re.findall(r'```(?:.*?)\n(.*?)```', text, re.DOTALL)
        if code_blocks:
            text = code_blocks[0]  # 使用第一个代码块
        
        # 先估算一个合理的行号宽度（用于正则表达式匹配）
        # 通过扫描文本中的行号模式来动态确定
        line_width = self._detect_line_width_from_text(text)
        
        # 解析每一行
        has_pipe_format = False
        parsed_count = 0
        skipped_count = 0
        
        for line_num, original_line in enumerate(text.split('\n'), 1):
            # ⚠️ 关键：完全不修改原始行，保留所有缩进和空白！
            # rstrip()会破坏有缩进的空行（如 "0294+        "）
            line = original_line
            
            # 检查是否为空行（或只有空白的行）
            if not line or not line.strip():
                continue
            
            # 检测原始代码显示格式: |0001content 或动态宽度的行号
            if re.match(rf'\|\d{{{line_width}}}', line):
                has_pipe_format = True
                logger.debug(f"   过滤原始代码行: {line[:60]}...")
                # 自动跳过原始代码显示格式的行
                continue
            
            # 尝试多种解析模式
            instruction = self._try_parse_instruction_line(line, line_width)
            
            if instruction:
                instructions.append(instruction)
                parsed_count += 1
                logger.debug(f"   解析成功: {line[:60]}...")
            else:
                # 记录解析失败的行，用于诊断
                if len(parse_errors) < 10:  # 限制错误日志数量
                    parse_errors.append(f"行 {line_num}: '{line[:50]}...'")
                skipped_count += 1
        
        if has_pipe_format:
            logger.debug("   自动过滤了原始代码显示行（带竖线|的行）")
        
        logger.info(f"解析完成: 成功 {parsed_count} 条, 跳过 {skipped_count} 条")
        
        if parse_errors:
            logger.debug(f"解析失败的行（前{min(10, len(parse_errors))}个）:")
            for error in parse_errors:
                logger.debug(f"   {error}")
        
        logger.info(f"解析出 {len(instructions)} 条编辑指令")
        for inst in instructions[:5]:  # 显示前5条
            logger.debug(f"  {inst}")
        
        # 检测重复指令（同一行号、同一操作类型）
        line_operation_count = {}
        for inst in instructions:
            key = (inst.line_number, inst.modifier)
            line_operation_count[key] = line_operation_count.get(key, 0) + 1
        
        duplicates = {k: v for k, v in line_operation_count.items() if v > 1}
        if duplicates:
            logger.warning(f"⚠️  检测到 {len(duplicates)} 个行号有重复的指令:")
            # 使用合理的行号宽度进行显示（基于最大行号）
            max_line_num = max(line_num for line_num, _ in duplicates.keys()) if duplicates else 9999
            display_width = max(4, len(str(max_line_num)))
            
            for (line_num, modifier), count in sorted(duplicates.items())[:10]:  # 只显示前10个
                formatted_line = f"{line_num:0{display_width}d}"
                logger.warning(f"   行 {formatted_line}{modifier}: 重复 {count} 次")
            if len(duplicates) > 10:
                logger.warning(f"   ... 还有 {len(duplicates) - 10} 个重复")
        
        return instructions
    
    def _try_parse_instruction_line(self, line: str, line_width: int) -> Optional[EditInstruction]:
        """
        尝试解析单行编辑指令（多模式匹配）
        
        Args:
            line: 要解析的行
            line_width: 期望的行号宽度
            
        Returns:
            解析出的EditInstruction，如果解析失败返回None
        """
        # 清理行首尾的空白字符（但保留内容中的空白）
        line = line.strip()
        if not line:
            return None
        
        # 模式1: 标准格式 [cmd][line][content]（动态宽度）
        # 示例: r0005def hello():, +0010# comment
        match = re.match(rf'^([+\-r])(\d{{{line_width}}})(.*)', line)
        if match:
            modifier = match.group(1)
            try:
                line_num = int(match.group(2))
                content = match.group(3)
                return EditInstruction(
                    line_number=line_num,
                    modifier=modifier,
                    content=content
                )
            except ValueError:
                pass
        
        # 模式2: 备选格式 [line][cmd][content]（行号在前，动态宽度）
        # 示例: 0005rdef hello():, 0010+# comment
        match = re.match(rf'^(\d{{{line_width}}})([+\-r])(.*)', line)
        if match:
            try:
                line_num = int(match.group(1))
                modifier = match.group(2)
                content = match.group(3)
                
                # 处理可能存在的空格
                if content.startswith(' '):
                    content = content[1:]
                
                return EditInstruction(
                    line_number=line_num,
                    modifier=modifier,
                    content=content
                )
            except ValueError:
                pass
        
        # 模式3: 带空格的格式 [cmd] [line] [content]
        # 示例: r 0005 def hello():, + 0010 # comment
        match = re.match(rf'^([+\-r])\s+(\d{{{line_width}}})\s+(.*)', line)
        if match:
            modifier = match.group(1)
            try:
                line_num = int(match.group(2))
                content = match.group(3)
                return EditInstruction(
                    line_number=line_num,
                    modifier=modifier,
                    content=content
                )
            except ValueError:
                pass
        
        # 模式4: 兼容旧的4位格式（如果动态宽度不是4）
        if line_width != 4:
            match = re.match(r'^([+\-r])(\d{4})(.*)', line)
            if match:
                modifier = match.group(1)
                try:
                    line_num = int(match.group(2))
                    content = match.group(3)
                    return EditInstruction(
                        line_number=line_num,
                        modifier=modifier,
                        content=content
                    )
                except ValueError:
                    pass
        
        # 模式5: 带括号的格式 [cmd]([line])[content]
        # 示例: r(0005)def hello():, +(0010)# comment
        match = re.match(rf'^([+\-r])\((\d{{{line_width}}})\)(.*)', line)
        if match:
            modifier = match.group(1)
            try:
                line_num = int(match.group(2))
                content = match.group(3)
                return EditInstruction(
                    line_number=line_num,
                    modifier=modifier,
                    content=content
                )
            except ValueError:
                pass
        
        return None
    
    def _validate_instructions(
        self, 
        instructions: List[EditInstruction],
        original_code: str
    ) -> Tuple[bool, List[str]]:
        """
        验证编辑指令的有效性（增强版 - 使用LineNumberHandler）
        
        Args:
            instructions: 编辑指令列表
            original_code: 原始代码
            
        Returns:
            (是否有效, 错误列表)
        """
        errors = []
        warnings = []
        lines = original_code.split('\n')
        total_lines = len(lines)
        
        # 使用LineNumberHandler进行行号验证
        line_handler = LineNumberHandler(total_lines)
        
        # 按行号分组，检测同一行的操作组合
        line_operations = {}
        for inst in instructions:
            if inst.line_number not in line_operations:
                line_operations[inst.line_number] = []
            line_operations[inst.line_number].append(inst)
        
        for inst in instructions:
            # 使用LineNumberHandler验证行号范围和格式
            try:
                line_handler.format_line_number(inst.line_number)
            except ValueError as e:
                errors.append(f"行号验证失败: {e}")
                continue
            
            # 检查行号范围
            if inst.modifier in ['-', 'r']:
                if inst.line_number < 1 or inst.line_number > total_lines:
                    errors.append(
                        f"行号 {inst.line_number} 超出范围 (1-{total_lines})"
                    )
            elif inst.modifier == '+':
                if inst.line_number < 0 or inst.line_number > total_lines:
                    errors.append(
                        f"插入位置 {inst.line_number} 无效"
                    )
            
            # 检查修改符有效性
            if inst.modifier not in ['+', '-', 'r']:
                errors.append(f"无效的修改符: {inst.modifier}")
            
            # 检查内容
            # 注意：
            # - '+' 操作允许空内容（添加空行）
            # - 'r' 操作也允许空内容（替换为空行，用于格式化代码）
            # - '-' 操作不需要内容（删除操作）
            # 所以这里不做内容检查
        
        # 检查同一行号的操作组合是否合理
        for line_num, ops in line_operations.items():
            if len(ops) > 1:
                modifiers = [op.modifier for op in ops]
                
                # 检查：同一行有 r 和 +
                if 'r' in modifiers and '+' in modifiers:
                    # 检查内容是否相似（可能是错误）
                    r_ops = [op for op in ops if op.modifier == 'r']
                    plus_ops = [op for op in ops if op.modifier == '+']
                    
                    for r_op in r_ops:
                        for plus_op in plus_ops:
                            # 简单检查：如果内容相似度高，可能是LLM的错误
                            if r_op.content and plus_op.content:
                                r_clean = r_op.content.strip()
                                p_clean = plus_op.content.strip()
                                if r_clean and p_clean and (r_clean in p_clean or p_clean in r_clean):
                                    warnings.append(
                                        f"行 {line_num}: 同时使用 'r' 和 '+' 且内容相似，"
                                        f"可能是LLM理解错误。建议只使用 'r' 替换"
                                    )
                
                # 检查：同一行有 - 和 r（会自动修复，只需提示）
                if '-' in modifiers and 'r' in modifiers:
                    # 注意：这个冲突会在应用阶段自动修复（忽略delete，保留replace）
                    # 所以这里只是info级别的提示，不是warning
                    pass  # 不记录为warning，因为会自动修复
        
        # 输出警告（不阻止执行，但提醒用户）
        if warnings:
            logger.warning("⚠️  发现潜在的指令问题：")
            for warn in warnings:
                logger.warning(f"   {warn}")
        
        return len(errors) == 0, errors
    
    def _apply_instructions(
        self, 
        original_code: str,
        instructions: List[EditInstruction],
        transactional: bool = False
    ) -> Tuple[str, List[str]]:
        """
        应用编辑指令 - 使用SourceCode类管理行号映射
        
        改进策略：
        1. 使用SourceCode类维护原始行号映射
        2. 支持乱序应用任意编辑指令
        3. 每个指令的行号都是相对于原始代码，避免行号偏移问题
        4. 支持事务性应用，失败时自动回滚
        5. 应用成功的指令不需要重复应用
        
        Args:
            original_code: 原始代码
            instructions: 编辑指令列表
            transactional: 是否启用事务性应用（失败时回滚）
            
        Returns:
            (新代码, 错误列表)
        """
        errors = []
        total_lines = len(original_code.split('\n'))
        
        logger.info(f"🔨 开始应用编辑指令：原始代码 {total_lines} 行，共 {len(instructions)} 条指令")
        
        # 创建SourceCode对象
        source = SourceCode(original_code)
        
        # 统计指令类型
        op_stats = {'delete': 0, 'replace': 0, 'add': 0}
        for inst in instructions:
            if inst.modifier == 'r':
                op_stats['replace'] += 1
            elif inst.modifier == '-':
                op_stats['delete'] += 1
            elif inst.modifier == '+':
                op_stats['add'] += 1
        
        logger.info(f"  操作统计: {op_stats['delete']}删除, {op_stats['replace']}替换, {op_stats['add']}插入")
        
        # 应用所有指令
        applied_count = 0
        for inst in instructions:
            apply_errors = source.apply_edit_instruction(inst)
            if apply_errors:
                errors.extend(apply_errors)
                logger.warning(f"  ❌ {inst.modifier}{inst.line_number:04d} 应用失败: {apply_errors[0]}")
            else:
                applied_count += 1
                logger.debug(f"  ✅ {inst.modifier}{inst.line_number:04d} 应用成功")
        
        # 获取最终代码
        new_code = source.get_code()
        actual_lines = len(new_code.split('\n'))
        
        logger.info(f"应用完成：新代码 {actual_lines} 行（原始 {total_lines} 行）")
        logger.info(f"  成功应用 {applied_count}/{len(instructions)} 条指令")
        
        # 如果有错误处理
        if errors:
            logger.warning(f"应用指令时发生 {len(errors)} 个错误")
            
            # 事务性回滚：如果启用事务模式，返回原始代码
            if transactional:
                logger.error(f"事务性应用失败，自动回滚到原始代码")
                return original_code, errors
            
            # 非事务模式：保存中间结果供调试
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                    f.write(new_code)
                    logger.info(f"已保存中间结果到: {f.name}")
            except Exception as e:
                logger.debug(f"无法保存中间结果: {e}")
        
        return new_code, errors
    
    def _rollback_instructions(self, original_code: str, applied_instructions: List[EditInstruction], 
                               failure_point: int, error_reason: str) -> str:
        """
        回滚部分应用的指令
        
        Args:
            original_code: 原始代码
            applied_instructions: 已成功应用的指令列表
            failure_point: 失败的指令索引
            error_reason: 失败原因
            
        Returns:
            回滚后的代码（即原始代码）
        """
        logger.error(f"在指令 {failure_point} 处失败: {error_reason}")
        logger.info(f"已应用 {len(applied_instructions)} 条指令，开始回滚...")
        
        # 目前实现：直接返回原始代码
        # 未来可以优化为只回滚失败的部分
        logger.info("✓ 回滚完成：返回到原始代码状态")
        return original_code
    
    def _generate_diff(self, old_code: str, new_code: str) -> str:
        """生成diff"""
        import difflib
        
        old_lines = old_code.split('\n')
        new_lines = new_code.split('\n')
        
        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            lineterm='',
            n=3  # 上下文行数
        )
        
        return '\n'.join(diff)
    
    def apply_instruction_string(
        self, 
        original_code: str,
        instruction_string: str
    ) -> EditResult:
        """
        直接应用编辑指令字符串（不通过LLM）
        
        Args:
            original_code: 原始代码
            instruction_string: 编辑指令字符串
            
        Returns:
            EditResult
        """
        logger.info("应用预定义的编辑指令")
        
        # 解析指令
        instructions = self._parse_edit_instructions(instruction_string)
        
        if not instructions:
            return EditResult(
                success=False,
                new_code=original_code,
                applied_instructions=[],
                errors=["未能解析编辑指令"],
                diff=""
            )
        
        # 验证指令
        valid, errors = self._validate_instructions(instructions, original_code)
        if not valid:
            return EditResult(
                success=False,
                new_code=original_code,
                applied_instructions=[],
                errors=errors,
                diff=""
            )
        
        # 应用指令（启用事务性模式）
        new_code, apply_errors = self._apply_instructions(original_code, instructions, transactional=True)
        
        if apply_errors:
            return EditResult(
                success=False,
                new_code=original_code,
                applied_instructions=[],
                errors=apply_errors,
                diff=""
            )
        
        # 验证结果代码的结构完整性
        validator = CodeStructureValidator()
        validation_report = validator.get_comprehensive_report(new_code, "<edited>")
        
        if not validation_report['syntax_valid']:
            syntax_errors = [err for err in validation_report['errors'] if '语法错误' in err]
            return EditResult(
                success=False,
                new_code=original_code,
                applied_instructions=[],
                errors=[f"结果代码语法错误: {'; '.join(syntax_errors)}"],
                diff=""
            )
        
        if validation_report['warnings']:
            logger.warning(f"⚠️  结果代码存在警告: {validation_report['warnings']}")
        
        # 生成diff
        diff = self._generate_diff(original_code, new_code)
        
        return EditResult(
            success=True,
            new_code=new_code,
            applied_instructions=instructions,
            errors=[],
            diff=diff
        )


if __name__ == '__main__':
    """测试"""
    
    # 测试1: 解析编辑指令
    editor = LLMCodeEditor()
    
    instruction_text = """
+0002# 新增的注释
r0005    return x * 3  # 修改返回值
-0010
"""
    
    instructions = editor._parse_edit_instructions(instruction_text)
    print(f"解析了 {len(instructions)} 条指令:")
    for inst in instructions:
        print(f"  {inst}")
    
    # 测试2: 应用编辑指令
    original = """def calculate(x):
    result = x * 2
    return result

def other():
    pass
"""
    
    result = editor.apply_instruction_string(original, instruction_text)
    
    if result.success:
        print("\n✅ 编辑成功")
        print("\n新代码:")
        print(result.new_code)
        print("\nDiff:")
        print(result.diff)
    else:
        print(f"\n❌ 编辑失败: {result.errors}")
    
    # 测试3: CodeStructureValidator
    print("\n" + "="*50)
    print("测试 CodeStructureValidator")
    print("="*50)
    
    validator = CodeStructureValidator()
    
    # 测试有效的代码
    valid_code = """def hello():
    print("Hello, World!")
    return True
"""
    
    report = validator.get_comprehensive_report(valid_code, "test.py")
    print(f"有效代码验证结果:")
    print(f"  语法有效: {report['syntax_valid']}")
    print(f"  结构有效: {report['structure_valid']}")
    print(f"  导入有效: {report['import_valid']}")
    print(f"  错误数: {len(report['errors'])}")
    print(f"  警告数: {len(report['warnings'])}")
    
    # 测试有语法错误的代码
    invalid_code = """def hello(
    print("Hello, World!")
    return True
"""
    
    report = validator.get_comprehensive_report(invalid_code, "test.py")
    print(f"\n无效代码验证结果:")
    print(f"  语法有效: {report['syntax_valid']}")
    print(f"  错误: {report['errors']}")
    
    # 测试结构问题
    problematic_code = """def hello():
    print("Hello, World!")
        print("Bad indent")
    return True
"""
    
    report = validator.get_comprehensive_report(problematic_code, "test.py")
    print(f"\n结构问题代码验证结果:")
    print(f"  语法有效: {report['syntax_valid']}")
    print(f"  结构有效: {report['structure_valid']}")
    print(f"  错误: {report['errors']}")

