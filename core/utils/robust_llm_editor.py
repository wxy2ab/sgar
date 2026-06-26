#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RobustLLMEditor - 鲁棒性LLM代码编辑器

核心设计：
- Search/Replace Block 为主编辑协议（业界验证最可靠）
- 绝不依赖行号（LLM天生不擅长精确行号）
- 三策略级联回退保证高成功率
- 支持修改/查找/咨询三种模式
"""

import re
import ast
import difflib
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

try:
    from core.utils.log import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        _h = logging.StreamHandler()
        _h.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        logger.addHandler(_h)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CodeLocation:
    """代码位置信息"""
    name: str
    type: str  # 'function' | 'class' | 'import' | 'block'
    start_line: int
    end_line: int
    content: str
    relevance: float = 0.0

    def __repr__(self):
        return f"<{self.type} {self.name} L{self.start_line}-{self.end_line} rel={self.relevance:.2f}>"


@dataclass
class EditResult:
    """编辑结果"""
    success: bool
    new_code: str
    diff: str = ""
    applied_edits: List[str] = field(default_factory=list)
    failed_edits: List[str] = field(default_factory=list)
    strategy_used: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    diff_line_count: int = 0


@dataclass
class FindResult:
    """查找结果"""
    locations: List[CodeLocation] = field(default_factory=list)
    query: str = ""
    summary: str = ""


@dataclass
class ConsultResult:
    """咨询结果"""
    analysis: str = ""
    suggestions: List[str] = field(default_factory=list)
    related_code: List[CodeLocation] = field(default_factory=list)
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# CodeLocator – AST + keyword based code location
# ---------------------------------------------------------------------------

class CodeLocator:
    """根据查询定位代码中的相关段落"""

    def locate(self, code: str, query: str, *, llm_client=None) -> List[CodeLocation]:
        symbols = self._extract_symbols(code)
        keywords = self._extract_keywords(query)
        self._score_symbols(symbols, keywords, code, query)
        symbols.sort(key=lambda s: s.relevance, reverse=True)

        if llm_client and symbols and symbols[0].relevance < 0.3:
            llm_locs = self._llm_assisted_locate(code, query, symbols, llm_client)
            if llm_locs:
                return llm_locs

        return symbols

    # ---- AST extraction ----------------------------------------------------

    def _extract_symbols(self, code: str) -> List[CodeLocation]:
        symbols: List[CodeLocation] = []
        lines = code.split('\n')
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return self._fallback_extract(code)

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols.append(self._node_to_location(node, lines, 'function'))
            elif isinstance(node, ast.ClassDef):
                loc = self._node_to_location(node, lines, 'class')
                symbols.append(loc)
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_loc = self._node_to_location(item, lines, 'function')
                        method_loc.name = f"{node.name}.{item.name}"
                        symbols.append(method_loc)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                end = getattr(node, 'end_lineno', node.lineno) or node.lineno
                content = '\n'.join(lines[node.lineno - 1:end])
                names = [a.name for a in node.names]
                symbols.append(CodeLocation(
                    name=', '.join(names), type='import',
                    start_line=node.lineno, end_line=end,
                    content=content,
                ))
        return symbols

    def _node_to_location(self, node: ast.AST, lines: List[str], sym_type: str) -> CodeLocation:
        start = node.lineno
        end = getattr(node, 'end_lineno', start) or start

        dec_start = start
        if hasattr(node, 'decorator_list') and node.decorator_list:
            dec_start = node.decorator_list[0].lineno

        content = '\n'.join(lines[dec_start - 1:end])
        return CodeLocation(
            name=getattr(node, 'name', '?'),
            type=sym_type,
            start_line=dec_start,
            end_line=end,
            content=content,
        )

    def _fallback_extract(self, code: str) -> List[CodeLocation]:
        """AST失败时用正则提取"""
        symbols: List[CodeLocation] = []
        lines = code.split('\n')
        pattern = re.compile(r'^(\s*)(def |async def |class )(\w+)')
        for i, line in enumerate(lines):
            m = pattern.match(line)
            if m:
                indent = len(m.group(1))
                name = m.group(3)
                sym_type = 'class' if 'class ' in m.group(2) else 'function'
                end = i + 1
                for j in range(i + 1, len(lines)):
                    stripped = lines[j]
                    if stripped.strip() == '':
                        continue
                    cur_indent = len(stripped) - len(stripped.lstrip())
                    if cur_indent <= indent and stripped.strip():
                        break
                    end = j + 1
                content = '\n'.join(lines[i:end])
                symbols.append(CodeLocation(
                    name=name, type=sym_type,
                    start_line=i + 1, end_line=end,
                    content=content,
                ))
        return symbols

    # ---- keyword scoring ---------------------------------------------------

    def _extract_keywords(self, query: str) -> List[str]:
        stop = {'的', '和', '或', '是', '在', '了', '把', '将', '对', '被', '用',
                'the', 'and', 'or', 'is', 'in', 'to', 'a', 'an', 'of', 'for',
                'with', 'this', 'that', 'from', 'not', 'but', 'are', 'be',
                '函数', '方法', '类', '代码', '修改', '添加', '删除', '修复',
                '错误', '问题', '优化', '改进', '需要', '请', '帮', '我'}
        tokens = re.findall(r'[a-zA-Z_]\w*', query)
        cn_tokens = re.findall(r'[\u4e00-\u9fff]+', query)
        all_tokens = tokens + cn_tokens
        return [t for t in all_tokens if t.lower() not in stop and len(t) > 1]

    def _score_symbols(self, symbols: List[CodeLocation], keywords: List[str],
                       code: str, query: str) -> None:
        query_lower = query.lower()
        for sym in symbols:
            score = 0.0
            name_lower = sym.name.lower()
            content_lower = sym.content.lower()

            for kw in keywords:
                kw_l = kw.lower()
                if kw_l in name_lower:
                    score += 0.4
                if kw_l in content_lower:
                    score += 0.15

            if sym.name.lower() in query_lower:
                score += 0.5

            if any(term in query_lower for term in ['error', 'bug', '错误', '异常', 'exception']):
                if any(w in content_lower for w in ['try', 'except', 'raise', 'error']):
                    score += 0.1

            sym.relevance = min(1.0, score)

    # ---- LLM-assisted locate -----------------------------------------------

    def _llm_assisted_locate(self, code: str, query: str,
                             symbols: List[CodeLocation],
                             llm_client) -> List[CodeLocation]:
        symbol_list = '\n'.join(
            f"  {i+1}. [{s.type}] {s.name} (L{s.start_line}-{s.end_line})"
            for i, s in enumerate(symbols)
        )
        prompt = (
            f"以下代码包含这些符号：\n{symbol_list}\n\n"
            f"用户查询：{query}\n\n"
            f"请返回与查询最相关的符号编号（用逗号分隔），只返回数字，不要其他内容。"
        )
        try:
            resp = llm_client.one_chat(prompt)
            nums = [int(x.strip()) for x in re.findall(r'\d+', resp)]
            result = []
            for n in nums:
                if 1 <= n <= len(symbols):
                    s = symbols[n - 1]
                    s.relevance = max(s.relevance, 0.8)
                    result.append(s)
            return result if result else symbols
        except Exception as e:
            logger.debug(f"LLM辅助定位失败: {e}")
            return symbols


# ---------------------------------------------------------------------------
# CodeValidator
# ---------------------------------------------------------------------------

class CodeValidator:
    """代码验证器"""

    def validate_syntax(self, code: str) -> Tuple[bool, List[str]]:
        try:
            compile(code, '<editor>', 'exec')
            return True, []
        except SyntaxError as e:
            msg = f"SyntaxError L{e.lineno}: {e.msg}"
            return False, [msg]

    def validate_integrity(self, old_code: str, new_code: str,
                           *, protected_names: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
        errors: List[str] = []

        old_len = len(old_code)
        new_len = len(new_code)
        if old_len > 500 and new_len < old_len * 0.3:
            errors.append(f"代码大幅缩减 ({old_len}->{new_len} chars), 可能丢失内容")

        if protected_names:
            for name in protected_names:
                pattern_func = rf'def\s+{re.escape(name)}\s*\('
                pattern_cls = rf'class\s+{re.escape(name)}\b'
                if re.search(pattern_func, old_code) and not re.search(pattern_func, new_code):
                    errors.append(f"函数 '{name}' 丢失")
                if re.search(pattern_cls, old_code) and not re.search(pattern_cls, new_code):
                    errors.append(f"类 '{name}' 丢失")

        return len(errors) == 0, errors

    def extract_protected_names(self, code: str) -> List[str]:
        names: List[str] = []
        try:
            tree = ast.parse(code)
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    names.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    names.append(node.name)
        except SyntaxError:
            names = re.findall(r'(?:def|class)\s+(\w+)', code)
        return names

    def full_validate(self, old_code: str, new_code: str) -> Tuple[bool, List[str], List[str]]:
        errors: List[str] = []
        warnings: List[str] = []

        ok, syn_errs = self.validate_syntax(new_code)
        if not ok:
            return False, syn_errs, warnings

        protected = self.extract_protected_names(old_code)
        ok, int_errs = self.validate_integrity(old_code, new_code, protected_names=protected)
        if not ok:
            errors.extend(int_errs)

        old_lines = len(old_code.split('\n'))
        new_lines = len(new_code.split('\n'))
        if old_lines > 0 and abs(new_lines - old_lines) / old_lines > 0.5:
            warnings.append(f"行数变化较大: {old_lines} -> {new_lines}")

        return len(errors) == 0, errors, warnings


# ---------------------------------------------------------------------------
# Search/Replace parser and matcher
# ---------------------------------------------------------------------------

class SearchReplaceMatcher:
    """解析并应用 Search/Replace 块"""

    BLOCK_PATTERN = re.compile(
        r'<{6,7}\s*SEARCH\s*\n(.*?)\n={6,7}\s*\n(.*?)\n>{6,7}\s*REPLACE',
        re.DOTALL
    )

    def parse_blocks(self, text: str) -> List[Tuple[str, str]]:
        blocks: List[Tuple[str, str]] = []
        for m in self.BLOCK_PATTERN.finditer(text):
            search = m.group(1)
            replace = m.group(2)
            blocks.append((search, replace))
        if not blocks:
            blocks = self._try_alternative_parse(text)
        return blocks

    def _try_alternative_parse(self, text: str) -> List[Tuple[str, str]]:
        """支持更宽松的标记格式"""
        blocks: List[Tuple[str, str]] = []
        alt = re.compile(
            r'```*\s*(?:SEARCH|search|原代码|原始代码)\s*\n(.*?)\n```*\s*\n'
            r'```*\s*(?:REPLACE|replace|新代码|修改后代码)\s*\n(.*?)\n```*',
            re.DOTALL
        )
        for m in alt.finditer(text):
            blocks.append((m.group(1), m.group(2)))
        return blocks

    def apply_blocks(self, code: str, blocks: List[Tuple[str, str]]) -> Tuple[str, List[str], List[str]]:
        applied: List[str] = []
        failed: List[str] = []
        current = code

        for idx, (search, replace) in enumerate(blocks):
            new_code = self._apply_single(current, search, replace)
            preview = search.strip().split('\n')[0][:60]
            if new_code is None:
                failed.append(f"未匹配: {preview}...")
                continue
            try:
                compile(new_code, '<block_check>', 'exec')
                current = new_code
                applied.append(f"替换: {preview}...")
            except SyntaxError:
                repaired = self._try_indent_repair(current, search, replace)
                if repaired is not None:
                    current = repaired
                    applied.append(f"替换(缩进修正): {preview}...")
                else:
                    failed.append(f"语法错误回滚: {preview}...")

        return current, applied, failed

    def _try_indent_repair(self, code: str, search: str, replace: str) -> Optional[str]:
        """When normal indent alignment fails, probe common indent levels."""
        already_tried = _get_base_indent(search)
        match_indent = self._detect_match_indent(code, search)

        candidates: List[int] = []
        if match_indent is not None:
            for delta in (0, 4, -4, 8):
                candidates.append(match_indent + delta)
        for level in (0, 4, 8, 12, 16, 20):
            candidates.append(level)

        seen = {already_tried}
        for target in candidates:
            if target < 0 or target in seen:
                continue
            seen.add(target)
            adjusted = _reindent(replace, target)
            result = self._apply_single(code, search, adjusted, _skip_reindent=True)
            if result is None:
                continue
            try:
                compile(result, '<indent_repair>', 'exec')
                logger.info(f"[RobustEditor] 缩进探测修复成功: {already_tried} -> {target}")
                return result
            except SyntaxError:
                continue
        return None

    def _detect_match_indent(self, code: str, search: str) -> Optional[int]:
        """Detect the indent level of the region in code that search targets."""
        if search in code:
            return _get_base_indent(search)
        first_line = search.split('\n')[0].strip()
        if not first_line:
            return None
        for line in code.split('\n'):
            if line.strip() == first_line:
                return len(line) - len(line.lstrip(' '))
        for line in code.split('\n'):
            if first_line in line:
                return len(line) - len(line.lstrip(' '))
        return None

    def _apply_single(self, code: str, search: str, replace: str,
                      *, _skip_reindent: bool = False) -> Optional[str]:
        if not _skip_reindent:
            replace = _reindent(replace, _get_base_indent(search))

        # 1) exact match
        if search in code:
            return code.replace(search, replace, 1)

        # 2) strip trailing whitespace per line then match
        def strip_trailing(s: str) -> str:
            return '\n'.join(l.rstrip() for l in s.split('\n'))

        code_stripped = strip_trailing(code)
        search_stripped = strip_trailing(search)
        if search_stripped in code_stripped:
            idx = code_stripped.index(search_stripped)
            pre_lines = code_stripped[:idx].count('\n')
            search_line_count = search_stripped.count('\n') + 1
            code_lines = code.split('\n')
            before = '\n'.join(code_lines[:pre_lines])
            after = '\n'.join(code_lines[pre_lines + search_line_count:])
            parts = [before, replace, after] if before else [replace, after]
            return '\n'.join(p for p in parts if p is not None)

        # 3) normalized whitespace match
        result = self._normalized_match_replace(code, search, replace)
        if result is not None:
            return result

        # 4) fuzzy line match (similarity >= 0.9)
        return self._fuzzy_match_replace(code, search, replace, threshold=0.9)

    def _normalized_match_replace(self, code: str, search: str, replace: str) -> Optional[str]:
        def norm(s: str) -> str:
            return re.sub(r'[ \t]+', ' ', s)

        code_lines = code.split('\n')
        search_lines = search.split('\n')
        search_norm = [norm(l) for l in search_lines]
        code_norm = [norm(l) for l in code_lines]
        w = len(search_norm)

        for i in range(len(code_norm) - w + 1):
            if code_norm[i:i + w] == search_norm:
                original_indent = _get_base_indent('\n'.join(code_lines[i:i + w]))
                aligned = _reindent(replace, original_indent)
                new_lines = code_lines[:i] + aligned.split('\n') + code_lines[i + w:]
                return '\n'.join(new_lines)
        return None

    def _fuzzy_match_replace(self, code: str, search: str, replace: str,
                             threshold: float = 0.85) -> Optional[str]:
        code_lines = code.split('\n')
        search_lines = [l for l in search.split('\n')]
        w = len(search_lines)
        if w == 0 or w > len(code_lines):
            return None

        best_idx = -1
        best_score = 0.0

        for i in range(len(code_lines) - w + 1):
            window = code_lines[i:i + w]
            score = self._block_similarity(window, search_lines)
            if score > best_score:
                best_score = score
                best_idx = i

        if best_score >= threshold and best_idx >= 0:
            original_indent = _get_base_indent('\n'.join(code_lines[best_idx:best_idx + w]))
            aligned = _reindent(replace, original_indent)
            new_lines = code_lines[:best_idx] + aligned.split('\n') + code_lines[best_idx + w:]
            return '\n'.join(new_lines)
        return None

    def _block_similarity(self, lines_a: List[str], lines_b: List[str]) -> float:
        if len(lines_a) != len(lines_b):
            return 0.0
        total = 0.0
        for a, b in zip(lines_a, lines_b):
            total += difflib.SequenceMatcher(None, a.rstrip(), b.rstrip()).ratio()
        return total / len(lines_a)


# ---------------------------------------------------------------------------
# Edit strategies
# ---------------------------------------------------------------------------

class _Strategy1_SearchReplace:
    """主策略：Search/Replace Block"""

    PROMPT_TEMPLATE = """你是代码编辑专家。根据用户指令修改Python代码。

## 规则（必须严格遵守）
1. 用 SEARCH/REPLACE 块描述每处修改
2. SEARCH 内容必须是原代码中**逐字存在**的片段（包含缩进空格）
3. SEARCH 要包含足够上下文以唯一定位（至少3行）
4. REPLACE 中的缩进必须与 SEARCH 中对应位置的缩进完全一致（空格数相同）
5. 不要破坏 try/except/finally、if/elif/else、with 等复合语句的完整性
6. 如果修改涉及 try 块内部的代码，SEARCH 应包含完整的 try...except 结构
7. 只输出 SEARCH/REPLACE 块，不要输出任何其他内容
8. **最小修改原则**：只修改与指令直接相关的代码，不要做无关的重构或格式调整

## 格式
<<<<<<< SEARCH
（原代码片段，逐字匹配，包含缩进）
=======
（修改后的代码，保持相同缩进层级）
>>>>>>> REPLACE

## 原始代码（左侧行号仅供参考定位，SEARCH 块中不要包含行号前缀）
```python
{numbered_code}
```

## 修改指令
{instruction}
"""

    def __init__(self, llm_client, matcher: SearchReplaceMatcher, validator: CodeValidator):
        self.llm = llm_client
        self.matcher = matcher
        self.validator = validator

    def execute(self, code: str, instruction: str, context: str = "") -> EditResult:
        numbered_code = '\n'.join(
            f"{i+1:4d}| {line}" for i, line in enumerate(code.split('\n'))
        )
        prompt = self.PROMPT_TEMPLATE.format(
            numbered_code=numbered_code, instruction=instruction
        )
        if context:
            prompt += f"\n\n## 额外上下文\n{context}"

        try:
            resp = self.llm.one_chat(prompt)
        except Exception as e:
            return EditResult(success=False, new_code=code, errors=[f"LLM调用失败: {e}"])

        blocks = self.matcher.parse_blocks(resp)
        if not blocks:
            return EditResult(success=False, new_code=code,
                              errors=["LLM未返回有效的SEARCH/REPLACE块"],
                              strategy_used="search_replace")

        new_code, applied, failed = self.matcher.apply_blocks(code, blocks)

        if not applied:
            return EditResult(success=False, new_code=code,
                              applied_edits=applied, failed_edits=failed,
                              errors=["所有SEARCH块均未匹配到原代码"],
                              strategy_used="search_replace")

        ok, syn_errs = self.validator.validate_syntax(new_code)
        if not ok:
            return EditResult(success=False, new_code=new_code,
                              applied_edits=applied, failed_edits=failed,
                              errors=syn_errs, strategy_used="search_replace")

        diff = _make_diff(code, new_code)
        return EditResult(
            success=True, new_code=new_code, diff=diff,
            applied_edits=applied, failed_edits=failed,
            strategy_used="search_replace",
            warnings=[f"{len(failed)}个块未匹配"] if failed else [],
        )


class _Strategy2_FunctionReplace:
    """二级策略：函数/类整体替换"""

    PROMPT_TEMPLATE = """你是代码编辑专家。请根据指令修改以下函数/类。

## 规则
1. 返回修改后的**完整**函数或类定义（包含装饰器、签名、函数体）
2. 只返回修改后的代码，不要返回其他解释
3. 用 ```python 代码块包裹
4. 保持原有缩进风格

## 需要修改的代码
```python
{target_code}
```

## 修改指令
{instruction}
"""

    def __init__(self, llm_client, locator: CodeLocator, validator: CodeValidator):
        self.llm = llm_client
        self.locator = locator
        self.validator = validator

    def execute(self, code: str, instruction: str, context: str = "") -> EditResult:
        locations = self.locator.locate(code, instruction)
        targets = [loc for loc in locations if loc.type in ('function', 'class') and loc.relevance > 0.1]

        if not targets:
            return EditResult(success=False, new_code=code,
                              errors=["未找到与指令相关的函数/类"],
                              strategy_used="function_replace")

        target = targets[0]
        prompt = self.PROMPT_TEMPLATE.format(
            target_code=target.content, instruction=instruction
        )
        if context:
            prompt += f"\n\n## 额外上下文\n{context}"

        try:
            resp = self.llm.one_chat(prompt)
        except Exception as e:
            return EditResult(success=False, new_code=code, errors=[f"LLM调用失败: {e}"],
                              strategy_used="function_replace")

        new_fragment = _extract_code_block(resp)
        if not new_fragment:
            return EditResult(success=False, new_code=code,
                              errors=["LLM未返回有效的代码块"],
                              strategy_used="function_replace")

        lines = code.split('\n')
        new_lines = (
            lines[:target.start_line - 1]
            + new_fragment.split('\n')
            + lines[target.end_line:]
        )
        new_code = '\n'.join(new_lines)

        ok, syn_errs = self.validator.validate_syntax(new_code)
        if not ok:
            return EditResult(success=False, new_code=new_code, errors=syn_errs,
                              strategy_used="function_replace")

        diff = _make_diff(code, new_code)
        return EditResult(
            success=True, new_code=new_code, diff=diff,
            applied_edits=[f"替换 {target.type} '{target.name}'"],
            strategy_used="function_replace",
        )


class _Strategy3_FullRewrite:
    """末级回退策略：完整文件重写"""

    MAX_LINES = 800

    PROMPT_TEMPLATE = """你是代码编辑专家。请根据指令修改以下代码并返回完整的修改后代码。

## 重要规则
1. 返回修改后的**完整**文件代码（不能省略任何部分）
2. 不要使用 `...` 或注释代替省略的代码
3. 用 ```python 代码块包裹
4. 保持所有现有函数和类
5. 除了代码块外不要输出任何其他内容

## 原始代码
```python
{code}
```

## 修改指令
{instruction}
"""

    def __init__(self, llm_client, validator: CodeValidator):
        self.llm = llm_client
        self.validator = validator

    def execute(self, code: str, instruction: str, context: str = "") -> EditResult:
        line_count = len(code.split('\n'))
        if line_count > self.MAX_LINES:
            return EditResult(success=False, new_code=code,
                              errors=[f"文件过大({line_count}行), 全量重写策略仅支持{self.MAX_LINES}行以内"],
                              strategy_used="full_rewrite")

        prompt = self.PROMPT_TEMPLATE.format(code=code, instruction=instruction)
        if context:
            prompt += f"\n\n## 额外上下文\n{context}"

        try:
            resp = self.llm.one_chat(prompt)
        except Exception as e:
            return EditResult(success=False, new_code=code, errors=[f"LLM调用失败: {e}"],
                              strategy_used="full_rewrite")

        new_code = _extract_code_block(resp)
        if not new_code:
            return EditResult(success=False, new_code=code,
                              errors=["LLM未返回有效的代码块"],
                              strategy_used="full_rewrite")

        ok, errors, warnings = self.validator.full_validate(code, new_code)
        if not ok:
            return EditResult(success=False, new_code=code, errors=errors,
                              warnings=warnings, strategy_used="full_rewrite")

        if code.strip() == new_code.strip():
            return EditResult(success=False, new_code=code,
                              errors=["LLM返回的代码与原代码相同，未进行任何修改"],
                              strategy_used="full_rewrite")

        diff = _make_diff(code, new_code)
        return EditResult(
            success=True, new_code=new_code, diff=diff,
            applied_edits=["全量重写"],
            strategy_used="full_rewrite",
            warnings=warnings,
        )


# ---------------------------------------------------------------------------
# RobustLLMEditor – main entry point
# ---------------------------------------------------------------------------

class RobustLLMEditor:
    """
    鲁棒性LLM代码编辑器

    三种工作模式:
        - modify : 定位 + 修改 + 验证
        - find   : 定位相关代码
        - consult: 分析 + 返回建议（不修改代码）
    """

    def __init__(self, llm_client=None):
        self.llm = llm_client
        self.locator = CodeLocator()
        self.validator = CodeValidator()
        self.matcher = SearchReplaceMatcher()

    # ---- 修改模式 --------------------------------------------------------

    def modify(self, code: str, instruction: str, *,
               context: str = "", file_path: str = "",
               max_retries: int = 2) -> EditResult:
        if not self.llm:
            return EditResult(success=False, new_code=code, errors=["LLM客户端不可用"])

        logger.info(f"[RobustEditor] modify: instruction='{instruction[:80]}...'")

        strategies = [
            ("search_replace", lambda inst: _Strategy1_SearchReplace(
                self.llm, self.matcher, self.validator
            ).execute(code, inst, context)),
            ("function_replace", lambda inst: _Strategy2_FunctionReplace(
                self.llm, self.locator, self.validator
            ).execute(code, inst, context)),
            ("full_rewrite", lambda inst: _Strategy3_FullRewrite(
                self.llm, self.validator
            ).execute(code, inst, context)),
        ]

        all_errors: List[str] = []

        for strategy_name, run_fn in strategies:
            last_errors: List[str] = []
            last_failed_code: Optional[str] = None

            for attempt in range(max_retries):
                current_instruction = instruction
                if last_errors:
                    current_instruction = self._build_retry_instruction(
                        instruction, last_errors, last_failed_code
                    )

                logger.info(f"[RobustEditor] 尝试 {strategy_name} (attempt {attempt+1}/{max_retries})")
                result = run_fn(current_instruction)

                if result.success:
                    ok, int_errs = self.validator.validate_integrity(
                        code, result.new_code,
                        protected_names=self.validator.extract_protected_names(code)
                    )
                    if not ok:
                        result.warnings.extend(int_errs)
                        logger.warning(f"[RobustEditor] 完整性警告: {int_errs}")

                    result.diff_line_count = _count_diff_lines(code, result.new_code)
                    logger.info(f"[RobustEditor] 成功: strategy={strategy_name}")
                    return result

                all_errors.extend(result.errors)
                last_errors = result.errors
                last_failed_code = result.new_code if result.new_code != code else None
                logger.warning(f"[RobustEditor] {strategy_name} attempt {attempt+1} 失败: {result.errors}")

        logger.error(f"[RobustEditor] 所有策略均失败")
        return EditResult(
            success=False, new_code=code,
            errors=all_errors[-5:],
            strategy_used="all_failed",
        )

    def _build_retry_instruction(self, instruction: str, errors: List[str],
                                 failed_code: Optional[str] = None) -> str:
        error_text = '\n'.join(errors)
        parts = [
            instruction,
            f"\n\n## 上次尝试的错误\n{error_text}",
            "\n请仔细检查缩进和语法，避免破坏 try/except/finally 等复合语句结构。",
        ]
        if failed_code:
            error_context = _extract_error_context(failed_code, errors)
            if error_context:
                parts.append(f"\n\n## 错误位置附近的代码\n{error_context}")
        return ''.join(parts)

    # ---- 查找模式 --------------------------------------------------------

    def find(self, code: str, query: str, *, use_llm: bool = False) -> FindResult:
        logger.info(f"[RobustEditor] find: query='{query[:80]}...'")

        llm = self.llm if use_llm else None
        locations = self.locator.locate(code, query, llm_client=llm)
        relevant = [loc for loc in locations if loc.relevance > 0.05]

        summary = ""
        if use_llm and self.llm and relevant:
            summary = self._generate_find_summary(code, query, relevant)

        return FindResult(locations=relevant, query=query, summary=summary)

    def _generate_find_summary(self, code: str, query: str,
                               locations: List[CodeLocation]) -> str:
        loc_desc = '\n'.join(
            f"- [{l.type}] {l.name} (L{l.start_line}-{l.end_line}, 相关度={l.relevance:.2f})"
            for l in locations[:5]
        )
        prompt = (
            f"用户查询: {query}\n\n"
            f"找到以下相关代码段:\n{loc_desc}\n\n"
            f"请用1-2句话简要总结这些代码段与查询的关系。只返回总结，不要返回代码。"
        )
        try:
            return self.llm.one_chat(prompt).strip()
        except Exception:
            return ""

    # ---- 咨询模式 --------------------------------------------------------

    def consult(self, code: str, query: str) -> ConsultResult:
        if not self.llm:
            return ConsultResult(analysis="LLM客户端不可用")

        logger.info(f"[RobustEditor] consult: query='{query[:80]}...'")

        locations = self.locator.locate(code, query, llm_client=self.llm)
        relevant = [loc for loc in locations if loc.relevance > 0.05][:5]

        relevant_code = '\n\n'.join(
            f"# --- {loc.type}: {loc.name} (L{loc.start_line}-{loc.end_line}) ---\n{loc.content}"
            for loc in relevant
        ) if relevant else code[:3000]

        prompt = f"""你是代码分析专家。分析以下代码并回答用户的问题。

## 用户问题
{query}

## 相关代码
```python
{relevant_code}
```

## 要求
请返回以下格式的JSON（不要返回其他内容）:
{{
    "analysis": "分析结果（详细说明问题所在和原因）",
    "suggestions": ["建议1", "建议2", ...],
    "confidence": 0.0到1.0之间的数字
}}
"""
        try:
            resp = self.llm.one_chat(prompt)
            return self._parse_consult_response(resp, relevant)
        except Exception as e:
            return ConsultResult(
                analysis=f"咨询失败: {e}",
                related_code=relevant,
            )

    def _parse_consult_response(self, resp: str, locations: List[CodeLocation]) -> ConsultResult:
        import json

        text = resp.strip()
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                return ConsultResult(
                    analysis=data.get('analysis', ''),
                    suggestions=data.get('suggestions', []),
                    related_code=locations,
                    confidence=float(data.get('confidence', 0.5)),
                )
            except (json.JSONDecodeError, ValueError):
                pass

        return ConsultResult(
            analysis=text,
            suggestions=[],
            related_code=locations,
            confidence=0.5,
        )


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _extract_code_block(text: str) -> Optional[str]:
    """从LLM响应中提取代码块"""
    m = re.search(r'```python\s*\n(.*?)\n```', text, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r'```\s*\n(.*?)\n```', text, re.DOTALL)
    if m:
        return m.group(1)
    if text.strip().startswith(('import ', 'from ', 'def ', 'class ', '#')):
        return text.strip()
    return None


def _make_diff(old: str, new: str) -> str:
    return '\n'.join(difflib.unified_diff(
        old.split('\n'), new.split('\n'),
        lineterm='', n=3,
    ))


def _get_base_indent(text: str) -> int:
    """Get indentation level of first non-empty line."""
    for line in text.split('\n'):
        if line.strip():
            return len(line) - len(line.lstrip(' '))
    return 0


def _reindent(text: str, target_indent: int) -> str:
    """Re-indent text so its base indentation becomes *target_indent*."""
    lines = text.split('\n')
    base = _get_base_indent(text)
    delta = target_indent - base
    if delta == 0:
        return text
    result: List[str] = []
    for line in lines:
        if not line.strip():
            result.append(line)
        elif delta > 0:
            result.append(' ' * delta + line)
        else:
            stripped = line.lstrip(' ')
            cur = len(line) - len(stripped)
            result.append(' ' * max(0, cur + delta) + stripped)
    return '\n'.join(result)


def _count_diff_lines(old_code: str, new_code: str) -> int:
    """Count changed lines in unified diff."""
    diff = list(difflib.unified_diff(
        old_code.split('\n'), new_code.split('\n'), lineterm='', n=0,
    ))
    return sum(
        1 for line in diff
        if (line.startswith('+') or line.startswith('-'))
        and not line.startswith('+++') and not line.startswith('---')
    )


def _extract_error_context(code: str, errors: List[str],
                           context_lines: int = 5) -> str:
    """Extract code lines around error locations mentioned in error messages."""
    lines = code.split('\n')
    contexts: List[str] = []
    seen: set = set()
    for err in errors:
        m = re.search(r'L(\d+)', err)
        if not m:
            continue
        lineno = int(m.group(1))
        if lineno in seen:
            continue
        seen.add(lineno)
        start = max(0, lineno - context_lines - 1)
        end = min(len(lines), lineno + context_lines)
        snippet = '\n'.join(
            f"{'>>>' if i + 1 == lineno else '   '} {i+1:4d}| {lines[i]}"
            for i in range(start, end)
        )
        contexts.append(f"错误 '{err}' 附近:\n{snippet}")
    return '\n\n'.join(contexts)
