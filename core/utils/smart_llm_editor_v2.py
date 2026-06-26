#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SmartLLMEditorV2 - 智能LLM代码编辑器 V2

四层流水线架构:
    A. 索引感知层 - TreeSitterIndexer / RepoMapper / CodeSearcher
    B. 规划决策层 - EditPlanner / ContextManager
    C. 执行编辑层 - SearchReplaceEngine
    D. 验证纠错层 - EditValidator / CorrectionLoop

核心使命: 在有限的上下文窗口内，精准定位代码并安全地实施局部修改
"""

import re
import os
import ast
import difflib
from dataclasses import dataclass, field
from typing import (
    List, Optional, Tuple, Dict, Any, Callable, Set,
)

# ---------------------------------------------------------------------------
# Optional heavy deps – graceful degradation
# ---------------------------------------------------------------------------

try:
    import tree_sitter_python as _tspython
    from tree_sitter import Language as _TSLanguage, Parser as _TSParser
    _TS_AVAILABLE = True
except ImportError:
    _TS_AVAILABLE = False

try:
    from grep_ast import TreeContext as _GrepTreeContext
    _GREP_AST_AVAILABLE = True
except ImportError:
    _GREP_AST_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi as _BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

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


# ===================================================================
# Data classes
# ===================================================================

@dataclass
class SymbolDef:
    """代码符号定义"""
    name: str
    type: str               # 'function' | 'class' | 'method' | 'import'
    start_line: int          # 1-based
    end_line: int            # 1-based inclusive
    signature: str = ""      # e.g. "def foo(x, y) -> int:"
    parent: str = ""         # enclosing class name for methods
    children: List['SymbolDef'] = field(default_factory=list)

    def qualified_name(self) -> str:
        return f"{self.parent}.{self.name}" if self.parent else self.name

    def __repr__(self):
        return f"<{self.type} {self.qualified_name()} L{self.start_line}-{self.end_line}>"


@dataclass
class SearchHit:
    """代码搜索命中"""
    file_path: str
    score: float
    snippet: str = ""
    symbol: Optional[SymbolDef] = None


@dataclass
class SearchReplaceBlock:
    """一个 SEARCH/REPLACE 块"""
    search: str
    replace: str
    file_hint: str = ""


@dataclass
class ApplyResult:
    """单次 apply 的结果"""
    success: bool
    new_code: str
    applied: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    strategy_used: str = ""


@dataclass
class ValidationResult:
    """验证结果"""
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class EditPlan:
    """修改计划"""
    target_files: List[str] = field(default_factory=list)
    modifications: List[Dict[str, str]] = field(default_factory=list)
    summary: str = ""


@dataclass
class EditResultV2:
    """编辑结果 V2"""
    success: bool
    new_code: str
    diff: str = ""
    applied_edits: List[str] = field(default_factory=list)
    failed_edits: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    rounds_used: int = 0
    strategy_used: str = ""


@dataclass
class ContextFile:
    """上下文中的文件"""
    path: str
    content: str
    relevance: float = 0.0
    token_estimate: int = 0


# ===================================================================
# A. 索引感知层
# ===================================================================

# -------------------------------------------------------------------
# A1. TreeSitterIndexer
# -------------------------------------------------------------------

class TreeSitterIndexer:
    """tree-sitter 驱动的代码索引器，降级到 Python ast"""

    def __init__(self):
        self._parser: Optional[Any] = None
        self._language: Optional[Any] = None
        if _TS_AVAILABLE:
            try:
                self._language = _TSLanguage(_tspython.language())
                self._parser = _TSParser(self._language)
            except Exception as e:
                logger.warning(f"tree-sitter 初始化失败, 降级到 ast: {e}")
                self._parser = None

    @property
    def backend(self) -> str:
        return "tree-sitter" if self._parser else "ast"

    # -- public API ---------------------------------------------------

    def parse_file(self, code: str, language: str = "python") -> List[SymbolDef]:
        if self._parser and language == "python":
            return self._ts_parse(code)
        return self._ast_parse(code)

    def extract_signatures(self, code: str, language: str = "python") -> str:
        symbols = self.parse_file(code, language)
        return self._symbols_to_signature_text(symbols)

    def get_symbol_at_line(self, code: str, line: int,
                           language: str = "python") -> Optional[SymbolDef]:
        symbols = self.parse_file(code, language)
        best: Optional[SymbolDef] = None
        for s in symbols:
            if s.start_line <= line <= s.end_line:
                if best is None or (s.end_line - s.start_line) < (best.end_line - best.start_line):
                    best = s
            for c in s.children:
                if c.start_line <= line <= c.end_line:
                    if best is None or (c.end_line - c.start_line) < (best.end_line - best.start_line):
                        best = c
        return best

    def get_symbol_range(self, code: str, symbol_name: str,
                         language: str = "python") -> Optional[Tuple[int, int]]:
        """返回符号的 (start_line, end_line) 1-based"""
        symbols = self.parse_file(code, language)
        for s in self._flatten(symbols):
            if s.name == symbol_name or s.qualified_name() == symbol_name:
                return (s.start_line, s.end_line)
        return None

    # -- tree-sitter backend ------------------------------------------

    def _ts_parse(self, code: str) -> List[SymbolDef]:
        tree = self._parser.parse(code.encode('utf-8'))
        root = tree.root_node
        symbols: List[SymbolDef] = []
        for child in root.children:
            sym = self._ts_visit(child, code)
            if sym:
                symbols.append(sym)
        return symbols

    def _ts_visit(self, node, code: str, parent_name: str = "") -> Optional[SymbolDef]:
        ntype = node.type
        if ntype == 'function_definition':
            return self._ts_function(node, code, parent_name)
        elif ntype == 'decorated_definition':
            for child in node.children:
                if child.type in ('function_definition', 'class_definition'):
                    sym = self._ts_visit(child, code, parent_name)
                    if sym:
                        sym.start_line = node.start_point[0] + 1
                    return sym
        elif ntype == 'class_definition':
            return self._ts_class(node, code)
        elif ntype in ('import_statement', 'import_from_statement'):
            start = node.start_point[0] + 1
            end = node.end_point[0] + 1
            text = code.encode('utf-8')[node.start_byte:node.end_byte].decode('utf-8', errors='replace')
            return SymbolDef(
                name=text.split()[1] if len(text.split()) > 1 else text,
                type='import', start_line=start, end_line=end,
                signature=text.strip(),
            )
        return None

    def _ts_function(self, node, code: str, parent_name: str = "") -> SymbolDef:
        name_node = node.child_by_field_name('name')
        name = self._node_text(name_node, code) if name_node else '?'
        params_node = node.child_by_field_name('parameters')
        params_text = self._node_text(params_node, code) if params_node else '()'
        ret_node = node.child_by_field_name('return_type')
        ret_text = f" -> {self._node_text(ret_node, code)}" if ret_node else ""
        sig = f"def {name}{params_text}{ret_text}:"
        stype = 'method' if parent_name else 'function'
        return SymbolDef(
            name=name, type=stype,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            signature=sig, parent=parent_name,
        )

    def _ts_class(self, node, code: str) -> SymbolDef:
        name_node = node.child_by_field_name('name')
        name = self._node_text(name_node, code) if name_node else '?'
        superclasses = node.child_by_field_name('superclasses')
        bases = self._node_text(superclasses, code) if superclasses else ""
        sig = f"class {name}{bases}:" if bases else f"class {name}:"
        sym = SymbolDef(
            name=name, type='class',
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            signature=sig,
        )
        body_node = node.child_by_field_name('body')
        if body_node:
            for child in body_node.children:
                child_sym = self._ts_visit(child, code, parent_name=name)
                if child_sym and child_sym.type in ('function', 'method'):
                    child_sym.type = 'method'
                    child_sym.parent = name
                    sym.children.append(child_sym)
        return sym

    def _node_text(self, node, code: str) -> str:
        return code.encode('utf-8')[node.start_byte:node.end_byte].decode('utf-8', errors='replace')

    # -- Python ast fallback -------------------------------------------

    def _ast_parse(self, code: str) -> List[SymbolDef]:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return self._regex_parse(code)

        lines = code.split('\n')
        symbols: List[SymbolDef] = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols.append(self._ast_func(node, lines))
            elif isinstance(node, ast.ClassDef):
                symbols.append(self._ast_class(node, lines))
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                end = getattr(node, 'end_lineno', node.lineno) or node.lineno
                text = '\n'.join(lines[node.lineno - 1:end])
                names = [a.name for a in node.names]
                symbols.append(SymbolDef(
                    name=', '.join(names), type='import',
                    start_line=node.lineno, end_line=end,
                    signature=text.strip(),
                ))
        return symbols

    def _ast_func(self, node, lines: List[str], parent: str = "") -> SymbolDef:
        start = node.lineno
        end = getattr(node, 'end_lineno', start) or start
        if hasattr(node, 'decorator_list') and node.decorator_list:
            start = node.decorator_list[0].lineno
        args_str = ', '.join(a.arg for a in node.args.args)
        ret = ""
        if node.returns:
            try:
                ret = f" -> {ast.unparse(node.returns)}"
            except Exception:
                pass
        sig = f"def {node.name}({args_str}){ret}:"
        stype = 'method' if parent else 'function'
        return SymbolDef(
            name=node.name, type=stype,
            start_line=start, end_line=end,
            signature=sig, parent=parent,
        )

    def _ast_class(self, node, lines: List[str]) -> SymbolDef:
        start = node.lineno
        end = getattr(node, 'end_lineno', start) or start
        if hasattr(node, 'decorator_list') and node.decorator_list:
            start = node.decorator_list[0].lineno
        bases = []
        for b in node.bases:
            try:
                bases.append(ast.unparse(b))
            except Exception:
                bases.append('?')
        bases_str = f"({', '.join(bases)})" if bases else ""
        sig = f"class {node.name}{bases_str}:"
        sym = SymbolDef(
            name=node.name, type='class',
            start_line=start, end_line=end,
            signature=sig,
        )
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                child = self._ast_func(item, lines, parent=node.name)
                sym.children.append(child)
        return sym

    # -- regex fallback ------------------------------------------------

    def _regex_parse(self, code: str) -> List[SymbolDef]:
        symbols: List[SymbolDef] = []
        lines = code.split('\n')
        pat = re.compile(r'^(\s*)((?:async\s+)?def |class )(\w+)')
        for i, line in enumerate(lines):
            m = pat.match(line)
            if not m:
                continue
            indent = len(m.group(1))
            name = m.group(3)
            stype = 'class' if 'class ' in m.group(2) else 'function'
            end = i + 1
            for j in range(i + 1, len(lines)):
                s = lines[j]
                if s.strip() == '':
                    continue
                if len(s) - len(s.lstrip()) <= indent and s.strip():
                    break
                end = j + 1
            sig = line.strip()
            if not sig.endswith(':'):
                sig += ':'
            symbols.append(SymbolDef(
                name=name, type=stype,
                start_line=i + 1, end_line=end,
                signature=sig,
            ))
        return symbols

    # -- helpers -------------------------------------------------------

    def _symbols_to_signature_text(self, symbols: List[SymbolDef], indent: int = 0) -> str:
        parts: List[str] = []
        prefix = "  " * indent
        for s in symbols:
            if s.type == 'import':
                continue
            parts.append(f"{prefix}{s.signature}")
            for c in s.children:
                parts.append(f"{prefix}  {c.signature}")
        return '\n'.join(parts)

    def _flatten(self, symbols: List[SymbolDef]) -> List[SymbolDef]:
        out: List[SymbolDef] = []
        for s in symbols:
            out.append(s)
            out.extend(s.children)
        return out


# -------------------------------------------------------------------
# A2. RepoMapper
# -------------------------------------------------------------------

class RepoMapper:
    """生成仓库地图 — 将项目结构压缩为几 KB 的文本"""

    def __init__(self, indexer: Optional[TreeSitterIndexer] = None):
        self._indexer = indexer or TreeSitterIndexer()

    def build_map(self, file_paths: List[str],
                  max_tokens: int = 4000,
                  read_fn: Optional[Callable[[str], str]] = None) -> str:
        entries: List[Tuple[str, str]] = []
        for fp in sorted(file_paths):
            if not fp.endswith('.py'):
                continue
            try:
                if read_fn:
                    code = read_fn(fp)
                else:
                    with open(fp, 'r', encoding='utf-8', errors='replace') as f:
                        code = f.read()
                sig_text = self._indexer.extract_signatures(code)
                if sig_text.strip():
                    entries.append((fp, sig_text))
            except Exception:
                continue

        lines: List[str] = []
        token_est = 0
        for fp, sig in entries:
            block = f"{fp}\n{self._indent(sig, 2)}\n"
            est = len(block) // 4
            if token_est + est > max_tokens:
                lines.append(f"... ({len(entries) - len(lines)} more files truncated)")
                break
            lines.append(block)
            token_est += est
        return '\n'.join(lines)

    def build_file_map(self, file_path: str, code: str) -> str:
        sig_text = self._indexer.extract_signatures(code)
        if sig_text.strip():
            return f"{file_path}\n{self._indent(sig_text, 2)}"
        return file_path

    @staticmethod
    def _indent(text: str, n: int) -> str:
        prefix = ' ' * n
        return '\n'.join(prefix + line for line in text.split('\n'))


# -------------------------------------------------------------------
# A3. CodeSearcher
# -------------------------------------------------------------------

class CodeSearcher:
    """BM25 代码搜索（降级到简单关键词匹配）"""

    def __init__(self):
        self._corpus_paths: List[str] = []
        self._corpus_tokens: List[List[str]] = []
        self._corpus_contents: Dict[str, str] = {}
        self._bm25: Optional[Any] = None

    def index_codebase(self, files: Dict[str, str]) -> None:
        self._corpus_paths = []
        self._corpus_tokens = []
        self._corpus_contents = files.copy()

        for path, content in files.items():
            self._corpus_paths.append(path)
            self._corpus_tokens.append(self._tokenize(content))

        if _BM25_AVAILABLE and self._corpus_tokens:
            self._bm25 = _BM25Okapi(self._corpus_tokens)
        else:
            self._bm25 = None

    def search(self, query: str, top_k: int = 5) -> List[SearchHit]:
        if not self._corpus_paths:
            return []
        tokens = self._tokenize(query)
        if self._bm25:
            return self._bm25_search(tokens, top_k)
        return self._keyword_search(tokens, top_k)

    def _bm25_search(self, tokens: List[str], top_k: int) -> List[SearchHit]:
        try:
            scores = self._bm25.get_scores(tokens)
        except Exception:
            return self._keyword_search(tokens, top_k)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        hits: List[SearchHit] = []
        for idx, score in ranked[:top_k]:
            if score <= 0:
                continue
            path = self._corpus_paths[idx]
            content = self._corpus_contents.get(path, "")
            snippet = content[:200] if content else ""
            hits.append(SearchHit(file_path=path, score=float(score), snippet=snippet))
        if not hits:
            return self._keyword_search(tokens, top_k)
        return hits

    def _keyword_search(self, tokens: List[str], top_k: int) -> List[SearchHit]:
        scores: List[Tuple[str, float]] = []
        for path, content in self._corpus_contents.items():
            content_lower = content.lower()
            score = sum(1.0 for t in tokens if t.lower() in content_lower)
            if score > 0:
                scores.append((path, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [
            SearchHit(
                file_path=p, score=s,
                snippet=self._corpus_contents.get(p, "")[:200],
            )
            for p, s in scores[:top_k]
        ]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        tokens = re.findall(r'[a-zA-Z_]\w*', text)
        cn_tokens = re.findall(r'[\u4e00-\u9fff]+', text)
        return [t.lower() for t in tokens + cn_tokens if len(t) > 1]


# ===================================================================
# B. 规划决策层
# ===================================================================

# -------------------------------------------------------------------
# B1. ContextManager
# -------------------------------------------------------------------

class ContextManager:
    """管理活动文件列表与 token 预算"""

    def __init__(self, max_context_tokens: int = 8000,
                 indexer: Optional[TreeSitterIndexer] = None):
        self._max_tokens = max_context_tokens
        self._files: List[ContextFile] = []
        self._indexer = indexer or TreeSitterIndexer()

    def reset(self):
        self._files.clear()

    def add_file(self, path: str, content: str, relevance: float = 1.0):
        est = self._estimate_tokens(content)
        self._files.append(ContextFile(
            path=path, content=content,
            relevance=relevance, token_estimate=est,
        ))
        self._files.sort(key=lambda f: f.relevance, reverse=True)

    def add_snippet(self, path: str, code: str, symbol_name: str,
                    relevance: float = 1.0):
        """只添加某个符号的代码片段（大文件优化）"""
        rng = self._indexer.get_symbol_range(code, symbol_name)
        if rng:
            lines = code.split('\n')
            snippet = '\n'.join(lines[rng[0] - 1:rng[1]])
            self.add_file(f"{path}:{symbol_name}", snippet, relevance)
        else:
            self.add_file(path, code, relevance)

    def build_context(self) -> str:
        parts: List[str] = []
        used = 0
        for f in self._files:
            if used + f.token_estimate > self._max_tokens:
                remaining_budget = self._max_tokens - used
                if remaining_budget > 0:
                    truncated = self._truncate(f.content, max(remaining_budget, 20))
                    parts.append(f"## {f.path} (truncated)\n```python\n{truncated}\n```")
                break
            parts.append(f"## {f.path}\n```python\n{f.content}\n```")
            used += f.token_estimate
        return '\n\n'.join(parts)

    def get_token_usage(self) -> int:
        return sum(f.token_estimate for f in self._files)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return len(text) // 4 + 1

    @staticmethod
    def _truncate(text: str, max_tokens: int) -> str:
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n... (truncated)"


# -------------------------------------------------------------------
# B2. EditPlanner
# -------------------------------------------------------------------

class EditPlanner:
    """LLM 驱动的任务分拆"""

    _PLAN_PROMPT = """你是代码修改规划专家。根据以下项目结构和用户需求，输出修改计划。

## 项目结构
{repo_map}

## 用户需求
{instruction}

## 输出要求
用JSON格式返回修改计划，格式如下（不要返回其他内容）：
```json
{{
    "target_files": ["file1.py", "file2.py"],
    "modifications": [
        {{"file": "file1.py", "description": "修改描述"}},
        {{"file": "file2.py", "description": "修改描述"}}
    ],
    "summary": "一句话总结"
}}
```
"""

    def __init__(self, llm_client=None):
        self._llm = llm_client

    def plan(self, instruction: str, repo_map: str = "") -> EditPlan:
        if not self._llm or not repo_map:
            return EditPlan(summary="单文件直接编辑模式")

        prompt = self._PLAN_PROMPT.format(
            repo_map=repo_map, instruction=instruction,
        )
        try:
            resp = self._llm.one_chat(prompt)
            return self._parse_plan(resp)
        except Exception as e:
            logger.warning(f"EditPlanner 调用失败: {e}")
            return EditPlan(summary=f"规划失败: {e}")

    def _parse_plan(self, resp: str) -> EditPlan:
        import json
        m = re.search(r'\{[\s\S]*\}', resp)
        if not m:
            return EditPlan(summary=resp.strip()[:200])
        try:
            data = json.loads(m.group(0))
            return EditPlan(
                target_files=data.get('target_files', []),
                modifications=data.get('modifications', []),
                summary=data.get('summary', ''),
            )
        except (json.JSONDecodeError, KeyError):
            return EditPlan(summary=resp.strip()[:200])


# ===================================================================
# C. 执行编辑层 — SearchReplaceEngine
# ===================================================================

class SearchReplaceEngine:
    """增强版 Search/Replace 引擎

    五级匹配策略:
        1. 精确匹配
        2. 行尾空白归一化
        3. 空白归一化 (\\s+ -> 单空格)
        4. 模糊逐行匹配 (similarity >= threshold)
        5. tree-sitter 辅助范围收窄后重试模糊匹配
    """

    # -- SEARCH/REPLACE 块的正则 ----------------------------------------

    _BLOCK_PATTERN = re.compile(
        r'<{6,7}\s*SEARCH\s*\n(.*?)\n={6,7}\s*\n(.*?)\n>{6,7}\s*REPLACE',
        re.DOTALL,
    )

    _ALT_PATTERNS = [
        re.compile(
            r'```*\s*(?:SEARCH|search|原代码|原始代码)\s*\n(.*?)\n```*\s*\n'
            r'```*\s*(?:REPLACE|replace|新代码|修改后代码)\s*\n(.*?)\n```*',
            re.DOTALL,
        ),
        re.compile(
            r'<<<+\s*SEARCH\s*\n(.*?)\n===+\n(.*?)\n>>>+\s*REPLACE',
            re.DOTALL,
        ),
    ]

    _EDIT_PROMPT = """你是代码编辑专家。根据用户指令修改Python代码。

## 规则（必须严格遵守）
1. 用 SEARCH/REPLACE 块描述每处修改
2. SEARCH 内容必须是原代码中**逐字存在**的片段（包含缩进和空白）
3. SEARCH 要包含足够上下文（至少3行完整行）以唯一定位
4. 只输出需要修改的部分，不要输出未修改的代码
5. 除了 SEARCH/REPLACE 块，不要输出任何其他内容
6. 保持SEARCH块中的缩进与原代码**完全一致**
7. REPLACE 中的缩进必须与 SEARCH 中对应位置的缩进完全一致
8. 不要破坏 try/except/finally、if/elif/else、with、for/while 等复合语句的完整性
9. 如果修改涉及 try 块内部的代码，SEARCH 应包含完整的 try...except 结构
10. 代码左边的行号仅供参考定位，**禁止**将行号写入 SEARCH 或 REPLACE 块

## 格式
<<<<<<< SEARCH
（原代码片段，逐字匹配，包含完整的缩进）
=======
（修改后的代码，保持正确缩进）
>>>>>>> REPLACE

## 原始代码
{numbered_code}

## 修改指令
{instruction}
"""

    def __init__(self, indexer: Optional[TreeSitterIndexer] = None,
                 fuzzy_threshold: float = 0.85):
        self._indexer = indexer or TreeSitterIndexer()
        self._fuzzy_threshold = fuzzy_threshold

    # -- prompt 生成 --------------------------------------------------

    def build_edit_prompt(self, code: str, instruction: str,
                          context: str = "") -> str:
        numbered = self._add_line_numbers(code)
        prompt = self._EDIT_PROMPT.format(
            numbered_code=numbered, instruction=instruction,
        )
        if context:
            prompt += f"\n\n## 额外上下文\n{context}"
        return prompt

    # -- 解析 LLM 响应 -------------------------------------------------

    def parse_response(self, llm_response: str) -> List[SearchReplaceBlock]:
        blocks: List[SearchReplaceBlock] = []
        for m in self._BLOCK_PATTERN.finditer(llm_response):
            blocks.append(SearchReplaceBlock(search=m.group(1), replace=m.group(2)))
        if not blocks:
            for pat in self._ALT_PATTERNS:
                for m in pat.finditer(llm_response):
                    blocks.append(SearchReplaceBlock(search=m.group(1), replace=m.group(2)))
                if blocks:
                    break
        return blocks

    # -- 应用 blocks ---------------------------------------------------

    def apply(self, code: str, blocks: List[SearchReplaceBlock]) -> ApplyResult:
        applied: List[str] = []
        failed: List[str] = []
        current = code

        for blk in blocks:
            new_code, strategy = self._apply_single(current, blk.search, blk.replace)
            preview = blk.search.strip().split('\n')[0][:60]
            if new_code is None:
                failed.append(f"未匹配: {preview}...")
                continue
            try:
                compile(new_code, '<block_check>', 'exec')
                current = new_code
                applied.append(f"[{strategy}] {preview}...")
            except SyntaxError:
                repaired = self._try_indent_repair(
                    current, blk.search, blk.replace,
                )
                if repaired is not None:
                    current = repaired
                    applied.append(f"[indent_fix] {preview}...")
                else:
                    failed.append(f"syntax_rollback: {preview}...")

        success = len(applied) > 0
        return ApplyResult(
            success=success, new_code=current,
            applied=applied, failed=failed,
            strategy_used="search_replace",
        )

    # -- 五级匹配策略 ---------------------------------------------------

    def _apply_single(self, code: str, search: str, replace: str,
                      *, _skip_prealign: bool = False
                      ) -> Tuple[Optional[str], str]:
        if not _skip_prealign:
            indent_diff = self._detect_indent_diff(
                search.split('\n'), replace.split('\n'),
            )
            if indent_diff != 0:
                replace = '\n'.join(
                    self._adjust_indent(replace.split('\n'), indent_diff)
                )

        # Level 1: 精确匹配
        if search in code:
            return code.replace(search, replace, 1), "exact"

        # Level 2: 行尾空白归一化
        result = self._strip_trailing_match(code, search, replace)
        if result is not None:
            return result, "strip_trailing"

        # Level 3: 空白归一化
        result = self._normalized_match(code, search, replace)
        if result is not None:
            return result, "normalized"

        # Level 4: 模糊逐行匹配
        result = self._fuzzy_match(code, search, replace, self._fuzzy_threshold)
        if result is not None:
            return result, "fuzzy"

        # Level 5: tree-sitter 辅助范围收窄
        result = self._ts_assisted_match(code, search, replace)
        if result is not None:
            return result, "ts_assisted"

        return None, "none"

    def _strip_trailing_match(self, code: str, search: str, replace: str
                              ) -> Optional[str]:
        def strip_tr(s: str) -> str:
            return '\n'.join(l.rstrip() for l in s.split('\n'))

        code_s = strip_tr(code)
        search_s = strip_tr(search)
        if search_s not in code_s:
            return None

        idx = code_s.index(search_s)
        pre_lines = code_s[:idx].count('\n')
        search_lc = search_s.count('\n') + 1
        code_lines = code.split('\n')
        before = code_lines[:pre_lines]
        after = code_lines[pre_lines + search_lc:]
        replace_lines = replace.split('\n')
        indent_diff = self._detect_indent_diff(
            code_lines[pre_lines:pre_lines + search_lc], search.split('\n'),
        )
        if indent_diff != 0:
            replace_lines = self._adjust_indent(replace_lines, indent_diff)
        return '\n'.join(before + replace_lines + after)

    def _normalized_match(self, code: str, search: str, replace: str
                          ) -> Optional[str]:
        def norm(s: str) -> str:
            return re.sub(r'[ \t]+', ' ', s)

        code_lines = code.split('\n')
        search_lines = search.split('\n')
        code_norm = [norm(l) for l in code_lines]
        search_norm = [norm(l) for l in search_lines]
        w = len(search_norm)
        for i in range(len(code_norm) - w + 1):
            if code_norm[i:i + w] == search_norm:
                indent_diff = self._detect_indent_diff(
                    code_lines[i:i + w], search_lines,
                )
                replace_lines = replace.split('\n')
                if indent_diff != 0:
                    replace_lines = self._adjust_indent(replace_lines, indent_diff)
                return '\n'.join(code_lines[:i] + replace_lines + code_lines[i + w:])
        return None

    def _fuzzy_match(self, code: str, search: str, replace: str,
                     threshold: float) -> Optional[str]:
        code_lines = code.split('\n')
        search_lines = search.split('\n')
        w = len(search_lines)
        if w == 0 or w > len(code_lines):
            return None

        best_idx, best_score = -1, 0.0
        for i in range(len(code_lines) - w + 1):
            score = self._block_similarity(code_lines[i:i + w], search_lines)
            if score > best_score:
                best_score = score
                best_idx = i

        if best_score >= threshold and best_idx >= 0:
            indent_diff = self._detect_indent_diff(
                code_lines[best_idx:best_idx + w], search_lines,
            )
            replace_lines = replace.split('\n')
            if indent_diff != 0:
                replace_lines = self._adjust_indent(replace_lines, indent_diff)
            return '\n'.join(code_lines[:best_idx] + replace_lines + code_lines[best_idx + w:])
        return None

    def _ts_assisted_match(self, code: str, search: str, replace: str
                           ) -> Optional[str]:
        """Level 5: 用 tree-sitter 找到 SEARCH 块可能的符号范围，收窄搜索"""
        search_stripped = search.strip()
        first_line = search_stripped.split('\n')[0].strip()

        func_match = re.match(r'(?:async\s+)?def\s+(\w+)', first_line)
        class_match = re.match(r'class\s+(\w+)', first_line)
        symbol_name = None
        if func_match:
            symbol_name = func_match.group(1)
        elif class_match:
            symbol_name = class_match.group(1)

        if not symbol_name:
            return None

        rng = self._indexer.get_symbol_range(code, symbol_name)
        if not rng:
            return None

        code_lines = code.split('\n')
        sub_start = max(0, rng[0] - 1)
        sub_end = min(len(code_lines), rng[1])
        sub_code = '\n'.join(code_lines[sub_start:sub_end])

        result = self._fuzzy_match(sub_code, search, replace,
                                   threshold=self._fuzzy_threshold - 0.1)
        if result is None:
            return None

        result_lines = result.split('\n')
        return '\n'.join(code_lines[:sub_start] + result_lines + code_lines[sub_end:])

    # -- 缩进处理 -------------------------------------------------------

    @staticmethod
    def _detect_indent_diff(code_lines: List[str], search_lines: List[str]) -> int:
        """检测代码原文与 SEARCH 块之间的缩进差异"""
        for cl, sl in zip(code_lines, search_lines):
            if cl.strip() and sl.strip():
                c_indent = len(cl) - len(cl.lstrip())
                s_indent = len(sl) - len(sl.lstrip())
                return c_indent - s_indent
        return 0

    @staticmethod
    def _adjust_indent(lines: List[str], diff: int) -> List[str]:
        """调整缩进"""
        result: List[str] = []
        for line in lines:
            if not line.strip():
                result.append(line)
                continue
            if diff > 0:
                result.append(' ' * diff + line)
            else:
                remove = abs(diff)
                if line[:remove] == ' ' * remove:
                    result.append(line[remove:])
                else:
                    result.append(line.lstrip())
        return result

    # -- 工具方法 -------------------------------------------------------

    @staticmethod
    def _block_similarity(lines_a: List[str], lines_b: List[str]) -> float:
        if len(lines_a) != len(lines_b):
            return 0.0
        total = 0.0
        for a, b in zip(lines_a, lines_b):
            total += difflib.SequenceMatcher(None, a.rstrip(), b.rstrip()).ratio()
        return total / len(lines_a)

    @staticmethod
    def _add_line_numbers(code: str) -> str:
        lines = code.split('\n')
        width = max(4, len(str(len(lines))))
        numbered = []
        for i, line in enumerate(lines, 1):
            numbered.append(f"{i:>{width}}| {line}")
        return '\n'.join(numbered)

    @staticmethod
    def _get_base_indent(text: str) -> int:
        for line in text.split('\n'):
            if line.strip():
                return len(line) - len(line.lstrip(' '))
        return 0

    @staticmethod
    def _force_indent(text: str, target: int) -> str:
        lines = text.split('\n')
        base = None
        for line in lines:
            if line.strip():
                base = len(line) - len(line.lstrip(' '))
                break
        if base is None or base == target:
            return text
        delta = target - base
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

    def _detect_match_indent(self, code: str, search: str) -> Optional[int]:
        if search in code:
            return self._get_base_indent(search)
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

    def _try_indent_repair(self, code: str, search: str, replace: str
                           ) -> Optional[str]:
        """When normal indent alignment fails, probe common indent levels."""
        already_tried = self._get_base_indent(search)
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
            adjusted = self._force_indent(replace, target)
            result, _ = self._apply_single(
                code, search, adjusted, _skip_prealign=True,
            )
            if result is None:
                continue
            try:
                compile(result, '<indent_repair>', 'exec')
                logger.info(
                    f"[SearchReplaceEngine] 缩进探测修复成功: "
                    f"{already_tried} -> {target}"
                )
                return result
            except SyntaxError:
                continue
        return None


# ===================================================================
# D. 验证纠错层
# ===================================================================

# -------------------------------------------------------------------
# D1. EditValidator
# -------------------------------------------------------------------

class EditValidator:
    """多级验证器"""

    def __init__(self, indexer: Optional[TreeSitterIndexer] = None):
        self._indexer = indexer or TreeSitterIndexer()

    def validate(self, old_code: str, new_code: str) -> ValidationResult:
        errors: List[str] = []
        warnings: List[str] = []

        # Level 1: compile() 语法检查
        syn_ok, syn_errs = self._check_syntax(new_code)
        if not syn_ok:
            return ValidationResult(ok=False, errors=syn_errs)

        # Level 2: 完整性 — 函数/类是否丢失
        int_errs = self._check_integrity(old_code, new_code)
        errors.extend(int_errs)

        # Level 3: tree-sitter 重新解析，结构合理性
        ts_warns = self._check_ts_structure(new_code)
        warnings.extend(ts_warns)

        # Level 4: 行数大幅变化预警
        old_lc = len(old_code.split('\n'))
        new_lc = len(new_code.split('\n'))
        if old_lc > 0 and abs(new_lc - old_lc) / old_lc > 0.5:
            warnings.append(f"行数变化较大: {old_lc} -> {new_lc}")

        if old_lc > 50 and new_lc < old_lc * 0.3:
            errors.append(f"代码大幅缩减 ({old_lc}->{new_lc} 行), 可能丢失内容")

        return ValidationResult(
            ok=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def _check_syntax(self, code: str) -> Tuple[bool, List[str]]:
        try:
            compile(code, '<editor>', 'exec')
            return True, []
        except SyntaxError as e:
            return False, [f"SyntaxError L{e.lineno}: {e.msg}"]

    def _check_integrity(self, old_code: str, new_code: str) -> List[str]:
        errors: List[str] = []
        old_names = self._extract_toplevel_names(old_code)
        new_names = self._extract_toplevel_names(new_code)
        for name, ntype in old_names.items():
            if name not in new_names:
                errors.append(f"{ntype} '{name}' 在修改后丢失")
        return errors

    def _extract_toplevel_names(self, code: str) -> Dict[str, str]:
        names: Dict[str, str] = {}
        try:
            tree = ast.parse(code)
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    names[node.name] = 'function'
                elif isinstance(node, ast.ClassDef):
                    names[node.name] = 'class'
        except SyntaxError:
            for m in re.finditer(r'^(?:def|class)\s+(\w+)', code, re.MULTILINE):
                names[m.group(1)] = 'symbol'
        return names

    def _check_ts_structure(self, code: str) -> List[str]:
        warnings: List[str] = []
        try:
            symbols = self._indexer.parse_file(code)
            for s in symbols:
                if s.type in ('function', 'method') and s.end_line - s.start_line < 1:
                    warnings.append(f"符号 {s.qualified_name()} 体为空")
        except Exception:
            pass
        return warnings


# -------------------------------------------------------------------
# D2. CorrectionLoop
# -------------------------------------------------------------------

class CorrectionLoop:
    """多轮迭代纠错引擎"""

    def __init__(self, llm_client, engine: SearchReplaceEngine,
                 validator: EditValidator, max_rounds: int = 3):
        self._llm = llm_client
        self._engine = engine
        self._validator = validator
        self._max_rounds = max_rounds

    def run(self, code: str, instruction: str,
            context: str = "") -> EditResultV2:
        error_history: List[str] = []
        current_instruction = instruction

        for round_idx in range(1, self._max_rounds + 1):
            logger.info(f"[CorrectionLoop] 第 {round_idx}/{self._max_rounds} 轮")

            # 1. 生成 prompt
            prompt = self._engine.build_edit_prompt(
                code, current_instruction, context,
            )

            # 2. 调用 LLM
            try:
                resp = self._llm.one_chat(prompt)
            except Exception as e:
                error_history.append(f"第{round_idx}轮 LLM 调用失败: {e}")
                continue

            # 3. 解析 blocks
            blocks = self._engine.parse_response(resp)
            if not blocks:
                error_history.append(f"第{round_idx}轮 LLM 未返回有效的 SEARCH/REPLACE 块")
                current_instruction = self._augment_instruction(
                    instruction, error_history,
                )
                continue

            # 4. 应用
            result = self._engine.apply(code, blocks)

            if not result.success:
                error_history.append(
                    f"第{round_idx}轮 所有 SEARCH 块均未匹配: {result.failed}"
                )
                current_instruction = self._augment_instruction(
                    instruction, error_history,
                )
                continue

            # 5. 验证
            vr = self._validator.validate(code, result.new_code)
            if not vr.ok:
                error_history.extend(
                    f"第{round_idx}轮 验证失败: {e}" for e in vr.errors
                )
                current_instruction = self._augment_instruction(
                    instruction, error_history,
                    failed_code=result.new_code,
                )
                continue

            # 成功
            diff = _make_diff(code, result.new_code)
            return EditResultV2(
                success=True, new_code=result.new_code, diff=diff,
                applied_edits=result.applied, failed_edits=result.failed,
                warnings=vr.warnings, rounds_used=round_idx,
                strategy_used="search_replace",
            )

        # 所有轮次耗尽
        return EditResultV2(
            success=False, new_code=code,
            errors=error_history[-5:],
            rounds_used=self._max_rounds,
            strategy_used="all_rounds_failed",
        )

    def _augment_instruction(self, original: str,
                             errors: List[str], *,
                             failed_code: Optional[str] = None) -> str:
        recent = errors[-3:]
        error_text = '\n'.join(f"- {e}" for e in recent)
        parts = [
            original,
            "\n\n## 重要：之前的尝试失败了，请务必避免以下错误\n",
            error_text,
        ]
        if failed_code:
            ctx = self._extract_error_context(failed_code, recent)
            if ctx:
                parts.append(f"\n\n## 错误行附近的代码\n{ctx}")
        parts.append(
            "\n\n请确保 SEARCH 块中的内容与原代码**完全一致**（包括缩进和空白）。"
        )
        return ''.join(parts)

    @staticmethod
    def _extract_error_context(code: str, errors: List[str],
                               context_lines: int = 5) -> str:
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


# ===================================================================
# 主入口类
# ===================================================================

class SmartLLMEditorV2:
    """
    智能 LLM 代码编辑器 V2

    四层流水线:
        A. 索引感知 -> B. 规划决策 -> C. 执行编辑 -> D. 验证纠错

    主要接口:
        edit()         - 单文件编辑（最常用）
        edit_project() - 多文件项目编辑
        find()         - 查找代码符号
        analyze()      - 代码分析
    """

    def __init__(self, llm_client=None, max_context_tokens: int = 8000):
        self._llm = llm_client
        self._indexer = TreeSitterIndexer()
        self._mapper = RepoMapper(self._indexer)
        self._searcher = CodeSearcher()
        self._context_mgr = ContextManager(max_context_tokens, self._indexer)
        self._planner = EditPlanner(llm_client)
        self._engine = SearchReplaceEngine(self._indexer)
        self._validator = EditValidator(self._indexer)
        # Surface backend selection so production logs make degradations visible.
        self.backends = {
            "indexer": self._indexer.backend,
            "bm25": "ok" if _BM25_AVAILABLE else "fallback",
            "grep_ast": "ok" if _GREP_AST_AVAILABLE else "missing",
            "tree_sitter": "ok" if _TS_AVAILABLE else "missing",
        }
        logger.info(
            f"[SmartLLMEditorV2] init backends={self.backends}"
        )
        if self.backends["indexer"] != "tree-sitter":
            logger.warning(
                "[SmartLLMEditorV2] tree-sitter unavailable; falling back to ast/regex. "
                "Search quality will degrade on large or syntax-broken files."
            )

    # ---- 单文件编辑 --------------------------------------------------

    def edit(self, code: str, instruction: str, *,
             file_path: str = "", context: str = "",
             max_retries: int = 3) -> EditResultV2:
        if not self._llm:
            return EditResultV2(
                success=False, new_code=code,
                errors=["LLM 客户端不可用"],
            )

        logger.info(f"[SmartLLMEditorV2] edit: '{instruction[:80]}...'")

        # A. 索引感知 — 生成文件签名摘要作为额外上下文
        file_map = self._mapper.build_file_map(file_path or "<stdin>", code)
        enriched_context = context
        if file_map:
            enriched_context = f"## 文件结构\n{file_map}\n\n{context}" if context else f"## 文件结构\n{file_map}"

        # C+D. 执行 + 纠错循环
        loop = CorrectionLoop(
            self._llm, self._engine, self._validator,
            max_rounds=max_retries,
        )
        result = loop.run(code, instruction, enriched_context)

        if result.success:
            logger.info(
                f"[SmartLLMEditorV2] 编辑成功 "
                f"(rounds={result.rounds_used}, "
                f"applied={len(result.applied_edits)})"
            )
        else:
            logger.warning(
                f"[SmartLLMEditorV2] 编辑失败: {result.errors[:2]}"
            )

        return result

    # ---- 多文件项目编辑 ------------------------------------------------

    def edit_project(self, instruction: str,
                     file_paths: List[str],
                     read_file_fn: Callable[[str], str],
                     max_retries: int = 3,
                     *,
                     transactional: bool = False) -> List[EditResultV2]:
        """Edit multiple files driven by a single instruction.

        When ``transactional=True``, the returned ``new_code`` for every file
        is computed but **none** of them are considered "applied" if any single
        file failed. The caller is then responsible for either committing all
        results or discarding them. The default ``False`` keeps the legacy
        per-file behaviour where each result stands alone.
        """
        if not self._llm:
            return [EditResultV2(
                success=False, new_code="",
                errors=["LLM 客户端不可用"],
            )]

        logger.info(
            f"[SmartLLMEditorV2] edit_project: {len(file_paths)} files "
            f"(transactional={transactional})"
        )

        # A. 索引 — 建立 BM25 索引和 repo map
        files: Dict[str, str] = {}
        for fp in file_paths:
            try:
                files[fp] = read_file_fn(fp)
            except Exception as e:
                logger.warning(f"读取 {fp} 失败: {e}")

        self._searcher.index_codebase(files)
        repo_map = self._mapper.build_map(file_paths, read_fn=read_file_fn)

        # B. 规划
        plan = self._planner.plan(instruction, repo_map)
        logger.info(f"[SmartLLMEditorV2] 修改计划: {plan.summary}")

        if not plan.target_files:
            hits = self._searcher.search(instruction, top_k=3)
            plan.target_files = [h.file_path for h in hits]
            if not plan.target_files:
                plan.target_files = list(files.keys())[:3]

        # C+D. 逐文件编辑
        results: List[EditResultV2] = []
        for target_file in plan.target_files:
            if target_file not in files:
                results.append(EditResultV2(
                    success=False, new_code="",
                    errors=[f"文件不存在: {target_file}"],
                ))
                continue

            code = files[target_file]
            file_instruction = instruction
            for mod in plan.modifications:
                if mod.get('file') == target_file:
                    file_instruction = f"{instruction}\n\n具体要求: {mod.get('description', '')}"
                    break

            # 收集相关上下文
            self._context_mgr.reset()
            hits = self._searcher.search(instruction, top_k=3)
            for hit in hits:
                if hit.file_path != target_file and hit.file_path in files:
                    self._context_mgr.add_file(
                        hit.file_path, files[hit.file_path],
                        relevance=hit.score,
                    )
            extra_context = self._context_mgr.build_context()

            result = self.edit(
                code, file_instruction,
                file_path=target_file,
                context=extra_context,
                max_retries=max_retries,
            )
            results.append(result)

        if transactional and any(not r.success for r in results):
            failed = [r for r in results if not r.success]
            logger.warning(
                f"[SmartLLMEditorV2] transactional edit_project rolling back: "
                f"{len(failed)}/{len(results)} files failed"
            )
            for r in results:
                if r.success:
                    r.success = False
                    r.warnings.append(
                        "rolled back: another file in the same transaction failed"
                    )
        return results

    # ---- 查找代码 ----------------------------------------------------

    def find(self, code: str, query: str) -> List[SymbolDef]:
        symbols = self._indexer.parse_file(code)
        keywords = self._extract_keywords(query)
        scored: List[Tuple[SymbolDef, float]] = []

        for sym in self._indexer._flatten(symbols):
            score = 0.0
            name_lower = sym.name.lower()
            sig_lower = sym.signature.lower()
            for kw in keywords:
                kw_l = kw.lower()
                if kw_l in name_lower:
                    score += 0.5
                if kw_l in sig_lower:
                    score += 0.2
            if sym.name.lower() in query.lower():
                score += 0.5
            if score > 0:
                sym_copy = SymbolDef(
                    name=sym.name, type=sym.type,
                    start_line=sym.start_line, end_line=sym.end_line,
                    signature=sym.signature, parent=sym.parent,
                )
                scored.append((sym_copy, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in scored]

    # ---- 代码分析 ----------------------------------------------------

    def analyze(self, code: str, query: str) -> Dict[str, Any]:
        if not self._llm:
            return {"error": "LLM 客户端不可用"}

        symbols = self._indexer.parse_file(code)
        sig_text = self._indexer.extract_signatures(code)

        relevant = self.find(code, query)[:5]
        relevant_code = ""
        if relevant:
            lines = code.split('\n')
            parts = []
            for sym in relevant:
                snippet = '\n'.join(lines[sym.start_line - 1:sym.end_line])
                parts.append(f"# --- {sym.type}: {sym.qualified_name()} ---\n{snippet}")
            relevant_code = '\n\n'.join(parts)
        else:
            relevant_code = code[:3000]

        prompt = f"""你是代码分析专家。分析以下代码并回答用户的问题。

## 用户问题
{query}

## 文件结构
{sig_text}

## 相关代码
```python
{relevant_code}
```

请给出详细的分析和建议。"""

        try:
            resp = self._llm.one_chat(prompt)
            return {
                "analysis": resp,
                "symbols_found": len(symbols),
                "relevant_symbols": [s.qualified_name() for s in relevant],
            }
        except Exception as e:
            return {"error": str(e)}

    # ---- 工具方法 ----------------------------------------------------

    @staticmethod
    def _extract_keywords(query: str) -> List[str]:
        stop = {
            '的', '和', '或', '是', '在', '了', '把', '将', '对', '被', '用',
            'the', 'and', 'or', 'is', 'in', 'to', 'a', 'an', 'of', 'for',
            'with', 'this', 'that', 'from', 'not', 'but', 'are', 'be',
            '函数', '方法', '类', '代码', '修改', '添加', '删除', '修复',
            '错误', '问题', '优化', '改进', '需要', '请', '帮', '我',
        }
        tokens = re.findall(r'[a-zA-Z_]\w*', query)
        cn_tokens = re.findall(r'[\u4e00-\u9fff]+', query)
        return [t for t in tokens + cn_tokens if t.lower() not in stop and len(t) > 1]


# ===================================================================
# Utility
# ===================================================================

def _make_diff(old: str, new: str) -> str:
    return '\n'.join(difflib.unified_diff(
        old.split('\n'), new.split('\n'), lineterm='', n=3,
    ))
