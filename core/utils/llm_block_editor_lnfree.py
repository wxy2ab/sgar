#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import ast
import difflib
import gc
import hashlib
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from core.utils.log import logger


def _stable_source_hash(source: str) -> str:
    """Stable, collision-resistant hash for AST cache keys.

    The previous implementation used :func:`hash` which is process-randomised
    and collidable on very large source files. SHA-256 keeps the cache safe.
    """
    return hashlib.sha256(source.encode("utf-8", errors="replace")).hexdigest()


@dataclass
class LNFreeInstruction:
    type: str
    old: Optional[str] = None
    new: Optional[str] = None
    locator: Optional[str] = None
    content: Optional[str] = None
    nth: int = 1
    context_before: Optional[str] = None
    raw_text: str = ""


@dataclass
class LNFreeResult:
    success: bool
    new_code: str
    applied_instructions: List[LNFreeInstruction]
    errors: List[str]
    warnings: List[str]
    diff: str
    original_code: str = ""


class _CodeNormalizer:
    @staticmethod
    def normalize_ws(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip())

    @staticmethod
    def strip_comments(code: str) -> str:
        lines = []
        for line in code.split("\n"):
            lines.append(re.sub(r"#.*$", "", line))
        return "\n".join(lines)

    @staticmethod
    def canonical(code: str) -> str:
        return _CodeNormalizer.normalize_ws(_CodeNormalizer.strip_comments(code))


class ContentLocator:
    def __init__(self):
        self._ast_cache: Optional[Tuple[str, ast.AST]] = None
        self._line_hashes: Optional[List[int]] = None
        self.last_strategy_name: Optional[str] = None

    def _load_ast(self, source: str) -> Optional[ast.AST]:
        if not source.strip():
            return None
        source_hash = _stable_source_hash(source)
        if self._ast_cache and self._ast_cache[0] == source_hash:
            return self._ast_cache[1]
        try:
            tree = ast.parse(source)
            self._ast_cache = (source_hash, tree)
            return tree
        except Exception:
            self._ast_cache = None
            return None

    def clear_cache(self):
        self._ast_cache = None
        self._line_hashes = None
        try:
            gc.collect()
        except Exception:
            pass

    def _is_def(self, pattern: str) -> bool:
        p = pattern.strip()
        return p.startswith("def ") or p.startswith("class ")

    def _ast_locate(self, source: str, pattern: str) -> List[Tuple[int, int]]:
        matches: List[Tuple[int, int]] = []
        tree = self._load_ast(source)
        if not tree:
            return matches
        src_lines = source.split("\n")
        pat = _CodeNormalizer.canonical(pattern)
        pat_lines = [l for l in pat.split("\n") if l.strip()]
        if not pat_lines:
            return matches
            
        class DefinitionVisitor(ast.NodeVisitor):
            def __init__(self, src_lines, pat, pat_lines):
                self.src_lines = src_lines
                self.pat = pat
                self.pat_lines = pat_lines
                self.matches: List[Tuple[int, int]] = []
                
            def _check(self, node):
                if hasattr(node, 'lineno') :
                    start = getattr(node, 'lineno', 0) - 1
                    end = getattr(node, 'end_lineno', start + 1)
                    if hasattr(node, 'name') and self.pat_lines[0].strip().endswith(node.name + '('):
                        self.matches.append((start, end))
                    if 0 <= start < end <= len(self.src_lines):
                        # 优先尝试完整匹配
                        seg = "\n".join(self.src_lines[start:end])
                        if _CodeNormalizer.canonical(seg) == self.pat:
                            self.matches.append((start, end))
                            return
                        
                        # 完整匹配失败时尝试签名行匹配
                        first_line = self.src_lines[start].strip()
                        pat_first_line = self.pat_lines[0].strip()
                        if (first_line.startswith(('def ', 'class ', '@')) and
                            _CodeNormalizer.normalize_ws(first_line) == _CodeNormalizer.normalize_ws(pat_first_line)):
                            self.matches.append((start, end))
            
            def visit_FunctionDef(self, node):
                self._check(node); self.generic_visit(node)
            def visit_AsyncFunctionDef(self, node):
                self._check(node); self.generic_visit(node)
            def visit_ClassDef(self, node):
                self._check(node); self.generic_visit(node)
        
        v = DefinitionVisitor(src_lines, pat, pat_lines)
        try:
            v.visit(tree)
        except Exception:
            return matches
        return v.matches

    def _exact_match(self, source: str, pattern: str) -> List[Tuple[int, int]]:
        matches: List[Tuple[int, int]] = []
        src = source
        pat = pattern
        start = 0
        while True:
            idx = src.find(pat, start)
            if idx == -1:
                break
            end = idx + len(pat)
            pre = src[:idx].count("\n")
            lines_pat = pat.count("\n") + 1
            matches.append((pre, pre + lines_pat))
            start = idx + 1
        return matches

    def _normalized_match(self, source: str, pattern: str) -> List[Tuple[int, int]]:
        s_norm = _CodeNormalizer.strip_comments(source)
        p_norm = _CodeNormalizer.strip_comments(pattern)
        return self._exact_match(s_norm, p_norm)

    def _similarity_match(self, source: str, pattern: str, strict: bool = True) -> List[Tuple[int, int]]:
        src_lines = source.split("\n")
        pat_lines = [l for l in pattern.split("\n") if l.strip()]
        if not pat_lines:
            return []
        thr = getattr(self, "similarity_threshold", 0.8) if strict else max(0.5, getattr(self, "similarity_threshold", 0.8) - 0.15)
        matches: List[Tuple[int, int]] = []
        w = len(pat_lines)
        for i in range(len(src_lines) - w + 1):
            window = src_lines[i:i + w]
            sim = self._calculate_similarity(window, pat_lines)
            if sim >= thr:
                matches.append((i, i + w))
        matches.sort(key=lambda x: self._calculate_similarity(src_lines[x[0]:x[1]], pat_lines), reverse=True)
        return matches

    def _score_instruction(self, instr: LNFreeInstruction) -> float:
        """评估指令质量，0.0-1.0分"""
        score = 1.0
        # LOCATOR/OLD 内容长度评分
        target = instr.old or instr.locator or ""
        if len(target.strip()) < 10:
            score *= 0.6
        # 上下文质量评分
        if not instr.context_before or len(instr.context_before) < 5:
            score *= 0.8
        return score

    def _calculate_similarity(self, lines1: List[str], lines2: List[str]) -> float:
        if len(lines1) != len(lines2) or not lines1:
            return 0.0
        weights = self._calculate_line_weights(lines2)
        total = 0.0
        for i, (a, b) in enumerate(zip(lines1, lines2)):
            base = difflib.SequenceMatcher(None, a.strip(), b.strip()).ratio()
            w = weights[i] if i < len(weights) else 1.0
            total += base * w
        return total / len(lines1)

    def _calculate_line_weights(self, pattern_lines: List[str]) -> List[float]:
        ws: List[float] = []
        for line in pattern_lines:
            s = line.strip()
            w = 1.0
            if s.startswith(('def ', 'class ', '@')):
                w = 2.0
            elif any(k in s for k in ['return', 'import', 'from', 'if ', 'for ', 'while ']):
                w = 1.5
            elif not s or s.startswith('#'):
                w = 0.3
            ws.append(w)
        return ws

    def _build_line_hashes(self, source: str) -> List[int]:
        if self._line_hashes is None:
            self._line_hashes = [hash(line.strip()) for line in source.split("\n")]
        return self._line_hashes

    def _fingerprint_match(self, source: str, pattern: str) -> List[Tuple[int, int]]:
        if len(pattern) > len(source) * 1.2:
            return []
        src_hashes = self._build_line_hashes(source)
        pat_lines = [line.strip() for line in pattern.split("\n") if line.strip()]
        pat_hashes = [hash(line) for line in pat_lines]
        if not pat_hashes:
            return []
        matches: List[Tuple[int, int]] = []
        for i in range(len(src_hashes) - len(pat_hashes) + 1):
            if src_hashes[i:i + len(pat_hashes)] == pat_hashes:
                matches.append((i, i + len(pat_hashes)))
        return matches

    def _fuzzy_match(self, source: str, pattern: str) -> List[Tuple[int, int]]:
        """Allow for minor differences in whitespace and punctuation"""
        def simplify(s):
            return re.sub(r'\s+', '', s).replace("'", '"')
        
        src_lines = source.split("\n")
        pat_lines = [l for l in pattern.split("\n") if l.strip()]
        if not pat_lines:
            return []
            
        matches = []
        pat_simple = [simplify(l) for l in pat_lines]
        w = len(pat_lines)
        
        for i in range(len(src_lines) - w + 1):
            window = src_lines[i:i+w]
            window_simple = [simplify(l) for l in window]
            
            match_count = sum(1 for a, b in zip(window_simple, pat_simple) if a == b)
            if match_count / w > 0.9: # 90% match
                matches.append((i, i+w))
        return matches

    def locate(self, source_code: str, pattern: str, nth: int = 1, context_before: Optional[str] = None) -> Optional[Tuple[int, int]]:
        # Early-stop: if exact match yields a single hit and the caller asked
        # for the first one, skip the cascade entirely. Big files used to pay
        # for ast/fingerprint/normalized/similarity/fuzzy passes even when an
        # exact match was already unique.
        if nth == 1:
            exact = self._exact_match(source_code, pattern)
            if len(exact) == 1:
                pos = exact[0]
                if not context_before or self._context_match(source_code, pos, context_before):
                    self.last_strategy_name = "exact"
                    return pos

        scored: List[Tuple[Tuple[int, int], float, str]] = []
        strategies = [
            ("exact", lambda: self._exact_match(source_code, pattern)),
            ("ast", lambda: self._ast_locate(source_code, pattern) if self._is_def(pattern) else []),
            ("fingerprint", lambda: self._fingerprint_match(source_code, pattern)),
            ("normalized", lambda: self._normalized_match(source_code, pattern)),
            ("similarity", lambda: self._similarity_match(source_code, pattern, strict=True)),
            ("fuzzy", lambda: self._fuzzy_match(source_code, pattern)),
            ("similarity_relaxed", lambda: self._similarity_match(source_code, pattern, strict=False)),
        ]
        for name, strat in strategies:
            try:
                res = strat()
            except Exception:
                res = []
            for pos in res:
                score = self._calculate_match_confidence(name, pos, source_code, pattern, context_before)
                scored.append((pos, score, name))
        scored.sort(key=lambda x: x[1], reverse=True)
        if context_before and scored:
            filtered = []
            for pos, score, name in scored:
                if self._context_match(source_code, pos, context_before):
                    filtered.append((pos, score * 1.5, name))
            if filtered:
                scored = filtered
        if not scored or nth < 1 or nth > len(scored):
            self._log_diagnosis(source_code, pattern, context_before, scored)
            return None
        best_pos, best_score, strategy_used = scored[nth - 1]
        self.last_strategy_name = strategy_used
        return best_pos

    def _calculate_match_confidence(self, strategy_name: str, pos: Tuple[int, int], source: str, pattern: str, context: Optional[str]) -> float:
        base = {"exact": 1.0, "ast": 0.95, "fingerprint": 0.9, "normalized": 0.85, "fuzzy": 0.8, "similarity": 0.75, "similarity_relaxed": 0.6}
        score = base.get(strategy_name, 0.5)
        if context and self._context_match(source, pos, context):
            score += 0.2
        src_lines = source.split("\n")
        snippet = "\n".join(src_lines[pos[0]:pos[1]])
        try:
            ast.parse(snippet)
            score += 0.1
        except Exception:
            pass
        if len(pattern.strip()) < 10:
            score *= 0.7
        return 1.0 if score > 1.0 else score

    def _context_match(self, source: str, pos: Tuple[int, int], context: str) -> bool:
        if self._semantic_context_match(source, pos, context):
            return True
        src_lines = source.split("\n")
        ctx_lines = [l for l in context.strip().split("\n") if l.strip()]
        if not ctx_lines:
            return True
        window = max(3, min(15, len(ctx_lines) + 2))
        start_idx = max(0, pos[0] - window)
        end_idx = min(len(src_lines), pos[0] + 1)
        window_text = "\n".join(src_lines[start_idx:end_idx])
        if context.strip() in window_text:
            return True
        norm_context = _CodeNormalizer.canonical(context)
        norm_window = _CodeNormalizer.canonical(window_text)
        if norm_context in norm_window:
            return True
        sim = difflib.SequenceMatcher(None, norm_context, norm_window).ratio()
        return sim > 0.65

    def _semantic_context_match(self, source: str, pos: Tuple[int, int], context: str) -> bool:
        """
        Check if context matches semantically (ignoring comments/whitespace)
        """
        try:
            src_lines = source.split("\n")
            # Look at a window before the position
            window_size = len(context.split("\n")) + 5
            start = max(0, pos[0] - window_size)
            end = pos[0]
            window_text = "\n".join(src_lines[start:end])
            
            norm_context = _CodeNormalizer.canonical(context)
            norm_window = _CodeNormalizer.canonical(window_text)
            
            return norm_context in norm_window
        except Exception:
            return False

    def _log_diagnosis(self, source_code: str, pattern: str, context_before: Optional[str], candidates: List[Tuple[Tuple[int, int], float, str]]):
        # 扩大样本范围至10行
        sample_lines = source_code.split("\n")[:10]
        sample = "\n".join(sample_lines)
        cb = (context_before or "")
        # 添加模式清理后的日志
        clean_pat = re.sub(r'\.\.\.', '', pattern).strip()
        logger.warning(
            f"定位失败: 原始模式='{pattern[:50]}...' 清理后='{clean_pat[:50]}'\n"
            f"上下文: '{cb[:30]}' 候选:{len(candidates)}\n"
            f"源代码样本:\n{sample}"
        )

class TransactionalEditor:
    def __init__(self, source_code: str, config: 'EditorConfig' = None, locator: ContentLocator = None):
        self.code = source_code
        self.locator = locator or ContentLocator()
        self.config = config or EditorConfig()
        self.locator.similarity_threshold = self.config.similarity_threshold
        self._original_code = source_code
        self._stats: Dict[str, Any] = {"ast_parses": 0, "locate_calls": 0, "validation_time": 0.0}

    def _timed_operation(self, name: str):
        import time
        class _Ctx:
            def __enter__(_self):
                _self._start = time.time()
            def __exit__(_self, exc_type, exc, tb):
                duration = time.time() - _self._start
                logger.debug(f"{name} 耗时: {duration:.3f}s")
                if name == "代码验证":
                    self._stats["validation_time"] += duration
        return _Ctx()

    def _validate(self, region_start: int, region_end: int) -> bool:
        with self._timed_operation("代码验证"):
            if self.config.max_modification_ratio < 1.0:
                o = len(self._original_code.split('\n'))
                n = len(self.code.split('\n'))
                ratio = abs(n - o) / o if o > 0 else 0.0
                if ratio > self.config.max_modification_ratio:
                    return False
            if self.config.enable_ast_validation:
                try:
                    ast.parse(self.code)
                    self._stats["ast_parses"] += 1
                except SyntaxError:
                    return False
            if self.config.enable_structure_check:
                if not self._check_structure_integrity():
                    return False
            if not self._check_brace_balance():
                return False
            return True

    def get_stats(self) -> Dict[str, Any]:
        return dict(self._stats)

    def _replace_region(self, start_line: int, end_line: int, new_text: str) -> None:
        lines = self.code.split("\n")
        if start_line < 0 or end_line > len(lines) or start_line > end_line:
            raise ValueError(f"无效的行范围: {start_line}-{end_line}")
        new_lines = new_text.split("\n") if new_text is not None else []
        if new_lines and self.config.preserve_code_style and start_line < len(lines):
            base_indent = self._get_indent_level(lines[start_line - 1]) if start_line > 0 else 0
            new_lines = self._adjust_indentation(new_lines, base_indent)
        self.code = "\n".join(lines[:start_line] + new_lines + lines[end_line:])
        try:
            self.locator.clear_cache()
        except Exception:
            pass

    def _get_indent_level(self, line: str) -> int:
        return len(line) - len(line.lstrip())

    def _adjust_indentation(self, lines: List[str], base_indent: int) -> List[str]:
        if not lines:
            return lines
        first_non_empty = next((l for l in lines if l.strip()), None)
        if not first_non_empty:
            return lines
        current_indent = self._get_indent_level(first_non_empty)
        diff = base_indent - current_indent
        if diff == 0:
            return lines
        adjusted: List[str] = []
        for l in lines:
            if l.strip():
                new_indent = max(0, self._get_indent_level(l) + diff)
                adjusted.append(' ' * new_indent + l.lstrip())
            else:
                adjusted.append(l)
        return adjusted

    def apply_operations(self, ops: List[LNFreeInstruction]) -> Tuple[str, List[LNFreeInstruction], List[str]]:
        batches = self._smart_group_operations(ops)
        original_code = self.code
        applied: List[LNFreeInstruction] = []
        errors: List[str] = []
        batches.sort(key=lambda b: self._batch_risk_level(b))
        for idx, batch in enumerate(batches):
            a, e = self._apply_batch_with_fallback(batch, idx + 1, len(batches))
            applied.extend(a)
            errors.extend(e)
            if e and self._batch_risk_level(batch) == "high" and not self.config.allow_partial_success:
                self.code = original_code
                return self.code, applied, errors + ["高风险操作失败，已回滚所有更改"]
        return self.code, applied, errors

    def _apply_batch_with_fallback(self, batch: List[LNFreeInstruction], i: int, n: int) -> Tuple[List[LNFreeInstruction], List[str]]:
        a: List[LNFreeInstruction] = []
        e: List[str] = []
        order = self._sort_operations(batch)
        for op in order:
            snapshot = self.code
            try:
                pos = None
                if op.type in ("REPLACE", "REPLACE NTH"):
                    target = op.old or ""
                    pos = self.locator.locate(self.code, target, nth=op.nth, context_before=op.context_before)
                    if not pos:
                        raise RuntimeError("未匹配到OLD片段")
                    self._replace_region(pos[0], pos[1], op.new or "")
                elif op.type in ("INSERT AFTER", "INSERT AFTER NTH"):
                    target = op.locator or ""
                    pos = self.locator.locate(self.code, target, nth=op.nth, context_before=op.context_before)
                    if not pos:
                        raise RuntimeError("未匹配到LOCATOR片段")
                    self._replace_region(pos[1], pos[1], op.content or "")
                elif op.type == "INSERT BEFORE":
                    target = op.locator or ""
                    pos = self.locator.locate(self.code, target, nth=op.nth, context_before=op.context_before)
                    if not pos:
                        raise RuntimeError("未匹配到LOCATOR片段")
                    self._replace_region(pos[0], pos[0], op.content or "")
                elif op.type == "DELETE":
                    target = op.locator or ""
                    pos = self.locator.locate(self.code, target, nth=op.nth, context_before=op.context_before)
                    if not pos:
                        raise RuntimeError("未匹配到LOCATOR片段")
                    self._replace_region(pos[0], pos[1], "")
                else:
                    raise RuntimeError("未知指令类型")
                ok, err = self._validate_with_healing(op)
                if not ok:
                    self.code = snapshot
                    e.append(err or "验证失败")
                else:
                    a.append(op)
                if not pos:
                    # 详细记录定位失败的上下文
                    snippet = target[:100] + "..." if len(target) > 100 else target
                    msg = f"定位失败 - 操作类型: {op.type}\n查找内容:\n{snippet}\n上下文:\n{op.context_before or 'None'}"
                    logger.warning(msg)
                    raise RuntimeError(f"无法定位代码块: {snippet[:30]}...")
            except Exception as ex:
                self.code = snapshot
                e.append(str(ex))
        return a, e

    def _validate_with_healing(self, operation: LNFreeInstruction) -> Tuple[bool, Optional[str]]:
        if not self._validate(1, 1):
            return False, "基础语法验证失败"
        if not self._semantic_consistency_check(operation):
            fix = self._attempt_semantic_healing(operation)
            if fix:
                return True, None
            return False, "语义一致性检查失败"
        if operation.type == "REPLACE":
            if not self._validate(1, 1):
                return False, "替换语义验证失败"
        return True, None

    def _semantic_consistency_check(self, operation: LNFreeInstruction) -> bool:
        try:
            target = (operation.old or operation.locator or "").strip()
            if operation.type in ("REPLACE", "INSERT AFTER") and target.startswith(("def ", "class ")):
                return True
            if "import" in target.lower():
                return True
            return True
        except Exception:
            return True

    def _attempt_semantic_healing(self, operation: LNFreeInstruction) -> Optional[str]:
        original = self.code
        try:
            healed = self._heal_indentation(operation)
        except Exception:
            healed = None
        if healed and healed != original and self._validate(1, 1):
            self.code = healed
            return "_heal_indentation"
        return None

    def _heal_indentation(self, operation: LNFreeInstruction) -> Optional[str]:
        """修复常见的缩进问题"""
        lines = self.code.split("\n")
        fixed = False
        for i in range(1, len(lines)):
            if lines[i].strip() and not lines[i-1].strip():
                prev_indent = len(lines[i-2]) - len(lines[i-2].lstrip()) if i >= 2 else 0
                curr_indent = len(lines[i]) - len(lines[i].lstrip())
                if abs(curr_indent - prev_indent) > 2:
                    lines[i] = ' ' * prev_indent + lines[i].lstrip()
                    fixed = True
        if fixed:
            return "\n".join(lines)
        return None

    def _smart_group_operations(self, ops: List[LNFreeInstruction]) -> List[List[LNFreeInstruction]]:
        positioned = []
        for op in ops:
            p = self._pre_locate_operation(op)
            positioned.append((op, p[0] if p else -1, p[1] if p else -1))
        positioned.sort(key=lambda x: x[1])
        batches: List[List[LNFreeInstruction]] = []
        current: List[LNFreeInstruction] = []
        last_end = -1
        for op, start_pos, end_pos in positioned:
            if start_pos > last_end + 5:
                current.append(op)
                last_end = max(last_end, end_pos if end_pos >= 0 else start_pos + 10)
            else:
                if current:
                    batches.append(current)
                current = [op]
                last_end = end_pos if end_pos >= 0 else start_pos + 10
        if current:
            batches.append(current)
        merged: List[List[LNFreeInstruction]] = []
        for b in batches:
            if not merged or len(merged[-1]) + len(b) > 3:
                merged.append(b)
            else:
                merged[-1].extend(b)
        return merged

    def _batch_risk_level(self, batch: List[LNFreeInstruction]):
        size = len(batch)
        types = set(op.type for op in batch)
        if size > 3 or "REPLACE" in types or "DELETE" in types:
            return "high"
        return "low"

    def _sort_operations(self, ops: List[LNFreeInstruction]) -> List[LNFreeInstruction]:
        positioned = []
        for op in ops:
            p = self._pre_locate_operation(op)
            positioned.append((p, op))
        positioned.sort(key=lambda x: x[0][0] if x[0] else 0, reverse=True)
        return [op for _, op in positioned]

    def _pre_locate_operation(self, op: LNFreeInstruction) -> Optional[Tuple[int, int]]:
        try:
            if op.type in ("REPLACE", "REPLACE NTH"):
                return self.locator.locate(self.code, op.old or "", op.nth, op.context_before)
            if op.type in ("INSERT AFTER", "INSERT AFTER NTH", "INSERT BEFORE", "DELETE"):
                return self.locator.locate(self.code, op.locator or "", op.nth, op.context_before)
        except Exception:
            return None
        return None

    def _check_structure_integrity(self) -> bool:
        try:
            tree = ast.parse(self.code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    if not hasattr(node, "body") or not node.body:
                        return False
            return True
        except Exception:
            return False

    def _check_brace_balance(self) -> bool:
        s = self.code
        stack: List[str] = []
        pairs = {"(": ")", "[": "]", "{": "}"}
        for ch in s:
            if ch in pairs:
                stack.append(ch)
            elif ch in pairs.values():
                if not stack or pairs[stack.pop()] != ch:
                    return False
        return len(stack) == 0


@dataclass
class EditorConfig:
    similarity_threshold: float = 0.85
    max_operations_per_batch: int = 10
    max_modification_ratio: float = 0.4
    enable_ast_validation: bool = True
    enable_structure_check: bool = True
    preserve_code_style: bool = True
    allow_partial_success: bool = False

class LineNumberFreeLLMBlockEditor:
    def __init__(self, llm_client=None, max_retries: int = 2, config: EditorConfig = None, locator_factory=None):
        self.llm_client = llm_client
        self.max_retries = max_retries
        self.config = config or EditorConfig()
        self.locator_factory = locator_factory or (lambda: ContentLocator())

    def edit_with_llm(self, original_code: str, instruction: str, context: str = "", file_path: str = "") -> LNFreeResult:
        if not self.llm_client:
            return LNFreeResult(False, original_code, [], ["LLM客户端不可用"], [], "", original_code)
        
        # 参数类型验证和安全转换，避免循环引用导致递归错误
        try:
            if not isinstance(instruction, str):
                instruction = str(instruction)
            if not isinstance(context, str):
                try:
                    context = str(context)
                except RecursionError:
                    context = "[错误: context 包含循环引用]"
                except Exception:
                    context = ""
        except RecursionError:
            return LNFreeResult(False, original_code, [], ["参数包含循环引用，导致递归错误"], [], "", original_code)
        except Exception as e:
            return LNFreeResult(False, original_code, [], [f"参数验证失败: {str(e)}"], [], "", original_code)
        
        parsed = self._parse_instructions(self.llm_client.one_chat(self._build_prompt(original_code, instruction, context)), original_code)
        if not parsed:
            try:
                from core.utils.editor_fallback import FallbackLLMEditor
                fb = FallbackLLMEditor(self.llm_client, prefer="block")
                fr = fb.edit_with_llm(original_code, instruction, context, file_path)
                if getattr(fr, 'success', False):
                    return LNFreeResult(True, fr.new_code, [], [], [], fr.diff, original_code)
            except Exception:
                pass
            return LNFreeResult(False, original_code, [], ["未能解析指令"], [], "", original_code)
        if len(parsed) > self.config.max_operations_per_batch:
            warn = f"操作数量超过限制: {len(parsed)} > {self.config.max_operations_per_batch}"
            return LNFreeResult(False, original_code, [], [warn], [], "", original_code)
        editor = TransactionalEditor(original_code, self.config, locator=self.locator_factory())
        editor._original_code = original_code
        new_code, applied, errs = editor.apply_operations(parsed)
        diff = self._diff(original_code, new_code)
        warnings: List[str] = []
        if errs and applied:
            warnings.append(f"部分操作成功: {len(applied)}/{len(parsed)}")
        success = len(errs) == 0 or (len(applied) > 0 and self.config.allow_partial_success)
        if not success:
            try:
                from core.utils.editor_fallback import FallbackLLMEditor
                fb = FallbackLLMEditor(self.llm_client, prefer="block")
                fr = fb.edit_with_llm(original_code, instruction, context, file_path)
                if getattr(fr, 'success', False):
                    return LNFreeResult(True, fr.new_code, applied, [], warnings, fr.diff, original_code)
            except Exception:
                pass
        return LNFreeResult(success, new_code, applied, errs, warnings, diff, original_code)

    def apply_instruction_string(self, original_code: str, instruction_string: str) -> LNFreeResult:
        parsed = self._parse_instructions(instruction_string, original_code)
        if not parsed:
            return LNFreeResult(False, original_code, [], ["未能解析指令"], [], "", original_code)
        if len(parsed) > self.config.max_operations_per_batch:
            warning = f"操作数量超过限制: {len(parsed)} > {self.config.max_operations_per_batch}"
            return LNFreeResult(False, original_code, [], [warning], [], "", original_code)
        editor = TransactionalEditor(original_code, self.config, locator=self.locator_factory())
        editor._original_code = original_code
        new_code, applied, errors = editor.apply_operations(parsed)
        diff = self._diff(original_code, new_code)
        warnings: List[str] = []
        if errors and applied:
            warnings.append(f"部分操作成功: {len(applied)}/{len(parsed)}")
        success = len(errors) == 0 or (len(applied) > 0 and self.config.allow_partial_success)
        return LNFreeResult(success, new_code, applied, errors, warnings, diff, original_code)

    def _build_prompt(self, original_code: str, instruction: str, context: str) -> str:
        # 安全转换参数，避免循环引用
        def safe_str(obj, default=""):
            try:
                if obj is None:
                    return default
                if isinstance(obj, str):
                    return obj
                return str(obj)
            except RecursionError:
                return "[错误: 参数包含循环引用]"
            except Exception:
                return default
        
        instruction = safe_str(instruction, "无修改需求")
        context = safe_str(context, "无")
        
        rel = self._extract_relevant_code(original_code, instruction, context)
        p = []
        p.append("# 无行号代码编辑指令系统")
        p.append("## 任务")
        p.append("根据需求修改以下Python代码，使用指定的无行号指令语法。")
        p.append("## 关键规则")
        p.append("1. **唯一性**: 必须提供足够上下文(CONTEXT_BEFORE)以唯一确定位置。")
        p.append("2. **完整性**: OLD/LOCATOR 必须完整展示要修改的代码块，包括缩进。")
        p.append("3. **缩进**: NEW/CONTENT 的缩进必须与源代码保持一致。")
        p.append("4. **签名**: 修改函数时必须包含完整函数签名。")
        p.append("5. **禁止**: 禁止输出任何额外解释，只输出指令块。")
        p.append("## 操作类型")
        p.append("- REPLACE: 替换现有代码块")
        p.append("- INSERT AFTER: 在指定代码块后插入")
        p.append("- INSERT BEFORE: 在指定代码块前插入")
        p.append("- DELETE: 删除指定代码块")
        p.append("## 指令格式示例")
        p.append("### 替换示例")
        p.append(">> REPLACE")
        p.append(">> OLD")
        p.append("def foo(x):")
        p.append("    return x + 1")
        p.append(">> NEW")
        p.append("def foo(x):")
        p.append("    return x + 2")
        p.append(">> CONTEXT_BEFORE")
        p.append("# This is a comment before foo")
        p.append("<< END")
        p.append("### 插入示例")
        p.append(">> INSERT AFTER")
        p.append(">> LOCATOR")
        p.append("    x = 1")
        p.append(">> CONTENT")
        p.append("    y = 2")
        p.append(">> CONTEXT_BEFORE")
        p.append("def bar():")
        p.append("<< END")
        p.append("## 待修改代码")
        p.append("```python")
        p.append(rel)
        p.append("```")
        p.append("## 需求")
        p.append(instruction)
        p.append("## 上下文")
        p.append(context)
        p.append("## 你的指令")
        return "\n".join(p)

    def _extract_relevant_code(self, code: str, instruction: str, context: str) -> str:
        import re as _re
        kws = _re.findall(r"\b\w+\b", (instruction or "") + (context or ""))
        imp = [k.lower() for k in kws if len(k) > 3 and k.lower() not in ["the", "and", "with", "for", "this"]]
        lines = code.split("\n")
        idxs = set()
        for i, line in enumerate(lines):
            lo = line.lower()
            if any(k in lo for k in imp[:5]):
                for j in range(max(0, i - 5), min(len(lines), i + 6)):
                    idxs.add(j)
        if not idxs or len(idxs) < 10:
            end_idx = min(50, len(lines))
            return "\n".join(lines[:end_idx])
        
        mn = max(0, min(idxs) - 3)
        mx = min(len(lines), max(idxs) + 3)
        return "\n".join(lines[mn:mx]) 

    def _parse_instructions(self, text: str, original_code: str) -> List[LNFreeInstruction]:
        cleaned = re.sub(r"```(?:\w+)?\s*\n", "", text)
        cleaned = re.sub(r"\n\s*```\s*$", "", cleaned)
        blocks: List[LNFreeInstruction] = []
        
        # 尝试JSON样式
        try:
            import ast as _ast
            # 尝试提取JSON块
            json_match = re.search(r"\[\s*\{.*\}\s*\]", cleaned, re.DOTALL)
            if json_match:
                data = _ast.literal_eval(json_match.group(0))
                if isinstance(data, list):
                    for item in data:
                        instr = self._parse_json_instruction(item)
                        if instr:
                            blocks.append(instr)
                    if blocks:
                        return blocks
        except Exception:
            pass

        # 增强的正则匹配，支持缺失 << END 的情况
        # 优先匹配完整的块
        # 注意：(?=>>\s*(?:REPLACE|INSERT|DELETE)) 确保只在遇到新指令头时停止，而不是遇到 >> OLD 等内部标记时停止
        full_pattern = r"(>>\s*(?:REPLACE|INSERT|DELETE)[\s\S]*?)(?:<<\s*END|(?=>>\s*(?:REPLACE|INSERT|DELETE))|$)"
        
        matches = list(re.finditer(full_pattern, cleaned, re.IGNORECASE))
        for m in matches:
            block_text = m.group(1).strip()
            # 补全 << END 如果缺失
            if not block_text.endswith("END"):
                block_text += "\n<< END"
            
            instr = self._parse_single_instruction(block_text)
            if instr:
                blocks.append(instr)
                
        if blocks:
            return blocks

        # 回退：状态机扫描 (保留原有逻辑作为最后防线)
        lines = cleaned.split("\n")
        i = 0
        while i < len(lines):
            ln = lines[i].strip()
            if ln.startswith(">>"):
                buf: List[str] = []
                while i < len(lines) and not lines[i].strip().startswith("<< END") and not (lines[i].strip().startswith(">>") and len(buf) > 0):
                    buf.append(lines[i])
                    i += 1
                if i < len(lines) and lines[i].strip().startswith("<< END"):
                    buf.append(lines[i])
                    i += 1
                else:
                    buf.append("<< END") # 强制结束
                
                instr = self._parse_single_instruction("\n".join(buf))
                if instr:
                    blocks.append(instr)
            else:
                i += 1
        return blocks

    def _parse_json_instruction(self, item: Any) -> Optional[LNFreeInstruction]:
        try:
            t = str(item.get('type', '')).upper()
            nth = int(item.get('nth', 1)) if isinstance(item.get('nth', 1), (int, str)) else 1
            old = item.get('old')
            new = item.get('new')
            locator = item.get('locator')
            content = item.get('content')
            context_before = item.get('context_before')
            instr = LNFreeInstruction(type=t, old=old, new=new, locator=locator, content=content, nth=max(1, nth), context_before=context_before, raw_text=str(item))
            if self._validate_instruction(instr.type if ' ' in instr.type else instr.type.replace('_', ' '), {
                'OLD': old or '',
                'NEW': new or '',
                'LOCATOR': locator or '',
                'CONTENT': content or '',
                'CONTEXT_BEFORE': context_before or ''
            }):
            
                return instr
        except Exception:
            return None

        return None

    def _parse_single_instruction(self, block_text: str) -> Optional[LNFreeInstruction]:
        logger.debug(f"Parsing block:\n{block_text}")
        lines = [ln.rstrip() for ln in block_text.split('\n') if ln.strip()]
        if not lines or not lines[0].startswith('>>'):
            return None
        
        header = lines[0].strip()
        t, nth = self._parse_instruction_header(header)
        if len(lines) == 1:
            return None
        
        sections = self._extract_sections_robust(lines[1:])
        logger.debug(f"Extracted sections: {list(sections.keys())}")
        
        # 清理所有部分中的省略号和多余空格
        for key in sections:
            if isinstance(sections[key], str):
                # 移除省略号、多余空格和常见LLM生成的占位符
                cleaned = re.sub(r'\.\.\.|<omitted>|# \.\.\.|# omitted', '', sections[key])
                sections[key] = re.sub(r'\n\s*\n', '\n', cleaned).strip()
        
        # 添加缺失的上下文（当LOCATOR存在但CONTEXT_BEFORE缺失时）
        if "LOCATOR" in sections and "CONTEXT_BEFORE" not in sections:
            locator_lines = sections["LOCATOR"].split("\n")
            if len(locator_lines) > 1:
                # 使用定位器的第一行作为上下文
                sections["CONTEXT_BEFORE"] = locator_lines[0].strip()
            else:
                # 尝试从原始代码中提取上下文
                sections["CONTEXT_BEFORE"] = locator_lines[0].strip()[:20]
        
        if not self._validate_instruction(t, sections):
            logger.warning(f"指令验证失败: {t}, 内容: {sections}")
            return None
        
        # 创建指令对象
        instr = LNFreeInstruction(
            type=t,
            old=sections.get('OLD'),
            new=sections.get('NEW'),
            locator=sections.get('LOCATOR'),
            content=sections.get('CONTENT'),
            nth=nth,
            context_before=sections.get('CONTEXT_BEFORE'),
            raw_text=block_text,
        )
        
        # 额外验证：确保关键字段不为空
        if instr.type == "REPLACE" and (not instr.old or not instr.new):
            logger.warning(f"REPLACE指令缺少必要字段: old={bool(instr.old)}, new={bool(instr.new)}")
            return None
        if instr.type in ("INSERT AFTER", "INSERT BEFORE", "DELETE") and not instr.locator:
            logger.warning(f"{instr.type}指令缺少LOCATOR字段")
            return None
        
        return instr

    def _parse_instruction_header(self, header: str) -> Tuple[str, int]:
        try:
            nth = 1
            t = header.replace(">>", "").strip()
            m = re.search(r"NTH\s+(\d+)", t, re.IGNORECASE)
            if m:
                nth = max(1, int(m.group(1)))
                t = re.sub(r"\s*NTH\s+\d+", "", t, flags=re.IGNORECASE).strip()
            t = t.upper()
            return t, nth
        except Exception as e:
            logger.warning(f"指令头解析失败: {header}, 错误: {e}")
            return "UNKNOWN", 1

    def _extract_sections_robust(self, lines: List[str]) -> Dict[str, str]:
        sections: Dict[str, str] = {}
        current = None
        buf: List[str] = []
        for line in lines:
            s = line.strip()
            if s == '<< END':
                if current and buf:
                    sections[current] = '\n'.join(buf).strip()
                break
            if s.startswith('>>'):
                if current and buf:
                    sections[current] = '\n'.join(buf).strip()
                current = s[2:].strip().upper()
                buf = []
            elif current:
                buf.append(line)
        if current and buf and current not in sections:
            sections[current] = '\n'.join(buf).strip()
        
        logger.debug(f"Extracted raw sections: {list(sections.keys())}")
        return sections

    def _validate_instruction(self, t: str, sections: Dict[str, str]) -> bool:
        """
        验证指令的必要字段是否存在且有效
        """
        if not isinstance(sections, dict):
            logger.warning(f"无效的sections类型: {type(sections)}, 预期dict")
            return False
        
        # 1. 清理字段值（防御性编程）
        cleaned = {}
        for key, value in sections.items():
            if isinstance(value, str):
                # 移除可能导致问题的特殊字符和LLM占位符
                cleaned_value = re.sub(r'[`*]', '', value)
                cleaned_value = re.sub(r'\.\.\.|<omitted>|# \.\.\.|# omitted', '', cleaned_value)
                cleaned_value = cleaned_value.strip()
                if cleaned_value:  # 仅保留非空值
                    cleaned[key] = cleaned_value
        
        # 2. 基本类型验证
        valid_types = {
            "REPLACE": ["OLD", "NEW"],
            "INSERT AFTER": ["LOCATOR", "CONTENT"],
            "INSERT BEFORE": ["LOCATOR", "CONTENT"],
            "DELETE": ["LOCATOR"],
            "REPLACE NTH": ["OLD", "NEW"],
            "INSERT AFTER NTH": ["LOCATOR", "CONTENT"]
        }
        
        if t not in valid_types:
            logger.warning(f"无效的指令类型: '{t}'，支持的类型: {list(valid_types.keys())}")
            return False
        
        # 3. 检查必要字段
        required_fields = valid_types[t]
        missing = [field for field in required_fields if field not in cleaned or not cleaned[field]]
        
        if missing:
            logger.warning(
                f"指令 '{t}' 缺少必要字段: {missing}\n"
                f"原始Sections keys: {list(sections.keys())}\n"
                f"Cleaned keys: {list(cleaned.keys())}\n"
                f"Cleaned content: { {k: v[:30]+'...' if len(v)>30 else v for k,v in cleaned.items()} }"
            )
            return False
        
        # 4. 内容长度验证（防止空/无效操作）
        if t in ("REPLACE", "REPLACE NTH"):
            if len(cleaned["OLD"]) < 5 or len(cleaned["NEW"]) < 5:
                logger.warning(f"REPLACE内容过短: old={len(cleaned['OLD'])}, new={len(cleaned['NEW'])}")
                return False
            # 防止替换为空内容
            if not cleaned["NEW"].strip():
                logger.warning("禁止替换为空内容")
                return False
        
        if t in ("INSERT AFTER", "INSERT BEFORE", "DELETE", "INSERT AFTER NTH"):
            if len(cleaned["LOCATOR"]) < 5:
                logger.warning(f"LOCATOR内容过短 ({len(cleaned['LOCATOR'])}字符): '{cleaned['LOCATOR'][:20]}...'")
                return False
        
        # 5. 语义验证：防止无效替换
        if t in ("REPLACE", "REPLACE NTH") and "LOCATOR" in cleaned:
            locator_norm = _CodeNormalizer.canonical(cleaned["LOCATOR"])
            old_norm = _CodeNormalizer.canonical(cleaned["OLD"])
            # LOCATOR应该与OLD内容高度相似
            similarity = difflib.SequenceMatcher(None, locator_norm, old_norm).ratio()
            if similarity < 0.6:
                logger.warning(f"LOCATOR与OLD内容不匹配 (相似度={similarity:.2f}):\nLOCATOR: {locator_norm[:50]}\nOLD: {old_norm[:50]}")
                return False
        
        # 6. 防御性检查：LOCATOR不应与NEW内容相同
        if t in ("REPLACE", "REPLACE NTH") and "NEW" in cleaned:
            old_norm = _CodeNormalizer.canonical(cleaned["OLD"])
            new_norm = _CodeNormalizer.canonical(cleaned["NEW"])
            if old_norm == new_norm:
                logger.warning("无效指令：OLD与NEW内容完全相同")
                return False
        
        # 7. 上下文验证
        if "CONTEXT_BEFORE" in cleaned:
            if len(cleaned["CONTEXT_BEFORE"]) < 3:
                logger.warning(f"上下文过短 ({len(cleaned['CONTEXT_BEFORE'])}字符)")
                return False
            # 防止包含特殊字符
            if re.search(r'[`*{}[\]]', cleaned["CONTEXT_BEFORE"]):
                cleaned["CONTEXT_BEFORE"] = re.sub(r'[`*{}[\]]', '', cleaned["CONTEXT_BEFORE"])
                logger.debug("清理了CONTEXT_BEFORE中的特殊字符")
        
        return True

    def _diff(self, old_code: str, new_code: str) -> str:
        return "\n".join(difflib.unified_diff(old_code.split("\n"), new_code.split("\n"), lineterm="", n=3))