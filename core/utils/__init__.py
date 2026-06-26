"""Utility package exports with lazy loading.

This module intentionally avoids importing every helper eagerly because
many of them pull in optional heavyweight dependencies. The public API
remains the same, but attributes are loaded on first access.
"""

from __future__ import annotations

from importlib import import_module
from typing import Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "CodeIndex": ("core.utils.code_index", "CodeIndex"),
    "CodeSummarizer": ("core.utils.code_summarizer", "CodeSummarizer"),
    "ContextExtractor": ("core.utils.context_extractor", "ContextExtractor"),
    "CodeValidator": ("core.utils.code_validator", "CodeValidator"),
    "BackupManager": ("core.utils.code_validator", "BackupManager"),
    "ErrorChecker": ("core.utils.code_validator", "ErrorChecker"),
    "CodeAnalyzer": ("core.utils.code_analyzer", "CodeAnalyzer"),
    "CacheManager": ("core.utils.cache_manager", "CacheManager"),
    "CodeEditor": ("core.utils.code_editor", "CodeEditor"),
    "fetch_complete_code_with_chat": ("core.utils.complete_code_fetcher", "fetch_complete_code_with_chat"),
    "LineNumberFreeLLMBlockEditor": ("core.utils.llm_block_editor_lnfree", "LineNumberFreeLLMBlockEditor"),
    "FallbackLLMEditor": ("core.utils.editor_fallback", "FallbackLLMEditor"),
    "RobustLLMEditor": ("core.utils.robust_llm_editor", "RobustLLMEditor"),
    "SmartLLMEditorV2": ("core.utils.smart_llm_editor_v2", "SmartLLMEditorV2"),
    "AutonomousCodeAgent": ("core.utils.autonomous_code_agent", "AutonomousCodeAgent"),
    "InlineCodeEditor": ("core.utils.inline_code_editor", "InlineCodeEditor"),
    "InlineEditResult": ("core.utils.inline_code_editor", "InlineEditResult"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'core.utils' has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
