from __future__ import annotations

from ..config import CCConfig
from .models import MemoryWriteCandidate


STRUCTURAL_MEMORY_KINDS = {
    "repo_structure",
    "module_map",
    "architecture",
    "decision",
    "failure_mode",
    "compact_summary",
}

_STRUCTURAL_KEYWORDS = (
    "architecture",
    "component",
    "directory",
    "module",
    "repository",
    "structure",
    "tree",
    "仓库",
    "模块",
    "架构",
    "目录",
    "结构",
)


def recall_enabled(config: CCConfig) -> bool:
    return bool(config.memory_enabled and config.memory_auto_recall)


def store_enabled(config: CCConfig) -> bool:
    return bool(config.memory_enabled and config.memory_auto_store)


def is_structural_kind(memory_kind: str) -> bool:
    return memory_kind in STRUCTURAL_MEMORY_KINDS


def infer_room(memory_kind: str) -> str:
    return {
        "repo_structure": "repo-structure",
        "module_map": "module-map",
        "architecture": "architecture",
        "decision": "design-decisions",
        "failure_mode": "failure-modes",
        "compact_summary": "compact-summary",
    }.get(memory_kind, "general")


def should_store_candidate(candidate: MemoryWriteCandidate, config: CCConfig) -> bool:
    if not store_enabled(config):
        return False
    if config.memory_store_structural_only and not is_structural_kind(candidate.memory_kind):
        return False
    if candidate.memory_kind in STRUCTURAL_MEMORY_KINDS:
        return True
    lowered = f"{candidate.summary}\n{candidate.text}".lower()
    return any(keyword in lowered for keyword in _STRUCTURAL_KEYWORDS)


def trim_text(text: str, max_chars: int) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max(0, max_chars - 3)]}..."
