"""Loader for cc-side system prompts that ccx mode runners reuse.

cc's ``core/cc/prompts/system/`` already hosts polished doc / ask
prompts in both Chinese and English. ccx mirrors the same prompt
contract so we treat cc's files as the single source of truth and just
read them on demand. Avoids duplication / drift.
"""

from __future__ import annotations

import functools
from pathlib import Path


_CC_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "cc" / "prompts" / "system"


@functools.lru_cache(maxsize=16)
def load_cc_system_prompt(name: str, language: str = "en") -> str:
    """Load and cache one of cc's system prompts.

    Args:
        name: prompt stem, e.g. ``doc_mode`` or ``ask_mode`` (no
            extension, no ``.zh`` / ``.en`` suffix).
        language: ``zh`` or ``en``. Anything starting with ``zh`` maps to
            Chinese; everything else to English (matching cc's convention).

    Returns the raw markdown text. Raises ``FileNotFoundError`` if cc's
    prompts directory is missing or the requested file does not exist —
    caller should treat this as a configuration error.
    """
    suffix = "zh" if language.startswith("zh") else "en"
    path = _CC_PROMPTS_DIR / f"{name}.{suffix}.md"
    return path.read_text(encoding="utf-8")


def cc_prompts_dir() -> Path:
    """Return the directory cc system prompts live in. Mostly for tests."""
    return _CC_PROMPTS_DIR


__all__ = ["load_cc_system_prompt", "cc_prompts_dir"]
