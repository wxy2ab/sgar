"""cross-run persistent memory; for single-chain resume see deepstack_v5.memory."""

from __future__ import annotations

from typing import Any


MEMORY_PROMPT_METADATA_KEY = "ccx.memory.prompt_block"


def install_memory_metadata(metadata: dict[str, Any], block: str) -> dict[str, Any]:
    new_meta = dict(metadata or {})
    text = str(block or "")
    if text:
        new_meta[MEMORY_PROMPT_METADATA_KEY] = text
    return new_meta


def read_memory_block(metadata: dict[str, Any] | None) -> str:
    if not metadata:
        return ""
    value = metadata.get(MEMORY_PROMPT_METADATA_KEY)
    if not isinstance(value, str):
        return ""
    return value.strip()


__all__ = [
    "MEMORY_PROMPT_METADATA_KEY",
    "install_memory_metadata",
    "read_memory_block",
]
