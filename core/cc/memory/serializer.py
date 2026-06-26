from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import MemoryRecallBundle, MemoryWriteCandidate
from .policy import trim_text


def summarize_recall_bundle(bundle: MemoryRecallBundle, *, max_chars: int) -> str:
    if not bundle.available:
        return bundle.error or "Memory provider unavailable."
    lines = [f"provider={bundle.provider}", f'query="{trim_text(bundle.query, 120)}"']
    if bundle.hits:
        for index, hit in enumerate(bundle.hits, start=1):
            lines.append(
                f"{index}. [{hit.wing}/{hit.room}] {trim_text(hit.summary, 180)} "
                f"(source={hit.source_file}, similarity={hit.similarity})"
            )
    if bundle.facts:
        for fact in bundle.facts[:5]:
            lines.append(f"fact: {fact.subject} -> {fact.predicate} -> {fact.object}")
    return trim_text("\n".join(lines), max_chars)


def render_candidate_text(candidate: MemoryWriteCandidate, *, max_chars: int) -> str:
    payload = {
        "memory_kind": candidate.memory_kind,
        "subject": candidate.subject,
        "summary": candidate.summary,
        "details": candidate.details,
        "sources": candidate.sources,
    }
    prefix = json.dumps(payload, ensure_ascii=False, indent=2)
    suffix = trim_text(candidate.text, max_chars)
    combined = f"{prefix}\n\n{suffix}" if suffix else prefix
    return trim_text(combined, max_chars)


def outline_to_details(outline: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(outline, dict):
        return {}
    return {
        "root": outline.get("root"),
        "max_depth": outline.get("max_depth"),
        "max_entries_per_dir": outline.get("max_entries_per_dir"),
        "text": outline.get("text"),
    }


def repository_subject(cwd: str) -> str:
    return Path(cwd).resolve().name or "repository"
