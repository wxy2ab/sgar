"""cross-run persistent memory; for single-chain resume see deepstack_v5.memory."""

from __future__ import annotations

import logging
from typing import Any

from core.cc.api import AgentRunRequest, AgentRunResult

from ..modes import LLMCallable, text_of
from ..modes.parsing import parse_llm_json
from .models import (
    MemoryEntry,
    MemoryOptions,
    make_memory_entry,
    memory_disabled,
    normalize_tags,
    request_memory_tags,
)


logger = logging.getLogger(__name__)

GOAL_EXCERPT_CHARS = 2000
RESULT_EXCERPT_CHARS = 6000

SYSTEM_PROMPT = """\
You are the memory writer for an autonomous coding agent. After each run
you distill what the NEXT run in this project must remember.

Rules:
- Return STRICT JSON: {{"entries": [...]}} and nothing else.
- 0 entries is a normal answer. Store nothing for routine, uneventful runs.
- At most {max_entries} entries.
- Each entry: {{"kind": "decision|outcome|constraint|failure_mode",
  "title": "...", "text": "...", "tags": ["..."], "keywords": ["..."]}}
- title <= 100 chars. text <= {entry_text_max_chars} chars, plain markdown,
  self-contained (the next reader has no transcript of this run).
- Write title and text in the same language as the goal.
- tags: lowercase, [a-z0-9_-] only. REUSE the existing vocabulary below
  whenever a tag fits; invent a new tag only for a genuinely new topic.
- keywords: short literal strings likely to appear in future task texts
  (file names, module names, domain terms; Chinese or English).
- Prefer durable facts: decisions made, constraints discovered, what
  failed and why. Do not narrate the steps taken.
- Never include secrets, tokens, API keys, or credentials.

Existing tag vocabulary (tag: count):
{vocab_lines}
"""


def summarize_run(
    *,
    llm: LLMCallable,
    request: AgentRunRequest,
    result: AgentRunResult,
    options: MemoryOptions,
    existing_tags: list[tuple[str, int]],
    mode: str | None = None,
) -> list[MemoryEntry]:
    if not options.auto_summarize:
        return []
    if memory_disabled(request.metadata):
        return []
    if not result.final_text and not (result.failed and result.error_message):
        return []
    try:
        raw = llm(
            system=_system_prompt(options=options, existing_tags=existing_tags),
            user=_user_prompt(request=request, result=result),
            purpose="ccx.memory.summarize",
        )
        parsed = parse_llm_json(
            text_of(raw),
            schema_name="ccx_memory_summary",
            fallback_factory=lambda _text: {},
            expected_type=dict,
        )
        entries_raw = parsed.get("entries") if isinstance(parsed, dict) else None
        if not isinstance(entries_raw, list):
            return []
        return _entries_from_payload(
            entries_raw,
            request=request,
            result=result,
            options=options,
            mode=mode,
        )
    except Exception:
        logger.warning("ccx memory: summarize failed", exc_info=True)
        return []


def _system_prompt(
    *,
    options: MemoryOptions,
    existing_tags: list[tuple[str, int]],
) -> str:
    vocab_lines = "\n".join(
        f"{tag}: {count}" for tag, count in existing_tags[:30]
    ) or "(empty)"
    return SYSTEM_PROMPT.format(
        max_entries=max(0, options.max_entries_per_run),
        entry_text_max_chars=max(0, options.entry_text_max_chars),
        vocab_lines=vocab_lines,
    )


def _user_prompt(
    *,
    request: AgentRunRequest,
    result: AgentRunResult,
) -> str:
    goal_excerpt = str(request.instruction or "")[:GOAL_EXCERPT_CHARS]
    result_excerpt = str(result.final_text or "")[:RESULT_EXCERPT_CHARS]
    parts = [
        f"Run status: {result.session_snapshot.get('status', '')}, "
        f"failed={result.failed}",
        "Goal:",
        goal_excerpt,
        "",
        "Final result:",
        result_excerpt,
    ]
    if result.failed and result.error_message:
        parts.extend(["", f"Error: {result.error_message}"])
    parts.extend(["", "Return the JSON now."])
    return "\n".join(parts)


def _entries_from_payload(
    entries_raw: list[Any],
    *,
    request: AgentRunRequest,
    result: AgentRunResult,
    options: MemoryOptions,
    mode: str | None,
) -> list[MemoryEntry]:
    extra_tags = normalize_tags((
        *options.tags,
        *request_memory_tags(request.metadata),
    ))
    out: list[MemoryEntry] = []
    resolved_mode = mode or request.agent_mode or ""
    for raw in entries_raw:
        if len(out) >= max(0, options.max_entries_per_run):
            break
        if not isinstance(raw, dict):
            continue
        title = str(raw.get("title") or "").strip()
        text = str(raw.get("text") or "").strip()
        if not title or not text:
            continue
        tags_raw = raw.get("tags") if isinstance(raw.get("tags"), list) else []
        keywords_raw = (
            raw.get("keywords") if isinstance(raw.get("keywords"), list) else []
        )
        entry = make_memory_entry(
            run_id=result.session_id,
            mode=resolved_mode,
            kind=raw.get("kind") or "outcome",
            title=title,
            text=text,
            tags=(*extra_tags, *tags_raw),
            keywords=keywords_raw,
            entry_text_max_chars=options.entry_text_max_chars,
        )
        if entry.title and entry.text:
            out.append(entry)
    return out


__all__ = ["summarize_run"]
