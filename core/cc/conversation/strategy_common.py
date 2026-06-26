from __future__ import annotations

import json
import re
from typing import Any


PATH_RE = re.compile(
    r"([A-Za-z]:)?[A-Za-z0-9_./\\-]+\.(py|md|ts|tsx|js|jsx|json|yml|yaml|toml|ini|txt)\b",
    re.IGNORECASE,
)


# Broader matcher than ``PATH_RE``: catches directory-only path tokens
# like ``core/deepstack-agent/stock_rec_v3`` that ``PATH_RE`` rejects
# because there's no recognized file extension. Used by the "Paths in
# this task" block (see ``mode_strategy.build_paths_in_request_block``)
# to surface user-named directories the truncated repository outline
# may have hidden.
_URL_RE = re.compile(r"\b(?:https?|ftps?|ssh|git)://\S+", re.IGNORECASE)
_PATH_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?:[A-Za-z]:)?"
    r"[A-Za-z0-9_.][A-Za-z0-9_./\\-]*"
    r"/[A-Za-z0-9_./\\-]+",
)


def coerce_text(user_input: str | list[dict[str, object]]) -> str:
    if isinstance(user_input, str):
        text = user_input.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                return text
            if isinstance(payload, dict):
                pieces = [
                    str(payload.get("goal") or ""),
                    " ".join(str(item) for item in payload.get("constraints", []) if item),
                    " ".join(str(item) for item in payload.get("acceptance_criteria", []) if item),
                ]
                merged = " ".join(part.strip() for part in pieces if part and str(part).strip())
                if merged:
                    return merged
        return text
    return str(user_input)


def count_hint_hits(text: str, hints: tuple[str, ...]) -> int:
    return sum(1 for hint in hints if hint in text)


def summarize_source_text(text: str, *, limit: int = 240) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def contains_target_path(text: str) -> bool:
    return bool(PATH_RE.search(text))


def extract_path_tokens(text: str) -> list[str]:
    """Return path-like substrings from ``text``.

    Heuristics:
    * Must contain at least one ``/`` (extension-only references like
      ``foo.py`` are picked up by ``PATH_RE`` separately).
    * URLs are stripped first so domain components don't leak in.
    * Backslash separators are normalized to forward slashes.
    * Trailing punctuation (``.,;:`` and a trailing ``/``) is stripped.
    * Returns unique tokens preserving first-seen order.

    Used to surface user-named paths even when the repository outline
    truncates them out (e.g. ``core/`` has 51 subdirectories but the
    outline only shows the alphabetically-first ~8).
    """
    if not text:
        return []
    cleaned_text = _URL_RE.sub(" ", text)
    seen: set[str] = set()
    out: list[str] = []
    for raw in _PATH_TOKEN_RE.findall(cleaned_text):
        cleaned = raw.replace("\\", "/").rstrip(" .,;:/")
        if not cleaned or "/" not in cleaned:
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out
