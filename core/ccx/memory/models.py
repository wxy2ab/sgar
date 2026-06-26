"""cross-run persistent memory; for single-chain resume see deepstack_v5.memory."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any


SCHEMA_VERSION = 1
VALID_KINDS = {"decision", "outcome", "constraint", "failure_mode"}
TAG_RE = re.compile(r"^[a-z0-9_-]{1,32}$")
DEFAULT_ENTRY_TEXT_MAX_CHARS = 1200
MAX_TITLE_CHARS = 120
MAX_TAGS = 10
MAX_KEYWORDS = 8
MAX_KEYWORD_CHARS = 40


@dataclass(frozen=True, slots=True)
class MemoryOptions:
    enabled: bool = True
    root: str | None = None
    auto_recall: bool = True
    auto_summarize: bool = True
    tags: tuple[str, ...] = ()
    max_entries_per_run: int = 3
    entry_text_max_chars: int = DEFAULT_ENTRY_TEXT_MAX_CHARS
    recall_top_k: int = 5
    recall_char_budget: int = 4000
    recall_max_age_days: int = 90
    half_life_days: float = 14.0
    max_total_entries: int = 500
    summary_route: str | None = None


@dataclass(slots=True)
class MemoryEntry:
    id: str
    schema_version: int = SCHEMA_VERSION
    created_at: str = ""
    run_id: str = ""
    mode: str = ""
    kind: str = "outcome"
    title: str = ""
    text: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)
    keywords: tuple[str, ...] = field(default_factory=tuple)
    pinned: bool = False
    expires_at: str | None = None
    fingerprint: str = ""
    source: str = "auto_summary"

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "run_id": self.run_id,
            "mode": self.mode,
            "kind": self.kind,
            "title": self.title,
            "text": self.text,
            "tags": list(self.tags),
            "keywords": list(self.keywords),
            "pinned": self.pinned,
            "expires_at": self.expires_at,
            "fingerprint": self.fingerprint,
            "source": self.source,
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "MemoryEntry":
        created_at = str(data.get("created_at") or now_iso())
        entry = cls(
            id=str(data.get("id") or ""),
            schema_version=int(data.get("schema_version") or SCHEMA_VERSION),
            created_at=created_at,
            run_id=str(data.get("run_id") or ""),
            mode=str(data.get("mode") or ""),
            kind=str(data.get("kind") or "outcome"),
            title=str(data.get("title") or ""),
            text=str(data.get("text") or ""),
            tags=_tuple_if_iterable(data.get("tags")),
            keywords=_tuple_if_iterable(data.get("keywords")),
            pinned=bool(data.get("pinned") or False),
            expires_at=(
                str(data.get("expires_at"))
                if data.get("expires_at") is not None else None
            ),
            fingerprint=str(data.get("fingerprint") or ""),
            source=str(data.get("source") or "auto_summary"),
        )
        return normalize_entry(entry, truncate=False, preserve_fingerprint=True)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def parse_datetime(value: str | None, *, default_now: bool = False) -> datetime | None:
    if not value:
        return datetime.now().astimezone() if default_now else None
    try:
        normalized = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
    except (TypeError, ValueError):
        return datetime.now().astimezone() if default_now else None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return dt


def normalize_text_for_fingerprint(value: str) -> str:
    return " ".join(str(value or "").lower().split())


def compute_fingerprint(kind: str, title: str, text: str) -> str:
    raw = "|".join((
        normalize_kind(kind),
        normalize_text_for_fingerprint(title),
        normalize_text_for_fingerprint(text),
    ))
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def make_memory_id(fingerprint: str, created_at: str) -> str:
    raw = f"{fingerprint}{created_at}"
    return "mem_" + hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]


def normalize_kind(kind: Any) -> str:
    value = str(kind or "").strip()
    return value if value in VALID_KINDS else "outcome"


def truncate_text(value: str, max_chars: int) -> str:
    text = str(value or "").strip()
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars == 1:
        return "…"
    return text[: max_chars - 1].rstrip() + "…"


def normalize_tags(values: Iterable[Any]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        tag = value.strip().lower()
        if not TAG_RE.fullmatch(tag):
            continue
        if tag in seen:
            continue
        seen.add(tag)
        out.append(tag)
        if len(out) >= MAX_TAGS:
            break
    return tuple(out)


def normalize_keywords(values: Iterable[Any]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        keyword = value.strip()
        if not keyword:
            continue
        keyword = keyword[:MAX_KEYWORD_CHARS]
        dedupe_key = keyword.lower()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        out.append(keyword)
        if len(out) >= MAX_KEYWORDS:
            break
    return tuple(out)


def request_memory_tags(metadata: dict[str, Any] | None) -> tuple[str, ...]:
    raw = (metadata or {}).get("ccx_memory_tags")
    if isinstance(raw, str):
        return normalize_tags((raw,))
    if isinstance(raw, Iterable):
        return normalize_tags(raw)
    return ()


def _tuple_if_iterable(value: Any) -> tuple[Any, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(value)
    return ()


def memory_disabled(metadata: dict[str, Any] | None) -> bool:
    return str((metadata or {}).get("ccx_memory") or "").lower() == "off"


def normalize_entry(
    entry: MemoryEntry,
    *,
    entry_text_max_chars: int = DEFAULT_ENTRY_TEXT_MAX_CHARS,
    truncate: bool = True,
    preserve_fingerprint: bool = False,
) -> MemoryEntry:
    kind = normalize_kind(entry.kind)
    title = truncate_text(entry.title, MAX_TITLE_CHARS)
    text = (
        truncate_text(entry.text, entry_text_max_chars)
        if truncate else str(entry.text or "").strip()
    )
    created_at = entry.created_at or now_iso()
    stored_fingerprint = str(entry.fingerprint or "").strip()
    fingerprint = (
        stored_fingerprint
        if preserve_fingerprint and stored_fingerprint
        else compute_fingerprint(kind, title, text)
    )
    entry_id = entry.id or make_memory_id(fingerprint, created_at)
    return replace(
        entry,
        id=entry_id,
        schema_version=SCHEMA_VERSION,
        created_at=created_at,
        kind=kind,
        title=title,
        text=text,
        tags=normalize_tags(entry.tags),
        keywords=normalize_keywords(entry.keywords),
        fingerprint=fingerprint,
        source=entry.source or "auto_summary",
    )


def make_memory_entry(
    *,
    run_id: str,
    mode: str,
    kind: Any,
    title: str,
    text: str,
    tags: Iterable[Any] = (),
    keywords: Iterable[Any] = (),
    entry_text_max_chars: int = DEFAULT_ENTRY_TEXT_MAX_CHARS,
    created_at: str | None = None,
    pinned: bool = False,
    expires_at: str | None = None,
    source: str = "auto_summary",
) -> MemoryEntry:
    created = created_at or now_iso()
    entry = MemoryEntry(
        id="",
        created_at=created,
        run_id=str(run_id or ""),
        mode=str(mode or ""),
        kind=str(kind or "outcome"),
        title=title,
        text=text,
        tags=tuple(tags),
        keywords=tuple(keywords),
        pinned=bool(pinned),
        expires_at=expires_at,
        source=source,
    )
    return normalize_entry(entry, entry_text_max_chars=entry_text_max_chars)


__all__ = [
    "MemoryEntry",
    "MemoryOptions",
    "SCHEMA_VERSION",
    "compute_fingerprint",
    "make_memory_entry",
    "memory_disabled",
    "normalize_entry",
    "normalize_keywords",
    "normalize_tags",
    "parse_datetime",
    "request_memory_tags",
    "truncate_text",
]
