"""ResumeSnapshot — build a priority-filtered view of a prior run.

Reads the existing v5 ``events`` table (no new schema). A snapshot is a
*derived* view, not stored state: build it on demand at the start of a
new run that wants context from a previous run, or right before a
compaction to capture state that would otherwise be lost.

Algorithm (greedy with priority ordering, char budget bounded):

1. Read all events for ``run_id`` with a single ``EventStore.read_after``
   call. Because the query is one SQL statement, the result set is
   atomic at SQLite level — no torn read possible. The largest sequence
   in the result is recorded as ``highwater_sequence`` so a downstream
   consumer can prove "this snapshot reflects events up through N".

2. Classify each event with :func:`priority_for`. Priority 4 (critical
   failures) is included newest-first up to ``max_priority4_events``.
   Priorities 3, 2, 1 are added in descending priority order, newest-first
   within a priority bucket, until ``token_budget_chars`` is exhausted.
   Large payload excerpt values are byte-capped before rendering.

3. Aggregate a short ``summary`` from the failures + completion counts
   so the LLM gets a one-paragraph headline before the event list.

The output is intentionally simple: a list of ``EventRef`` dataclasses
plus a summary string. Downstream code (``resume.py``) renders them
into a prompt block.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from .priority import priority_for

if TYPE_CHECKING:
    from ..persistence.stores import EventStore


@dataclass(slots=True)
class EventRef:
    """A compact, prompt-friendly reference to one v5 event."""

    sequence: int
    kind: str
    priority: int
    payload_excerpt: dict[str, Any]
    occurred_at_ms: int


@dataclass(slots=True)
class ResumeSnapshot:
    run_id: str
    summary: str
    highwater_sequence: int
    events: list[EventRef] = field(default_factory=list)
    built_at_ms: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.events and self.highwater_sequence == 0


# Keys we keep when copying a payload into ``payload_excerpt`` — every
# other key is dropped to keep the snapshot prompt-sized. Listed
# explicitly so a newly added cc bridge field doesn't silently bloat
# every snapshot.
_PAYLOAD_EXCERPT_KEYS: frozenset[str] = frozenset({
    "tool_name",
    "tool_use_id",
    "success",
    "error_code",
    "duration_ms",
    "reason",
    "preview",
    "command",
    "pattern",
    "path",
    "query",
    "node_id",
    "message",
    "kind",
})

_MAX_EXCERPT_VALUE_BYTES = 2_048
_TRUNCATION_SUFFIX = "...[truncated]"


def _truncate_excerpt_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    encoded = value.encode("utf-8")
    if len(encoded) <= _MAX_EXCERPT_VALUE_BYTES:
        return value
    suffix = _TRUNCATION_SUFFIX.encode("utf-8")
    head_limit = max(0, _MAX_EXCERPT_VALUE_BYTES - len(suffix))
    return encoded[:head_limit].decode("utf-8", errors="ignore") + _TRUNCATION_SUFFIX


def _excerpt(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        k: _truncate_excerpt_value(v)
        for k, v in payload.items()
        if k in _PAYLOAD_EXCERPT_KEYS and v is not None
    }


def _event_priority(kind: str, payload: dict[str, Any]) -> int:
    raw = payload.get("priority")
    if type(raw) is int and 1 <= raw <= 4:
        return raw
    return priority_for(kind, payload)


def _row_chars(event: dict[str, Any]) -> int:
    """Approximate char weight of an event for budget accounting."""
    payload = event.get("payload") or {}
    return len(str(event.get("kind", ""))) + sum(
        len(str(v)) for v in _excerpt(payload).values()
    ) + 24  # constant overhead per row (sequence, brackets, separators)


def _build_summary(
    run_id: str,
    raw_events: list[dict[str, Any]],
) -> str:
    """One-paragraph headline derived from the event list.

    Surfaces:
    * counts: node.succeeded / node.failed
    * the most recent failure's tool_name + error_code (if any)
    * the most recent succeeded node_id (if any)
    """
    succeeded = 0
    failed_events: list[dict[str, Any]] = []
    last_succeeded: dict[str, Any] | None = None
    for ev in raw_events:
        kind = ev.get("kind", "")
        if kind == "node.succeeded":
            succeeded += 1
            last_succeeded = ev
        elif kind in {"node.failed", "cc.tool_failed", "cc.turn_failed"}:
            failed_events.append(ev)

    parts: list[str] = []
    if not raw_events:
        return "(no prior activity)"

    if failed_events:
        parts.append(
            f"{len(failed_events)} failure(s); {succeeded} node(s) succeeded"
        )
        last_fail = failed_events[-1]
        fp = last_fail.get("payload") or {}
        tool = fp.get("tool_name") or fp.get("tool") or fp.get("node_id")
        ec = fp.get("error_code")
        message = fp.get("message") or fp.get("reason") or fp.get("preview") or ""
        suffix = f" - {ec}" if ec else (f" - {message[:120]}" if message else "")
        subject = f" on {tool}" if tool else ""
        parts.append(f"last failure: {last_fail.get('kind')}{subject}{suffix}")
    else:
        parts.append(f"{succeeded} node(s) succeeded, no failures")

    if last_succeeded is not None:
        sp = last_succeeded.get("payload") or {}
        node_id = sp.get("node_id") or "(unknown)"
        parts.append(f"last succeeded node: {node_id}")

    return f"Run {run_id}: " + "; ".join(parts) + "."


def build_snapshot(
    event_store: "EventStore",
    run_id: str,
    *,
    token_budget_chars: int = 12_000,
    max_events: int = 5_000,
    max_priority4_events: int = 200,
) -> ResumeSnapshot:
    """Read ``run_id``'s events and build a priority-filtered snapshot.

    ``token_budget_chars`` bounds the rendered prompt block size; rough
    correspondence is ~4 chars per token, so 12_000 chars ≈ 3000 tokens.

    ``max_events`` is the upper limit on the raw read — protects against
    a runaway run with millions of events. Default 5k is plenty for
    real ccx workloads; tune up if you ever need to snapshot a giant
    historical run.

    ``max_priority4_events`` is a hard cap for critical rows so a failure
    storm cannot create an unbounded snapshot row. The newest failures win.
    """
    if hasattr(event_store, "read_last"):
        raw = event_store.read_last(run_id, limit=max_events)
    else:
        raw = event_store.read_after(0, limit=max_events, run_id=run_id)
    highwater = raw[-1]["sequence"] if raw else 0
    summary = _build_summary(run_id, raw)

    # Classify and bucket by priority.
    annotated: list[tuple[int, dict[str, Any]]] = []
    for ev in raw:
        kind = str(ev.get("kind", ""))
        payload = ev.get("payload") or {}
        annotated.append((_event_priority(kind, payload), ev))

    selected: list[dict[str, Any]] = []
    budget = token_budget_chars

    # Priority 4: keep newest-first up to a hard cap and include those critical
    # rows even if they exceed the soft prompt budget.
    p4 = [ev for p, ev in annotated if p == 4]
    p4.sort(key=lambda e: e.get("sequence", 0), reverse=True)
    for ev in p4[:max_priority4_events]:
        cost = _row_chars(ev)
        selected.append(ev)
        budget -= cost

    # Priorities 3, 2, 1: take newest-first within each bucket.
    for pri in (3, 2, 1):
        bucket = [ev for p, ev in annotated if p == pri]
        bucket.sort(key=lambda e: e.get("sequence", 0), reverse=True)
        for ev in bucket:
            cost = _row_chars(ev)
            if cost > budget:
                break
            selected.append(ev)
            budget -= cost
        if budget <= 0:
            break

    # Sort final selection by sequence ascending so the prompt reads
    # chronologically (failures first only in the summary, not the list).
    selected.sort(key=lambda e: e.get("sequence", 0))

    refs = [
        EventRef(
            sequence=int(ev.get("sequence", 0)),
            kind=str(ev.get("kind", "")),
            priority=_event_priority(str(ev.get("kind", "")), ev.get("payload") or {}),
            payload_excerpt=_excerpt(ev.get("payload") or {}),
            occurred_at_ms=int(ev.get("created_at_ms", 0)),
        )
        for ev in selected
    ]

    return ResumeSnapshot(
        run_id=run_id,
        summary=summary,
        highwater_sequence=int(highwater),
        events=refs,
        built_at_ms=int(time.time() * 1000),
    )


__all__ = [
    "EventRef",
    "ResumeSnapshot",
    "build_snapshot",
]
