from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any

from .observability import EventRecord

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RuntimeAuditSnapshot:
    audit_root: Path
    session_events: list[EventRecord] = field(default_factory=list)
    task_events: list[EventRecord] = field(default_factory=list)
    tool_events: list[EventRecord] = field(default_factory=list)

    @property
    def all_events(self) -> list[EventRecord]:
        return [*self.session_events, *self.task_events, *self.tool_events]


@dataclass(slots=True)
class RuntimeAuditSummary:
    audit_root: Path
    session_event_count: int
    task_event_count: int
    tool_event_count: int
    total_event_count: int
    failed_event_count: int
    event_type_counts: dict[str, int]
    error_code_counts: dict[str, int]
    latest_failures: list[dict[str, Any]] = field(default_factory=list)
    memory_event_count: int = 0
    memory_event_type_counts: dict[str, int] = field(default_factory=dict)
    memory_provider_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_root": str(self.audit_root),
            "session_event_count": self.session_event_count,
            "task_event_count": self.task_event_count,
            "tool_event_count": self.tool_event_count,
            "total_event_count": self.total_event_count,
            "failed_event_count": self.failed_event_count,
            "event_type_counts": dict(self.event_type_counts),
            "error_code_counts": dict(self.error_code_counts),
            "latest_failures": list(self.latest_failures),
            "memory_event_count": self.memory_event_count,
            "memory_event_type_counts": dict(self.memory_event_type_counts),
            "memory_provider_counts": dict(self.memory_provider_counts),
        }


@dataclass(slots=True)
class RuntimeAuditQuery:
    session_id: str | None = None
    turn_id: str | None = None
    task_id: str | None = None
    tool_name: str | None = None
    event_types: list[str] = field(default_factory=list)
    error_code: str | None = None
    limit: int | None = None


def read_runtime_audit(runtime_root: str | Path) -> RuntimeAuditSnapshot:
    audit_root = _resolve_audit_root(runtime_root)
    return RuntimeAuditSnapshot(
        audit_root=audit_root,
        session_events=_read_event_records(audit_root / "session_events.jsonl"),
        task_events=_read_event_records(audit_root / "task_events.jsonl"),
        tool_events=_read_event_records(audit_root / "tool_events.jsonl"),
    )


def query_runtime_audit(
    runtime_root: str | Path,
    *,
    session_id: str | None = None,
    turn_id: str | None = None,
    task_id: str | None = None,
    tool_name: str | None = None,
    event_types: list[str] | None = None,
    error_code: str | None = None,
    limit: int | None = None,
) -> RuntimeAuditSnapshot:
    snapshot = read_runtime_audit(runtime_root)
    query = RuntimeAuditQuery(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        tool_name=tool_name,
        event_types=list(event_types or []),
        error_code=error_code,
        limit=limit,
    )
    filtered = RuntimeAuditSnapshot(
        audit_root=snapshot.audit_root,
        session_events=_filter_events(snapshot.session_events, query),
        task_events=_filter_events(snapshot.task_events, query),
        tool_events=_filter_events(snapshot.tool_events, query),
    )
    return _apply_limit(filtered, query.limit)


def summarize_runtime_audit(
    runtime_root: str | Path,
    *,
    session_id: str | None = None,
    turn_id: str | None = None,
    task_id: str | None = None,
    tool_name: str | None = None,
    event_types: list[str] | None = None,
    error_code: str | None = None,
    limit: int | None = None,
) -> RuntimeAuditSummary:
    snapshot = query_runtime_audit(
        runtime_root,
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        tool_name=tool_name,
        event_types=event_types,
        error_code=error_code,
        limit=limit,
    )
    all_events = sorted(snapshot.all_events, key=lambda item: item.timestamp)
    event_type_counts = Counter(event.event_type for event in all_events)
    error_code_counts = Counter(event.error_code for event in all_events if event.error_code)
    memory_events = [
        event
        for event in all_events
        if event.event_type.startswith("memory_") or event.event_type == "memory_provider_unavailable"
    ]
    memory_event_type_counts = Counter(event.event_type for event in memory_events)
    memory_provider_counts = Counter(
        str(event.details.get("provider"))
        for event in memory_events
        if event.details.get("provider")
    )
    failure_events = [
        event
        for event in all_events
        if event.success is False or event.error_code is not None or event.event_type.endswith("_failed")
    ]
    latest_failures = [
        {
            "timestamp": event.timestamp,
            "event_type": event.event_type,
            "session_id": event.session_id,
            "turn_id": event.turn_id,
            "task_id": event.task_id,
            "tool_name": event.tool_name,
            "error_code": event.error_code,
            "details": dict(event.details),
        }
        for event in failure_events[-10:]
    ]
    return RuntimeAuditSummary(
        audit_root=snapshot.audit_root,
        session_event_count=len(snapshot.session_events),
        task_event_count=len(snapshot.task_events),
        tool_event_count=len(snapshot.tool_events),
        total_event_count=len(all_events),
        failed_event_count=len(failure_events),
        event_type_counts=dict(event_type_counts),
        error_code_counts=dict(error_code_counts),
        latest_failures=latest_failures,
        memory_event_count=len(memory_events),
        memory_event_type_counts=dict(memory_event_type_counts),
        memory_provider_counts=dict(memory_provider_counts),
    )


def _resolve_audit_root(runtime_root: str | Path) -> Path:
    root = Path(runtime_root)
    return root if root.name == "audit" else root / "audit"


def _filter_events(events: list[EventRecord], query: RuntimeAuditQuery) -> list[EventRecord]:
    return [event for event in events if _matches_query(event, query)]


def _apply_limit(snapshot: RuntimeAuditSnapshot, limit: int | None) -> RuntimeAuditSnapshot:
    if limit is None or limit < 0:
        return snapshot
    if limit == 0:
        return RuntimeAuditSnapshot(audit_root=snapshot.audit_root)
    selected_ids = {
        id(event)
        for event in sorted(snapshot.all_events, key=lambda item: item.timestamp)[-limit:]
    }
    return RuntimeAuditSnapshot(
        audit_root=snapshot.audit_root,
        session_events=[event for event in snapshot.session_events if id(event) in selected_ids],
        task_events=[event for event in snapshot.task_events if id(event) in selected_ids],
        tool_events=[event for event in snapshot.tool_events if id(event) in selected_ids],
    )


def _matches_query(event: EventRecord, query: RuntimeAuditQuery) -> bool:
    if query.session_id is not None and event.session_id != query.session_id:
        return False
    if query.turn_id is not None and event.turn_id != query.turn_id:
        return False
    if query.task_id is not None and event.task_id != query.task_id:
        return False
    if query.tool_name is not None and event.tool_name != query.tool_name:
        return False
    if query.event_types and event.event_type not in query.event_types:
        return False
    if query.error_code is not None and event.error_code != query.error_code:
        return False
    return True


def _read_event_records(path: Path) -> list[EventRecord]:
    if not path.exists():
        return []
    events: list[EventRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSON line in %s: %s", path, exc)
                continue
            if not isinstance(payload, dict):
                continue
            events.append(
                EventRecord(
                    event_type=str(payload.get("event_type") or ""),
                    timestamp=float(payload.get("timestamp") or 0.0),
                    session_id=_optional_str(payload.get("session_id")),
                    turn_id=_optional_str(payload.get("turn_id")),
                    task_id=_optional_str(payload.get("task_id")),
                    tool_name=_optional_str(payload.get("tool_name")),
                    success=payload.get("success"),
                    error_code=_optional_str(payload.get("error_code")),
                    details=dict(payload.get("details") or {}),
                )
            )
    return events


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
