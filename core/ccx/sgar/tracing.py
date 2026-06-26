"""Structured trace records for SGAR operations."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator
from uuid import uuid4

from .models import ProjectState
from .store import SgarStore, utc_now


TRACE_FILENAME = "trace.jsonl"


class SgarTracer:
    """Append-only JSONL tracer for SGAR workspace operations."""

    def __init__(self, store: SgarStore) -> None:
        self.store = store

    @property
    def trace_path(self) -> Path:
        return self.store.root / TRACE_FILENAME

    @contextmanager
    def operation(
        self,
        operation: str,
        *,
        inputs: dict[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        event_id = uuid4().hex
        started = perf_counter()
        before = self._state_snapshot()
        context: dict[str, Any] = {"artifacts": []}
        self.record(
            event_id=event_id,
            operation=operation,
            status="started",
            inputs=inputs or {},
            state_before=before,
        )
        try:
            yield context
        except Exception as exc:
            self.record(
                event_id=event_id,
                operation=operation,
                status="failed",
                inputs=inputs or {},
                state_before=before,
                state_after=self._state_snapshot(),
                artifacts=context.get("artifacts") or [],
                duration_ms=_elapsed_ms(started),
                error={"type": type(exc).__name__, "message": str(exc)},
            )
            raise
        else:
            self.record(
                event_id=event_id,
                operation=operation,
                status="completed",
                inputs=inputs or {},
                state_before=before,
                state_after=self._state_snapshot(),
                artifacts=context.get("artifacts") or [],
                duration_ms=_elapsed_ms(started),
            )

    def artifact(self, path: str | Path, *, role: str = "artifact") -> dict[str, Any]:
        path_obj = Path(path)
        try:
            resolved = path_obj.resolve()
        except OSError:
            resolved = path_obj
        rel_path = _relative_to(resolved, self.store.cwd)
        record: dict[str, Any] = {
            "role": role,
            "path": rel_path,
            "exists": resolved.exists(),
        }
        if resolved.is_file():
            record["kind"] = "file"
            try:
                record["size_bytes"] = resolved.stat().st_size
                record["sha256"] = _sha256_file(resolved)
            except OSError:
                # The file vanished between is_file() and the stat/hash — a
                # concurrent archive/abandon can delete a verification.json
                # mid-trace. Size/hash is best-effort metadata; degrade to an
                # honest 'missing' record rather than crash the governed op
                # with an uncaught FileNotFoundError. Byte-equivalent on the
                # single-writer path (the file is always present there).
                record["exists"] = False
                record["kind"] = "missing"
                record.pop("size_bytes", None)
                record.pop("sha256", None)
        elif resolved.is_dir():
            record["kind"] = "dir"
        else:
            record["kind"] = "missing"
        return record

    def record(
        self,
        *,
        event_id: str,
        operation: str,
        status: str,
        inputs: dict[str, Any],
        state_before: dict[str, Any] | None = None,
        state_after: dict[str, Any] | None = None,
        artifacts: list[dict[str, Any]] | None = None,
        duration_ms: int | None = None,
        error: dict[str, str] | None = None,
    ) -> None:
        if not self.store.root.exists() and operation != "init":
            return
        self.store.root.mkdir(parents=True, exist_ok=True)
        event: dict[str, Any] = {
            "timestamp": utc_now(),
            "event_id": event_id,
            "operation": operation,
            "status": status,
            "inputs": inputs,
        }
        if state_before is not None:
            event["state_before"] = state_before
        if state_after is not None:
            event["state_after"] = state_after
        if artifacts:
            event["artifacts"] = artifacts
        if duration_ms is not None:
            event["duration_ms"] = duration_ms
        if error is not None:
            event["error"] = error
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")

    def _state_snapshot(self) -> dict[str, Any] | None:
        if not self.store.state_path.exists():
            return None
        try:
            return _state_summary(self.store.load_state())
        except Exception:  # noqa: BLE001 - trace must not hide primary errors.
            return {"unreadable": True}


def read_trace(store: SgarStore) -> list[dict[str, Any]]:
    path = store.root / TRACE_FILENAME
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def read_failed_trace(store: SgarStore) -> list[dict[str, Any]]:
    """Return only the ``status == "failed"`` trace records.

    Every ``SgarTracer.operation`` that raises (including a governance
    ``SgarError``) writes a ``failed`` record before re-raising, so this is
    the durable, out-of-band signal a CLI / outside-driven driver can scan
    to see governance refusals — the on-disk complement to the in-process
    ``governance_errors`` that ``ccx.CodeAgent`` surfaces on its result
    snapshot. Each record keeps its ``operation``, ``error`` (``type`` /
    ``message``), ``inputs`` and timing, so the caller can report *what* was
    refused and *why* without re-parsing free text.
    """
    return [r for r in read_trace(store) if r.get("status") == "failed"]


def _state_summary(state: ProjectState) -> dict[str, Any]:
    return {
        "project_name": state.project_name,
        "mode": state.mode,
        "current_stage_id": state.current_stage_id,
        "next_stage_id": state.next_stage_id,
        "last_closed_stage_id": state.last_closed_stage_id,
        "closed_stage_ids": list(state.closed_stage_ids),
        "roadmap_review_required": state.roadmap_review_required,
        "future_stage_validation_required": state.future_stage_validation_required,
    }


def _elapsed_ms(started: float) -> int:
    return max(0, int((perf_counter() - started) * 1000))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_to(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


__all__ = ["TRACE_FILENAME", "SgarTracer", "read_failed_trace", "read_trace"]
