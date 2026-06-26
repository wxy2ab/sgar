from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from ..jsonl import JsonlTailReader, append_jsonl_sync
from .task_model import AgentTask, AgentTaskStatus


class TaskStateStore:
    def __init__(self, runtime_root: str | Path, *, snapshot_interval: int = 20) -> None:
        self.runtime_root = Path(runtime_root)
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self.snapshot_path = self.runtime_root / "tasks.json"
        self.events_path = self.runtime_root / "task_events.jsonl"
        self.snapshot_interval = max(1, snapshot_interval)
        self._tasks: dict[str, AgentTask] = {}
        self._reader = JsonlTailReader(self.events_path)
        self._dirty_event_count = 0
        self.load()

    @property
    def tasks(self) -> dict[str, AgentTask]:
        return self._tasks

    def load(self) -> dict[str, AgentTask]:
        self._tasks = self._load_snapshot()
        self._reader.reset()
        self._sync_events()
        return self._tasks

    def sync(self) -> dict[str, AgentTask]:
        self._sync_events()
        return self._tasks

    def create_task(self, task: AgentTask) -> None:
        payload = {
            "kind": "task_created",
            "task": task.to_dict(),
            "created_at": time.time(),
        }
        self._tasks[task.task_id] = task
        append_jsonl_sync(self.events_path, payload)
        self._dirty_event_count += 1
        self._maybe_persist_snapshot()

    def update_task_status(
        self,
        task_id: str,
        *,
        status: AgentTaskStatus,
        result_payload: dict[str, Any] | None = None,
        updated_at: float | None = None,
    ) -> None:
        task = self._tasks.get(task_id)
        timestamp = updated_at or time.time()
        if task is not None:
            task.status = status
            if result_payload is not None:
                task.result_payload = result_payload
            task.updated_at = timestamp
        append_jsonl_sync(
            self.events_path,
            {
                "kind": "task_status_updated",
                "task_id": task_id,
                "status": status.value,
                "result_payload": result_payload,
                "updated_at": timestamp,
                "created_at": time.time(),
            },
        )
        self._dirty_event_count += 1
        self._maybe_persist_snapshot()

    def append_event(self, task_id: str, event_type: str, payload: dict[str, Any] | None = None) -> None:
        append_jsonl_sync(
            self.events_path,
            {
                "kind": "event",
                "task_id": task_id,
                "event_type": event_type,
                "payload": payload or {},
                "created_at": time.time(),
            },
        )

    def persist_snapshot(self) -> None:
        self.snapshot_path.write_text(
            json.dumps([task.to_dict() for task in self.all()], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._dirty_event_count = 0

    def all(self) -> list[AgentTask]:
        return [self._tasks[key] for key in sorted(self._tasks)]

    def _load_snapshot(self) -> dict[str, AgentTask]:
        payload = self._load_snapshot_payload()
        tasks: dict[str, AgentTask] = {}
        for item in payload:
            if not isinstance(item, dict):
                continue
            item["status"] = AgentTaskStatus(item.get("status", AgentTaskStatus.PENDING.value))
            task = AgentTask(**item)
            tasks[task.task_id] = task
        return tasks

    def _load_snapshot_payload(self) -> list[dict[str, Any]]:
        legacy_path = self.runtime_root / "tasks.snapshot.json"
        candidate = self.snapshot_path if self.snapshot_path.exists() else legacy_path
        if not candidate.exists():
            return []
        raw_text = candidate.read_text(encoding="utf-8")
        if not raw_text.strip():
            return []
        payload = json.loads(raw_text)
        return payload if isinstance(payload, list) else []

    def _sync_events(self) -> None:
        for event in self._reader.read_new():
            self._apply_event(event)

    def _apply_event(self, event: dict[str, Any]) -> None:
        kind = str(event.get("kind") or "")
        if kind == "task_created":
            payload = event.get("task")
            if isinstance(payload, dict):
                payload = dict(payload)
                payload["status"] = AgentTaskStatus(payload.get("status", AgentTaskStatus.PENDING.value))
                task = AgentTask(**payload)
                self._tasks[task.task_id] = task
            return
        if kind == "task_status_updated":
            task_id = str(event.get("task_id") or "")
            task = self._tasks.get(task_id)
            if task is None:
                return
            task.status = AgentTaskStatus(str(event.get("status") or AgentTaskStatus.PENDING.value))
            if "result_payload" in event and event.get("result_payload") is not None:
                task.result_payload = dict(event.get("result_payload") or {})
            task.updated_at = float(event.get("updated_at") or time.time())

    def _maybe_persist_snapshot(self) -> None:
        if self._dirty_event_count >= self.snapshot_interval:
            self.persist_snapshot()

