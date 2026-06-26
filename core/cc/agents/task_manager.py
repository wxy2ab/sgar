from __future__ import annotations

import json
from pathlib import Path
import time

from ..errors import AgentTaskError
from ..observability import EventRecord, JsonlAuditLogger
from .task_state_store import TaskStateStore
from .task_model import AgentTask, AgentTaskStatus, can_transition_status


class TaskManager:
    def __init__(self, runtime_root: str | Path) -> None:
        self.runtime_root = Path(runtime_root)
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self.tasks_path = self.runtime_root / "tasks.json"
        self.events_path = self.runtime_root / "task_events.jsonl"
        self.audit_logger = JsonlAuditLogger(self.runtime_root.parent / "audit" / "task_events.jsonl")
        self._store = TaskStateStore(self.runtime_root)
        self._tasks = self._store.tasks

    def create_task(self, task: AgentTask) -> AgentTask:
        self._tasks[task.task_id] = task
        self._store.create_task(task)
        self.record_event(task.task_id, "task_created", {"status": task.status.value, "backend": task.backend})
        return task

    def update_task_status(
        self,
        task_id: str,
        status: AgentTaskStatus,
        *,
        result_payload: dict | None = None,
    ) -> None:
        task = self.get(task_id)
        if task is None:
            self.load_tasks_from_disk()
            task = self.get(task_id)
        if task is None:
            raise AgentTaskError(f"Unknown task_id: {task_id}")
        previous_status = task.status.value
        if not can_transition_status(task.status, status):
            raise AgentTaskError(
                f"Invalid task status transition: {task.status.value} -> {status.value}",
                error_code="AG1004",
            )
        task.status = status
        if result_payload is not None:
            task.result_payload = result_payload
        task.updated_at = time.time()
        self._store.update_task_status(
            task.task_id,
            status=status,
            result_payload=result_payload,
            updated_at=task.updated_at,
        )
        self.record_event(
            task.task_id,
            "task_status_updated",
            {
                "from": previous_status,
                "to": status.value,
                "result_payload": result_payload or {},
            },
        )

    def get(self, task_id: str) -> AgentTask | None:
        return self._tasks.get(task_id)

    def get_by_runtime_id(self, runtime_id: str) -> AgentTask | None:
        for task in self._tasks.values():
            if task.runtime_id == runtime_id:
                return task
        return None

    def all(self) -> list[AgentTask]:
        return [self._tasks[key] for key in sorted(self._tasks)]

    def load_tasks_from_disk(self) -> None:
        try:
            self._store.load()
        except json.JSONDecodeError as exc:
            raise AgentTaskError(f"Invalid tasks.json: {self.tasks_path}") from exc
        self._tasks = self._store.tasks

    def persist_tasks(self) -> None:
        self._store.persist_snapshot()

    def record_event(self, task_id: str, event_type: str, payload: dict | None = None) -> None:
        self._store.append_event(task_id, event_type, payload)
        event_payload = dict(payload or {})
        self.audit_logger.append(
            EventRecord(
                event_type=event_type,
                task_id=task_id,
                success=False if event_type == "task_failed" else None,
                error_code=str(event_payload.get("error_code")) if event_payload.get("error_code") else None,
                details=event_payload,
            )
        )
