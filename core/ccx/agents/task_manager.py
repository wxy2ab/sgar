"""ccx TaskManager — v5-backed replacement for ``core.cc.agents.task_manager``.

.. warning:: EXPERIMENTAL — no production callers. Nothing outside the
   test suite constructs this TaskManager. Note also that the
   "multi-process safe via WAL" claim covers durability only:
   ``update_task_status`` is a read-validate-write guarded by an
   in-process lock, so two PROCESSES can still race the transition
   check. Reconcile before building on this layer.

Surface compatibility goals:
* same class name, same primary methods (``create_task``,
  ``update_task_status``, ``get``, ``all``)
* AgentTask / AgentTaskStatus types are re-exported from cc (so existing
  callers keep importing ``AgentTask`` from this module unchanged).

Behavioural improvements over cc's TaskManager:
* tasks live in v5 SQLite (durable, multi-process safe via WAL) instead
  of a single tasks.json + tasks_events.jsonl pair
* each task can be promoted to a v5 NodeSpec when the manager is asked
  to dispatch — so dependencies (``depends_on_task_ids``) become v5 DAG
  edges and parallel siblings run concurrently
* status transitions still validated via cc's existing
  ``can_transition_status``
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import asdict
from pathlib import Path


logger = logging.getLogger(__name__)
from typing import Any, Iterable

from core.cc.agents.task_model import (
    AgentTask,
    AgentTaskStatus,
    can_transition_status,
)
from core.cc.errors import AgentTaskError
from core.deepstack_v5 import (
    NodeSpec,
    NodeState,
    RunStatus,
    RuntimeV5,
)
from core.deepstack_v5.persistence import SQLiteRuntimeDB

# Map cc task status to v5 node state for round-trip persistence.
_TASK_TO_NODE_STATE: dict[AgentTaskStatus, NodeState] = {
    AgentTaskStatus.PENDING: NodeState.PENDING,
    AgentTaskStatus.RUNNING: NodeState.RUNNING,
    AgentTaskStatus.WAITING_MESSAGE: NodeState.APPROVAL_HANG,
    AgentTaskStatus.COMPLETED: NodeState.SUCCEEDED,
    AgentTaskStatus.FAILED: NodeState.ABANDONED,
    AgentTaskStatus.KILLED: NodeState.CANCELLED,
}

_NODE_TO_TASK_STATE: dict[NodeState, AgentTaskStatus] = {
    NodeState.PENDING: AgentTaskStatus.PENDING,
    NodeState.READY: AgentTaskStatus.PENDING,
    NodeState.RUNNING: AgentTaskStatus.RUNNING,
    NodeState.APPROVAL_HANG: AgentTaskStatus.WAITING_MESSAGE,
    NodeState.TIMER_HANG: AgentTaskStatus.WAITING_MESSAGE,
    NodeState.SUCCEEDED: AgentTaskStatus.COMPLETED,
    NodeState.ABANDONED: AgentTaskStatus.FAILED,
    NodeState.SKIPPED: AgentTaskStatus.FAILED,
    NodeState.CANCELLED: AgentTaskStatus.KILLED,
    NodeState.FAILED: AgentTaskStatus.RUNNING,  # transient; in v5 retries
    NodeState.BLOCKED: AgentTaskStatus.PENDING,
}


# --------------------------------------------------------------------------- #
# TaskManager
# --------------------------------------------------------------------------- #

class TaskManager:
    """v5-backed task manager.

    Simpler API than cc's: no separate tasks.json or tasks_events.jsonl —
    everything lives in the v5 SQLite DB at ``runtime_root/runtime.db``.
    All tasks belong to a single virtual run id (``cc-tasks``) so they
    can be queried in one ``GraphStore.list_nodes`` call.
    """

    _RUN_ID = "cc-tasks"

    def __init__(self, runtime_root: str | Path) -> None:
        self.runtime_root = Path(runtime_root)
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self._db = SQLiteRuntimeDB(self.runtime_root / "runtime.db")
        # Reuse v5 stores; we don't need the full RuntimeV5 wiring here.
        from core.deepstack_v5.persistence import GraphStore, RunStore
        self._run_store = RunStore(self._db)
        self._graph_store = GraphStore(self._db)
        self._lock = threading.Lock()

        # Ensure the synthetic run exists so foreign keys / list_nodes
        # behave consistently.
        if self._run_store.get(self._RUN_ID) is None:
            try:
                self._run_store.create(
                    self._RUN_ID,
                    goal="cc-task-tracker",
                    status=RunStatus.COMPLETED,
                )
            except sqlite3.IntegrityError:
                # Another process/thread won the first-open race.
                pass
        run = self._run_store.get(self._RUN_ID)
        if (
            run is not None
            and run.get("goal") == "cc-task-tracker"
            and run.get("status") == RunStatus.RUNNING.value
        ):
            self._run_store.update_status(self._RUN_ID, RunStatus.COMPLETED)

    # -- API: same shape as cc.TaskManager -----------------------------------

    def create_task(self, task: AgentTask) -> AgentTask:
        with self._lock:
            self._upsert(task)
        return task

    def update_task_status(
        self,
        task_id: str,
        status: AgentTaskStatus,
        *,
        result_payload: dict | None = None,
    ) -> None:
        with self._lock:
            task = self._read(task_id)
            if task is None:
                raise AgentTaskError(f"Unknown task_id: {task_id}")
            if not can_transition_status(task.status, status):
                raise AgentTaskError(
                    f"Invalid task status transition: "
                    f"{task.status.value} -> {status.value}",
                    error_code="AG1004",
                )
            task.status = status
            if result_payload is not None:
                task.result_payload = result_payload
            task.updated_at = time.time()
            self._upsert(task)

    def get(self, task_id: str) -> AgentTask | None:
        with self._lock:
            return self._read(task_id)

    def get_by_runtime_id(self, runtime_id: str) -> AgentTask | None:
        with self._lock:
            for row in self._graph_store.list_nodes(self._RUN_ID):
                task = _row_to_task(row)
                if task.runtime_id == runtime_id:
                    return task
        return None

    def all(self) -> list[AgentTask]:
        with self._lock:
            rows = self._graph_store.list_nodes(self._RUN_ID)
        # Sort to mirror cc's behaviour (sorted by task_id).
        tasks = [_row_to_task(r) for r in rows]
        tasks.sort(key=lambda t: t.task_id)
        return tasks

    def load_tasks_from_disk(self) -> None:
        # No-op: all reads go to SQLite directly.
        pass

    def persist_tasks(self) -> None:
        # No-op: every mutation is already durable.
        pass

    def record_event(
        self, task_id: str, event_type: str, payload: dict | None = None,
    ) -> None:
        # Event goes to the v5 events table for unified streaming.
        from core.deepstack_v5.persistence import EventStore
        es = EventStore(self._db)
        es.append(self._RUN_ID, f"task.{event_type}",
                  {"task_id": task_id, **(dict(payload) if payload else {})})

    # -- v5-specific extensions ----------------------------------------------

    def to_node_specs(
        self, *, default_tool: str = "ccx.agent",
    ) -> list[NodeSpec]:
        """Convert every PENDING task into a v5 NodeSpec for dispatch.

        Tasks with ``input_payload['depends_on_task_ids']`` get their v5
        ``depends_on`` populated; siblings without deps run in parallel.
        """
        tasks = self.all()
        by_id = {task.task_id: task for task in tasks}
        pending = {
            task.task_id: task
            for task in tasks
            if task.status == AgentTaskStatus.PENDING
        }

        fail_reasons: dict[str, str] = {}
        blocked_unresolved: set[str] = set()
        deps_by_task: dict[str, tuple[str, ...]] = {}
        for task_id, task in pending.items():
            deps: list[str] = []
            seen: set[str] = set()
            for raw_dep in task.input_payload.get("depends_on_task_ids") or ():
                dep_id = str(raw_dep)
                if dep_id in seen:
                    continue
                seen.add(dep_id)
                dep_task = by_id.get(dep_id)
                if dep_task is None:
                    fail_reasons[task_id] = f"missing dependency: {dep_id}"
                    break
                if dep_task.status == AgentTaskStatus.COMPLETED:
                    continue
                if dep_task.status in {
                    AgentTaskStatus.FAILED,
                    AgentTaskStatus.KILLED,
                }:
                    fail_reasons[task_id] = f"failed dependency: {dep_id}"
                    break
                if dep_task.status == AgentTaskStatus.PENDING:
                    if dep_id in pending:
                        deps.append(dep_id)
                    continue
                blocked_unresolved.add(task_id)
                break
            if task_id not in fail_reasons and task_id not in blocked_unresolved:
                deps_by_task[task_id] = tuple(deps)

        changed = True
        while changed:
            changed = False
            for task_id, deps in list(deps_by_task.items()):
                failed_dep = next(
                    (dep for dep in deps if dep in fail_reasons),
                    None,
                )
                if failed_dep is not None:
                    fail_reasons[task_id] = (
                        f"failed dependency: {failed_dep}"
                    )
                    deps_by_task.pop(task_id, None)
                    changed = True
                    continue
                unresolved_dep = next(
                    (
                        dep for dep in deps
                        if dep in blocked_unresolved
                        or (dep in pending and dep not in deps_by_task)
                    ),
                    None,
                )
                if unresolved_dep is not None:
                    blocked_unresolved.add(task_id)
                    deps_by_task.pop(task_id, None)
                    changed = True

        # Kahn topo sort over the remaining pending subset. Any leftover
        # nodes are cyclic and cannot be handed to v5 as a valid DAG.
        ordered_ids: list[str] = []
        remaining = set(deps_by_task)
        while remaining:
            ready = sorted(
                task_id for task_id in remaining
                if all(dep not in remaining for dep in deps_by_task[task_id])
            )
            if not ready:
                for task_id in remaining:
                    fail_reasons[task_id] = "cyclic dependency"
                remaining.clear()
                break
            ordered_ids.extend(ready)
            remaining.difference_update(ready)

        if fail_reasons:
            now = time.time()
            with self._lock:
                for task_id, reason in fail_reasons.items():
                    task = pending.get(task_id)
                    if task is None:
                        continue
                    task.status = AgentTaskStatus.FAILED
                    task.result_payload = {
                        **dict(task.result_payload or {}),
                        "error_code": "CCX_TASK_DEPENDENCY_UNSATISFIED",
                        "message": reason,
                    }
                    task.updated_at = now
                    self._upsert(task)

        specs: list[NodeSpec] = []
        for task_id in ordered_ids:
            if task_id in fail_reasons:
                continue
            task = pending[task_id]
            deps = deps_by_task.get(task_id, ())
            specs.append(NodeSpec(
                node_id=task.task_id,
                tool=task.input_payload.get("tool", default_tool),
                params={
                    "goal": str(
                        task.input_payload.get("prompt") or task.title or ""
                    ),
                    "metadata": {
                        "task_id": task.task_id,
                        "runtime_id": task.runtime_id,
                        "agent_type": task.agent_type,
                    },
                },
                depends_on=deps,
                metadata={
                    "ccx_task_id": task.task_id,
                    "ccx_origin": "task_manager",
                },
            ))
        return specs

    def shutdown(self) -> None:
        try:
            self._db.close()
        except Exception:
            logger.warning(
                "TaskManager db.close() raised on shutdown; ignoring",
                exc_info=True,
            )

    # -- internals -----------------------------------------------------------

    def _read(self, task_id: str) -> AgentTask | None:
        row = self._graph_store.get_node(self._RUN_ID, task_id)
        if row is None:
            return None
        return _row_to_task(row)

    def _upsert(self, task: AgentTask) -> None:
        node_state = _TASK_TO_NODE_STATE.get(
            task.status, NodeState.PENDING,
        )
        self._graph_store.upsert_node(
            self._RUN_ID,
            task.task_id,
            state=node_state,
            spec={
                "node_id": task.task_id,
                "tool": "ccx.agent",
                "params": dict(task.input_payload or {}),
                "depends_on": list(task.input_payload.get("depends_on_task_ids") or []),
                "max_attempts": 3,
                "timeout_s": None,
                "requires_approval": False,
                "metadata": {
                    "ccx_task_id": task.task_id,
                    "ccx_runtime_id": task.runtime_id,
                    "ccx_agent_type": task.agent_type,
                    "ccx_backend": task.backend,
                    "ccx_title": task.title,
                    "ccx_prompt_language": task.prompt_language,
                    "ccx_created_at": task.created_at,
                    "ccx_updated_at": task.updated_at,
                    "ccx_status": task.status.value,
                },
            },
            attempts=[],
            result=task.result_payload or None,
        )


def _row_to_task(row: dict[str, Any]) -> AgentTask:
    spec = row.get("spec") or {}
    meta = spec.get("metadata") or {}
    cc_status = AgentTaskStatus(meta.get("ccx_status",
                                _NODE_TO_TASK_STATE.get(
                                    NodeState(row["state"]),
                                    AgentTaskStatus.PENDING,
                                ).value))
    return AgentTask(
        task_id=row["node_id"],
        runtime_id=meta.get("ccx_runtime_id", ""),
        agent_type=meta.get("ccx_agent_type", "worker"),
        backend=meta.get("ccx_backend", "in_process"),
        status=cc_status,
        prompt_language=meta.get("ccx_prompt_language", "en"),
        title=meta.get("ccx_title", ""),
        input_payload=dict(spec.get("params") or {}),
        result_payload=dict(row.get("result") or {}) if isinstance(row.get("result"), dict) else {},
        created_at=float(meta.get("ccx_created_at") or row.get("created_at_ms", 0) / 1000),
        updated_at=float(meta.get("ccx_updated_at") or row.get("updated_at_ms", 0) / 1000),
    )


__all__ = ["AgentTask", "AgentTaskStatus", "TaskManager"]
