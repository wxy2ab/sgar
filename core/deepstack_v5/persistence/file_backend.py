"""In-memory / file backend for tests and ephemeral runs.

Production deployments use `SQLiteRuntimeDB`. This module exists only to give
unit tests a zero-IO substitute that exposes the same store interfaces. It
does not implement the full event/outbox semantics — multi-process and crash
recovery require SQLite.

NOT for production use.
"""

from __future__ import annotations

import threading
from typing import Any, Iterable, Sequence

from ..types import (
    Lease,
    NodeState,
    RunStatus,
    TERMINAL_NODE_STATES,
    TERMINAL_RUN_STATUSES,
    now_ms,
)
from .stores import BudgetIncrementResult, _apply_budget_delta, _budget_is_warning


class InMemoryRunStore:
    def __init__(self) -> None:
        self._runs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create(
        self,
        run_id: str,
        goal: str,
        *,
        status: RunStatus | str = RunStatus.RUNNING,
        budget: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        ts = now_ms()
        with self._lock:
            self._runs[run_id] = {
                "run_id": run_id,
                "goal": goal,
                "status": status.value if isinstance(status, RunStatus) else status,
                "created_at_ms": ts,
                "updated_at_ms": ts,
                "budget": budget,
                "config": config,
                "metadata": metadata,
            }

    def update_status(
        self,
        run_id: str,
        status: RunStatus | str,
        *,
        budget: dict[str, Any] | None = None,
        expected_status: RunStatus | str | Sequence[RunStatus | str] | None = None,
        refuse_if_terminal: bool = False,
    ) -> bool:
        with self._lock:
            run = self._runs.get(run_id)
            if not run:
                return False
            status_val = status.value if isinstance(status, RunStatus) else str(status)
            expected_values = _coerce_status_values(expected_status)
            if expected_values is not None and run["status"] not in expected_values:
                return False
            terminal_values = {s.value for s in TERMINAL_RUN_STATUSES}
            if (
                refuse_if_terminal
                and run["status"] in terminal_values
                and run["status"] != status_val
            ):
                return False
            run["status"] = status_val
            run["updated_at_ms"] = now_ms()
            if budget is not None:
                run["budget"] = budget
            return True

    def update_metadata(self, run_id: str, metadata: dict[str, Any]) -> None:
        with self._lock:
            run = self._runs.get(run_id)
            if not run:
                return
            run["metadata"] = dict(metadata)
            run["updated_at_ms"] = now_ms()

    def increment_budget_usage(
        self,
        run_id: str,
        *,
        tokens: int = 0,
        cost: float = 0.0,
        elapsed_s: float | None = None,
    ) -> BudgetIncrementResult | None:
        with self._lock:
            run = self._runs.get(run_id)
            if not run:
                return None
            budget = dict(run.get("budget") or {})
            was_warning = _budget_is_warning(budget)
            _apply_budget_delta(
                budget,
                tokens=int(tokens or 0),
                cost=float(cost or 0.0),
                elapsed_s=elapsed_s,
            )
            warning_crossed = _budget_is_warning(budget) and not was_warning
            run["budget"] = budget
            run["updated_at_ms"] = now_ms()
            return BudgetIncrementResult(
                budget=dict(budget),
                warning_crossed=warning_crossed,
            )

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            run = self._runs.get(run_id)
            return dict(run) if run else None

    def list_active(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                dict(r) for r in self._runs.values()
                if r["status"] == RunStatus.RUNNING.value
            ]


def _coerce_status_values(
    value: RunStatus | str | Sequence[RunStatus | str] | None,
) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, (RunStatus, str)):
        values: Sequence[RunStatus | str] = (value,)
    else:
        values = value
    return {
        item.value if isinstance(item, RunStatus) else str(item)
        for item in values
    }


class InMemoryGraphStore:
    def __init__(self) -> None:
        self._nodes: dict[tuple[str, str], dict[str, Any]] = {}
        self._edges: set[tuple[str, str, str]] = set()
        self._leases: dict[str, Lease] = {}
        self._lock = threading.Lock()

    def upsert_node(
        self,
        run_id: str,
        node_id: str,
        *,
        state: NodeState | str,
        spec: dict[str, Any],
        attempts: list[dict[str, Any]],
        result: Any | None = None,
        failure: dict[str, Any] | None = None,
        history: list | None = None,
        lease_id: str | None = None,
        require_active_lease: bool = False,
        expected_state: NodeState | str | None = None,
        refuse_if_terminal: bool = False,
        refuse_if_running_unowned: bool = False,
    ) -> bool:
        ts = now_ms()
        state_val = state.value if isinstance(state, NodeState) else state
        expected_state_val = (
            expected_state.value
            if isinstance(expected_state, NodeState)
            else (str(expected_state) if expected_state is not None else None)
        )
        terminal_values = {s.value for s in TERMINAL_NODE_STATES}
        with self._lock:
            if require_active_lease:
                if lease_id is None:
                    return False
                lease = self._leases.get(lease_id)
                if (
                    lease is None
                    or lease.run_id != run_id
                    or lease.node_id != node_id
                    or lease.expires_at_ms <= ts
                ):
                    return False
            existing = self._nodes.get((run_id, node_id))
            if expected_state_val is not None:
                if existing is None or existing["state"] != expected_state_val:
                    return False
            terminal_guard = refuse_if_terminal or require_active_lease
            if (
                terminal_guard
                and existing is not None
                and existing["state"] in terminal_values
                and existing["state"] != state_val
            ):
                return False
            if (
                refuse_if_running_unowned
                and not require_active_lease
                and existing is not None
                and existing["state"] == NodeState.RUNNING.value
            ):
                active = next(
                    (
                        lease for lease in self._leases.values()
                        if lease.run_id == run_id
                        and lease.node_id == node_id
                        and lease.expires_at_ms > ts
                    ),
                    None,
                )
                if active is not None and active.lease_id != lease_id:
                    return False
            created_at = existing["created_at_ms"] if existing else ts
            self._nodes[(run_id, node_id)] = {
                "run_id": run_id,
                "node_id": node_id,
                "state": state_val,
                "spec": spec,
                "attempts": list(attempts),
                "result": result,
                "failure": failure,
                "history": list(history or []),
                "created_at_ms": created_at,
                "updated_at_ms": ts,
            }
        return True

    def upsert_node_with_edges(
        self,
        run_id: str,
        node_id: str,
        *,
        state: NodeState | str,
        spec: dict[str, Any],
        attempts: list[dict[str, Any]],
        edges: Iterable[tuple[str, str]] = (),
        result: Any | None = None,
        failure: dict[str, Any] | None = None,
        history: list | None = None,
        lease_id: str | None = None,
        require_active_lease: bool = False,
        expected_state: NodeState | str | None = None,
        refuse_if_terminal: bool = False,
        refuse_if_running_unowned: bool = False,
    ) -> bool:
        ok = self.upsert_node(
            run_id,
            node_id,
            state=state,
            spec=spec,
            attempts=attempts,
            result=result,
            failure=failure,
            history=history,
            lease_id=lease_id,
            require_active_lease=require_active_lease,
            expected_state=expected_state,
            refuse_if_terminal=refuse_if_terminal,
            refuse_if_running_unowned=refuse_if_running_unowned,
        )
        if ok:
            self.add_edges(run_id, edges)
        return ok

    def get_node(self, run_id: str, node_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._nodes.get((run_id, node_id))
            return dict(row) if row else None

    def list_nodes(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return [
                dict(r) for (rid, _), r in self._nodes.items() if rid == run_id
            ]

    def list_nodes_by_state(
        self, run_id: str, state: NodeState | str
    ) -> list[dict[str, Any]]:
        state_val = state.value if isinstance(state, NodeState) else state
        with self._lock:
            return [
                dict(r) for (rid, _), r in self._nodes.items()
                if rid == run_id and r["state"] == state_val
            ]

    def count_by_state(self, run_id: str) -> dict[str, int]:
        out: dict[str, int] = {}
        with self._lock:
            for (rid, _), r in self._nodes.items():
                if rid != run_id:
                    continue
                out[r["state"]] = out.get(r["state"], 0) + 1
        return out

    def add_edges(self, run_id: str, edges: Iterable[tuple[str, str]]) -> None:
        with self._lock:
            for src, dst in edges:
                self._edges.add((run_id, src, dst))

    def list_edges(self, run_id: str) -> list[tuple[str, str]]:
        with self._lock:
            return [(s, d) for (rid, s, d) in self._edges if rid == run_id]

    def list_attempts(
        self, run_id: str, node_id: str | None = None
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        with self._lock:
            for (rid, nid), node in self._nodes.items():
                if rid != run_id:
                    continue
                if node_id is not None and nid != node_id:
                    continue
                for a in node.get("attempts", []):
                    out.append({
                        "run_id": rid,
                        "node_id": nid,
                        "attempt_id": a.get("attempt_id"),
                        "worker_id": a.get("worker_id"),
                        "started_at_ms": a.get("started_at_ms"),
                        "ended_at_ms": a.get("ended_at_ms"),
                        "outcome": a.get("outcome"),
                    })
        return out

    def grant_lease(self, lease: Lease) -> None:
        with self._lock:
            for lease_id, existing in list(self._leases.items()):
                if (
                    existing.run_id == lease.run_id
                    and existing.node_id == lease.node_id
                    and existing.expires_at_ms <= now_ms()
                ):
                    self._leases.pop(lease_id, None)
            for existing in self._leases.values():
                if existing.run_id == lease.run_id and existing.node_id == lease.node_id:
                    raise ValueError(
                        f"Node {lease.node_id} already leased: {existing.lease_id}"
                    )
            self._leases[lease.lease_id] = lease

    def heartbeat_lease(self, lease_id: str, expires_at_ms: int) -> bool:
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None:
                return False
            if lease.expires_at_ms <= now_ms():
                return False
            new_lease = Lease(
                lease_id=lease.lease_id,
                run_id=lease.run_id,
                node_id=lease.node_id,
                worker_id=lease.worker_id,
                granted_at_ms=lease.granted_at_ms,
                expires_at_ms=expires_at_ms,
                heartbeat_at_ms=now_ms(),
            )
            self._leases[lease_id] = new_lease
            return True

    def release_lease(self, lease_id: str) -> bool:
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None or lease.expires_at_ms <= now_ms():
                return False
            return self._leases.pop(lease_id, None) is not None

    def find_expired(self, *, now: int) -> list[Lease]:
        with self._lock:
            return [l for l in self._leases.values() if l.expires_at_ms <= now]

    def reclaim_expired(
        self, *, now: int, run_id: str | None = None
    ) -> list[Lease]:
        with self._lock:
            expired = [
                l for l in self._leases.values()
                if l.expires_at_ms <= now and (run_id is None or l.run_id == run_id)
            ]
            for lease in expired:
                self._leases.pop(lease.lease_id, None)
            return expired

    def find_lease_for(self, run_id: str, node_id: str) -> Lease | None:
        with self._lock:
            for lease in self._leases.values():
                if lease.run_id == run_id and lease.node_id == node_id:
                    return lease
            return None

    def count_leases(self, run_id: str) -> int:
        with self._lock:
            return sum(1 for lease in self._leases.values() if lease.run_id == run_id)


class InMemoryEventStore:
    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []
        self._seq = 0
        self._lock = threading.Lock()

    def append(self, run_id: str, kind: str, payload: dict[str, Any]) -> int:
        with self._lock:
            self._seq += 1
            self._events.append({
                "sequence": self._seq,
                "run_id": run_id,
                "kind": kind,
                "created_at_ms": now_ms(),
                "payload": dict(payload),
            })
            return self._seq

    def read_after(
        self, sequence: int = 0, *, limit: int = 1000, run_id: str | None = None
    ) -> list[dict[str, Any]]:
        with self._lock:
            out: list[dict[str, Any]] = []
            for ev in self._events:
                if ev["sequence"] <= sequence:
                    continue
                if run_id is not None and ev["run_id"] != run_id:
                    continue
                out.append(dict(ev))
                if len(out) >= limit:
                    break
            return out

    def read_sequences(
        self, sequences: Sequence[int], *, run_id: str | None = None
    ) -> list[dict[str, Any]]:
        wanted = {int(s) for s in sequences}
        with self._lock:
            return [
                dict(ev)
                for ev in self._events
                if ev["sequence"] in wanted
                and (run_id is None or ev["run_id"] == run_id)
            ]

    def read_last(self, run_id: str, *, limit: int = 1000) -> list[dict[str, Any]]:
        with self._lock:
            rows = [
                dict(ev) for ev in self._events
                if ev["run_id"] == run_id
            ][-limit:]
            return rows

    def max_sequence(self, run_id: str | None = None) -> int:
        with self._lock:
            if run_id is None:
                return self._seq
            return max(
                (e["sequence"] for e in self._events if e["run_id"] == run_id),
                default=0,
            )


class InMemoryOutbox:
    def __init__(self) -> None:
        self._undelivered: set[int] = set()
        self._run_ids: dict[int, str | None] = {}
        self._lock = threading.Lock()

    def stage(self, sequence: int, *, run_id: str | None = None) -> None:
        with self._lock:
            self._undelivered.add(sequence)
            self._run_ids[sequence] = run_id

    def claim_pending(
        self,
        *,
        limit: int = 100,
        run_id: str | None = None,
    ) -> list[int]:
        with self._lock:
            pending = sorted(self._undelivered)
            if run_id is not None:
                pending = [
                    sequence for sequence in pending
                    if self._run_ids.get(sequence) == run_id
                ]
            return pending[:limit]

    def mark_delivered(self, sequences: Sequence[int]) -> None:
        with self._lock:
            for s in sequences:
                self._undelivered.discard(s)

    def reset_pending(self) -> int:
        return 0  # no notion of "delivered" outside undelivered set

    def pending_count(self) -> int:
        with self._lock:
            return len(self._undelivered)


__all__ = [
    "InMemoryEventStore",
    "InMemoryGraphStore",
    "InMemoryOutbox",
    "InMemoryRunStore",
]
