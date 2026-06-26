"""Stores for DeepStack v5 persistence.

Four stores:
* `RunStore`    – `runs` table, one row per engine.run() invocation.
* `GraphStore`  – `nodes` + `edges` + `attempts_index` tables.
* `EventStore`  – append-only `events` table (DB-sequenced).
* `ClaimStore`  – `claims` table (knowledge layer, mirrored by ClaimStore in
  knowledge/claims.py for in-memory access).

Stores deliberately do not import in-memory dataclasses (NodeExecution etc.)
to avoid circular imports; they accept and return plain dicts. JSON
encoding/decoding happens here.
"""

from __future__ import annotations

import enum
import json
import logging
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from ..types import (
    Lease,
    NodeState,
    RunStatus,
    TERMINAL_NODE_STATES,
    TERMINAL_RUN_STATUSES,
    now_ms,
)
from .db import SQLiteRuntimeDB

logger = logging.getLogger(__name__)


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, enum.Enum):
        return value.value
    logger.warning(
        "DeepStack v5 persistence: coerced non-JSON value of type %s to repr",
        type(value).__name__,
    )
    return repr(value)


def _loads(value: str | None) -> Any:
    if value is None:
        return None
    return json.loads(value)


# --------------------------------------------------------------------------- #
# RunStore
# --------------------------------------------------------------------------- #

@dataclass(slots=True, frozen=True)
class BudgetIncrementResult:
    budget: dict[str, Any]
    warning_crossed: bool


class RunStore:
    def __init__(self, db: SQLiteRuntimeDB):
        self.db = db

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
        self.db.execute(
            """
            INSERT INTO runs (
                run_id, goal, status, created_at_ms, updated_at_ms,
                budget_json, config_json, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                goal,
                status.value if isinstance(status, RunStatus) else str(status),
                ts,
                ts,
                _dumps(budget) if budget is not None else None,
                _dumps(config) if config is not None else None,
                _dumps(metadata) if metadata is not None else None,
            ),
        )

    def update_status(
        self,
        run_id: str,
        status: RunStatus | str,
        *,
        budget: dict[str, Any] | None = None,
        expected_status: RunStatus | str | Sequence[RunStatus | str] | None = None,
        refuse_if_terminal: bool = False,
    ) -> bool:
        status_val = status.value if isinstance(status, RunStatus) else str(status)
        expected_values = _coerce_status_values(expected_status)
        terminal_values = {s.value for s in TERMINAL_RUN_STATUSES}
        with self.db.transaction():
            existing = self.db.query_one(
                "SELECT status FROM runs WHERE run_id = ?", (run_id,)
            )
            if existing is None:
                return False
            existing_status = existing["status"]
            if expected_values is not None and existing_status not in expected_values:
                return False
            if (
                refuse_if_terminal
                and existing_status in terminal_values
                and existing_status != status_val
            ):
                return False
            if budget is None:
                cur = self.db.execute(
                    "UPDATE runs SET status = ?, updated_at_ms = ? WHERE run_id = ?",
                    (
                        status_val,
                        now_ms(),
                        run_id,
                    ),
                )
            else:
                cur = self.db.execute(
                    """
                    UPDATE runs SET status = ?, updated_at_ms = ?, budget_json = ?
                    WHERE run_id = ?
                    """,
                    (
                        status_val,
                        now_ms(),
                        _dumps(budget),
                        run_id,
                    ),
                )
            return cur.rowcount > 0

    def update_metadata(self, run_id: str, metadata: dict[str, Any]) -> None:
        self.db.execute(
            """
            UPDATE runs SET metadata_json = ?, updated_at_ms = ?
            WHERE run_id = ?
            """,
            (_dumps(metadata), now_ms(), run_id),
        )

    def increment_budget_usage(
        self,
        run_id: str,
        *,
        tokens: int = 0,
        cost: float = 0.0,
        elapsed_s: float | None = None,
    ) -> BudgetIncrementResult | None:
        """Atomically add token/cost deltas to a run budget.

        WorkerHarness instances can run in separate processes, so callers
        must not read/modify/write budget snapshots outside the DB
        transaction. This method serializes the merge with BEGIN IMMEDIATE.
        """
        token_delta = int(tokens or 0)
        cost_delta = float(cost or 0.0)
        with self.db.transaction():
            row = self.db.query_one(
                "SELECT budget_json FROM runs WHERE run_id = ?", (run_id,)
            )
            if row is None:
                return None
            budget = dict(_loads(row["budget_json"]) or {})
            was_warning = _budget_is_warning(budget)
            _apply_budget_delta(
                budget,
                tokens=token_delta,
                cost=cost_delta,
                elapsed_s=elapsed_s,
            )
            warning_crossed = _budget_is_warning(budget) and not was_warning
            cur = self.db.execute(
                """
                UPDATE runs SET budget_json = ?, updated_at_ms = ?
                WHERE run_id = ?
                """,
                (_dumps(budget), now_ms(), run_id),
            )
            if cur.rowcount <= 0:
                return None
            return BudgetIncrementResult(
                budget=budget,
                warning_crossed=warning_crossed,
            )

    def get(self, run_id: str) -> dict[str, Any] | None:
        row = self.db.query_one(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        )
        return _row_to_run(row) if row else None

    def list_active(self) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM runs WHERE status = ? ORDER BY created_at_ms ASC",
            (RunStatus.RUNNING.value,),
        )
        return [_row_to_run(r) for r in rows]


def _row_to_run(row: Any) -> dict[str, Any]:
    return {
        "run_id": row["run_id"],
        "goal": row["goal"],
        "status": row["status"],
        "created_at_ms": row["created_at_ms"],
        "updated_at_ms": row["updated_at_ms"],
        "budget": _loads(row["budget_json"]),
        "config": _loads(row["config_json"]),
        "metadata": _loads(row["metadata_json"]),
    }


def _apply_budget_delta(
    budget: dict[str, Any],
    *,
    tokens: int,
    cost: float,
    elapsed_s: float | None,
) -> None:
    if tokens:
        budget["consumed_tokens"] = int(budget.get("consumed_tokens") or 0) + tokens
    else:
        budget["consumed_tokens"] = int(budget.get("consumed_tokens") or 0)
    if cost:
        budget["consumed_cost"] = float(budget.get("consumed_cost") or 0.0) + cost
    else:
        budget["consumed_cost"] = float(budget.get("consumed_cost") or 0.0)
    if elapsed_s is not None:
        current_elapsed = float(budget.get("elapsed_s") or 0.0)
        budget["elapsed_s"] = max(current_elapsed, float(elapsed_s))
    elif "elapsed_s" in budget:
        budget["elapsed_s"] = float(budget.get("elapsed_s") or 0.0)
    if "iterations" in budget:
        budget["iterations"] = int(budget.get("iterations") or 0)


def _budget_is_warning(budget: dict[str, Any]) -> bool:
    ratio = float(budget.get("warning_ratio") or 0.8)
    max_tokens = budget.get("max_tokens")
    if (
        max_tokens
        and int(budget.get("consumed_tokens") or 0) >= float(max_tokens) * ratio
    ):
        return True
    max_cost = budget.get("max_cost")
    if (
        max_cost
        and float(budget.get("consumed_cost") or 0.0) >= float(max_cost) * ratio
    ):
        return True
    max_wallclock_s = budget.get("max_wallclock_s")
    if (
        max_wallclock_s
        and float(budget.get("elapsed_s") or 0.0) >= float(max_wallclock_s) * ratio
    ):
        return True
    return False


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


# --------------------------------------------------------------------------- #
# GraphStore
# --------------------------------------------------------------------------- #

class GraphStore:
    """Persists nodes, edges, attempts (denormalised index for querying)."""

    def __init__(self, db: SQLiteRuntimeDB):
        self.db = db

    # -- nodes ---------------------------------------------------------------

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
        state_val = state.value if isinstance(state, NodeState) else str(state)
        expected_state_val = (
            expected_state.value
            if isinstance(expected_state, NodeState)
            else (str(expected_state) if expected_state is not None else None)
        )
        terminal_values = {s.value for s in TERMINAL_NODE_STATES}
        with self.db.transaction():
            if require_active_lease:
                if lease_id is None:
                    return False
                lease = self.db.query_one(
                    """
                    SELECT lease_id FROM leases
                    WHERE run_id = ? AND node_id = ? AND lease_id = ?
                      AND expires_at_ms > ?
                    """,
                    (run_id, node_id, lease_id, ts),
                )
                if lease is None:
                    return False
            existing = self.db.query_one(
                "SELECT state FROM nodes WHERE run_id = ? AND node_id = ?",
                (run_id, node_id),
            )
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
                active = self.db.query_one(
                    """
                    SELECT lease_id FROM leases
                    WHERE run_id = ? AND node_id = ? AND expires_at_ms > ?
                    """,
                    (run_id, node_id, ts),
                )
                if active is not None and active["lease_id"] != lease_id:
                    return False
            self.db.execute(
                """
                INSERT INTO nodes (
                    run_id, node_id, state, spec_json, attempts_json,
                    result_json, failure_json, history_json,
                    created_at_ms, updated_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, node_id) DO UPDATE SET
                    state = excluded.state,
                    spec_json = excluded.spec_json,
                    attempts_json = excluded.attempts_json,
                    result_json = excluded.result_json,
                    failure_json = excluded.failure_json,
                    history_json = excluded.history_json,
                    updated_at_ms = excluded.updated_at_ms
                """,
                (
                    run_id,
                    node_id,
                    state_val,
                    _dumps(spec),
                    _dumps(attempts),
                    _dumps(result) if result is not None else None,
                    _dumps(failure) if failure is not None else None,
                    _dumps(history or []),
                    ts,
                    ts,
                ),
            )
            # Refresh attempts_index in the same transaction as the node row.
            self.db.execute(
                "DELETE FROM attempts_index WHERE run_id = ? AND node_id = ?",
                (run_id, node_id),
            )
            if attempts:
                self.db.executemany(
                    """
                    INSERT INTO attempts_index (
                        run_id, node_id, attempt_id, worker_id,
                        started_at_ms, ended_at_ms, outcome
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            run_id,
                            node_id,
                            a.get("attempt_id"),
                            a.get("worker_id"),
                            a.get("started_at_ms"),
                            a.get("ended_at_ms"),
                            a.get("outcome"),
                        )
                        for a in attempts
                    ],
                )
        return True

    def get_node(self, run_id: str, node_id: str) -> dict[str, Any] | None:
        row = self.db.query_one(
            "SELECT * FROM nodes WHERE run_id = ? AND node_id = ?",
            (run_id, node_id),
        )
        return _row_to_node(row) if row else None

    def list_nodes(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM nodes WHERE run_id = ? ORDER BY created_at_ms ASC",
            (run_id,),
        )
        return [_row_to_node(r) for r in rows]

    def list_nodes_by_state(
        self, run_id: str, state: NodeState | str
    ) -> list[dict[str, Any]]:
        state_val = state.value if isinstance(state, NodeState) else str(state)
        rows = self.db.query(
            "SELECT * FROM nodes WHERE run_id = ? AND state = ?",
            (run_id, state_val),
        )
        return [_row_to_node(r) for r in rows]

    def count_by_state(self, run_id: str) -> dict[str, int]:
        rows = self.db.query(
            "SELECT state, COUNT(*) AS n FROM nodes WHERE run_id = ? GROUP BY state",
            (run_id,),
        )
        return {r["state"]: int(r["n"]) for r in rows}

    # -- edges ----------------------------------------------------------------

    def add_edges(
        self, run_id: str, edges: Iterable[tuple[str, str]]
    ) -> None:
        rows = [(run_id, src, dst) for src, dst in edges]
        if not rows:
            return
        with self.db.transaction():
            self.db.executemany(
                """
                INSERT OR IGNORE INTO edges (run_id, src_node_id, dst_node_id)
                VALUES (?, ?, ?)
                """,
                rows,
            )

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
        ts = now_ms()
        state_val = state.value if isinstance(state, NodeState) else str(state)
        expected_state_val = (
            expected_state.value
            if isinstance(expected_state, NodeState)
            else (str(expected_state) if expected_state is not None else None)
        )
        terminal_values = {s.value for s in TERMINAL_NODE_STATES}
        edge_rows = [(run_id, src, dst) for src, dst in edges]
        with self.db.transaction():
            if require_active_lease:
                if lease_id is None:
                    return False
                lease = self.db.query_one(
                    """
                    SELECT lease_id FROM leases
                    WHERE run_id = ? AND node_id = ? AND lease_id = ?
                      AND expires_at_ms > ?
                    """,
                    (run_id, node_id, lease_id, ts),
                )
                if lease is None:
                    return False
            existing = self.db.query_one(
                "SELECT state FROM nodes WHERE run_id = ? AND node_id = ?",
                (run_id, node_id),
            )
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
                active = self.db.query_one(
                    """
                    SELECT lease_id FROM leases
                    WHERE run_id = ? AND node_id = ? AND expires_at_ms > ?
                    """,
                    (run_id, node_id, ts),
                )
                if active is not None and active["lease_id"] != lease_id:
                    return False
            self.db.execute(
                """
                INSERT INTO nodes (
                    run_id, node_id, state, spec_json, attempts_json,
                    result_json, failure_json, history_json,
                    created_at_ms, updated_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, node_id) DO UPDATE SET
                    state = excluded.state,
                    spec_json = excluded.spec_json,
                    attempts_json = excluded.attempts_json,
                    result_json = excluded.result_json,
                    failure_json = excluded.failure_json,
                    history_json = excluded.history_json,
                    updated_at_ms = excluded.updated_at_ms
                """,
                (
                    run_id, node_id, state_val, _dumps(spec),
                    _dumps(attempts),
                    _dumps(result) if result is not None else None,
                    _dumps(failure) if failure is not None else None,
                    _dumps(history or []), ts, ts,
                ),
            )
            self.db.execute(
                "DELETE FROM attempts_index WHERE run_id = ? AND node_id = ?",
                (run_id, node_id),
            )
            if attempts:
                self.db.executemany(
                    """
                    INSERT INTO attempts_index (
                        run_id, node_id, attempt_id, worker_id,
                        started_at_ms, ended_at_ms, outcome
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            run_id, node_id, a.get("attempt_id"),
                            a.get("worker_id"), a.get("started_at_ms"),
                            a.get("ended_at_ms"), a.get("outcome"),
                        )
                        for a in attempts
                    ],
                )
            if edge_rows:
                self.db.executemany(
                    """
                    INSERT OR IGNORE INTO edges (run_id, src_node_id, dst_node_id)
                    VALUES (?, ?, ?)
                    """,
                    edge_rows,
                )
        return True

    def list_edges(self, run_id: str) -> list[tuple[str, str]]:
        rows = self.db.query(
            "SELECT src_node_id, dst_node_id FROM edges WHERE run_id = ?",
            (run_id,),
        )
        return [(r["src_node_id"], r["dst_node_id"]) for r in rows]

    def list_attempts(
        self, run_id: str, node_id: str | None = None
    ) -> list[dict[str, Any]]:
        if node_id is None:
            rows = self.db.query(
                "SELECT * FROM attempts_index WHERE run_id = ?",
                (run_id,),
            )
        else:
            rows = self.db.query(
                "SELECT * FROM attempts_index WHERE run_id = ? AND node_id = ?",
                (run_id, node_id),
            )
        return [
            {
                "run_id": r["run_id"],
                "node_id": r["node_id"],
                "attempt_id": r["attempt_id"],
                "worker_id": r["worker_id"],
                "started_at_ms": r["started_at_ms"],
                "ended_at_ms": r["ended_at_ms"],
                "outcome": r["outcome"],
            }
            for r in rows
        ]

    # -- leases ---------------------------------------------------------------

    def grant_lease(self, lease: Lease) -> None:
        """Insert lease; raises sqlite3.IntegrityError if (run_id, node_id) exists."""
        with self.db.transaction():
            self.db.execute(
                """
                DELETE FROM leases
                WHERE run_id = ? AND node_id = ? AND expires_at_ms <= ?
                """,
                (lease.run_id, lease.node_id, now_ms()),
            )
            self.db.execute(
                """
                INSERT INTO leases (
                    lease_id, run_id, node_id, worker_id,
                    granted_at_ms, expires_at_ms, heartbeat_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    lease.lease_id,
                    lease.run_id,
                    lease.node_id,
                    lease.worker_id,
                    lease.granted_at_ms,
                    lease.expires_at_ms,
                    lease.heartbeat_at_ms,
                ),
            )

    def heartbeat_lease(self, lease_id: str, expires_at_ms: int) -> bool:
        cur = self.db.execute(
            """
            UPDATE leases SET heartbeat_at_ms = ?, expires_at_ms = ?
            WHERE lease_id = ? AND expires_at_ms > ?
            """,
            (now_ms(), expires_at_ms, lease_id, now_ms()),
        )
        return cur.rowcount > 0

    def release_lease(self, lease_id: str) -> bool:
        cur = self.db.execute(
            "DELETE FROM leases WHERE lease_id = ? AND expires_at_ms > ?",
            (lease_id, now_ms()),
        )
        return cur.rowcount > 0

    def find_expired(self, *, now: int) -> list[Lease]:
        rows = self.db.query(
            "SELECT * FROM leases WHERE expires_at_ms <= ?",
            (now,),
        )
        return [_row_to_lease(r) for r in rows]

    def reclaim_expired(
        self, *, now: int, run_id: str | None = None
    ) -> list[Lease]:
        if run_id is None:
            rows = self.db.query(
                "DELETE FROM leases WHERE expires_at_ms <= ? RETURNING *",
                (now,),
            )
        else:
            rows = self.db.query(
                """
                DELETE FROM leases
                WHERE expires_at_ms <= ? AND run_id = ?
                RETURNING *
                """,
                (now, run_id),
            )
        return [_row_to_lease(r) for r in rows]

    def find_lease_for(self, run_id: str, node_id: str) -> Lease | None:
        row = self.db.query_one(
            "SELECT * FROM leases WHERE run_id = ? AND node_id = ?",
            (run_id, node_id),
        )
        return _row_to_lease(row) if row else None

    def count_leases(self, run_id: str) -> int:
        row = self.db.query_one(
            "SELECT COUNT(*) AS n FROM leases WHERE run_id = ?",
            (run_id,),
        )
        return int(row["n"]) if row else 0


def _row_to_node(row: Any) -> dict[str, Any]:
    return {
        "run_id": row["run_id"],
        "node_id": row["node_id"],
        "state": row["state"],
        "spec": _loads(row["spec_json"]),
        "attempts": _loads(row["attempts_json"]) or [],
        "result": _loads(row["result_json"]),
        "failure": _loads(row["failure_json"]),
        "history": _loads(row["history_json"]) or [],
        "created_at_ms": row["created_at_ms"],
        "updated_at_ms": row["updated_at_ms"],
    }


def _row_to_lease(row: Any) -> Lease:
    return Lease(
        lease_id=row["lease_id"],
        run_id=row["run_id"],
        node_id=row["node_id"],
        worker_id=row["worker_id"],
        granted_at_ms=row["granted_at_ms"],
        expires_at_ms=row["expires_at_ms"],
        heartbeat_at_ms=row["heartbeat_at_ms"],
    )


# --------------------------------------------------------------------------- #
# EventStore
# --------------------------------------------------------------------------- #

class EventStore:
    """Append-only event log. AUTOINCREMENT gives a global monotonic sequence."""

    def __init__(self, db: SQLiteRuntimeDB):
        self.db = db

    def append(
        self, run_id: str, kind: str, payload: dict[str, Any]
    ) -> int:
        cur = self.db.execute(
            """
            INSERT INTO events (run_id, kind, created_at_ms, payload_json)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, kind, now_ms(), _dumps(payload)),
        )
        return int(cur.lastrowid)

    def read_after(
        self, sequence: int = 0, *, limit: int = 1000, run_id: str | None = None
    ) -> list[dict[str, Any]]:
        if run_id is None:
            rows = self.db.query(
                """
                SELECT * FROM events WHERE sequence > ?
                ORDER BY sequence ASC LIMIT ?
                """,
                (sequence, limit),
            )
        else:
            rows = self.db.query(
                """
                SELECT * FROM events WHERE sequence > ? AND run_id = ?
                ORDER BY sequence ASC LIMIT ?
                """,
                (sequence, run_id, limit),
            )
        return [
            {
                "sequence": r["sequence"],
                "run_id": r["run_id"],
                "kind": r["kind"],
                "created_at_ms": r["created_at_ms"],
                "payload": _loads(r["payload_json"]) or {},
            }
            for r in rows
        ]

    def read_sequences(
        self, sequences: Sequence[int], *, run_id: str | None = None
    ) -> list[dict[str, Any]]:
        if not sequences:
            return []
        out: list[dict[str, Any]] = []
        chunk_size = 500
        for i in range(0, len(sequences), chunk_size):
            chunk = [int(s) for s in sequences[i : i + chunk_size]]
            placeholders = ",".join("?" * len(chunk))
            if run_id is None:
                rows = self.db.query(
                    f"""
                    SELECT * FROM events
                    WHERE sequence IN ({placeholders})
                    ORDER BY sequence ASC
                    """,
                    tuple(chunk),
                )
            else:
                rows = self.db.query(
                    f"""
                    SELECT * FROM events
                    WHERE sequence IN ({placeholders}) AND run_id = ?
                    ORDER BY sequence ASC
                    """,
                    (*chunk, run_id),
                )
            out.extend(
                {
                    "sequence": r["sequence"],
                    "run_id": r["run_id"],
                    "kind": r["kind"],
                    "created_at_ms": r["created_at_ms"],
                    "payload": _loads(r["payload_json"]) or {},
                }
                for r in rows
            )
        out.sort(key=lambda e: int(e["sequence"]))
        return out

    def read_last(self, run_id: str, *, limit: int = 1000) -> list[dict[str, Any]]:
        rows = self.db.query(
            """
            SELECT * FROM events WHERE run_id = ?
            ORDER BY sequence DESC LIMIT ?
            """,
            (run_id, limit),
        )
        events = [
            {
                "sequence": r["sequence"],
                "run_id": r["run_id"],
                "kind": r["kind"],
                "created_at_ms": r["created_at_ms"],
                "payload": _loads(r["payload_json"]) or {},
            }
            for r in rows
        ]
        events.sort(key=lambda e: int(e["sequence"]))
        return events

    def max_sequence(self, run_id: str | None = None) -> int:
        if run_id is None:
            row = self.db.query_one("SELECT MAX(sequence) AS s FROM events")
        else:
            row = self.db.query_one(
                "SELECT MAX(sequence) AS s FROM events WHERE run_id = ?",
                (run_id,),
            )
        if row is None or row["s"] is None:
            return 0
        return int(row["s"])


# --------------------------------------------------------------------------- #
# ClaimStore (persistence side)
# --------------------------------------------------------------------------- #

class ClaimStorePersistence:
    """SQL-backed persistence for claims. The in-memory cache lives in
    knowledge/claims.py."""

    def __init__(self, db: SQLiteRuntimeDB):
        self.db = db

    def upsert(
        self,
        claim_id: str,
        run_id: str,
        kind: str,
        statement: str,
        confidence: float,
        evidence: list[dict[str, Any]],
    ) -> None:
        ts = now_ms()
        self.db.execute(
            """
            INSERT INTO claims (
                claim_id, run_id, kind, statement, confidence, evidence_json,
                created_at_ms, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(claim_id) DO UPDATE SET
                kind = excluded.kind,
                statement = excluded.statement,
                confidence = excluded.confidence,
                evidence_json = excluded.evidence_json,
                updated_at_ms = excluded.updated_at_ms
            """,
            (claim_id, run_id, kind, statement, confidence,
             _dumps(evidence), ts, ts),
        )

    def archive(self, claim_id: str) -> None:
        self.db.execute(
            "UPDATE claims SET archived_at_ms = ? WHERE claim_id = ?",
            (now_ms(), claim_id),
        )

    def list_active(
        self, run_id: str, *, limit: int | None = None
    ) -> list[dict[str, Any]]:
        if limit is not None:
            sql = """
            SELECT * FROM (
                SELECT rowid AS _rowid, * FROM claims
                WHERE run_id = ? AND archived_at_ms IS NULL
                ORDER BY created_at_ms DESC, rowid DESC
                LIMIT ?
            ) AS latest_claims
            ORDER BY created_at_ms ASC, _rowid ASC
            """
            params = (run_id, int(limit))
        else:
            sql = """
            SELECT * FROM claims
            WHERE run_id = ? AND archived_at_ms IS NULL
            ORDER BY created_at_ms ASC, claim_id ASC
            """
            params = (run_id,)
        rows = self.db.query(sql, params)
        return [_row_to_claim(r) for r in rows]

    def list_all_active(
        self, *, limit: int | None = None
    ) -> list[dict[str, Any]]:
        if limit is not None:
            sql = """
            SELECT * FROM (
                SELECT rowid AS _rowid, * FROM claims
                WHERE archived_at_ms IS NULL
                ORDER BY created_at_ms DESC, rowid DESC
                LIMIT ?
            ) AS latest_claims
            ORDER BY created_at_ms ASC, _rowid ASC
            """
            params = (int(limit),)
        else:
            sql = """
            SELECT * FROM claims
            WHERE archived_at_ms IS NULL
            ORDER BY created_at_ms ASC, claim_id ASC
            """
            params: tuple[Any, ...] = ()
        rows = self.db.query(sql, params)
        return [_row_to_claim(r) for r in rows]

    def count_active(self, run_id: str) -> int:
        row = self.db.query_one(
            "SELECT COUNT(*) AS n FROM claims WHERE run_id = ? AND archived_at_ms IS NULL",
            (run_id,),
        )
        return int(row["n"]) if row else 0


def _row_to_claim(row: Any) -> dict[str, Any]:
    return {
        "claim_id": row["claim_id"],
        "run_id": row["run_id"],
        "kind": row["kind"],
        "statement": row["statement"],
        "confidence": float(row["confidence"]),
        "evidence": _loads(row["evidence_json"]) or [],
        "created_at_ms": row["created_at_ms"],
        "updated_at_ms": row["updated_at_ms"],
        "archived_at_ms": row["archived_at_ms"],
    }


# --------------------------------------------------------------------------- #
# SnapshotStore (Phase 3)
# --------------------------------------------------------------------------- #

class SnapshotStore:
    """Persistent ResumeSnapshot writes for a run.

    Snapshots are derived views generated by ``CompactionStrategy`` at
    compaction points (and on demand). Persisting them means a later
    process — same machine or different — can call
    ``EngineV5.get_resume_snapshot(run_id)`` and recover the
    priority-filtered view without re-reading every event.

    Records are append-only; each compaction trigger writes a new row
    so the full history of "what mattered at each compaction point" is
    queryable. ``get_latest`` is the common read path; ``list_for_run``
    is for analytics / debugging.
    """

    def __init__(self, db: SQLiteRuntimeDB) -> None:
        self.db = db

    def save(
        self,
        *,
        snapshot_id: str,
        run_id: str,
        triggered_by: str,
        highwater_sequence: int,
        summary: str,
        payload: dict[str, Any],
    ) -> None:
        """Insert a single snapshot row.

        ``payload`` is JSON-serialised; the canonical layout is the
        dict form of a :class:`ResumeSnapshot` so a reader can
        round-trip it directly.
        """
        self.db.execute(
            """
            INSERT INTO snapshots (
                snapshot_id, run_id, triggered_by, highwater_sequence,
                summary, payload_json, created_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot_id, run_id, triggered_by, int(highwater_sequence),
                summary, _dumps(payload), now_ms(),
            ),
        )

    def get_latest(self, run_id: str) -> dict[str, Any] | None:
        row = self.db.query_one(
            """
            SELECT snapshot_id, run_id, triggered_by, highwater_sequence,
                   summary, payload_json, created_at_ms
            FROM snapshots
            WHERE run_id = ?
            ORDER BY created_at_ms DESC, snapshot_id DESC
            LIMIT 1
            """,
            (run_id,),
        )
        if row is None:
            return None
        return _row_to_snapshot(row)

    def list_for_run(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.db.query(
            """
            SELECT snapshot_id, run_id, triggered_by, highwater_sequence,
                   summary, payload_json, created_at_ms
            FROM snapshots
            WHERE run_id = ?
            ORDER BY created_at_ms ASC, snapshot_id ASC
            """,
            (run_id,),
        )
        return [_row_to_snapshot(r) for r in rows]


def _row_to_snapshot(row: Any) -> dict[str, Any]:
    return {
        "snapshot_id": row["snapshot_id"],
        "run_id": row["run_id"],
        "triggered_by": row["triggered_by"],
        "highwater_sequence": int(row["highwater_sequence"] or 0),
        "summary": row["summary"] or "",
        "payload": _loads(row["payload_json"]) or {},
        "created_at_ms": int(row["created_at_ms"]),
    }


__all__ = [
    "ClaimStorePersistence",
    "EventStore",
    "GraphStore",
    "RunStore",
    "SnapshotStore",
]
