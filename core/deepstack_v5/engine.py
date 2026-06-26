"""EngineV5 — main loop tying control + execution + persistence together.

Loop body (one iteration):
  1. Reclaim expired leases; affected nodes transition back to READY.
  2. Promote PENDING nodes whose deps are satisfied.
  3. Ask Controller.decide() for the next action.
  4. ENQUEUE: dispatch ready nodes (sync or via thread pool); handle
     each result (retry / replan / abandon).
  5. WAIT: sleep poll_interval_s.
  6. HALT: exit loop.
  7. Persist deltas to GraphStore. Fire budget warnings → compaction.
"""

from __future__ import annotations

import enum
import logging
import os
import sqlite3
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

from .control.controller import Controller, ControllerInputs
from .control.escalation import EscalationContext, EscalationPolicy
from .execution.dispatcher import Dispatcher, DispatchResult
from .execution.graph import WorkGraph
from .execution.node import NodeExecution
from .types import (
    DecisionKind,
    Budget,
    Failure,
    FailureKind,
    NodeSpec,
    NodeState,
    RunStatus,
    Scope,
    ScopeLevel,
    StepResult,
    TERMINAL_RUN_STATUSES,
    ToolCallState,
    Verdict,
    new_id,
    now_ms,
)
from .execution.toolcall import ToolCall

if TYPE_CHECKING:
    from .memory.snapshot import ResumeSnapshot
    from .runtime import RuntimeV5


def _snapshot_from_row(row: dict[str, Any] | None) -> "ResumeSnapshot | None":
    """Rehydrate a persisted snapshot row into a ResumeSnapshot dataclass.

    Inline import keeps the engine module decoupled from
    ``memory/snapshot`` at load time. Returns ``None`` if the row is
    missing or malformed (payload corruption shouldn't crash callers).
    """
    if row is None:
        return None
    try:
        from .memory.snapshot import EventRef, ResumeSnapshot
        payload = row.get("payload") or {}
        return ResumeSnapshot(
            run_id=row["run_id"],
            summary=str(payload.get("summary") or row.get("summary") or ""),
            highwater_sequence=int(
                payload.get("highwater_sequence")
                or row.get("highwater_sequence")
                or 0
            ),
            events=[
                EventRef(
                    sequence=int(ev.get("sequence", 0)),
                    kind=str(ev.get("kind", "")),
                    priority=int(ev.get("priority", 2)),
                    payload_excerpt=dict(ev.get("payload_excerpt") or {}),
                    occurred_at_ms=int(ev.get("occurred_at_ms", 0)),
                )
                for ev in (payload.get("events") or [])
            ],
            built_at_ms=int(payload.get("built_at_ms") or 0),
        )
    except Exception:
        logger.exception(
            "EngineV5._snapshot_from_row: failed to rehydrate snapshot row"
        )
        return None


@dataclass(slots=True)
class _ReplanState:
    """Per-node replan accounting threaded into EscalationContext."""
    local_used: int = 0
    global_used: int = 0
    same_id_reuses: int = 0


class _PersistResult(enum.Enum):
    SUCCESS = "success"
    FENCE_REJECTED = "fence_rejected"
    DB_ERROR = "db_error"


class _LeaseLostError(RuntimeError):
    """A node's lease expired mid-flight AND another worker has genuinely
    superseded the node (re-leased it or wrote a terminal state for it).

    Distinct from an opaque ``dispatch crashed`` exception: this is a clean,
    *retryable* worker-lost condition — the engine converts it to
    ``Failure(kind=WORKER_LOST, retryable=True)`` so the node is re-dispatched
    rather than abandoned. It is raised by the engine's lease-persist callback
    only after a salvage attempt (committing the completed result without the
    lease fence) has itself been refused by the store's competitor guards —
    i.e. only when re-running really is the correct response.
    """


_REPLAN_METADATA_KEY = "replan_state_v1"


def _replan_int(value: Any, *, run_id: str, field: str) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        logger.warning(
            "EngineV5: malformed %s in %s for run_id=%s; using 0",
            _REPLAN_METADATA_KEY,
            field,
            run_id,
        )
        return 0


class EngineV5:
    def __init__(self, runtime: "RuntimeV5") -> None:
        self._rt = runtime
        # run_id -> WorkGraph
        self._graphs: dict[str, WorkGraph] = {}
        self._replan_state: dict[tuple[str, str], _ReplanState] = {}
        self._replan_run_totals: dict[str, int] = {}
        self._halt_reasons: dict[str, str] = {}
        # Track which leases / dispatchers belong to which run.
        self._dispatchers: dict[str, Dispatcher] = {}
        self._escalation = EscalationPolicy(
            local_to_global_threshold=self._rt.config.local_to_global_threshold,
        )
        # Parallel-dispatch executors currently in flight. Tracked so an
        # external teardown (EngineV5.shutdown, called from Bundle.shutdown)
        # can cancel still-queued work without blocking. Guarded by a lock
        # because a batch registers/unregisters from the engine-driving
        # thread while shutdown() may be invoked from another (the api.py
        # teardown path runs on a separate thread / asyncio executor).
        self._active_executors: set[ThreadPoolExecutor] = set()
        self._executors_lock = threading.Lock()
        self._shutting_down = False

    # -- shutdown ------------------------------------------------------------

    def _register_executor(self, ex: ThreadPoolExecutor) -> None:
        """Track ``ex`` for teardown; refuse if shutdown already requested.

        Converting "register during shutdown" into a clean ``RuntimeError``
        (caught by the engine's existing ``BaseException`` handlers) is what
        prevents the ``cannot schedule new futures after interpreter
        shutdown`` failure mode from surfacing as an unhandled crash.
        """
        with self._executors_lock:
            if self._shutting_down:
                ex.shutdown(wait=False, cancel_futures=True)
                raise RuntimeError("EngineV5 is shutting down")
            self._active_executors.add(ex)

    def _unregister_executor(self, ex: ThreadPoolExecutor) -> None:
        with self._executors_lock:
            self._active_executors.discard(ex)

    def shutdown(self, *, timeout_s: float = 2.0) -> None:
        """Cancel any in-flight parallel-dispatch executors, fast.

        Deliberately uses ``shutdown(wait=False, cancel_futures=True)`` — it
        cancels still-queued futures and returns immediately rather than
        joining (a wedged worker would otherwise block teardown forever).
        ``timeout_s`` is accepted for API symmetry but intentionally not used
        to join workers; the bound is "never wait". Idempotent and safe to
        call when no batch is in flight. A worker already executing a wedged
        node cannot be killed in-process — that residual is the harness's
        ``os._exit`` responsibility (see scripts/ccx_soak.py).
        """
        with self._executors_lock:
            self._shutting_down = True
            executors = list(self._active_executors)
            self._active_executors.clear()
        for ex in executors:
            try:
                ex.shutdown(wait=False, cancel_futures=True)
            except Exception:
                logger.warning(
                    "EngineV5.shutdown: executor shutdown raised; ignoring",
                    exc_info=True,
                )

    # -- entry points --------------------------------------------------------

    def run(self, goal: str) -> Verdict:
        run_id = new_id("run")
        self._rt.run_store.create(
            run_id, goal,
            status=RunStatus.RUNNING,
            budget=self._rt.budget.budget.snapshot(),
        )
        graph = WorkGraph()
        self._graphs[run_id] = graph
        dispatcher = self._make_dispatcher(run_id, graph)
        self._dispatchers[run_id] = dispatcher

        interrupt: BaseException | None = None
        try:
            # Initial proposal.
            specs = self._rt.controller.propose_initial(goal)
            for spec in specs:
                graph.add(spec)
                self._persist_node(run_id, graph.get(spec.node_id))
        except BaseException as exc:  # noqa: BLE001
            logger.exception("EngineV5 initial proposal failed for run_id=%s", run_id)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                interrupt = exc
            verdict = self._build_verdict(
                run_id,
                goal,
                graph,
                None,
                error=f"{type(exc).__name__}: {exc}",
            )
            self._rt.run_store.update_status(
                run_id,
                verdict.status,
                budget=self._rt.budget.budget.snapshot(),
            )
            if interrupt is not None:
                raise interrupt
            return verdict

        return self._loop(run_id, goal, graph)

    def resume(self, run_id: str, *, budget: Budget | dict[str, Any] | None = None) -> Verdict:
        run = self._rt.run_store.get(run_id)
        if run is None:
            raise KeyError(f"unknown run_id {run_id}")

        graph = self._reconstruct_graph(run_id)
        self._graphs[run_id] = graph

        stored_status = RunStatus(run["status"])
        self._load_replan_state(run_id, run.get("metadata"))
        budget_snapshot = self._budget_snapshot_for_resume(run.get("budget"), budget)
        self._rt.budget.restore(budget_snapshot)
        if stored_status == RunStatus.BUDGET_EXHAUSTED and budget is not None:
            stored_status = RunStatus.RUNNING
            self._rt.run_store.update_status(
                run_id,
                RunStatus.RUNNING,
                budget=self._rt.budget.budget.snapshot(),
            )
        if stored_status not in (RunStatus.RUNNING, RunStatus.WAITING_APPROVAL):
            verdict = self._build_verdict(run_id, run["goal"], graph, None)
            verdict.status = stored_status
            return verdict

        # Reclaim leases that may have been left in flight when the prior
        # process died.
        self._rt.assignment.reclaim_expired(now=now_ms(), run_id=run_id)
        # Push RUNNING nodes whose lease was reclaimed back to READY so the
        # new workers can pick them up. Any node still RUNNING without a
        # current lease is ambiguous: mark it FAILED with WORKER_LOST.
        for node_id, node in graph.nodes().items():
            if node.state == NodeState.RUNNING:
                lease = self._rt.assignment.find_for(run_id, node_id)
                if lease is None:
                    self._mark_worker_lost(
                        node,
                        message="resumed: lease lost, prior worker likely died",
                    )
                    self._persist_node(run_id, node)

        dispatcher = self._make_dispatcher(run_id, graph)
        self._dispatchers[run_id] = dispatcher

        # Replay outbox so subscribers see any prior events.
        self._rt.event_bus.replay_outbox(run_id=run_id)

        return self._loop(run_id, run["goal"], graph)

    def step(self, run_id: str) -> StepResult:
        """Single iteration — useful for tests / external orchestration."""
        graph = self._graphs.get(run_id)
        if graph is None:
            raise KeyError(f"no active graph for {run_id}")
        return self._step_once(run_id, graph)

    def approve(self, run_id: str, node_id: str, approved: bool) -> Verdict:
        """Approve or reject an approval-pending node and continue the run."""
        run = self._rt.run_store.get(run_id)
        if run is None:
            raise KeyError(f"unknown run_id {run_id}")
        self._rt.budget.restore(run.get("budget"))
        graph = self._graphs.get(run_id)
        if graph is None:
            graph = self._reconstruct_graph(run_id)
            self._graphs[run_id] = graph
        self._load_replan_state(run_id, run.get("metadata"))
        stored_status = RunStatus(run["status"])
        if stored_status in TERMINAL_RUN_STATUSES:
            verdict = self._build_verdict(run_id, run["goal"], graph, None)
            verdict.status = stored_status
            return verdict
        if self._rt.budget.should_halt():
            verdict = self._build_verdict(
                run_id,
                run["goal"],
                graph,
                StepResult(
                    iteration=self._rt.budget.budget.iterations,
                    decision_kind=DecisionKind.HALT,
                    should_halt=True,
                    halt_reason="budget exhausted",
                ),
            )
            verdict.status = RunStatus.BUDGET_EXHAUSTED
            self._rt.run_store.update_status(
                run_id,
                RunStatus.BUDGET_EXHAUSTED,
                budget=self._rt.budget.budget.snapshot(),
                expected_status=(RunStatus.RUNNING, RunStatus.WAITING_APPROVAL),
                refuse_if_terminal=True,
            )
            return verdict
        dispatcher = self._make_dispatcher(run_id, graph)
        self._dispatchers[run_id] = dispatcher
        result = dispatcher.resume_after_approval(node_id, approved=approved)
        self._handle_dispatch_result(run_id, graph, result)
        if result.skipped:
            return self._build_verdict(
                run_id,
                run["goal"],
                graph,
                StepResult(
                    iteration=self._rt.budget.budget.iterations,
                    decision_kind=DecisionKind.HALT,
                    should_halt=True,
                    halt_reason=result.skip_reason,
                ),
            )
        return self._loop(run_id, run["goal"], graph)

    def cancel(self, run_id: str, node_id: str | None = None) -> Verdict:
        """Cancel a run or one non-terminal node."""
        run = self._rt.run_store.get(run_id)
        if run is None:
            raise KeyError(f"unknown run_id {run_id}")
        stored_status = RunStatus(run["status"])
        graph = self._graphs.get(run_id)
        if graph is None:
            graph = self._reconstruct_graph(run_id)
            self._graphs[run_id] = graph
        self._load_replan_state(run_id, run.get("metadata"))
        if run_id not in self._dispatchers:
            self._dispatchers[run_id] = self._make_dispatcher(run_id, graph)
        if stored_status not in (RunStatus.RUNNING, RunStatus.WAITING_APPROVAL):
            verdict = self._build_verdict(run_id, run["goal"], graph, None)
            verdict.status = stored_status
            return verdict

        targets = (
            [graph.get(node_id)]
            if node_id is not None
            else list(graph.nodes().values())
        )
        for node in targets:
            if self._cancel_node(run_id, graph, node):
                graph.cascade_skip_from(
                    node.node_id,
                    reason=f"upstream {node.node_id} cancelled",
                )
            for changed in graph.nodes().values():
                if changed.state == NodeState.SKIPPED:
                    self._persist_node(run_id, changed)
        if node_id is not None:
            return self._loop(run_id, run["goal"], graph)
        verdict = self._build_verdict(
            run_id,
            run["goal"],
            graph,
            StepResult(
                iteration=self._rt.budget.budget.iterations,
                decision_kind=DecisionKind.HALT,
                should_halt=True,
                halt_reason="cancelled",
            ),
        )
        verdict.status = RunStatus.CANCELLED
        updated = self._rt.run_store.update_status(
            run_id,
            RunStatus.CANCELLED,
            budget=self._rt.budget.budget.snapshot(),
            expected_status=(RunStatus.RUNNING, RunStatus.WAITING_APPROVAL),
            refuse_if_terminal=True,
        )
        if not updated:
            latest = self._rt.run_store.get(run_id)
            if latest is not None:
                verdict.status = RunStatus(latest["status"])
        return verdict

    def get_node_result(self, run_id: str, node_id: str) -> Any | None:
        """Return a node result from the in-memory graph when available."""
        graph = self._graphs.get(run_id)
        if graph is None:
            return None
        try:
            return graph.get(node_id).result
        except KeyError:
            return None

    def list_node_results(self, run_id: str) -> dict[str, Any]:
        """Return node_id -> result from the in-memory graph."""
        graph = self._graphs.get(run_id)
        if graph is None:
            return {}
        return {node_id: node.result for node_id, node in graph.nodes().items()}

    def _cancel_node(
        self,
        run_id: str,
        graph: WorkGraph,
        node: NodeExecution,
    ) -> bool:
        """Cancel one node, refreshing from DB once if a fence rejects it."""
        node_id = node.node_id
        for _attempt in range(2):
            self._release_node_lease(run_id, node_id)
            if node.is_terminal():
                return node.state == NodeState.CANCELLED
            old_state = node.state
            self._close_cancelled_attempt(node)
            node.transition(NodeState.CANCELLED, reason="cancelled")
            result = self._persist_node(
                run_id,
                node,
                expected_state=old_state,
            )
            if result != _PersistResult.FENCE_REJECTED:
                return True
            if not self._rt.config.persist_to_db:
                return True
            row = self._rt.graph_store.get_node(run_id, node_id)
            if row is None:
                return True
            self._refresh_node_from_row(graph, row)
            node = graph.get(node_id)
        return graph.get(node_id).state == NodeState.CANCELLED

    def _release_node_lease(self, run_id: str, node_id: str) -> None:
        try:
            lease = self._rt.assignment.find_for(run_id, node_id)
            if lease is not None:
                self._rt.assignment.release(lease.lease_id)
        except sqlite3.DatabaseError as exc:
            logger.warning(
                "EngineV5.cancel: failed to release lease for node=%s "
                "(run_id=%s); continuing with fenced write: %s",
                node_id,
                run_id,
                exc,
            )

    # -- snapshots (Phase 3) -------------------------------------------------

    def get_resume_snapshot(self, run_id: str):
        """Return the most recent persisted ResumeSnapshot for ``run_id``.

        Returns ``None`` if no snapshot has been persisted yet (the
        run never crossed a compaction trigger, or compaction wasn't
        wired). Use :meth:`list_snapshots` to walk history.

        The returned object is a :class:`ResumeSnapshot` reconstructed
        from the row's ``payload_json``; callers can render it via
        ``ResumeContext`` or read its fields directly.
        """
        return _snapshot_from_row(self._rt.snapshot_store.get_latest(run_id))

    def list_snapshots(self, run_id: str) -> list:
        """Return all persisted snapshots for ``run_id`` in age order.

        Useful for ``ccx watch stats`` / analytics that want to see
        how compaction unfolded over a long run.
        """
        rows = self._rt.snapshot_store.list_for_run(run_id)
        return [
            snap for snap in (_snapshot_from_row(r) for r in rows)
            if snap is not None
        ]

    def _budget_snapshot_for_resume(
        self,
        persisted: dict[str, Any] | None,
        override: Budget | dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if override is None:
            return persisted
        override_data = (
            override.snapshot()
            if isinstance(override, Budget)
            else dict(override)
        )
        data = dict(persisted or override_data)
        for key in ("max_tokens", "max_cost", "max_wallclock_s", "max_iterations"):
            if override_data.get(key) is not None:
                data[key] = override_data[key]
        return data

    def _load_replan_state(
        self,
        run_id: str,
        metadata: dict[str, Any] | None,
    ) -> None:
        for key in [key for key in self._replan_state if key[0] == run_id]:
            self._replan_state.pop(key, None)
        self._halt_reasons[run_id] = ""
        has_replan_hook = self._rt.controller.has_replan_hook
        if metadata is None:
            if has_replan_hook:
                self._replan_run_totals[run_id] = 0
            return
        if not isinstance(metadata, dict):
            if has_replan_hook:
                self._replan_run_totals[run_id] = 0
            logger.warning(
                "EngineV5: run metadata for run_id=%s is not a dict; "
                "ignoring %s",
                run_id,
                _REPLAN_METADATA_KEY,
            )
            return
        data = metadata.get(_REPLAN_METADATA_KEY)
        if data is None:
            if has_replan_hook:
                self._replan_run_totals[run_id] = 0
            return
        if not isinstance(data, dict):
            if has_replan_hook:
                self._replan_run_totals[run_id] = 0
            logger.warning(
                "EngineV5: %s for run_id=%s is not a dict; ignoring",
                _REPLAN_METADATA_KEY,
                run_id,
            )
            return
        if not data:
            if has_replan_hook:
                self._replan_run_totals[run_id] = 0
            return
        nodes = data.get("nodes") or {}
        if not isinstance(nodes, dict):
            logger.warning(
                "EngineV5: %s.nodes for run_id=%s is not a dict; ignoring",
                _REPLAN_METADATA_KEY,
                run_id,
            )
            nodes = {}
        # Persisted replan metadata is authoritative even on hook-less
        # resume: the old run's counters and halt reason remain useful
        # for reporting. Runs without persisted state only initialise
        # totals when this engine can actually replan.
        self._replan_run_totals[run_id] = _replan_int(
            data.get("run_total"), run_id=run_id, field="run_total"
        )
        self._halt_reasons[run_id] = str(data.get("halt_reason") or "")
        for node_id, item in nodes.items():
            if not isinstance(item, dict):
                logger.warning(
                    "EngineV5: %s.nodes[%s] for run_id=%s is not a dict; "
                    "using empty counters",
                    _REPLAN_METADATA_KEY,
                    node_id,
                    run_id,
                )
                item = {}
            self._replan_state[(run_id, str(node_id))] = _ReplanState(
                local_used=_replan_int(
                    item.get("local_used"), run_id=run_id,
                    field=f"nodes.{node_id}.local_used",
                ),
                global_used=_replan_int(
                    item.get("global_used"), run_id=run_id,
                    field=f"nodes.{node_id}.global_used",
                ),
                same_id_reuses=_replan_int(
                    item.get("same_id_reuses"), run_id=run_id,
                    field=f"nodes.{node_id}.same_id_reuses",
                ),
            )

    def _persist_replan_state(self, run_id: str) -> None:
        if not self._rt.config.persist_to_db:
            return
        try:
            run = self._rt.run_store.get(run_id)
        except sqlite3.DatabaseError as exc:
            logger.warning(
                "engine._persist_replan_state: failed to read run metadata "
                "for run_id=%s; continuing with in-memory counters: %s",
                run_id,
                exc,
            )
            return
        if run is None:
            return
        existing_metadata = run.get("metadata") or {}
        metadata = (
            dict(existing_metadata)
            if isinstance(existing_metadata, dict)
            else {}
        )
        nodes: dict[str, dict[str, int]] = {}
        for (rid, node_id), state in self._replan_state.items():
            if rid != run_id:
                continue
            nodes[node_id] = {
                "local_used": state.local_used,
                "global_used": state.global_used,
                "same_id_reuses": state.same_id_reuses,
            }
        metadata[_REPLAN_METADATA_KEY] = {
            "run_total": self._replan_run_totals.get(run_id, 0),
            "halt_reason": self._halt_reasons.get(run_id, ""),
            "nodes": nodes,
        }
        try:
            self._rt.run_store.update_metadata(run_id, metadata)
        except sqlite3.DatabaseError as exc:
            logger.warning(
                "engine._persist_replan_state: failed to persist metadata "
                "for run_id=%s; continuing with in-memory counters: %s",
                run_id,
                exc,
            )

    # -- loop ----------------------------------------------------------------

    def _loop(self, run_id: str, goal: str, graph: WorkGraph) -> Verdict:
        config = self._rt.config
        result: StepResult | None = None
        error: str | None = None
        interrupt: BaseException | None = None
        try:
            for _it in range(config.max_loop_iterations):
                result = self._step_once(run_id, graph)
                if result.should_halt:
                    break
                if result.decision_kind == DecisionKind.WAIT:
                    if config.poll_interval_s > 0:
                        time.sleep(config.poll_interval_s)
        except BaseException as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            logger.exception("EngineV5 loop failed for run_id=%s", run_id)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                interrupt = exc

        verdict = self._build_verdict(run_id, goal, graph, result, error=error)
        self._rt.run_store.update_status(
            run_id, verdict.status,
            budget=self._rt.budget.budget.snapshot(),
            expected_status=(RunStatus.RUNNING, RunStatus.WAITING_APPROVAL),
            refuse_if_terminal=True,
        )
        if interrupt is not None:
            raise interrupt
        return verdict

    def _step_once(self, run_id: str, graph: WorkGraph) -> StepResult:
        # 1. Reclaim leases.
        # SQLite corruption (``database disk image is malformed``) can
        # surface here during ``reclaim_expired`` when the persistence
        # backend has been left in a bad state by a prior interrupted
        # run. Cleanup is best-effort — losing one round of lease
        # reclamation does not affect the in-memory WorkGraph (the
        # source of truth for the rest of the run), so degrade
        # gracefully instead of crashing the whole process after
        # artifacts have already been written. Log loudly so the user
        # knows the runtime DB needs rebuilding.
        try:
            expired = self._rt.assignment.reclaim_expired(
                now=now_ms(), run_id=run_id
            )
        except sqlite3.DatabaseError as exc:
            logger.warning(
                "v5 engine: lease-reclaim hit SQLite corruption (%s); "
                "skipping cleanup for this step. Quarantine %s and "
                "re-run to rebuild.",
                exc,
                getattr(self._rt.db, "path", "<runtime.db>"),
            )
            expired = []
        for lease in expired:
            try:
                node = graph.get(lease.node_id)
            except KeyError:
                continue
            if node.state == NodeState.RUNNING:
                self._mark_worker_lost(
                    node,
                    message=f"lease {lease.lease_id} expired",
                    worker_id=lease.worker_id,
                )
                self._persist_node(
                    run_id,
                    node,
                    expected_state=NodeState.RUNNING,
                )

        self._patrol_stale_running_nodes(run_id, graph)

        # 2. Handle any FAILED nodes before deciding. This catches resume /
        # sweep failures even when no other node is READY.
        self._handle_failures(run_id, graph)

        # 3. Promote PENDING -> READY and persist the promotion so workers can
        # pick up engine-seeded DAGs.
        for node_id in graph.transition_pending_to_ready():
            self._persist_node(run_id, graph.get(node_id))

        # 4. Build inputs and decide.
        inputs = self._build_controller_inputs(run_id, graph)
        decision = self._rt.controller.decide(inputs)
        self._rt.budget.consume(iteration=True)

        # Fire compaction-related warning once on warning crossing.
        if self._rt.budget.fire_warning_if_needed():
            self._rt.event_bus.publish(run_id, "budget.warning",
                                       self._rt.budget.budget.snapshot())

        nodes_started: tuple[str, ...] = ()
        nodes_completed: tuple[str, ...] = ()

        if decision.kind == DecisionKind.HALT:
            return StepResult(
                iteration=self._rt.budget.budget.iterations,
                decision_kind=decision.kind,
                should_halt=True,
                halt_reason=decision.reason,
            )

        if decision.kind == DecisionKind.WAIT:
            return StepResult(
                iteration=self._rt.budget.budget.iterations,
                decision_kind=decision.kind,
            )

        if decision.kind == DecisionKind.ENQUEUE:
            ready = sorted(
                inputs.ready_nodes,
                key=lambda nid: (-graph.get(nid).spec.priority, nid),
            )
            nodes_started, nodes_completed = self._dispatch_batch(run_id, graph, ready)
            # Handle failures from this batch.
            self._handle_failures(run_id, graph)
            if not nodes_started and self._rt.assignment.count_for_run(run_id) > 0:
                return StepResult(
                    iteration=self._rt.budget.budget.iterations,
                    decision_kind=DecisionKind.WAIT,
                )

        return StepResult(
            iteration=self._rt.budget.budget.iterations,
            decision_kind=decision.kind,
            nodes_started=nodes_started,
            nodes_completed=nodes_completed,
        )

    # -- dispatch ------------------------------------------------------------

    def _dispatch_batch(
        self,
        run_id: str,
        graph: WorkGraph,
        ready: list[str],
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if not ready:
            return ((), ())
        config = self._rt.config
        dispatcher = self._dispatchers[run_id]
        started: list[str] = []
        completed: list[str] = []
        if config.parallelism <= 1:
            for node_id in ready:
                if self._rt.budget.should_halt():
                    break
                if not self._fresh_ready_check(run_id, graph, node_id):
                    continue
                try:
                    res = dispatcher.dispatch_one(node_id)
                except Exception as exc:
                    res = self._dispatch_crashed_result(graph, node_id, exc)
                self._handle_dispatch_result(run_id, graph, res)
                if not res.skipped:
                    started.append(node_id)
                    completed.append(node_id)
        else:
            dispatchable: list[str] = []
            for node_id in ready:
                if self._rt.budget.should_halt():
                    break
                if self._fresh_ready_check(run_id, graph, node_id):
                    dispatchable.append(node_id)
            if dispatchable:
                # Explicit executor lifecycle instead of ``with ... as ex:``.
                # The context-manager ``__exit__`` calls ``shutdown(wait=True)``,
                # which blocks forever joining non-daemon workers when a
                # BaseException (SystemExit / KeyboardInterrupt) escapes — the
                # exact zombie-process hang this replaces. Here every exit path
                # tears the pool down with ``wait=False, cancel_futures=True``,
                # so teardown never blocks.
                ex = ThreadPoolExecutor(max_workers=config.parallelism)
                self._register_executor(ex)
                try:
                    futures = [
                        ex.submit(dispatcher.dispatch_one, n) for n in dispatchable
                    ]
                    for n, fut in zip(dispatchable, futures):
                        try:
                            res = fut.result(
                                timeout=self._future_timeout_for(graph, n)
                            )
                        except FutureTimeoutError as exc:
                            # Backstop: the worker overran its own deadline.
                            # Treat as a re-drivable worker-loss, not an opaque
                            # crash, and do not block the remaining futures.
                            res = self._dispatch_timed_out_result(graph, n, exc)
                        except Exception as exc:
                            res = self._dispatch_crashed_result(graph, n, exc)
                        self._handle_dispatch_result(run_id, graph, res)
                        if not res.skipped:
                            started.append(n)
                            completed.append(n)
                except BaseException:
                    # SystemExit / KeyboardInterrupt: cancel queued work before
                    # unwinding so __exit__-style wait=True joins never happen.
                    ex.shutdown(wait=False, cancel_futures=True)
                    raise
                finally:
                    self._unregister_executor(ex)
                    ex.shutdown(wait=False, cancel_futures=True)
        return (tuple(started), tuple(completed))

    def _dispatch_crashed_result(
        self,
        graph: WorkGraph,
        node_id: str,
        exc: Exception,
    ) -> DispatchResult:
        # A lost lease whose salvage was refused (a real competitor took the
        # node) is a clean, retryable worker-lost condition — NOT an opaque
        # crash. Classifying it as retryable WORKER_LOST lets _handle_failures
        # re-dispatch the node instead of abandoning completed-or-superseded
        # work, mirroring _patrol_stale_running_nodes / _mark_worker_lost.
        if isinstance(exc, _LeaseLostError):
            return DispatchResult(
                node_id=node_id,
                final_state=graph.get(node_id).state,
                failure=Failure(
                    kind=FailureKind.WORKER_LOST,
                    message=str(exc),
                    retryable=True,
                ),
            )
        return DispatchResult(
            node_id=node_id,
            final_state=graph.get(node_id).state,
            failure=Failure(
                kind=FailureKind.UNKNOWN,
                message=f"dispatch crashed: {exc}",
                retryable=False,
                details={"traceback": traceback.format_exc()},
            ),
        )

    def _dispatch_timed_out_result(
        self,
        graph: WorkGraph,
        node_id: str,
        exc: BaseException,
    ) -> DispatchResult:
        # The dispatch future exceeded its backstop deadline — the worker
        # overran (or ignored) its own ``timeout_s``. This is a re-drivable
        # worker-loss, NOT an opaque crash: same semantics as a refused lease
        # salvage, so _handle_failures re-dispatches rather than abandons.
        return DispatchResult(
            node_id=node_id,
            final_state=graph.get(node_id).state,
            failure=Failure(
                kind=FailureKind.WORKER_LOST,
                message=f"dispatch future backstop timeout: {exc}",
                retryable=True,
            ),
        )

    @staticmethod
    def _future_timeout_margin_s() -> float:
        """Slack added to a node's own deadline before the future backstop.

        Read per call from ``CCX_DISPATCH_FUTURE_MARGIN_S`` (default 60.0),
        mirroring ``_node_idle_timeout_s`` so a launch/test can set it in the
        environment without import-order surprises. The margin keeps the
        worker's own ``_call_with_timeout`` the *first* deadline to fire (a
        clean TIMEOUT classification); the future backstop only triggers when
        a worker overruns its own bound. Non-positive / malformed ⇒ 0.0.
        """
        raw = os.environ.get("CCX_DISPATCH_FUTURE_MARGIN_S", "").strip()
        if not raw:
            return 60.0
        try:
            value = float(raw)
        except ValueError:
            return 60.0
        return value if value > 0 else 0.0

    def _future_timeout_for(self, graph: WorkGraph, node_id: str) -> float | None:
        """Backstop wall-clock for one dispatch future, or ``None``.

        ``None`` (legacy unbounded ``fut.result()``) is returned when the node
        has no effective per-node deadline (``node.spec.timeout_s`` and the
        capability's ``timeout_s`` both unset) — such a node runs without an
        inner ``_call_with_timeout``, so a finite future timeout would
        wrongly classify a legitimately-long node as WORKER_LOST. Otherwise
        the backstop is the node's effective ``timeout_s`` plus a margin, so
        the worker's own deadline fires first.
        """
        node = graph.get(node_id)
        cap = self._rt.capabilities.get(node.spec.tool)
        # Mirror the dispatcher's own ``node.spec.timeout_s or cap.timeout_s``
        # (dispatcher.py:260) and its ``timeout_s > 0`` guard (dispatcher.py:300):
        # a 0 / None effective deadline means the node runs unbounded, so the
        # backstop must also be unbounded.
        eff = node.spec.timeout_s or (cap.timeout_s if cap is not None else None)
        if not eff or eff <= 0:
            return None
        return float(eff) + self._future_timeout_margin_s()

    def _fresh_ready_check(
        self, run_id: str, graph: WorkGraph, node_id: str
    ) -> bool:
        if not self._rt.config.persist_to_db:
            return True
        row = self._rt.graph_store.get_node(run_id, node_id)
        if row is None:
            return True
        if row["state"] == NodeState.READY.value:
            return True
        self._refresh_node_from_row(graph, row)
        return False

    def _refresh_node_from_row(
        self, graph: WorkGraph, row: dict[str, Any]
    ) -> None:
        node = self._node_from_row(row)
        if graph.has(node.node_id):
            graph.replace_execution(node, validate_deps=False)
        else:
            graph.add_execution(node, validate_deps=False)

    def _node_from_row(self, row: dict[str, Any]) -> NodeExecution:
        return NodeExecution.from_dict({
            "spec": row["spec"],
            "state": row["state"],
            "attempts": row["attempts"],
            "result": row["result"],
            "failure": row["failure"],
            "history": row.get("history") or [],
            "created_at_ms": row["created_at_ms"],
            "updated_at_ms": row["updated_at_ms"],
        })

    def _patrol_stale_running_nodes(self, run_id: str, graph: WorkGraph) -> None:
        if not self._rt.config.persist_to_db:
            return
        for node_id, node in list(graph.nodes().items()):
            if node.state != NodeState.RUNNING:
                continue
            if self._rt.assignment.find_for(run_id, node_id) is not None:
                continue
            row = self._rt.graph_store.get_node(run_id, node_id)
            if row is None:
                self._mark_worker_lost(
                    node,
                    message="running node has no active lease or DB row",
                )
                self._persist_node(run_id, node)
                continue
            row_state = NodeState(row["state"])
            if row_state != NodeState.RUNNING:
                self._refresh_node_from_row(graph, row)
                continue
            self._mark_worker_lost(
                node,
                message="running node has no active lease",
            )
            self._persist_node(
                run_id,
                node,
                expected_state=NodeState.RUNNING,
            )

    def _handle_dispatch_result(
        self,
        run_id: str,
        graph: WorkGraph,
        result: DispatchResult,
    ) -> None:
        try:
            node = graph.get(result.node_id)
        except KeyError:
            return
        if result.skipped:
            if self._rt.config.persist_to_db:
                row = self._rt.graph_store.get_node(run_id, result.node_id)
                if row is not None:
                    self._refresh_node_from_row(graph, row)
            return
        if result.failure is not None and node.state not in (
            NodeState.FAILED,
            NodeState.ABANDONED,
        ):
            self._force_fail_node(node, result.failure)
        self._persist_node(run_id, node)
        if result.final_state == NodeState.ABANDONED:
            for changed in graph.nodes().values():
                if changed.state == NodeState.SKIPPED:
                    self._persist_node(run_id, changed)
        # Persist any children spawned via SpawnResult during this dispatch.
        for child_id in result.spawned_node_ids:
            try:
                child = graph.get(child_id)
            except KeyError:
                continue
            self._persist_node(run_id, child)
        if result.final_state == NodeState.SUCCEEDED:
            self._rt.event_bus.publish(run_id, "node.completed", {
                "node_id": result.node_id,
                "result_summary": str(result.result)[:200],
                "spawned": list(result.spawned_node_ids),
                "tokens": result.tokens_reported,
                "cost": result.cost_reported,
            })

    def _force_fail_node(self, node: NodeExecution, failure: Failure) -> None:
        if node.current_attempt() is None:
            if node.state == NodeState.READY:
                node.transition(NodeState.RUNNING, reason="dispatch failed")
            node.new_attempt(worker_id="engine")
        att = node.current_attempt()
        if att is not None and att.outcome is None:
            node.finish_attempt(outcome="failure", failure=failure)
        else:
            node.failure = failure
            if att is not None:
                att.outcome = "failure"
                att.failure = failure
                att.ended_at_ms = now_ms()
            if node.is_terminal() and node.state != NodeState.ABANDONED:
                node.history.append(
                    (
                        node.state.value,
                        NodeState.FAILED.value,
                        now_ms(),
                        failure.message[:80],
                    )
                )
                node.state = NodeState.FAILED
                node.result = None
                node.updated_at_ms = now_ms()
                return
        if node.state == NodeState.RUNNING:
            node.transition(NodeState.FAILED, reason=failure.message[:80])

    # -- failure handling / replan ------------------------------------------

    def _handle_failures(self, run_id: str, graph: WorkGraph) -> None:
        for node_id, node in list(graph.nodes().items()):
            if node.state != NodeState.FAILED:
                continue
            if node.failure is None:
                # Defensive: synthesize a generic failure.
                node.failure = Failure(
                    kind=FailureKind.UNKNOWN,
                    message="failed without failure record",
                )
            self._rt.controller.notify_failure(node, node.failure)

            replan_state = self._replan_state.setdefault(
                (run_id, node_id), _ReplanState()
            )
            ctx = EscalationContext(
                node_id=node_id,
                attempts_used=node.attempt_count(),
                max_attempts=node.spec.max_attempts,
                local_replans_used=replan_state.local_used,
                global_replans_used=replan_state.global_used,
            )
            scope = self._escalation.classify(node.failure, ctx)

            unknown_effect = self._has_unknown_effect(node)
            cap = self._rt.capabilities.get(node.spec.tool)
            non_idempotent_unknown = (
                unknown_effect and (cap is None or not cap.idempotent)
            )
            if scope.level == ScopeLevel.STEP and non_idempotent_unknown:
                scope = Scope(
                    level=ScopeLevel.LOCAL,
                    node_id=node_id,
                    reason=f"unknown-effect-gate: {scope.reason}",
                )

            if (
                scope.level == ScopeLevel.STEP
                and node.can_retry()
            ):
                if scope.retry_after_ms is not None and scope.retry_after_ms > 0:
                    time.sleep(min(scope.retry_after_ms, 5_000) / 1000.0)
                node.transition(NodeState.READY, reason=f"step retry: {scope.reason}")
                self._persist_node(run_id, node)
                continue

            # LOCAL or GLOBAL — invoke replan hook if one is configured.
            new_specs: list[NodeSpec] = []
            if self._rt.controller.has_replan_hook:
                if (
                    self._replan_run_totals.get(run_id, 0)
                    >= self._rt.config.max_replans_per_run
                ):
                    self._abandon_for_replan_budget(run_id, graph, node_id, scope)
                    continue
                new_specs = self._rt.controller.replan(scope, node, scope.reason)
                self._replan_run_totals[run_id] = (
                    self._replan_run_totals.get(run_id, 0) + 1
                )
                if scope.level == ScopeLevel.LOCAL:
                    replan_state.local_used += 1
                else:
                    replan_state.global_used += 1
                self._persist_replan_state(run_id)

            added: list[str] = []
            reused_current = False
            if new_specs:
                for spec in new_specs:
                    if spec.node_id == node_id:
                        if (
                            replan_state.same_id_reuses
                            >= self._rt.config.max_replans_per_node
                        ):
                            continue
                        replan_state.same_id_reuses += 1
                        graph.replace_spec(node_id, spec)
                        node.failure = None
                        node.transition(
                            NodeState.READY,
                            reason=f"replan reused node id: {scope.reason}",
                        )
                        self._persist_node(run_id, node)
                        self._persist_replan_state(run_id)
                        added.append(spec.node_id)
                        reused_current = True
                    elif not graph.has(spec.node_id):
                        graph.add(spec)
                        self._persist_node(run_id, graph.get(spec.node_id))
                        added.append(spec.node_id)
                self._rt.event_bus.publish(run_id, "replan.applied", {
                    "scope": scope.level.value,
                    "added": added,
                    "skipped_existing": [
                        s.node_id
                        for s in new_specs
                        if s.node_id not in added
                    ],
                    "trigger_node": node_id,
                })

            if reused_current:
                continue

            graph.mark(
                node_id,
                NodeState.ABANDONED,
                reason=f"{scope.level.value}: {scope.reason}",
            )
            for changed in graph.nodes().values():
                if changed.state in (NodeState.ABANDONED, NodeState.SKIPPED):
                    self._persist_node(run_id, changed)

    def _abandon_for_replan_budget(
        self,
        run_id: str,
        graph: WorkGraph,
        node_id: str,
        scope: Scope,
    ) -> None:
        reason = "replan budget exhausted"
        self._halt_reasons[run_id] = reason
        graph.mark(
            node_id,
            NodeState.ABANDONED,
            reason=f"{scope.level.value}: {reason}",
        )
        for changed in graph.nodes().values():
            if changed.state in (NodeState.ABANDONED, NodeState.SKIPPED):
                self._persist_node(run_id, changed)
        self._persist_replan_state(run_id)

    def _has_unknown_effect(self, node: NodeExecution) -> bool:
        for attempt in node.attempts:
            for tc in attempt.tool_calls:
                if tc.state == ToolCallState.UNKNOWN_EFFECT:
                    return True
        return False

    # -- helpers -------------------------------------------------------------

    def _make_dispatcher(self, run_id: str, graph: WorkGraph) -> Dispatcher:
        def emit(kind: str, payload: dict[str, Any]) -> None:
            self._rt.event_bus.publish(run_id, kind, payload)
        def report_cost(tokens: int, cost: float) -> None:
            self._rt.budget.consume(tokens=tokens, cost=cost)
        def persist_under_lease(node: NodeExecution, lease_id: str) -> None:
            result = self._persist_node(
                run_id,
                node,
                lease_id=lease_id,
                require_active_lease=True,
            )
            if result != _PersistResult.FENCE_REJECTED:
                return
            # The lease no longer owns the node. This is frequently a FALSE
            # POSITIVE: the in-process engine blocks on the dispatch future and
            # reclaims nothing mid-flight, so a healthily-running node can have
            # its lease expire purely by wall-clock (a briefly-starved
            # heartbeat) with NO competing worker. Discarding genuinely-
            # completed work in that case is the bug. Try to SALVAGE: re-persist
            # WITHOUT the lease fence. The store still refuses the write if a
            # real competitor has finished the node (refuse_if_terminal) or
            # holds an active lease on it while RUNNING (refuse_if_running_
            # unowned), so this preserves the anti-double-run invariant — it
            # only commits when no other worker actually took over.
            salvage = self._persist_node(run_id, node)
            if salvage == _PersistResult.SUCCESS:
                return
            # A genuine competitor owns or finished the node — re-running is the
            # correct response, NOT abandoning. Surface a typed, retryable
            # worker-lost signal (mirrors _patrol_stale_running_nodes).
            raise _LeaseLostError(
                f"lease {lease_id} no longer owns {node.node_id} and the node "
                "was superseded by another worker; clean retry required"
            )
        return Dispatcher(
            run_id=run_id,
            graph=graph,
            assignment=self._rt.assignment,
            capabilities=self._rt.capabilities,
            event_emitter=emit,
            worker_id=f"engine-{run_id[-8:]}",
            on_node_started_with_lease=persist_under_lease,
            on_node_finished_with_lease=persist_under_lease,
            on_toolcall_started_with_lease=persist_under_lease,
            budget_reporter=report_cost,
        )

    def _mark_worker_lost(
        self,
        node: NodeExecution,
        *,
        message: str,
        worker_id: str | None = None,
    ) -> None:
        failure = Failure(
            kind=FailureKind.WORKER_LOST,
            message=message,
            retryable=True,
            worker_id=worker_id,
        )
        att = node.current_attempt()
        if att is not None and att.outcome is None:
            if not att.tool_calls:
                tc = ToolCall.new(node.spec.tool, node.spec.params)
                tc.mark_running()
                tc.mark_unknown(message, effect_signature="synthetic:worker-lost")
                att.tool_calls.append(tc)
            for tc in att.tool_calls:
                if tc.state == ToolCallState.RUNNING:
                    tc.mark_unknown(message)
            node.finish_attempt(outcome="abandoned", failure=failure)
        else:
            node.failure = failure
        if node.state == NodeState.RUNNING:
            node.transition(NodeState.FAILED, reason=message[:80])

    def _close_cancelled_attempt(self, node: NodeExecution) -> None:
        failure = Failure(
            kind=FailureKind.UNKNOWN,
            message="cancelled",
            retryable=False,
        )
        att = node.current_attempt()
        if att is not None and att.outcome is None:
            if not att.tool_calls:
                tc = ToolCall.new(node.spec.tool, node.spec.params)
                tc.mark_running()
                tc.mark_unknown("cancelled", effect_signature="synthetic:cancelled")
                att.tool_calls.append(tc)
            for tc in att.tool_calls:
                if tc.state == ToolCallState.RUNNING:
                    tc.mark_unknown("cancelled")
                elif tc.state in (ToolCallState.PENDING, ToolCallState.APPROVAL_PENDING):
                    tc.reject()
            node.finish_attempt(outcome="cancelled", failure=failure)
        else:
            node.failure = failure

    def _build_controller_inputs(
        self, run_id: str, graph: WorkGraph
    ) -> ControllerInputs:
        ready: list[str] = []
        blocked: list[str] = []
        approval_pending: list[str] = []
        timer_hang: list[str] = []
        for node_id, node in graph.nodes().items():
            if node.state == NodeState.READY:
                ready.append(node_id)
            elif node.state == NodeState.BLOCKED:
                blocked.append(node_id)
            elif node.state == NodeState.APPROVAL_HANG:
                approval_pending.append(node_id)
            elif node.state == NodeState.TIMER_HANG:
                timer_hang.append(node_id)
        return ControllerInputs(
            goal="",
            counts_by_state=graph.counts_by_state(),
            ready_nodes=tuple(ready),
            blocked_nodes=tuple(blocked),
            approval_pending=tuple(approval_pending),
            timer_hang=tuple(timer_hang),
            in_flight_leases=self._rt.assignment.count_for_run(run_id),
            all_terminal=graph.all_terminal(),
            has_failed_terminal=graph.has_failed(),
        )

    def _persist_node(
        self,
        run_id: str,
        node: NodeExecution,
        *,
        lease_id: str | None = None,
        require_active_lease: bool = False,
        expected_state: NodeState | str | None = None,
    ) -> _PersistResult:
        if not self._rt.config.persist_to_db:
            return _PersistResult.SUCCESS
        graph = self._graphs.get(run_id)
        # Persist node row.
        spec_dict = node.to_dict()["spec"]
        attempts_dicts = [a.to_dict() for a in node.attempts]
        result_dict = node.result
        failure_dict = node.to_dict()["failure"]
        # Persistence is best-effort: a transient SQLite hiccup
        # ("database disk image is malformed" that the next open
        # recovers from, lock contention from a peer process, etc.)
        # must not crash the engine. The in-memory WorkGraph is the
        # source of truth for the rest of the run; losing a single
        # node-row write only costs us the ability to ``resume()`` from
        # this exact point — the run itself keeps making forward
        # progress.
        try:
            edges = []
            if graph is not None:
                edges = [(dep, node.node_id) for dep in node.spec.depends_on]
            ok = self._rt.graph_store.upsert_node_with_edges(
                run_id, node.node_id,
                state=node.state,
                spec=spec_dict,
                attempts=attempts_dicts,
                result=result_dict,
                failure=failure_dict,
                history=list(node.history),
                edges=edges,
                lease_id=lease_id,
                require_active_lease=require_active_lease,
                expected_state=expected_state,
                refuse_if_terminal=True,
                refuse_if_running_unowned=not require_active_lease,
            )
            return (
                _PersistResult.SUCCESS
                if ok
                else _PersistResult.FENCE_REJECTED
            )
        except sqlite3.DatabaseError as exc:
            logger.warning(
                "engine._persist_node: persistence failed for node=%s "
                "(run_id=%s, state=%s) — continuing with in-memory state: %s",
                node.node_id, run_id, node.state.value, exc,
            )
            return _PersistResult.DB_ERROR

    def _reconstruct_graph(self, run_id: str) -> WorkGraph:
        graph = WorkGraph()
        rows = self._rt.graph_store.list_nodes(run_id)
        # Topologically sort by dep dependency to satisfy add() invariant.
        rows_by_id = {r["node_id"]: r for r in rows}
        added: set[str] = set()

        def add_node(row: dict[str, Any]) -> None:
            if row["node_id"] in added:
                return
            for dep_id in row["spec"].get("depends_on") or ():
                if dep_id in rows_by_id:
                    add_node(rows_by_id[dep_id])
            ne = NodeExecution.from_dict({
                "spec": row["spec"],
                "state": row["state"],
                "attempts": row["attempts"],
                "result": row["result"],
                "failure": row["failure"],
                "history": row.get("history") or [],
                "created_at_ms": row["created_at_ms"],
                "updated_at_ms": row["updated_at_ms"],
            })
            graph.add_execution(ne, validate_deps=False)
            added.add(row["node_id"])

        for row in rows:
            add_node(row)
        return graph

    def _build_verdict(
        self,
        run_id: str,
        goal: str,
        graph: WorkGraph,
        last_step: StepResult | None,
        *,
        error: str | None = None,
    ) -> Verdict:
        counts = graph.counts_by_state()
        succeeded = counts.get("succeeded", 0)
        abandoned = counts.get("abandoned", 0)
        failed = counts.get("failed", 0)
        skipped = counts.get("skipped", 0)
        cancelled = counts.get("cancelled", 0)
        node_count = sum(counts.values())

        halt_reason = (
            self._halt_reasons.get(run_id, "")
            or (
                last_step.halt_reason
                if last_step is not None and last_step.halt_reason
                else ""
            )
        )
        approval_pending = counts.get("approval_hang", 0)

        if error is not None:
            status = RunStatus.FAILED
        elif approval_pending > 0:
            status = RunStatus.WAITING_APPROVAL
        elif abandoned > 0 and succeeded == 0 and node_count > 0:
            status = RunStatus.FAILED
        elif graph.all_terminal():
            # Partial completion (some succeeded, some abandoned) is
            # treated as COMPLETED — the abandoned count remains visible
            # on the Verdict so callers can decide how to react. This is
            # an explicit "best-effort completion" choice; flip to FAILED
            # if your callers must distinguish.
            status = RunStatus.COMPLETED
        elif last_step is not None and last_step.decision_kind == DecisionKind.WAIT:
            status = RunStatus.RUNNING
        elif self._rt.budget.should_halt():
            status = RunStatus.BUDGET_EXHAUSTED
        else:
            status = RunStatus.ABORTED

        budget_snap = self._rt.budget.snapshot()
        summary_parts = []
        if succeeded:
            summary_parts.append(f"{succeeded} succeeded")
        if abandoned:
            summary_parts.append(f"{abandoned} abandoned")
        if skipped:
            summary_parts.append(f"{skipped} skipped")
        if cancelled:
            summary_parts.append(f"{cancelled} cancelled")
        if failed:
            summary_parts.append(f"{failed} failed")
        if approval_pending:
            summary_parts.append(f"{approval_pending} awaiting approval")
        summary = ", ".join(summary_parts) or "no nodes"

        return Verdict(
            run_id=run_id,
            status=status,
            summary=summary,
            node_count=node_count,
            succeeded=succeeded,
            failed=failed,
            abandoned=abandoned,
            skipped=skipped,
            cancelled=cancelled,
            elapsed_s=budget_snap.elapsed_s,
            iterations=budget_snap.iterations,
            budget=self._rt.budget.budget.snapshot(),
            halt_reason=halt_reason,
            error=error,
        )


__all__ = ["EngineV5"]
