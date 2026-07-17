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
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError,
    wait,
)
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
    TERMINAL_NODE_STATES,
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


@dataclass(slots=True)
class _InflightOrphan:
    """A dispatch Future that outlived backstop / budget cancel.

    ``kind`` distinguishes drain behaviour:
    - ``backstop``: synthetic WORKER_LOST already consumed; ignore late result.
    - ``budget``: soft-deferred to READY with parking lease; drain must not
      apply SUCCEEDED and must release the lease after the orphan finishes.
    """

    future: Future[Any]
    kind: str  # "backstop" | "budget"
    parking_stop: threading.Event | None = None
    parking_thread: threading.Thread | None = None


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


class _PersistDbError(RuntimeError):
    """Durable persist under lease failed with a SQLite/backend error.

    Must NOT be treated like a successful write: the dispatcher would otherwise
    release the lease while the DB row still looks runnable, and another
    worker could double-execute. Raised from ``persist_under_lease`` so the
    engine crash path re-attempts an unfenced persist via
    ``_handle_dispatch_result``.
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
        # run_id -> node_ids soft-preempted by parallel budget halt. Workers
        # that have not yet entered RUNNING check this and skip instead of
        # racing a READY→RUNNING after the defer pass.
        self._budget_preempted: dict[str, set[str]] = {}
        # Shared with Runtime: every ``rt.engine()`` must see the same
        # budget/backstop orphans so resume / preempt cannot drop a live
        # parking fence or double-dispatch while an orphan tool still runs.
        self._inflight_orphans = self._rt._engine_inflight_orphans
        self._orphans_lock = self._rt._engine_orphans_lock

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

        Also finalizes budget-orphan parking heartbeats for every run still
        tracked here — same teardown as ``_loop`` finally / full-run
        ``cancel``. Without this, an external teardown that never reaches
        those paths leaves daemon parking threads renewing leases after
        ``resume()`` has already ``drop_for_node``.
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
        with self._orphans_lock:
            orphan_run_ids = list(self._inflight_orphans.keys())
        for rid in orphan_run_ids:
            try:
                self._finalize_inflight_orphans(rid)
            except Exception:
                logger.warning(
                    "EngineV5.shutdown: finalize inflight orphans failed "
                    "for run_id=%s",
                    rid,
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
            self._rt.run_store.persist_budget_snapshot(
                run_id, self._rt.budget.budget.snapshot()
            )
            self._rt.run_store.update_status(
                run_id,
                verdict.status,
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
            self._rt.run_store.persist_budget_snapshot(
                run_id, self._rt.budget.budget.snapshot()
            )
            self._rt.run_store.update_status(
                run_id,
                RunStatus.RUNNING,
            )
        if stored_status not in (RunStatus.RUNNING, RunStatus.WAITING_APPROVAL):
            verdict = self._build_verdict(run_id, run["goal"], graph, None)
            verdict.status = stored_status
            return verdict

        # Keep live budget-orphan parking fences across reclaim — same
        # discipline as ``_step_once``. An unprotected reclaim_expired would
        # DELETE an expired parking row while the shared orphan Future still
        # runs, opening a READY/no-lease window for harness double-run.
        self._reclaim_expired_protecting_parking(run_id)
        # Push RUNNING nodes whose lease was reclaimed back to READY so the
        # new workers can pick them up. Any node still RUNNING without a
        # current lease is ambiguous: mark it FAILED with WORKER_LOST.
        #
        # In the default single-process deployment (no external workers) any
        # surviving lease belongs to the dead prior process, so force-reclaim
        # every RUNNING node NOW rather than leaving it stuck until the lease
        # TTL expires (defect 15). A genuine multi-process topology keeps the
        # TTL slack: a still-live peer may legitimately own the node, so we only
        # reclaim nodes whose lease is already gone.
        force_reclaim = not self._rt.config.expect_external_workers
        for node_id, node in graph.nodes().items():
            if node.state == NodeState.RUNNING:
                # Soft-defer may have failed to durable-write READY while a
                # budget orphan still runs; GraphStore can remain RUNNING with
                # a live parking/engine lease. Do not WORKER_LOST + release —
                # same keep-guard as the READY branch below.
                if self._has_inflight_orphan(run_id, node_id):
                    continue
                lease = self._rt.assignment.find_for(run_id, node_id)
                if lease is None or force_reclaim:
                    if lease is not None:
                        # Release the dead process's lease first so the FAILED
                        # writeback isn't fenced by refuse_if_running_unowned.
                        self._rt.assignment.release(lease.lease_id)
                    self._mark_worker_lost(
                        node,
                        message="resumed: lease lost, prior worker likely died",
                    )
                    self._persist_node(run_id, node)
            elif node.state == NodeState.READY and force_reclaim:
                # Budget soft-defer may have left a parking lease on READY.
                # A *dead* prior process's fence must be dropped so workers can
                # re-lease without waiting out the TTL. But a still-registered
                # inflight orphan (same Runtime, possibly a prior EngineV5)
                # still owns that fence — dropping it lets harness / this
                # engine double-run a non-idempotent tool beside the orphan.
                if self._has_inflight_orphan(run_id, node_id):
                    continue
                self._rt.assignment.drop_for_node(run_id, node_id)

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
            self._rt.run_store.persist_budget_snapshot(
                run_id, self._rt.budget.budget.snapshot()
            )
            self._rt.run_store.update_status(
                run_id,
                RunStatus.BUDGET_EXHAUSTED,
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
        # Same as approve/resume: load durable budget before any persist so a
        # fresh EngineV5 (zero local tracker) cannot overwrite iterations /
        # spend that workers or a prior engine pass already wrote.
        self._rt.budget.restore(run.get("budget"))
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
        # Full-run cancel never re-enters ``_loop``, so tear down budget
        # orphan parking HB / leases here (same as ``_loop`` finally).
        try:
            self._finalize_inflight_orphans(run_id)
        except Exception:
            logger.warning(
                "v5 engine: finalize inflight orphans failed on cancel "
                "for run_id=%s",
                run_id,
                exc_info=True,
            )
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
        self._rt.run_store.persist_budget_snapshot(
            run_id, self._rt.budget.budget.snapshot()
        )
        updated = self._rt.run_store.update_status(
            run_id,
            RunStatus.CANCELLED,
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
        finally:
            # HALT / iteration exhaustion / crash must stop parking keep-alives
            # and drain orphans. Otherwise daemon parking threads outlive the
            # Verdict and can re-grant leases after resume() drop_for_node.
            try:
                self._finalize_inflight_orphans(run_id)
            except Exception:
                logger.warning(
                    "v5 engine: finalize inflight orphans failed for "
                    "run_id=%s",
                    run_id,
                    exc_info=True,
                )

        # Pull any worker spend that landed after the last _step_once absorb
        # so Verdict.budget and the terminal budget_json are not understated.
        self._absorb_remote_budget(run_id)
        verdict = self._build_verdict(run_id, goal, graph, result, error=error)
        self._rt.run_store.persist_budget_snapshot(
            run_id, self._rt.budget.budget.snapshot()
        )
        self._rt.run_store.update_status(
            run_id, verdict.status,
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
            # Refresh + protect_node_ids renews parking fences in place and
            # excludes them from DELETE — closes the READY/no-lease steal
            # window between reclaim and a later re-grant. Uses the parking
            # row's real worker_id (may be ``in-process`` from a prior
            # EngineV5), not only this instance's dispatcher id.
            expired = self._reclaim_expired_protecting_parking(run_id)
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

        self._refresh_parking_leases(run_id)

        self._patrol_stale_running_nodes(run_id, graph)

        # 1b. Merge GraphStore rows into the in-memory WorkGraph so nodes that
        # WorkerHarness (or another process) spawned/finished in SQLite are
        # visible to promote / Controller.decide / _build_verdict.
        self._sync_graph_from_db(run_id, graph)

        # 2. Handle any FAILED nodes before deciding. This catches resume /
        # sweep failures even when no other node is READY.
        self._handle_failures(run_id, graph)

        # 3. Promote PENDING -> READY and persist the promotion so workers can
        # pick up engine-seeded DAGs.
        for node_id in graph.transition_pending_to_ready():
            self._persist_node(run_id, graph.get(node_id))

        # 3b. Absorb worker-reported budget usage from runs.budget_json so
        # Controller.decide / should_halt / dispatch gates see tokens+cost that
        # external WorkerHarness processes wrote via increment_budget_usage.
        self._absorb_remote_budget(run_id)

        # 4. Build inputs and decide.
        inputs = self._build_controller_inputs(run_id, graph)
        decision = self._rt.controller.decide(inputs)

        # Consume one iteration only for a work-doing decision. WAIT idle polls
        # (blocked on slow workers) and HALT steps must not burn iteration
        # budget, or a run bounded only by max_iterations could hit a false
        # BUDGET_EXHAUSTED while merely waiting (defect 11).
        #
        # Ticket is taken *before* dispatch and force-flushed so external
        # WorkerHarness refresh sees ``max_iterations`` mid-batch. The batch
        # that just paid the ticket dispatches with
        # ``ignore_iteration_halt=True`` so ``should_halt`` from this consume
        # cannot zero-start the ready set. Token/cost/wallclock gates still
        # apply. Idle ENQUEUE→WAIT refunds and force-flushes again.
        if decision.kind == DecisionKind.ENQUEUE:
            self._rt.budget.consume(iteration=True)

        # Fire the compaction warning once on crossing — checked EVERY step so a
        # token/cost crossing from the previous step's dispatch is caught even
        # when this step's decision is HALT/WAIT. Periodically persist the
        # in-flight budget so a hard SIGKILL/OOM cannot resurrect already-spent
        # budget on resume; always flush on a warning crossing.
        warning_crossed = self._rt.budget.fire_warning_if_needed()
        if warning_crossed:
            self._rt.event_bus.publish(run_id, "budget.warning",
                                       self._rt.budget.budget.snapshot())

        nodes_started: tuple[str, ...] = ()
        nodes_completed: tuple[str, ...] = ()

        if decision.kind == DecisionKind.HALT:
            self._maybe_flush_budget(run_id, warning_crossed)
            return StepResult(
                iteration=self._rt.budget.budget.iterations,
                decision_kind=decision.kind,
                should_halt=True,
                halt_reason=decision.reason,
            )

        if decision.kind == DecisionKind.WAIT:
            self._maybe_flush_budget(run_id, warning_crossed)
            return StepResult(
                iteration=self._rt.budget.budget.iterations,
                decision_kind=decision.kind,
            )

        if decision.kind == DecisionKind.ENQUEUE:
            # Publish the ticket before dispatch so workers absorb it during
            # the batch window (default flush is every N steps only).
            self._maybe_flush_budget(run_id, warning_crossed, force=True)
            ready = sorted(
                inputs.ready_nodes,
                key=lambda nid: (-graph.get(nid).spec.priority, nid),
            )
            nodes_started, nodes_completed = self._dispatch_batch(
                run_id,
                graph,
                ready,
                ignore_iteration_halt=True,
            )
            # Handle failures from this batch.
            self._handle_failures(run_id, graph)
            idle_waiting = (
                not nodes_started
                and (
                    self._rt.assignment.count_for_run(run_id) > 0
                    or self._run_has_inflight_orphan(run_id)
                )
            )
            if idle_waiting:
                self._rt.budget.refund_iteration()
                # Flush after refund so DB drops the idle ticket. Keep a
                # token/cost warning flush if still in the warning band.
                still_warn = (
                    warning_crossed and self._rt.budget.budget.is_warning()
                )
                self._maybe_flush_budget(run_id, still_warn, force=True)
                return StepResult(
                    iteration=self._rt.budget.budget.iterations,
                    decision_kind=DecisionKind.WAIT,
                )
            self._maybe_flush_budget(run_id, warning_crossed, force=True)

        return StepResult(
            iteration=self._rt.budget.budget.iterations,
            decision_kind=decision.kind,
            nodes_started=nodes_started,
            nodes_completed=nodes_completed,
        )

    def _sync_graph_from_db(self, run_id: str, graph: WorkGraph) -> None:
        """Pull GraphStore rows into the in-memory WorkGraph.

        WorkerHarness may persist ``SpawnResult`` children (and worker-finished
        states) that the engine process never added locally. Without this merge,
        ``Controller.decide`` / ``_build_verdict`` can halt as COMPLETED while
        READY spawned children still exist only in SQLite.

        Existing local nodes are refreshed only when the DB row is ahead —
        never regress a local terminal (e.g. SUCCEEDED after a missed
        ``persist_under_lease``) with a stale RUNNING/READY row.
        """
        if not self._rt.config.persist_to_db:
            return
        try:
            rows = self._rt.graph_store.list_nodes(run_id)
        except sqlite3.DatabaseError as exc:
            logger.warning(
                "engine._sync_graph_from_db: list_nodes failed for run_id=%s: %s",
                run_id,
                exc,
            )
            return
        for row in rows:
            self._refresh_node_from_row(graph, row, allow_regress=False)

    def _absorb_remote_budget(self, run_id: str) -> None:
        """Pull worker-written budget counters from DB into the local tracker.

        Without this, multi-process topologies where WorkerHarness reports cost
        via ``increment_budget_usage`` leave the engine's in-process
        ``BudgetTracker`` at zero for tokens/cost — so ``should_halt()`` and
        dispatch gates never fire on remote spend.
        """
        if not self._rt.config.persist_to_db:
            return
        try:
            run = self._rt.run_store.get(run_id)
        except sqlite3.DatabaseError as exc:
            logger.warning(
                "engine._absorb_remote_budget: failed to read run_id=%s: %s",
                run_id,
                exc,
            )
            return
        if run is None:
            return
        self._rt.budget.absorb_remote(run.get("budget"))

    def _maybe_flush_budget(
        self,
        run_id: str,
        warning_crossed: bool,
        *,
        force: bool = False,
    ) -> None:
        """Persist the in-flight budget snapshot mid-loop via the monotonic
        merge (never clobbers concurrent worker deltas). Flushes on a warning
        crossing, every ``budget_flush_every_n_iterations`` iterations, or
        when ``force=True`` (ENQUEUE ticket visibility for external workers).
        """
        if not self._rt.config.persist_to_db:
            return
        every_n = self._rt.config.budget_flush_every_n_iterations
        iterations = self._rt.budget.budget.iterations
        if force or warning_crossed or (every_n > 0 and iterations % every_n == 0):
            self._rt.run_store.persist_budget_snapshot(
                run_id, self._rt.budget.budget.snapshot()
            )

    def _run_has_inflight_orphan(self, run_id: str) -> bool:
        """True while any orphan Future is still registered for this run."""
        with self._orphans_lock:
            by_node = self._inflight_orphans.get(run_id)
            return bool(by_node)

    # -- dispatch ------------------------------------------------------------

    def _dispatch_batch(
        self,
        run_id: str,
        graph: WorkGraph,
        ready: list[str],
        *,
        ignore_iteration_halt: bool = False,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if not ready:
            return ((), ())
        config = self._rt.config
        dispatcher = self._dispatchers[run_id]
        started: list[str] = []
        completed: list[str] = []
        self._reap_inflight_orphans(run_id)

        def _budget_blocks() -> bool:
            return self._rt.budget.should_halt(
                ignore_iterations=ignore_iteration_halt
            )

        # A prior parallel budget halt may have marked nodes preempted. Once
        # the budget allows dispatch again, clear marks for this ready set so
        # they can run — but never while an orphan future is still in flight.
        if not _budget_blocks():
            marked = self._budget_preempted.get(run_id)
            if marked:
                for nid in ready:
                    if not self._has_inflight_orphan(run_id, nid):
                        marked.discard(nid)
        if config.parallelism <= 1:
            for node_id in ready:
                if _budget_blocks():
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
                if _budget_blocks():
                    break
                if self._has_inflight_orphan(run_id, node_id):
                    continue
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
                budget_halted_ids: list[str] = []
                try:
                    # Collect by completion, not submit order. An unbounded
                    # future (timeout=None) must not stall siblings that are
                    # already done — zip+fut.result(None) would do exactly that.
                    futures = {
                        ex.submit(dispatcher.dispatch_one, n): n
                        for n in dispatchable
                    }
                    started_at = {
                        n: time.monotonic() for n in dispatchable
                    }
                    pending = set(futures)

                    def _consume_result(n: str, res: DispatchResult) -> None:
                        try:
                            self._handle_dispatch_result(run_id, graph, res)
                        except _PersistDbError:
                            # Durable write missed — same as serial batch: surface
                            # so the run does not pretend the node completed.
                            raise
                        except Exception:
                            # A late worker thread the backstop could not stop
                            # (shutdown(wait=False) cannot kill it) can race
                            # _force_fail_node on the same NodeExecution and raise
                            # an illegal-transition error. That must not reach the
                            # outer handler and abort every other in-flight/pending
                            # node — the node self-heals on the next _step_once.
                            # (The per-node lock makes torn state rare; this is
                            # the belt-and-braces so one node's race is contained.)
                            logger.warning(
                                "v5 engine: dispatch-result handling raced for "
                                "node=%s (run_id=%s); skipping, self-heals next "
                                "step", n, run_id, exc_info=True,
                            )
                            return
                        if not res.skipped:
                            started.append(n)
                            completed.append(n)

                    while pending:
                        # Match serial batch: once the budget is exhausted, do
                        # not keep waiting on (or completing) siblings that are
                        # still in flight — they would continue spending.
                        # Iteration-only exhaustion is ignored when this batch
                        # already paid the ENQUEUE ticket.
                        if _budget_blocks():
                            # Single pass: cancel still-queued work, consume any
                            # sibling that already finished (including races
                            # during this loop), and soft-defer only true
                            # in-flight orphans. A successful cancel() means
                            # the worker never started — do not treat
                            # CancelledError as a node failure.
                            for fut in list(pending):
                                n = futures[fut]
                                pending.discard(fut)
                                if not fut.done():
                                    if fut.cancel():
                                        budget_halted_ids.append(n)
                                        self._budget_preempted.setdefault(
                                            run_id, set()
                                        ).add(n)
                                        self._defer_for_budget_halt(
                                            run_id, graph, n
                                        )
                                        continue
                                if fut.done() and not fut.cancelled():
                                    try:
                                        res = fut.result(timeout=0)
                                    except FutureTimeoutError as exc:
                                        res = self._dispatch_timed_out_result(
                                            graph, n, exc
                                        )
                                    except Exception as exc:
                                        res = self._dispatch_crashed_result(
                                            graph, n, exc
                                        )
                                    _consume_result(n, res)
                                    continue
                                budget_halted_ids.append(n)
                                # Mark preempted before registering the
                                # orphan so retain_lease_check cannot miss
                                # the register→defer window.
                                self._budget_preempted.setdefault(
                                    run_id, set()
                                ).add(n)
                                self._register_inflight_orphan(
                                    run_id, n, fut, kind="budget"
                                )
                                # Soft-preempt: discard open attempt, restore
                                # READY. Parking lease is kept while the orphan
                                # future is still running so external harnesses
                                # cannot re-lease. Do not synthesize
                                # FAILED/BUDGET_EXHAUSTED — that burned attempt
                                # slots and raced orphan workers.
                                self._defer_for_budget_halt(run_id, graph, n)
                            # A cancel can race a worker that has not yet
                            # flipped READY→RUNNING; re-defer any that landed
                            # in RUNNING after the first pass.
                            for n in budget_halted_ids:
                                self._defer_for_budget_halt(run_id, graph, n)
                            break

                        now = time.monotonic()
                        overdue: list[tuple[Any, str]] = []
                        wait_timeout: float | None = None
                        for fut in pending:
                            n = futures[fut]
                            backstop = self._future_timeout_for(graph, n)
                            if backstop is None:
                                continue
                            remaining = backstop - (now - started_at[n])
                            if remaining <= 0:
                                overdue.append((fut, n))
                            elif (
                                wait_timeout is None
                                or remaining < wait_timeout
                            ):
                                wait_timeout = remaining

                        if overdue:
                            for fut, n in overdue:
                                pending.discard(fut)
                                if fut.done():
                                    try:
                                        res = fut.result(timeout=0)
                                    except Exception as exc:
                                        res = self._dispatch_crashed_result(
                                            graph, n, exc
                                        )
                                else:
                                    # Backstop: worker overran its own deadline.
                                    res = self._dispatch_timed_out_result(
                                        graph, n, FutureTimeoutError()
                                    )
                                    self._register_inflight_orphan(
                                        run_id, n, fut, kind="backstop"
                                    )
                                _consume_result(n, res)
                            continue

                        done, pending = wait(
                            pending,
                            timeout=wait_timeout,
                            return_when=FIRST_COMPLETED,
                        )
                        if not done:
                            # Bounded wait elapsed with no completion — loop
                            # again so overdue nodes are classified.
                            continue
                        for fut in done:
                            n = futures[fut]
                            try:
                                res = fut.result(timeout=0)
                            except FutureTimeoutError as exc:
                                res = self._dispatch_timed_out_result(
                                    graph, n, exc
                                )
                            except Exception as exc:
                                res = self._dispatch_crashed_result(
                                    graph, n, exc
                                )
                            _consume_result(n, res)
                except BaseException:
                    # SystemExit / KeyboardInterrupt: cancel queued work before
                    # unwinding so __exit__-style wait=True joins never happen.
                    ex.shutdown(wait=False, cancel_futures=True)
                    raise
                finally:
                    # Last chance: workers that raced past the first defer pass
                    # (including pre-RUNNING → RUNNING) must still be soft-
                    # preempted before the pool is abandoned.
                    for n in budget_halted_ids:
                        self._defer_for_budget_halt(run_id, graph, n)
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
        if isinstance(exc, (_LeaseLostError, _PersistDbError)):
            # Lease lost / durable write missed. Retryable so a still-RUNNING
            # node can be re-driven. If memory already reached SUCCEEDED,
            # _handle_dispatch_result skips force-fail and retries an unfenced
            # persist of that success + SpawnResult children recovered from
            # the in-memory graph (exception paths have no spawned_node_ids).
            node = graph.get(node_id)
            spawned = (
                self._spawned_child_ids(graph, node_id)
                if node.state == NodeState.SUCCEEDED
                else ()
            )
            return DispatchResult(
                node_id=node_id,
                final_state=node.state,
                result=node.result,
                failure=Failure(
                    kind=FailureKind.WORKER_LOST,
                    message=str(exc),
                    retryable=True,
                ),
                spawned_node_ids=spawned,
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
        # If memory already reached SUCCEEDED (tool finished; backstop raced
        # the result), recover SpawnResult children like _dispatch_crashed_result
        # so _handle_dispatch_result can durable-write them.
        node = graph.get(node_id)
        spawned = (
            self._spawned_child_ids(graph, node_id)
            if node.state == NodeState.SUCCEEDED
            else ()
        )
        return DispatchResult(
            node_id=node_id,
            final_state=node.state,
            result=node.result,
            failure=Failure(
                kind=FailureKind.WORKER_LOST,
                message=f"dispatch future backstop timeout: {exc}",
                retryable=True,
            ),
            spawned_node_ids=spawned,
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
        self,
        graph: WorkGraph,
        row: dict[str, Any],
        *,
        allow_regress: bool = True,
    ) -> None:
        node = self._node_from_row(row)
        if graph.has(node.node_id):
            if not allow_regress and not self._db_row_is_ahead(
                graph.get(node.node_id), row
            ):
                return
            graph.replace_execution(node, validate_deps=False)
        else:
            graph.add_execution(node, validate_deps=False)

    @staticmethod
    def _db_row_is_ahead(local: NodeExecution, row: dict[str, Any]) -> bool:
        """Return True when a GraphStore row should replace local execution.

        Protects against the persist-gap race: memory already advanced to a
        terminal state while SQLite still holds an older RUNNING/READY row
        after ``DB_ERROR``. Worker-finished terminals and newer timestamps
        still win so external harness progress is adopted.
        """
        try:
            db_state = NodeState(row["state"])
        except (KeyError, ValueError):
            return False
        db_ts = int(row.get("updated_at_ms") or 0)
        local_ts = int(local.updated_at_ms or 0)
        local_terminal = local.state in TERMINAL_NODE_STATES
        db_terminal = db_state in TERMINAL_NODE_STATES
        if local_terminal and not db_terminal:
            return False
        if db_terminal and not local_terminal:
            return True
        return db_ts >= local_ts

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
            # Soft-defer may leave GraphStore RUNNING while a budget orphan
            # still executes (READY durable-write failed). ``find_for`` ignores
            # expired parking rows, so without this keep-guard patrol would
            # false-WORKER_LOST and later re-dispatch beside the orphan.
            if self._has_inflight_orphan(run_id, node_id):
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
        # A failure result must never overwrite an already-terminal node
        # (defect 4). SUCCEEDED is excluded here so a failure carrying
        # final_state=SUCCEEDED — e.g. a _LeaseLostError raised AFTER the node
        # already succeeded, or a backstop timeout racing a just-finished
        # worker — cannot flip a genuinely-completed node to FAILED. The
        # unconditional _persist_node below is still fenced
        # (refuse_if_terminal / refuse_if_running_unowned), so it never commits
        # the local result over a real competitor's DB state.
        if result.failure is not None and node.state not in (
            NodeState.FAILED,
            NodeState.ABANDONED,
            NodeState.SUCCEEDED,
        ):
            self._force_fail_node(node, result.failure)
        # A wedged in-process worker may still hold (and heartbeat) its lease
        # after a future backstop / budget-halt force-fail. Release our own
        # lease first so refuse_if_running_unowned cannot FENCE_REJECT the
        # FAILED write — otherwise allow_regress would wipe WORKER_LOST and
        # leave the node RUNNING forever.
        if result.failure is not None and node.state == NodeState.FAILED:
            self._release_own_node_lease(run_id, result.node_id)
        persist = self._persist_node(run_id, node)
        if persist == _PersistResult.DB_ERROR:
            raise _PersistDbError(
                f"dispatch result persist failed for {node.node_id} "
                f"(run_id={run_id}, state={node.state.value})"
            )
        if persist == _PersistResult.FENCE_REJECTED:
            # Retry once after another own-lease release (heartbeat race).
            if result.failure is not None and node.state == NodeState.FAILED:
                self._release_own_node_lease(run_id, result.node_id)
                persist = self._persist_node(run_id, node)
                if persist == _PersistResult.DB_ERROR:
                    raise _PersistDbError(
                        f"dispatch result persist failed for {node.node_id} "
                        f"(run_id={run_id}, state={node.state.value})"
                    )
            if persist == _PersistResult.FENCE_REJECTED:
                # Competitor / reclaim owns durable state. Adopt DB truth (even
                # if that regresses a local SUCCEEDED) and do not emit
                # node.completed for a success that never landed.
                if self._rt.config.persist_to_db:
                    row = self._rt.graph_store.get_node(run_id, result.node_id)
                    if row is not None:
                        self._refresh_node_from_row(
                            graph, row, allow_regress=True
                        )
                return
        if result.final_state == NodeState.ABANDONED:
            for changed in graph.nodes().values():
                if changed.state == NodeState.SKIPPED:
                    skipped_persist = self._persist_node(run_id, changed)
                    if skipped_persist == _PersistResult.DB_ERROR:
                        raise _PersistDbError(
                            f"skipped-node persist failed for {changed.node_id} "
                            f"(run_id={run_id})"
                        )
        # Persist any children spawned via SpawnResult during this dispatch
        # (idempotent with the earlier on_spawned hook).
        self._persist_spawned_children(run_id, graph, result.spawned_node_ids)
        # Only announce completion when the durable write of SUCCEEDED landed.
        if (
            result.final_state == NodeState.SUCCEEDED
            and node.state == NodeState.SUCCEEDED
        ):
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
                # Never null an existing result here: a failure result must not
                # erase work a node already produced (defect 4). The guard in
                # _handle_dispatch_result also excludes SUCCEEDED so this branch
                # is not reached for an already-succeeded node.
                node.updated_at_ms = now_ms()
                return
        if node.state == NodeState.RUNNING:
            node.transition(NodeState.FAILED, reason=failure.message[:80])

    # -- failure handling / replan ------------------------------------------

    def _handle_failures(self, run_id: str, graph: WorkGraph) -> None:
        self._reap_inflight_orphans(run_id)
        for node_id, node in list(graph.nodes().items()):
            if node.state != NodeState.FAILED:
                continue
            if node.failure is None:
                # Defensive: synthesize a generic failure.
                node.failure = Failure(
                    kind=FailureKind.UNKNOWN,
                    message="failed without failure record",
                )
            # Parallel mid-batch halt marks unfinished siblings BUDGET_EXHAUSTED
            # (legacy path) or soft-defers via `_defer_for_budget_halt`. Either
            # way this is a run-level pause: restore READY without burning
            # attempt slots, and never replan/abandon (would cascade SKIPPED
            # and consume replan counters while the budget is already exhausted).
            if node.failure.kind == FailureKind.BUDGET_EXHAUSTED:
                node.discard_last_attempt_with_failure(
                    FailureKind.BUDGET_EXHAUSTED
                )
                node.transition(
                    NodeState.READY,
                    reason="budget exhausted; deferred until budget allows",
                )
                node.failure = None
                persist = self._persist_node(run_id, node)
                if persist != _PersistResult.SUCCESS:
                    self._recover_ready_persist_failure(
                        run_id, graph, node_id, persist
                    )
                continue
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
                # Parallel backstop / cancelled futures leave the worker thread
                # running after WORKER_LOST. Do not READY-retry until that
                # orphan future completes — otherwise a second dispatch_one
                # overlaps the first on the shared WorkGraph.
                if (
                    node.failure.kind == FailureKind.WORKER_LOST
                    and self._has_inflight_orphan(run_id, node_id)
                ):
                    continue
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

            if added:
                # Only count a replan that actually applied at least one spec.
                # A replan that returned nothing, or whose specs were all skipped
                # (same-id reuse cap already hit, or already present), is a
                # no-op and must not consume the run-wide or per-node replan
                # budget (defect 12).
                self._replan_run_totals[run_id] = (
                    self._replan_run_totals.get(run_id, 0) + 1
                )
                if scope.level == ScopeLevel.LOCAL:
                    replan_state.local_used += 1
                else:
                    replan_state.global_used += 1
                self._persist_replan_state(run_id)

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

    @staticmethod
    def _spawned_child_ids(graph: WorkGraph, parent_id: str) -> tuple[str, ...]:
        """Return in-memory children tagged with ``parent_node_id`` metadata."""
        out: list[str] = []
        for node_id, node in graph.nodes().items():
            meta = node.spec.metadata or {}
            if meta.get("parent_node_id") == parent_id:
                out.append(node_id)
        return tuple(out)

    def _persist_spawned_children(
        self,
        run_id: str,
        graph: WorkGraph,
        spawned_ids: tuple[str, ...],
    ) -> None:
        """Durably write SpawnResult children (and promote deps-satisfied ones).

        Raises ``_PersistDbError`` on SQLite/backend failure so ``on_spawned``
        aborts before parent finish-persist, and the post-dispatch path does
        not treat a missing child row as success.
        """
        if not spawned_ids:
            return
        for child_id in spawned_ids:
            try:
                child = graph.get(child_id)
            except KeyError:
                continue
            self._require_spawn_persist(
                run_id, child, self._persist_node(run_id, child)
            )
        for node_id in graph.transition_pending_to_ready():
            if node_id not in spawned_ids:
                continue
            child = graph.get(node_id)
            self._require_spawn_persist(
                run_id, child, self._persist_node(run_id, child)
            )

    @staticmethod
    def _require_spawn_persist(
        run_id: str,
        node: NodeExecution,
        result: _PersistResult,
    ) -> None:
        if result == _PersistResult.SUCCESS:
            return
        if result == _PersistResult.DB_ERROR:
            raise _PersistDbError(
                f"spawn persist failed for {node.node_id} "
                f"(run_id={run_id}, state={node.state.value})"
            )
        # FENCE_REJECTED: refuse to let the parent finish while a child
        # write did not land (e.g. competitor already owns the row).
        raise _PersistDbError(
            f"spawn persist refused for {node.node_id} "
            f"(run_id={run_id}, state={node.state.value}, result={result.value})"
        )

    # -- helpers -------------------------------------------------------------

    def _make_dispatcher(self, run_id: str, graph: WorkGraph) -> Dispatcher:
        def emit(kind: str, payload: dict[str, Any]) -> None:
            self._rt.event_bus.publish(run_id, kind, payload)
        def report_cost(tokens: int, cost: float) -> None:
            # Multi-process: WorkerHarness gates leases on runs.budget_json.
            # Write the shared DB first (same as harness), then mirror into the
            # local tracker so concurrent workers see engine spend promptly —
            # not only on the next periodic _maybe_flush_budget.
            # On DB failure / missing run: do NOT fall back to local consume —
            # that would inflate the in-process tracker while shared budget
            # stays understated and workers keep leasing past the cap.
            if not self._rt.config.persist_to_db:
                self._rt.budget.consume(tokens=tokens, cost=cost)
                return
            elapsed_s = self._rt.budget.snapshot().elapsed_s
            try:
                result = self._rt.run_store.increment_budget_usage(
                    run_id,
                    tokens=tokens,
                    cost=cost,
                    elapsed_s=elapsed_s,
                )
            except sqlite3.DatabaseError:
                logger.warning(
                    "engine.report_cost: increment_budget_usage failed for "
                    "run_id=%s; leaving local budget unchanged",
                    run_id,
                    exc_info=True,
                )
                return
            if result is None:
                return
            self._rt.budget.absorb_remote(result.budget)
            if result.warning_crossed:
                # Claim the once-only flag here. Publishing without arming
                # ``_warning_fired`` lets the next ``_step_once`` fire
                # ``fire_warning_if_needed`` again → duplicate budget.warning
                # and a second compaction for the same crossing.
                if self._rt.budget.fire_warning_if_needed():
                    self._rt.event_bus.publish(
                        run_id, "budget.warning", result.budget
                    )

        def persist_spawned(spawned_ids: tuple[str, ...]) -> None:
            self._persist_spawned_children(run_id, graph, spawned_ids)

        def persist_under_lease(node: NodeExecution, lease_id: str) -> None:
            result = self._persist_node(
                run_id,
                node,
                lease_id=lease_id,
                require_active_lease=True,
            )
            if result == _PersistResult.SUCCESS:
                return
            if result == _PersistResult.DB_ERROR:
                # Must not look like success: a silent return lets the
                # dispatcher release the lease while the DB row is still
                # RUNNING/READY for another worker to re-execute.
                raise _PersistDbError(
                    f"persist under lease failed for {node.node_id} "
                    f"(run_id={run_id}, state={node.state.value})"
                )
            # FENCE_REJECTED: the lease no longer owns the node. This is
            # frequently a FALSE POSITIVE: the in-process engine blocks on the
            # dispatch future and reclaims nothing mid-flight, so a healthily-
            # running node can have its lease expire purely by wall-clock (a
            # briefly-starved heartbeat) with NO competing worker. Discarding
            # genuinely-completed work in that case is the bug. Try to SALVAGE:
            # re-persist WITHOUT the lease fence, but ONLY if the DB row is
            # still RUNNING. expected_state=RUNNING refuses writes over
            # FAILED/READY that a reclaim / worker-loss recovery already
            # committed; refuse_if_terminal still blocks overwriting
            # SUCCEEDED/ABANDONED/etc.; refuse_if_running_unowned still blocks
            # when a live competitor holds the lease.
            salvage = self._persist_node(
                run_id, node, expected_state=NodeState.RUNNING
            )
            if salvage == _PersistResult.SUCCESS:
                return
            if salvage == _PersistResult.DB_ERROR:
                raise _PersistDbError(
                    f"salvage persist failed for {node.node_id} "
                    f"(run_id={run_id}, state={node.state.value})"
                )
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
            on_spawned=persist_spawned,
            budget_reporter=report_cost,
            interaction_fn=self._rt.interaction_fn,
            preempt_check=lambda nid: (
                nid in self._budget_preempted.get(run_id, ())
                or self._has_inflight_orphan(run_id, nid)
            ),
            # Only budget soft-defer retains the lease past dispatch_one;
            # backstop orphans must still release (WORKER_LOST already applied).
            # Also retain while a budget orphan is registered but
            # ``_budget_preempted`` has not been written yet (register → defer
            # gap): otherwise finally releases the parking fence and
            # WorkerHarness can double-run.
            retain_lease_check=lambda nid: (
                nid in self._budget_preempted.get(run_id, ())
                or self._has_budget_inflight_orphan(run_id, nid)
            ),
        )

    def _register_inflight_orphan(
        self,
        run_id: str,
        node_id: str,
        fut: Future[Any],
        *,
        kind: str,
    ) -> None:
        """Remember a dispatch Future that outlived backstop / budget cancel."""
        parking_stop: threading.Event | None = None
        parking_thread: threading.Thread | None = None
        if kind == "budget":
            parking_stop, parking_thread = self._start_parking_heartbeat(
                run_id, node_id, fut
            )
        with self._orphans_lock:
            self._inflight_orphans.setdefault(run_id, {})[node_id] = (
                _InflightOrphan(
                    future=fut,
                    kind=kind,
                    parking_stop=parking_stop,
                    parking_thread=parking_thread,
                )
            )
        # Budget HALT / loop exit may leave READY orphans with no further
        # ``_step_once`` reap. When the Future completes, drain so parking HB
        # and the shared registry entry do not pin the node forever.
        fut.add_done_callback(
            lambda _f, rid=run_id: self._schedule_orphan_reap(rid)
        )

    def _schedule_orphan_reap(self, run_id: str) -> None:
        """Drain finished orphans from a Future done-callback (any thread)."""
        try:
            self._reap_inflight_orphans(run_id)
        except Exception:
            logger.warning(
                "v5 engine: orphan done-callback reap failed for run_id=%s",
                run_id,
                exc_info=True,
            )

    def _start_parking_heartbeat(
        self,
        run_id: str,
        node_id: str,
        _fut: Future[Any],
    ) -> tuple[threading.Event, threading.Thread]:
        """Keep the parking lease alive until drain sets ``stop``.

        Do not exit when the orphan Future completes: ``dispatch_one`` may
        finish before ``_reap_inflight_orphans`` clears the orphan entry.
        The fence must hold across that gap (retain_lease_check + this loop).
        """
        stop = threading.Event()
        interval_s = max(
            self._rt.assignment.heartbeat_interval_ms / 1000.0, 0.01
        )

        def loop() -> None:
            # Renew immediately: dispatch_one's finally may stop the main
            # heartbeat as soon as retain_lease_check parks the fence, and
            # waiting a full interval first can let the TTL lapse before the
            # first parking renew — opening a READY/no-lease steal window.
            while True:
                self._refresh_one_parking_lease(run_id, node_id)
                if stop.wait(interval_s):
                    return

        t = threading.Thread(
            target=loop,
            daemon=True,
            name=f"park-hb-{node_id[:12]}",
        )
        t.start()
        return stop, t

    def _engine_worker_id(self, run_id: str) -> str:
        dispatcher = self._dispatchers.get(run_id)
        return dispatcher.worker_id if dispatcher is not None else "in-process"

    @staticmethod
    def _is_engine_parking_worker(worker_id: str) -> bool:
        """True for engine / in-process parking owners (not harness workers)."""
        return worker_id == "in-process" or worker_id.startswith("engine-")

    def _parking_protect_worker_id(
        self, run_id: str, protect: tuple[str, ...]
    ) -> str:
        """Worker id used to renew protected parking rows during reclaim.

        Prefer the lease row's owner: a prior EngineV5 may have granted the
        fence as ``in-process`` or a different ``engine-*`` id. Using only
        this instance's dispatcher id makes owner-scoped renew silently fail
        and leaves an expired-but-excluded fence until heartbeat goes stale.
        """
        for nid in protect:
            row = self._rt.assignment.find_row(run_id, nid)
            if row is not None:
                return row.worker_id
        return self._engine_worker_id(run_id)

    def _reclaim_expired_protecting_parking(self, run_id: str) -> list:
        """Reclaim expired leases while keeping live budget-orphan parking.

        Refreshes parking first, then ``reclaim_expired`` with
        ``protect_node_ids`` + owner-scoped ``protect_worker_id`` so an
        expired orphan fence is renewed in place instead of DELETE'd.
        """
        self._refresh_parking_leases(run_id)
        protect = self._budget_parking_node_ids(run_id)
        if not protect:
            return self._rt.assignment.reclaim_expired(
                now=now_ms(), run_id=run_id
            )
        return self._rt.assignment.reclaim_expired(
            now=now_ms(),
            run_id=run_id,
            protect_node_ids=protect,
            protect_worker_id=self._parking_protect_worker_id(run_id, protect),
        )

    def _refresh_one_parking_lease(self, run_id: str, node_id: str) -> None:
        """Heartbeat or revive the engine parking lease for one budget orphan.

        Prefer in-place renew (works even when the row is already expired) so
        ``grant_lease`` cannot DELETE-then-steal an expired parking fence.
        Only grant a new lease when no row exists at all.

        Renew is owner-scoped: never extend a WorkerHarness-stolen lease.
        When the row belongs to a prior EngineV5 / ``in-process`` owner and
        this node is still a registered budget orphan, renew as that owner
        so heartbeat cannot go stale under a mismatched dispatcher id.
        """
        worker_id = self._engine_worker_id(run_id)
        try:
            if self._rt.assignment.renew_for_node(
                run_id, node_id, worker_id=worker_id
            ):
                return
        except Exception:
            logger.warning(
                "v5 engine: parking renew failed for node=%s "
                "(run_id=%s); will try grant",
                node_id,
                run_id,
                exc_info=True,
            )
        existing = self._rt.assignment.find_row(run_id, node_id)
        if existing is not None and existing.worker_id != worker_id:
            # Keep a prior engine/in-process fence alive for our orphan;
            # never extend a harness-stolen row.
            if (
                self._has_budget_inflight_orphan(run_id, node_id)
                and self._is_engine_parking_worker(existing.worker_id)
            ):
                try:
                    self._rt.assignment.renew_for_node(
                        run_id, node_id, worker_id=existing.worker_id
                    )
                except Exception:
                    logger.warning(
                        "v5 engine: parking owner-renew failed for "
                        "node=%s (run_id=%s, owner=%s)",
                        node_id,
                        run_id,
                        existing.worker_id,
                        exc_info=True,
                    )
            return
        try:
            self._rt.assignment.lease(run_id, node_id, worker_id)
        except Exception:
            logger.warning(
                "v5 engine: parking re-grant failed for node=%s "
                "(run_id=%s)",
                node_id,
                run_id,
                exc_info=True,
            )

    def _budget_parking_node_ids(self, run_id: str) -> tuple[str, ...]:
        """Budget-orphan node ids still awaiting drain (including fut.done())."""
        with self._orphans_lock:
            by_node = self._inflight_orphans.get(run_id) or {}
            return tuple(
                nid
                for nid, orphan in by_node.items()
                if orphan.kind == "budget"
            )

    def _refresh_parking_leases(self, run_id: str) -> None:
        """Extend / re-grant parking leases for all live budget orphans."""
        for nid in self._budget_parking_node_ids(run_id):
            self._refresh_one_parking_lease(run_id, nid)

    def _reap_inflight_orphans(self, run_id: str) -> None:
        """Drain finished orphan Futures (result handling + lease cleanup)."""
        self._refresh_parking_leases(run_id)
        with self._orphans_lock:
            by_node = self._inflight_orphans.get(run_id)
            if not by_node:
                return
            done = [
                (nid, orphan)
                for nid, orphan in list(by_node.items())
                if orphan.future.done()
            ]
            for nid, _ in done:
                by_node.pop(nid, None)
            if not by_node:
                self._inflight_orphans.pop(run_id, None)
        graph = self._graphs.get(run_id)
        for nid, orphan in done:
            self._drain_inflight_orphan(run_id, graph, nid, orphan)

    def _finalize_inflight_orphans(self, run_id: str) -> None:
        """Drain finished orphans; tear down only when safe for still-running ones.

        Called from ``_loop`` finally, full-run ``cancel``, and ``shutdown``.

        Done futures are drained normally. Still-running orphans whose node is
        already terminal (e.g. CANCELLED) drop parking HB / lease and leave the
        map — they cannot be re-leased as READY. Still-running budget orphans
        left on READY (soft-defer after budget halt) **keep** their map entry
        and parking fence until the Future's done-callback runs
        ``_reap_inflight_orphans``; clearing them early let ``preempt_check`` /
        harness re-lease the same node while the orphan tool thread was still
        executing.

        Do not call ``_reap_inflight_orphans`` here: that path refreshes parking
        leases first and would re-grant fences we tear down on the terminal path.
        """
        with self._orphans_lock:
            by_node = self._inflight_orphans.get(run_id)
            if not by_node:
                return
            items = list(by_node.items())
        graph = self._graphs.get(run_id)
        for nid, orphan in items:
            node = None
            if graph is not None:
                try:
                    node = graph.get(nid)
                except KeyError:
                    node = None
            terminal = (
                node is not None and node.state in TERMINAL_NODE_STATES
            )
            if orphan.future.done():
                with self._orphans_lock:
                    cur = self._inflight_orphans.get(run_id)
                    if cur is not None:
                        cur.pop(nid, None)
                        if not cur:
                            self._inflight_orphans.pop(run_id, None)
                self._drain_inflight_orphan(run_id, graph, nid, orphan)
            elif terminal:
                # Cancelled / abandoned / etc.: stop forever-renew and drop the
                # fence; the node is not READY so harness will not steal.
                if orphan.parking_stop is not None:
                    orphan.parking_stop.set()
                if orphan.parking_thread is not None:
                    orphan.parking_thread.join(timeout=2.0)
                with self._orphans_lock:
                    cur = self._inflight_orphans.get(run_id)
                    if cur is not None:
                        cur.pop(nid, None)
                        if not cur:
                            self._inflight_orphans.pop(run_id, None)
                self._release_own_node_lease(run_id, nid)
            # else: READY (or unknown) + still running — leave orphan + parking
            # HB; the Future done-callback will reap when the tool finishes.

    def _has_inflight_orphan(self, run_id: str, node_id: str) -> bool:
        """True while an orphan entry exists (including done-but-not-yet-drained)."""
        with self._orphans_lock:
            by_node = self._inflight_orphans.get(run_id)
            return bool(by_node and node_id in by_node)

    def _has_budget_inflight_orphan(self, run_id: str, node_id: str) -> bool:
        """True while a *budget* orphan entry exists (not backstop)."""
        with self._orphans_lock:
            by_node = self._inflight_orphans.get(run_id)
            if not by_node:
                return False
            orphan = by_node.get(node_id)
            return orphan is not None and orphan.kind == "budget"

    def _drain_inflight_orphan(
        self,
        run_id: str,
        graph: WorkGraph | None,
        node_id: str,
        orphan: _InflightOrphan,
    ) -> None:
        """Consume a finished orphan Future without re-applying success."""
        if orphan.parking_stop is not None:
            orphan.parking_stop.set()
        if orphan.parking_thread is not None:
            orphan.parking_thread.join(timeout=2.0)
        res: DispatchResult | None = None
        try:
            res = orphan.future.result(timeout=0)
        except Exception:
            if orphan.kind == "budget":
                logger.warning(
                    "v5 engine: budget-orphan future raised for node=%s "
                    "(run_id=%s); ignoring after soft-defer",
                    node_id,
                    run_id,
                    exc_info=True,
                )
            # backstop: synthetic WORKER_LOST already applied — ignore late crash.
        if orphan.kind == "backstop":
            return
        # Budget soft-defer: never let a late non-skipped result flip READY
        # to SUCCEEDED. Dispatcher normally returns skipped; this is belt-and-
        # braces if a race produced a success-shaped DispatchResult.
        if (
            graph is not None
            and isinstance(res, DispatchResult)
            and not res.skipped
            and res.failure is None
        ):
            try:
                node = graph.get(node_id)
            except KeyError:
                node = None
            if node is not None and node.state == NodeState.READY:
                logger.warning(
                    "v5 engine: ignoring late non-skipped budget-orphan "
                    "result for node=%s (run_id=%s, final_state=%s)",
                    node_id,
                    run_id,
                    res.final_state,
                )
        # Parking lease: orphan dispatch_one.finally should have released; if
        # the future was cancelled before acquire or release raced, drop ours.
        self._release_own_node_lease(run_id, node_id)

    def _recover_ready_persist_failure(
        self,
        run_id: str,
        graph: WorkGraph,
        node_id: str,
        persist: _PersistResult,
    ) -> None:
        """Undo a divergent in-memory READY after a failed defer persist.

        Memory may already show READY with attempts discarded; if the durable
        write missed, reload DB truth so patrol does not invent WORKER_LOST
        over a stale RUNNING row and burn ``max_attempts``.
        """
        if persist == _PersistResult.DB_ERROR:
            if self._rt.config.persist_to_db:
                try:
                    row = self._rt.graph_store.get_node(run_id, node_id)
                except Exception:
                    row = None
                if row is not None:
                    self._refresh_node_from_row(
                        graph, row, allow_regress=True
                    )
            raise _PersistDbError(
                f"ready-defer persist failed for {node_id} "
                f"(run_id={run_id})"
            )
        if persist == _PersistResult.FENCE_REJECTED:
            if self._rt.config.persist_to_db:
                row = self._rt.graph_store.get_node(run_id, node_id)
                if row is not None:
                    self._refresh_node_from_row(
                        graph, row, allow_regress=True
                    )
            return

    def _release_own_node_lease(self, run_id: str, node_id: str) -> None:
        """Drop the lease held by this engine's dispatcher for ``node_id``.

        Used before persisting a local FAILED after a backstop / budget halt so
        ``refuse_if_running_unowned`` does not reject the write, and so orphan
        in-process workers see the lease gone and skip claiming SUCCEEDED.
        Leases owned by other workers are left alone when
        ``expect_external_workers`` is set.

        Also clears *expired* parking rows: ``find_for`` / ``release`` ignore
        them, but WorkerHarness would otherwise forever renew READY+expired
        fences and starve re-lease after drain or a dead prior process.
        """
        lease = self._rt.assignment.find_for(run_id, node_id)
        if lease is None:
            expired = [
                row
                for row in self._rt.assignment.find_expired(run_id=run_id)
                if row.node_id == node_id
            ]
            if not expired:
                return
            lease = expired[0]
        dispatcher = self._dispatchers.get(run_id)
        own_worker = (
            dispatcher.worker_id if dispatcher is not None else "in-process"
        )
        if lease.worker_id != own_worker:
            if self._rt.config.expect_external_workers:
                return
        # Force-drop (including expired): plain release refuses expired rows.
        self._rt.assignment.drop_for_node(run_id, node_id)

    def _defer_for_budget_halt(
        self,
        run_id: str,
        graph: WorkGraph,
        node_id: str,
    ) -> None:
        """Soft-preempt a still-pending parallel node when the budget is exhausted.

        Marks the node preempted (so late ``dispatch_one`` starts skip). While an
        orphan future is still running, keeps this engine's lease as a parking
        fence so external harnesses cannot re-lease the READY row; once the
        orphan is gone (or was never registered), releases the lease. Discards
        any open attempt without burning ``max_attempts``, and restores
        ``READY`` for a larger-budget resume. Nodes that never left READY are
        left untouched aside from the preempt mark.

        The preempt mark and READY/RUNNING decision are taken under the node's
        lock so they cannot race ``Dispatcher._execute``'s check→RUNNING window.
        """
        try:
            node = graph.get(node_id)
        except KeyError:
            self._budget_preempted.setdefault(run_id, set()).add(node_id)
            return

        need_persist = False
        with node._lock:
            self._budget_preempted.setdefault(run_id, set()).add(node_id)
            if node.state == NodeState.READY:
                pass
            elif node.state == NodeState.RUNNING:
                node.discard_open_attempt()
                node.failure = None
                node.transition(
                    NodeState.READY,
                    reason="budget exhausted; deferred until budget allows",
                )
                need_persist = True
            elif (
                node.state == NodeState.FAILED
                and node.failure is not None
                and node.failure.kind == FailureKind.BUDGET_EXHAUSTED
            ):
                node.discard_last_attempt_with_failure(
                    FailureKind.BUDGET_EXHAUSTED
                )
                node.failure = None
                node.transition(
                    NodeState.READY,
                    reason="budget exhausted; deferred until budget allows",
                )
                need_persist = True
            else:
                return

        # Keep the engine lease while an orphan dispatch_one is still running
        # (parking). UNIQUE(run_id, node_id) then blocks WorkerHarness from
        # re-leasing the soft-deferred READY row. Release only once the orphan
        # is gone (drain) or was never registered (cold cancel).
        if not self._has_inflight_orphan(run_id, node_id):
            self._release_own_node_lease(run_id, node_id)
        if not need_persist:
            return
        # When parking an orphan lease, pass lease_id so
        # refuse_if_running_unowned does not reject the READY write.
        parked = self._rt.assignment.find_for(run_id, node_id)
        persist = self._persist_node(
            run_id,
            node,
            lease_id=parked.lease_id if parked is not None else None,
        )
        if persist != _PersistResult.SUCCESS:
            self._recover_ready_persist_failure(
                run_id, graph, node_id, persist
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
            # Include registered orphans even when their parking lease has
            # expired: Controller WAIT must cover soft-defer / backstop
            # futures still executing, same as idle ENQUEUE refund.
            in_flight_leases=(
                self._rt.assignment.count_for_run(run_id)
                + self._inflight_orphan_count(run_id)
            ),
            all_terminal=graph.all_terminal(),
            has_failed_terminal=graph.has_failed(),
        )

    def _inflight_orphan_count(self, run_id: str) -> int:
        """Number of registered orphan Futures for ``run_id`` (any kind)."""
        with self._orphans_lock:
            by_node = self._inflight_orphans.get(run_id)
            return len(by_node) if by_node else 0

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
