"""Dispatcher — runs a single ready node end-to-end.

The dispatcher is the bridge between a ready node and a real Python
callable. For in-process runs, the EngineV5 calls
`Dispatcher.dispatch_one()` directly; the multi-process WorkerHarness
calls the same code path from a worker subprocess.

Responsibilities:
1. Lease the node from AssignmentManager (skip if already leased).
2. Transition the node READY → RUNNING; open an Attempt.
3. Look up the capability and produce a ToolCall.
4. If the tool requires approval, transition the node to APPROVAL_HANG
   and return — engine will wake it up via `resume_after_approval`.
5. Otherwise execute the callable. Catch exceptions; convert hard
   timeouts to UNKNOWN_EFFECT so reconciliation can decide later.
6. Update node state to SUCCEEDED or FAILED based on outcome.
7. Always release the lease.
"""

from __future__ import annotations

import contextvars
import os
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ..types import (
    Capability,
    Failure,
    FailureKind,
    NodeSpec,
    NodeState,
    SpawnResult,
    ToolCallState,
    now_ms,
)
from .assignment import AssignmentManager
from .dispatch_context import DispatchContext, set_dispatch_context
from .graph import WorkGraph
from .node import NodeExecution
from .toolcall import ToolCall


@dataclass(slots=True)
class DispatchResult:
    node_id: str
    final_state: NodeState
    attempt_id: str | None = None
    result: Any = None
    failure: Failure | None = None
    skipped: bool = False
    skip_reason: str = ""
    tokens_reported: int = 0
    cost_reported: float = 0.0
    # Children added via SpawnResult during this dispatch. Caller (engine
    # or harness) is responsible for persisting these new nodes.
    spawned_node_ids: tuple[str, ...] = ()


class CapabilityNotFound(KeyError):
    pass


# Optional event-emit signature: callable invoked at boundary points.
EventEmitter = Callable[[str, dict[str, Any]], None]
LeasePersistCallback = Callable[[NodeExecution, str], None]


class Dispatcher:
    def __init__(
        self,
        run_id: str,
        graph: WorkGraph,
        assignment: AssignmentManager,
        capabilities: Mapping[str, Capability],
        *,
        worker_id: str = "in-process",
        event_emitter: EventEmitter | None = None,
        on_node_started: Callable[[NodeExecution], None] | None = None,
        on_node_finished: Callable[[NodeExecution], None] | None = None,
        on_node_started_with_lease: LeasePersistCallback | None = None,
        on_node_finished_with_lease: LeasePersistCallback | None = None,
        on_toolcall_started_with_lease: LeasePersistCallback | None = None,
        budget_reporter: Callable[[int, float], None] | None = None,
        interaction_fn: Callable[[Any], Any] | None = None,
    ) -> None:
        self.run_id = run_id
        self.graph = graph
        self.assignment = assignment
        self.capabilities = dict(capabilities)
        self.worker_id = worker_id
        self._emit = event_emitter or _noop_emit
        self._on_started = on_node_started
        self._on_finished = on_node_finished
        self._on_started_with_lease = on_node_started_with_lease
        self._on_finished_with_lease = on_node_finished_with_lease
        self._on_toolcall_started_with_lease = on_toolcall_started_with_lease
        self._budget_reporter = budget_reporter
        self._interaction_fn = interaction_fn

    def dispatch_one(
        self,
        node_id: str,
        *,
        pre_leased_id: str | None = None,
    ) -> DispatchResult:
        """Run a node end-to-end.

        If `pre_leased_id` is given, the caller has already secured the
        lease and the dispatcher only releases it on completion. This is
        used by WorkerHarness which leases-then-double-checks-state to
        avoid re-executing nodes whose status changed under it.
        """
        node = self.graph.get(node_id)
        if node.state != NodeState.READY:
            return DispatchResult(
                node_id=node_id,
                final_state=node.state,
                skipped=True,
                skip_reason=f"not READY (was {node.state.value})",
            )

        if pre_leased_id is not None:
            lease_id = pre_leased_id
        else:
            lease_result = self.assignment.lease(self.run_id, node_id, self.worker_id)
            if not lease_result.granted:
                return DispatchResult(
                    node_id=node_id,
                    final_state=node.state,
                    skipped=True,
                    skip_reason=lease_result.reason,
                )
            lease_id = lease_result.lease.lease_id

        stop_hb = threading.Event()
        hb_thread = self._start_heartbeat(lease_id, stop_hb)
        try:
            return self._execute(node, lease_id)
        finally:
            stop_hb.set()
            hb_thread.join(timeout=2.0)
            self.assignment.release(lease_id)

    # -- internal ------------------------------------------------------------

    #: How many *consecutive* heartbeat failures (transient SQLite lock
    #: contention under parallelism, momentary backend errors) the loop
    #: tolerates before giving up. A single hiccup must not kill the
    #: heartbeat thread: that lets the lease expire by wall-clock and a
    #: healthily-running node's completed work gets fence-rejected. Any
    #: success resets the counter. Genuine lease loss (the row is gone, so
    #: every beat fails) still terminates after this many tries; the
    #: engine's salvage path then preserves the result if no competitor
    #: actually took the node.
    _MAX_CONSECUTIVE_HEARTBEAT_FAILURES = 3

    def _start_heartbeat(
        self,
        lease_id: str,
        stop_event: threading.Event,
    ) -> threading.Thread:
        """Periodically extend the lease while the tool runs.

        Resilient to transient failures: a single ``heartbeat`` returning
        False or raising (e.g. SQLite write contention) no longer kills the
        thread — only ``_MAX_CONSECUTIVE_HEARTBEAT_FAILURES`` in a row, or
        ``stop_event``, stops it. This keeps the lease alive through brief
        contention so a still-running node doesn't lose it by wall-clock.
        On genuine, persistent loss the loop gives up and the lease reclaim
        sweeper / engine salvage path handles recovery.
        """
        interval_s = max(self.assignment.heartbeat_interval_ms / 1000.0, 0.01)
        max_failures = self._MAX_CONSECUTIVE_HEARTBEAT_FAILURES

        def loop() -> None:
            consecutive_failures = 0
            while not stop_event.wait(interval_s):
                try:
                    ok = self.assignment.heartbeat(lease_id)
                except Exception:
                    ok = False
                if ok:
                    consecutive_failures = 0
                    continue
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    return

        t = threading.Thread(
            target=loop,
            daemon=True,
            name=f"hb-{lease_id[:8]}",
        )
        t.start()
        return t

    def _execute(self, node: NodeExecution, lease_id: str) -> DispatchResult:
        node.transition(NodeState.RUNNING, reason="dispatched")
        attempt = node.new_attempt(worker_id=self.worker_id)
        if self._on_started_with_lease:
            self._on_started_with_lease(node, lease_id)
        elif self._on_started:
            self._on_started(node)
        self._emit("node.running", {
            "run_id": self.run_id, "node_id": node.node_id,
            "attempt_id": attempt.attempt_id, "worker_id": self.worker_id,
        })

        cap = self.capabilities.get(node.spec.tool)
        if cap is None:
            failure = Failure(
                kind=FailureKind.TOOL_ERROR,
                message=f"capability '{node.spec.tool}' not registered",
                retryable=False,
            )
            return self._finish_failed(node, attempt, failure, lease_id=lease_id)

        tc = ToolCall.new(
            tool_name=cap.name,
            params=node.spec.params,
            requires_approval=node.spec.requires_approval or cap.requires_approval,
        )
        attempt.tool_calls.append(tc)

        if tc.requires_approval:
            tc.request_approval()
            node.transition(NodeState.APPROVAL_HANG, reason="awaiting approval")
            self._emit("node.approval_pending", {
                "run_id": self.run_id, "node_id": node.node_id,
                "tool": cap.name,
            })
            if self._on_finished_with_lease:
                self._on_finished_with_lease(node, lease_id)
            elif self._on_finished:
                self._on_finished(node)
            return DispatchResult(
                node_id=node.node_id,
                final_state=NodeState.APPROVAL_HANG,
                attempt_id=attempt.attempt_id,
            )

        return self._run_tool(node, attempt, tc, cap, lease_id)

    def _run_tool(
        self,
        node: NodeExecution,
        attempt: Any,
        tc: ToolCall,
        cap: Capability,
        lease_id: str,
    ) -> DispatchResult:
        # Idempotent transition: if already RUNNING (post-approval) skip;
        # otherwise transition PENDING -> RUNNING.
        if tc.state != ToolCallState.RUNNING:
            tc.mark_running()
            if self._on_toolcall_started_with_lease:
                self._on_toolcall_started_with_lease(node, lease_id)
        timeout_s = node.spec.timeout_s or cap.timeout_s
        # Install a per-call dispatch context so tools that opt in (e.g.
        # the ccx → cc event bridge) can read run_id/node_id and publish
        # extra events through the same event_bus the dispatcher uses.
        # Tools that don't care never look at it.
        tokens_reported = 0
        cost_reported = 0.0
        # Activity heartbeat for the idle-abandon watchdog (see
        # ``_node_idle_timeout_s``): bumped on every observable progress
        # signal — an LLM cost report and any emitted event. A turn that goes
        # silent (no progress for the idle window) is abandoned + retried
        # instead of blocking until the whole-node ``timeout_s`` deadline.
        last_activity = [time.monotonic()]

        def report_cost(tokens: int, cost: float) -> None:
            nonlocal tokens_reported, cost_reported
            last_activity[0] = time.monotonic()
            tokens_reported += int(tokens or 0)
            cost_reported += float(cost or 0.0)
            if self._budget_reporter is not None:
                self._budget_reporter(int(tokens or 0), float(cost or 0.0))

        def _tracking_emit(kind: str, payload: dict[str, Any]) -> None:
            last_activity[0] = time.monotonic()
            self._emit(kind, payload)

        cancel_event = threading.Event()
        dispatch_ctx = DispatchContext(
            run_id=self.run_id,
            node_id=node.node_id,
            attempt_id=attempt.attempt_id,
            attempt_ordinal=node.attempt_count(),
            emit=_tracking_emit,
            report_cost_fn=report_cost,
            cancel_event=cancel_event,
            interaction_fn=self._interaction_fn,
        )
        idle_timeout_s = _node_idle_timeout_s()
        try:
            with set_dispatch_context(dispatch_ctx):
                try:
                    if timeout_s is not None and timeout_s > 0:
                        outcome = _call_with_timeout(
                            cap.fn,
                            tc.params,
                            timeout_s,
                            idle_timeout_s=idle_timeout_s,
                            activity=last_activity,
                        )
                    else:
                        outcome = cap.fn(**tc.params)
                finally:
                    # Close the observability context for any background or
                    # timed-out work that outlives this dispatch attempt.
                    cancel_event.set()
        except _ToolTimeout as exc:
            # Timeout — we cannot know if side effects landed.
            tc.mark_unknown(
                f"timeout after {timeout_s}s",
                effect_signature=str(exc),
            )
            failure = Failure(
                kind=FailureKind.TIMEOUT,
                message=f"tool '{cap.name}' timed out after {timeout_s}s",
                retryable=True,
                worker_id=self.worker_id,
            )
            return self._finish_failed(
                node,
                attempt,
                failure,
                lease_id=lease_id,
                tokens_reported=tokens_reported,
                cost_reported=cost_reported,
            )
        except BaseException as exc:  # noqa: BLE001
            tc.mark_failed(f"{type(exc).__name__}: {exc}")
            failure = Failure(
                kind=FailureKind.TOOL_ERROR,
                message=str(exc),
                retryable=not isinstance(exc, (KeyboardInterrupt, SystemExit)),
                worker_id=self.worker_id,
                details={"traceback": traceback.format_exc()},
            )
            result = self._finish_failed(
                node,
                attempt,
                failure,
                lease_id=lease_id,
                tokens_reported=tokens_reported,
                cost_reported=cost_reported,
            )
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            return result

        # Unwrap SpawnResult — children get added before we finish so they
        # are visible on the next promote tick.
        spawned_ids: tuple[str, ...] = ()
        if isinstance(outcome, SpawnResult):
            try:
                spawned_ids = self._spawn_children(node.node_id, outcome.spawn)
            except Exception as exc:
                tc.mark_failed(f"{type(exc).__name__}: {exc}")
                failure = Failure(
                    kind=FailureKind.TOOL_ERROR,
                    message=f"spawn failed: {exc}",
                    retryable=False,
                    worker_id=self.worker_id,
                    details={"traceback": traceback.format_exc()},
                )
                return self._finish_failed(
                    node,
                    attempt,
                    failure,
                    lease_id=lease_id,
                    tokens_reported=tokens_reported,
                    cost_reported=cost_reported,
                )
            outcome = outcome.value

        if node.state == NodeState.CANCELLED:
            return DispatchResult(
                node_id=node.node_id,
                final_state=NodeState.CANCELLED,
                attempt_id=attempt.attempt_id,
                skipped=True,
                skip_reason="node was cancelled while tool was running",
                tokens_reported=tokens_reported,
                cost_reported=cost_reported,
            )

        tc.mark_completed(outcome)
        node.finish_attempt(outcome="success", result=outcome)
        node.transition(NodeState.SUCCEEDED, reason="tool returned")
        self._emit("node.succeeded", {
            "run_id": self.run_id, "node_id": node.node_id,
            "attempt_id": attempt.attempt_id,
            "spawned": list(spawned_ids),
        })
        if self._on_finished_with_lease:
            self._on_finished_with_lease(node, lease_id)
        elif self._on_finished:
            self._on_finished(node)
        return DispatchResult(
            node_id=node.node_id,
            final_state=NodeState.SUCCEEDED,
            attempt_id=attempt.attempt_id,
            result=outcome,
            tokens_reported=tokens_reported,
            cost_reported=cost_reported,
            spawned_node_ids=spawned_ids,
        )

    def _spawn_children(
        self, parent_id: str, specs: list[NodeSpec]
    ) -> tuple[str, ...]:
        added: list[str] = []
        unique_specs: list[NodeSpec] = []
        seen_in_batch: set[str] = set()
        for spec in specs:
            if spec.node_id in seen_in_batch:
                self._emit("node.spawn_skipped", {
                    "run_id": self.run_id,
                    "parent_node_id": parent_id,
                    "node_id": spec.node_id,
                    "reason": "duplicate_in_batch",
                })
                continue
            seen_in_batch.add(spec.node_id)
            unique_specs.append(spec)
        for spec in self._order_spawn_specs(unique_specs):
            if self.graph.has(spec.node_id):
                self._emit("node.spawn_skipped", {
                    "run_id": self.run_id,
                    "parent_node_id": parent_id,
                    "node_id": spec.node_id,
                    "reason": "duplicate_node_id",
                })
                continue
            new_meta = dict(spec.metadata or {})
            new_meta["parent_node_id"] = parent_id
            new_spec = NodeSpec(
                node_id=spec.node_id,
                tool=spec.tool,
                params=dict(spec.params or {}),
                depends_on=tuple(spec.depends_on or ()),
                max_attempts=spec.max_attempts,
                timeout_s=spec.timeout_s,
                requires_approval=spec.requires_approval,
                priority=spec.priority,
                metadata=new_meta,
            )
            self.graph.add(new_spec)
            added.append(new_spec.node_id)
        return tuple(added)

    def _order_spawn_specs(self, specs: list[NodeSpec]) -> list[NodeSpec]:
        """Topologically order spawned children by sibling dependencies."""
        if not specs:
            return []
        by_id = {spec.node_id: spec for spec in specs}
        existing = set(self.graph.nodes())
        ordered: list[NodeSpec] = []
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in visited:
                return
            if node_id in visiting:
                raise ValueError(f"cycle in spawned children at {node_id}")
            spec = by_id[node_id]
            visiting.add(node_id)
            for dep in spec.depends_on or ():
                if dep in by_id:
                    visit(dep)
                elif dep not in existing:
                    raise KeyError(
                        f"spawned node {spec.node_id} depends on unknown node {dep}"
                    )
            visiting.remove(node_id)
            visited.add(node_id)
            ordered.append(spec)

        for spec in specs:
            visit(spec.node_id)
        return ordered

    def _finish_failed(
        self,
        node: NodeExecution,
        attempt: Any,
        failure: Failure,
        *,
        lease_id: str | None = None,
        tokens_reported: int = 0,
        cost_reported: float = 0.0,
    ) -> DispatchResult:
        if node.state == NodeState.CANCELLED:
            return DispatchResult(
                node_id=node.node_id,
                final_state=NodeState.CANCELLED,
                attempt_id=attempt.attempt_id,
                failure=failure,
                skipped=True,
                skip_reason="node was cancelled while dispatch was failing",
                tokens_reported=tokens_reported,
                cost_reported=cost_reported,
            )
        node.finish_attempt(outcome="failure", failure=failure)
        target = NodeState.FAILED
        if node.state == NodeState.APPROVAL_HANG:
            target = NodeState.ABANDONED
        if target == NodeState.ABANDONED:
            self.graph.mark(node.node_id, target, reason=failure.message[:80])
        else:
            node.transition(target, reason=failure.message[:80])
        self._emit("node.failed", {
            "run_id": self.run_id, "node_id": node.node_id,
            "attempt_id": attempt.attempt_id,
            "kind": failure.kind.value, "message": failure.message,
        })
        if self._on_finished_with_lease and lease_id is not None:
            self._on_finished_with_lease(node, lease_id)
        elif self._on_finished:
            self._on_finished(node)
        return DispatchResult(
            node_id=node.node_id,
            final_state=target,
            attempt_id=attempt.attempt_id,
            failure=failure,
            tokens_reported=tokens_reported,
            cost_reported=cost_reported,
        )

    # -- approval lifecycle --------------------------------------------------

    def resume_after_approval(
        self,
        node_id: str,
        *,
        approved: bool,
        pre_leased_id: str | None = None,
    ) -> DispatchResult:
        node = self.graph.get(node_id)
        if node.state != NodeState.APPROVAL_HANG:
            return DispatchResult(
                node_id=node_id,
                final_state=node.state,
                skipped=True,
                skip_reason=f"not APPROVAL_HANG (was {node.state.value})",
            )
        if pre_leased_id is None:
            lease_result = self.assignment.lease(
                self.run_id, node_id, self.worker_id
            )
            if not lease_result.granted:
                return DispatchResult(
                    node_id=node_id,
                    final_state=node.state,
                    skipped=True,
                    skip_reason=lease_result.reason,
                )
            lease_id = lease_result.lease.lease_id
        else:
            lease_id = pre_leased_id

        stop_hb = threading.Event()
        hb_thread = self._start_heartbeat(lease_id, stop_hb)
        try:
            return self._resume_after_approval_locked(
                node_id, approved=approved, lease_id=lease_id
            )
        finally:
            stop_hb.set()
            hb_thread.join(timeout=2.0)
            self.assignment.release(lease_id)

    def _resume_after_approval_locked(
        self,
        node_id: str,
        *,
        approved: bool,
        lease_id: str,
    ) -> DispatchResult:
        node = self.graph.get(node_id)
        attempt = node.current_attempt()
        if attempt is None:
            # No attempt was ever opened on this node — node was nudged
            # into APPROVAL_HANG out-of-band (e.g. crash-restored from a
            # partial write, manual state edit). We can't close an attempt
            # we don't have, so transition the node directly to ABANDONED
            # and surface the failure on the node itself.
            failure = Failure(
                kind=FailureKind.UNKNOWN,
                message="approval resume found no pending attempt",
                retryable=False,
            )
            node.failure = failure
            self.graph.mark(node.node_id, NodeState.ABANDONED,
                            reason=failure.message[:80])
            self._emit("node.failed", {
                "run_id": self.run_id, "node_id": node.node_id,
                "attempt_id": None,
                "kind": failure.kind.value, "message": failure.message,
            })
            if self._on_finished_with_lease:
                self._on_finished_with_lease(node, lease_id)
            elif self._on_finished:
                self._on_finished(node)
            return DispatchResult(
                node_id=node.node_id,
                final_state=NodeState.ABANDONED,
                attempt_id=None,
                failure=failure,
            )
        if not attempt.tool_calls:
            failure = Failure(
                kind=FailureKind.UNKNOWN,
                message="approval resume found no pending tool call",
                retryable=False,
            )
            return self._finish_failed(node, attempt, failure, lease_id=lease_id)
        tc = attempt.tool_calls[-1]
        if not approved:
            tc.reject()
            failure = Failure(
                kind=FailureKind.TOOL_ERROR,
                message="approval rejected",
                retryable=False,
            )
            node.finish_attempt(outcome="abandoned", failure=failure)
            self.graph.mark(node.node_id, NodeState.ABANDONED,
                            reason="approval rejected")
            if self._on_finished_with_lease:
                self._on_finished_with_lease(node, lease_id)
            elif self._on_finished:
                self._on_finished(node)
            return DispatchResult(
                node_id=node_id,
                final_state=NodeState.ABANDONED,
                attempt_id=attempt.attempt_id,
                failure=failure,
            )
        # Approved: re-acquire lease (or trust caller has it) and run.
        cap = self.capabilities.get(node.spec.tool)
        if cap is None:
            failure = Failure(
                kind=FailureKind.TOOL_ERROR,
                message=f"capability '{node.spec.tool}' not registered",
                retryable=False,
            )
            return self._finish_failed(node, attempt, failure, lease_id=lease_id)
        tc.approve()
        node.transition(NodeState.RUNNING, reason="approval granted")
        if self._on_toolcall_started_with_lease:
            self._on_toolcall_started_with_lease(node, lease_id)
        return self._run_tool(node, attempt, tc, cap, lease_id)


# --------------------------------------------------------------------------- #
# Timeout helpers
# --------------------------------------------------------------------------- #

class _ToolTimeout(Exception):
    pass


def _node_idle_timeout_s() -> float | None:
    """Idle (no-progress) abandon window for a node, read from the env.

    When ``CCX_NODE_IDLE_TIMEOUT_S`` is set to a positive value, a node whose
    activity heartbeat does not advance for that many seconds is abandoned and
    (TIMEOUT ⇒ retryable) retried, recovering a *wedged* agent turn well before
    the whole-node ``timeout_s`` deadline. The heartbeat is bumped on every
    observable progress signal (LLM cost report + emitted event), so a node
    doing real work is never falsely abandoned; only a turn that goes silent
    (no progress, 0% CPU — the observed wedge signature) trips it.

    Unset / non-positive / malformed ⇒ ``None`` ⇒ legacy behaviour (the node is
    bounded only by ``timeout_s``). Read per call so a launch/test can set it in
    the environment without import-order surprises; this keeps the default path
    byte-identical.
    """
    raw = os.environ.get("CCX_NODE_IDLE_TIMEOUT_S", "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _call_with_timeout(
    fn: Callable,
    params: Mapping[str, Any],
    timeout_s: float,
    *,
    idle_timeout_s: float | None = None,
    activity: list[float] | None = None,
) -> Any:
    """Run fn in a worker thread; raise _ToolTimeout on deadline.

    Note: Python threads cannot be killed safely. The worker thread
    continues after the timeout but its result is discarded; this is the
    classic "side effect may have landed" case that motivates UNKNOWN_EFFECT.

    The fn runs under a copy of the caller's contextvars context: a fresh
    thread starts with an EMPTY context (PEP 567), which would sever the
    DispatchContext installed by ``_run_tool`` and silently disable every
    context-reading consumer (ccx event bridge, cost/steer telemetry).

    When ``idle_timeout_s`` is a positive value the join is *polled* and the
    worker is abandoned if no progress is observed for ``idle_timeout_s``
    seconds, where progress is ``activity[0]`` (a ``time.monotonic()`` timestamp
    the caller bumps on each observable event). This recovers a wedged turn
    long before the whole-node ``timeout_s`` deadline while never abandoning a
    node that is actively making progress. Default (``None``) keeps the legacy
    single-``join(timeout_s)`` behaviour byte-for-byte.
    """
    result_box: dict[str, Any] = {}
    error_box: dict[str, BaseException] = {}
    ctx = contextvars.copy_context()

    def target() -> None:
        try:
            result_box["v"] = ctx.run(fn, **dict(params))
        except BaseException as exc:  # noqa: BLE001
            error_box["e"] = exc

    t = threading.Thread(target=target, daemon=True)
    t.start()
    if idle_timeout_s is not None and idle_timeout_s > 0:
        deadline = (
            time.monotonic() + timeout_s if (timeout_s and timeout_s > 0) else None
        )
        poll = min(idle_timeout_s, 5.0)
        while True:
            t.join(timeout=poll)
            if not t.is_alive():
                break
            now = time.monotonic()
            last = activity[0] if activity else now
            if now - last >= idle_timeout_s:
                raise _ToolTimeout(
                    f"idle {idle_timeout_s}s exceeded "
                    f"(no progress — wedged turn abandoned)"
                )
            if deadline is not None and now >= deadline:
                raise _ToolTimeout(f"deadline {timeout_s}s exceeded")
    else:
        t.join(timeout=timeout_s)
        if t.is_alive():
            raise _ToolTimeout(f"deadline {timeout_s}s exceeded")
    if "e" in error_box:
        raise error_box["e"]
    return result_box.get("v")


def _noop_emit(_kind: str, _payload: dict[str, Any]) -> None:
    pass


__all__ = ["CapabilityNotFound", "Dispatcher", "DispatchResult"]
