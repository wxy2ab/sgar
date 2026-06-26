"""ccx SwarmCoordinator — v5-backed replacement for cc's
``core.cc.agents.swarm.coordinator.SwarmCoordinator``.

Same outer dataclasses (``WorkerAssignment``, ``AssignmentRunResult``,
``SwarmRunSummary``) so callers can swap imports with no field changes.

Internally the coordinator maps each WorkerAssignment to a v5 NodeSpec
(tool ``ccx.agent`` by default) and runs them through a single v5 engine
invocation. v5's parallel dispatch + lease/heartbeat replaces cc's
asyncio.gather + retry-by-mailbox machinery.

Caveats vs cc:
* Mailbox semantics (per-worker event streams) are not reproduced;
  events come back via v5's EventBus instead. Callers that depended on
  the mailbox API need ``runtime.event_bus.subscribe`` instead.
* ``timeout_seconds`` is honoured per-node via NodeSpec.timeout_s.
* ``max_retries`` becomes NodeSpec.max_attempts.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.deepstack_v5 import NodeSpec, NodeState, RunStatus

from ...modes.llm_client import LLMCallable
from ...runtime import build_runtime, CCX_MODE_TOOL_MAP
from .mailbox_bridge import MailboxBridge


logger = logging.getLogger(__name__)
_CANCEL_CLEANUP_TIMEOUT_S = 5.0
_CANCEL_GRACE_S = 5.0


def _retrieve_task_exception(task: asyncio.Task[Any]) -> None:
    try:
        task.exception()
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.debug(
            "SwarmCoordinator: background task ended after cleanup",
            exc_info=True,
        )


async def _wait_for_run_id(
    run_id_box: dict[str, str], *, timeout_s: float = 0.5,
) -> str | None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while True:
        run_id = run_id_box.get("run_id")
        if run_id:
            return run_id
        if asyncio.get_running_loop().time() >= deadline:
            return None
        await asyncio.sleep(0.02)


async def _await_cleanup(awaitable: Any) -> Any:
    task = asyncio.ensure_future(awaitable)
    was_cancelled = False
    while True:
        try:
            result = await asyncio.shield(task)
            if was_cancelled:
                raise asyncio.CancelledError
            return result
        except asyncio.CancelledError:
            if task.done():
                if was_cancelled:
                    try:
                        task.result()
                    except Exception:
                        logger.debug(
                            "SwarmCoordinator: cleanup task failed after "
                            "caller cancellation",
                            exc_info=True,
                        )
                    raise
                return task.result()
            was_cancelled = True
            continue
        except Exception:
            if was_cancelled:
                logger.debug(
                    "SwarmCoordinator: cleanup task failed after caller "
                    "cancellation",
                    exc_info=True,
                )
                raise asyncio.CancelledError
            raise


# Re-use cc's outer dataclasses to keep field-level compatibility.
@dataclass(slots=True)
class WorkerAssignment:
    description: str
    prompt: str
    runtime_id: str | None = None
    preferred_runtime_ids: list[str] = field(default_factory=list)
    timeout_seconds: float | None = None
    max_retries: int = 0


@dataclass(slots=True)
class AssignmentRunResult:
    runtime_id: str
    description: str
    success: bool
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    events: list = field(default_factory=list)
    attempts: list = field(default_factory=list)
    event_count_total: int = 0
    event_count_captured: int = 0
    event_count_dropped: int = 0
    attempt_count: int = 0
    total_duration_ms: float = 0.0


@dataclass(slots=True)
class SwarmRunSummary:
    team_id: str
    runs: list[AssignmentRunResult] = field(default_factory=list)

    @property
    def final_texts(self) -> list[str]:
        return [item.result.get("final_text", "") for item in self.runs]

    @property
    def event_count(self) -> int:
        return sum(item.event_count_total for item in self.runs)

    @property
    def failed_runs(self) -> list[AssignmentRunResult]:
        return [item for item in self.runs if not item.success]


class SwarmCoordinator:
    """v5-backed swarm coordinator.

    Constructor takes the bare minimum needed to dispatch work:
    workspace, llm callable, language. Cc's TeamRuntime dependency is
    replaced by v5's runtime — simpler and more flexible.
    """

    def __init__(
        self,
        *,
        workspace: Path | str,
        llm: LLMCallable,
        team_id: str = "ccx-swarm",
        language: str = "en",
        parallelism: int = 4,
        mailbox_bridge: MailboxBridge | None = None,
    ) -> None:
        self.workspace = Path(workspace)
        self.llm = llm
        self.team_id = team_id
        self.language = language
        self.parallelism = parallelism
        self.mailbox_bridge = mailbox_bridge

    # -- API: matches cc's SwarmCoordinator.coordinate ----------------------

    async def coordinate(
        self,
        *,
        assignments: list[WorkerAssignment],
        ack_events: bool = True,
        poll_interval: float = 0.05,
        event_sink: Any | None = None,
        default_timeout_seconds: float | None = None,
        stop_on_failure: bool = False,
    ) -> SwarmRunSummary:
        if not assignments:
            return SwarmRunSummary(team_id=self.team_id)

        # Build one NodeSpec per assignment. They are siblings without
        # explicit deps — v5 will dispatch them in parallel up to
        # `parallelism`. To enforce stop_on_failure, we use cc's behaviour
        # of marking subsequent assignments as skipped on failure: in
        # v5, the cleanest mapping is sequential deps. For
        # stop_on_failure we therefore chain the nodes left-to-right.
        prev_id: str | None = None
        specs: list[NodeSpec] = []
        for index, asg in enumerate(assignments):
            node_id = asg.runtime_id or f"swarm-{index}"
            depends_on: tuple[str, ...] = ()
            if stop_on_failure and prev_id is not None:
                depends_on = (prev_id,)
            specs.append(NodeSpec(
                node_id=node_id,
                tool=CCX_MODE_TOOL_MAP["agent"],
                params={
                    "goal": asg.prompt,
                    "metadata": {
                        "description": asg.description,
                        "runtime_id": node_id,
                    },
                },
                depends_on=depends_on,
                max_attempts=max(1, asg.max_retries + 1),
                timeout_s=asg.timeout_seconds or default_timeout_seconds,
                metadata={
                    "ccx_swarm_team_id": self.team_id,
                    "ccx_swarm_index": index,
                },
            ))
            prev_id = node_id

        bundle = build_runtime(
            workspace=self.workspace,
            llm=self.llm,
            language=self.language,
            parallelism=self.parallelism,
            propose_initial=lambda _g: list(specs),
        )

        # Subscribe to node events so we can attribute counts back to
        # the right AssignmentRunResult. When a MailboxBridge is wired,
        # we also route events through it so the per-runtime envelope
        # queues populate; the resulting AssignmentRunResult.events
        # carries MailboxEnvelope objects (drop-in compatible with cc).
        events_by_node: dict[str, list] = {}
        run_id_box: dict[str, str] = {}

        def _on_event(event: dict) -> None:
            if event.get("run_id"):
                run_id_box.setdefault("run_id", str(event.get("run_id")))
            payload = event.get("payload") or {}
            node_id = payload.get("node_id")
            if not node_id:
                return
            if self.mailbox_bridge is not None:
                envelope = self.mailbox_bridge.route_event(event)
                if envelope is not None:
                    events_by_node.setdefault(node_id, []).append(envelope)
                # Also keep the raw v5 event when no envelope was emitted
                # so debugging stays possible.
                else:
                    events_by_node.setdefault(node_id, []).append(event)
            else:
                events_by_node.setdefault(node_id, []).append(event)
            if event_sink is not None:
                try:
                    event_sink(event)
                except Exception:
                    logger.warning(
                        "SwarmCoordinator event_sink raised; isolating",
                        exc_info=True,
                    )

        bundle.runtime.event_bus.subscribe(_on_event, kind="node.")

        t0 = time.monotonic()
        runs: list[AssignmentRunResult] = []
        try:
            engine = bundle.runtime.engine()
            worker = asyncio.create_task(asyncio.to_thread(engine.run, goal="swarm"))
            try:
                verdict = await asyncio.shield(worker)
                run_id_box["run_id"] = verdict.run_id
            except asyncio.CancelledError:
                run_id = await _await_cleanup(_wait_for_run_id(run_id_box))
                cancel_sent = False
                deadline = asyncio.get_running_loop().time() + _CANCEL_GRACE_S
                while not worker.done() and asyncio.get_running_loop().time() < deadline:
                    run_id = run_id_box.get("run_id") or run_id
                    if run_id and not cancel_sent:
                        try:
                            await _await_cleanup(
                                asyncio.wait_for(
                                    asyncio.to_thread(engine.cancel, run_id),
                                    timeout=_CANCEL_CLEANUP_TIMEOUT_S,
                                )
                            )
                        except Exception:
                            logger.warning(
                                "SwarmCoordinator: engine.cancel failed during cancellation",
                                exc_info=True,
                            )
                        cancel_sent = True
                    remaining = max(0.0, deadline - asyncio.get_running_loop().time())
                    try:
                        await _await_cleanup(
                            asyncio.wait_for(
                                asyncio.shield(worker),
                                timeout=min(0.1, remaining),
                            )
                        )
                    except asyncio.TimeoutError:
                        continue
                    except asyncio.CancelledError:
                        logger.debug(
                            "SwarmCoordinator: worker cancellation observed during "
                            "cleanup",
                            exc_info=True,
                        )
                        break
                    except Exception:
                        logger.debug(
                            "SwarmCoordinator: worker ended with an exception during "
                            "cancellation cleanup",
                            exc_info=True,
                        )
                        break
                if not worker.done():
                    worker.add_done_callback(_retrieve_task_exception)
                    logger.warning(
                        "SwarmCoordinator: engine worker did not finish within "
                        "cancellation grace period"
                    )
                raise
            elapsed_ms = (time.monotonic() - t0) * 1000

            # Read back per-node state through the LIVE bundle's GraphStore
            # so we don't have to reopen the DB after shutdown. Reopening
            # while engine ThreadPoolExecutor threads still hold per-thread
            # connections has occasionally been observed to hit busy/locked
            # WAL state on Windows; doing the read inside the bundle's
            # lifetime sidesteps that entirely.
            graph_store = bundle.runtime.graph_store
            for asg, spec in zip(assignments, specs):
                row = graph_store.get_node(verdict.run_id, spec.node_id)
                node_events = events_by_node.get(spec.node_id, [])
                if row is None:
                    runs.append(AssignmentRunResult(
                        runtime_id=spec.node_id,
                        description=asg.description,
                        success=False,
                        error="node not found in graph",
                        attempt_count=0,
                    ))
                    continue
                state = row["state"]
                success = state == NodeState.SUCCEEDED.value
                result = row.get("result") or {}
                if not isinstance(result, dict):
                    result = {"final_text": str(result)}
                error = None
                if not success:
                    err_obj = row.get("failure")
                    if err_obj:
                        error = err_obj.get("message") or str(err_obj)
                    else:
                        error = f"node ended in state {state!r}"
                runs.append(AssignmentRunResult(
                    runtime_id=spec.node_id,
                    description=asg.description,
                    success=success,
                    result=result,
                    error=error,
                    events=list(node_events),
                    event_count_total=len(node_events),
                    event_count_captured=len(node_events),
                    event_count_dropped=0,
                    attempt_count=len(row.get("attempts") or []),
                    total_duration_ms=elapsed_ms,
                ))
        finally:
            bundle.shutdown(run_id=run_id_box.get("run_id"))

        return SwarmRunSummary(team_id=self.team_id, runs=runs)


__all__ = [
    "AssignmentRunResult",
    "SwarmCoordinator",
    "SwarmRunSummary",
    "WorkerAssignment",
]
