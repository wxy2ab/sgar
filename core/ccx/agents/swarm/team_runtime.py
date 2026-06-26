"""ccx TeamRuntime — v5-backed equivalent of cc's TeamRuntime.

.. warning:: EXPERIMENTAL — no production callers. Nothing outside the
   test suite constructs ``TeamRuntime``, and the "drop-in cc
   compatible" claim below has known semantic divergences (envelope
   direction is coordinator→worker where cc emits worker→lead; a
   successful node yields two ``task_completed`` envelopes via the
   ``node.succeeded`` + ``node.completed`` kind mapping). Reconcile
   those before building on this layer.

cc's TeamRuntime is built around long-lived **worker controllers** that
sit idle in subprocesses waiting for task assignments delivered via a
file-backed mailbox. That model is sensible for cc's local-subprocess
backend; on v5 it'd require keeping subprocess agents alive between
assignments. Instead, ccx maps the same conceptual surface (workers,
assignment, broadcast) onto v5's per-task DAG dispatch:

* ``spawn_worker(description)`` — pre-allocates a ``runtime_id`` slot
  (no subprocess yet)
* ``assign_task(runtime_id, prompt)`` — runs ONE v5 node for that
  runtime_id; worker is "alive" only for the duration of that node
* ``broadcast(prompt)`` — assigns to ALL members in parallel via
  SwarmCoordinator
* ``coordinate_assignment(...)`` — assign + wait for result + emit
  envelopes via MailboxBridge

This loses cc's "stand by for assignments" semantics — workers can't
hold session state between assignments — but in exchange every
assignment is durable in v5 SQLite, can resume across restarts, and
benefits from lease-based fault tolerance.

Drop-in compatibility: same class name + same method signatures for
``spawn_worker``, ``broadcast``, ``assign_task``, ``coordinate_assignment``,
``close``. Field shapes match cc's where possible.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.cc.agents.swarm.mailbox import MailboxEnvelope
from core.cc.config import CCConfig
from core.cc.llm import DefaultLLMClientProvider, LLMClientProvider

from ...modes.llm_client import LLMCallable, from_provider
from .coordinator import (
    AssignmentRunResult,
    SwarmCoordinator,
    WorkerAssignment,
)
from .mailbox_bridge import MailboxBridge


# --------------------------------------------------------------------------- #
# Lightweight TeamDefinition (no inheritance from cc's; same shape).
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class TeamDefinition:
    team_id: str
    lead_runtime_id: str = "ccx-coordinator"
    members: list[str] = field(default_factory=list)
    shared_context: dict[str, Any] = field(default_factory=dict)
    shared_allowed_paths: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# TeamRuntime
# --------------------------------------------------------------------------- #

class TeamRuntime:
    """v5-backed team runtime.

    Construct with the same shape as cc.TeamRuntime, but the heavy
    dependencies (RuntimeBackend, AgentRuntime, MailboxStore on disk)
    are replaced by ccx primitives.
    """

    def __init__(
        self,
        *,
        definition: TeamDefinition,
        config: CCConfig | None = None,
        cwd: str | None = None,
        llm_client_provider: LLMClientProvider | None = None,
        llm: LLMCallable | None = None,
        workspace: Path | str | None = None,
        parallelism: int = 4,
    ) -> None:
        self.definition = definition
        self.config = config or CCConfig()
        self.cwd = str(Path(cwd or ".").resolve())
        self.llm_client_provider = llm_client_provider or (
            DefaultLLMClientProvider() if llm is None else None
        )
        self._llm: LLMCallable
        if llm is not None:
            self._llm = llm
        else:
            self._llm = from_provider(self.llm_client_provider, self.config)

        if workspace is not None:
            self.workspace = Path(workspace)
        else:
            self.workspace = Path(self.cwd) / ".ccx" / "teams" / definition.team_id
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.parallelism = parallelism

        # Per-team mailbox bridge — coordinators created for each
        # assignment route their events through this single bridge so
        # the team-wide history is queryable in one place.
        self.mailbox = MailboxBridge(
            team_id=definition.team_id,
            coordinator_runtime_id=definition.lead_runtime_id,
        )
        # Worker descriptions, indexed by runtime_id (no live processes;
        # we materialise workers per-assignment).
        self._workers: dict[str, dict[str, Any]] = {}

    # -- Worker lifecycle ----------------------------------------------------

    async def spawn_worker(
        self,
        *,
        description: str,
        cwd: str | None = None,
        backend: str = "in_process",
        agent_id: str = "worker",
        name: str | None = None,
    ) -> dict[str, Any]:
        """Allocate a worker slot. No live process is started — a v5 node
        runs only when ``assign_task`` is called for this runtime_id."""
        runtime_id = f"rt_{uuid.uuid4().hex[:10]}"
        task_id = f"task_{uuid.uuid4().hex[:10]}"
        record = {
            "team_id": self.definition.team_id,
            "task_id": task_id,
            "runtime_id": runtime_id,
            "description": description,
            "agent_id": agent_id,
            "name": name or agent_id,
            "backend": backend,
            "cwd": cwd or self.cwd,
            "created_at": time.time(),
        }
        self._workers[runtime_id] = record
        if runtime_id not in self.definition.members:
            self.definition.members.append(runtime_id)
        return {
            "team_id": self.definition.team_id,
            "task_id": task_id,
            "runtime_id": runtime_id,
            "backend": backend,
            "launch": {
                "task_id": task_id,
                "runtime_id": runtime_id,
                "status": "ready",
                "background": False,
                "backend": backend,
                "waiting_reason": "waiting_for_assignment",
            },
        }

    def get_worker(self, runtime_id: str) -> dict[str, Any] | None:
        return self._workers.get(runtime_id)

    def list_workers(self) -> list[dict[str, Any]]:
        return list(self._workers.values())

    # -- Assignment ----------------------------------------------------------

    async def assign_task(
        self,
        *,
        runtime_id: str,
        description: str,
        prompt: str,
        from_runtime_id: str | None = None,
    ) -> dict[str, Any]:
        """Run ONE v5 agent node bound to this runtime_id."""
        worker = self._workers.get(runtime_id)
        if worker is None:
            raise ValueError(f"Unknown worker runtime: {runtime_id!r}")
        run = await self._run_single_assignment(
            runtime_id=runtime_id,
            description=description,
            prompt=prompt,
            from_runtime_id=from_runtime_id,
        )
        return self._assignment_run_to_dict(run)

    async def coordinate_assignment(
        self,
        *,
        runtime_id: str,
        description: str,
        prompt: str,
        from_runtime_id: str | None = None,
        event_sink: Any | None = None,
        ack_events: bool = True,
        poll_interval: float = 0.05,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Like ``assign_task`` but emits envelope events to the mailbox
        bridge (and the optional ``event_sink``) as the v5 node executes.
        Returns the same payload shape as ``assign_task``."""
        del ack_events, poll_interval  # honoured intrinsically by v5 lease
        run = await self._run_single_assignment(
            runtime_id=runtime_id,
            description=description,
            prompt=prompt,
            from_runtime_id=from_runtime_id,
            event_sink=event_sink,
            timeout_seconds=timeout_seconds,
        )
        return self._assignment_run_to_dict(run)

    async def broadcast(
        self,
        *,
        description: str,
        prompt: str,
        from_runtime_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Send the same prompt to every team member; runs them in parallel.

        ``from_runtime_id`` is accepted for cc-API compatibility but is
        not propagated to the per-node runs (cc uses it for sender
        accountability inside its mailbox; ccx logs it on the bridge
        side and not at dispatch time).
        """
        del from_runtime_id  # accepted for API parity, not threaded through
        member_ids = list(self.definition.members)
        if not member_ids:
            return []
        coord = self._make_coordinator()
        assignments = [
            WorkerAssignment(
                description=description,
                prompt=prompt,
                runtime_id=runtime_id,
            )
            for runtime_id in member_ids
        ]
        summary = await coord.coordinate(assignments=assignments)
        # Both success and failure entries share the same top-level shape:
        # {"runtime_id", "result", "error"}. ``result`` is the per-worker
        # payload (status="completed" or "failed", plus mode-specific
        # fields). ``error`` is None on success, a string on failure.
        # This lets callers iterate without branching on key presence.
        results: list[dict[str, Any]] = []
        for runtime_id, run in zip(member_ids, summary.runs):
            results.append({
                "runtime_id": runtime_id,
                "result": self._assignment_run_to_dict(run),
                "error": (
                    None if run.success
                    else (run.error or "broadcast assignment failed")
                ),
            })
        return results

    # -- Mailbox-style accessors --------------------------------------------

    def collect_worker_events(
        self,
        *,
        message_types: set[str] | None = None,
        ack: bool = False,
    ) -> list[MailboxEnvelope]:
        """Snapshot of envelopes routed to the team lead (coordinator)."""
        del ack  # ack semantics are advisory in ccx
        envelopes = self.mailbox.all_envelopes()
        if message_types is not None:
            envelopes = [
                e for e in envelopes if e.message_type in message_types
            ]
        return envelopes

    def collect_worker_results(
        self, *, ack: bool = False,
    ) -> list[MailboxEnvelope]:
        return self.collect_worker_events(
            message_types={"task_completed"},
            ack=ack,
        )

    # -- Cleanup -------------------------------------------------------------

    async def close(self, reason: str = "Team runtime closed.") -> None:
        del reason
        self._workers.clear()

    def close_sync(self, reason: str = "Team runtime closed.") -> None:
        del reason
        self._workers.clear()

    # -- internals -----------------------------------------------------------

    def _make_coordinator(self) -> SwarmCoordinator:
        return SwarmCoordinator(
            workspace=self.workspace,
            llm=self._llm,
            team_id=self.definition.team_id,
            language=self.config.prompt_language,
            parallelism=self.parallelism,
            mailbox_bridge=self.mailbox,
        )

    async def _run_single_assignment(
        self,
        *,
        runtime_id: str,
        description: str,
        prompt: str,
        from_runtime_id: str | None,
        event_sink: Any | None = None,
        timeout_seconds: float | None = None,
    ) -> AssignmentRunResult:
        del from_runtime_id  # cc carries this for accountability; ccx logs via metadata only
        coord = self._make_coordinator()
        assignments = [
            WorkerAssignment(
                description=description,
                prompt=prompt,
                runtime_id=runtime_id,
                timeout_seconds=timeout_seconds,
            ),
        ]
        summary = await coord.coordinate(
            assignments=assignments,
            event_sink=event_sink,
        )
        return summary.runs[0]

    @staticmethod
    def _assignment_run_to_dict(run: AssignmentRunResult) -> dict[str, Any]:
        if run.success:
            final = run.result.get("final_text", "")
            return {
                "runtime_id": run.runtime_id,
                "status": "completed",
                "final_text": final,
                "result": dict(run.result),
                "attempts": run.attempt_count,
                "events_count": run.event_count_total,
            }
        return {
            "runtime_id": run.runtime_id,
            "status": "failed",
            "error": run.error or "unknown",
            "attempts": run.attempt_count,
            "events_count": run.event_count_total,
        }


__all__ = [
    "TeamDefinition",
    "TeamRuntime",
]
