"""Bridge between v5 EventBus events and cc's MailboxEnvelope shape.

cc's SwarmCoordinator yields ``MailboxEnvelope`` objects via per-runtime
queues. Existing cc callers depend on that shape: they read
``summary.runs[i].events`` expecting a list of envelopes with fields
like ``message_type``, ``payload``, ``from_runtime_id`` etc.

ccx SwarmCoordinator runs on v5, so its native events are dicts. The
``MailboxBridge`` converts each v5 event into a MailboxEnvelope, routes
it to the right per-runtime queue, and exposes the envelopes back to
the coordinator so it can populate ``AssignmentRunResult.events`` with
mailbox-shaped objects.

This is purely an adapter — no MailboxStore is involved (we don't
persist envelopes to disk; that's cc's mailbox-store concern). Callers
that need persistence can wire a MailboxStore on top via the
``on_envelope`` hook.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from core.cc.agents.swarm.mailbox import MailboxEnvelope


logger = logging.getLogger(__name__)


# v5 event kind → mailbox message_type
_DEFAULT_KIND_MAP: dict[str, str] = {
    "node.running": "task_started",
    "node.succeeded": "task_completed",
    "node.failed": "task_failed",
    "node.completed": "task_completed",  # engine-emitted variant
    "node.approval_pending": "approval_pending",
}


OnEnvelope = Callable[[MailboxEnvelope], None]


class MailboxBridge:
    """Routes v5 events into per-runtime mailbox queues.

    Construct with `team_id` and the set of runtime_ids the swarm will
    spawn (so we can pre-seed the queues; queues for unknown runtimes
    are created on first event).
    """

    def __init__(
        self,
        *,
        team_id: str = "ccx-swarm",
        coordinator_runtime_id: str = "ccx-coordinator",
        on_envelope: OnEnvelope | None = None,
        kind_map: dict[str, str] | None = None,
    ) -> None:
        self.team_id = team_id
        self.coordinator_runtime_id = coordinator_runtime_id
        self.on_envelope = on_envelope
        self.kind_map = dict(kind_map) if kind_map is not None else dict(_DEFAULT_KIND_MAP)
        self._queues: dict[str, list[MailboxEnvelope]] = {}
        # v5 EventBus is at-least-once: replay_outbox() re-delivers undelivered
        # events on resume, and idempotency is the subscriber's responsibility.
        # Dedup by (run_id, sequence) so a replayed event doesn't enqueue a
        # second envelope. Only events with a real persisted sequence (seq > 0)
        # are deduped — seq <= 0 marks in-memory-only events (no db / a
        # persistence failure) that are never replayed, so they pass through.
        self._seen: set[tuple[str, int]] = set()
        self._lock = threading.Lock()

    # -- routing -------------------------------------------------------------

    def route_event(self, event: dict[str, Any]) -> MailboxEnvelope | None:
        """Convert one v5 event dict into a MailboxEnvelope; route into
        the right per-runtime queue. Returns the envelope or None if the
        event was filtered (no mapping)."""
        kind = event.get("kind") or ""
        message_type = self.kind_map.get(kind)
        if message_type is None:
            # No mailbox mapping for this kind (e.g. node.spawn_skipped). The
            # event is intentionally dropped from the mailbox view; log at debug
            # so an operator diagnosing "missing envelope" can see what fell
            # through instead of guessing.
            logger.debug(
                "MailboxBridge: no mapping for event kind %r; dropping", kind,
            )
            return None
        payload = event.get("payload") or {}
        runtime_id = (
            payload.get("node_id")
            or payload.get("runtime_id")
            or "unknown"
        )
        # Dedup key for at-least-once replay (see __init__). Only persisted
        # events (seq > 0) carry a stable identity; seq <= 0 events are never
        # replayed so they bypass dedup.
        seq = event.get("sequence")
        dedup_key: tuple[str, int] | None = None
        if isinstance(seq, int) and seq > 0:
            dedup_key = (str(event.get("run_id")), seq)
        envelope = MailboxEnvelope(
            envelope_id=f"env-{uuid.uuid4().hex[:12]}",
            team_id=self.team_id,
            from_runtime_id=self.coordinator_runtime_id,
            to_runtime_id=str(runtime_id),
            message_type=message_type,
            payload=dict(payload),
        )
        with self._lock:
            if dedup_key is not None:
                if dedup_key in self._seen:
                    logger.debug(
                        "MailboxBridge: dropping duplicate event run_id=%s seq=%s kind=%r",
                        event.get("run_id"), seq, kind,
                    )
                    return None
                self._seen.add(dedup_key)
            self._queues.setdefault(envelope.to_runtime_id, []).append(envelope)
        if self.on_envelope is not None:
            try:
                self.on_envelope(envelope)
            except Exception:
                logger.warning(
                    "MailboxBridge on_envelope hook raised; isolating",
                    exc_info=True,
                )
        return envelope

    def envelopes_for(self, runtime_id: str) -> list[MailboxEnvelope]:
        """Snapshot of envelopes targeted at a runtime, in arrival order."""
        with self._lock:
            return list(self._queues.get(runtime_id, ()))

    def all_envelopes(self) -> list[MailboxEnvelope]:
        with self._lock:
            out: list[MailboxEnvelope] = []
            for queue in self._queues.values():
                out.extend(queue)
        # Sort by created_at for deterministic readback.
        out.sort(key=lambda e: (e.created_at, e.envelope_id))
        return out

    def clear(self) -> None:
        with self._lock:
            self._queues.clear()
            self._seen.clear()


__all__ = ["MailboxBridge", "OnEnvelope"]
