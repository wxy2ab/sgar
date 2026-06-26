"""Single in-process EventBus, optionally backed by SQLite outbox.

When `db` is provided, every publish writes to `events` + `outbox` in a
single SQLite transaction, then notifies in-process subscribers. On
resume, undelivered events from the outbox are replayed.

When `db` is None (test mode), the bus is purely in-memory.

Subscribers are called synchronously in the publish thread. Slow
subscribers should offload to their own queue.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from typing import Any, Callable

from .persistence.db import SQLiteRuntimeDB
from .persistence.outbox import Outbox
from .persistence.stores import EventStore


logger = logging.getLogger(__name__)


Subscriber = Callable[[dict[str, Any]], None]


class EventBus:
    def __init__(
        self,
        *,
        db: SQLiteRuntimeDB | None = None,
        event_store: EventStore | None = None,
        outbox: Outbox | None = None,
    ) -> None:
        self._db = db
        self._event_store = event_store
        self._outbox = outbox
        self._subscribers: list[tuple[str | None, Subscriber]] = []
        self._lock = threading.Lock()
        # Count of publishes whose persistence raised (best-effort, in-memory
        # only). Lets operators/tests distinguish "no db configured" (seq == 0)
        # from "persistence failed" (seq == -1) without crashing the engine.
        self._persist_failures = 0

    @property
    def persist_failures(self) -> int:
        """Number of publishes whose DB persistence failed (seq == -1)."""
        return self._persist_failures

    # -- subscribe -----------------------------------------------------------

    def subscribe(self, callback: Subscriber, *, kind: str | None = None) -> None:
        """Subscribe to events, optionally filtered by kind prefix."""
        with self._lock:
            self._subscribers.append((kind, callback))

    def unsubscribe(self, callback: Subscriber) -> None:
        with self._lock:
            self._subscribers = [
                (k, cb) for (k, cb) in self._subscribers if cb is not callback
            ]

    # -- publish -------------------------------------------------------------

    def publish(self, run_id: str, kind: str, payload: dict[str, Any]) -> int:
        """Persist (best-effort) + notify subscribers. Returns the sequence.

        Return semantics let callers tell apart the two non-persisted cases:
          * ``> 0`` — persisted with that sequence number.
          * ``0``   — no db configured (pure in-memory bus).
          * ``-1``  — db configured but persistence FAILED (in-memory only;
                      not replayable). ``persist_failures`` is also bumped.
        """
        seq = 0
        if self._db is not None and self._event_store is not None:
            try:
                with self._db.transaction():
                    seq = self._event_store.append(run_id, kind, payload)
                    if self._outbox is not None:
                        self._outbox.stage(seq, run_id=run_id)
            except sqlite3.DatabaseError as exc:
                # Persistence is best-effort. A transient SQLite hiccup
                # (e.g. a brief "database disk image is malformed" that
                # the next open recovers from cleanly) must not crash
                # the engine and lose an entire long-running v5 DAG.
                # In-process subscribers (the live event_sink that
                # streams progress to stdout, the v5 dispatcher's own
                # state machine) still need to fire — they're driven by
                # the in-memory ``_notify`` below, not the DB row.
                # seq = -1 (not 0) so the caller can distinguish a
                # persistence failure from a db-less bus.
                logger.warning(
                    "EventBus.publish: persistence failed for kind=%r "
                    "(run_id=%s) — continuing in-memory only: %s",
                    kind, run_id, exc,
                )
                seq = -1
                self._persist_failures += 1
        self._notify({
            "sequence": seq,
            "run_id": run_id,
            "kind": kind,
            "payload": dict(payload),
        })
        # Mark delivered immediately for in-process subscribers; outbox is
        # primarily a crash-recovery WAL.
        if self._outbox is not None and seq > 0:
            try:
                self._outbox.mark_delivered([seq])
            except sqlite3.DatabaseError as exc:
                logger.warning(
                    "EventBus.publish: outbox mark_delivered failed seq=%d "
                    "(run_id=%s): %s — outbox replay on resume will "
                    "redeliver, harmless for in-process subscribers.",
                    seq, run_id, exc,
                )
        return seq

    def _notify(self, event: dict[str, Any]) -> None:
        with self._lock:
            subs = list(self._subscribers)
        kind = event["kind"]
        for filter_kind, cb in subs:
            if filter_kind is not None and not kind.startswith(filter_kind):
                continue
            try:
                cb(event)
            except Exception:
                # Subscriber exceptions must not crash the publisher,
                # but they must surface in logs so the failing subscriber
                # is debuggable.
                logger.warning(
                    "EventBus subscriber raised on kind=%r; isolating",
                    kind, exc_info=True,
                )

    # -- replay (resume) -----------------------------------------------------

    def replay_outbox(self, *, limit: int = 1000, run_id: str | None = None) -> int:
        """Replay undelivered events from the outbox to current subscribers.

        Returns the number of events delivered. Idempotency is the
        subscribers' responsibility (events carry a sequence number).
        """
        if self._outbox is None or self._event_store is None:
            return 0
        pending = self._outbox.claim_pending(limit=limit, run_id=run_id)
        if not pending:
            return 0
        events = self._event_store.read_sequences(pending, run_id=run_id)
        delivered: list[int] = []
        for ev in events:
            self._notify(ev)
            delivered.append(ev["sequence"])
        if delivered:
            self._outbox.mark_delivered(delivered)
        return len(delivered)


__all__ = ["EventBus", "Subscriber"]
