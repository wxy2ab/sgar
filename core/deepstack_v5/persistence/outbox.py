"""Append-only outbox for event delivery (Write-Ahead Log).

Pattern:
1. EventBus.publish() appends to `events` (DB-sequenced) AND inserts the same
   sequence into `outbox` with delivered_at_ms = NULL — both within the same
   SQLite transaction. This guarantees the event is durable before any
   subscriber is notified.
2. A reader (single thread within EngineV5, or a separate process) polls
   `claim_pending()` to get a batch of undelivered sequence numbers, fans out
   to subscribers, then calls `mark_delivered()` once acknowledged.
3. On crash before mark_delivered, the events remain in the outbox and will be
   re-delivered on resume — at-least-once semantics. Subscribers must be
   idempotent.

This is also how multi-worker harnesses get authoritative "world state": they
poll the outbox.
"""

from __future__ import annotations

from typing import Sequence

from ..types import now_ms
from .db import SQLiteRuntimeDB


class Outbox:
    def __init__(self, db: SQLiteRuntimeDB):
        self.db = db

    def stage(self, sequence: int, *, run_id: str | None = None) -> None:
        """Insert a new event sequence into the outbox as undelivered.

        Must be called inside the same transaction as the events INSERT to
        guarantee atomicity.
        """
        self.db.execute(
            "INSERT OR IGNORE INTO outbox (sequence, delivered_at_ms) VALUES (?, NULL)",
            (sequence,),
        )

    def claim_pending(
        self,
        *,
        limit: int = 100,
        run_id: str | None = None,
    ) -> list[int]:
        """Return up to `limit` sequence numbers awaiting delivery."""
        if run_id is None:
            rows = self.db.query(
                """
                SELECT sequence FROM outbox
                WHERE delivered_at_ms IS NULL
                ORDER BY sequence ASC LIMIT ?
                """,
                (limit,),
            )
        else:
            rows = self.db.query(
                """
                SELECT o.sequence FROM outbox AS o
                JOIN events AS e ON e.sequence = o.sequence
                WHERE o.delivered_at_ms IS NULL AND e.run_id = ?
                ORDER BY o.sequence ASC LIMIT ?
                """,
                (run_id, limit),
            )
        return [int(r["sequence"]) for r in rows]

    def mark_delivered(self, sequences: Sequence[int]) -> None:
        if not sequences:
            return
        ts = now_ms()
        # Chunk to avoid SQLite parameter limits on huge batches.
        chunk_size = 500
        for i in range(0, len(sequences), chunk_size):
            chunk = sequences[i : i + chunk_size]
            placeholders = ",".join("?" * len(chunk))
            self.db.execute(
                f"UPDATE outbox SET delivered_at_ms = ? WHERE sequence IN ({placeholders})",
                (ts, *chunk),
            )

    def reset_pending(self) -> int:
        """Mark all delivered events as undelivered. Used by tests / replay."""
        cur = self.db.execute(
            "UPDATE outbox SET delivered_at_ms = NULL WHERE delivered_at_ms IS NOT NULL"
        )
        return cur.rowcount

    def pending_count(self) -> int:
        row = self.db.query_one(
            "SELECT COUNT(*) AS n FROM outbox WHERE delivered_at_ms IS NULL"
        )
        return int(row["n"]) if row else 0


__all__ = ["Outbox"]
