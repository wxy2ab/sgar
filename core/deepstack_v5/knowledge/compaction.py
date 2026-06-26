"""Compaction — proactive context reduction strategy.

cc's existing compaction (`core/cc/conversation/compact.py`) is reactive:
it watches a character/message threshold. v5's CompactionStrategy is
proactive: it subscribes to engine events and decides ahead of time
when to compact, before the context grows too large.

Triggers (any of):
* `budget.warning` — token budget crossed warning ratio.
* `turn.boundary` — engine reports a logical turn end (e.g. node group
  completed).
* `node.succeeded` followed by token-usage delta exceeding `incremental_threshold`.
* Manual: `force_compact()` for caller-driven compaction.

The strategy itself doesn't manipulate cc messages — it returns a
`CompactionPlan` describing what to drop / archive / summarise. Callers
(engine, cc bridge) act on the plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Sequence, TYPE_CHECKING

from .claims import ClaimStore

if TYPE_CHECKING:
    from ..persistence.stores import EventStore, SnapshotStore


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CompactionPlan:
    triggered_by: str
    archive_claim_ids: list[str] = field(default_factory=list)
    estimated_tokens_freed: int = 0
    summary_message: str = ""
    snapshot_id: str | None = None
    snapshot_highwater_sequence: int | None = None


class CompactionStrategy:
    def __init__(
        self,
        *,
        max_active_claims: int = 200,
        min_claim_confidence: float = 0.2,
        incremental_token_threshold: int = 5_000,
    ) -> None:
        self.max_active_claims = max_active_claims
        self.min_claim_confidence = min_claim_confidence
        self.incremental_token_threshold = incremental_token_threshold
        self._tokens_since_last_compact: int = 0
        # Phase 3: persistence hooks. RuntimeV5 attaches these after
        # the stores exist; ``_compact`` writes a ResumeSnapshot row at
        # every trigger so a later run can resume context-aware.
        # Both ``None`` keeps Phase 1/2 behaviour (claims-only).
        self._event_store: "EventStore | None" = None
        self._snapshot_store: "SnapshotStore | None" = None
        self._snapshot_token_budget_chars: int = 12_000

    # -- event-driven entry points ------------------------------------------

    def on_node_succeeded(
        self,
        run_id: str,
        claim_store: ClaimStore,
        *,
        tokens_added: int = 0,
    ) -> CompactionPlan | None:
        self._tokens_since_last_compact += tokens_added
        if self._tokens_since_last_compact < self.incremental_token_threshold:
            return None
        plan = self._compact(run_id, claim_store, "node.succeeded")
        self._tokens_since_last_compact = 0
        return plan

    def on_budget_warning(
        self, run_id: str, claim_store: ClaimStore
    ) -> CompactionPlan:
        return self._compact(run_id, claim_store, "budget.warning")

    def on_turn_boundary(
        self, run_id: str, claim_store: ClaimStore
    ) -> CompactionPlan | None:
        # On turn boundaries we only compact if claims exceed soft cap.
        if claim_store.count_active(run_id) <= self.max_active_claims // 2:
            return None
        return self._compact(run_id, claim_store, "turn.boundary")

    def force_compact(
        self, run_id: str, claim_store: ClaimStore, *, reason: str = "manual"
    ) -> CompactionPlan:
        return self._compact(run_id, claim_store, reason)

    # -- core compaction -----------------------------------------------------

    def _compact(
        self,
        run_id: str,
        claim_store: ClaimStore,
        triggered_by: str,
    ) -> CompactionPlan:
        archived = claim_store.compact(
            run_id,
            max_claims=self.max_active_claims,
            min_confidence=self.min_claim_confidence,
        )
        # Rough token estimate: 50 tokens per claim is a conservative average.
        estimated = len(archived) * 50

        snapshot_id, highwater = self._persist_snapshot(run_id, triggered_by)

        return CompactionPlan(
            triggered_by=triggered_by,
            archive_claim_ids=archived,
            estimated_tokens_freed=estimated,
            summary_message=(
                f"compacted {len(archived)} claims (trigger={triggered_by})"
                if archived
                else f"no claims to compact (trigger={triggered_by})"
            ),
            snapshot_id=snapshot_id,
            snapshot_highwater_sequence=highwater,
        )

    # -- persistence hook (Phase 3) ------------------------------------------

    def attach_stores(
        self,
        *,
        event_store: "EventStore | None",
        snapshot_store: "SnapshotStore | None",
        snapshot_token_budget_chars: int = 12_000,
    ) -> None:
        """Wire the ResumeSnapshot persistence path.

        Called once by RuntimeV5 after stores exist. Either store
        being ``None`` disables snapshot writes (the strategy still
        archives claims as before). Idempotent: safe to call again
        with new instances when a runtime is re-wired in tests.
        """
        self._event_store = event_store
        self._snapshot_store = snapshot_store
        self._snapshot_token_budget_chars = snapshot_token_budget_chars

    def _persist_snapshot(
        self, run_id: str, triggered_by: str,
    ) -> tuple[str | None, int | None]:
        """Build a ResumeSnapshot from the events table and persist it.

        Returns ``(snapshot_id, highwater_sequence)`` or ``(None, None)``
        when no stores are attached, or on any error. The return tuple
        gets folded into the CompactionPlan so callers can confirm
        persistence happened (e.g. ``ccx watch`` could mark the
        compaction line with "snapshot saved").
        """
        if self._event_store is None or self._snapshot_store is None:
            return None, None
        try:
            # Local import to keep ``knowledge/`` decoupled from
            # ``memory/`` at module load — both packages import
            # ``persistence/stores`` and a top-level import chain
            # would create a cycle on cold start.
            from ..memory.snapshot import build_snapshot
            from ..types import new_id
            snapshot = build_snapshot(
                self._event_store, run_id,
                token_budget_chars=self._snapshot_token_budget_chars,
            )
            if snapshot.is_empty:
                return None, None
            snapshot_id = new_id("snap")
            self._snapshot_store.save(
                snapshot_id=snapshot_id,
                run_id=run_id,
                triggered_by=triggered_by,
                highwater_sequence=snapshot.highwater_sequence,
                summary=snapshot.summary,
                payload={
                    "summary": snapshot.summary,
                    "highwater_sequence": snapshot.highwater_sequence,
                    "built_at_ms": snapshot.built_at_ms,
                    "events": [
                        {
                            "sequence": ev.sequence,
                            "kind": ev.kind,
                            "priority": ev.priority,
                            "payload_excerpt": ev.payload_excerpt,
                            "occurred_at_ms": ev.occurred_at_ms,
                        }
                        for ev in snapshot.events
                    ],
                },
            )
            return snapshot_id, snapshot.highwater_sequence
        except Exception:
            logger.exception(
                "CompactionStrategy: failed to persist ResumeSnapshot for "
                "run_id=%r trigger=%r; claims compaction still applied",
                run_id, triggered_by,
            )
            return None, None


__all__ = ["CompactionPlan", "CompactionStrategy"]
