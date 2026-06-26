"""ClaimStore — knowledge layer for v5.

Combines what v3 split across ClaimStore + EvidenceLedger into a single
store: every claim carries its own evidence list. Within a run, claims
accumulate; the Compaction strategy archives stale/low-confidence ones.

Persistence is optional. If a `ClaimStorePersistence` is supplied, every
mutation is mirrored to SQLite; otherwise the store is in-memory only
(handy for tests and small jobs).

Public API:
    record(run_id, kind, statement, evidence=..., confidence=...)
    get(claim_id)
    list_active(run_id)
    archive(claim_id)
    add_evidence(claim_id, evidence_item)
    compact(run_id, *, max_claims, min_confidence)
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..types import new_id, now_ms


@dataclass(slots=True)
class Evidence:
    source: str
    kind: str
    payload: Any
    recorded_at_ms: int = field(default_factory=now_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "kind": self.kind,
            "payload": self.payload,
            "recorded_at_ms": self.recorded_at_ms,
        }


@dataclass(slots=True)
class Claim:
    claim_id: str
    run_id: str
    kind: str  # 'fact' | 'hypothesis' | 'observation' | 'rule'
    statement: str
    confidence: float = 0.5
    evidence: list[Evidence] = field(default_factory=list)
    created_at_ms: int = field(default_factory=now_ms)
    updated_at_ms: int = field(default_factory=now_ms)
    archived: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "run_id": self.run_id,
            "kind": self.kind,
            "statement": self.statement,
            "confidence": self.confidence,
            "evidence": [e.to_dict() for e in self.evidence],
            "created_at_ms": self.created_at_ms,
            "updated_at_ms": self.updated_at_ms,
            "archived": self.archived,
        }


class ClaimStore:
    def __init__(self, persistence: Any | None = None) -> None:
        self._claims: dict[str, Claim] = {}
        self._persistence = persistence
        self._lock = threading.Lock()

    # -- mutations -----------------------------------------------------------

    def record(
        self,
        run_id: str,
        kind: str,
        statement: str,
        *,
        confidence: float = 0.5,
        evidence: Iterable[Evidence] | None = None,
    ) -> Claim:
        claim = Claim(
            claim_id=new_id("claim"),
            run_id=run_id,
            kind=kind,
            statement=statement,
            confidence=confidence,
            evidence=list(evidence or []),
        )
        with self._lock:
            self._claims[claim.claim_id] = claim
        self._persist(claim)
        return claim

    def add_evidence(self, claim_id: str, evidence: Evidence) -> None:
        with self._lock:
            claim = self._claims.get(claim_id)
            if claim is None or claim.archived:
                return
            claim.evidence.append(evidence)
            claim.updated_at_ms = now_ms()
            persist = claim
        self._persist(persist)

    def update_confidence(self, claim_id: str, confidence: float) -> None:
        with self._lock:
            claim = self._claims.get(claim_id)
            if claim is None:
                return
            claim.confidence = max(0.0, min(1.0, confidence))
            claim.updated_at_ms = now_ms()
            persist = claim
        self._persist(persist)

    def archive(self, claim_id: str) -> None:
        with self._lock:
            claim = self._claims.get(claim_id)
            if claim is None or claim.archived:
                return
            claim.archived = True
            claim.updated_at_ms = now_ms()
        if self._persistence is not None:
            try:
                self._persistence.archive(claim_id)
            except Exception:
                pass

    # -- queries -------------------------------------------------------------

    def get(self, claim_id: str) -> Claim | None:
        with self._lock:
            return self._claims.get(claim_id)

    def list_active(self, run_id: str) -> list[Claim]:
        with self._lock:
            return [
                c for c in self._claims.values()
                if c.run_id == run_id and not c.archived
            ]

    def count_active(self, run_id: str) -> int:
        with self._lock:
            return sum(
                1 for c in self._claims.values()
                if c.run_id == run_id and not c.archived
            )

    def load(self, run_id: str | None = None, *, limit: int | None = None) -> int:
        """Load active persisted claims into the in-memory cache."""
        if self._persistence is None:
            return 0
        try:
            rows = (
                self._persistence.list_active(run_id, limit=limit)
                if run_id is not None
                else self._persistence.list_all_active(limit=limit)
            )
        except Exception:
            return 0
        loaded = 0
        with self._lock:
            for row in rows:
                claim = Claim(
                    claim_id=row["claim_id"],
                    run_id=row["run_id"],
                    kind=row["kind"],
                    statement=row["statement"],
                    confidence=float(row["confidence"]),
                    evidence=[
                        Evidence(
                            source=e.get("source", ""),
                            kind=e.get("kind", ""),
                            payload=e.get("payload"),
                            recorded_at_ms=int(
                                e.get("recorded_at_ms") or now_ms()
                            ),
                        )
                        for e in (row.get("evidence") or [])
                    ],
                    created_at_ms=int(row.get("created_at_ms") or now_ms()),
                    updated_at_ms=int(row.get("updated_at_ms") or now_ms()),
                    archived=False,
                )
                self._claims[claim.claim_id] = claim
                loaded += 1
        return loaded

    # -- compaction (called by Compaction strategy) --------------------------

    def compact(
        self,
        run_id: str,
        *,
        max_claims: int | None = None,
        min_confidence: float | None = None,
    ) -> list[str]:
        """Archive low-confidence and oldest claims so that:
        * remaining count <= max_claims (oldest archived first)
        * all remaining claims have confidence >= min_confidence

        Returns archived claim IDs.
        """
        with self._lock:
            actives = [
                c for c in self._claims.values()
                if c.run_id == run_id and not c.archived
            ]

        archived: list[str] = []

        # 1. Drop low-confidence claims first.
        if min_confidence is not None:
            for c in actives:
                if c.confidence < min_confidence:
                    self.archive(c.claim_id)
                    archived.append(c.claim_id)
            actives = [c for c in actives if c.claim_id not in archived]

        # 2. Cap count by archiving oldest first.
        if max_claims is not None and len(actives) > max_claims:
            actives.sort(key=lambda c: c.updated_at_ms)
            overflow = len(actives) - max_claims
            for c in actives[:overflow]:
                self.archive(c.claim_id)
                archived.append(c.claim_id)

        return archived

    # -- persistence sync ----------------------------------------------------

    def _persist(self, claim: Claim) -> None:
        if self._persistence is None:
            return
        try:
            self._persistence.upsert(
                claim_id=claim.claim_id,
                run_id=claim.run_id,
                kind=claim.kind,
                statement=claim.statement,
                confidence=claim.confidence,
                evidence=[e.to_dict() for e in claim.evidence],
            )
        except Exception:
            # Persistence failures are non-fatal for in-memory operation.
            pass


__all__ = ["Claim", "ClaimStore", "Evidence"]
