"""AssignmentManager — worker leases with heartbeat-based reclamation.

A lease binds (run_id, node_id) to a single worker for a TTL window. The
worker must heartbeat before TTL elapses or the lease is reclaimed and
the node becomes eligible for reassignment.

This is the substrate for v4's worker-failure recovery: if a worker
crashes mid-execution, a sweeper detects expired leases, the node either
gets a fresh attempt or the in-flight attempt is marked UNKNOWN_EFFECT
(because side effects may have happened before the crash).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Protocol

from ..types import Lease, new_id, now_ms


# Kept in sync with ConfigV5.lease_ttl_ms / heartbeat_interval_ms. The 5-min
# TTL gives a healthily-running node slack to survive a briefly-starved
# heartbeat (GIL/SQLite contention) without losing its lease.
DEFAULT_LEASE_TTL_MS = 300_000
DEFAULT_HEARTBEAT_INTERVAL_MS = 10_000


class _GraphLeaseBackend(Protocol):
    def grant_lease(self, lease: Lease) -> None: ...
    def heartbeat_lease(self, lease_id: str, expires_at_ms: int) -> bool: ...
    def renew_lease_for_node(
        self,
        run_id: str,
        node_id: str,
        expires_at_ms: int,
        *,
        worker_id: str,
    ) -> bool: ...
    def release_lease(self, lease_id: str) -> bool: ...
    def drop_lease_for_node(self, run_id: str, node_id: str) -> bool: ...
    def find_expired(self, *, now: int) -> list[Lease]: ...
    def find_lease_row(self, run_id: str, node_id: str) -> Lease | None: ...
    def reclaim_expired(
        self,
        *,
        now: int,
        run_id: str | None = None,
        exclude_node_ids: tuple[str, ...] | list[str] | None = None,
    ) -> list[Lease]: ...
    def find_lease_for(self, run_id: str, node_id: str) -> Lease | None: ...
    def count_leases(self, run_id: str) -> int: ...


@dataclass(slots=True)
class LeaseGrantResult:
    lease: Lease
    granted: bool
    reason: str = ""


class AssignmentManager:
    def __init__(
        self,
        backend: _GraphLeaseBackend,
        *,
        ttl_ms: int = DEFAULT_LEASE_TTL_MS,
        heartbeat_interval_ms: int = DEFAULT_HEARTBEAT_INTERVAL_MS,
    ) -> None:
        self._backend = backend
        self.ttl_ms = ttl_ms
        self.heartbeat_interval_ms = heartbeat_interval_ms

    # -- core ----------------------------------------------------------------

    def lease(
        self,
        run_id: str,
        node_id: str,
        worker_id: str,
        *,
        ttl_ms: int | None = None,
    ) -> LeaseGrantResult:
        ttl = self.ttl_ms if ttl_ms is None else ttl_ms
        ts = now_ms()
        # Refuse to steal a still-live engine parking fence: an expired row
        # with a fresh heartbeat means the engine orphan keep-alive is alive
        # (or just hiccupped). Stale heartbeats are dead-engine leftovers and
        # may be replaced via grant_lease's expired-row DELETE.
        #
        # Fence freshness must use at least DEFAULT_LEASE_TTL_MS (engine
        # ConfigV5 default). A WorkerHarness that grants with a shorter
        # ``ttl_ms`` must not treat a 120–300s-old engine heartbeat as dead
        # and steal beside a still-running budget orphan.
        existing_row = self._backend.find_lease_row(run_id, node_id)
        fence_ttl = max(int(ttl), int(self.ttl_ms), DEFAULT_LEASE_TTL_MS)
        if (
            existing_row is not None
            and existing_row.expires_at_ms <= ts
            and self.parking_fence_is_live(
                existing_row, now=ts, ttl_ms=fence_ttl
            )
        ):
            return LeaseGrantResult(
                lease=existing_row,
                granted=False,
                reason=f"parking fence held by {existing_row.worker_id}",
            )
        lease = Lease(
            lease_id=new_id("L"),
            run_id=run_id,
            node_id=node_id,
            worker_id=worker_id,
            granted_at_ms=ts,
            expires_at_ms=ts + ttl,
            heartbeat_at_ms=ts,
        )
        try:
            self._backend.grant_lease(lease)
            return LeaseGrantResult(lease=lease, granted=True)
        except (sqlite3.IntegrityError, ValueError) as exc:
            # Already leased; report current owner.
            existing = self._backend.find_lease_for(run_id, node_id)
            if existing is not None:
                return LeaseGrantResult(
                    lease=existing,
                    granted=False,
                    reason=f"already leased to {existing.worker_id}",
                )
            return LeaseGrantResult(lease=lease, granted=False, reason=str(exc))

    @staticmethod
    def parking_fence_is_live(
        lease: Lease,
        *,
        now: int | None = None,
        ttl_ms: int,
    ) -> bool:
        """True when an (possibly expired) lease still looks engine-kept-alive.

        Engine parking / dispatch heartbeats bump ``heartbeat_at_ms``. Once the
        engine is dead that stamp freezes; after one TTL the fence is stale and
        reclaim / grant may clear it. Callers must NOT renew in the harness —
        renewing would refresh heartbeat and forever revive dead fences.
        """
        ts = now if now is not None else now_ms()
        return (ts - int(lease.heartbeat_at_ms)) <= max(int(ttl_ms), 1)

    def heartbeat(
        self,
        lease_id: str,
        *,
        ttl_ms: int | None = None,
    ) -> bool:
        ttl = self.ttl_ms if ttl_ms is None else ttl_ms
        return self._backend.heartbeat_lease(lease_id, now_ms() + ttl)

    def renew_for_node(
        self,
        run_id: str,
        node_id: str,
        *,
        ttl_ms: int | None = None,
        worker_id: str,
    ) -> bool:
        """Extend a lease row even if it is already expired.

        Used for budget parking fences: an expired row must be revived in
        place so ``grant_lease`` / ``reclaim_expired`` cannot open a READY
        window with no lease for WorkerHarness to steal.

        ``worker_id`` must match the row owner — otherwise a late engine
        parking heartbeat would extend a harness-stolen lease and enable
        double-dispatch beside a still-running budget orphan.
        """
        ttl = self.ttl_ms if ttl_ms is None else ttl_ms
        return self._backend.renew_lease_for_node(
            run_id, node_id, now_ms() + ttl, worker_id=worker_id
        )

    def release(self, lease_id: str) -> bool:
        return self._backend.release_lease(lease_id)

    def drop_for_node(self, run_id: str, node_id: str) -> bool:
        """Delete any lease row for ``(run_id, node_id)``, even if expired.

        Used when draining parking fences: ``release`` refuses expired rows and
        ``find_for`` ignores them, which would leave a READY+expired row for
        WorkerHarness to forever ``renew_for_node``.
        """
        return self._backend.drop_lease_for_node(run_id, node_id)

    def reclaim_expired(
        self,
        *,
        now: int | None = None,
        run_id: str | None = None,
        protect_node_ids: tuple[str, ...] | list[str] | None = None,
        renew_protected: bool = True,
        protect_worker_id: str | None = None,
    ) -> list[Lease]:
        ts = now if now is not None else now_ms()
        protect = tuple(protect_node_ids or ())
        if run_id is not None and protect:
            # Engine reclaim renews parking fences in place so they stay
            # active. Harness reclaim sets renew_protected=False: excluding
            # from DELETE is enough, and renewing would refresh heartbeat_at
            # and forever revive a dead engine's fence.
            if renew_protected:
                if not protect_worker_id:
                    raise ValueError(
                        "protect_worker_id is required when renew_protected=True"
                    )
                for node_id in protect:
                    self._backend.renew_lease_for_node(
                        run_id,
                        node_id,
                        now_ms() + self.ttl_ms,
                        worker_id=protect_worker_id,
                    )
            return self._backend.reclaim_expired(
                now=ts, run_id=run_id, exclude_node_ids=protect
            )
        return self._backend.reclaim_expired(now=ts, run_id=run_id)

    def find_expired(
        self,
        *,
        now: int | None = None,
        run_id: str | None = None,
    ) -> list[Lease]:
        """List expired leases (does not delete). Optional ``run_id`` filter."""
        ts = now if now is not None else now_ms()
        leases = self._backend.find_expired(now=ts)
        if run_id is None:
            return leases
        return [lease for lease in leases if lease.run_id == run_id]

    def find_row(self, run_id: str, node_id: str) -> Lease | None:
        """Return any lease row for the node, including expired."""
        return self._backend.find_lease_row(run_id, node_id)

    def find_for(self, run_id: str, node_id: str) -> Lease | None:
        return self._backend.find_lease_for(run_id, node_id)

    def count_for_run(self, run_id: str) -> int:
        return self._backend.count_leases(run_id)


__all__ = ["AssignmentManager", "LeaseGrantResult",
           "DEFAULT_HEARTBEAT_INTERVAL_MS", "DEFAULT_LEASE_TTL_MS"]
