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
    def release_lease(self, lease_id: str) -> bool: ...
    def find_expired(self, *, now: int) -> list[Lease]: ...
    def reclaim_expired(
        self, *, now: int, run_id: str | None = None
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

    def heartbeat(
        self,
        lease_id: str,
        *,
        ttl_ms: int | None = None,
    ) -> bool:
        ttl = self.ttl_ms if ttl_ms is None else ttl_ms
        return self._backend.heartbeat_lease(lease_id, now_ms() + ttl)

    def release(self, lease_id: str) -> bool:
        return self._backend.release_lease(lease_id)

    def reclaim_expired(
        self, *, now: int | None = None, run_id: str | None = None
    ) -> list[Lease]:
        ts = now if now is not None else now_ms()
        return self._backend.reclaim_expired(now=ts, run_id=run_id)

    def find_for(self, run_id: str, node_id: str) -> Lease | None:
        return self._backend.find_lease_for(run_id, node_id)

    def count_for_run(self, run_id: str) -> int:
        return self._backend.count_leases(run_id)


__all__ = ["AssignmentManager", "LeaseGrantResult",
           "DEFAULT_HEARTBEAT_INTERVAL_MS", "DEFAULT_LEASE_TTL_MS"]
