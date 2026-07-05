"""ConfigV5 — single dataclass holding all advanced overrides.

Most users instantiate `RuntimeV5.create(...)` without a ConfigV5; the
defaults below cover the common case. ConfigV5 is for power users who
want to tune the engine loop, lease TTLs, compaction thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .control.budget import BudgetTracker  # noqa: F401  (re-exported via runtime)
from .knowledge.compaction import CompactionStrategy


@dataclass(slots=True)
class ConfigV5:
    # Engine loop -------------------------------------------------------------
    poll_interval_s: float = 0.05
    """Sleep when WAIT decision is returned by Controller."""

    max_loop_iterations: int = 10_000
    """Hard ceiling on engine.run() iterations to prevent runaway loops."""

    budget_flush_every_n_iterations: int = 10
    """Persist the budget snapshot to the DB every N loop iterations (plus on a
    warning crossing). Bounds the in-flight budget delta lost to a hard
    SIGKILL/OOM so resume cannot resurrect already-spent budget. 0 disables the
    periodic flush (loop-boundary + warning-crossing flushes still fire)."""

    parallelism: int = 1
    """How many ready nodes to dispatch concurrently (in-process threads).
    Use multi-process via worker_count on RuntimeV5 instead for isolation."""

    # Leases ------------------------------------------------------------------
    # TTL is generous (5 min) on purpose: a healthily-running node whose
    # heartbeat thread is briefly starved (GIL held by CPU-bound work, or
    # SQLite write contention under parallelism) must not lose its lease and
    # have its completed work fence-rejected. The heartbeat (every 10s) keeps
    # the lease alive; the TTL is the slack before a genuinely-missed
    # heartbeat expires it irrecoverably (heartbeat_lease cannot revive an
    # already-expired lease). See Engine._make_dispatcher's salvage path for
    # the recovery when a lease does expire mid-flight.
    lease_ttl_ms: int = 300_000
    heartbeat_interval_ms: int = 10_000
    harness_reclaim_interval_s: float = 1.0
    """Minimum seconds between WorkerHarness expired-lease reclaim sweeps."""

    expect_external_workers: bool = False
    """Set True only when external WorkerHarness processes compete for this
    run's leases. When False (the default single-process deployment), resume()
    force-reclaims every RUNNING node immediately — any surviving lease must
    belong to the dead prior process — instead of waiting out the lease TTL. A
    genuine multi-process topology keeps the TTL slack so a still-live peer's
    node is not yanked mid-flight."""

    # Compaction --------------------------------------------------------------
    compaction_max_active_claims: int = 200
    compaction_min_claim_confidence: float = 0.2
    compaction_token_threshold: int = 5_000
    claim_store_load_limit: int = 5_000
    """Maximum active claims loaded into memory at runtime startup."""

    # Escalation --------------------------------------------------------------
    local_to_global_threshold: int = 2
    """LOCAL replans for one node before escalation is classified as GLOBAL."""

    max_replans_per_node: int = 3
    """Same-node-id replan reuse cap before the failed chain is abandoned."""

    max_replans_per_run: int = 50
    """Run-wide cap across all LOCAL and GLOBAL replan applications."""

    # Misc --------------------------------------------------------------------
    prompt_language: str = "en"
    """Inherited from v3 — surfaced to LLM-using hooks."""

    persist_to_db: bool = True
    """If False, run state stays in memory only. Useful for tests."""

    def build_compaction_strategy(self) -> CompactionStrategy:
        return CompactionStrategy(
            max_active_claims=self.compaction_max_active_claims,
            min_claim_confidence=self.compaction_min_claim_confidence,
            incremental_token_threshold=self.compaction_token_threshold,
        )

    def validate(self) -> None:
        if self.parallelism < 1:
            raise ValueError("parallelism must be >= 1")
        if self.lease_ttl_ms <= 0:
            raise ValueError("lease_ttl_ms must be positive")
        if self.heartbeat_interval_ms >= self.lease_ttl_ms:
            raise ValueError(
                "heartbeat_interval_ms must be < lease_ttl_ms"
            )
        if self.poll_interval_s < 0:
            raise ValueError("poll_interval_s must be >= 0")
        if self.max_loop_iterations < 1:
            raise ValueError("max_loop_iterations must be >= 1")
        if self.budget_flush_every_n_iterations < 0:
            raise ValueError("budget_flush_every_n_iterations must be >= 0")
        if self.max_replans_per_node < 1:
            raise ValueError("max_replans_per_node must be >= 1")
        if self.max_replans_per_run < 1:
            raise ValueError("max_replans_per_run must be >= 1")
        if self.harness_reclaim_interval_s < 0:
            raise ValueError("harness_reclaim_interval_s must be >= 0")
        if self.claim_store_load_limit < 1:
            raise ValueError("claim_store_load_limit must be >= 1")


__all__ = ["ConfigV5"]
