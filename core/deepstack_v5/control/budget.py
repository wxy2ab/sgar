"""BudgetTracker — global token / cost / wallclock / iteration budget.

Distinct from per-attempt budgets: this tracks the run as a whole and is
the substrate for cc's "global cost cap" requirement, plus the trigger
source for the Compaction strategy (token usage hitting `warning_ratio`
emits a `budget.warning` event).

Thread-safe: `consume()` is called from any number of dispatcher threads.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

from ..types import Budget


WarningCallback = Callable[[Budget], None]


@dataclass(slots=True)
class BudgetSnapshot:
    """Read-only snapshot for decision logic."""
    consumed_tokens: int
    consumed_cost: float
    elapsed_s: float
    iterations: int
    is_exhausted: bool
    is_warning: bool


class BudgetTracker:
    def __init__(
        self,
        budget: Budget | None = None,
        *,
        warning_callback: WarningCallback | None = None,
    ) -> None:
        self._budget = budget if budget is not None else Budget()
        self._lock = threading.Lock()
        self._started_at = time.monotonic()
        self._warning_callback = warning_callback
        self._warning_fired = False

    @property
    def budget(self) -> Budget:
        return self._budget

    def consume(
        self,
        *,
        tokens: int = 0,
        cost: float = 0.0,
        iteration: bool = False,
    ) -> BudgetSnapshot:
        with self._lock:
            if tokens:
                self._budget.consumed_tokens += tokens
            # Use the LLM-reported price when it gave one (cost > 0). Otherwise,
            # if a token→cost price is configured, derive a cost from tokens so
            # a cost budget bites for token-only clients (reasoning models that
            # report tokens but not dollars). Deriving only when cost == 0 means
            # a real price is never double-counted, and an unset price leaves
            # consumed_cost byte-identical (the `if cost:` legacy path).
            derived = cost
            price = self._budget.cost_per_1k_tokens
            if not cost and tokens and price:
                derived = (tokens / 1000.0) * price
            if derived:
                self._budget.consumed_cost += derived
            if iteration:
                self._budget.iterations += 1
            self._budget.elapsed_s = time.monotonic() - self._started_at
            return self._snapshot_locked()

    def refund_iteration(self) -> None:
        """Undo one ``consume(iteration=True)`` ticket.

        Used when Controller returns ENQUEUE but dispatch starts no nodes and
        the step falls back to idle WAIT (parking / in-flight leases). Without
        a refund, ``max_iterations`` would burn while merely waiting.
        """
        with self._lock:
            if self._budget.iterations > 0:
                self._budget.iterations -= 1
            self._budget.elapsed_s = time.monotonic() - self._started_at

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            self._budget.elapsed_s = time.monotonic() - self._started_at
            return self._snapshot_locked()

    def restore(self, data: dict | None) -> None:
        """Restore persisted budget counters during resume."""
        if not data:
            return
        with self._lock:
            self._budget.max_tokens = data.get("max_tokens")
            self._budget.max_cost = data.get("max_cost")
            self._budget.max_wallclock_s = data.get("max_wallclock_s")
            self._budget.max_iterations = data.get("max_iterations")
            warning_ratio = data.get("warning_ratio", 0.8)
            if warning_ratio is None or warning_ratio == "":
                warning_ratio = 0.8
            self._budget.warning_ratio = float(warning_ratio)
            # Round-trip the optional price so a resumed run keeps deriving
            # cost the same way (None for snapshots that predate the field).
            price = data.get("cost_per_1k_tokens")
            self._budget.cost_per_1k_tokens = (
                float(price) if price is not None else None
            )
            self._budget.consumed_tokens = int(data.get("consumed_tokens") or 0)
            self._budget.consumed_cost = float(data.get("consumed_cost") or 0.0)
            self._budget.elapsed_s = float(data.get("elapsed_s") or 0.0)
            self._budget.iterations = int(data.get("iterations") or 0)
            self._started_at = time.monotonic() - self._budget.elapsed_s
            self._warning_fired = self._budget.is_warning()

    def absorb_remote(
        self,
        data: dict | None,
        *,
        include_iterations: bool = False,
    ) -> None:
        """Raise local consumed_* counters to at least a remote (DB) snapshot.

        Used by the engine to observe WorkerHarness usage written via
        ``RunStore.increment_budget_usage`` without clobbering engine-owned
        fields (``iterations``, ``max_*``). Monotonic on
        ``consumed_tokens`` / ``consumed_cost`` / ``elapsed_s`` only by
        default.

        Pass ``include_iterations=True`` for WorkerHarness refresh so the
        engine's ENQUEUE ticket count (persisted via
        ``persist_budget_snapshot``) can trip ``max_iterations`` /
        ``should_halt`` before the worker leases more work. When set,
        ``iterations`` is mirrored from the remote snapshot (including
        decreases after idle ``refund_iteration``); tokens/cost/elapsed stay
        monotonic. Engine callers keep the default so a local
        ``refund_iteration`` cannot be undone by a stale higher DB value.

        Does **not** set ``_warning_fired``: harness ``budget.warning`` events
        stay on the worker process bus and never reach the engine. Leaving the
        flag clear lets ``fire_warning_if_needed`` run the engine-local
        compaction callback when remote spend first crosses the band.
        """
        if not data:
            return
        with self._lock:
            remote_tokens = int(data.get("consumed_tokens") or 0)
            remote_cost = float(data.get("consumed_cost") or 0.0)
            remote_elapsed = float(data.get("elapsed_s") or 0.0)
            if remote_tokens > self._budget.consumed_tokens:
                self._budget.consumed_tokens = remote_tokens
            if remote_cost > self._budget.consumed_cost:
                self._budget.consumed_cost = remote_cost
            if remote_elapsed > self._budget.elapsed_s:
                self._budget.elapsed_s = remote_elapsed
                self._started_at = time.monotonic() - remote_elapsed
            if include_iterations:
                # Engine-authoritative, including idle ENQUEUE refunds.
                self._budget.iterations = int(data.get("iterations") or 0)

    def _snapshot_locked(self) -> BudgetSnapshot:
        return BudgetSnapshot(
            consumed_tokens=self._budget.consumed_tokens,
            consumed_cost=self._budget.consumed_cost,
            elapsed_s=self._budget.elapsed_s,
            iterations=self._budget.iterations,
            is_exhausted=self._budget.is_exhausted(),
            is_warning=self._budget.is_warning(),
        )

    def fire_warning_if_needed(self) -> bool:
        """Fires `warning_callback` once if budget crosses warning ratio.

        Separate from snapshot() to keep that side-effect free.
        Returns True if the callback fired this call.
        """
        snap = self.snapshot()
        with self._lock:
            if snap.is_warning and not self._warning_fired:
                self._warning_fired = True
                fire = True
            else:
                fire = False
        if fire and self._warning_callback is not None:
            self._warning_callback(self._budget)
        return fire

    def should_halt(self, *, ignore_iterations: bool = False) -> bool:
        """True when the run budget is exhausted.

        ``ignore_iterations=True`` is for the ENQUEUE batch that just paid an
        iteration ticket: that ticket must still be allowed to dispatch, while
        tokens/cost/wallclock gates remain active mid-batch.
        """
        snap = self.snapshot()
        if ignore_iterations:
            return self._budget.is_exhausted(ignore_iterations=True)
        return snap.is_exhausted


__all__ = ["BudgetSnapshot", "BudgetTracker", "WarningCallback"]
