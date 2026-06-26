"""Sidecar collector for cross-node findings exchange.

ccx's doc mode decomposes a goal into N parallel investigator nodes
plus a synthesizer node. The synthesizer's input is the union of the
investigators' findings — but v5's dispatcher only uses ``depends_on``
for ordering; it does NOT automatically inject completed predecessor
results into a dependent's ``params`` (see
``core/deepstack_v5/execution/dispatcher.py:243`` and
``core/deepstack_v5/engine.py:402``).

So we need a sidecar: every investigator pushes its structured finding
keyed by ``run_id`` + ``dimension_id``; the synthesizer pops the full
list when it starts. The collector lives in the run-scoped
``CcxRuntimeBundle`` so its lifecycle = one ccx run.

Thread-safe push / pop because v5 may run investigators concurrently
across worker threads.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FindingsCollector:
    _by_run: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def push(
        self,
        run_id: str,
        dimension_id: str,
        findings: dict[str, Any],
    ) -> None:
        """Record findings from one investigator. Last write wins for
        the (run_id, dimension_id) tuple — investigators are expected to
        be single-shot.
        """
        if not run_id or not dimension_id:
            return
        with self._lock:
            bucket = self._by_run.setdefault(run_id, {})
            if dimension_id in bucket:
                logger.warning(
                    "findings collector overwriting duplicate dimension_id=%s "
                    "for run_id=%s",
                    dimension_id,
                    run_id,
                )
            bucket[dimension_id] = dict(findings)

    def pop_all(self, run_id: str) -> list[dict[str, Any]]:
        """Drain all findings for a run. Returns a list ordered by
        dimension_id so the synthesizer prompt is deterministic.

        Subsequent calls for the same ``run_id`` return an empty list.
        """
        if not run_id:
            return []
        with self._lock:
            bucket = self._by_run.pop(run_id, None)
        if not bucket:
            return []
        return [bucket[k] for k in sorted(bucket.keys())]

    def peek(self, run_id: str) -> list[dict[str, Any]]:
        """Non-destructive read of a run's findings. Used by tests."""
        if not run_id:
            return []
        with self._lock:
            bucket = self._by_run.get(run_id)
            if not bucket:
                return []
            return [dict(bucket[k]) for k in sorted(bucket.keys())]


__all__ = ["FindingsCollector"]
