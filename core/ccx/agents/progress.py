"""Monotone convergence measure for governed repair loops (default ON).

The governed spawn / goal / run loops decide whether to keep re-driving by
a PROGRESS signal. The legacy signal is *count-delta* (failing-check count
went DOWN). That detects stalls but does not prove convergence: an
oscillating repair can bounce the count forever and only stop on
``max_iters``.

Default (ON) uses a strictly-monotone measure:

* progress iff a check passed that had NEVER passed before.

(Failing-set shrink alone is intentionally NOT progress: an oscillating
2↔1 count would otherwise reset the no-progress counter forever.)

Opt OUT with ``CCX_MONOTONE_PROGRESS=0`` / ``false`` / ``off`` / ``no`` to
restore the legacy count-delta signal.
"""

from __future__ import annotations

import os
from typing import Iterable

_ENV = "CCX_MONOTONE_PROGRESS"
_FALSEY = frozenset({"0", "false", "no", "off"})


def monotone_progress_enabled() -> bool:
    """True unless ``CCX_MONOTONE_PROGRESS`` explicitly opts the measure out.

    Default ON (unset or empty ⇒ monotone). Falsy = ``0`` / ``false`` /
    ``no`` / ``off`` (case-insensitive).
    """
    raw = os.environ.get(_ENV)
    if raw is None:
        return True
    text = raw.strip().lower()
    if text == "":
        return True
    return text not in _FALSEY


class EverPassedTracker:
    """Tracks the union of criterion ids that have passed in ANY round."""

    __slots__ = ("_ever",)

    def __init__(self) -> None:
        self._ever: set[str] = set()

    def observe(self, passed_ids: Iterable[str]) -> bool:
        """Fold ``passed_ids`` into the ever-passed set; return True if it grew."""
        before = len(self._ever)
        self._ever.update(pid for pid in passed_ids if pid is not None)
        return len(self._ever) > before

    @property
    def ever_passed(self) -> frozenset[str]:
        return frozenset(self._ever)


def observe_progress(
    *,
    monotone: bool,
    ever_tracker: EverPassedTracker | None,
    prev_failing_ids: frozenset[str] | None,
    passed_ids: Iterable[str],
    failing_ids: Iterable[str],
    prev_failing_count: int | None,
) -> tuple[bool, frozenset[str]]:
    """Return ``(made_progress, failing_ids_frozenset)``.

    * Monotone ON: progress iff ever-passed grew (a check passed that had
      never passed before). Failing-set shrink alone is NOT progress — an
      oscillating 2↔1 failing count would otherwise reset forever.
    * Monotone OFF: progress iff failing count decreased (legacy).

    On the first failing round (no previous baseline) returns
    ``(True, failing)`` so callers do not yet increment ``no_progress``.
    """
    failing = frozenset(fid for fid in failing_ids if fid is not None)
    if not monotone:
        if prev_failing_count is None:
            return True, failing
        return len(failing) < prev_failing_count, failing

    newly = False
    if ever_tracker is not None:
        newly = ever_tracker.observe(passed_ids)
    if prev_failing_ids is None and prev_failing_count is None:
        return True, failing
    return newly, failing


__all__ = [
    "EverPassedTracker",
    "monotone_progress_enabled",
    "observe_progress",
]
