"""Monotone convergence measure for governed repair loops (default OFF).

The governed spawn / goal / run loops decide whether to keep re-driving by
a PROGRESS signal: a round "made progress" if it moved the loop closer to
"all checks green". The default signal is *count-delta* — the number of
failing checks went DOWN versus the previous round. That is a stall
DETECTOR, not a convergence PROOF: an oscillating repair (fixing check B
regresses check A, then the next round fixes A and regresses B) keeps the
failing COUNT bouncing 2↔3 forever, resetting the no-progress counter every
other round, so the loop only ever terminates on the ``max_iters`` clock.
Raise that clock (a large iteration budget) and such a loop never
terminates — the budget is the only thing standing between it and
non-termination.

Enabling ``CCX_MONOTONE_PROGRESS`` swaps the count-delta signal for a
strictly-monotone one: a round makes progress iff it SATISFIED A CHECK THAT
WAS NEVER SATISFIED BEFORE. The set of ever-passed criteria only grows and
is bounded by the criteria count, so at most ``len(criteria)`` rounds can
report progress; between any two the no-progress counter climbs toward
``no_progress_stop``. Termination is therefore guaranteed in
``O(len(criteria) * no_progress_stop)`` rounds INDEPENDENT of ``max_iters``
— a real convergence proof. A check that flips back to failing no longer
resets the counter, because re-passing an already-ever-passed check does
not grow the set.

This is strictly a *tightening* on oscillation, never a false stall on
genuine forward motion: the only round the monotone measure scores as
"no progress" while count-delta scores "progress" is a RE-pass of a
previously-passed check — exactly the oscillation we want to stop. A check
passing for the first time is progress under both.

Default OFF keeps every governed loop byte-identical: the tracker is only
consulted inside an ``if monotone_progress_enabled()`` branch; the ``else``
branch is the unchanged count-delta code.
"""

from __future__ import annotations

import os
from typing import Iterable

_ENV = "CCX_MONOTONE_PROGRESS"


def monotone_progress_enabled() -> bool:
    """True when ``CCX_MONOTONE_PROGRESS`` opts the monotone measure in.

    Truthy = ``1`` / ``true`` / ``yes`` / ``on`` (case-insensitive). Any
    other value — and the unset default — is OFF, so governed loops keep
    their byte-identical count-delta behaviour.
    """
    return os.environ.get(_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


class EverPassedTracker:
    """Tracks the union of criterion ids that have passed in ANY round.

    :meth:`observe` folds one round's passed-criteria ids into the running
    set and returns whether the set GREW — i.e. at least one
    never-before-passed check passed this round. That boolean is the
    monotone progress signal. The set only grows and is bounded by the
    total criteria count, so ``True`` is returned at most ``len(criteria)``
    times over the loop's whole life regardless of how the failing COUNT
    oscillates.
    """

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


__all__ = ["monotone_progress_enabled", "EverPassedTracker"]
