"""Mutation kill-test engine — the shared "does this test have teeth?" core.

A mutation kill-test turns an authoritative decision point into a RUBBER STAMP
(a one-line edit that makes a gate always pass / drops a clamp / fakes success)
inside an isolated git worktree, then runs that point's own test suite:

* tests go **RED**   → the behaviour has TEST TEETH (a regression that silently
  removed the gate would be caught);
* tests stay **GREEN** → a TEST BLIND SPOT (correct today, but nothing pins it;
  a refactor could quietly turn it into a stamp and no test would notice).

This module is the *mechanics* only — anchor check → apply → run tests →
**always revert** → label RED/GREEN. It is deliberately **campaign-agnostic**:

* the mutation list is DATA the caller passes in. No campaign's mutations are
  ever hardcoded here (that would ossify a general engine into a one-shot
  script);
* how a result is *recorded* (finding prose, verdict string, ledger path) is the
  caller's job, supplied via the ``on_result`` hook. The three historical
  drivers (``scripts/ccx_gate_mutation.py``, ``scripts/sgar_gate_mutation.py``,
  ``scripts/ccx_memory_mutation.py``) disagree on all of those — verdict
  mappings differ (``confirmed`` vs ``by_design``), ledger paths differ, prose
  differs — so only the mechanics live here.

Worktree handling is raw ``git`` subprocess (matching the original scripts). We
do NOT import the cc-side worktree tools: ccx must never depend on cc, and the
per-file revert here is the cheapest possible real rollback. **Nothing is ever
committed**; every mutation is reverted with ``git checkout -- <file>`` in a
``finally`` so the tree returns to HEAD even if the test run raises.
"""

from __future__ import annotations

import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

__all__ = [
    "Mutation",
    "MutationResult",
    "run_mutation_campaign",
    "filter_mutations",
    "is_git_worktree",
    "ephemeral_worktree",
]


@dataclass(frozen=True)
class Mutation:
    """One rubber-stamp edit and the tests that should catch it.

    The canonical field is ``track`` (the historical ``sgar_gate_mutation.py``
    called it ``gate``; that driver normalises to ``track`` when it builds these).
    """

    name: str           # short id, e.g. "M1_goal_met_always_true"
    track: str          # finding track / gate family this mutation probes
    file: str           # repo-relative path to mutate (within the worktree)
    old: str            # exact source to replace — MUST be unique in the file
    new: str            # rubber-stamp replacement
    tests: list[str]    # test files to run as the teeth oracle
    rationale: str      # what teeth this proves


@dataclass
class MutationResult:
    """Neutral per-mutation outcome. Recording/verdict policy is the caller's.

    ``applied is False`` means the anchor was not found exactly once (source
    drifted); ``rc``/``red``/``is_blind_spot`` are then not meaningful and the
    caller should record an anchor-miss instead of a teeth verdict.

    ``unproven``/``timed_out`` mean the mutation WAS applied but the teeth
    oracle rendered no verdict (a non-running pytest — usage/collection error
    or no tests collected — or a timeout). ``red`` and ``is_blind_spot`` are
    then both ``False``; the caller must refuse the mutation rather than count
    a non-verdict as proven teeth.
    """

    name: str
    track: str
    file: str
    tests: list[str] = field(default_factory=list)
    rationale: str = ""
    applied: bool = False
    anchor_count: int = 0
    rc: int | None = None
    tail: str = ""
    red: bool = False
    is_blind_spot: bool = False
    #: The teeth oracle never rendered a real verdict — pytest returned a
    #: usage/collection error or "no tests collected" (rc ∉ {0, 1}), so the
    #: mutation is neither proven RED teeth nor a GREEN blind spot. The caller
    #: must REFUSE it, not count it as teeth (see ``regression_capture``).
    unproven: bool = False
    #: The teeth run exceeded ``test_timeout_s`` — same class as ``unproven``:
    #: no verdict was rendered, an unrunnable-harness condition, not evidence
    #: of a blind spot.
    timed_out: bool = False


# ResultHook: called once per mutation (applied or anchor-miss) after revert.
ResultHook = Callable[[MutationResult, Mutation], None]
Logger = Callable[[str], None]


def is_git_worktree(path: str | Path) -> bool:
    """True if ``path`` looks like a git worktree (has a ``.git`` entry).

    A linked worktree's ``.git`` is a file (gitdir pointer), a primary clone's
    is a directory — ``exists()`` covers both.
    """
    return (Path(path) / ".git").exists()


def filter_mutations(
    mutations: Sequence[Mutation], tokens: Iterable[str]
) -> list[Mutation]:
    """Restrict to mutations whose ``name`` contains one of ``tokens``.

    Empty ``tokens`` ⇒ all mutations (identical to the drivers' ``only`` logic).
    """
    toks = [t for t in tokens]
    return [m for m in mutations if not toks or any(tok in m.name for tok in toks)]


def _run_tests(
    worktree: Path, tests: Sequence[str], *, pybin: str, timeout_s: float
) -> tuple[int, str]:
    """Run ``pytest`` on ``tests`` inside ``worktree``; return (rc, tail).

    Mirrors the drivers byte-for-byte: ``-q -p no:cacheprovider`` and a 6-line
    tail of combined stdout+stderr.
    """
    proc = subprocess.run(
        [pybin, "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests],
        cwd=str(worktree), capture_output=True, text=True, timeout=timeout_s,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    tail = "\n".join(line for line in out.strip().splitlines()[-6:])
    return proc.returncode, tail


def _git_revert(worktree: Path, file: str) -> None:
    """Restore ``file`` to HEAD in ``worktree`` — never commits, never stages."""
    subprocess.run(
        ["git", "-C", str(worktree), "checkout", "--", file],
        capture_output=True, text=True,
    )


def run_mutation_campaign(
    worktree: str | Path,
    mutations: Sequence[Mutation],
    *,
    pybin: str = sys.executable,
    test_timeout_s: float = 600.0,
    on_result: ResultHook | None = None,
    log: Logger | None = None,
) -> list[MutationResult]:
    """Apply each mutation, run its tests, revert, and label RED/GREEN.

    For each mutation:

    1. Read ``<worktree>/<file>`` and count occurrences of ``old``. If not
       exactly one ⇒ ``MutationResult(applied=False, anchor_count=...)`` and
       skip (anchor drifted — do not silently mutate the wrong spot).
    2. Write the rubber-stamp (``old`` → ``new``).
    3. ``try`` run the tests ``finally`` revert the file. The ``finally``
       guarantees the worktree returns to HEAD even on timeout / exception.
    4. Label the outcome from the pytest exit code: ``rc == 1`` ⇒ RED teeth
       (the guard caught the mutation); ``rc == 0`` ⇒ GREEN blind spot; a
       timeout ⇒ ``timed_out``; any other ``rc`` (usage/collection error, no
       tests collected) ⇒ ``unproven``. ``timed_out``/``unproven`` are NEITHER
       teeth NOR a blind spot — the oracle rendered no verdict, so the caller
       must refuse the mutation rather than count it either way.

    ``on_result`` (if given) is called once per mutation — for anchor-misses
    too — *after* revert, so a recording callback can never observe a mutated
    tree. Returns the full ``list[MutationResult]`` in input order.

    The mutation list is the caller's data; verdict/ledger policy lives in
    ``on_result``. This function hardcodes neither.
    """
    wt = Path(worktree)
    results: list[MutationResult] = []
    for m in mutations:
        target = wt / m.file
        src = target.read_text(encoding="utf-8")
        count = src.count(m.old)
        if count != 1:
            if log is not None:
                log(f"[{m.name}] SKIP — anchor found {count}x in {m.file} (need 1)")
            res = MutationResult(
                name=m.name, track=m.track, file=m.file, tests=list(m.tests),
                rationale=m.rationale, applied=False, anchor_count=count,
            )
            results.append(res)
            if on_result is not None:
                on_result(res, m)
            continue

        target.write_text(src.replace(m.old, m.new), encoding="utf-8")
        rc: int | None = None
        tail = ""
        timed_out = False
        try:
            try:
                rc, tail = _run_tests(
                    wt, m.tests, pybin=pybin, timeout_s=test_timeout_s
                )
            except subprocess.TimeoutExpired:
                # A timed-out proof is NOT evidence the guard is a blind spot —
                # the oracle never rendered a verdict. Surface it as its own
                # (unrunnable-harness) state instead of folding a timeout into
                # the RED/GREEN teeth label; ``code_task._run_suite`` handles a
                # subprocess timeout the same lenient way.
                timed_out = True
                tail = "TIMEOUT"
        finally:
            _git_revert(wt, m.file)

        # pytest exit codes: 0 = all passed (guard did NOT catch the mutation ⇒
        # blind spot); 1 = a test failed (the guard CAUGHT it ⇒ genuine teeth).
        # Any other rc (2/3 usage or collection error, 4 bad cmdline, 5 no tests
        # collected, or a killed process) means the teeth ORACLE never ran the
        # guard's assertions — neither RED teeth nor a GREEN blind spot, but an
        # UNPROVEN result the caller must refuse (a stale/renamed ``m.tests``
        # would otherwise score as proven teeth). Mirrors code_task._run_suite's
        # lenient rc==5 / collection-error handling.
        unproven = (not timed_out) and (rc not in (0, 1))
        red = rc != 0
        if timed_out or unproven:
            red = False
        res = MutationResult(
            name=m.name, track=m.track, file=m.file, tests=list(m.tests),
            rationale=m.rationale, applied=True, anchor_count=count,
            rc=rc, tail=tail, red=red,
            is_blind_spot=(not red and not unproven and not timed_out),
            unproven=unproven, timed_out=timed_out,
        )
        results.append(res)
        if log is not None:
            if timed_out:
                teeth = "UNPROVEN (timed out)"
            elif unproven:
                teeth = f"UNPROVEN (rc={rc}, no verdict)"
            elif red:
                teeth = "TEETH (caught)"
            else:
                teeth = "BLIND SPOT (uncaught)"
            log(f"[{m.name}] {m.file} -> rc={rc} :: {teeth}")
        if on_result is not None:
            on_result(res, m)

    return results


@contextmanager
def ephemeral_worktree(
    repo_root: str | Path, *, ref: str = "HEAD", suffix: str = "mut"
) -> Iterator[Path]:
    """Create a detached git worktree of ``repo_root`` at ``ref``; remove on exit.

    Yields the worktree path. The whole tree is torn down with
    ``git worktree remove --force`` in ``finally`` — the clean boundary for a
    campaign that scaffolds *new* (untracked) files, which per-file
    ``git checkout`` cannot restore. Never commits.

    Used by ``regression_capture`` to own an isolated tree for the mandatory
    teeth-proof; the historical drivers pass an operator-created worktree
    straight to :func:`run_mutation_campaign` and do not use this.
    """
    import tempfile

    repo = Path(repo_root)
    tmp = Path(tempfile.mkdtemp(prefix=f"ccx_{suffix}_wt_"))
    wt_path = tmp / "wt"
    add = subprocess.run(
        ["git", "-C", str(repo), "worktree", "add", "--detach",
         str(wt_path), ref],
        capture_output=True, text=True,
    )
    if add.returncode != 0:
        _rmtree(tmp)
        raise RuntimeError(
            f"git worktree add failed (rc={add.returncode}): "
            f"{(add.stderr or add.stdout).strip()}"
        )
    try:
        yield wt_path
    finally:
        subprocess.run(
            ["git", "-C", str(repo), "worktree", "remove", "--force",
             str(wt_path)],
            capture_output=True, text=True,
        )
        # Best-effort scrub of the temp parent (and the worktree dir if the
        # git remove left anything behind).
        _rmtree(tmp)


def _rmtree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)
