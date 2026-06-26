"""Criterion A — wiring analysis: is new code actually reachable from production?

Two strengths, deliberately calibrated to keep the false-FAIL rate low (we'd
rather miss a subtle shelf-ware case than block a legitimate run):

* **HARD gate** — a brand-new, non-test, non-``__init__`` ``.py`` *module* that
  NOTHING imports anywhere (tracked or untracked, excluding test files and the
  module itself). This is the exact "wrote ``regime_aware_weighting.py``, nobody
  imports it" shelf-ware shape from the postmortem.
* **WARNING** — a new public top-level ``def``/``class`` *added to an existing*
  file that has no production reference. Recorded in the ledger, never fails the
  exit code: ``git grep`` cannot see re-exports / decorator registries /
  string-or-getattr dispatch / framework hooks reliably, so a hard gate here
  would false-fail common patterns.

Reference search errs toward "wired": any word-boundary hit in a non-test,
non-defining ``.py`` counts. A generic name that coincidentally matches elsewhere
is treated as wired (a false PASS we accept to avoid false FAILs).
"""

from __future__ import annotations

import subprocess
from typing import Any

from .gitdiff import ChangeSet, is_test_path


def _grep_files(cwd: str, token: str) -> set[str]:
    """Files (``*.py``) containing ``token`` as a whole word. Tracked + untracked.

    Prefers ``git grep --untracked`` (so a same-run sibling file counts as a call
    site); falls back to ripgrep, then to an empty set. Returns relative paths.
    """
    p = subprocess.run(
        ["git", "grep", "-I", "-l", "-w", "--untracked", "-e", token, "--", "*.py"],
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    if p.returncode in (0, 1):  # 0 = matches, 1 = no matches (both are "ran ok")
        return {ln for ln in p.stdout.splitlines() if ln}
    rg = subprocess.run(
        ["rg", "-l", "-t", "py", "-w", "--", token],
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    if rg.returncode in (0, 1):
        return {ln for ln in rg.stdout.splitlines() if ln}
    return set()


def _production_refs(cwd: str, token: str, *, defining_file: str) -> list[str]:
    refs = _grep_files(cwd, token)
    refs.discard(defining_file)
    return sorted(r for r in refs if not is_test_path(r))


def _module_basename(path: str) -> str:
    return path.replace("\\", "/").rsplit("/", 1)[-1][:-3]  # strip dir + ".py"


def analyze_wiring(cwd: str, changes: ChangeSet) -> dict[str, Any]:
    """Return the ledger ``wiring`` block (``mode`` is set by the caller's flag)."""
    symbols: list[dict[str, Any]] = []
    unwired_fail = 0
    unwired_warn = 0
    failed_module_files: set[str] = set()

    # HARD gate: new non-test modules with zero importers.
    for fc in changes.new_prod_modules:
        modname = _module_basename(fc.path)
        prod_refs = _production_refs(cwd, modname, defining_file=fc.path)
        wired = bool(prod_refs)
        if not wired:
            unwired_fail += 1
            failed_module_files.add(fc.path)
        symbols.append(
            {
                "name": modname,
                "file": fc.path,
                "kind": "module",
                "call_sites": prod_refs,
                "wired": wired,
                "verdict": "WIRED" if wired else "UNWIRED_FAIL",
            }
        )

    # WARNING: new public symbols added to existing (modified) files.
    for fc in changes.prod_py:
        if fc.is_new or fc.path.endswith("__init__.py"):
            continue  # new files covered by the module gate above
        for sym in fc.added_symbols:
            if sym.startswith("_"):
                continue  # intentionally private; often wired later in the task
            if _production_refs(cwd, sym, defining_file=fc.path):
                continue  # wired ⇒ no warning, keep the ledger compact
            unwired_warn += 1
            symbols.append(
                {
                    "name": sym,
                    "file": fc.path,
                    "kind": "symbol",
                    "call_sites": [],
                    "wired": False,
                    "verdict": "UNWIRED_WARN",
                }
            )

    return {
        "mode": "enforce",
        "symbols": symbols,
        "unwired_fail": unwired_fail,
        "unwired_warn": unwired_warn,
    }


__all__ = ["analyze_wiring"]
