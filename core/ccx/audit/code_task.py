"""Deterministic CODE-task definition-of-done audit CLI.

    python -m core.ccx.audit.code_task [--cwd .] [--base HEAD] [--test-cmd CMD] ...

Run over a git working tree, it decides whether a code task is *done*:

* **self-gating** — no ``.py`` changed ⇒ ``NO_CODE_TASK`` (exit 0); only test
  ``.py`` changed ⇒ ``NEEDS_REVIEW`` (exit 1, closes the "gut the test to force
  green" cheat); production ``.py`` changed ⇒ run the full audit.
* **A (wiring)** — a new module nobody imports hard-fails; unwired new symbols
  in existing files warn (see :mod:`core.ccx.audit.wiring`).
* **C (suite)** — a scoped pytest must exit 0; collection/import errors fail.
* **B (failure-first)** — production change without an accompanying test change
  warns (advisory; never the sole reason to fail).

Exit code is the trust root (``run_criterion_check`` reads only 0 vs non-0):
0 = FIXED / NO_CODE_TASK, 1 = INCOMPLETE / NEEDS_REVIEW, 2 = AUDIT_ERROR.
The JSON ledger is printed to stdout (compact one line) followed by a
``SUMMARY:`` line, so the most important verdict survives the bounded evidence
tail ``run_criterion_check`` captures.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from typing import Any

from . import CODE_TASK_TEST_CMD_ENV
from . import gitdiff
from .ledger import (
    STATUS_AUDIT_ERROR,
    STATUS_FIXED,
    STATUS_INCOMPLETE,
    STATUS_NEEDS_REVIEW,
    STATUS_NO_CODE_TASK,
    exit_code_for,
    make_ledger,
)
from .wiring import analyze_wiring

_EVIDENCE_TAIL_LINES = 20
_EVIDENCE_MAX_CHARS = 2000


def _tail(text: str) -> str:
    if not text:
        return ""
    tail = "\n".join(text.strip().splitlines()[-_EVIDENCE_TAIL_LINES:])
    return tail[-_EVIDENCE_MAX_CHARS:]


def _changed_block(changes: gitdiff.ChangeSet) -> dict[str, list[str]]:
    return {
        "prod_py": [f.path for f in changes.prod_py],
        "test_py": [f.path for f in changes.test_py],
        "new_files": sorted(f.path for f in changes.files if f.is_new),
        "untracked": sorted(f.path for f in changes.files if f.status == "untracked"),
    }


# --------------------------------------------------------------------------- #
# Criterion C — scoped test suite
# --------------------------------------------------------------------------- #

def _nearest_tests_dir(path: str) -> str | None:
    parts = path.replace("\\", "/").split("/")
    for i, seg in enumerate(parts):
        if seg in ("tests", "test"):
            return "/".join(parts[: i + 1])
    return os.path.dirname(path) or None


def _package_tests_dir(path: str) -> str | None:
    parts = path.replace("\\", "/").split("/")
    if len(parts) >= 2 and parts[0] == "core":
        return f"{parts[0]}/{parts[1]}/tests"
    if parts and parts[0]:
        return f"{parts[0]}/tests"
    return None


def _scoped_test_dirs(cwd: str, changes: gitdiff.ChangeSet) -> list[str]:
    dirs: set[str] = set()
    for fc in changes.test_py:
        d = _nearest_tests_dir(fc.path)
        if d:
            dirs.add(d)
    for fc in changes.prod_py:
        d = _package_tests_dir(fc.path)
        if d:
            dirs.add(d)
    return sorted(d for d in dirs if os.path.isdir(os.path.join(str(cwd), d)))


def _derive_test_argv(cwd: str, changes: gitdiff.ChangeSet, args: argparse.Namespace) -> list[str] | None:
    if args.test_cmd:
        return shlex.split(args.test_cmd)
    env_cmd = os.environ.get(CODE_TASK_TEST_CMD_ENV)
    if env_cmd:
        return shlex.split(env_cmd)
    dirs = _scoped_test_dirs(cwd, changes)
    if not dirs:
        return None
    return [sys.executable, "-m", "pytest", *dirs, "-q", "-p", "no:cacheprovider"]


def _run_suite(cwd: str, changes: gitdiff.ChangeSet, args: argparse.Namespace) -> dict[str, Any]:
    argv = _derive_test_argv(cwd, changes, args)
    if argv is None:
        return {
            "cmd": None,
            "returncode": None,
            "passed": None,
            "collection_error": False,
            "tail": "",
            "note": "no test scope derived (no changed tests, no package tests dir)",
        }
    # The inner test run legitimately needs to import the repo under test. The
    # audit itself runs under the hermetic env (PYTHONSAFEPATH set, PYTHONPATH
    # dropped); re-establish import-ability for the CHILD by putting the
    # workspace on PYTHONPATH and clearing PYTHONSAFEPATH for it only.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(cwd), env.get("PYTHONPATH", "")) if p
    )
    env.pop("PYTHONSAFEPATH", None)
    try:
        proc = subprocess.run(
            argv,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=args.timeout_s,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {
            "cmd": argv,
            "returncode": None,
            "passed": False,
            "collection_error": False,
            "tail": "TIMEOUT",
        }
    except (OSError, ValueError) as exc:
        return {
            "cmd": argv,
            "returncode": None,
            "passed": False,
            "collection_error": False,
            "tail": str(exc),
            "error": True,
        }
    rc = proc.returncode
    combined = (proc.stdout or "") + (proc.stderr or "")
    if rc == 5:  # pytest: no tests collected — lenient, don't false-fail a scope miss
        return {
            "cmd": argv,
            "returncode": rc,
            "passed": None,
            "collection_error": False,
            "tail": _tail(combined),
            "note": "no tests collected for the derived scope",
        }
    collection_error = rc == 2 or "errors during collection" in combined.lower()
    return {
        "cmd": argv,
        "returncode": rc,
        "passed": rc == 0,
        "collection_error": collection_error,
        "tail": _tail(combined),
    }


# --------------------------------------------------------------------------- #
# Criterion B — failure-first proxy (advisory)
# --------------------------------------------------------------------------- #

def _failure_first(changes: gitdiff.ChangeSet, args: argparse.Namespace) -> dict[str, Any]:
    if not args.require_test_per_prod:
        return {"required": False, "prod_without_test": [], "passed": True}
    has_test_change = bool(changes.test_py)
    missing: list[str] = []
    for fc in changes.prod_py:
        if fc.path.endswith("__init__.py"):
            continue
        if not fc.added_symbols and not fc.is_new:
            continue  # comment/format-only change ⇒ no test obligation
        if not has_test_change:
            missing.append(fc.path)
    return {
        "required": True,
        "prod_without_test": sorted(missing),
        "passed": not missing,
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    cwd = os.path.abspath(args.cwd)

    if not gitdiff.is_inside_work_tree(cwd):
        # Not a git tree ⇒ cannot diff. Do NOT block (the criterion is injected
        # broadly): pass trivially as "no code task".
        return make_ledger(
            STATUS_NO_CODE_TASK,
            "not a git work tree; nothing to audit",
            base_ref=args.base,
        )

    changes = gitdiff.collect_changes(cwd, args.base)
    changed = _changed_block(changes)
    has_prod = bool(changes.prod_py)
    has_test = bool(changes.test_py)

    if not args.force:
        if not has_prod and not has_test:
            return make_ledger(
                STATUS_NO_CODE_TASK,
                "no .py changes vs base; not a code task",
                base_ref=changes.base_ref,
                changed=changed,
            )
        if has_test and not has_prod:
            return make_ledger(
                STATUS_NEEDS_REVIEW,
                "only test .py changed and no production .py — possible test-gaming, needs review",
                base_ref=changes.base_ref,
                changed=changed,
            )

    wiring = analyze_wiring(cwd, changes)
    if args.wiring_mode == "warn":
        wiring["unwired_warn"] += wiring["unwired_fail"]
        wiring["unwired_fail"] = 0
        wiring["mode"] = "warn"
        for sym in wiring["symbols"]:
            if sym.get("verdict") == "UNWIRED_FAIL":
                sym["verdict"] = "UNWIRED_WARN"

    suite = _run_suite(cwd, changes, args)
    failure_first = _failure_first(changes, args)

    if suite.get("error"):
        status = STATUS_AUDIT_ERROR
        summary = f"audit could not run the test suite: {suite.get('tail', '')[:120]}"
    elif wiring["unwired_fail"] > 0 or suite.get("passed") is False:
        status = STATUS_INCOMPLETE
        reasons = []
        if wiring["unwired_fail"] > 0:
            reasons.append(f"{wiring['unwired_fail']} new module(s) with no production importer")
        if suite.get("passed") is False:
            rc = suite.get("returncode")
            reasons.append(
                "scoped test suite "
                + ("collection/import error" if suite.get("collection_error") else f"failed (exit={rc})")
            )
        summary = "INCOMPLETE: " + "; ".join(reasons)
    else:
        status = STATUS_FIXED
        summary = (
            f"FIXED: wiring ok ({wiring['unwired_warn']} warning(s)), "
            f"suite {'green' if suite.get('passed') else suite.get('note', 'n/a')}"
        )

    return make_ledger(
        status,
        summary,
        base_ref=changes.base_ref,
        changed=changed,
        wiring=wiring,
        suite=suite,
        failure_first=failure_first,
        mutation={"enabled": args.mutation, "status": "skipped"},
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="core.ccx.audit.code_task")
    parser.add_argument("--cwd", default=".")
    parser.add_argument("--base", default="HEAD")
    parser.add_argument("--test-cmd", dest="test_cmd", default=None)
    parser.add_argument(
        "--wiring-mode", dest="wiring_mode", choices=("enforce", "warn"), default="enforce",
    )
    parser.add_argument(
        "--require-test-per-prod",
        dest="require_test_per_prod",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--mutation", action="store_true", default=False)
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--timeout-s", dest="timeout_s", type=float, default=120.0)
    parser.add_argument("--format", choices=("json", "text"), default="json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    ledger = run_audit(args)
    if args.format == "text":
        print(ledger["summary"])
    else:
        # Compact (single line) so the trailing SUMMARY line is what the bounded
        # evidence tail keeps last.
        print(json.dumps(ledger, sort_keys=True))
        print("SUMMARY: " + ledger["summary"])
    return exit_code_for(ledger["status"])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
