"""Status vocabulary, exit-code mapping, and ledger assembly for the audit.

The exit code is the ONLY thing :func:`core.ccx.sgar.checks.run_criterion_check`
observes (0 == pass). The JSON ledger is human/operator evidence captured in the
check's output tail. Keep the mapping in one place so the CLI and any future
caller agree on what each status means.
"""

from __future__ import annotations

from typing import Any

SCHEMA = "ccx.code_task_audit/v1"

#: Done satisfied, or self-gated trivial pass — exit 0.
STATUS_FIXED = "FIXED"
STATUS_NO_CODE_TASK = "NO_CODE_TASK"
#: A real, attributable failure — exit 1.
STATUS_INCOMPLETE = "INCOMPLETE"
STATUS_NEEDS_REVIEW = "NEEDS_REVIEW"
#: The audit itself could not run (harness defect) — exit 2.
STATUS_AUDIT_ERROR = "AUDIT_ERROR"

_PASS_STATUSES = frozenset({STATUS_FIXED, STATUS_NO_CODE_TASK})


def exit_code_for(status: str) -> int:
    """Map an audit ``status`` to a process exit code.

    0 = pass (FIXED / NO_CODE_TASK); 2 = harness defect (AUDIT_ERROR, mirrors
    ``check_unrunnable``'s "could not execute" notion); 1 = everything else
    (INCOMPLETE / NEEDS_REVIEW), i.e. a genuine "not done" verdict.
    """
    if status in _PASS_STATUSES:
        return 0
    if status == STATUS_AUDIT_ERROR:
        return 2
    return 1


def empty_changed() -> dict[str, list[str]]:
    return {"prod_py": [], "test_py": [], "new_files": [], "untracked": []}


def make_ledger(
    status: str,
    summary: str,
    *,
    base_ref: str = "HEAD",
    changed: dict[str, Any] | None = None,
    wiring: dict[str, Any] | None = None,
    suite: dict[str, Any] | None = None,
    failure_first: dict[str, Any] | None = None,
    mutation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a ledger dict with every key always present (stable schema)."""
    return {
        "schema": SCHEMA,
        "status": status,
        "summary": summary,
        "base_ref": base_ref,
        "changed": changed if changed is not None else empty_changed(),
        "wiring": wiring
        if wiring is not None
        else {"mode": "enforce", "symbols": [], "unwired_fail": 0, "unwired_warn": 0},
        "suite": suite
        if suite is not None
        else {
            "cmd": None,
            "returncode": None,
            "passed": None,
            "collection_error": False,
            "tail": "",
        },
        "failure_first": failure_first
        if failure_first is not None
        else {"required": False, "prod_without_test": [], "passed": True},
        "mutation": mutation if mutation is not None else {"enabled": False, "status": "skipped"},
    }


__all__ = [
    "SCHEMA",
    "STATUS_AUDIT_ERROR",
    "STATUS_FIXED",
    "STATUS_INCOMPLETE",
    "STATUS_NEEDS_REVIEW",
    "STATUS_NO_CODE_TASK",
    "empty_changed",
    "exit_code_for",
    "make_ledger",
]
