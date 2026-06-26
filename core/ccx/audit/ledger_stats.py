"""Read-only metrics over a finding ledger (theory §14).

The ratchet accumulates findings in an append-only JSONL ledger
(:mod:`core.ccx.audit.finding_ledger`). This module reads that ledger and
computes the §14 audit metrics over it — it never writes, and (unlike the
auditing machinery in :mod:`core.ccx.audit`) it is *not* behind an env gate:
reporting on what already happened is always safe.

Metrics (all derived per distinct ``(file, func)`` location, folding records in
timestamp order via the same lifecycle reducer the ledger uses):

* **Recurrence Rate** — share of locations that ever reached
  ``regression_passed`` and later *reopened* (a defect family that returned
  despite a resident guard).
* **Regression Pass Rate** — share of passed locations whose fix has *held*
  (``1 − recurrence``): the stability of old fixes.
* **Over-Audit Rate** — ``false_positive`` records over all records. (On the
  promotion pipeline ``false_positive`` is overloaded for teeth-proof failures,
  so this is cleanest on a pure probe ledger; the raw count is reported too.)
* Counts by ``mismatch_type`` / ``severity`` (theory §3 controlled fields).

CLI::

    python -m core.ccx.audit.ledger_stats <ledger.jsonl>
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from .finding_ledger import (
    _advance_lifecycle,
    _lifecycle_signal,
    read_findings,
)

__all__ = [
    "group_by_location",
    "location_history",
    "recurrence_rate",
    "regression_pass_rate",
    "over_audit_rate",
    "counts_by_mismatch_type",
    "counts_by_severity",
    "summarize",
    "main",
]


def group_by_location(
    findings: list[dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Group records by their ``evidence.code_location`` ``(file, func)`` key.

    Records without a well-formed code-location are dropped (they carry no
    cross-run identity to attribute to a defect family).
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for rec in findings:
        loc = (rec.get("evidence") or {}).get("code_location")
        if isinstance(loc, dict) and loc.get("file") and loc.get("func"):
            groups.setdefault((loc["file"], loc["func"]), []).append(rec)
    return groups


def location_history(recs: list[dict[str, Any]]) -> tuple[str, bool, int]:
    """Replay one location's records → ``(final_state, ever_passed, reopens)``.

    Uses the ledger's own signal/advance reducer so this stays the single source
    of truth. ``reopens`` counts ``open`` signals that landed while the location
    was already ``regression_passed`` — i.e. a resident guard that did not hold.
    """
    ordered = sorted(recs, key=lambda r: str(r.get("ts") or ""))
    state = "unknown"
    ever_passed = False
    reopens = 0
    for rec in ordered:
        sig = _lifecycle_signal(rec)
        prev = state
        state = _advance_lifecycle(state, sig)
        if state == "regression_passed":
            ever_passed = True
        if sig == "open" and prev == "regression_passed":
            reopens += 1
    return state, ever_passed, reopens


def _passed_and_reopened(
    findings: list[dict[str, Any]],
) -> tuple[int, int]:
    """``(passed_keys, reopened_keys)`` across all distinct locations."""
    passed = 0
    reopened = 0
    for recs in group_by_location(findings).values():
        _state, ever_passed, reopens = location_history(recs)
        if ever_passed:
            passed += 1
            if reopens > 0:
                reopened += 1
    return passed, reopened


def recurrence_rate(findings: list[dict[str, Any]]) -> float | None:
    """Reopened / passed locations. ``None`` when no location ever passed."""
    passed, reopened = _passed_and_reopened(findings)
    if passed == 0:
        return None
    return reopened / passed


def regression_pass_rate(findings: list[dict[str, Any]]) -> float | None:
    """Share of passed locations whose fix held (``1 − recurrence``).

    ``None`` when no location ever passed (nothing to be stable *about*)."""
    passed, reopened = _passed_and_reopened(findings)
    if passed == 0:
        return None
    return (passed - reopened) / passed


def over_audit_rate(findings: list[dict[str, Any]]) -> float | None:
    """``false_positive`` records / total records. ``None`` on an empty ledger."""
    total = len(findings)
    if total == 0:
        return None
    fp = sum(1 for r in findings if r.get("verdict") == "false_positive")
    return fp / total


def counts_by_mismatch_type(findings: list[dict[str, Any]]) -> dict[str, int]:
    """Count records carrying each ``evidence.mismatch_type`` (theory §3)."""
    c: Counter[str] = Counter()
    for rec in findings:
        mt = (rec.get("evidence") or {}).get("mismatch_type")
        if mt:
            c[str(mt)] += 1
    return dict(c)


def counts_by_severity(findings: list[dict[str, Any]]) -> dict[str, int]:
    """Count records carrying each ``evidence.severity`` (theory §3)."""
    c: Counter[str] = Counter()
    for rec in findings:
        sv = (rec.get("evidence") or {}).get("severity")
        if sv:
            c[str(sv)] += 1
    return dict(c)


def _counts_by_verdict(findings: list[dict[str, Any]]) -> dict[str, int]:
    c: Counter[str] = Counter()
    for rec in findings:
        v = rec.get("verdict")
        if v:
            c[str(v)] += 1
    return dict(c)


def _counts_by_state(findings: list[dict[str, Any]]) -> dict[str, int]:
    c: Counter[str] = Counter()
    for recs in group_by_location(findings).values():
        state, _ever, _re = location_history(recs)
        c[state] += 1
    return dict(c)


def summarize(path: str | Path) -> dict[str, Any]:
    """Compute every §14 metric over the ledger at ``path``.

    Returns a JSON-serialisable dict. Raw counts accompany every rate so a
    ``None`` rate (zero denominator) is never ambiguous.
    """
    findings = read_findings(path)
    passed, reopened = _passed_and_reopened(findings)
    locations = group_by_location(findings)
    return {
        "ledger_path": str(path),
        "total_records": len(findings),
        "distinct_locations": len(locations),
        "passed_locations": passed,
        "reopened_locations": reopened,
        "false_positive_records": sum(
            1 for r in findings if r.get("verdict") == "false_positive"
        ),
        "recurrence_rate": recurrence_rate(findings),
        "regression_pass_rate": regression_pass_rate(findings),
        "over_audit_rate": over_audit_rate(findings),
        "by_verdict": _counts_by_verdict(findings),
        "by_state": _counts_by_state(findings),
        "by_mismatch_type": counts_by_mismatch_type(findings),
        "by_severity": counts_by_severity(findings),
    }


def _fmt_rate(v: float | None) -> str:
    return "n/a (no samples)" if v is None else f"{v:.4f}"


def _render(summary: dict[str, Any]) -> str:
    lines = [
        f"Ledger: {summary['ledger_path']}",
        f"  total records        : {summary['total_records']}",
        f"  distinct locations   : {summary['distinct_locations']}",
        f"  passed locations     : {summary['passed_locations']}",
        f"  reopened locations   : {summary['reopened_locations']}",
        f"  false-positive records: {summary['false_positive_records']}",
        "",
        "Metrics (theory §14):",
        f"  Recurrence Rate      : {_fmt_rate(summary['recurrence_rate'])}",
        f"  Regression Pass Rate : {_fmt_rate(summary['regression_pass_rate'])}",
        f"  Over-Audit Rate      : {_fmt_rate(summary['over_audit_rate'])}",
    ]

    def _block(title: str, counts: dict[str, int]) -> None:
        lines.append("")
        lines.append(f"{title}:")
        if not counts:
            lines.append("  (none)")
            return
        for k in sorted(counts):
            lines.append(f"  {k:24s}: {counts[k]}")

    _block("By verdict", summary["by_verdict"])
    _block("By lifecycle state", summary["by_state"])
    _block("By mismatch_type", summary["by_mismatch_type"])
    _block("By severity", summary["by_severity"])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """``python -m core.ccx.audit.ledger_stats <ledger.jsonl> [--json]``."""
    args = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in args
    args = [a for a in args if a != "--json"]
    if len(args) != 1:
        print(
            "usage: python -m core.ccx.audit.ledger_stats <ledger.jsonl> [--json]",
            file=sys.stderr,
        )
        return 2
    summary = summarize(args[0])
    if as_json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    else:
        print(_render(summary))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
