"""Core-side finding ledger for promoted regression guards.

Mirrors the supervised-probe finding schema (``scripts/ccx_supervised_probe``'s
``append_finding``) byte-for-byte — same 9 keys, same verdict vocabulary — so a
promotion record is readable by the same operator tooling that reads the probe
campaigns' ``*_findings.jsonl``. We reimplement it here rather than import the
script because ``core/`` must never depend on ``scripts/`` (layering); the schema
is small and stable.

The one addition the ratchet needs is a **cross-run code-location axis**:
``runtime.db`` keys events by ``run_id``/``node_id``/``sequence`` but has no
"this (file, func) broke before" axis, which is exactly what recidivism
detection asks. We carry that inside the existing free-form ``evidence`` dict
(``evidence["code_location"] = {"file", "func"}``) so the record shape is
unchanged, and :func:`count_prior` groups by it.
"""

from __future__ import annotations

import datetime as _dt
import json
import uuid
from pathlib import Path
from typing import Any, Iterable

#: Verdict vocabulary — IDENTICAL to the supervised probe ledger. A blind-spot
#: refusal records ``false_positive`` (the test claimed teeth it does not have);
#: do not invent new verdict strings.
_VALID_VERDICTS = frozenset({"confirmed", "false_positive", "uncertain", "by_design"})

#: Verdicts that count as "this location has a real prior incident" for
#: recidivism. ``uncertain`` (anchor-miss / harness defect) and
#: ``false_positive`` (refused promotion) do NOT count.
_RECIDIVISM_VERDICTS = frozenset({"confirmed", "by_design"})

__all__ = [
    "append_finding",
    "read_findings",
    "count_prior",
    "code_location_key",
]


def _now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def code_location_key(code_location: tuple[str, str] | dict[str, str]) -> dict[str, str]:
    """Normalise a ``(file, func)`` (or ``{"file","func"}``) into a stable dict."""
    if isinstance(code_location, dict):
        file = str(code_location.get("file") or "")
        func = str(code_location.get("func") or "")
    else:
        file, func = code_location
        file, func = str(file), str(func)
    if not file or not func:
        raise ValueError(
            f"code_location must be a non-empty (file, func); got {code_location!r}"
        )
    return {"file": file, "func": func}


def append_finding(
    *,
    track: str,
    hypothesis: str,
    expected: Any,
    observed: Any,
    verdict: str,
    repro: str,
    path: str | Path,
    evidence: dict[str, Any] | None = None,
    id: str | None = None,
) -> dict[str, Any]:
    """Append one finding as a JSONL line. Returns the written record.

    ``verdict`` must be one of confirmed | false_positive | uncertain |
    by_design (raises ``ValueError`` otherwise — matching the probe ledger).
    """
    if verdict not in _VALID_VERDICTS:
        raise ValueError(
            f"verdict must be one of {sorted(_VALID_VERDICTS)}, got {verdict!r}"
        )
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "id": id or f"{track}-{uuid.uuid4().hex[:8]}",
        "ts": _now_iso(),
        "track": track,
        "hypothesis": hypothesis,
        "expected": expected,
        "observed": observed,
        "verdict": verdict,
        "repro": repro,
        "evidence": evidence or {},
    }
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    return rec


def read_findings(path: str | Path) -> list[dict[str, Any]]:
    """Parse a findings JSONL ledger; tolerant of blank / malformed lines."""
    p = Path(path)
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def count_prior(
    path: str | Path,
    code_location: tuple[str, str] | dict[str, str],
    *,
    verdicts: Iterable[str] = _RECIDIVISM_VERDICTS,
) -> int:
    """Count prior real findings (recidivism) at this ``(file, func)`` location.

    The cross-run "did this location break + get fixed before?" signal: groups
    the ledger by ``evidence.code_location`` and counts records whose verdict is
    a real incident (``confirmed``/``by_design`` by default).
    """
    key = code_location_key(code_location)
    want = frozenset(verdicts)
    n = 0
    for rec in read_findings(path):
        if rec.get("verdict") not in want:
            continue
        loc = (rec.get("evidence") or {}).get("code_location")
        if isinstance(loc, dict) and loc.get("file") == key["file"] and \
                loc.get("func") == key["func"]:
            n += 1
    return n
