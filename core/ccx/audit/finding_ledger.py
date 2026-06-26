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

#: Controlled vocabulary for the six-mismatch classification (theory §3 / §9).
#: Optional metadata stamped into ``evidence`` — an illegal value is a
#: ``ValueError`` (mirrors the verdict guard); the record stays 9 keys.
_VALID_MISMATCH_TYPES = frozenset(
    {
        "aggregation",
        "support",
        "state",
        "specification",
        "fitting_boundary",
        "observation_representation",
    }
)

#: Controlled vocabulary for finding severity (theory §3). Metadata only for now
#: — no gate consumes it, and the teeth proof stays mandatory for every real
#: promote; severity does not weaken it.
_VALID_SEVERITIES = frozenset({"blocker", "major", "minor", "note"})

#: Defect lifecycle states (theory §7.3):
#: ``unknown → open → patched → regression_passed → accepted_risk → revoked``.
#: ``unknown`` is the pre-discovery sentinel for a location with no recognised
#: records. The rank orders only the monotone-forward prefix; ``accepted_risk``
#: and ``revoked`` are explicit terminal overrides, not ranked.
_LIFECYCLE_RANK = {"unknown": 0, "open": 1, "patched": 2, "regression_passed": 3}

__all__ = [
    "append_finding",
    "read_findings",
    "count_prior",
    "code_location_key",
    "defect_lifecycle",
    "mark_accepted_risk",
    "mark_revoked",
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
    mismatch_type: str | None = None,
    severity: str | None = None,
) -> dict[str, Any]:
    """Append one finding as a JSONL line. Returns the written record.

    ``verdict`` must be one of confirmed | false_positive | uncertain |
    by_design (raises ``ValueError`` otherwise — matching the probe ledger).

    ``mismatch_type`` (theory §3 six-mismatch vocabulary) and ``severity``
    (blocker | major | minor | note) are optional controlled fields. When
    supplied they are validated and stamped *inside* ``evidence`` so the record
    keeps its 9-key shape and stays readable by the probe schema; an illegal
    value raises ``ValueError``.
    """
    if verdict not in _VALID_VERDICTS:
        raise ValueError(
            f"verdict must be one of {sorted(_VALID_VERDICTS)}, got {verdict!r}"
        )
    if mismatch_type is not None and mismatch_type not in _VALID_MISMATCH_TYPES:
        raise ValueError(
            f"mismatch_type must be one of {sorted(_VALID_MISMATCH_TYPES)}, "
            f"got {mismatch_type!r}"
        )
    if severity is not None and severity not in _VALID_SEVERITIES:
        raise ValueError(
            f"severity must be one of {sorted(_VALID_SEVERITIES)}, got {severity!r}"
        )
    # Copy so a caller-owned dict is never mutated; classification fields live
    # inside evidence (the record's 9 top-level keys are unchanged).
    ev = dict(evidence or {})
    if mismatch_type is not None:
        ev["mismatch_type"] = mismatch_type
    if severity is not None:
        ev["severity"] = severity
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
        "evidence": ev,
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


def _loc_matches(
    rec: dict[str, Any], key: dict[str, str], mismatch_type: str | None
) -> bool:
    """Whether ``rec``'s ``evidence.code_location`` matches ``key`` (and, if
    ``mismatch_type`` is given, its ``evidence.mismatch_type`` too)."""
    ev = rec.get("evidence") or {}
    loc = ev.get("code_location")
    if not (
        isinstance(loc, dict)
        and loc.get("file") == key["file"]
        and loc.get("func") == key["func"]
    ):
        return False
    if mismatch_type is not None and ev.get("mismatch_type") != mismatch_type:
        return False
    return True


def count_prior(
    path: str | Path,
    code_location: tuple[str, str] | dict[str, str],
    *,
    verdicts: Iterable[str] = _RECIDIVISM_VERDICTS,
    mismatch_type: str | None = None,
) -> int:
    """Count prior real findings (recidivism) at this ``(file, func)`` location.

    The cross-run "did this location break + get fixed before?" signal: groups
    the ledger by ``evidence.code_location`` and counts records whose verdict is
    a real incident (``confirmed``/``by_design`` by default).

    Pass ``mismatch_type`` to narrow the recidivism group to
    ``(file, func, mismatch_type)`` — same location, same *kind* of defect.
    ``None`` (default) preserves the original ``(file, func)`` grouping exactly.
    """
    key = code_location_key(code_location)
    want = frozenset(verdicts)
    n = 0
    for rec in read_findings(path):
        if rec.get("verdict") not in want:
            continue
        if _loc_matches(rec, key, mismatch_type):
            n += 1
    return n


def _lifecycle_signal(rec: dict[str, Any]) -> str | None:
    """Map one ledger record to a defect-lifecycle signal, or ``None`` if it
    does not move the state (probe-style false positives / unphased noise).

    The join key is (``verdict``, ``evidence.phase``) — the phase vocabulary the
    promotion pipeline and the :func:`mark_accepted_risk` / :func:`mark_revoked`
    helpers already stamp; no new top-level record key is introduced.
    """
    verdict = rec.get("verdict")
    phase = (rec.get("evidence") or {}).get("phase")
    if verdict == "by_design":
        if phase == "revoked":
            return "revoked"
        # ``accepted_risk`` phase, or a legacy by_design with no/other phase.
        return "accepted_risk"
    if verdict == "confirmed":
        if phase == "promoted":
            return "regression_passed"
        # probe-style confirmed: a real defect, not yet a resident guard.
        return "open"
    if verdict == "uncertain":
        if phase in ("post_fix", "backstop"):
            return "patched"  # repair attempted, not validated
        return None  # unphased / harness-defect uncertain — not a defect signal
    if verdict == "false_positive":
        if phase == "teeth":
            return "patched"  # fix + guard attempted but teeth proof failed
        return None  # probe-style false positive — not a real defect
    return None


def _advance_lifecycle(state: str, sig: str | None) -> str:
    """Fold one signal into the running lifecycle state.

    Forward progress is monotone over ``unknown < open < patched <
    regression_passed``; ``accepted_risk`` / ``revoked`` are explicit terminal
    overrides. An ``open`` signal arriving after a passed/terminal state is a
    **reopen** — the resident guard did not hold — and drops the state back to
    ``open``.
    """
    if sig is None:
        return state
    if sig in ("accepted_risk", "revoked"):
        return sig
    if sig == "regression_passed":
        return "regression_passed"
    if sig == "open":
        # discovery, or re-discovery after a passed/terminal state (reopen).
        return "open"
    if sig == "patched":
        # a repair attempt only advances forward; it never downgrades a
        # passed/accepted/revoked state (those move only via an open reopen).
        if _LIFECYCLE_RANK.get(state, -1) < _LIFECYCLE_RANK["patched"]:
            return "patched"
        return state
    return state


def defect_lifecycle(
    path: str | Path,
    code_location: tuple[str, str] | dict[str, str],
) -> str:
    """Reduce the time-ordered records for one ``(file, func)`` to its current
    lifecycle state (theory §7.3):
    ``unknown → open → patched → regression_passed → accepted_risk → revoked``.

    Append-only and read-only: this never writes. Records are folded in
    timestamp order (stable on ledger/append order for ties). A location with no
    recognised records is ``unknown`` — missing/legacy records (no ``phase``)
    are inferred from their verdict and never raise.
    """
    key = code_location_key(code_location)
    recs = [rec for rec in read_findings(path) if _loc_matches(rec, key, None)]
    recs.sort(key=lambda r: str(r.get("ts") or ""))
    state = "unknown"
    for rec in recs:
        state = _advance_lifecycle(state, _lifecycle_signal(rec))
    return state


def _mark_lifecycle(
    *,
    phase: str,
    verdict: str,
    path: str | Path,
    code_location: tuple[str, str] | dict[str, str],
    reason: str,
    track: str,
    finding_id: str | None,
    evidence: dict[str, Any] | None,
) -> dict[str, Any]:
    """Shared body for the explicit operator lifecycle transitions."""
    key = code_location_key(code_location)
    ev = dict(evidence or {})
    ev["code_location"] = key
    ev["phase"] = phase
    ev["reason"] = reason
    if finding_id is not None:
        ev["finding_id"] = finding_id
    return append_finding(
        track=track,
        hypothesis=(
            f"Operator lifecycle transition → {phase} at "
            f"{key['file']}::{key['func']}."
        ),
        expected=f"defect lifecycle state = {phase}",
        observed=reason,
        verdict=verdict,
        repro="(operator transition; no automated repro)",
        path=path,
        evidence=ev,
    )


def mark_accepted_risk(
    *,
    path: str | Path,
    code_location: tuple[str, str] | dict[str, str],
    reason: str,
    track: str = "lifecycle",
    finding_id: str | None = None,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record an explicit operator transition to ``accepted_risk`` (theory §7.3).

    A deliberate, audited act: the residual defect is knowingly accepted. Appends
    one ``by_design`` record with ``evidence.phase="accepted_risk"`` so
    :func:`defect_lifecycle` reduces the location to ``accepted_risk``. Returns
    the written record.
    """
    return _mark_lifecycle(
        phase="accepted_risk",
        verdict="by_design",
        path=path,
        code_location=code_location,
        reason=reason,
        track=track,
        finding_id=finding_id,
        evidence=evidence,
    )


def mark_revoked(
    *,
    path: str | Path,
    code_location: tuple[str, str] | dict[str, str],
    reason: str,
    track: str = "lifecycle",
    finding_id: str | None = None,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record an explicit operator transition to ``revoked`` (theory §7.3).

    The defect (and its guard) is deliberately retired — no longer applicable.
    Appends one ``by_design`` record with ``evidence.phase="revoked"`` so
    :func:`defect_lifecycle` reduces the location to ``revoked``. Returns the
    written record.
    """
    return _mark_lifecycle(
        phase="revoked",
        verdict="by_design",
        path=path,
        code_location=code_location,
        reason=reason,
        track=track,
        finding_id=finding_id,
        evidence=evidence,
    )
