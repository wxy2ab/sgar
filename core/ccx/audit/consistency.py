"""Deterministic consistency-oracle teeth for the narrative-fidelity audit.

Phase 1 of the hard-feedback audit agent
(``docs/audit_agent_design_2026-06-28.md``, §1.5). This is the **consistency
oracle** (fidelity source #1 — logic / internal consistency): it audits whether
a run's narrative report (the *claim* side) is consistent with the machine
ground-truth the run actually produced (the *evidence* side).

Per §1.5 the "teeth" for this oracle are NOT a pytest subprocess (that is the
*execution* oracle, ``corrective.is_actionable``). They are a **pure-Python
re-read**: re-derive the ground-truth and re-evaluate the same boolean relation
the candidate asserts. ``recheck_consistency_claim`` returns ``confirmed=True``
ONLY when that deterministic re-read reproduces the mismatch — never on the LLM's
prose (别信 ``goal_verdict.passed``). The LLM (the AuditRunner) only *proposes*
which claim to check; Python *disposes* by re-deriving the truth.

Two-sided grounding (the scs_v6 lesson, §3 — enforced, not documented):

* **claim side** — the candidate's ``claim_text`` must be a real span of the
  report AND carry a positive assertion. A claim the report never made cannot be
  confirmed (no fabricated over-claim).
* **evidence side** — the kind-specific re-check must re-derive the contradiction
  from the *real* ground-truth. A candidate naming a non-existent check, or a
  check that actually passed, is refused.

Scope (decided MVP): mismatch **kinds 1-3** below, which elevate the two existing
baseline detectors — ``llm_monitor._heuristic_performative_completion`` and
``watch.degraded_completion`` — from "node-counts vs a verdict flag" to
"narrative text vs machine ground-truth". Kind 4 (``evidence_citation_invalid``,
a cited file span lacking the claimed excerpt) is deferred.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "MISMATCH_KINDS",
    "ConsistencyVerdict",
    "assemble_ground_truth",
    "recheck_consistency_claim",
]

# Mirror of ``governed_goal.CCX_GOAL_VERDICT_SNAPSHOT_KEY`` ("goal_verdict").
# Hardcoded to keep this module's import surface light (the constant lives in the
# heavy ``governed_goal`` module); a unit test asserts the two never drift.
_GOAL_VERDICT_KEY = "goal_verdict"

#: The deterministically-checkable narrative-fidelity mismatch kinds (MVP scope).
VERDICT_CONTRADICTION = "verdict_contradiction"
CHECK_OUTCOME_CONTRADICTION = "check_outcome_contradiction"
NODE_STATE_CONTRADICTION = "node_state_contradiction"
MISMATCH_KINDS = frozenset(
    {VERDICT_CONTRADICTION, CHECK_OUTCOME_CONTRADICTION, NODE_STATE_CONTRADICTION}
)

#: Small fixed lexicon of positive-assertion tokens. A ``claim_text`` must
#: contain at least one (substring, case-folded) to count as a success/completion
#: claim — so the auditor can't confirm a "contradiction" against a report span
#: that makes no positive assertion at all.
_POSITIVE_ASSERTION_TOKENS = frozenset({
    "pass", "green", "success", "succeed", "complete", "done", "verified",
    "verify", "works", "working", "fixed", "resolved", "all checks", "no error",
    "no failures", "100%", "clean",
})

_WS_RE = re.compile(r"\s+")


@dataclass(slots=True)
class ConsistencyVerdict:
    """Result of the deterministic consistency re-check for one candidate."""

    confirmed: bool
    mismatch_kind: str
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)


def _norm(text: Any) -> str:
    """Case-fold + collapse whitespace for robust substring grounding."""
    return _WS_RE.sub(" ", str(text or "")).strip().lower()


def assemble_ground_truth(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize the machine ground-truth from a run's ``session_snapshot``.

    The snapshot is the harness-produced machine truth (the same source
    ``_debug_advisories`` reads): ``goal_verdict`` (with ``passed`` and the
    per-criterion ``check_evidence`` list), the run ``status``, and the node
    ``succeeded/failed/abandoned`` counts. ``check_evidence`` is snapshot-only
    (it is not in the ``runtime.db`` tables), so the snapshot — not the DB — is
    the authoritative source for kinds 1-2.

    Returns a flat dict the kind-specific re-checks consume:
    ``{passed, status, counts, check_evidence (by criterion_id), degraded}``.
    Reuses ``watch.degraded_completion`` (the kind-3 baseline) verbatim.
    """
    snap = snapshot or {}
    gv = snap.get(_GOAL_VERDICT_KEY)
    gv = gv if isinstance(gv, dict) else {}

    check_evidence: dict[str, dict[str, Any]] = {}
    for entry in gv.get("check_evidence") or []:
        if isinstance(entry, dict) and entry.get("criterion_id") is not None:
            check_evidence[str(entry["criterion_id"])] = entry

    counts = {
        "succeeded": int(snap.get("succeeded", 0) or 0),
        "failed": int(snap.get("failed", 0) or 0),
        "abandoned": int(snap.get("abandoned", 0) or 0),
    }
    status = snap.get("status")

    from ..watch import degraded_completion  # light; reuse the kind-3 detector

    degraded = degraded_completion(
        status, {"abandoned": counts["abandoned"], "failed": counts["failed"]}
    )
    return {
        "passed": gv.get("passed"),
        "status": status,
        "counts": counts,
        "check_evidence": check_evidence,
        "degraded": degraded,
    }


def _claim_grounded(claim_text: str, report_text: str) -> tuple[bool, str]:
    """The cited claim span must be REAL (a substring of the report) and a
    positive assertion. Refuses a fabricated span or a non-assertive one."""
    ct = _norm(claim_text)
    if not ct:
        return False, "empty claim_text"
    if ct not in _norm(report_text):
        return False, "claim_text is not a span of the report (fabricated citation)"
    if not any(tok in ct for tok in _POSITIVE_ASSERTION_TOKENS):
        return False, "claim_text carries no positive assertion (nothing to contradict)"
    return True, ""


def _recheck_verdict(gt: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    passed = gt.get("passed")
    ev = {"goal_verdict_passed": passed}
    if passed is False:
        return True, "report asserts success but goal_verdict.passed is False", ev
    return False, f"goal_verdict.passed={passed!r} does not contradict a success claim", ev


def _recheck_check_outcome(
    locator: dict[str, Any], gt: dict[str, Any]
) -> tuple[bool, str, dict[str, Any]]:
    cid = str((locator or {}).get("criterion_id") or "").strip()
    if not cid:
        return False, "no criterion_id in locator (ungrounded)", {}
    entry = (gt.get("check_evidence") or {}).get(cid)
    if entry is None:
        return (
            False,
            f"claimed check {cid!r} is not in check_evidence (ungrounded / hallucinated)",
            {"criterion_id": cid},
        )
    ev = {
        "criterion_id": cid,
        "passed": entry.get("passed"),
        "executable": entry.get("executable"),
        "command": entry.get("command"),
        "line": entry.get("line"),
    }
    if entry.get("passed") is False and entry.get("executable") is True:
        return True, f"report claims check {cid} passes, but it is a genuine RED", ev
    return (
        False,
        f"check {cid}: passed={entry.get('passed')!r} executable={entry.get('executable')!r}"
        " — no real contradiction (passed, or a harness defect)",
        ev,
    )


def _recheck_node_state(gt: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    degraded = gt.get("degraded")
    if degraded:
        return (
            True,
            f"report asserts completion but the run is degraded ({degraded})",
            {"degraded": degraded, "status": gt.get("status")},
        )
    return (
        False,
        "no degraded completion (no abandoned/failed nodes on a completed run)",
        {"status": gt.get("status"), "counts": gt.get("counts")},
    )


_RECHECKERS = {
    VERDICT_CONTRADICTION: lambda loc, gt: _recheck_verdict(gt),
    CHECK_OUTCOME_CONTRADICTION: _recheck_check_outcome,
    NODE_STATE_CONTRADICTION: lambda loc, gt: _recheck_node_state(gt),
}


def recheck_consistency_claim(
    candidate: dict[str, Any],
    *,
    report_text: str,
    ground_truth: dict[str, Any],
) -> ConsistencyVerdict:
    """Deterministically re-check one LLM-proposed mismatch candidate.

    ``candidate = {mismatch_kind, locator, claim_text}``. Returns
    ``confirmed=True`` ONLY when (a) the claim is grounded (a real, positive-
    assertion span of ``report_text``) AND (b) the kind-specific re-derivation of
    ``ground_truth`` reproduces the contradiction. Anything else is
    ``confirmed=False`` — the caller downgrades it to advisory (``uncertain``).
    The LLM's prose is never trusted; the truth is re-derived here.
    """
    kind = str(candidate.get("mismatch_kind") or "")
    claim_text = str(candidate.get("claim_text") or "")
    locator = candidate.get("locator") or {}

    if kind not in MISMATCH_KINDS:
        return ConsistencyVerdict(False, kind, f"unknown mismatch_kind {kind!r}")

    grounded, why = _claim_grounded(claim_text, report_text)
    if not grounded:
        return ConsistencyVerdict(
            False, kind, f"claim not grounded: {why}", {"claim_text": claim_text}
        )

    ok, reason, ev = _RECHECKERS[kind](locator, ground_truth)
    ev = {**ev, "claim_text": claim_text}
    return ConsistencyVerdict(ok, kind, reason, ev)
