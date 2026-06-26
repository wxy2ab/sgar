"""Emit the run-level governance verdict into the runtime.db event stream.

ccx computes an authoritative run-level governance verdict
(``contract_verdict`` / ``run_audit_verdict`` / ``goal_verdict`` /
``abandoned_warning``) and stamps it onto ``result.session_snapshot``. But the
three operator renderers — ``watch`` / ``report`` / ``llm_monitor`` — read ONLY
the ``runtime.db`` event stream, never the in-memory ``AgentRunResult``. So the
verdict was invisible to them: a goal-mode run whose 17 nodes all SUCCEEDED but
whose ``goal_verdict.passed`` is ``False`` rendered as a clean "all green",
forcing the operator to "inspect the artifacts, not the flags".

This module closes that gap by publishing ONE summary event,
``ccx.governance.verdict``, carrying the derived overall ``passed`` plus each
sub-verdict's ``passed``/``stop_reason``, so a renderer can show the run-level
truth alongside the node-level state.

Default OFF — strictly gated by ``CCX_EMIT_GOVERNANCE_EVENTS``. With the flag
unset (or falsey) :func:`emit_governance_verdict` returns immediately and
publishes nothing, so the ``runtime.db`` event stream and every renderer stay
byte-for-byte unchanged. This is the byte-equivalence anchor. The whole emit is
best-effort (wrapped in ``try/except`` → ``logger.debug``) so an emission
failure never disturbs the run it is reporting on.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


#: The single event kind this module publishes. Namespaced under ``ccx.`` like
#: the other ccx-layer kinds (``ccx.steer.injected`` etc.) so a renderer can key
#: off the prefix.
GOVERNANCE_VERDICT_EVENT_KIND = "ccx.governance.verdict"

_EMIT_FLAG_ENV = "CCX_EMIT_GOVERNANCE_EVENTS"
_TRUTHY = frozenset({"1", "true", "on", "yes"})


def emission_enabled() -> bool:
    """Whether run-level governance verdict emission is turned on (default OFF).

    Set ``CCX_EMIT_GOVERNANCE_EVENTS`` to ``1``/``true``/``on``/``yes`` to opt
    in. Anything else (including unset) keeps emission off — the
    byte-equivalence default. Callers that build an event bus / reopen a DB to
    emit should gate that work on this so the default path does zero extra I/O.
    """
    raw = os.environ.get(_EMIT_FLAG_ENV)
    if raw is None:
        return False
    return raw.strip().lower() in _TRUTHY


def _verdict_summary(verdict: Any) -> dict[str, Any] | None:
    """Compact, renderer-friendly view of a sub-verdict (or ``None``)."""
    if not isinstance(verdict, dict):
        return None
    return {
        "passed": verdict.get("passed"),
        "status": verdict.get("status"),
        "stop_reason": verdict.get("stop_reason"),
        "unrunnable_criterion_ids": verdict.get("unrunnable_criterion_ids"),
    }


def _derive_overall_passed(verdicts: list[dict[str, Any]]) -> bool | None:
    """Overall ``passed`` from the present sub-verdicts.

    ``None`` when no governance verdict is present (a plain run that was never
    governed — nothing to assert about). Otherwise ``True`` only if EVERY present
    verdict passed; a single not-passed verdict makes the run not-passed. This is
    what lets a renderer flag "nodes all green but the run-level verdict says
    NO" — exactly the performative-completion signal.
    """
    if not verdicts:
        return None
    return all(bool(v.get("passed")) for v in verdicts)


def emit_governance_verdict(
    event_bus: Any, run_id: str, snapshot: dict[str, Any] | None,
) -> None:
    """Publish one ``ccx.governance.verdict`` event for a finished run.

    ``snapshot`` is the FULLY-stamped ``result.session_snapshot`` (so all three
    sub-verdicts are final — that is why the caller emits at the outermost
    ``run()`` boundary, not inside ``_build_result`` where ``run_audit_verdict``
    and ``goal_verdict`` are still ``None``).

    Default OFF: returns immediately when :func:`emission_enabled` is false —
    publishes nothing, the byte-equivalence anchor. Best-effort: any failure is
    swallowed to ``logger.debug`` and never propagates to the run.
    """
    if not emission_enabled():
        return
    try:
        snap = snapshot or {}
        contract = snap.get("contract_verdict")
        run_audit = snap.get("run_audit_verdict")
        goal = snap.get("goal_verdict")
        present = [
            v for v in (contract, run_audit, goal) if isinstance(v, dict)
        ]
        payload = {
            "run_id": run_id,
            "status": snap.get("status"),
            "passed": _derive_overall_passed(present),
            "succeeded": snap.get("succeeded"),
            "failed": snap.get("failed"),
            "abandoned": snap.get("abandoned"),
            "abandoned_warning": bool(snap.get("abandoned_warning")),
            "contract_verdict": _verdict_summary(contract),
            "run_audit_verdict": _verdict_summary(run_audit),
            "goal_verdict": _verdict_summary(goal),
        }
        event_bus.publish(run_id, GOVERNANCE_VERDICT_EVENT_KIND, payload)
    except Exception:
        # Observability must never break the run it reports on.
        logger.debug(
            "ccx: failed to emit %s event (run_id=%s)",
            GOVERNANCE_VERDICT_EVENT_KIND, run_id, exc_info=True,
        )


__all__ = [
    "GOVERNANCE_VERDICT_EVENT_KIND",
    "emission_enabled",
    "emit_governance_verdict",
]
