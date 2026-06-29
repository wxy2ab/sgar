"""Turnkey autonomous SGAR project builder.

Drives a multi-stage SGAR project to completion from a project plan
(blueprint + roadmap + per-stage specs). The design choice that makes this
*turnkey* rather than *fragile*:

* **Governance is deterministic.** The hard state machine —
  ``init → validate/accept → start-stage → verify → close-stage`` — is driven
  by Python here, never by an LLM. That is the entire point of SGAR: the gates
  are code, not model whim. (Contrast the agent-driven driver, where an LLM
  supervisor issues the governance ops and needs careful forcing prompts.)
* **The LLM only implements.** A pluggable ``implement`` callback is the only
  place a model is involved — its job is to satisfy a stage's spec. Swap in a
  ``CodeAgent`` turn for production, or a stub that writes files for tests.
* **Verification is machine-gated (P2).** A stage closes only when its spec's
  ``[check: <cmd>]`` criteria actually pass: the runtime runs ``run_checks``
  at verify/close, so ``autobuild`` simply marks every criterion ``--pass`` and
  lets the runtime refuse a pass the checks contradict. A refusal carries the
  failing-check evidence, which is fed back to the next implement attempt
  (bounded repair). Criteria WITHOUT a ``[check:]`` are trust-the-implementer:
  the spec author opts a criterion into hard gating by adding a check.
* **Resumable.** State lives on disk under ``.sgar/``; re-running picks up from
  the current state (already-closed stages skipped, a started-but-unclosed
  stage resumes its repair loop). "Resumes" is literal: the repair budget and
  the last refusal's failing-``[check:]`` evidence are persisted on the stage
  record, so a process killed mid-stage CONTINUES from the consumed budget with
  the prior evidence re-fed — it does not cold-restart with a refilled
  ``max_verify_attempts`` (the cost-amplifier that defeated "bounded repair").
  A stage that has already exhausted its budget is not silently granted more on
  re-run; raise ``max_verify_attempts`` to deliberately extend it.

This module has no LLM or task.py dependency — it's pure orchestration over
``SgarRuntime`` + a callback, so it is unit-testable with a stub implementer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .models import CriterionResult, SgarError, StageRecord
from .runtime import SgarRuntime
from .validation import parse_exit_criteria


# Cap on the persisted failure detail so a long, repeatedly-refused stage cannot
# bloat state.json. The [check:] evidence tail is already bounded in
# checks.py (~2k chars); this is a belt-and-suspenders ceiling on the whole
# SgarError text.
_FAILURE_DETAIL_MAX_CHARS = 2000


@dataclass(slots=True)
class StagePlan:
    stage_id: str
    spec_text: str


@dataclass(slots=True)
class ProjectPlan:
    blueprint: str
    roadmap: str
    stages: list[StagePlan]


@dataclass(slots=True)
class StageReport:
    stage_id: str
    closed: bool
    attempts: int
    last_error: str | None = None


@dataclass(slots=True)
class AutobuildReport:
    success: bool
    stages: list[StageReport] = field(default_factory=list)
    reason: str = ""


# implement(stage_plan, attempt, failure_detail) -> None
#   attempt: 1-based attempt counter for this stage.
#   failure_detail: None on the first attempt; otherwise the SgarError text
#     from the previous verify/close refusal (includes failing-check evidence).
Implementer = Callable[[StagePlan, int, "str | None"], None]


def _noop_log(_message: str) -> None:
    return None


def autobuild(
    plan: ProjectPlan,
    *,
    cwd: str | Path,
    implement: Implementer,
    session: str | None = None,
    max_verify_attempts: int = 4,
    check_timeout_s: float = 120.0,
    log: Callable[[str], None] = _noop_log,
) -> AutobuildReport:
    """Drive ``plan`` to completion. Returns a structured report.

    Never raises for an ordinary build failure (a stage that exhausts its
    repair attempts) — that is reported as ``success=False`` with the offending
    stage's ``last_error``. SgarError still propagates for *structural*
    problems (e.g. a malformed plan that can't even bootstrap), which are
    programmer errors, not build outcomes.
    """
    if max_verify_attempts < 1:
        raise ValueError("max_verify_attempts must be >= 1")
    runtime = SgarRuntime(
        cwd,
        session_id=session,
        run_criterion_checks=True,
        criterion_check_timeout_s=check_timeout_s,
    )
    _bootstrap(runtime, plan, log)

    reports: list[StageReport] = []
    for stage in plan.stages:
        closed = set(runtime.store.load_state().closed_stage_ids)
        if stage.stage_id in closed:
            log(f"{stage.stage_id}: already closed — skip")
            reports.append(StageReport(stage.stage_id, True, 0))
            continue
        report = _drive_stage(runtime, stage, implement, max_verify_attempts, log)
        reports.append(report)
        if not report.closed:
            return AutobuildReport(
                success=False,
                stages=reports,
                reason=f"{stage.stage_id} not closed: {report.last_error}",
            )
    return AutobuildReport(success=True, stages=reports, reason="all stages closed")


def _bootstrap(runtime: SgarRuntime, plan: ProjectPlan, log: Callable[[str], None]) -> None:
    if not runtime.store.state_path.exists():
        runtime.init()
        log("init")
    state = runtime.store.load_state()
    if not state.accepted_blueprint_hash:
        runtime.set_blueprint(plan.blueprint)
        runtime.validate_blueprint(accept=True).require_ok()
        log("blueprint accepted")
    state = runtime.store.load_state()
    if not state.accepted_roadmap_hash or state.roadmap_review_required:
        runtime.set_roadmap(plan.roadmap)
        runtime.validate_roadmap(accept=True).require_ok()
        log("roadmap accepted")


def _drive_stage(
    runtime: SgarRuntime,
    stage: StagePlan,
    implement: Implementer,
    max_attempts: int,
    log: Callable[[str], None],
) -> StageReport:
    state = runtime.store.load_state()
    # Set up + start only if this isn't already the current (resumed) stage.
    if state.current_stage_id != stage.stage_id:
        runtime.set_stage_spec(stage.stage_id, stage.spec_text)
        runtime.validate_stage_spec(stage.stage_id).require_ok()
        runtime.start_stage(stage.stage_id)
        log(f"{stage.stage_id}: started")
        prior_attempts = 0
        detail: str | None = None
    else:
        # Resume a started-but-unclosed stage. The repair-loop budget and the
        # last refusal's evidence live on the stage record (persisted on every
        # refusal below), so we CONTINUE the loop from where the killed run left
        # off — the budget is consumed, not silently refilled with a fresh
        # max_attempts every restart — and re-feed the failing-[check:] detail
        # the Implementer contract promises (previously both were lost: attempt
        # reset to 1, detail to None).
        record = state.stages.get(stage.stage_id)
        prior_attempts = record.repair_attempts if record else 0
        detail = record.last_failure_detail if record else None
        log(
            f"{stage.stage_id}: resuming current stage "
            f"(repair attempt {prior_attempts + 1}/{max_attempts})"
        )

    criteria = parse_exit_criteria(stage.spec_text)
    for attempt in range(prior_attempts + 1, max_attempts + 1):
        implement(stage, attempt, detail)
        try:
            runtime.record_verification(
                stage.stage_id,
                results=[
                    CriterionResult(c.criterion_id, True, "autobuild")
                    for c in criteria
                ],
            )
            runtime.close_stage(stage.stage_id)
            # Closed: clear the pending failure evidence (a closed stage has no
            # outstanding refusal) but keep the attempt count as history.
            _persist_repair_progress(runtime, stage.stage_id, attempt, None)
            log(f"{stage.stage_id}: closed on attempt {attempt}")
            return StageReport(stage.stage_id, closed=True, attempts=attempt)
        except SgarError as exc:
            detail = str(exc)
            _persist_repair_progress(runtime, stage.stage_id, attempt, detail)
            log(f"{stage.stage_id}: attempt {attempt} refused: {detail}")
    # Budget exhausted (this run, or already-exhausted on entry → empty loop).
    return StageReport(
        stage.stage_id,
        closed=False,
        attempts=max(prior_attempts, max_attempts),
        last_error=detail,
    )


def _persist_repair_progress(
    runtime: SgarRuntime,
    stage_id: str,
    attempts: int,
    detail: str | None,
) -> None:
    """Persist the repair-loop control-state (cumulative attempts consumed +
    last refusal evidence) onto the stage record so a mid-stage process kill
    resumes deterministically. ``detail=None`` clears the pending evidence (the
    stage closed). Turnkey autobuild is the single writer of ``.sgar/`` here, so
    a plain atomic ``write_state`` is sufficient — no CAS needed."""
    state = runtime.store.load_state()
    record = state.stages.get(stage_id) or StageRecord(stage_id=stage_id)
    record.repair_attempts = attempts
    record.last_failure_detail = (
        detail[:_FAILURE_DETAIL_MAX_CHARS] if detail is not None else None
    )
    state.stages[stage_id] = record
    runtime.store.write_state(state)


__all__ = [
    "AutobuildReport",
    "Implementer",
    "ProjectPlan",
    "StagePlan",
    "StageReport",
    "autobuild",
]
