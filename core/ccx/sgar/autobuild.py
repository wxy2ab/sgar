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
* **Optional candidate frontier (sgarx).** With ``use_candidate_frontier=True``,
  the repair loop proposes/patches/audits/promotes nodes under ``.sgarx/``
  instead of an anonymous workspace trajectory. Default remains False so stable
  sgar callers are byte-compatible.

This module has no LLM or task.py dependency — it's pure orchestration over
``SgarRuntime`` / ``SgarxRuntime`` + a callback, so it is unit-testable with a
stub implementer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..agents.patch_first import patch_first_hint
from ..agents.syndrome import workspace_fingerprint
from .checks import stop_on_unrunnable_enabled
from .models import CriterionResult, SgarError, StageRecord
from .runtime import SgarRuntime
from .validation import parse_exit_criteria


_FAILURE_DETAIL_MAX_CHARS = 2000

_UNRUNNABLE_MARKERS = ("(TIMEOUT)", "(ERROR:", "(exit=127)")

_FRONTIER_BUDGET_MARKERS = (
    "candidate budget exhausted",
    "audit budget exhausted",
    "patch budget exhausted",
)


def _looks_unrunnable(detail: str | None) -> bool:
    """Best-effort: does this refusal's text look like an unrunnable ``[check:]``."""
    if not detail:
        return False
    if any(marker in detail for marker in _UNRUNNABLE_MARKERS):
        return True
    return "(exit=2)" in detail and "syntax error" in detail.lower()


def _looks_frontier_budget_exhausted(detail: str | None) -> bool:
    if not detail:
        return False
    return any(marker in detail for marker in _FRONTIER_BUDGET_MARKERS)


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
class ImplementResult:
    """Optional richer return from an implementer under frontier mode."""

    artifact_paths: list[str]
    summary: str = ""


@dataclass(slots=True)
class StageReport:
    stage_id: str
    closed: bool
    attempts: int
    last_error: str | None = None
    harness_defect_suspected: bool = False
    frontier_budget_exhausted: bool = False
    last_candidate_id: str | None = None


@dataclass(slots=True)
class AutobuildReport:
    success: bool
    stages: list[StageReport] = field(default_factory=list)
    reason: str = ""


Implementer = Callable[[StagePlan, int, "str | None"], Any]


def _noop_log(_message: str) -> None:
    return None


def _parse_implement_result(result: Any) -> tuple[list[str], str]:
    if result is None:
        return [], ""
    if isinstance(result, ImplementResult):
        return [str(p) for p in result.artifact_paths], str(result.summary or "")
    if isinstance(result, (list, tuple)):
        return [str(p) for p in result], ""
    return [], ""


def autobuild(
    plan: ProjectPlan,
    *,
    cwd: str | Path,
    implement: Implementer,
    session: str | None = None,
    max_verify_attempts: int = 4,
    check_timeout_s: float = 120.0,
    log: Callable[[str], None] = _noop_log,
    use_candidate_frontier: bool = False,
) -> AutobuildReport:
    """Drive ``plan`` to completion. Returns a structured report.

    ``use_candidate_frontier`` (default False) opts into sgarx frontier
    propose → audit-from-checks → patch → promote before verify/close.
    """
    if max_verify_attempts < 1:
        raise ValueError("max_verify_attempts must be >= 1")
    if use_candidate_frontier:
        from ..sgarx.runtime import SgarxRuntime

        runtime: SgarRuntime = SgarxRuntime(
            cwd,
            session_id=session,
            run_criterion_checks=True,
            criterion_check_timeout_s=check_timeout_s,
        )
    else:
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
        if use_candidate_frontier:
            report = _drive_stage_with_frontier(
                runtime, stage, implement, max_verify_attempts, log
            )
        else:
            report = _drive_stage(
                runtime, stage, implement, max_verify_attempts, log
            )
        reports.append(report)
        if not report.closed:
            if report.frontier_budget_exhausted:
                reason = (
                    f"{stage.stage_id} not closed (frontier_budget_exhausted: "
                    f"{report.last_error})"
                )
            elif report.harness_defect_suspected:
                reason = (
                    f"{stage.stage_id} not closed (harness_defect suspected — the "
                    f"exit-criterion check could not execute, not a candidate "
                    f"failure): {report.last_error}"
                )
            else:
                reason = f"{stage.stage_id} not closed: {report.last_error}"
            return AutobuildReport(
                success=False,
                stages=reports,
                reason=reason,
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


def _bind_detail_to_candidate(
    detail: str | None,
    *,
    cwd: Path | str,
    attempt: int,
    prev_candidate_hash: str | None,
) -> tuple[str | None, str]:
    candidate_hash, git_head = workspace_fingerprint(cwd)
    if not detail:
        return detail, candidate_hash
    lines = [
        f"VERSION-BOUND SYNDROME (residual of the CURRENT candidate) — "
        f"attempt {attempt}",
        f"candidate_hash={candidate_hash}",
    ]
    if git_head:
        lines.append(f"git_head={git_head}")
    if prev_candidate_hash and prev_candidate_hash != candidate_hash:
        lines.append(
            f"NOTE: the workspace changed since the last refusal "
            f"(was {prev_candidate_hash}). The evidence below was produced "
            "against the PREVIOUS candidate — re-read the files it names "
            "before acting on it."
        )
    lines.append(
        "The failing criteria below apply to this candidate version only; "
        "do not reuse an older syndrome."
    )
    lines.append("")
    lines.append(detail)
    if attempt >= 2:
        lines.append("")
        lines.append(patch_first_hint())
    return "\n".join(lines), candidate_hash


def _drive_stage(
    runtime: SgarRuntime,
    stage: StagePlan,
    implement: Implementer,
    max_attempts: int,
    log: Callable[[str], None],
) -> StageReport:
    state = runtime.store.load_state()
    if state.current_stage_id != stage.stage_id:
        runtime.set_stage_spec(stage.stage_id, stage.spec_text)
        runtime.validate_stage_spec(stage.stage_id).require_ok()
        runtime.start_stage(stage.stage_id)
        log(f"{stage.stage_id}: started")
        prior_attempts = 0
        detail: str | None = None
    else:
        record = state.stages.get(stage.stage_id)
        prior_attempts = record.repair_attempts if record else 0
        detail = record.last_failure_detail if record else None
        log(
            f"{stage.stage_id}: resuming current stage "
            f"(repair attempt {prior_attempts + 1}/{max_attempts})"
        )

    criteria = parse_exit_criteria(stage.spec_text)
    prev_candidate_hash: str | None = None
    for attempt in range(prior_attempts + 1, max_attempts + 1):
        fed_detail, prev_candidate_hash = _bind_detail_to_candidate(
            detail,
            cwd=runtime.store.cwd,
            attempt=attempt,
            prev_candidate_hash=prev_candidate_hash,
        )
        implement(stage, attempt, fed_detail)
        try:
            runtime.record_verification(
                stage.stage_id,
                results=[
                    CriterionResult(c.criterion_id, True, "autobuild")
                    for c in criteria
                ],
            )
            runtime.close_stage(stage.stage_id)
            _persist_repair_progress(runtime, stage.stage_id, attempt, None)
            log(f"{stage.stage_id}: closed on attempt {attempt}")
            return StageReport(stage.stage_id, closed=True, attempts=attempt)
        except SgarError as exc:
            detail = str(exc)
            _persist_repair_progress(runtime, stage.stage_id, attempt, detail)
            log(f"{stage.stage_id}: attempt {attempt} refused: {detail}")
            if stop_on_unrunnable_enabled() and _looks_unrunnable(detail):
                log(
                    f"{stage.stage_id}: attempt {attempt} refusal looks like a "
                    "harness defect (the exit-criterion check could not execute), "
                    "not a candidate failure — the repair budget cannot fix this"
                )
    suspected = stop_on_unrunnable_enabled() and _looks_unrunnable(detail)
    return StageReport(
        stage.stage_id,
        closed=False,
        attempts=max(prior_attempts, max_attempts),
        last_error=detail,
        harness_defect_suspected=suspected,
    )


def _next_autobuild_candidate_id(runtime: SgarRuntime, stage_id: str) -> str:
    from ..sgarx.runtime import SgarxRuntime

    assert isinstance(runtime, SgarxRuntime)
    existing = {
        c.candidate_id for c in runtime.list_candidates(stage_id)
    }
    # Also treat on-disk meta as occupied even if list races.
    for n in range(1, 10_000):
        cid = f"ab-{n:03d}"
        if cid in existing:
            continue
        meta = runtime.store.candidate_dir(stage_id, cid) / "meta.json"
        if meta.is_file():
            continue
        return cid
    raise SgarError(f"exhausted autobuild candidate ids for {stage_id!r}")


def _pick_frontier_cursor(
    runtime: SgarRuntime, stage_id: str
) -> tuple[str | None, str | None]:
    from ..sgarx.candidates.models import CandidateStatus
    from ..sgarx.runtime import SgarxRuntime

    assert isinstance(runtime, SgarxRuntime)
    frontier = runtime.list_frontier(stage_id)
    if frontier.promoted_candidate_id:
        return frontier.promoted_candidate_id, None
    candidates = [
        c
        for c in runtime.list_candidates(stage_id)
        if c.status
        not in (
            CandidateStatus.DISCARDED.value,
            CandidateStatus.SUPERSEDED.value,
        )
    ]
    if not candidates:
        return None, None
    candidates.sort(
        key=lambda c: (c.updated_at or "", c.created_at or "", c.candidate_id)
    )
    cur = candidates[-1]
    detail: str | None = None
    if cur.status == CandidateStatus.AUDITED_FAIL.value and cur.audit is not None:
        bound = cur.audit.extras.get("bound_detail")
        if isinstance(bound, str) and bound.strip():
            detail = bound
        elif cur.audit.findings:
            detail = "\n".join(cur.audit.findings)
    return cur.candidate_id, detail


def _feed_frontier_detail(detail: str | None, *, attempt: int) -> str | None:
    if not detail:
        return None
    return f"autobuild attempt {attempt}\n{detail}"


def _looks_candidate_hash_drift(detail: str) -> bool:
    text = (detail or "").lower()
    return "hash drift" in text


def _try_close_frontier_cursor(
    runtime: SgarRuntime,
    stage: StagePlan,
    criteria: list[Any],
    current_id: str,
    *,
    prior_attempts: int,
    log: Callable[[str], None],
) -> StageReport | tuple[str, str]:
    """Try promote (if needed) + verify + close without calling implement.

    Returns a successful :class:`StageReport`, or ``(error_detail, discarded_id)``
    after force-discarding the stale cursor so the caller can re-propose.
    """
    from ..sgarx.candidates.models import CandidateStatus
    from ..sgarx.runtime import SgarxRuntime

    assert isinstance(runtime, SgarxRuntime)
    cur = runtime.get_candidate(stage.stage_id, current_id)
    try:
        if cur.status == CandidateStatus.AUDITED_PASS.value:
            runtime.promote_candidate(stage.stage_id, current_id)
            log(f"{stage.stage_id}: promoted {current_id} (resume close)")
        elif cur.status != CandidateStatus.PROMOTED.value:
            raise SgarError(
                f"resume close expected audited_pass/promoted, got {cur.status!r}"
            )
        runtime.record_verification(
            stage.stage_id,
            results=[
                CriterionResult(c.criterion_id, True, "autobuild")
                for c in criteria
            ],
        )
        runtime.close_stage(stage.stage_id)
        _persist_repair_progress(
            runtime, stage.stage_id, max(prior_attempts, 1), None
        )
        log(f"{stage.stage_id}: closed from existing {current_id}")
        return StageReport(
            stage.stage_id,
            closed=True,
            attempts=max(prior_attempts, 1),
            last_candidate_id=current_id,
        )
    except SgarError as exc:
        detail = str(exc)
        try:
            runtime.force_discard_candidate(stage.stage_id, current_id)
            log(
                f"{stage.stage_id}: discarded stale {current_id} after "
                f"resume-close failure: {detail}"
            )
        except SgarError as discard_exc:
            detail = f"{detail}; discard failed: {discard_exc}"
        _persist_repair_progress(
            runtime, stage.stage_id, max(prior_attempts, 1), detail
        )
        return detail, current_id


def _drive_stage_with_frontier(
    runtime: SgarRuntime,
    stage: StagePlan,
    implement: Implementer,
    max_attempts: int,
    log: Callable[[str], None],
) -> StageReport:
    from ..sgarx.candidates.models import CandidateStatus
    from ..sgarx.runtime import SgarxRuntime

    assert isinstance(runtime, SgarxRuntime)
    state = runtime.store.load_state()
    if state.current_stage_id != stage.stage_id:
        runtime.set_stage_spec(stage.stage_id, stage.spec_text)
        runtime.validate_stage_spec(stage.stage_id).require_ok()
        runtime.start_stage(stage.stage_id)
        log(f"{stage.stage_id}: started (frontier)")
        prior_attempts = 0
        detail: str | None = None
        current_id: str | None = None
    else:
        record = state.stages.get(stage.stage_id)
        prior_attempts = record.repair_attempts if record else 0
        persisted = record.last_failure_detail if record else None
        current_id, audit_detail = _pick_frontier_cursor(runtime, stage.stage_id)
        detail = audit_detail or persisted
        log(
            f"{stage.stage_id}: resuming current stage with frontier "
            f"(repair attempt {prior_attempts + 1}/{max_attempts}; "
            f"current={current_id or '-'})"
        )

    criteria = parse_exit_criteria(stage.spec_text)
    last_candidate_id = current_id
    frontier_budget_hit = False

    if current_id is not None:
        try:
            cur = runtime.get_candidate(stage.stage_id, current_id)
        except SgarError:
            cur = None
        if cur is not None and cur.status in (
            CandidateStatus.AUDITED_PASS.value,
            CandidateStatus.PROMOTED.value,
        ):
            outcome = _try_close_frontier_cursor(
                runtime,
                stage,
                criteria,
                current_id,
                prior_attempts=prior_attempts,
                log=log,
            )
            if isinstance(outcome, StageReport):
                return outcome
            detail, _discarded = outcome
            frontier_budget_hit = _looks_frontier_budget_exhausted(detail)
            current_id = None
            last_candidate_id = None

    for attempt in range(prior_attempts + 1, max_attempts + 1):
        # Never implement on top of an already-audited/promoted cursor.
        if current_id is not None:
            try:
                cur = runtime.get_candidate(stage.stage_id, current_id)
            except SgarError:
                cur = None
            if cur is not None and cur.status in (
                CandidateStatus.AUDITED_PASS.value,
                CandidateStatus.PROMOTED.value,
            ):
                outcome = _try_close_frontier_cursor(
                    runtime,
                    stage,
                    criteria,
                    current_id,
                    prior_attempts=max(prior_attempts, attempt - 1),
                    log=log,
                )
                if isinstance(outcome, StageReport):
                    return outcome
                detail, _discarded = outcome
                current_id = None
                last_candidate_id = None

        fed_detail = _feed_frontier_detail(detail, attempt=attempt)
        result = implement(stage, attempt, fed_detail)
        paths, summary = _parse_implement_result(result)
        if not summary:
            summary = f"autobuild attempt {attempt}"
        try:
            if current_id is None:
                if not paths:
                    raise SgarError(
                        "propose-candidate requires --artifact / artifact_paths"
                    )
                cid = _next_autobuild_candidate_id(runtime, stage.stage_id)
                record = runtime.propose_candidate(
                    stage.stage_id,
                    candidate_id=cid,
                    summary=summary,
                    artifact_paths=paths,
                    origin="propose",
                )
                current_id = record.candidate_id
                last_candidate_id = current_id
                log(f"{stage.stage_id}: proposed {current_id}")
            else:
                cur = runtime.get_candidate(stage.stage_id, current_id)
                if cur.status == CandidateStatus.AUDITED_FAIL.value:
                    if not paths:
                        raise SgarError(
                            "patch-candidate requires --artifact / artifact_paths"
                        )
                    cid = _next_autobuild_candidate_id(runtime, stage.stage_id)
                    record = runtime.patch_candidate(
                        stage.stage_id,
                        parent_id=current_id,
                        candidate_id=cid,
                        summary=summary,
                        artifact_paths=paths,
                    )
                    current_id = record.candidate_id
                    last_candidate_id = current_id
                    log(f"{stage.stage_id}: patched → {current_id}")
                elif cur.status == CandidateStatus.PROPOSED.value:
                    last_candidate_id = current_id
                    # Implement may have mutated artifacts after propose;
                    # drift is detected at from-checks below.
                else:
                    raise SgarError(
                        f"autobuild frontier cannot continue from status="
                        f"{cur.status!r} candidate={current_id!r}"
                    )

            cur = runtime.get_candidate(stage.stage_id, current_id)
            # Only skip verify/close re-checks when THIS attempt just ran
            # from-checks successfully (same tree, no further implement).
            audited_this_attempt = False
            if cur.status == CandidateStatus.PROPOSED.value:
                try:
                    cur = runtime.audit_candidate_from_checks(
                        stage.stage_id, current_id
                    )
                except SgarError as audit_exc:
                    if _looks_candidate_hash_drift(str(audit_exc)):
                        runtime.force_discard_candidate(
                            stage.stage_id, current_id
                        )
                        log(
                            f"{stage.stage_id}: discarded drifted "
                            f"{current_id}: {audit_exc}"
                        )
                        detail = str(audit_exc)
                        current_id = None
                        last_candidate_id = None
                        _persist_repair_progress(
                            runtime, stage.stage_id, attempt, detail
                        )
                        continue
                    raise
                audited_this_attempt = (
                    cur.status == CandidateStatus.AUDITED_PASS.value
                )
                log(f"{stage.stage_id}: audited {current_id} → {cur.status}")

            if cur.status == CandidateStatus.AUDITED_FAIL.value:
                if cur.audit and isinstance(cur.audit.extras.get("bound_detail"), str):
                    detail = cur.audit.extras["bound_detail"]
                elif cur.audit and cur.audit.findings:
                    detail = "\n".join(cur.audit.findings)
                else:
                    detail = f"candidate {current_id} audited_fail"
                _persist_repair_progress(runtime, stage.stage_id, attempt, detail)
                log(f"{stage.stage_id}: attempt {attempt} audited_fail: {current_id}")
                if stop_on_unrunnable_enabled() and _looks_unrunnable(detail):
                    log(
                        f"{stage.stage_id}: attempt {attempt} refusal looks like a "
                        "harness defect (the exit-criterion check could not execute), "
                        "not a candidate failure — the repair budget cannot fix this"
                    )
                continue

            if cur.status == CandidateStatus.AUDITED_PASS.value:
                runtime.promote_candidate(stage.stage_id, current_id)
                log(f"{stage.stage_id}: promoted {current_id}")
            elif cur.status != CandidateStatus.PROMOTED.value:
                raise SgarError(
                    f"expected audited_pass before promote, got {cur.status!r}"
                )

            # Fresh from-checks already gated [check:] criteria; avoid a second
            # (or third) run on record_verification/close. Resume of an older
            # audited_pass/promoted node keeps checks on.
            skip_recheck = audited_this_attempt
            prev_run_checks = runtime.run_criterion_checks
            if skip_recheck:
                runtime.run_criterion_checks = False
            try:
                runtime.record_verification(
                    stage.stage_id,
                    results=[
                        CriterionResult(c.criterion_id, True, "autobuild")
                        for c in criteria
                    ],
                )
                runtime.close_stage(stage.stage_id)
            finally:
                if skip_recheck:
                    runtime.run_criterion_checks = prev_run_checks
            _persist_repair_progress(runtime, stage.stage_id, attempt, None)
            log(f"{stage.stage_id}: closed on attempt {attempt} (frontier)")
            return StageReport(
                stage.stage_id,
                closed=True,
                attempts=attempt,
                last_candidate_id=current_id,
            )
        except SgarError as exc:
            detail = str(exc)
            if _looks_frontier_budget_exhausted(detail):
                frontier_budget_hit = True
                _persist_repair_progress(runtime, stage.stage_id, attempt, detail)
                log(f"{stage.stage_id}: frontier budget exhausted: {detail}")
                return StageReport(
                    stage.stage_id,
                    closed=False,
                    attempts=attempt,
                    last_error=detail,
                    frontier_budget_exhausted=True,
                    last_candidate_id=last_candidate_id,
                )
            if (
                current_id is not None
                and _looks_candidate_hash_drift(detail)
            ):
                try:
                    runtime.force_discard_candidate(stage.stage_id, current_id)
                    log(
                        f"{stage.stage_id}: discarded drifted "
                        f"{current_id} after refuse: {detail}"
                    )
                    current_id = None
                    last_candidate_id = None
                except SgarError:
                    pass
            _persist_repair_progress(runtime, stage.stage_id, attempt, detail)
            log(f"{stage.stage_id}: attempt {attempt} refused: {detail}")
            if stop_on_unrunnable_enabled() and _looks_unrunnable(detail):
                log(
                    f"{stage.stage_id}: attempt {attempt} refusal looks like a "
                    "harness defect (the exit-criterion check could not execute), "
                    "not a candidate failure — the repair budget cannot fix this"
                )

    suspected = stop_on_unrunnable_enabled() and _looks_unrunnable(detail)
    return StageReport(
        stage.stage_id,
        closed=False,
        attempts=max(prior_attempts, max_attempts),
        last_error=detail,
        harness_defect_suspected=suspected,
        frontier_budget_exhausted=frontier_budget_hit,
        last_candidate_id=last_candidate_id,
    )


def _persist_repair_progress(
    runtime: SgarRuntime,
    stage_id: str,
    attempts: int,
    detail: str | None,
) -> None:
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
    "ImplementResult",
    "Implementer",
    "ProjectPlan",
    "StagePlan",
    "StageReport",
    "autobuild",
]
