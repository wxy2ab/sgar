"""Validated SGAR runtime operations."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..modes.llm_client import text_of
from .models import (
    CriterionResult,
    DoctorResult,
    ProjectMode,
    ProjectState,
    SgarError,
    StageRecord,
    StageStatus,
    ValidationResult,
    VerificationReport,
)
from .missions import (
    complete_mission,
    create_mission,
    format_mission_list,
    format_mission_status,
    list_missions,
    load_mission,
)
from .checks import run_criterion_check
from .store import SgarStore, utc_now
from .tracing import SgarTracer, read_failed_trace, read_trace
from .validation import (
    doctor as run_doctor,
    extract_stage_ids_from_roadmap,
    load_verification,
    parse_exit_criteria,
    validate_blueprint_text,
    validate_roadmap_text,
    validate_stage_spec_text,
    validate_verification,
)

LlmDraftCallable = Callable[[str, str, str], str] | Callable[..., str]


ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    ProjectMode.BLUEPRINT.value: {
        ProjectMode.BLUEPRINT.value,
        ProjectMode.ROADMAP.value,
    },
    ProjectMode.ROADMAP.value: {
        ProjectMode.ROADMAP.value,
        ProjectMode.STAGE_READY.value,
    },
    ProjectMode.STAGE_READY.value: {
        ProjectMode.ROADMAP.value,
        ProjectMode.STAGE_READY.value,
        ProjectMode.EXECUTION.value,
    },
    ProjectMode.EXECUTION.value: {
        ProjectMode.EXECUTION.value,
        ProjectMode.VERIFICATION.value,
    },
    ProjectMode.VERIFICATION.value: {
        ProjectMode.VERIFICATION.value,
        ProjectMode.NEXT_STAGE_READY.value,
    },
    ProjectMode.NEXT_STAGE_READY.value: {
        ProjectMode.ROADMAP.value,
        ProjectMode.STAGE_READY.value,
        ProjectMode.EXECUTION.value,
        ProjectMode.NEXT_STAGE_READY.value,
    },
}


class SgarRuntime:
    def __init__(
        self,
        cwd: str | Path = ".",
        session_id: str | None = None,
        *,
        run_criterion_checks: bool = False,
        criterion_check_timeout_s: float = 120.0,
    ) -> None:
        self.store = SgarStore(cwd, session_id=session_id)
        self.tracer = SgarTracer(self.store)
        # P2: substantive verification. When True, verify/close run any
        # ``[check: <cmd>]`` declared on an exit criterion and refuse a pass
        # the command contradicts. Operator-controlled, default OFF — a spec
        # with no check, or this flag left off, behaves exactly as before.
        self.run_criterion_checks = run_criterion_checks
        self.criterion_check_timeout_s = criterion_check_timeout_s

    def init(
        self, *, project_name: str | None = None, force: bool = False,
    ) -> ProjectState:
        with self.tracer.operation(
            "init",
            inputs={"project_name": project_name, "force": force},
        ) as trace:
            state = self.store.initialize(project_name=project_name, force=force)
            trace["artifacts"] = self._artifacts(
                ("config", self.store.config_path),
                ("state", self.store.state_path),
                ("blueprint", self.store.blueprint_path),
                ("roadmap", self.store.roadmap_path),
                ("stage_spec", self.store.stage_spec_path("stage-01")),
            )
            return state

    def status(self) -> str:
        with self.tracer.operation("status") as trace:
            state = self._refresh_flags()
            trace["artifacts"] = self._artifacts(("state", self.store.state_path))
            lines = [
                f"Project: {state.project_name}",
                f"Mode: {state.mode}",
                f"Current stage: {state.current_stage_id or '-'}",
                f"Next stage: {state.next_stage_id or '-'}",
                f"Last closed stage: {state.last_closed_stage_id or '-'}",
            ]
            if state.roadmap_review_required:
                lines.append("Roadmap review required: yes")
            if state.future_stage_validation_required:
                lines.append("Future stage validation required: yes")
            return "\n".join(lines)

    def validate_blueprint(self, *, accept: bool = False) -> ValidationResult:
        with self.tracer.operation(
            "validate_blueprint",
            inputs={"accept": accept},
        ) as trace:
            state = self._refresh_flags()
            text = self.store.read_text(self.store.blueprint_path)
            result = validate_blueprint_text(text)
            if accept:
                result.require_ok()
                state.accepted_blueprint_hash = self.store.file_hash(
                    self.store.blueprint_path
                )
                if state.accepted_roadmap_hash:
                    state.roadmap_review_required = True
                self._set_mode(state, ProjectMode.ROADMAP.value)
                self.store.write_state(state)
            trace["artifacts"] = self._artifacts(
                ("blueprint", self.store.blueprint_path),
                ("state", self.store.state_path),
            )
            return result

    def set_blueprint(self, text: str) -> Path:
        with self.tracer.operation("write_blueprint") as trace:
            state = self._refresh_flags()
            self.store.write_text(self.store.blueprint_path, _ensure_trailing_newline(text))
            self._mark_blueprint_changed(state)
            self.store.write_state(state)
            trace["artifacts"] = self._artifacts(
                ("blueprint", self.store.blueprint_path),
                ("state", self.store.state_path),
            )
            return self.store.blueprint_path

    def set_roadmap(self, text: str) -> Path:
        with self.tracer.operation("write_roadmap") as trace:
            state = self._refresh_flags()
            self.store.write_text(self.store.roadmap_path, _ensure_trailing_newline(text))
            self._mark_roadmap_changed(state)
            self.store.write_state(state)
            trace["artifacts"] = self._artifacts(
                ("roadmap", self.store.roadmap_path),
                ("state", self.store.state_path),
            )
            return self.store.roadmap_path

    def set_stage_spec(self, stage_id: str, text: str) -> Path:
        with self.tracer.operation(
            "write_stage_spec",
            inputs={"stage_id": stage_id},
        ) as trace:
            self.store.load_state()
            self.store.write_text(
                self.store.stage_spec_path(stage_id),
                _ensure_trailing_newline(text),
            )
            trace["artifacts"] = self._artifacts(
                ("stage_spec", self.store.stage_spec_path(stage_id))
            )
            return self.store.stage_spec_path(stage_id)

    def draft_blueprint(
        self,
        *,
        llm: LlmDraftCallable,
        prompt: str = "",
    ) -> Path:
        with self.tracer.operation(
            "draft_blueprint",
            inputs={"prompt": prompt},
        ) as trace:
            context = self._draft_context(prompt)
            text = _call_draft_llm(
                llm,
                system=_draft_system("blueprint"),
                user=(
                    "Draft an SGAR blueprint.md for this repository. "
                    "Return Markdown only, with these headings: Problem, Goals, "
                    "Non-Goals, Constraints, Success Criteria.\n\n"
                    f"{context}"
                ),
                purpose="sgar_draft_blueprint",
            )
            self.store.write_text(self.store.blueprint_path, _strip_markdown_fence(text))
            state = self._refresh_flags()
            self._mark_blueprint_changed(state)
            self.store.write_state(state)
            trace["artifacts"] = self._artifacts(
                ("blueprint", self.store.blueprint_path),
                ("state", self.store.state_path),
            )
            return self.store.blueprint_path

    def draft_roadmap(
        self,
        *,
        llm: LlmDraftCallable,
        prompt: str = "",
    ) -> Path:
        with self.tracer.operation(
            "draft_roadmap",
            inputs={"prompt": prompt},
        ) as trace:
            context = self._draft_context(prompt)
            blueprint = self.store.read_text(self.store.blueprint_path)
            text = _call_draft_llm(
                llm,
                system=_draft_system("roadmap"),
                user=(
                    "Draft an SGAR roadmap.md for this repository. Return Markdown "
                    "only. It must include a '## Stages' section and at least "
                    "stage-01.\n\n"
                    f"Current blueprint:\n{blueprint}\n\n{context}"
                ),
                purpose="sgar_draft_roadmap",
            )
            self.store.write_text(self.store.roadmap_path, _strip_markdown_fence(text))
            state = self._refresh_flags()
            self._mark_roadmap_changed(state)
            self.store.write_state(state)
            trace["artifacts"] = self._artifacts(
                ("roadmap", self.store.roadmap_path),
                ("state", self.store.state_path),
            )
            return self.store.roadmap_path

    def draft_stage_spec(
        self,
        stage_id: str,
        *,
        llm: LlmDraftCallable,
        prompt: str = "",
    ) -> Path:
        with self.tracer.operation(
            "draft_stage_spec",
            inputs={"stage_id": stage_id, "prompt": prompt},
        ) as trace:
            context = self._draft_context(prompt)
            blueprint = self.store.read_text(self.store.blueprint_path)
            roadmap = self.store.read_text(self.store.roadmap_path)
            text = _call_draft_llm(
                llm,
                system=_draft_system("stage spec"),
                user=(
                    f"Draft an SGAR stages/{stage_id}/spec.md file. Return Markdown "
                    "only. It must include Objective, Scope, and Exit Criteria with "
                    "criterion ids such as C1 and C2.\n\n"
                    f"Blueprint:\n{blueprint}\n\nRoadmap:\n{roadmap}\n\n{context}"
                ),
                purpose="sgar_draft_stage_spec",
            )
            self.store.write_text(
                self.store.stage_spec_path(stage_id),
                _strip_markdown_fence(text),
            )
            trace["artifacts"] = self._artifacts(
                ("stage_spec", self.store.stage_spec_path(stage_id))
            )
            return self.store.stage_spec_path(stage_id)

    def validate_roadmap(self, *, accept: bool = False) -> ValidationResult:
        with self.tracer.operation(
            "validate_roadmap",
            inputs={"accept": accept},
        ) as trace:
            state = self._refresh_flags()
            text = self.store.read_text(self.store.roadmap_path)
            result = validate_roadmap_text(text)
            if accept:
                result.require_ok()
                current_blueprint = self.store.file_hash(self.store.blueprint_path)
                if state.accepted_blueprint_hash != current_blueprint:
                    raise SgarError(
                        "roadmap cannot be accepted without current accepted blueprint"
                    )
                state.accepted_roadmap_hash = self.store.file_hash(
                    self.store.roadmap_path
                )
                state.roadmap_review_required = False
                state.future_stage_validation_required = False
                stage_ids = extract_stage_ids_from_roadmap(text)
                state.next_stage_id = self._first_open_stage(state, stage_ids)
                self._set_mode(state, ProjectMode.STAGE_READY.value)
                self.store.write_state(state)
            trace["artifacts"] = self._artifacts(
                ("roadmap", self.store.roadmap_path),
                ("state", self.store.state_path),
            )
            return result

    def validate_stage_spec(self, stage_id: str) -> ValidationResult:
        with self.tracer.operation(
            "validate_stage_spec",
            inputs={"stage_id": stage_id},
        ) as trace:
            state = self._refresh_flags()
            text = self.store.read_text(self.store.stage_spec_path(stage_id))
            result = validate_stage_spec_text(text)
            if result.ok:
                state.validated_stage_spec_hashes[stage_id] = self.store.file_hash(
                    self.store.stage_spec_path(stage_id)
                )
                self.store.write_state(state)
            trace["artifacts"] = self._artifacts(
                ("stage_spec", self.store.stage_spec_path(stage_id)),
                ("state", self.store.state_path),
            )
            return result

    def start_stage(self, stage_id: str) -> ProjectState:
        with self.tracer.operation(
            "start_stage",
            inputs={"stage_id": stage_id},
        ) as trace:
            state = self._refresh_flags()
            current_roadmap = self.store.file_hash(self.store.roadmap_path)
            if not state.accepted_blueprint_hash:
                raise SgarError("stage cannot start without an accepted blueprint")
            if state.accepted_roadmap_hash != current_roadmap:
                raise SgarError("stage cannot start without an accepted roadmap")
            if state.roadmap_review_required:
                raise SgarError("stage cannot start while roadmap review is required")
            if state.future_stage_validation_required:
                raise SgarError(
                    "stage cannot start while future stage validation is required"
                )
            if stage_id in state.closed_stage_ids:
                raise SgarError(
                    f"stage {stage_id} is already closed; use sgarx reopen-stage to revisit it"
                )
            if state.current_stage_id is not None:
                raise SgarError(
                    f"cannot start {stage_id}; current stage is {state.current_stage_id}"
                )
            stage_ids = extract_stage_ids_from_roadmap(
                self.store.read_text(self.store.roadmap_path)
            )
            if stage_id not in stage_ids:
                raise SgarError(f"stage is not listed in roadmap: {stage_id}")
            spec_hash = self.store.file_hash(self.store.stage_spec_path(stage_id))
            if state.validated_stage_spec_hashes.get(stage_id) != spec_hash:
                raise SgarError(
                    "stage cannot start without a current validated stage spec"
                )
            spec_text = self.store.read_text(self.store.stage_spec_path(stage_id))
            spec_result = validate_stage_spec_text(spec_text)
            spec_result.require_ok()
            criteria = parse_exit_criteria(spec_text)
            self._validate_transition(state, ProjectMode.EXECUTION.value)
            self._write_stage_execution_files(stage_id, criteria)
            record = state.stages.get(stage_id) or StageRecord(stage_id=stage_id)
            record.status = StageStatus.EXECUTION.value
            record.started_at = record.started_at or utc_now()
            state.stages[stage_id] = record
            state.current_stage_id = stage_id
            state.next_stage_id = stage_id
            self._set_mode(state, ProjectMode.EXECUTION.value)
            self.store.write_state(state)
            trace["artifacts"] = self._artifacts(
                ("stage_spec", self.store.stage_spec_path(stage_id)),
                ("stage_plan", self.store.stage_plan_path(stage_id)),
                ("stage_tasks", self.store.stage_tasks_path(stage_id)),
                ("stage_context", self.store.stage_context_path(stage_id)),
                ("state", self.store.state_path),
            )
            return state

    def record_verification(
        self,
        stage_id: str,
        *,
        results: list[CriterionResult],
        notes: str = "",
        artifact_paths: list[str | Path] | None = None,
    ) -> VerificationReport:
        with self.tracer.operation(
            "record_verification",
            inputs={
                "stage_id": stage_id,
                "criterion_ids": [result.criterion_id for result in results],
            },
        ) as trace:
            state = self._refresh_flags()
            if state.current_stage_id != stage_id:
                raise SgarError(
                    f"cannot verify {stage_id}; current stage is {state.current_stage_id}"
                )
            spec_text = self.store.read_text(self.store.stage_spec_path(stage_id))
            criteria = parse_exit_criteria(spec_text)
            if not criteria:
                raise SgarError("stage verification requires exit criteria")
            known = {criterion.criterion_id for criterion in criteria}
            for result in results:
                if result.criterion_id not in known:
                    raise SgarError(f"unknown exit criterion: {result.criterion_id}")

            # P2: machine-back the agent's pass claims. For each criterion the
            # agent is marking --pass that declares a [check: ...], run the
            # check ourselves and REFUSE the pass if it fails — the false pass
            # is never recorded. Runs before any write so a failure aborts the
            # whole op (and is captured as a failed trace record + governance
            # error). Opt-in; a --fail result skips its check (under-claiming
            # is not a governance risk), and a disabled runtime is a no-op.
            if self.run_criterion_checks:
                by_criterion = {c.criterion_id: c for c in criteria}
                for result in results:
                    criterion = by_criterion.get(result.criterion_id)
                    if criterion is None or not criterion.check or not result.passed:
                        continue
                    outcome = run_criterion_check(
                        criterion,
                        cwd=self.store.cwd,
                        timeout_s=self.criterion_check_timeout_s,
                    )
                    if not outcome.passed:
                        raise SgarError(
                            f"exit criterion {result.criterion_id} machine "
                            f"check failed: {outcome.evidence_line()}"
                        )
                    result.evidence = _join_evidence(
                        result.evidence, outcome.evidence_line(),
                    )

            existing: dict[str, CriterionResult] = {}
            if self.store.verification_json_path(stage_id).exists():
                for result in load_verification(self.store, stage_id).results:
                    existing[result.criterion_id] = result
            for result in results:
                existing[result.criterion_id] = result
            report = VerificationReport(
                stage_id=stage_id,
                results=list(existing.values()),
                notes=notes,
            )
            self._validate_transition(state, ProjectMode.VERIFICATION.value)
            self.store.write_json(
                self.store.verification_json_path(stage_id),
                report.to_dict(),
            )
            self.store.write_text(
                self.store.verification_md_path(stage_id),
                self._format_verification_md(criteria, report),
            )
            record = state.stages.get(stage_id) or StageRecord(stage_id=stage_id)
            record.status = StageStatus.VERIFICATION.value
            state.stages[stage_id] = record
            state.current_stage_id = stage_id
            self._set_mode(state, ProjectMode.VERIFICATION.value)
            self.store.write_state(state)
            artifacts = self._artifacts(
                ("verification_json", self.store.verification_json_path(stage_id)),
                ("verification_md", self.store.verification_md_path(stage_id)),
                ("state", self.store.state_path),
            )
            for path in artifact_paths or []:
                artifacts.append(self.tracer.artifact(self.store.cwd / path))
            trace["artifacts"] = artifacts
            return report

    def close_stage(self, stage_id: str) -> ProjectState:
        with self.tracer.operation(
            "close_stage",
            inputs={"stage_id": stage_id},
        ) as trace:
            state = self._refresh_flags()
            if state.current_stage_id != stage_id:
                raise SgarError(
                    f"cannot close {stage_id}; current stage is {state.current_stage_id}"
                )
            spec_text = self.store.read_text(self.store.stage_spec_path(stage_id))
            criteria = parse_exit_criteria(spec_text)
            if not self.store.verification_json_path(stage_id).exists():
                raise SgarError("stage cannot close without a verification report")
            report = load_verification(self.store, stage_id)
            validate_verification(criteria=criteria, report=report).require_ok()
            # Code-task definition-of-done (default OFF). Append the same
            # planner-independent code-task criterion the other governed modes
            # inject, so a stage that edits production code cannot close without
            # wiring + scoped tests green. Injected AFTER ``validate_verification``
            # (which requires a CriterionResult per spec criterion) so the
            # appended criterion is only ever a machine-run close gate, not a
            # report-coverage requirement. Self-gates to a trivial pass when no
            # production .py changed. Flag unset ⇒ ``criteria`` is untouched.
            from core.ccx.audit import code_task_audit_enabled
            if self.run_criterion_checks and code_task_audit_enabled():
                from core.ccx.audit import build_code_task_contract
                criteria = [
                    *criteria,
                    *build_code_task_contract("criteria", cwd=str(self.store.cwd)),
                ]
            # P2: final hard gate. Re-run machine checks for blocking checked
            # criteria so the DAG cannot advance on a check that no longer
            # holds (e.g. the working tree regressed between verify and
            # close). Opt-in; no-op when disabled or when no blocking
            # criterion declares a check.
            if self.run_criterion_checks:
                for criterion in criteria:
                    if not criterion.check or not criterion.blocking:
                        continue
                    outcome = run_criterion_check(
                        criterion,
                        cwd=self.store.cwd,
                        timeout_s=self.criterion_check_timeout_s,
                    )
                    if not outcome.passed:
                        raise SgarError(
                            f"stage {stage_id} cannot close: exit criterion "
                            f"{criterion.criterion_id} machine check failed: "
                            f"{outcome.evidence_line()}"
                        )
            stage_ids = extract_stage_ids_from_roadmap(
                self.store.read_text(self.store.roadmap_path)
            )
            next_stage = self._stage_after(stage_ids, stage_id)
            self._validate_transition(state, ProjectMode.NEXT_STAGE_READY.value)
            self.store.write_text(
                self.store.handoff_path(stage_id),
                self._format_handoff_md(stage_id, report, next_stage),
            )
            record = state.stages.get(stage_id) or StageRecord(stage_id=stage_id)
            record.status = StageStatus.CLOSED.value
            record.closed_at = utc_now()
            state.stages[stage_id] = record
            if stage_id not in state.closed_stage_ids:
                state.closed_stage_ids.append(stage_id)
            state.current_stage_id = None
            state.last_closed_stage_id = stage_id
            state.next_stage_id = next_stage
            self._set_mode(state, ProjectMode.NEXT_STAGE_READY.value)
            self.store.write_state(state)
            trace["artifacts"] = self._artifacts(
                ("handoff", self.store.handoff_path(stage_id)),
                ("state", self.store.state_path),
            )
            return state

    def doctor(self) -> DoctorResult:
        with self.tracer.operation("doctor") as trace:
            result = run_doctor(self.store)
            trace["artifacts"] = self._artifacts(("state", self.store.state_path))
            return result

    def create_mission(
        self,
        *,
        mission_id: str,
        kind: str,
        objective: str,
        input_paths: list[str | Path],
        expected_outputs: list[str],
        allowed_scope: list[str] | None = None,
    ) -> dict:
        with self.tracer.operation(
            "create_mission",
            inputs={
                "mission_id": mission_id,
                "kind": kind,
                "input_paths": [str(path) for path in input_paths],
                "expected_outputs": list(expected_outputs),
            },
        ) as trace:
            manifest = create_mission(
                self.store,
                mission_id=mission_id,
                kind=kind,
                objective=objective,
                input_paths=input_paths,
                expected_outputs=expected_outputs,
                allowed_scope=allowed_scope,
            )
            mission_dir = self.store.missions_root / mission_id
            trace["artifacts"] = self._artifacts(
                ("mission_manifest", mission_dir / "manifest.json"),
                ("mission_context", mission_dir / "context.md"),
            )
            return manifest

    def complete_mission(
        self,
        mission_id: str,
        *,
        result_path: str | Path,
    ) -> dict:
        with self.tracer.operation(
            "complete_mission",
            inputs={"mission_id": mission_id, "result_path": str(result_path)},
        ) as trace:
            manifest = complete_mission(
                self.store,
                mission_id=mission_id,
                result_path=result_path,
            )
            mission_dir = self.store.missions_root / mission_id
            artifacts = self._artifacts(
                ("mission_manifest", mission_dir / "manifest.json")
            )
            for output in manifest.get("recorded_outputs") or []:
                if isinstance(output, dict) and output.get("path"):
                    artifacts.append(
                        self.tracer.artifact(
                            mission_dir / str(output["path"]),
                            role="mission_output",
                        )
                    )
            trace["artifacts"] = artifacts
            return manifest

    def mission_status(self, mission_id: str) -> str:
        with self.tracer.operation(
            "mission_status",
            inputs={"mission_id": mission_id},
        ) as trace:
            trace["artifacts"] = self._artifacts(
                ("mission_manifest", self.store.missions_root / mission_id / "manifest.json")
            )
            return format_mission_status(self.store, mission_id)

    def list_missions(self) -> list[dict]:
        with self.tracer.operation("list_missions"):
            return list_missions(self.store)

    def mission_list_text(self) -> str:
        with self.tracer.operation("mission_list_text"):
            return format_mission_list(self.store)

    def load_mission(self, mission_id: str) -> dict:
        with self.tracer.operation(
            "load_mission",
            inputs={"mission_id": mission_id},
        ):
            return load_mission(self.store, mission_id)

    def trace_records(self) -> list[dict]:
        return read_trace(self.store)

    def failed_trace_records(self) -> list[dict]:
        """Trace records for operations that were refused (status=failed).

        Mirrors :meth:`trace_records`; outside-driven drivers already filter
        ``trace_records()`` for ``status == "completed"`` to report progress,
        so this is the symmetric way to surface governance refusals from the
        on-disk trace without re-implementing the scan at every call site.
        """
        return read_failed_trace(self.store)

    def _refresh_flags(self) -> ProjectState:
        state = self.store.load_state()
        changed = False
        if (
            state.accepted_blueprint_hash
            and self.store.blueprint_path.exists()
            and self.store.file_hash(self.store.blueprint_path)
            != state.accepted_blueprint_hash
        ):
            if not state.roadmap_review_required:
                state.roadmap_review_required = True
                changed = True
        if (
            state.accepted_roadmap_hash
            and self.store.roadmap_path.exists()
            and self.store.file_hash(self.store.roadmap_path)
            != state.accepted_roadmap_hash
        ):
            if not state.future_stage_validation_required:
                state.future_stage_validation_required = True
                changed = True
        if changed:
            self.store.write_state(state)
        return state

    def _mark_blueprint_changed(self, state: ProjectState) -> None:
        if (
            state.accepted_blueprint_hash
            and self.store.file_hash(self.store.blueprint_path)
            != state.accepted_blueprint_hash
        ):
            state.roadmap_review_required = True

    def _mark_roadmap_changed(self, state: ProjectState) -> None:
        if (
            state.accepted_roadmap_hash
            and self.store.file_hash(self.store.roadmap_path)
            != state.accepted_roadmap_hash
        ):
            state.future_stage_validation_required = True

    def _draft_context(self, prompt: str) -> str:
        state = self.store.load_state()
        parts = [
            f"Project: {state.project_name}",
            f"Current mode: {state.mode}",
        ]
        if prompt.strip():
            parts += ["Additional user prompt:", prompt.strip()]
        return "\n".join(parts)

    def _set_mode(self, state: ProjectState, new_mode: str) -> None:
        self._validate_transition(state, new_mode)
        state.mode = new_mode

    def _validate_transition(self, state: ProjectState, new_mode: str) -> None:
        allowed = ALLOWED_TRANSITIONS.get(state.mode, set())
        if new_mode not in allowed:
            raise SgarError(
                f"invalid SGAR transition: {state.mode} -> {new_mode}"
            )

    def _write_stage_execution_files(
        self, stage_id: str, criteria: list,
    ) -> None:
        self.store.write_text(
            self.store.stage_plan_path(stage_id),
            f"""# Plan: {stage_id}

This plan is flexible inside the current stage. Tasks may be split, merged,
reordered, or revised as implementation evidence changes, while the stage
scope and exit criteria remain governed by `spec.md`.
""",
        )
        tasks = [f"- [ ] Satisfy {c.criterion_id}: {c.description}" for c in criteria]
        self.store.write_text(
            self.store.stage_tasks_path(stage_id),
            "# Tasks\n\n" + "\n".join(tasks) + "\n",
        )
        self.store.write_text(
            self.store.stage_context_path(stage_id),
            f"""# Context Packet: {stage_id}

- Blueprint: `{self.store.blueprint_path}`
- Roadmap: `{self.store.roadmap_path}`
- Stage spec: `{self.store.stage_spec_path(stage_id)}`
- Blueprint hash: `{self.store.file_hash(self.store.blueprint_path)}`
- Roadmap hash: `{self.store.file_hash(self.store.roadmap_path)}`
""",
        )

    def _format_verification_md(
        self, criteria: list, report: VerificationReport,
    ) -> str:
        by_id = {result.criterion_id: result for result in report.results}
        lines = [f"# Verification: {report.stage_id}", ""]
        for criterion in criteria:
            result = by_id.get(criterion.criterion_id)
            if result is None:
                verdict = "missing"
                evidence = ""
            else:
                verdict = "pass" if result.passed else "fail"
                evidence = result.evidence
            blocking = "blocking" if criterion.blocking else "non-blocking"
            lines.append(
                f"- {criterion.criterion_id} ({blocking}): {verdict}"
            )
            if evidence:
                lines.append(f"  Evidence: {evidence}")
        if report.notes:
            lines += ["", "## Notes", "", report.notes]
        return "\n".join(lines) + "\n"

    def _format_handoff_md(
        self,
        stage_id: str,
        report: VerificationReport,
        next_stage: str | None,
    ) -> str:
        lines = [
            f"# Handoff: {stage_id}",
            "",
            f"Closed at: {utc_now()}",
            f"Next stage: {next_stage or '-'}",
            "",
            "## Verification Summary",
            "",
        ]
        for result in report.results:
            verdict = "pass" if result.passed else "fail"
            lines.append(f"- {result.criterion_id}: {verdict}")
        return "\n".join(lines) + "\n"

    def _first_open_stage(
        self, state: ProjectState, stage_ids: list[str],
    ) -> str | None:
        for stage_id in stage_ids:
            if stage_id not in state.closed_stage_ids:
                return stage_id
        return None

    def _stage_after(
        self, stage_ids: list[str], stage_id: str,
    ) -> str | None:
        try:
            idx = stage_ids.index(stage_id)
        except ValueError:
            return None
        if idx + 1 < len(stage_ids):
            return stage_ids[idx + 1]
        return None

    def _artifacts(self, *items: tuple[str, Path]) -> list[dict]:
        return [self.tracer.artifact(path, role=role) for role, path in items]


def result_to_text(result: ValidationResult | DoctorResult) -> str:
    if result.ok:
        lines = ["OK"]
    else:
        lines = ["FAILED"]
    for issue in result.issues:
        lines.append(f"- {issue}")
    for warning in result.warnings:
        lines.append(f"warning: {warning}")
    return "\n".join(lines)


def state_to_dict(state: ProjectState) -> dict:
    return state.to_dict()


def criterion_result_from_bool(
    criterion_id: str, passed: bool, evidence: str,
) -> CriterionResult:
    return CriterionResult(
        criterion_id=criterion_id,
        passed=passed,
        evidence=evidence,
    )


def _ensure_trailing_newline(text: str) -> str:
    return text if text.endswith("\n") else text + "\n"


def _join_evidence(existing: str, addition: str) -> str:
    existing = (existing or "").strip()
    addition = (addition or "").strip()
    if not existing:
        return addition
    if not addition:
        return existing
    return f"{existing}\n{addition}"


def _draft_system(kind: str) -> str:
    return (
        "You draft SGAR governance artifacts. SGAR is a hard state machine: "
        "your draft must not claim it is accepted, validated, or started. "
        f"Produce only the requested {kind} Markdown artifact."
    )


def _call_draft_llm(
    llm: LlmDraftCallable,
    *,
    system: str,
    user: str,
    purpose: str,
) -> str:
    try:
        return text_of(llm(system=system, user=user, purpose=purpose))
    except TypeError:
        return text_of(llm(system, user, purpose))  # type: ignore[misc]


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2:
            stripped = "\n".join(lines[1:-1]).strip()
    return _ensure_trailing_newline(stripped)


__all__ = [
    "SgarRuntime",
    "criterion_result_from_bool",
    "result_to_text",
    "state_to_dict",
]
