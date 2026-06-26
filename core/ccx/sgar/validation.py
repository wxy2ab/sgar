"""Validation helpers for SGAR documents and state."""

from __future__ import annotations

import re
from pathlib import Path

from .models import (
    DoctorResult,
    ExitCriterion,
    ProjectMode,
    ProjectState,
    SgarError,
    ValidationResult,
    VerificationReport,
)
from .missions import validate_mission_records
from .store import SgarStore


REQUIRED_BLUEPRINT_HEADINGS = [
    "Problem",
    "Goals",
    "Non-Goals",
    "Constraints",
    "Success Criteria",
]


def _heading_names(markdown: str) -> set[str]:
    headings: set[str] = set()
    for line in markdown.splitlines():
        match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", line)
        if match:
            heading = match.group(1).strip().rstrip("#").strip()
            headings.add(heading.lower())
    return headings


def has_heading(markdown: str, heading: str) -> bool:
    return heading.lower() in _heading_names(markdown)


def validate_blueprint_text(markdown: str) -> ValidationResult:
    headings = _heading_names(markdown)
    missing = [
        heading for heading in REQUIRED_BLUEPRINT_HEADINGS
        if heading.lower() not in headings
    ]
    return ValidationResult(
        ok=not missing,
        issues=[f"blueprint missing heading: {heading}" for heading in missing],
    )


def extract_stage_ids_from_roadmap(markdown: str) -> list[str]:
    if not has_heading(markdown, "Stages"):
        return []
    stage_ids: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"\bstage[-_][A-Za-z0-9_.-]+\b", markdown):
        stage_id = match.group(0)
        if stage_id not in seen:
            seen.add(stage_id)
            stage_ids.append(stage_id)
    return stage_ids


def validate_roadmap_text(markdown: str) -> ValidationResult:
    issues: list[str] = []
    if not has_heading(markdown, "Stages"):
        issues.append("roadmap missing heading: Stages")
    if not extract_stage_ids_from_roadmap(markdown):
        issues.append("roadmap defines no stage ids")
    return ValidationResult(ok=not issues, issues=issues)


def _section_lines(markdown: str, heading: str) -> list[str]:
    lines = markdown.splitlines()
    out: list[str] = []
    in_section = False
    start_level = 0
    for line in lines:
        match = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$", line)
        if match:
            level = len(match.group(1))
            name = match.group(2).strip().rstrip("#").strip()
            if in_section and level <= start_level:
                break
            if name.lower() == heading.lower():
                in_section = True
                start_level = level
                continue
        elif in_section:
            out.append(line)
    return out


def parse_exit_criteria(markdown: str) -> list[ExitCriterion]:
    lines = _section_lines(markdown, "Exit Criteria")
    criteria: list[ExitCriterion] = []
    for line in lines:
        match = re.match(r"^\s*[-*]\s+(.+?)\s*$", line)
        if not match:
            continue
        raw = match.group(1).strip()
        raw = re.sub(r"^\[[ xX]\]\s*", "", raw)
        # Extract an optional trailing ``[check: <command>]`` BEFORE any ':'
        # splitting, so a ':' inside the command (e.g. ``sh -c "a:b"``) can
        # never be mistaken for the criterion-id separator. The check is the
        # LAST bracketed element on the line; non-greedy + ``$`` anchor lets
        # the command itself contain ']'.
        check: str | None = None
        check_match = re.search(r"\[check:\s*(.+?)\]\s*$", raw, re.I)
        if check_match:
            check = check_match.group(1).strip() or None
            raw = raw[: check_match.start()].rstrip()
        marker = re.match(r"^\[(blocking|non-blocking)\]\s*(.+)$", raw, re.I)
        blocking = True
        if marker:
            blocking = marker.group(1).lower() != "non-blocking"
            raw = marker.group(2).strip()
        if ":" in raw:
            criterion_id, description = raw.split(":", 1)
            criterion_id = criterion_id.strip()
            description = description.strip()
        else:
            criterion_id = f"C{len(criteria) + 1}"
            description = raw
        if criterion_id and description:
            criteria.append(ExitCriterion(
                criterion_id=criterion_id,
                description=description,
                blocking=blocking,
                check=check,
            ))
    return criteria


def validate_stage_spec_text(markdown: str) -> ValidationResult:
    issues: list[str] = []
    if not has_heading(markdown, "Exit Criteria"):
        issues.append("stage spec missing heading: Exit Criteria")
    if not parse_exit_criteria(markdown):
        issues.append("stage spec has no exit criteria")
    return ValidationResult(ok=not issues, issues=issues)


def validate_verification(
    *, criteria: list[ExitCriterion], report: VerificationReport,
) -> ValidationResult:
    by_id = {result.criterion_id: result for result in report.results}
    issues: list[str] = []
    for criterion in criteria:
        result = by_id.get(criterion.criterion_id)
        if criterion.blocking and result is None:
            issues.append(
                f"blocking exit criterion not verified: {criterion.criterion_id}"
            )
        elif criterion.blocking and not result.passed:
            issues.append(
                f"blocking exit criterion failed: {criterion.criterion_id}"
            )
    return ValidationResult(ok=not issues, issues=issues)


def load_verification(store: SgarStore, stage_id: str) -> VerificationReport:
    data = store.read_json(store.verification_json_path(stage_id))
    return VerificationReport.from_dict(data)


def doctor(store: SgarStore) -> DoctorResult:
    issues: list[str] = []
    warnings: list[str] = []
    if not store.root.exists():
        return DoctorResult(
            ok=False,
            issues=[f"missing SGAR workspace: {store.root}"],
        )

    required: list[Path] = [
        store.config_path,
        store.state_path,
        store.blueprint_path,
        store.roadmap_path,
    ]
    for path in required:
        if not path.exists():
            issues.append(f"missing file: {path}")

    state: ProjectState | None = None
    if store.state_path.exists():
        try:
            state = store.load_state()
        except SgarError as exc:
            issues.append(str(exc))

    if state is not None:
        valid_modes = {mode.value for mode in ProjectMode}
        if state.mode not in valid_modes:
            issues.append(f"invalid project mode: {state.mode}")
        if state.accepted_roadmap_hash and not state.accepted_blueprint_hash:
            issues.append("roadmap accepted without an accepted blueprint")
        if state.current_stage_id:
            if state.current_stage_id not in state.stages:
                issues.append(
                    f"current stage missing from state: {state.current_stage_id}"
                )
            if not store.stage_spec_path(state.current_stage_id).exists():
                issues.append(
                    f"current stage spec missing: {state.current_stage_id}"
                )
        if state.mode in {
            ProjectMode.EXECUTION.value,
            ProjectMode.VERIFICATION.value,
        } and not state.current_stage_id:
            issues.append(f"mode {state.mode} requires current_stage_id")

    if store.blueprint_path.exists():
        result = validate_blueprint_text(
            store.blueprint_path.read_text(encoding="utf-8")
        )
        warnings.extend(result.issues)
    if store.roadmap_path.exists():
        result = validate_roadmap_text(
            store.roadmap_path.read_text(encoding="utf-8")
        )
        warnings.extend(result.issues)

    mission_issues, mission_warnings = validate_mission_records(store)
    issues.extend(mission_issues)
    warnings.extend(mission_warnings)

    return DoctorResult(ok=not issues, issues=issues, warnings=warnings)


__all__ = [
    "REQUIRED_BLUEPRINT_HEADINGS",
    "doctor",
    "extract_stage_ids_from_roadmap",
    "load_verification",
    "parse_exit_criteria",
    "validate_blueprint_text",
    "validate_roadmap_text",
    "validate_stage_spec_text",
    "validate_verification",
]
