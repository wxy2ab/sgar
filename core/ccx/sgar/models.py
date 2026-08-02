"""Data models for the Stage-Governed Agent Runtime.

SGAR keeps its durable state intentionally small: human-facing Markdown
documents carry governance intent, while JSON stores only the machine state
needed to validate hard transitions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Any


class SgarError(RuntimeError):
    """Raised when an SGAR operation violates governance rules."""


class ProjectMode(str, Enum):
    BLUEPRINT = "blueprint"
    ROADMAP = "roadmap"
    STAGE_READY = "stage_ready"
    EXECUTION = "execution"
    VERIFICATION = "verification"
    NEXT_STAGE_READY = "next_stage_ready"


class StageStatus(str, Enum):
    PLANNED = "planned"
    EXECUTION = "execution"
    VERIFICATION = "verification"
    CLOSED = "closed"


def _known_field_names(cls: type) -> frozenset[str]:
    return frozenset(f.name for f in fields(cls))


def _extract_extras(data: dict[str, Any], known: frozenset[str]) -> dict[str, Any]:
    """Capture unknown keys (and an explicit ``extras`` map) for round-trip."""
    extras: dict[str, Any] = {}
    nested = data.get("extras")
    if isinstance(nested, dict):
        extras.update(nested)
    for key, value in data.items():
        if key in known or key == "extras":
            continue
        extras[key] = value
    return extras


def _merge_extras(payload: dict[str, Any], extras: dict[str, Any]) -> dict[str, Any]:
    """Flatten ``extras`` into the serialized payload (no nested ``extras`` key)."""
    out = dict(payload)
    out.pop("extras", None)
    for key, value in extras.items():
        if key not in out:
            out[key] = value
    return out


@dataclass(slots=True)
class ExitCriterion:
    criterion_id: str
    description: str
    blocking: bool = True
    check: str | None = None
    """Optional machine-checkable command for this criterion.

    Parsed from a ``[check: <command>]`` suffix on the spec line. When set
    AND the runtime has check execution enabled (opt-in, default off), SGAR
    runs the command itself during verify/close and refuses a ``--pass`` the
    command contradicts (exit code 0 = pass). ``None`` → self-reported
    verification only, exactly as before.
    """
    scope: tuple[str, ...] = ()
    """Optional path prefixes this check depends on (principle 7).

    Empty ⇒ global check (always re-run). When set, incremental verification
    may skip the check if the latest patch's changed paths do not intersect.
    """

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExitCriterion":
        check = data.get("check")
        scope_raw = data.get("scope") or ()
        if isinstance(scope_raw, str):
            scope: tuple[str, ...] = (scope_raw,) if scope_raw.strip() else ()
        elif isinstance(scope_raw, (list, tuple)):
            scope = tuple(str(s) for s in scope_raw if str(s).strip())
        else:
            scope = ()
        return cls(
            criterion_id=str(data.get("criterion_id") or ""),
            description=str(data.get("description") or ""),
            blocking=bool(data.get("blocking", True)),
            check=str(check) if check else None,
            scope=scope,
        )


@dataclass(slots=True)
class CriterionResult:
    criterion_id: str
    passed: bool
    evidence: str = ""
    # Forward-compat: unknown keys from newer writers survive load→mutate→save.
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _merge_extras(
            {
                "criterion_id": self.criterion_id,
                "passed": self.passed,
                "evidence": self.evidence,
            },
            self.extras,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CriterionResult":
        known = _known_field_names(cls)
        return cls(
            criterion_id=str(data.get("criterion_id") or ""),
            passed=bool(data.get("passed", False)),
            evidence=str(data.get("evidence") or ""),
            extras=_extract_extras(data, known),
        )


@dataclass(slots=True)
class VerificationReport:
    stage_id: str
    results: list[CriterionResult] = field(default_factory=list)
    notes: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _merge_extras(
            {
                "stage_id": self.stage_id,
                "results": [result.to_dict() for result in self.results],
                "notes": self.notes,
            },
            self.extras,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VerificationReport":
        known = _known_field_names(cls)
        return cls(
            stage_id=str(data.get("stage_id") or ""),
            results=[
                CriterionResult.from_dict(item)
                for item in data.get("results", [])
                if isinstance(item, dict)
            ],
            notes=str(data.get("notes") or ""),
            extras=_extract_extras(data, known),
        )


@dataclass(slots=True)
class StageRecord:
    stage_id: str
    status: str = StageStatus.PLANNED.value
    started_at: str | None = None
    closed_at: str | None = None
    # Repair-loop control-state (autobuild). Persisted so a process killed
    # mid-stage RESUMES the bounded-repair loop deterministically instead of
    # cold-restarting it: ``repair_attempts`` is the cumulative attempts
    # consumed for this stage (the budget continues, it is not silently
    # refilled every restart), and ``last_failure_detail`` is the previous
    # verify/close refusal's failing-``[check:]`` evidence to re-feed the next
    # implement attempt (the Implementer contract). Both default to the
    # never-attempted state so pre-existing state.json rows load unchanged.
    repair_attempts: int = 0
    last_failure_detail: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _merge_extras(
            {
                "stage_id": self.stage_id,
                "status": self.status,
                "started_at": self.started_at,
                "closed_at": self.closed_at,
                "repair_attempts": self.repair_attempts,
                "last_failure_detail": self.last_failure_detail,
            },
            self.extras,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StageRecord":
        known = _known_field_names(cls)
        return cls(
            stage_id=str(data.get("stage_id") or ""),
            status=str(data.get("status") or StageStatus.PLANNED.value),
            started_at=data.get("started_at"),
            closed_at=data.get("closed_at"),
            repair_attempts=int(data.get("repair_attempts") or 0),
            last_failure_detail=(
                str(data["last_failure_detail"])
                if data.get("last_failure_detail") is not None else None
            ),
            extras=_extract_extras(data, known),
        )


@dataclass(slots=True)
class ProjectState:
    project_name: str
    mode: str = ProjectMode.BLUEPRINT.value
    current_stage_id: str | None = None
    next_stage_id: str | None = None
    last_closed_stage_id: str | None = None
    closed_stage_ids: list[str] = field(default_factory=list)
    accepted_blueprint_hash: str | None = None
    accepted_roadmap_hash: str | None = None
    validated_stage_spec_hashes: dict[str, str] = field(default_factory=dict)
    roadmap_review_required: bool = False
    future_stage_validation_required: bool = False
    stages: dict[str, StageRecord] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _merge_extras(
            {
                "project_name": self.project_name,
                "mode": self.mode,
                "current_stage_id": self.current_stage_id,
                "next_stage_id": self.next_stage_id,
                "last_closed_stage_id": self.last_closed_stage_id,
                "closed_stage_ids": list(self.closed_stage_ids),
                "accepted_blueprint_hash": self.accepted_blueprint_hash,
                "accepted_roadmap_hash": self.accepted_roadmap_hash,
                "validated_stage_spec_hashes": dict(self.validated_stage_spec_hashes),
                "roadmap_review_required": self.roadmap_review_required,
                "future_stage_validation_required": self.future_stage_validation_required,
                "stages": {
                    stage_id: record.to_dict()
                    for stage_id, record in self.stages.items()
                },
            },
            self.extras,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectState":
        known = _known_field_names(cls)
        stages = {
            str(stage_id): StageRecord.from_dict(record)
            for stage_id, record in (data.get("stages") or {}).items()
            if isinstance(record, dict)
        }
        return cls(
            project_name=str(data.get("project_name") or "sgar-project"),
            mode=str(data.get("mode") or ProjectMode.BLUEPRINT.value),
            current_stage_id=data.get("current_stage_id"),
            next_stage_id=data.get("next_stage_id"),
            last_closed_stage_id=data.get("last_closed_stage_id"),
            closed_stage_ids=[
                str(item) for item in data.get("closed_stage_ids", [])
            ],
            accepted_blueprint_hash=data.get("accepted_blueprint_hash"),
            accepted_roadmap_hash=data.get("accepted_roadmap_hash"),
            validated_stage_spec_hashes={
                str(stage_id): str(stage_hash)
                for stage_id, stage_hash in (
                    data.get("validated_stage_spec_hashes") or {}
                ).items()
            },
            roadmap_review_required=bool(
                data.get("roadmap_review_required", False)
            ),
            future_stage_validation_required=bool(
                data.get("future_stage_validation_required", False)
            ),
            stages=stages,
            extras=_extract_extras(data, known),
        )


@dataclass(slots=True)
class ValidationResult:
    ok: bool
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def require_ok(self) -> None:
        if not self.ok:
            raise SgarError("; ".join(self.issues) or "validation failed")


@dataclass(slots=True)
class DoctorResult:
    ok: bool
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


__all__ = [
    "CriterionResult",
    "DoctorResult",
    "ExitCriterion",
    "ProjectMode",
    "ProjectState",
    "SgarError",
    "StageRecord",
    "StageStatus",
    "ValidationResult",
    "VerificationReport",
]
