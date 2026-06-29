"""Data models for the Stage-Governed Agent Runtime.

SGAR keeps its durable state intentionally small: human-facing Markdown
documents carry governance intent, while JSON stores only the machine state
needed to validate hard transitions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExitCriterion":
        check = data.get("check")
        return cls(
            criterion_id=str(data.get("criterion_id") or ""),
            description=str(data.get("description") or ""),
            blocking=bool(data.get("blocking", True)),
            check=str(check) if check else None,
        )


@dataclass(slots=True)
class CriterionResult:
    criterion_id: str
    passed: bool
    evidence: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CriterionResult":
        return cls(
            criterion_id=str(data.get("criterion_id") or ""),
            passed=bool(data.get("passed", False)),
            evidence=str(data.get("evidence") or ""),
        )


@dataclass(slots=True)
class VerificationReport:
    stage_id: str
    results: list[CriterionResult] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "results": [asdict(result) for result in self.results],
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VerificationReport":
        return cls(
            stage_id=str(data.get("stage_id") or ""),
            results=[
                CriterionResult.from_dict(item)
                for item in data.get("results", [])
                if isinstance(item, dict)
            ],
            notes=str(data.get("notes") or ""),
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StageRecord":
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

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["stages"] = {
            stage_id: record.to_dict()
            for stage_id, record in self.stages.items()
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectState":
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
