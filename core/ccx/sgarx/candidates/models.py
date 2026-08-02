"""Stage-internal candidate frontier data models (sgarx only).

中文：候选树只存在于**单个 stage 内部**的 Multi-Candidate Frontier，
不是 ProjectMode / roadmap 搜索树，也不是一等 Step 状态机。
本模块仅定义可序列化契约；propose/audit/promote 等 runtime ops 属 Stage 2。

English: The candidate tree lives only inside a single stage's frontier —
not as a ProjectMode/roadmap search tree, and not as a first-class Step FSM.
This module freezes the JSON-serializable contract; runtime ops land in Stage 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any


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


class CandidateStatus(str, Enum):
    """Lifecycle of one stage-internal candidate."""

    PROPOSED = "proposed"
    AUDITING = "auditing"  # optional intermediate; core path may skip
    AUDITED_PASS = "audited_pass"
    AUDITED_FAIL = "audited_fail"
    PROMOTED = "promoted"
    DISCARDED = "discarded"
    SUPERSEDED = "superseded"  # optional; parent replaced by patch child


POLICY_AUDIT_THEN_PROMOTE_V1 = "audit_then_promote_v1"
SCHEMA_VERSION = 1

DEFAULT_MAX_CANDIDATES = 8
DEFAULT_MAX_AUDITS = 16
DEFAULT_MAX_PATCHES = 12


@dataclass(slots=True)
class AuditRecord:
    audited_at: str
    bound_candidate_hash: str
    passed: bool
    findings: list[str] = field(default_factory=list)
    method: str = "criterion_checks"
    evidence_paths: list[str] = field(default_factory=list)
    raw_ref: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _merge_extras(
            {
                "audited_at": self.audited_at,
                "bound_candidate_hash": self.bound_candidate_hash,
                "passed": self.passed,
                "findings": list(self.findings),
                "method": self.method,
                "evidence_paths": list(self.evidence_paths),
                "raw_ref": self.raw_ref,
            },
            self.extras,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuditRecord":
        known = _known_field_names(cls)
        findings_raw = data.get("findings") or []
        if isinstance(findings_raw, str):
            findings = [findings_raw] if findings_raw.strip() else []
        elif isinstance(findings_raw, (list, tuple)):
            findings = [str(item) for item in findings_raw]
        else:
            findings = []
        evidence_raw = data.get("evidence_paths") or []
        if isinstance(evidence_raw, str):
            evidence_paths = [evidence_raw] if evidence_raw.strip() else []
        elif isinstance(evidence_raw, (list, tuple)):
            evidence_paths = [str(item) for item in evidence_raw]
        else:
            evidence_paths = []
        raw_ref = data.get("raw_ref")
        return cls(
            audited_at=str(data.get("audited_at") or ""),
            bound_candidate_hash=str(data.get("bound_candidate_hash") or ""),
            passed=bool(data.get("passed", False)),
            findings=findings,
            method=str(data.get("method") or "criterion_checks"),
            evidence_paths=evidence_paths,
            raw_ref=str(raw_ref) if raw_ref is not None else None,
            extras=_extract_extras(data, known),
        )


@dataclass(slots=True)
class CandidateRecord:
    candidate_id: str
    parent_id: str | None = None
    status: str = CandidateStatus.PROPOSED.value
    created_at: str = ""
    updated_at: str = ""
    artifact_paths: list[str] = field(default_factory=list)
    candidate_hash: str = ""
    git_head: str | None = None
    origin: str = "propose"
    summary: str = ""
    audit: AuditRecord | None = None
    score: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_audit: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "candidate_id": self.candidate_id,
            "parent_id": self.parent_id,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "artifact_paths": list(self.artifact_paths),
            "candidate_hash": self.candidate_hash,
            "git_head": self.git_head,
            "origin": self.origin,
            "summary": self.summary,
            "score": self.score,
        }
        if include_audit and self.audit is not None:
            payload["audit"] = self.audit.to_dict()
        return _merge_extras(payload, self.extras)

    def to_meta_dict(self) -> dict[str, Any]:
        """Serialize for ``meta.json`` (audit lives in ``audit.json``)."""
        return self.to_dict(include_audit=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidateRecord":
        known = _known_field_names(cls)
        audit_raw = data.get("audit")
        audit: AuditRecord | None = None
        if isinstance(audit_raw, dict):
            audit = AuditRecord.from_dict(audit_raw)
        score_raw = data.get("score")
        score: float | None
        if score_raw is None or score_raw == "":
            score = None
        else:
            score = float(score_raw)
        paths_raw = data.get("artifact_paths") or []
        if isinstance(paths_raw, str):
            artifact_paths = [paths_raw] if paths_raw.strip() else []
        elif isinstance(paths_raw, (list, tuple)):
            artifact_paths = [str(p) for p in paths_raw]
        else:
            artifact_paths = []
        parent_id = data.get("parent_id")
        git_head = data.get("git_head")
        return cls(
            candidate_id=str(data.get("candidate_id") or ""),
            parent_id=str(parent_id) if parent_id is not None else None,
            status=str(data.get("status") or CandidateStatus.PROPOSED.value),
            created_at=str(data.get("created_at") or ""),
            updated_at=str(data.get("updated_at") or ""),
            artifact_paths=artifact_paths,
            candidate_hash=str(data.get("candidate_hash") or ""),
            git_head=str(git_head) if git_head is not None else None,
            origin=str(data.get("origin") or "propose"),
            summary=str(data.get("summary") or ""),
            audit=audit,
            score=score,
            extras=_extract_extras(data, known),
        )


@dataclass(slots=True)
class FrontierState:
    stage_id: str
    schema_version: int = SCHEMA_VERSION
    policy: str = POLICY_AUDIT_THEN_PROMOTE_V1
    max_candidates: int = DEFAULT_MAX_CANDIDATES
    max_audits: int = DEFAULT_MAX_AUDITS
    max_patches: int = DEFAULT_MAX_PATCHES
    audit_count: int = 0
    patch_count: int = 0
    active_candidate_ids: list[str] = field(default_factory=list)
    promoted_candidate_id: str | None = None
    updated_at: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _merge_extras(
            {
                "stage_id": self.stage_id,
                "schema_version": self.schema_version,
                "policy": self.policy,
                "max_candidates": self.max_candidates,
                "max_audits": self.max_audits,
                "max_patches": self.max_patches,
                "audit_count": self.audit_count,
                "patch_count": self.patch_count,
                "active_candidate_ids": list(self.active_candidate_ids),
                "promoted_candidate_id": self.promoted_candidate_id,
                "updated_at": self.updated_at,
            },
            self.extras,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FrontierState":
        known = _known_field_names(cls)
        promoted = data.get("promoted_candidate_id")
        active_raw = data.get("active_candidate_ids") or []
        if isinstance(active_raw, (list, tuple)):
            active = [str(item) for item in active_raw]
        else:
            active = []
        return cls(
            stage_id=str(data.get("stage_id") or ""),
            schema_version=int(data.get("schema_version") or SCHEMA_VERSION),
            policy=str(data.get("policy") or POLICY_AUDIT_THEN_PROMOTE_V1),
            max_candidates=int(
                data.get("max_candidates")
                if data.get("max_candidates") is not None
                else DEFAULT_MAX_CANDIDATES
            ),
            max_audits=int(
                data.get("max_audits")
                if data.get("max_audits") is not None
                else DEFAULT_MAX_AUDITS
            ),
            max_patches=int(
                data.get("max_patches")
                if data.get("max_patches") is not None
                else DEFAULT_MAX_PATCHES
            ),
            audit_count=int(data.get("audit_count") or 0),
            patch_count=int(data.get("patch_count") or 0),
            active_candidate_ids=active,
            promoted_candidate_id=(
                str(promoted) if promoted is not None else None
            ),
            updated_at=str(data.get("updated_at") or ""),
            extras=_extract_extras(data, known),
        )


__all__ = [
    "AuditRecord",
    "CandidateRecord",
    "CandidateStatus",
    "DEFAULT_MAX_AUDITS",
    "DEFAULT_MAX_CANDIDATES",
    "DEFAULT_MAX_PATCHES",
    "FrontierState",
    "POLICY_AUDIT_THEN_PROMOTE_V1",
    "SCHEMA_VERSION",
]
