"""Disk layout helpers for stage-internal candidate frontiers under ``.sgarx/``.

Layout (authoritative)::

    .sgarx/stages/<stage_id>/
      frontier.json
      candidates/
        <candidate_id>/
          meta.json
          audit.json     # latest audit, optional
          NOTES.md       # optional human notes

Artifacts themselves stay in workspace business paths referenced by
``artifact_paths``; Stage 1 does not copy whole trees into ``candidates/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...sgar.models import SgarError
from ...sgar.store import STAGE_ID_RE, _ensure_inside, utc_now, validate_stage_id
from ..store import SgarxStore
from .models import (
    DEFAULT_MAX_AUDITS,
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_MAX_PATCHES,
    POLICY_AUDIT_THEN_PROMOTE_V1,
    SCHEMA_VERSION,
    AuditRecord,
    CandidateRecord,
    FrontierState,
)


def validate_candidate_id(candidate_id: str) -> str:
    candidate_id = str(candidate_id or "").strip()
    if not STAGE_ID_RE.match(candidate_id):
        raise SgarError(
            "candidate id must be 1-128 chars of letters, digits, dot, dash, "
            "or underscore, and start with a letter or digit"
        )
    if ".." in candidate_id:
        raise SgarError(f"candidate id escapes path segments: {candidate_id!r}")
    return candidate_id


def normalize_artifact_paths(paths: list[str] | tuple[str, ...] | None) -> list[str]:
    """Normalize workspace-relative artifact paths; reject escapes."""
    result: list[str] = []
    for raw in paths or ():
        text = str(raw).strip().replace("\\", "/")
        if not text:
            continue
        if text.startswith("/") or (len(text) >= 2 and text[1] == ":"):
            raise SgarError(f"artifact path must be workspace-relative: {text!r}")
        parts = [p for p in text.split("/") if p not in ("", ".")]
        if any(p == ".." for p in parts):
            raise SgarError(f"artifact path escapes workspace: {text!r}")
        result.append("/".join(parts))
    return result


class CandidateStore:
    """Path helpers + load/save for frontier and candidate records."""

    def __init__(self, store: SgarxStore) -> None:
        self.store = store

    def frontier_path(self, stage_id: str) -> Path:
        return self.store.stage_dir(stage_id) / "frontier.json"

    def candidates_root(self, stage_id: str) -> Path:
        return self.store.stage_dir(stage_id) / "candidates"

    def candidate_dir(self, stage_id: str, candidate_id: str) -> Path:
        stage_id = validate_stage_id(stage_id)
        candidate_id = validate_candidate_id(candidate_id)
        root = self.candidates_root(stage_id).resolve()
        path = (root / candidate_id).resolve()
        _ensure_inside(path, root, "candidate directory")
        return path

    def candidate_meta_path(self, stage_id: str, candidate_id: str) -> Path:
        return self.candidate_dir(stage_id, candidate_id) / "meta.json"

    def candidate_audit_path(self, stage_id: str, candidate_id: str) -> Path:
        return self.candidate_dir(stage_id, candidate_id) / "audit.json"

    def load_frontier(self, stage_id: str) -> FrontierState:
        path = self.frontier_path(stage_id)
        data = self.store.read_json(path)
        frontier = FrontierState.from_dict(data)
        self._require_schema_version(frontier.schema_version, path)
        if frontier.stage_id and frontier.stage_id != stage_id:
            raise SgarError(
                f"frontier stage_id mismatch: file has {frontier.stage_id!r}, "
                f"expected {stage_id!r}"
            )
        frontier.stage_id = stage_id
        return frontier

    def write_frontier(self, frontier: FrontierState) -> None:
        self._require_schema_version(frontier.schema_version, "frontier")
        validate_stage_id(frontier.stage_id)
        path = self.frontier_path(frontier.stage_id)
        self.store.write_json(path, frontier.to_dict())

    def ensure_frontier(
        self,
        stage_id: str,
        *,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        max_audits: int = DEFAULT_MAX_AUDITS,
        max_patches: int = DEFAULT_MAX_PATCHES,
        policy: str = POLICY_AUDIT_THEN_PROMOTE_V1,
        defaults: dict[str, Any] | None = None,
    ) -> FrontierState:
        """Return existing frontier or create an empty one with default budgets."""
        stage_id = validate_stage_id(stage_id)
        path = self.frontier_path(stage_id)
        if path.is_file():
            return self.load_frontier(stage_id)
        overrides = dict(defaults or {})
        frontier = FrontierState(
            stage_id=stage_id,
            schema_version=SCHEMA_VERSION,
            policy=str(overrides.get("policy") or policy),
            max_candidates=int(
                overrides.get("max_candidates", max_candidates)
            ),
            max_audits=int(overrides.get("max_audits", max_audits)),
            max_patches=int(overrides.get("max_patches", max_patches)),
            audit_count=0,
            patch_count=0,
            active_candidate_ids=[],
            promoted_candidate_id=None,
            updated_at=utc_now(),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        self.write_frontier(frontier)
        return frontier

    def load_candidate(self, stage_id: str, candidate_id: str) -> CandidateRecord:
        meta_path = self.candidate_meta_path(stage_id, candidate_id)
        data = self.store.read_json(meta_path)
        record = CandidateRecord.from_dict(data)
        record.candidate_id = validate_candidate_id(
            record.candidate_id or candidate_id
        )
        if record.candidate_id != candidate_id:
            raise SgarError(
                f"candidate_id mismatch: meta has {record.candidate_id!r}, "
                f"expected {candidate_id!r}"
            )
        record.artifact_paths = normalize_artifact_paths(record.artifact_paths)
        audit_path = self.candidate_audit_path(stage_id, candidate_id)
        if audit_path.is_file():
            audit_data = self.store.read_json(audit_path)
            record.audit = AuditRecord.from_dict(audit_data)
        return record

    def write_candidate(self, stage_id: str, record: CandidateRecord) -> None:
        validate_stage_id(stage_id)
        record.candidate_id = validate_candidate_id(record.candidate_id)
        record.artifact_paths = normalize_artifact_paths(record.artifact_paths)
        meta_path = self.candidate_meta_path(stage_id, record.candidate_id)
        self.store.write_json(meta_path, record.to_meta_dict())
        if record.audit is not None:
            audit_path = self.candidate_audit_path(stage_id, record.candidate_id)
            self.store.write_json(audit_path, record.audit.to_dict())

    def list_candidates(self, stage_id: str) -> list[CandidateRecord]:
        root = self.candidates_root(stage_id)
        if not root.is_dir():
            return []
        records: list[CandidateRecord] = []
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            meta = child / "meta.json"
            if not meta.is_file():
                continue
            try:
                cid = validate_candidate_id(child.name)
            except SgarError:
                continue
            records.append(self.load_candidate(stage_id, cid))
        return records

    @staticmethod
    def _require_schema_version(version: int, where: Any) -> None:
        if int(version) != SCHEMA_VERSION:
            raise SgarError(
                f"unsupported frontier schema_version {version} at {where}; "
                f"expected {SCHEMA_VERSION}"
            )


__all__ = [
    "CandidateStore",
    "normalize_artifact_paths",
    "validate_candidate_id",
]
