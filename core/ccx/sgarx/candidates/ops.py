"""Runtime orchestration for stage-internal candidate frontier ops.

Used by :class:`core.ccx.sgarx.runtime.SgarxRuntime`. Policy checks stay in
:mod:`.policy`; disk I/O stays in :mod:`.store`.

中文：默认 **patch** 派生；full rewrite 非默认。条件边须绑 checker/audit +
``candidate_hash``（接口 C），禁止仅凭 LLM 自觉分支或从零构造（接口 B）。

English: Default to patch derivation; full rewrite is non-default. Condition
edges require checker/audit + ``candidate_hash`` (interface C) — never
LLM-only branching or from-scratch construction as the default (B).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...sgar.missions import MISSION_STATUS_COMPLETED, load_mission
from ...sgar.models import SgarError
from ...sgar.store import utc_now
from ..store import SgarxStore
from .fingerprint import (
    compute_candidate_fingerprint,
    compute_candidate_fingerprint_with_git,
)
from .binding import format_candidate_bound_detail
from .models import (
    POLICY_AUDIT_THEN_PROMOTE_V1,
    AuditRecord,
    CandidateRecord,
    CandidateStatus,
    FrontierState,
)
from .policy import (
    ACTION_AUDIT,
    ACTION_DISCARD,
    ACTION_PATCH,
    ACTION_PROPOSE,
    assert_action_legal,
    assert_audit_binding,
    assert_budget,
    assert_promote_legal,
)
from .store import CandidateStore, normalize_artifact_paths, validate_candidate_id


class CandidateOps:
    """Frontier lifecycle helpers bound to a :class:`SgarxStore`."""

    def __init__(self, store: SgarxStore) -> None:
        self.store = store
        self.candidates = CandidateStore(store)

    def ensure_frontier(self, stage_id: str) -> FrontierState:
        return self.candidates.ensure_frontier(stage_id)

    def list_frontier(self, stage_id: str) -> FrontierState:
        return self.candidates.load_frontier(stage_id)

    def get_candidate(self, stage_id: str, candidate_id: str) -> CandidateRecord:
        return self.candidates.load_candidate(stage_id, candidate_id)

    def list_candidates(self, stage_id: str) -> list[CandidateRecord]:
        return self.candidates.list_candidates(stage_id)

    def require_current_stage(self, stage_id: str) -> None:
        state = self.store.load_state()
        if state.current_stage_id != stage_id:
            raise SgarError(
                f"cannot mutate frontier for {stage_id}; "
                f"current stage is {state.current_stage_id!r}"
            )

    def propose_candidate(
        self,
        stage_id: str,
        *,
        candidate_id: str,
        summary: str,
        artifact_paths: list[str] | None = None,
        candidate_hash: str | None = None,
        git_head: str | None = None,
        origin: str = "propose",
        extras: dict[str, Any] | None = None,
        require_current: bool = True,
    ) -> CandidateRecord:
        if require_current:
            self.require_current_stage(stage_id)
        candidate_id = validate_candidate_id(candidate_id)
        frontier = self.candidates.ensure_frontier(stage_id)
        existing = self.candidates.list_candidates(stage_id)
        assert_budget(
            frontier,
            ACTION_PROPOSE,
            candidate_count=len(existing),
        )
        meta_path = self.candidates.candidate_meta_path(stage_id, candidate_id)
        if meta_path.is_file():
            raise SgarError(f"candidate already exists: {candidate_id!r}")

        paths = normalize_artifact_paths(artifact_paths)
        if not paths:
            raise SgarError(
                "propose-candidate requires --artifact / artifact_paths"
            )
        hash_value = str(candidate_hash or "").strip()
        head = git_head
        if not hash_value:
            hash_value, computed_head = compute_candidate_fingerprint_with_git(
                self.store.cwd, paths
            )
            if head is None:
                head = computed_head
        now = utc_now()
        record = CandidateRecord(
            candidate_id=candidate_id,
            parent_id=None,
            status=CandidateStatus.PROPOSED.value,
            created_at=now,
            updated_at=now,
            artifact_paths=paths,
            candidate_hash=hash_value,
            git_head=head,
            origin=str(origin or "propose"),
            summary=str(summary or ""),
            audit=None,
            score=None,
            extras=dict(extras or {}),
        )
        self.candidates.write_candidate(stage_id, record)
        if candidate_id not in frontier.active_candidate_ids:
            frontier.active_candidate_ids.append(candidate_id)
        frontier.updated_at = now
        self.candidates.write_frontier(frontier)
        return record

    def audit_candidate(
        self,
        stage_id: str,
        candidate_id: str,
        *,
        passed: bool = False,
        findings: list[str] | None = None,
        method: str = "criterion_checks",
        evidence_paths: list[str] | None = None,
        raw_ref: str | None = None,
        bound_candidate_hash: str | None = None,
        audit: AuditRecord | None = None,
        require_current: bool = True,
    ) -> CandidateRecord:
        if require_current:
            self.require_current_stage(stage_id)
        frontier = self.candidates.load_frontier(stage_id)
        record = self.candidates.load_candidate(stage_id, candidate_id)
        assert_budget(frontier, ACTION_AUDIT)
        assert_action_legal(record, ACTION_AUDIT)

        if audit is None:
            bound = (
                str(bound_candidate_hash).strip()
                if bound_candidate_hash is not None
                else record.candidate_hash
            )
            audit = AuditRecord(
                audited_at=utc_now(),
                bound_candidate_hash=bound,
                passed=bool(passed),
                findings=list(findings or []),
                method=str(method or "criterion_checks"),
                evidence_paths=normalize_artifact_paths(evidence_paths),
                raw_ref=raw_ref,
            )
        # Binding check BEFORE any write / budget consume.
        assert_audit_binding(record, audit)

        record.audit = audit
        record.status = (
            CandidateStatus.AUDITED_PASS.value
            if audit.passed
            else CandidateStatus.AUDITED_FAIL.value
        )
        record.updated_at = utc_now()
        self.candidates.write_candidate(stage_id, record)
        frontier.audit_count += 1
        frontier.updated_at = record.updated_at
        self.candidates.write_frontier(frontier)
        return record

    def audit_candidate_from_checks(
        self,
        stage_id: str,
        candidate_id: str,
        *,
        timeout_s: float = 120.0,
        changed_paths: list[str] | None = None,
        require_current: bool = True,
    ) -> CandidateRecord:
        """Run stage ``[check:]`` criteria and write a bound AuditRecord."""
        if require_current:
            self.require_current_stage(stage_id)
        frontier = self.candidates.load_frontier(stage_id)
        record = self.candidates.load_candidate(stage_id, candidate_id)
        assert_budget(frontier, ACTION_AUDIT)
        assert_action_legal(record, ACTION_AUDIT)

        from ...agents.incremental_verify import select_criteria_for_changes
        from ...sgar.checks import run_criterion_check
        from ...sgar.validation import parse_exit_criteria

        if not record.artifact_paths:
            raise SgarError(
                "audit --from-checks requires non-empty artifact_paths "
                "on the candidate"
            )
        live = compute_candidate_fingerprint(
            self.store.cwd, record.artifact_paths
        )
        if live != record.candidate_hash:
            raise SgarError(
                f"candidate hash drift before audit: "
                f"recorded={record.candidate_hash!r} current={live!r}"
            )

        spec_text = self.store.read_text(self.store.stage_spec_path(stage_id))
        criteria = parse_exit_criteria(spec_text)
        checked = [c for c in criteria if c.check]
        if not checked:
            raise SgarError(
                "audit --from-checks requires exit criteria with [check:]"
            )

        paths = (
            list(changed_paths)
            if changed_paths is not None
            else list(record.artifact_paths)
        )
        if not paths:
            raise SgarError(
                "audit --from-checks requires --artifact / artifact_paths"
            )
        # Scope is used only for incremental selection; non-intersecting scoped
        # criteria are still covered by the confirmation pass below. Do not
        # hard-abort — autobuild often registers a minimal artifact set.
        to_run = select_criteria_for_changes(checked, paths)
        outcomes = []
        for criterion in to_run:
            outcomes.append(
                run_criterion_check(
                    criterion,
                    cwd=self.store.cwd,
                    timeout_s=float(timeout_s),
                )
            )
        # Confirm full checked set when incremental subset was used.
        if len(to_run) < len(checked):
            seen = {o.criterion_id for o in outcomes}
            for criterion in checked:
                if criterion.criterion_id in seen:
                    continue
                outcomes.append(
                    run_criterion_check(
                        criterion,
                        cwd=self.store.cwd,
                        timeout_s=float(timeout_s),
                    )
                )

        by_id = {c.criterion_id: c for c in checked}
        findings: list[str] = []
        blocking_failed = False
        for outcome in outcomes:
            criterion = by_id.get(outcome.criterion_id)
            line = f"[{outcome.criterion_id}] {outcome.evidence_line()}"
            if outcome.passed:
                findings.append(f"PASS {line}")
                continue
            findings.append(f"FAIL {line}")
            if criterion is None or criterion.blocking:
                blocking_failed = True

        passed = not blocking_failed
        status = (
            CandidateStatus.AUDITED_PASS.value
            if passed
            else CandidateStatus.AUDITED_FAIL.value
        )
        # Temporarily set status for formatter; persist after binding.
        preview = CandidateRecord(
            candidate_id=record.candidate_id,
            parent_id=record.parent_id,
            status=status,
            created_at=record.created_at,
            updated_at=record.updated_at,
            artifact_paths=list(record.artifact_paths),
            candidate_hash=record.candidate_hash,
            git_head=record.git_head,
            origin=record.origin,
            summary=record.summary,
            audit=None,
            score=record.score,
            extras=dict(record.extras),
        )
        bound_detail = format_candidate_bound_detail(
            preview,
            findings=findings,
            include_patch_hint=not passed,
        )
        audit = AuditRecord(
            audited_at=utc_now(),
            bound_candidate_hash=record.candidate_hash,
            passed=passed,
            findings=findings,
            method="criterion_checks",
            evidence_paths=[],
            raw_ref=None,
            extras={"bound_detail": bound_detail},
        )
        assert_audit_binding(record, audit)

        record.audit = audit
        record.status = status
        record.updated_at = utc_now()
        self.candidates.write_candidate(stage_id, record)
        frontier.audit_count += 1
        frontier.updated_at = record.updated_at
        self.candidates.write_frontier(frontier)
        return record

    def promote_candidate(
        self,
        stage_id: str,
        candidate_id: str,
        *,
        require_current: bool = True,
    ) -> CandidateRecord:
        if require_current:
            self.require_current_stage(stage_id)
        frontier = self.candidates.load_frontier(stage_id)
        record = self.candidates.load_candidate(stage_id, candidate_id)
        assert_promote_legal(frontier, record)
        live = compute_candidate_fingerprint(
            self.store.cwd, record.artifact_paths
        )
        if live != record.candidate_hash:
            raise SgarError(
                f"candidate hash drift before promote: "
                f"recorded={record.candidate_hash!r} current={live!r}"
            )
        if record.audit is not None:
            bound = str(record.audit.bound_candidate_hash or "").strip()
            if bound and bound != live:
                raise SgarError(
                    f"candidate hash drift before promote: "
                    f"audit bound={bound!r} current={live!r}"
                )
        record.status = CandidateStatus.PROMOTED.value
        record.updated_at = utc_now()
        self.candidates.write_candidate(stage_id, record)
        frontier.promoted_candidate_id = candidate_id
        if candidate_id not in frontier.active_candidate_ids:
            frontier.active_candidate_ids.append(candidate_id)
        frontier.updated_at = record.updated_at
        self.candidates.write_frontier(frontier)
        return record

    def patch_candidate(
        self,
        stage_id: str,
        *,
        parent_id: str,
        candidate_id: str,
        summary: str,
        artifact_paths: list[str] | None = None,
        candidate_hash: str | None = None,
        git_head: str | None = None,
        extras: dict[str, Any] | None = None,
        require_current: bool = True,
    ) -> CandidateRecord:
        if require_current:
            self.require_current_stage(stage_id)
        parent_id = validate_candidate_id(parent_id)
        candidate_id = validate_candidate_id(candidate_id)
        frontier = self.candidates.load_frontier(stage_id)
        parent = self.candidates.load_candidate(stage_id, parent_id)
        assert_action_legal(parent, ACTION_PATCH)
        existing = self.candidates.list_candidates(stage_id)
        assert_budget(
            frontier,
            ACTION_PROPOSE,
            candidate_count=len(existing),
        )
        assert_budget(frontier, ACTION_PATCH)
        meta_path = self.candidates.candidate_meta_path(stage_id, candidate_id)
        if meta_path.is_file():
            raise SgarError(f"candidate already exists: {candidate_id!r}")

        paths = normalize_artifact_paths(artifact_paths)
        if not paths:
            raise SgarError(
                "patch-candidate requires --artifact / artifact_paths"
            )
        hash_value = str(candidate_hash or "").strip()
        head = git_head
        if not hash_value:
            hash_value, computed_head = compute_candidate_fingerprint_with_git(
                self.store.cwd, paths
            )
            if head is None:
                head = computed_head
        now = utc_now()
        child = CandidateRecord(
            candidate_id=candidate_id,
            parent_id=parent_id,
            status=CandidateStatus.PROPOSED.value,
            created_at=now,
            updated_at=now,
            artifact_paths=paths,
            candidate_hash=hash_value,
            git_head=head,
            origin="patch",
            summary=str(summary or ""),
            audit=None,
            score=None,
            extras=dict(extras or {}),
        )
        self.candidates.write_candidate(stage_id, child)
        frontier.patch_count += 1
        if candidate_id not in frontier.active_candidate_ids:
            frontier.active_candidate_ids.append(candidate_id)
        frontier.updated_at = now
        self.candidates.write_frontier(frontier)
        return child

    def discard_candidate(
        self,
        stage_id: str,
        candidate_id: str,
        *,
        require_current: bool = True,
    ) -> CandidateRecord:
        if require_current:
            self.require_current_stage(stage_id)
        frontier = self.candidates.load_frontier(stage_id)
        record = self.candidates.load_candidate(stage_id, candidate_id)
        assert_action_legal(record, ACTION_DISCARD)
        record.status = CandidateStatus.DISCARDED.value
        record.updated_at = utc_now()
        self.candidates.write_candidate(stage_id, record)
        if candidate_id in frontier.active_candidate_ids:
            frontier.active_candidate_ids = [
                cid for cid in frontier.active_candidate_ids if cid != candidate_id
            ]
        frontier.updated_at = record.updated_at
        self.candidates.write_frontier(frontier)
        return record

    def force_discard_candidate(
        self,
        stage_id: str,
        candidate_id: str,
        *,
        require_current: bool = True,
    ) -> CandidateRecord:
        """Discard even if promoted (autobuild recovery after hash drift).

        Clears ``frontier.promoted_candidate_id`` when it points at this node so
        a fresh propose/promote can proceed. Not a demote API for agents.
        """
        if require_current:
            self.require_current_stage(stage_id)
        frontier = self.candidates.load_frontier(stage_id)
        record = self.candidates.load_candidate(stage_id, candidate_id)
        record.status = CandidateStatus.DISCARDED.value
        record.updated_at = utc_now()
        self.candidates.write_candidate(stage_id, record)
        if frontier.promoted_candidate_id == candidate_id:
            frontier.promoted_candidate_id = None
        if candidate_id in frontier.active_candidate_ids:
            frontier.active_candidate_ids = [
                cid for cid in frontier.active_candidate_ids if cid != candidate_id
            ]
        frontier.updated_at = record.updated_at
        self.candidates.write_frontier(frontier)
        return record

    def import_candidate_from_mission(
        self,
        stage_id: str,
        *,
        mission_id: str,
        candidate_id: str,
        summary: str | None = None,
        artifact_paths: list[str] | None = None,
        require_current: bool = True,
    ) -> CandidateRecord:
        manifest = load_mission(self.store, mission_id)
        if manifest.get("status") != MISSION_STATUS_COMPLETED:
            raise SgarError(
                f"mission {mission_id!r} is not completed "
                f"(status={manifest.get('status')!r})"
            )
        paths = artifact_paths
        if paths is None:
            paths = []
            for output in manifest.get("recorded_outputs") or []:
                if not isinstance(output, dict):
                    continue
                source = output.get("source_path") or output.get("path")
                if source:
                    paths.append(str(source))
            if not paths:
                for item in manifest.get("expected_outputs") or []:
                    paths.append(str(item))
        text = summary
        if not text:
            text = str(manifest.get("objective") or f"mission {mission_id}")
        return self.propose_candidate(
            stage_id,
            candidate_id=candidate_id,
            summary=text,
            artifact_paths=paths,
            origin="mission_promote",
            extras={"mission_id": mission_id},
            require_current=require_current,
        )

    def assert_close_frontier_gate(self, stage_id: str) -> None:
        """Raise if frontier policy blocks close_stage; no-op when absent."""
        path = self.candidates.frontier_path(stage_id)
        if not path.is_file():
            return  # grandfather
        frontier = self.candidates.load_frontier(stage_id)
        if frontier.policy != POLICY_AUDIT_THEN_PROMOTE_V1:
            raise SgarError(
                f"stage cannot close: unknown frontier policy {frontier.policy!r}"
            )
        if not frontier.promoted_candidate_id:
            raise SgarError("stage cannot close without a promoted candidate")
        promoted = self.candidates.load_candidate(
            stage_id, frontier.promoted_candidate_id
        )
        current = compute_candidate_fingerprint(
            self.store.cwd, promoted.artifact_paths
        )
        if current != promoted.candidate_hash:
            raise SgarError(
                f"promoted candidate hash drift: recorded={promoted.candidate_hash!r} "
                f"current={current!r}"
            )


__all__ = ["CandidateOps"]
