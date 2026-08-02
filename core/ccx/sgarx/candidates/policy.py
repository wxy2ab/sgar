"""Pure policy checks for ``audit_then_promote_v1`` (no runtime I/O).

Condition edges require audit/check binding; LLM-only branching is rejected
by callers that invoke these asserts before mutating frontier state.

Default path is propose → audit (preferably ``--from-checks``) → patch →
promote. Full rewrite is non-default.
"""

from __future__ import annotations

from ...sgar.models import SgarError
from .models import (
    AuditRecord,
    CandidateRecord,
    CandidateStatus,
    FrontierState,
)

# Actions used by Stage 1 policy table (runtime ops land in Stage 2).
ACTION_PROPOSE = "propose"
ACTION_AUDIT = "audit"
ACTION_PROMOTE = "promote"
ACTION_PATCH = "patch"
ACTION_DISCARD = "discard"

_AUDITABLE = frozenset({
    CandidateStatus.PROPOSED.value,
    CandidateStatus.AUDITING.value,
})
_PATCHABLE = frozenset({
    CandidateStatus.AUDITED_PASS.value,
    CandidateStatus.AUDITED_FAIL.value,
})


def _status_value(status: str | CandidateStatus) -> str:
    if isinstance(status, CandidateStatus):
        return status.value
    return str(status or "").strip()


def can_transition(status: str | CandidateStatus, action: str) -> bool:
    """Return whether ``action`` is legal from ``status`` under v1 policy."""
    status_v = _status_value(status)
    action_v = str(action or "").strip()
    if action_v == ACTION_PROPOSE:
        # Propose registers a new node (typically starts as proposed); not a
        # transition of an existing candidate.
        return True
    if action_v == ACTION_AUDIT:
        return status_v in _AUDITABLE
    if action_v == ACTION_PROMOTE:
        return status_v == CandidateStatus.AUDITED_PASS.value
    if action_v == ACTION_PATCH:
        return status_v in _PATCHABLE
    if action_v == ACTION_DISCARD:
        return status_v != CandidateStatus.PROMOTED.value
    return False


def assert_budget(
    frontier: FrontierState,
    action: str,
    *,
    candidate_count: int | None = None,
) -> None:
    """Raise :class:`SgarError` when frontier budgets are exhausted."""
    action_v = str(action or "").strip()
    if action_v == ACTION_PROPOSE:
        count = (
            candidate_count
            if candidate_count is not None
            else len(frontier.active_candidate_ids)
        )
        if count >= frontier.max_candidates:
            raise SgarError(
                f"candidate budget exhausted: {count} >= "
                f"max_candidates={frontier.max_candidates}"
            )
        return
    if action_v == ACTION_AUDIT:
        if frontier.audit_count >= frontier.max_audits:
            raise SgarError(
                f"audit budget exhausted: {frontier.audit_count} >= "
                f"max_audits={frontier.max_audits}"
            )
        return
    if action_v == ACTION_PATCH:
        if frontier.patch_count >= frontier.max_patches:
            raise SgarError(
                f"patch budget exhausted: {frontier.patch_count} >= "
                f"max_patches={frontier.max_patches}"
            )
        return
    # promote / discard do not consume propose/audit/patch budgets


def assert_promote_legal(
    frontier: FrontierState,
    candidate: CandidateRecord,
) -> None:
    """Promote only from ``audited_pass`` when frontier has no promoted yet."""
    if frontier.promoted_candidate_id:
        raise SgarError(
            "frontier already has promoted candidate "
            f"{frontier.promoted_candidate_id!r}; second promote forbidden "
            "(v1 keeps other candidates but does not auto-discard)"
        )
    if not can_transition(candidate.status, ACTION_PROMOTE):
        raise SgarError(
            f"cannot promote candidate {candidate.candidate_id!r} "
            f"in status {candidate.status!r}; require audited_pass"
        )


def assert_audit_binding(
    candidate: CandidateRecord,
    audit: AuditRecord,
) -> None:
    """Reject audits whose bound hash does not match the candidate snapshot."""
    bound = str(audit.bound_candidate_hash or "").strip()
    current = str(candidate.candidate_hash or "").strip()
    if not bound or not current or bound != current:
        raise SgarError(
            "audit binding rejected: bound_candidate_hash "
            f"{bound!r} != candidate_hash {current!r}"
        )


def assert_action_legal(
    candidate: CandidateRecord,
    action: str,
) -> None:
    """Raise if ``action`` is illegal for ``candidate.status``."""
    if not can_transition(candidate.status, action):
        raise SgarError(
            f"illegal action {action!r} for candidate "
            f"{candidate.candidate_id!r} in status {candidate.status!r}"
        )


__all__ = [
    "ACTION_AUDIT",
    "ACTION_DISCARD",
    "ACTION_PATCH",
    "ACTION_PROMOTE",
    "ACTION_PROPOSE",
    "assert_action_legal",
    "assert_audit_binding",
    "assert_budget",
    "assert_promote_legal",
    "can_transition",
]
