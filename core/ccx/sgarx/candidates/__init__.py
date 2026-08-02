"""Stage-internal Multi-Candidate Frontier for sgarx.

中文：树只存在于单个 stage 内部（propose → audit → patch/select → promote），
不改变 ProjectMode / StageStatus 线性 FSM，也不引入一等 Step 状态机。

English: The tree exists only inside a single stage frontier; ProjectMode
and StageStatus remain a linear FSM. No first-class Step state machine.
"""

from __future__ import annotations

from .binding import format_candidate_bound_detail
from .fingerprint import (
    compute_candidate_fingerprint,
    compute_candidate_fingerprint_with_git,
)
from .models import (
    DEFAULT_MAX_AUDITS,
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_MAX_PATCHES,
    POLICY_AUDIT_THEN_PROMOTE_V1,
    SCHEMA_VERSION,
    AuditRecord,
    CandidateRecord,
    CandidateStatus,
    FrontierState,
)
from .ops import CandidateOps
from .policy import (
    ACTION_AUDIT,
    ACTION_DISCARD,
    ACTION_PATCH,
    ACTION_PROMOTE,
    ACTION_PROPOSE,
    assert_action_legal,
    assert_audit_binding,
    assert_budget,
    assert_promote_legal,
    can_transition,
)
from .store import CandidateStore, normalize_artifact_paths, validate_candidate_id

__all__ = [
    "ACTION_AUDIT",
    "ACTION_DISCARD",
    "ACTION_PATCH",
    "ACTION_PROMOTE",
    "ACTION_PROPOSE",
    "AuditRecord",
    "CandidateOps",
    "CandidateRecord",
    "CandidateStatus",
    "CandidateStore",
    "DEFAULT_MAX_AUDITS",
    "DEFAULT_MAX_CANDIDATES",
    "DEFAULT_MAX_PATCHES",
    "FrontierState",
    "POLICY_AUDIT_THEN_PROMOTE_V1",
    "SCHEMA_VERSION",
    "assert_action_legal",
    "assert_audit_binding",
    "assert_budget",
    "assert_promote_legal",
    "can_transition",
    "compute_candidate_fingerprint",
    "compute_candidate_fingerprint_with_git",
    "format_candidate_bound_detail",
    "normalize_artifact_paths",
    "validate_candidate_id",
]
