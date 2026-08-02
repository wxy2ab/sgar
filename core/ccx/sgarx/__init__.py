"""SGAR Extended (sgarx) — incubation parallel of sgar.

sgarx is a fresh namespace that lets us prototype new long-horizon
governance features without disturbing the stable sgar runtime. Stage A
of sgarx is *behaviorally equivalent* to sgar: same state machine, same
operations, same trace shape — but data lives under ``.sgarx/`` so the
two never share a workspace footprint.

Stage-internal candidate frontier types live under
:mod:`core.ccx.sgarx.candidates` (data model / store / policy only;
runtime ops are Stage 2+).
"""

from .candidates import (
    AuditRecord,
    CandidateRecord,
    CandidateStatus,
    CandidateStore,
    FrontierState,
    assert_audit_binding,
    assert_budget,
    assert_promote_legal,
    can_transition,
)
from .runtime import SgarxRuntime
from .store import SGARX_DIR, SgarxStore

__all__ = [
    "SGARX_DIR",
    "AuditRecord",
    "CandidateRecord",
    "CandidateStatus",
    "CandidateStore",
    "FrontierState",
    "SgarxRuntime",
    "SgarxStore",
    "assert_audit_binding",
    "assert_budget",
    "assert_promote_legal",
    "can_transition",
]
