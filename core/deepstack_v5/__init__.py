"""DeepStack v5 — long-running agent framework.

Inherits the simplicity of v3 and the engineering improvements of v4
(SQLite + worker lease/heartbeat + DAG state machine), without v4's
context_assets / cognition / multi-EventBus over-engineering.

Public API surface intentionally small:
    RuntimeV5 — entry point / dependency wiring
    EngineV5  — main loop (run / resume / step)
    ToolSpec  — register a callable as a capability
    NodeSpec  — declarative node description
    Verdict   — final outcome of a run
    Budget    — token / cost / wallclock limits
"""

from .config import ConfigV5
from .engine import EngineV5
from .runtime import RuntimeV5
from .types import (
    Budget,
    Capability,
    Decision,
    DecisionKind,
    Failure,
    FailureKind,
    Lease,
    NodeSpec,
    NodeState,
    RunStatus,
    Scope,
    ScopeLevel,
    SpawnResult,
    StepResult,
    TERMINAL_NODE_STATES,
    TERMINAL_RUN_STATUSES,
    ToolCallState,
    ToolSpec,
    Verdict,
    new_id,
)

__all__ = [
    "Budget",
    "Capability",
    "ConfigV5",
    "Decision",
    "DecisionKind",
    "EngineV5",
    "Failure",
    "FailureKind",
    "Lease",
    "NodeSpec",
    "NodeState",
    "RunStatus",
    "RuntimeV5",
    "Scope",
    "ScopeLevel",
    "SpawnResult",
    "StepResult",
    "TERMINAL_NODE_STATES",
    "TERMINAL_RUN_STATUSES",
    "ToolCallState",
    "ToolSpec",
    "Verdict",
    "new_id",
]
