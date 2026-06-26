"""Core data types for DeepStack v5.

All persistent values are JSON-serialisable. Datetime values are stored as
epoch milliseconds at the persistence boundary; in-memory objects use ints
for portability.
"""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence


def now_ms() -> int:
    return int(time.time() * 1000)


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------- #
# Node lifecycle (11 states)
# --------------------------------------------------------------------------- #

class NodeState(str, enum.Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"
    APPROVAL_HANG = "approval_hang"
    TIMER_HANG = "timer_hang"
    ABANDONED = "abandoned"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


TERMINAL_NODE_STATES: frozenset[NodeState] = frozenset({
    NodeState.SUCCEEDED,
    NodeState.ABANDONED,
    NodeState.SKIPPED,
    NodeState.CANCELLED,
})


# Legal node transitions. Validated by NodeExecution.transition().
LEGAL_NODE_TRANSITIONS: dict[NodeState, frozenset[NodeState]] = {
    NodeState.PENDING: frozenset({
        NodeState.READY, NodeState.SKIPPED, NodeState.CANCELLED,
    }),
    NodeState.READY: frozenset({
        NodeState.RUNNING, NodeState.BLOCKED, NodeState.CANCELLED,
        NodeState.SKIPPED,
    }),
    NodeState.RUNNING: frozenset({
        NodeState.SUCCEEDED, NodeState.FAILED, NodeState.BLOCKED,
        NodeState.APPROVAL_HANG, NodeState.TIMER_HANG, NodeState.CANCELLED,
    }),
    NodeState.FAILED: frozenset({
        NodeState.READY, NodeState.ABANDONED, NodeState.CANCELLED,
    }),
    NodeState.BLOCKED: frozenset({
        NodeState.READY, NodeState.CANCELLED, NodeState.ABANDONED,
        NodeState.SKIPPED,
    }),
    NodeState.APPROVAL_HANG: frozenset({
        NodeState.RUNNING, NodeState.ABANDONED, NodeState.CANCELLED,
    }),
    NodeState.TIMER_HANG: frozenset({
        NodeState.READY, NodeState.RUNNING, NodeState.CANCELLED,
        NodeState.ABANDONED,
    }),
    NodeState.SUCCEEDED: frozenset(),
    NodeState.ABANDONED: frozenset(),
    NodeState.SKIPPED: frozenset(),
    NodeState.CANCELLED: frozenset(),
}


def is_legal_node_transition(src: NodeState, dst: NodeState) -> bool:
    return dst in LEGAL_NODE_TRANSITIONS.get(src, frozenset())


# --------------------------------------------------------------------------- #
# Tool call lifecycle (8 states)
# --------------------------------------------------------------------------- #

class ToolCallState(str, enum.Enum):
    PENDING = "pending"
    APPROVAL_PENDING = "approval_pending"
    REJECTED = "rejected"
    RUNNING = "running"
    COMPLETED = "completed"
    UNKNOWN_EFFECT = "unknown_effect"
    RECONCILED = "reconciled"
    FAILED = "failed"


TERMINAL_TOOLCALL_STATES: frozenset[ToolCallState] = frozenset({
    ToolCallState.REJECTED,
    ToolCallState.COMPLETED,
    ToolCallState.RECONCILED,
    ToolCallState.FAILED,
})


LEGAL_TOOLCALL_TRANSITIONS: dict[ToolCallState, frozenset[ToolCallState]] = {
    ToolCallState.PENDING: frozenset({
        ToolCallState.APPROVAL_PENDING, ToolCallState.RUNNING,
        ToolCallState.REJECTED, ToolCallState.FAILED,
    }),
    ToolCallState.APPROVAL_PENDING: frozenset({
        ToolCallState.RUNNING, ToolCallState.REJECTED,
    }),
    ToolCallState.RUNNING: frozenset({
        ToolCallState.COMPLETED, ToolCallState.FAILED,
        ToolCallState.UNKNOWN_EFFECT,
    }),
    ToolCallState.UNKNOWN_EFFECT: frozenset({
        ToolCallState.RECONCILED, ToolCallState.FAILED,
    }),
    ToolCallState.REJECTED: frozenset(),
    ToolCallState.COMPLETED: frozenset(),
    ToolCallState.RECONCILED: frozenset(),
    ToolCallState.FAILED: frozenset(),
}


def is_legal_toolcall_transition(src: ToolCallState, dst: ToolCallState) -> bool:
    return dst in LEGAL_TOOLCALL_TRANSITIONS.get(src, frozenset())


# --------------------------------------------------------------------------- #
# Run / verdict
# --------------------------------------------------------------------------- #

class RunStatus(str, enum.Enum):
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BUDGET_EXHAUSTED = "budget_exhausted"
    ABORTED = "aborted"


TERMINAL_RUN_STATUSES: frozenset[RunStatus] = frozenset({
    RunStatus.COMPLETED,
    RunStatus.FAILED,
    RunStatus.CANCELLED,
    RunStatus.BUDGET_EXHAUSTED,
    RunStatus.ABORTED,
})


# --------------------------------------------------------------------------- #
# Specs (declarative input)
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class ToolSpec:
    """A registered capability the runtime can invoke."""

    name: str
    fn: Callable[..., Any]
    schema: Mapping[str, Any] = field(default_factory=dict)
    requires_approval: bool = False
    concurrent_safe: bool = True
    idempotent: bool = False
    timeout_s: float | None = None
    description: str = ""


# Capability is an alias retained for v3-style external imports.
Capability = ToolSpec


@dataclass(slots=True)
class NodeSpec:
    """Declarative description of a node to execute.

    Used as input by goal_resolver / Controller.replan to define DAG nodes.
    """

    node_id: str
    tool: str
    params: Mapping[str, Any] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    max_attempts: int = 3
    timeout_s: float | None = None
    requires_approval: bool = False
    priority: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def make(cls, tool: str, **params: Any) -> "NodeSpec":
        return cls(node_id=new_id("node"), tool=tool, params=params)


# --------------------------------------------------------------------------- #
# Failures, scopes, decisions
# --------------------------------------------------------------------------- #

class FailureKind(str, enum.Enum):
    TIMEOUT = "timeout"
    TOOL_ERROR = "tool_error"
    INVALID_OUTPUT = "invalid_output"
    BUDGET_EXHAUSTED = "budget_exhausted"
    WORKER_LOST = "worker_lost"
    DEPENDENCY_FAILED = "dependency_failed"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class Failure:
    kind: FailureKind
    message: str
    retryable: bool = True
    worker_id: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


class ScopeLevel(str, enum.Enum):
    STEP = "step"
    LOCAL = "local"
    GLOBAL = "global"


@dataclass(slots=True)
class Scope:
    level: ScopeLevel
    node_id: str | None = None
    reason: str = ""
    retry_after_ms: int | None = None


class DecisionKind(str, enum.Enum):
    ENQUEUE = "enqueue"
    REPLAN = "replan"
    HALT = "halt"
    WAIT = "wait"


@dataclass(slots=True)
class Decision:
    kind: DecisionKind
    node_specs: tuple[NodeSpec, ...] = ()
    scope: Scope | None = None
    reason: str = ""


# --------------------------------------------------------------------------- #
# Budget
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class Budget:
    max_tokens: int | None = None
    max_cost: float | None = None
    max_wallclock_s: float | None = None
    max_iterations: int | None = None
    warning_ratio: float = 0.8
    # Optional token→cost price. When set, BudgetTracker.consume derives a USD
    # cost from reported tokens whenever the LLM call reported no price of its
    # own (cost == 0) — e.g. the reasoning clients that surface tokens but not
    # dollars. ``None`` (default) leaves cost accounting byte-identical: a
    # tokens-only call accrues only ``consumed_tokens``, ``consumed_cost``
    # stays 0. See control/budget.py:consume.
    cost_per_1k_tokens: float | None = None

    consumed_tokens: int = 0
    consumed_cost: float = 0.0
    elapsed_s: float = 0.0
    iterations: int = 0

    def remaining_tokens(self) -> int | None:
        if self.max_tokens is None:
            return None
        return max(0, self.max_tokens - self.consumed_tokens)

    def remaining_cost(self) -> float | None:
        if self.max_cost is None:
            return None
        return max(0.0, self.max_cost - self.consumed_cost)

    def is_exhausted(self) -> bool:
        if self.max_tokens is not None and self.consumed_tokens >= self.max_tokens:
            return True
        if self.max_cost is not None and self.consumed_cost >= self.max_cost:
            return True
        if self.max_wallclock_s is not None and self.elapsed_s >= self.max_wallclock_s:
            return True
        if self.max_iterations is not None and self.iterations >= self.max_iterations:
            return True
        return False

    def is_warning(self) -> bool:
        ratio = self.warning_ratio
        if self.max_tokens and self.consumed_tokens >= self.max_tokens * ratio:
            return True
        if self.max_cost and self.consumed_cost >= self.max_cost * ratio:
            return True
        if self.max_wallclock_s and self.elapsed_s >= self.max_wallclock_s * ratio:
            return True
        return False

    def snapshot(self) -> dict[str, Any]:
        snap: dict[str, Any] = {
            "max_tokens": self.max_tokens,
            "max_cost": self.max_cost,
            "max_wallclock_s": self.max_wallclock_s,
            "max_iterations": self.max_iterations,
            "warning_ratio": self.warning_ratio,
            "consumed_tokens": self.consumed_tokens,
            "consumed_cost": self.consumed_cost,
            "elapsed_s": self.elapsed_s,
            "iterations": self.iterations,
        }
        # Gated so runs.budget_json is byte-identical for every run that does
        # not configure a price (the default).
        if self.cost_per_1k_tokens is not None:
            snap["cost_per_1k_tokens"] = self.cost_per_1k_tokens
        return snap


# --------------------------------------------------------------------------- #
# Verdict / step result
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class Verdict:
    run_id: str
    status: RunStatus
    summary: str = ""
    node_count: int = 0
    succeeded: int = 0
    failed: int = 0
    abandoned: int = 0
    skipped: int = 0
    cancelled: int = 0
    elapsed_s: float = 0.0
    iterations: int = 0
    budget: dict[str, Any] = field(default_factory=dict)
    halt_reason: str = ""
    error: str | None = None


@dataclass(slots=True)
class StepResult:
    iteration: int
    decision_kind: DecisionKind
    nodes_started: tuple[str, ...] = ()
    nodes_completed: tuple[str, ...] = ()
    should_halt: bool = False
    halt_reason: str = ""


# --------------------------------------------------------------------------- #
# Lease (worker assignment)
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class SpawnResult:
    """Returned by a ToolSpec.fn that wants to add child nodes to the graph
    mid-execution. The dispatcher unpacks this:
    * `value` becomes the node's terminal result.
    * each NodeSpec in `spawn` is added to the graph; `parent_node_id`
      metadata is auto-stamped so callers can trace lineage.

    Children may declare `depends_on` referencing each other — the engine
    will resolve order on the next promote tick. `depends_on` referencing
    *the parent* is implicit only if you set it explicitly (the parent is
    already SUCCEEDED by the time children run, so in practice deps are
    only between siblings).
    """
    value: Any = None
    spawn: list["NodeSpec"] = field(default_factory=list)


@dataclass(slots=True)
class Lease:
    lease_id: str
    run_id: str
    node_id: str
    worker_id: str
    granted_at_ms: int
    expires_at_ms: int
    heartbeat_at_ms: int

    def is_expired(self, now: int | None = None) -> bool:
        ts = now if now is not None else now_ms()
        return ts >= self.expires_at_ms


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def coerce_node_state(value: Any) -> NodeState:
    if isinstance(value, NodeState):
        return value
    return NodeState(str(value))


def coerce_toolcall_state(value: Any) -> ToolCallState:
    if isinstance(value, ToolCallState):
        return value
    return ToolCallState(str(value))


def deps_iterable(spec: NodeSpec) -> Iterable[str]:
    return tuple(spec.depends_on or ())


def specs_to_ids(specs: Sequence[NodeSpec]) -> tuple[str, ...]:
    return tuple(s.node_id for s in specs)


__all__ = [
    "Budget",
    "Capability",
    "Decision",
    "DecisionKind",
    "Failure",
    "FailureKind",
    "LEGAL_NODE_TRANSITIONS",
    "LEGAL_TOOLCALL_TRANSITIONS",
    "Lease",
    "NodeSpec",
    "NodeState",
    "RunStatus",
    "Scope",
    "ScopeLevel",
    "SpawnResult",
    "StepResult",
    "TERMINAL_NODE_STATES",
    "TERMINAL_TOOLCALL_STATES",
    "ToolCallState",
    "ToolSpec",
    "Verdict",
    "coerce_node_state",
    "coerce_toolcall_state",
    "deps_iterable",
    "is_legal_node_transition",
    "is_legal_toolcall_transition",
    "new_id",
    "now_ms",
    "specs_to_ids",
]
