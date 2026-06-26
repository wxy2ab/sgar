"""NodeExecution — runtime state for a single DAG node.

Combines what v4 split across NodeRunV4 + NodeAttemptV4 into a single
object: the node holds its own list of attempts inline. The
`attempts_index` table provides a denormalised view for queries that
care about cross-node attempt history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..types import (
    Failure,
    FailureKind,
    NodeSpec,
    NodeState,
    is_legal_node_transition,
    new_id,
    now_ms,
)
from .toolcall import ToolCall


@dataclass(slots=True)
class Attempt:
    attempt_id: str
    worker_id: str | None = None
    started_at_ms: int | None = None
    ended_at_ms: int | None = None
    outcome: str | None = None  # 'success' | 'failure' | 'abandoned' | 'cancelled'
    tool_calls: list[ToolCall] = field(default_factory=list)
    failure: Failure | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "worker_id": self.worker_id,
            "started_at_ms": self.started_at_ms,
            "ended_at_ms": self.ended_at_ms,
            "outcome": self.outcome,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "failure": _failure_to_dict(self.failure),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Attempt":
        return cls(
            attempt_id=d["attempt_id"],
            worker_id=d.get("worker_id"),
            started_at_ms=d.get("started_at_ms"),
            ended_at_ms=d.get("ended_at_ms"),
            outcome=d.get("outcome"),
            tool_calls=[ToolCall.from_dict(t) for t in (d.get("tool_calls") or [])],
            failure=_failure_from_dict(d.get("failure")),
        )


@dataclass(slots=True)
class NodeExecution:
    spec: NodeSpec
    state: NodeState = NodeState.PENDING
    attempts: list[Attempt] = field(default_factory=list)
    result: Any | None = None
    failure: Failure | None = None
    created_at_ms: int = field(default_factory=now_ms)
    updated_at_ms: int = field(default_factory=now_ms)
    history: list[tuple[str, str, int, str]] = field(default_factory=list)
    # Each entry: (from_state, to_state, ts_ms, reason)

    @property
    def node_id(self) -> str:
        return self.spec.node_id

    def is_terminal(self) -> bool:
        from ..types import TERMINAL_NODE_STATES
        return self.state in TERMINAL_NODE_STATES

    def attempt_count(self) -> int:
        return len(self.attempts)

    def can_retry(self) -> bool:
        return self.attempt_count() < self.spec.max_attempts

    # -- transitions ---------------------------------------------------------

    def transition(self, to: NodeState, *, reason: str = "") -> None:
        if not is_legal_node_transition(self.state, to):
            raise ValueError(
                f"illegal node transition {self.state.value} -> {to.value}"
                + (f" ({reason})" if reason else "")
            )
        ts = now_ms()
        self.history.append((self.state.value, to.value, ts, reason))
        self.state = to
        self.updated_at_ms = ts

    def new_attempt(self, *, worker_id: str | None = None) -> Attempt:
        att = Attempt(attempt_id=new_id("att"), worker_id=worker_id,
                      started_at_ms=now_ms())
        self.attempts.append(att)
        self.updated_at_ms = now_ms()
        return att

    def current_attempt(self) -> Attempt | None:
        return self.attempts[-1] if self.attempts else None

    def finish_attempt(
        self,
        *,
        outcome: str,
        result: Any | None = None,
        failure: Failure | None = None,
    ) -> None:
        att = self.current_attempt()
        if att is None:
            raise RuntimeError("finish_attempt with no current attempt")
        att.ended_at_ms = now_ms()
        att.outcome = outcome
        att.failure = failure
        if result is not None:
            self.result = result
        if failure is not None:
            self.failure = failure
        elif outcome == "success":
            self.failure = None
        self.updated_at_ms = now_ms()

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": _spec_to_dict(self.spec),
            "state": self.state.value,
            "attempts": [a.to_dict() for a in self.attempts],
            "result": self.result,
            "failure": _failure_to_dict(self.failure),
            "created_at_ms": self.created_at_ms,
            "updated_at_ms": self.updated_at_ms,
            "history": list(self.history),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "NodeExecution":
        return cls(
            spec=_spec_from_dict(d["spec"]),
            state=NodeState(d["state"]),
            attempts=[Attempt.from_dict(a) for a in (d.get("attempts") or [])],
            result=d.get("result"),
            failure=_failure_from_dict(d.get("failure")),
            created_at_ms=int(d.get("created_at_ms") or now_ms()),
            updated_at_ms=int(d.get("updated_at_ms") or now_ms()),
            history=[tuple(x) for x in (d.get("history") or [])],
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _spec_to_dict(spec: NodeSpec) -> dict[str, Any]:
    return {
        "node_id": spec.node_id,
        "tool": spec.tool,
        "params": dict(spec.params),
        "depends_on": list(spec.depends_on),
        "max_attempts": spec.max_attempts,
        "timeout_s": spec.timeout_s,
        "requires_approval": spec.requires_approval,
        "priority": spec.priority,
        "metadata": dict(spec.metadata),
    }


def _spec_from_dict(d: Mapping[str, Any]) -> NodeSpec:
    return NodeSpec(
        node_id=d["node_id"],
        tool=d["tool"],
        params=dict(d.get("params") or {}),
        depends_on=tuple(d.get("depends_on") or ()),
        max_attempts=int(d.get("max_attempts", 3)),
        timeout_s=d.get("timeout_s"),
        requires_approval=bool(d.get("requires_approval", False)),
        priority=int(d.get("priority", 0)),
        metadata=dict(d.get("metadata") or {}),
    )


def _failure_to_dict(f: Failure | None) -> dict[str, Any] | None:
    if f is None:
        return None
    return {
        "kind": f.kind.value,
        "message": f.message,
        "retryable": f.retryable,
        "worker_id": f.worker_id,
        "details": dict(f.details or {}),
    }


def _failure_from_dict(d: Mapping[str, Any] | None) -> Failure | None:
    if not d:
        return None
    return Failure(
        kind=FailureKind(d.get("kind", "unknown")),
        message=str(d.get("message", "")),
        retryable=bool(d.get("retryable", True)),
        worker_id=d.get("worker_id"),
        details=dict(d.get("details") or {}),
    )


__all__ = ["Attempt", "NodeExecution"]
