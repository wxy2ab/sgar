"""ToolCall state machine.

Models a single tool invocation within a node attempt. Adopts v4's
PENDING → APPROVAL_PENDING → RUNNING → COMPLETED / UNKNOWN_EFFECT → RECONCILED
sequence to handle worker crashes mid-call where the side effect (file
write, HTTP request) may have already happened.

Transition validity is enforced; illegal transitions raise ValueError so
bugs surface loudly instead of silently corrupting state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..types import (
    ToolCallState,
    is_legal_toolcall_transition,
    new_id,
    now_ms,
)


@dataclass(slots=True)
class ToolCall:
    call_id: str
    tool_name: str
    params: Mapping[str, Any]
    state: ToolCallState = ToolCallState.PENDING
    requires_approval: bool = False

    started_at_ms: int | None = None
    ended_at_ms: int | None = None
    result: Any | None = None
    error: str | None = None

    # `effect_signature` is set by the tool itself or the dispatcher to a
    # value that allows post-hoc reconciliation (e.g. a file path + content
    # hash, a request idempotency key). Only meaningful in UNKNOWN_EFFECT.
    effect_signature: str | None = None
    reconciled_outcome: str | None = None  # e.g. "success" / "no-op"

    history: list[tuple[str, str, int]] = field(default_factory=list)
    # Each entry: (from_state, to_state, ts_ms)

    @classmethod
    def new(
        cls,
        tool_name: str,
        params: Mapping[str, Any] | None = None,
        *,
        requires_approval: bool = False,
    ) -> "ToolCall":
        return cls(
            call_id=new_id("tc"),
            tool_name=tool_name,
            params=dict(params or {}),
            requires_approval=requires_approval,
        )

    # -- transitions ---------------------------------------------------------

    def _transition(self, target: ToolCallState, *, reason: str = "") -> None:
        if not is_legal_toolcall_transition(self.state, target):
            raise ValueError(
                f"illegal toolcall transition {self.state.value} -> {target.value}"
                + (f" ({reason})" if reason else "")
            )
        self.history.append((self.state.value, target.value, now_ms()))
        self.state = target

    def request_approval(self) -> None:
        self._transition(ToolCallState.APPROVAL_PENDING, reason="approval requested")

    def approve(self) -> None:
        self._transition(ToolCallState.RUNNING, reason="approved")

    def reject(self) -> None:
        self._transition(ToolCallState.REJECTED, reason="rejected")

    def mark_running(self) -> None:
        if self.started_at_ms is None:
            self.started_at_ms = now_ms()
        self._transition(ToolCallState.RUNNING)

    def mark_completed(self, result: Any) -> None:
        self.result = result
        self.ended_at_ms = now_ms()
        self._transition(ToolCallState.COMPLETED)

    def mark_failed(self, error: str) -> None:
        self.error = error
        self.ended_at_ms = now_ms()
        self._transition(ToolCallState.FAILED)

    def mark_unknown(self, reason: str, *, effect_signature: str | None = None) -> None:
        self.error = reason
        self.effect_signature = effect_signature or self.effect_signature
        self._transition(ToolCallState.UNKNOWN_EFFECT, reason=reason)

    def reconcile(self, *, outcome: str, result: Any = None) -> None:
        """Resolve UNKNOWN_EFFECT after external observation.

        outcome is a free-form string ("success", "no-op", "partial"); the
        ToolCall transitions to RECONCILED. If reconciliation reveals that
        the tool effectively did nothing or failed unrecoverably, the caller
        should mark it FAILED instead.
        """
        self.reconciled_outcome = outcome
        if result is not None:
            self.result = result
        self.ended_at_ms = now_ms()
        self._transition(ToolCallState.RECONCILED)

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "params": dict(self.params),
            "state": self.state.value,
            "requires_approval": self.requires_approval,
            "started_at_ms": self.started_at_ms,
            "ended_at_ms": self.ended_at_ms,
            "result": self.result,
            "error": self.error,
            "effect_signature": self.effect_signature,
            "reconciled_outcome": self.reconciled_outcome,
            "history": list(self.history),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ToolCall":
        return cls(
            call_id=d["call_id"],
            tool_name=d["tool_name"],
            params=dict(d.get("params") or {}),
            state=ToolCallState(d["state"]),
            requires_approval=bool(d.get("requires_approval", False)),
            started_at_ms=d.get("started_at_ms"),
            ended_at_ms=d.get("ended_at_ms"),
            result=d.get("result"),
            error=d.get("error"),
            effect_signature=d.get("effect_signature"),
            reconciled_outcome=d.get("reconciled_outcome"),
            history=[tuple(x) for x in (d.get("history") or [])],
        )


__all__ = ["ToolCall"]
