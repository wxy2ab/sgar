"""Escalation policy — classify a failure into step / local / global scope.

Inherits v3's three-tier philosophy:
* STEP   — retry the same node (within max_attempts).
* LOCAL  — the node has used up retries; replan around it (different tool,
  alternate approach) without touching the rest of the goal.
* GLOBAL — multiple LOCAL replans have failed; the whole plan needs revision.

The policy is rule-based by default and pure-functional: same inputs ⇒ same
output. Engine threads in the failure history; policy doesn't store state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..types import Failure, FailureKind, Scope, ScopeLevel


# Defaults chosen to roughly match v3's behaviour:
# - Step retries up to max_attempts (handled by NodeExecution.can_retry).
# - After max_attempts, escalate to LOCAL.
# - After 2 LOCAL escalations on related nodes, escalate to GLOBAL.
DEFAULT_LOCAL_TO_GLOBAL_THRESHOLD = 2


@dataclass(slots=True)
class EscalationContext:
    """Counts & context the engine threads in for each classification call."""

    node_id: str
    attempts_used: int
    max_attempts: int
    local_replans_used: int = 0
    global_replans_used: int = 0
    related_failed_nodes: Sequence[str] = ()


class EscalationPolicy:
    def __init__(
        self,
        *,
        local_to_global_threshold: int = DEFAULT_LOCAL_TO_GLOBAL_THRESHOLD,
    ) -> None:
        self.local_to_global_threshold = local_to_global_threshold

    def classify(
        self,
        failure: Failure,
        ctx: EscalationContext,
    ) -> Scope:
        # Non-retryable failures jump straight to local replanning.
        if not failure.retryable:
            if ctx.local_replans_used >= self.local_to_global_threshold:
                return Scope(
                    level=ScopeLevel.GLOBAL,
                    reason=f"non-retryable failure after {ctx.local_replans_used} local replans",
                )
            return Scope(
                level=ScopeLevel.LOCAL,
                node_id=ctx.node_id,
                reason=f"non-retryable: {failure.kind.value}",
            )

        # Budget exhaustion never retries.
        if failure.kind == FailureKind.BUDGET_EXHAUSTED:
            return Scope(
                level=ScopeLevel.GLOBAL,
                reason="budget exhausted",
            )

        # Worker lost (UNKNOWN_EFFECT path) — single retry of the same node
        # is fine, but if it keeps happening, escalate.
        if failure.kind == FailureKind.WORKER_LOST and ctx.attempts_used >= 2:
            return Scope(
                level=ScopeLevel.LOCAL,
                node_id=ctx.node_id,
                reason="repeated worker loss",
            )

        # Within retry budget → STEP (retry same node).
        if ctx.attempts_used < ctx.max_attempts:
            return Scope(
                level=ScopeLevel.STEP,
                node_id=ctx.node_id,
                reason=f"retry {ctx.attempts_used}/{ctx.max_attempts}",
            )

        # Out of retries on this node — go LOCAL.
        if ctx.local_replans_used < self.local_to_global_threshold:
            return Scope(
                level=ScopeLevel.LOCAL,
                node_id=ctx.node_id,
                reason=f"max attempts {ctx.max_attempts} reached",
            )

        # Too many LOCAL replans already — go GLOBAL.
        return Scope(
            level=ScopeLevel.GLOBAL,
            reason=(
                f"local replan budget exhausted "
                f"({ctx.local_replans_used}/{self.local_to_global_threshold})"
            ),
        )


__all__ = ["DEFAULT_LOCAL_TO_GLOBAL_THRESHOLD", "EscalationContext",
           "EscalationPolicy"]
