"""Controller — single class fusing v3's HierarchicalController + Proposer +
PolicyStack into one surface, with extension via three optional hooks
instead of v4's protocol-and-factory triplet.

Hooks (all optional, all simple Callables):

* `propose_initial(goal: str) -> list[NodeSpec]`
    Called once at engine.run(goal) start to populate the graph. If None,
    falls back to a no-op (caller is expected to add nodes manually).
* `replan(scope: Scope, failed_node, reason: str) -> list[NodeSpec]`
    Called after a node fails and EscalationPolicy returns LOCAL/GLOBAL.
    Receives the scope, the failed NodeExecution, and a human-readable
    reason. Default returns [] (no new nodes), causing the failed node
    to transition to ABANDONED.
* `on_failure(node, failure)` — diagnostic only; called for every failure.

Optional `llm_client` — passed through to hooks if they want to use it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from ..types import (
    Decision,
    DecisionKind,
    Failure,
    NodeSpec,
    NodeState,
    Scope,
    ScopeLevel,
)
from .budget import BudgetTracker


# Hook signatures
ProposeInitial = Callable[[str], Sequence[NodeSpec]]
ReplanHook = Callable[[Scope, Any, str], Sequence[NodeSpec]]
FailureHook = Callable[[Any, Failure], None]


@dataclass(slots=True)
class ControllerInputs:
    """Read-only snapshot the engine passes to Controller.decide()."""

    goal: str
    counts_by_state: dict[str, int]
    ready_nodes: tuple[str, ...]
    blocked_nodes: tuple[str, ...]
    approval_pending: tuple[str, ...]
    timer_hang: tuple[str, ...]
    in_flight_leases: int
    all_terminal: bool
    has_failed_terminal: bool


class Controller:
    def __init__(
        self,
        *,
        budget: BudgetTracker | None = None,
        propose_initial: ProposeInitial | None = None,
        replan_hook: ReplanHook | None = None,
        failure_hook: FailureHook | None = None,
        llm_client: Any | None = None,
    ) -> None:
        self.budget = budget
        self._propose_initial = propose_initial
        self._replan_hook = replan_hook
        self._failure_hook = failure_hook
        self.llm_client = llm_client

    @property
    def has_replan_hook(self) -> bool:
        return self._replan_hook is not None

    # -- initial planning ----------------------------------------------------

    def propose_initial(self, goal: str) -> list[NodeSpec]:
        if self._propose_initial is None:
            return []
        return list(self._propose_initial(goal))

    # -- decide --------------------------------------------------------------

    def decide(self, inputs: ControllerInputs) -> Decision:
        # 1. Budget exhausted → halt.
        if self.budget is not None and self.budget.should_halt():
            return Decision(
                kind=DecisionKind.HALT,
                reason="budget exhausted",
            )

        # 2. Graph fully terminal → halt.
        if inputs.all_terminal:
            return Decision(
                kind=DecisionKind.HALT,
                reason="all nodes terminal",
            )

        # 3. Ready work to do → enqueue.
        if inputs.ready_nodes:
            return Decision(
                kind=DecisionKind.ENQUEUE,
                reason=f"{len(inputs.ready_nodes)} ready",
            )

        # 4. In-flight leases (other workers running) → wait.
        if inputs.in_flight_leases > 0:
            return Decision(
                kind=DecisionKind.WAIT,
                reason="other workers in flight",
            )

        # 5. Pending approvals or timers but nothing else moving → halt.
        # The engine returns to caller, who must externally approve / fire
        # the timer and then call resume().
        if inputs.approval_pending or inputs.timer_hang:
            return Decision(
                kind=DecisionKind.HALT,
                reason=(
                    f"awaiting external action: "
                    f"{len(inputs.approval_pending)} approvals, "
                    f"{len(inputs.timer_hang)} timers"
                ),
            )

        # 5. Otherwise we're stuck — engine should escalate via replan path.
        return Decision(
            kind=DecisionKind.HALT,
            reason="no ready nodes and nothing in flight",
        )

    # -- replan --------------------------------------------------------------

    def replan(
        self,
        scope: Scope,
        failed_node: Any,
        reason: str,
    ) -> list[NodeSpec]:
        if self._replan_hook is None:
            return []
        return list(self._replan_hook(scope, failed_node, reason))

    # -- failure callback ----------------------------------------------------

    def notify_failure(self, node: Any, failure: Failure) -> None:
        if self._failure_hook is not None:
            try:
                self._failure_hook(node, failure)
            except Exception:
                # Hook errors must not break the engine loop.
                pass


__all__ = ["Controller", "ControllerInputs", "FailureHook", "ProposeInitial",
           "ReplanHook"]
