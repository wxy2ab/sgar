"""Subagent abstraction shared by plan/spec/agent modes.

The contract is mode-agnostic:

* `SubagentInvocation` — what a parent passes to a child (goal text + mode hint
  + arbitrary metadata such as parent goal lineage).
* `SubagentResult` — what a child returns: either terminal text, or a list of
  follow-up `SubagentInvocation`s with optional sequential ordering.

Each mode (plan / spec / agent) is just a different prompt + parser, but they
share this contract. ccx wraps each mode in a v5 `ToolSpec`; tool execution
calls the mode's prompt+parser and returns either a value (terminal) or a
`SpawnResult` (more work to do).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from core.deepstack_v5 import NodeSpec, SpawnResult, new_id

CCX_REQUIRES_APPROVAL_UNSUPPORTED = "CCX_REQUIRES_APPROVAL_UNSUPPORTED"


@dataclass(slots=True)
class SubagentInvocation:
    """A request for one subagent to execute, regardless of mode."""
    goal: str
    mode: str  # "plan" | "spec" | "agent"
    metadata: dict[str, Any] = field(default_factory=dict)
    requires_approval: bool = False
    max_attempts: int = 3
    timeout_s: float | None = None
    preferred_model: str | None = None
    """Optional model hint for LLM routing (R1).

    When set, ``build_runtime(llm_routes=...)`` looks up this key in
    its routes dict and uses the matched ``LLMCallable`` for this
    node. ``None`` (default) → falls through to the default ``llm`` so
    the pre-R1 behaviour is preserved exactly. The hint is opaque to
    ccx: any string the caller's ``llm_routes`` keys understand is
    valid (e.g. ``"pro"`` / ``"flash"`` / ``"deep"``).

    A planner may also set ``preferred_model`` on the spec / agent
    children it spawns (R1 Step B will add automatic
    ``<<<NEEDS_PRO>>>`` parsing — until then, callers can pre-populate
    children's preferred_model explicitly when they know one child is
    harder than the others).
    """


@dataclass(slots=True)
class SubagentResult:
    """Mode-agnostic output. Either terminal text, or follow-up invocations."""
    final_text: str = ""
    subtasks: list[SubagentInvocation] = field(default_factory=list)
    sequential: bool = False
    """If True, subtasks have an implicit chain: subtasks[i] depends on
    subtasks[i-1]. Use for linear pipelines (e.g. a spec's steps that must
    execute in order). Default False = parallel siblings."""

    extras: dict[str, Any] = field(default_factory=dict)
    """Free-form metadata returned with the result (for telemetry / claims)."""


# --------------------------------------------------------------------------- #
# Conversion: SubagentResult -> v5 SpawnResult
# --------------------------------------------------------------------------- #

def to_spawn_result(
    result: SubagentResult,
    *,
    parent_id: str,
    tool_for_mode: dict[str, str],
) -> SpawnResult:
    """Convert a mode-agnostic SubagentResult into a v5 SpawnResult.

    `tool_for_mode` maps "plan"/"spec"/"agent" -> the v5 tool name to invoke.

    Dependency wiring resolves in this order of precedence:

    1. ``inv.metadata["ccx_depends_on"]`` — explicit list of predecessor
       *indices into result.subtasks* (e.g. ``[0, 2]``). Lets a planner
       express true DAG ordering ("item 4 depends on items 2 and 3").
    2. ``inv.metadata["ccx_depends_on_previous"]`` — boolean. If true,
       this child depends on the immediately-preceding child only.
    3. ``result.sequential`` — legacy, applies to every child. Kept so
       existing callers that set this flag still get a chain.

    Mixing modes 2 and 3 is fine; per-item flags take precedence over
    the global flag. If neither is set, the child runs as a parallel
    sibling.
    """
    if not result.subtasks:
        return SpawnResult(value={"final_text": result.final_text,
                                  "extras": dict(result.extras)},
                          spawn=[])

    # Pre-compute node_ids so valid backward depends_on indices resolve
    # without mutating child metadata. Forward references are rejected
    # below to avoid cycles and scheduler ambiguity.
    invocations = list(result.subtasks)
    if any(inv.requires_approval for inv in invocations):
        raise ValueError(
            f"{CCX_REQUIRES_APPROVAL_UNSUPPORTED}: "
            "ccx does not support requires_approval=True; approval-gated "
            "v5 nodes would otherwise wait forever at the ccx boundary."
        )

    inv_modes = [str(inv.mode or "agent") for inv in invocations]
    node_ids: list[str] = [new_id(mode) for mode in inv_modes]

    specs: list[NodeSpec] = []
    dependency_issues: list[str] = []
    for i, inv in enumerate(invocations):
        tool_name = tool_for_mode.get(inv.mode)
        if tool_name is None:
            raise ValueError(f"no v5 tool registered for mode={inv.mode!r}")
        node_id = node_ids[i]

        explicit = inv.metadata.get("ccx_depends_on")
        depends_on: tuple[str, ...] = ()
        if isinstance(explicit, (list, tuple)) and explicit:
            resolved: list[str] = []
            for idx in explicit:
                if type(idx) is not int:
                    dependency_issues.append(
                        f"subtask {i} ignored non-integer ccx_depends_on "
                        f"index {idx!r}"
                    )
                    continue
                if not 0 <= idx < i:
                    dependency_issues.append(
                        f"subtask {i} ignored dangling ccx_depends_on "
                        f"index {idx}"
                    )
                    continue
                node_dep = node_ids[idx]
                if node_dep in resolved:
                    dependency_issues.append(
                        f"subtask {i} ignored duplicate ccx_depends_on "
                        f"index {idx}"
                    )
                    continue
                resolved.append(node_dep)
            depends_on = tuple(resolved)
        elif inv.metadata.get("ccx_depends_on_previous") and i > 0:
            depends_on = (node_ids[i - 1],)
        elif result.sequential and i > 0:
            depends_on = (node_ids[i - 1],)

        # R1: preferred_model flows from invocation to NodeSpec by
        # stashing it on metadata. The v5 dispatcher echoes metadata
        # back into the dispatched fn's params, where _make_mode_tool
        # re-extracts it onto the rebuilt SubagentInvocation for the
        # child's run. Explicit metadata key beats inv.preferred_model
        # so a caller can override per-child via inv.metadata when
        # they don't want to mutate the dataclass.
        child_metadata = dict(inv.metadata)
        child_metadata.pop("node_id", None)
        if inv.preferred_model is not None:
            child_metadata.setdefault("preferred_model", inv.preferred_model)

        meta = {
            "ccx_mode": inv.mode,
            "ccx_parent_invocation_goal": result.extras.get("goal", ""),
            **child_metadata,
        }
        specs.append(NodeSpec(
            node_id=node_id,
            tool=tool_name,
            params={"goal": inv.goal, "metadata": dict(child_metadata)},
            depends_on=depends_on,
            max_attempts=inv.max_attempts,
            timeout_s=inv.timeout_s,
            requires_approval=False,
            metadata=meta,
        ))
    extras = dict(result.extras)
    if dependency_issues:
        extras["ccx_dependency_issues"] = dependency_issues
    return SpawnResult(
        value={"final_text": result.final_text,
               "spawned_count": len(specs),
               "extras": extras},
        spawn=specs,
    )


# --------------------------------------------------------------------------- #
# Mode runner contract: a callable invoked inside the v5 tool fn
# --------------------------------------------------------------------------- #

class ModeRunner:
    """Contract a mode implementation must satisfy.

    Subclasses (or simple functions wrapped via `_FunctionRunner`) take an
    invocation and return a SubagentResult. They are responsible for prompt
    construction, LLM invocation, output parsing, and deciding whether to
    terminate or spawn.
    """

    mode_name: str = ""

    def run(self, invocation: SubagentInvocation) -> SubagentResult:
        raise NotImplementedError


__all__ = [
    "CCX_REQUIRES_APPROVAL_UNSUPPORTED",
    "ModeRunner",
    "SubagentInvocation",
    "SubagentResult",
    "to_spawn_result",
]
