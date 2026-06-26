"""ccx_spawn cc tool — recursive subagent spawning from inside a cc turn.

When the CcAgentRunner drives a cc QueryEngine and the LLM decides the
task is too large to handle alone, it can call the ``ccx_spawn`` tool
to enqueue child agents. The tool is *non-executing* in cc terms —
calling it queues a ``SubagentInvocation``-shaped record into a
SpawnBuffer; the actual v5 NodeSpec creation happens after the cc turn
returns, when CcAgentRunner drains the buffer and converts it into a
SpawnResult.

This decoupling matters because cc's tool execution path is synchronous
within a turn: if ``ccx_spawn`` immediately added v5 nodes the cc tool
result would have nothing useful to put back into the LLM's context.
By buffering, the LLM gets a deterministic "queued" acknowledgement and
the v5 layer takes over orchestration once the turn finishes.

Two patterns are supported:

* Single spawn:   ``ccx_spawn(goal="X", mode="agent")``
* Bulk spawn:     ``ccx_spawn(spawns=[{...}, {...}], sequential=False)``

Both shape into a list of records on the SpawnBuffer.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from core.cc.tools.base import (
    BaseTool,
    ToolCall,
    ToolResult,
    ToolSpec as CcToolSpec,
    ValidationResult,
)

from .subagent import CCX_REQUIRES_APPROVAL_UNSUPPORTED


_TOOL_NAME = "ccx_spawn"

# Allow-list of modes a parent may spawn through ccx_spawn. Sourced
# semantically from runtime.CCX_MODE_TOOL_MAP minus "research" — research has
# its own dedicated ccx_research tool with read-only buffer semantics, and
# mixing the two would defeat the safety/intent split.
_SPAWNABLE_MODES: tuple[str, ...] = (
    "plan",
    "spec",
    "agent",
    "doc",
    "ask",
    "blueprint",
    "sgar",
)

_TOOL_DESCRIPTION = (
    "Spawn one or more child subagents that will run in parallel after this "
    "turn finishes. Each spawned subagent receives its own goal and mode. "
    "Use 'plan' / 'spec' / 'agent' for ordinary decomposition, "
    "'sgar' / 'blueprint' to drive SGAR governance ops (init, set-blueprint, "
    "start-stage, verify, close-stage), and 'doc' / 'ask' for documentation "
    "or read-only Q&A. The spawned subagents become children of the current "
    "agent node in the orchestration DAG."
)

_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "goal": {
            "type": "string",
            "description": "Single goal to spawn (alternative to 'spawns').",
        },
        "mode": {
            "type": "string",
            "enum": list(_SPAWNABLE_MODES),
            "default": "agent",
        },
        "metadata": {
            "type": "object",
            "description": "Optional free-form metadata attached to the child.",
        },
        "contract": {
            "type": "object",
            "description": (
                "Optional machine-verified acceptance contract folded into "
                "the child's metadata['ccx_contract']. Spawner-authored; "
                "honored by 'agent' mode under cc_query_loop when the runner "
                "has the spawn-contract feature enabled."
            ),
        },
        "spawns": {
            "type": "array",
            "description": (
                "Bulk spawn — list of {goal, mode, metadata, contract} "
                "entries. Mutually exclusive with the top-level 'goal' field."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "goal": {"type": "string"},
                    "mode": {
                        "type": "string",
                        "enum": list(_SPAWNABLE_MODES),
                    },
                    "metadata": {"type": "object"},
                    "contract": {"type": "object"},
                },
                "required": ["goal"],
            },
        },
        "sequential": {
            "type": "boolean",
            "default": False,
            "description": (
                "When true and 'spawns' has multiple entries, each entry "
                "depends on its predecessor (left-to-right chain)."
            ),
        },
    },
}


def _with_contract(
    metadata: dict[str, Any], contract: Any,
) -> dict[str, Any]:
    """Fold an optional spawn contract into ``metadata['ccx_contract']``.

    ``setdefault`` so a caller who wrote ``ccx_contract`` straight into
    metadata wins over the convenience ``contract`` field. The contract is
    opaque here — structural validation happens at parse time in
    ``governed_spawn.parse_contract``.
    """
    if contract is not None:
        metadata.setdefault("ccx_contract", contract)
    return metadata


# --------------------------------------------------------------------------- #
# SpawnBuffer
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class SpawnRequest:
    """Buffered spawn record. Drained by CcAgentRunner after the turn."""
    goal: str
    mode: str = "agent"
    metadata: dict[str, Any] = field(default_factory=dict)
    sequential_with_previous: bool = False


class SpawnBuffer:
    """Thread-safe queue of SpawnRequests.

    Multiple agent threads may share a buffer if a custom build wants to;
    in the common case each ``CcAgentRunner`` invocation creates a fresh
    buffer scoped to that one cc turn.
    """

    def __init__(self) -> None:
        self._items: list[SpawnRequest] = []
        self._lock = threading.Lock()

    def append(self, request: SpawnRequest) -> None:
        with self._lock:
            self._items.append(request)

    def extend(self, requests: list[SpawnRequest]) -> None:
        with self._lock:
            self._items.extend(requests)

    def drain(self) -> list[dict[str, Any]]:
        """Empty the buffer and return raw dicts CcAgentRunner expects."""
        with self._lock:
            items = list(self._items)
            self._items.clear()
        return [
            {
                "goal": r.goal,
                "mode": r.mode,
                "metadata": dict(r.metadata),
                "sequential_with_previous": r.sequential_with_previous,
            }
            for r in items
        ]

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def snapshot(self) -> list[SpawnRequest]:
        """Read-only copy without clearing — for inspection / tests."""
        with self._lock:
            return list(self._items)


# --------------------------------------------------------------------------- #
# Tool implementation
# --------------------------------------------------------------------------- #

class CcxSpawnTool(BaseTool):
    """cc BaseTool that buffers ccx subagent spawn requests."""

    def __init__(self, buffer: SpawnBuffer) -> None:
        super().__init__(spec=CcToolSpec(
            name=_TOOL_NAME,
            description=_TOOL_DESCRIPTION,
            input_schema=_INPUT_SCHEMA,
            is_read_only=True,
            needs_confirmation=False,
            metadata={"ccx": True},
        ))
        self.buffer = buffer

    def is_enabled(self, ctx: Any) -> bool:
        # Hidden from the LLM-facing schema; the unified ``CcxUnifiedTool``
        # in ccx_tool.py replaces this surface. Kept as a class because
        # direct-instantiation tests still call ``execute`` on it, and the
        # SpawnBuffer / SpawnRequest types it exposes are imported by
        # CcAgentRunner's drain logic.
        del ctx
        return False

    def validate_input(self, arguments: dict[str, Any]) -> ValidationResult:
        spawns_provided = (
            "spawns" in arguments and arguments.get("spawns") is not None
        )
        spawns = arguments.get("spawns") if spawns_provided else None
        if not arguments.get("goal") and not spawns_provided:
            return ValidationResult(
                ok=False,
                message="ccx_spawn requires either 'goal' or 'spawns'",
            )
        if spawns_provided and not isinstance(spawns, list):
            return ValidationResult(
                ok=False,
                message="ccx_spawn: 'spawns' must be an array",
            )
        if isinstance(spawns, list) and not spawns:
            return ValidationResult(
                ok=False,
                message="ccx_spawn: 'spawns' must not be empty",
            )
        if arguments.get("goal") and spawns_provided:
            return ValidationResult(
                ok=False,
                message="ccx_spawn: pass 'goal' OR 'spawns', not both",
            )
        if arguments.get("requires_approval") is True:
            return ValidationResult(
                ok=False,
                message=(
                    f"{CCX_REQUIRES_APPROVAL_UNSUPPORTED}: "
                    "ccx_spawn does not support requires_approval=True"
                ),
            )
        # Some LLM providers honour JSON-schema enum loosely; enforce here too.
        top_mode = arguments.get("mode")
        if top_mode is not None and top_mode not in _SPAWNABLE_MODES:
            return ValidationResult(
                ok=False,
                message=(
                    f"ccx_spawn: mode={top_mode!r} not allowed; "
                    f"choose one of {list(_SPAWNABLE_MODES)}"
                ),
            )
        if isinstance(spawns, list):
            for i, entry in enumerate(spawns):
                if not isinstance(entry, dict) or not entry.get("goal"):
                    return ValidationResult(
                        ok=False,
                        message=f"ccx_spawn.spawns[{i}] missing 'goal'",
                    )
                if entry.get("requires_approval") is True:
                    return ValidationResult(
                        ok=False,
                        message=(
                            f"{CCX_REQUIRES_APPROVAL_UNSUPPORTED}: "
                            f"ccx_spawn.spawns[{i}] does not support "
                            "requires_approval=True"
                        ),
                    )
                entry_mode = entry.get("mode")
                if entry_mode is not None and entry_mode not in _SPAWNABLE_MODES:
                    return ValidationResult(
                        ok=False,
                        message=(
                            f"ccx_spawn.spawns[{i}].mode={entry_mode!r} not allowed; "
                            f"choose one of {list(_SPAWNABLE_MODES)}"
                        ),
                    )
        return ValidationResult(ok=True)

    def is_concurrency_safe(self, arguments: dict[str, Any]) -> bool:
        # Buffering is read-only from cc's perspective.
        return True

    async def execute(self, tool_call: ToolCall, ctx: Any) -> ToolResult:
        args = dict(tool_call.arguments or {})
        spawns_input: list[SpawnRequest] = []

        if args.get("goal"):
            spawns_input.append(SpawnRequest(
                goal=str(args["goal"]),
                mode=str(args.get("mode") or "agent"),
                metadata=_with_contract(
                    dict(args.get("metadata") or {}), args.get("contract"),
                ),
            ))
        else:
            sequential = bool(args.get("sequential", False))
            for index, entry in enumerate(args.get("spawns") or []):
                spawns_input.append(SpawnRequest(
                    goal=str(entry["goal"]),
                    mode=str(entry.get("mode") or "agent"),
                    metadata=_with_contract(
                        dict(entry.get("metadata") or {}),
                        entry.get("contract"),
                    ),
                    sequential_with_previous=(sequential and index > 0),
                ))

        self.buffer.extend(spawns_input)
        queued = [
            {"goal": s.goal, "mode": s.mode}
            for s in spawns_input
        ]
        return ToolResult(
            tool_use_id=tool_call.tool_use_id,
            tool_name=tool_call.tool_name,
            success=True,
            content=(
                f"Queued {len(queued)} subagent(s); "
                f"they will run after this turn finishes."
            ),
            data={"queued": queued, "buffer_size": len(self.buffer)},
        )


def make_ccx_spawn_tool(buffer: SpawnBuffer | None = None) -> CcxSpawnTool:
    """Factory: returns a tool ready to register into a ToolRegistry.

    The returned tool's `.buffer` is the same SpawnBuffer the caller can
    drain after the cc turn finishes. If no buffer is supplied, a fresh
    one is created.
    """
    return CcxSpawnTool(buffer or SpawnBuffer())


__all__ = [
    "CcxSpawnTool",
    "SpawnBuffer",
    "SpawnRequest",
    "make_ccx_spawn_tool",
    "_SPAWNABLE_MODES",
]
