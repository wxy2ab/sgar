"""ccx_sgar cc tool — buffer SGAR governance ops from inside a cc turn.

Companion to ``ccx_spawn`` and ``ccx_research``. When a cc turn driven by
``CcAgentRunner`` wants to drive an SGAR (Stage-Governed Agent Runtime)
operation — ``init``, ``set-blueprint``, ``start-stage``, ``verify``,
``close-stage``, etc. — the LLM calls this tool. Each call buffers a
``SgarRequest`` into ``SgarBuffer`` exactly like ``ccx_spawn``: execution
is deferred. ``CcAgentRunner`` drains the buffer after the turn finishes
and turns each request into a ``SubagentInvocation(mode="sgar", ...)``,
which v5 dispatches via the ``ccx.sgar`` ToolSpec → ``BlueprintModeRunner``.

Why a separate tool from ccx_spawn:

* the buffered shape is governance-specific (``instruction`` text the
  ``BlueprintModeRunner`` parses, optionally with fenced ``blueprint`` /
  ``roadmap`` / ``stage-spec`` blocks) — keeping it separate avoids
  forcing the LLM to remember ``mode="sgar"`` on every spawn
* drained items always become ``mode="sgar"``, so the v5 layer can treat
  governance ops as a distinct node class for tracing / metrics
* the tool description is governance-flavoured, not decompositional

Each drained op becomes its own v5 node, durable in the graph store and
visible on the event bus — so a workflow's SGAR ops can be replayed and
audited the same way ordinary subagent invocations are.

Two patterns are supported:

* Single op:   ``ccx_sgar(instruction="init --session demo")``
* Bulk ops:    ``ccx_sgar(instructions=[{...}, {...}], sequential=True)``

Use ``sequential=True`` when subsequent ops depend on the previous one
(e.g. ``init`` → ``set-blueprint`` → ``validate blueprint --accept``).
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


_TOOL_NAME = "ccx_sgar"

_TOOL_DESCRIPTION = (
    "Drive one or more SGAR governance operations against the local .sgar "
    "workspace (or .sgar/sessions/<id>/ when a session is in scope). Each "
    "instruction is a single SGAR command parsed by BlueprintModeRunner — "
    "examples: 'init --session demo', 'set-blueprint --text \"...\"', "
    "'validate blueprint --accept', 'start-stage stage-01', "
    "'verify --stage stage-01 C1 --pass --evidence green', "
    "'close-stage stage-01', 'doctor', 'trace'. Operations may include "
    "fenced ```blueprint / ```roadmap / ```stage-spec blocks for inline "
    "writes. Buffered ops are drained after the turn and dispatched as "
    "ccx.sgar v5 nodes; use sequential=True for ops that depend on the "
    "previous one (the common case for SGAR workflows)."
)

_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "instruction": {
            "type": "string",
            "description": (
                "Single SGAR instruction (alternative to 'instructions'). "
                "BlueprintModeRunner parses subcommand + flags from this text."
            ),
        },
        "metadata": {
            "type": "object",
            "description": (
                "Optional free-form metadata attached to the buffered op. "
                "Use to override the parent-inherited sgar_session / cwd "
                "or to pass extra context to BlueprintModeRunner."
            ),
        },
        "instructions": {
            "type": "array",
            "description": (
                "Bulk ops — list of {instruction, metadata} entries. "
                "Mutually exclusive with the top-level 'instruction' field."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "instruction": {"type": "string"},
                    "metadata": {"type": "object"},
                },
                "required": ["instruction"],
            },
        },
        "sequential": {
            "type": "boolean",
            "default": False,
            "description": (
                "When true and 'instructions' has multiple entries, each "
                "entry depends on its predecessor (left-to-right chain). "
                "Use for ordered SGAR flows (init → set-blueprint → "
                "validate → start-stage → ...)."
            ),
        },
    },
}


# --------------------------------------------------------------------------- #
# SgarBuffer
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class SgarRequest:
    """Buffered SGAR op record. Drained by CcAgentRunner after the turn.

    ``mode`` selects which governance runtime the drained op dispatches to:
    ``"sgar"`` → ``ccx.sgar`` / ``BlueprintModeRunner`` (``.sgar/``), or
    ``"sgarx"`` → ``ccx.sgarx`` / ``BlueprintxModeRunner`` (``.sgarx/``,
    adds ``reopen-stage`` / ``abandon-stage``). Both share this buffer and
    command surface; only the target runtime differs.
    """
    instruction: str
    metadata: dict[str, Any] = field(default_factory=dict)
    sequential_with_previous: bool = False
    mode: str = "sgar"


class SgarBuffer:
    """Thread-safe queue of SgarRequests.

    A fresh buffer is created per cc turn by ``CcAgentRunner`` so its
    contents are scoped to that one turn — drain semantics match
    ``SpawnBuffer`` / ``ResearchBuffer``.
    """

    def __init__(self) -> None:
        self._items: list[SgarRequest] = []
        self._lock = threading.Lock()

    def append(self, request: SgarRequest) -> None:
        with self._lock:
            self._items.append(request)

    def extend(self, requests: list[SgarRequest]) -> None:
        with self._lock:
            self._items.extend(requests)

    def drain(self) -> list[dict[str, Any]]:
        """Empty the buffer and return raw dicts CcAgentRunner expects."""
        with self._lock:
            items = list(self._items)
            self._items.clear()
        return [
            {
                "instruction": r.instruction,
                "metadata": dict(r.metadata),
                "sequential_with_previous": r.sequential_with_previous,
                "mode": r.mode,
            }
            for r in items
        ]

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def snapshot(self) -> list[SgarRequest]:
        """Read-only copy without clearing — for inspection / tests."""
        with self._lock:
            return list(self._items)


# --------------------------------------------------------------------------- #
# Tool implementation
# --------------------------------------------------------------------------- #

class CcxSgarTool(BaseTool):
    """cc BaseTool that buffers SGAR governance ops."""

    def __init__(self, buffer: SgarBuffer) -> None:
        super().__init__(spec=CcToolSpec(
            name=_TOOL_NAME,
            description=_TOOL_DESCRIPTION,
            # Buffering is read-only from cc's perspective; the actual
            # writes to .sgar/ happen later in v5 via BlueprintModeRunner.
            is_read_only=True,
            needs_confirmation=False,
            input_schema=_INPUT_SCHEMA,
            metadata={"ccx": True, "ccx_sgar": True},
        ))
        self.buffer = buffer

    def is_enabled(self, ctx: Any) -> bool:
        # Hidden from the LLM-facing schema; the unified ``CcxUnifiedTool``
        # in ccx_tool.py replaces this surface. Class kept for direct
        # instantiation by tests and for SgarBuffer/SgarRequest types
        # still consumed by CcAgentRunner's drain logic.
        del ctx
        return False

    def validate_input(self, arguments: dict[str, Any]) -> ValidationResult:
        if not arguments.get("instruction") and not arguments.get("instructions"):
            return ValidationResult(
                ok=False,
                message="ccx_sgar requires either 'instruction' or 'instructions'",
            )
        if arguments.get("instruction") and arguments.get("instructions"):
            return ValidationResult(
                ok=False,
                message="ccx_sgar: pass 'instruction' OR 'instructions', not both",
            )
        instructions = arguments.get("instructions")
        if isinstance(instructions, list):
            for i, entry in enumerate(instructions):
                if not isinstance(entry, dict) or not entry.get("instruction"):
                    return ValidationResult(
                        ok=False,
                        message=f"ccx_sgar.instructions[{i}] missing 'instruction'",
                    )
        return ValidationResult(ok=True)

    def is_concurrency_safe(self, arguments: dict[str, Any]) -> bool:
        # Buffering is read-only from cc's perspective.
        return True

    async def execute(self, tool_call: ToolCall, ctx: Any) -> ToolResult:
        args = dict(tool_call.arguments or {})
        ops: list[SgarRequest] = []

        if args.get("instruction"):
            ops.append(SgarRequest(
                instruction=str(args["instruction"]),
                metadata=dict(args.get("metadata") or {}),
            ))
        else:
            sequential = bool(args.get("sequential", False))
            for index, entry in enumerate(args.get("instructions") or []):
                ops.append(SgarRequest(
                    instruction=str(entry["instruction"]),
                    metadata=dict(entry.get("metadata") or {}),
                    sequential_with_previous=(sequential and index > 0),
                ))

        self.buffer.extend(ops)
        queued = [{"instruction": r.instruction} for r in ops]
        return ToolResult(
            tool_use_id=tool_call.tool_use_id,
            tool_name=tool_call.tool_name,
            success=True,
            content=(
                f"Queued {len(queued)} SGAR op(s); they will run as "
                f"ccx.sgar nodes after this turn finishes."
            ),
            data={"queued": queued, "buffer_size": len(self.buffer)},
        )


def make_ccx_sgar_tool(buffer: SgarBuffer | None = None) -> CcxSgarTool:
    """Factory: returns a tool ready to register into a ToolRegistry.

    The returned tool's ``.buffer`` is the same SgarBuffer the caller can
    drain after the cc turn finishes. If no buffer is supplied, a fresh
    one is created.
    """
    return CcxSgarTool(buffer or SgarBuffer())


__all__ = [
    "CcxSgarTool",
    "SgarBuffer",
    "SgarRequest",
    "make_ccx_sgar_tool",
]
