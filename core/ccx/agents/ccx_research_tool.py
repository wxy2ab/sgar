"""ccx_research cc tool — buffer read-only investigation requests.

Companion to ``ccx_spawn`` but specialised for *investigation*: when a cc
turn (driven by ``CcAgentRunner``) wants to fan out one or more parallel
read-only research questions that the engine should run as their own v5
nodes, the LLM calls this tool. Each call buffers a ``ResearchRequest``
into ``ResearchBuffer`` exactly like ccx_spawn — execution is deferred.
``CcAgentRunner`` drains the buffer after the turn finishes and turns
each request into a ``SubagentInvocation(mode="research", ...)`` which
v5 dispatches as parallel siblings.

Why a separate tool from ccx_spawn:

* the buffered shape is different (question + scope + focus_paths) and
  drained items become ``mode="research"``, routed to ResearchRunner
  with a read-only tool whitelist
* the LLM-facing description is investigative, not decompositional
* keeping the buffers separate avoids accidentally mixing "go do this
  whole subtask" with "go investigate this question"

Two patterns:

* Single research:   ``ccx_research(question="X", scope="src/auth")``
* Bulk research:     ``ccx_research(researches=[{question, scope}, ...])``

Concurrency: every buffered request becomes an independent v5 node, so
they run in parallel up to the runtime parallelism config. Use this
when you have N independent questions you'd otherwise ask sequentially.
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


_TOOL_NAME = "ccx_research"

_TOOL_DESCRIPTION = (
    "Spawn one or more PARALLEL read-only research subagents. Each "
    "research subagent investigates a single question against a scoped "
    "directory using ONLY read-only tools (Grep, Glob, Read). Use this "
    "when you have N independent investigative questions about the "
    "codebase that don't depend on each other — they will run "
    "concurrently and return structured findings (summary + evidence "
    "paths). Do NOT use this for tasks that need to write files; use "
    "ccx_spawn with mode='agent' for that. Researchers cannot write."
)

_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "question": {
            "type": "string",
            "description": (
                "Single research question (alternative to 'researches'). "
                "Phrase as a concrete question the subagent can answer "
                "by reading code, not as a task."
            ),
        },
        "scope": {
            "type": "string",
            "description": (
                "Directory or file path to scope the investigation to. "
                "Subagent will not look outside this scope. Default: cwd."
            ),
        },
        "focus_paths": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Optional list of file/dir hints the subagent should "
                "look at first (does not exclude others)."
            ),
        },
        "researches": {
            "type": "array",
            "description": (
                "Bulk research — list of {question, scope, focus_paths} "
                "entries. Mutually exclusive with the top-level "
                "'question' field."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "scope": {"type": "string"},
                    "focus_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["question"],
            },
        },
    },
}


def normalize_focus_paths(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    raise TypeError(
        "focus_paths must be a string or an array of path-like values"
    )


# --------------------------------------------------------------------------- #
# ResearchBuffer
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class ResearchRequest:
    """Buffered research record. Drained by CcAgentRunner after the turn.

    ``focus_paths`` is a soft hint — the runner's prompt encourages the
    LLM to start there but doesn't restrict to those paths.
    """
    question: str
    scope: str = ""
    focus_paths: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ResearchBuffer:
    """Thread-safe queue of ResearchRequests.

    A fresh buffer is created per cc turn by ``CcAgentRunner`` so its
    contents are scoped to that one turn — drain semantics match
    ``SpawnBuffer``.
    """

    def __init__(self) -> None:
        self._items: list[ResearchRequest] = []
        self._lock = threading.Lock()

    def append(self, request: ResearchRequest) -> None:
        request.focus_paths = normalize_focus_paths(request.focus_paths)
        with self._lock:
            self._items.append(request)

    def extend(self, requests: list[ResearchRequest]) -> None:
        for request in requests:
            self.append(request)

    def drain(self) -> list[dict[str, Any]]:
        """Empty the buffer and return raw dicts CcAgentRunner expects."""
        with self._lock:
            items = list(self._items)
            self._items.clear()
        return [
            {
                "question": r.question,
                "scope": r.scope,
                "focus_paths": list(r.focus_paths),
                "metadata": dict(r.metadata),
            }
            for r in items
        ]

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def snapshot(self) -> list[ResearchRequest]:
        """Read-only copy without clearing — for inspection / tests."""
        with self._lock:
            return list(self._items)


# --------------------------------------------------------------------------- #
# Tool implementation
# --------------------------------------------------------------------------- #

class CcxResearchTool(BaseTool):
    """cc BaseTool that buffers ccx research requests."""

    def __init__(self, buffer: ResearchBuffer) -> None:
        super().__init__(spec=CcToolSpec(
            name=_TOOL_NAME,
            description=_TOOL_DESCRIPTION,
            input_schema=_INPUT_SCHEMA,
            is_read_only=True,
            needs_confirmation=False,
            metadata={"ccx": True, "ccx_research": True},
        ))
        self.buffer = buffer

    def is_enabled(self, ctx: Any) -> bool:
        # Hidden from the LLM-facing schema; the unified ``CcxUnifiedTool``
        # in ccx_tool.py replaces this surface. Class kept for direct
        # instantiation by tests and for ResearchBuffer/ResearchRequest
        # types still consumed by CcAgentRunner's drain logic.
        del ctx
        return False

    def validate_input(self, arguments: dict[str, Any]) -> ValidationResult:
        if not arguments.get("question") and not arguments.get("researches"):
            return ValidationResult(
                ok=False,
                message="ccx_research requires either 'question' or 'researches'",
            )
        if arguments.get("question") and arguments.get("researches"):
            return ValidationResult(
                ok=False,
                message="ccx_research: pass 'question' OR 'researches', not both",
            )
        researches = arguments.get("researches")
        if isinstance(researches, list):
            for i, entry in enumerate(researches):
                if not isinstance(entry, dict) or not entry.get("question"):
                    return ValidationResult(
                        ok=False,
                        message=f"ccx_research.researches[{i}] missing 'question'",
                    )
                try:
                    normalize_focus_paths(entry.get("focus_paths"))
                except TypeError as exc:
                    return ValidationResult(
                        ok=False,
                        message=f"ccx_research.researches[{i}]: {exc}",
                    )
        else:
            try:
                normalize_focus_paths(arguments.get("focus_paths"))
            except TypeError as exc:
                return ValidationResult(
                    ok=False,
                    message=f"ccx_research: {exc}",
                )
        return ValidationResult(ok=True)

    def is_concurrency_safe(self, arguments: dict[str, Any]) -> bool:
        # Buffering is read-only from cc's perspective.
        return True

    async def execute(self, tool_call: ToolCall, ctx: Any) -> ToolResult:
        args = dict(tool_call.arguments or {})
        researches: list[ResearchRequest] = []

        if args.get("question"):
            researches.append(ResearchRequest(
                question=str(args["question"]),
                scope=str(args.get("scope") or ""),
                focus_paths=normalize_focus_paths(args.get("focus_paths")),
            ))
        else:
            for entry in args.get("researches") or []:
                researches.append(ResearchRequest(
                    question=str(entry["question"]),
                    scope=str(entry.get("scope") or ""),
                    focus_paths=normalize_focus_paths(entry.get("focus_paths")),
                ))

        self.buffer.extend(researches)
        queued = [
            {"question": r.question, "scope": r.scope}
            for r in researches
        ]
        return ToolResult(
            tool_use_id=tool_call.tool_use_id,
            tool_name=tool_call.tool_name,
            success=True,
            content=(
                f"Queued {len(queued)} research subagent(s); "
                f"they will run in parallel after this turn finishes "
                f"and return structured findings."
            ),
            data={"queued": queued, "buffer_size": len(self.buffer)},
        )


def make_ccx_research_tool(
    buffer: ResearchBuffer | None = None,
) -> CcxResearchTool:
    """Factory: returns a tool ready to register into a ToolRegistry.

    The returned tool's ``.buffer`` is the same ResearchBuffer the caller
    can drain after the cc turn finishes. If no buffer is supplied, a
    fresh one is created.
    """
    return CcxResearchTool(buffer or ResearchBuffer())


__all__ = [
    "CcxResearchTool",
    "ResearchBuffer",
    "ResearchRequest",
    "normalize_focus_paths",
    "make_ccx_research_tool",
]
