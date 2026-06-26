"""Unified MemoryTool — facade over the four memory primitives.

Replaces the previous 4-tool surface (``memory_status``, ``memory_search``,
``memory_store``, ``memory_fact``) with a single LLM-facing tool that
dispatches by ``action``. The four legacy classes remain as hidden alias
shims for one deprecation window so existing in-process callers and
tests keep working — see ``core/cc/tools/builtin.py`` for the
registration site.

Why one tool instead of four:

- All four operate on the same domain (memory provider) and were always
  visible to the LLM simultaneously. The LLM had to remember which-tool
  for which-action; ``action`` as a discriminator matches how the model
  already reasons about it.
- Per-action concurrency safety is preserved via ``is_concurrency_safe``:
  read actions (``status``, ``search``, ``fact_query``) run in parallel;
  write actions (``store``, ``fact_store``) serialize as before.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..memory import MemoryRuntime
from .base import BaseTool, ToolCall, ToolResult, ToolSpec, ValidationResult
from .context import ToolUseContext
from .memory_fact import MemoryFactTool
from .memory_search import MemorySearchTool
from .memory_status import MemoryStatusTool
from .memory_store import MemoryStoreTool


_ACTIONS = ("status", "search", "store", "fact_query", "fact_store")
_READ_ACTIONS = frozenset({"status", "search", "fact_query"})


class MemoryTool(BaseTool):
    def __init__(self, *, memory_runtime: MemoryRuntime) -> None:
        super().__init__(
            ToolSpec(
                name="memory",
                description=(
                    "Read or write structured memory via the configured provider. "
                    "Pick an action: 'status' to inspect provider availability; "
                    "'search' for semantic/structure-first retrieval (provide 'query'); "
                    "'store' to save a memory candidate (provide memory_kind, subject, "
                    "summary, text); 'fact_query' to look up RDF-style facts about an "
                    "entity (provide 'entity'); 'fact_store' to write a fact triple "
                    "(provide subject, predicate, object)."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": list(_ACTIONS),
                            "description": (
                                "status | search | store | fact_query | fact_store"
                            ),
                        },
                        # search
                        "query": {"type": "string"},
                        "wing": {"type": "string"},
                        "room": {"type": "string"},
                        "limit": {"type": "integer"},
                        "mode": {"type": "string"},
                        "structure_first": {"type": "boolean"},
                        # store
                        "memory_kind": {"type": "string"},
                        "subject": {"type": "string"},
                        "summary": {"type": "string"},
                        "text": {"type": "string"},
                        "sources": {"type": "array", "items": {"type": "string"}},
                        # fact_query
                        "entity": {"type": "string"},
                        "as_of": {"type": "string"},
                        "direction": {"type": "string"},
                        # fact_store
                        "predicate": {"type": "string"},
                        "object": {"type": "string"},
                        "valid_from": {"type": "string"},
                        "valid_to": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["action"],
                },
                is_read_only=False,
            )
        )
        self._status = MemoryStatusTool(memory_runtime=memory_runtime)
        self._search = MemorySearchTool(memory_runtime=memory_runtime)
        self._store = MemoryStoreTool(memory_runtime=memory_runtime)
        self._fact = MemoryFactTool(memory_runtime=memory_runtime)

    def is_concurrency_safe(self, arguments: dict[str, Any]) -> bool:
        return str(arguments.get("action") or "") in _READ_ACTIONS

    def validate_input(self, arguments: dict[str, Any]) -> ValidationResult:
        action = str(arguments.get("action") or "")
        if action not in _ACTIONS:
            return ValidationResult(
                ok=False,
                message=f"action must be one of {_ACTIONS}, got {action!r}.",
            )
        if action == "status":
            return ValidationResult(ok=True)
        if action == "search":
            return self._search.validate_input(arguments)
        if action == "store":
            return self._store.validate_input(arguments)
        if action == "fact_query":
            return self._fact.validate_input({**arguments, "action": "query"})
        if action == "fact_store":
            return self._fact.validate_input({**arguments, "action": "store"})
        return ValidationResult(ok=True)  # pragma: no cover - unreachable

    async def execute(self, tool_call: ToolCall, ctx: ToolUseContext) -> ToolResult:
        action = str(tool_call.arguments.get("action") or "")
        if action == "status":
            return await self._status.execute(tool_call, ctx)
        if action == "search":
            return await self._search.execute(tool_call, ctx)
        if action == "store":
            return await self._store.execute(tool_call, ctx)
        if action == "fact_query":
            forwarded = replace(
                tool_call,
                arguments={**tool_call.arguments, "action": "query"},
            )
            return await self._fact.execute(forwarded, ctx)
        if action == "fact_store":
            forwarded = replace(
                tool_call,
                arguments={**tool_call.arguments, "action": "store"},
            )
            return await self._fact.execute(forwarded, ctx)
        return ToolResult(
            tool_use_id=tool_call.tool_use_id,
            tool_name=tool_call.tool_name,
            success=False,
            content=f"Unknown memory action: {action!r}.",
            error_code="TL1002",
        )
