from __future__ import annotations

from typing import Any

from ..memory import MemoryRuntime
from .base import BaseTool, ToolCall, ToolResult, ToolSpec, ValidationResult
from .context import ToolUseContext


class MemorySearchTool(BaseTool):
    def __init__(self, *, memory_runtime: MemoryRuntime) -> None:
        super().__init__(
            ToolSpec(
                name="memory_search",
                description="Search structured memory using the configured provider.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "wing": {"type": "string"},
                        "room": {"type": "string"},
                        "limit": {"type": "integer"},
                        "mode": {"type": "string"},
                        "structure_first": {"type": "boolean"},
                    },
                    "required": ["query"],
                },
                is_read_only=True,
            )
        )
        self.memory_runtime = memory_runtime

    def is_enabled(self, ctx):
        del ctx
        return False

    def validate_input(self, arguments: dict[str, Any]) -> ValidationResult:
        if not str(arguments.get("query") or "").strip():
            return ValidationResult(ok=False, message="query is required.")
        return ValidationResult(ok=True)

    async def execute(self, tool_call: ToolCall, ctx: ToolUseContext) -> ToolResult:
        del ctx
        mode = str(tool_call.arguments.get("mode") or "semantic")
        if bool(tool_call.arguments.get("structure_first")) and mode == "semantic":
            mode = "structure_first"
        bundle = self.memory_runtime.explicit_search(
            query=str(tool_call.arguments["query"]),
            wing=tool_call.arguments.get("wing"),
            room=tool_call.arguments.get("room"),
            limit=tool_call.arguments.get("limit"),
            mode=mode,
        )
        content = bundle.summary or "No memory results."
        return ToolResult(
            tool_use_id=tool_call.tool_use_id,
            tool_name=tool_call.tool_name,
            success=bundle.available and not bundle.error,
            content=content,
            data=bundle.to_prompt_payload(),
            error_code="SC1001" if bundle.error else None,
        )
