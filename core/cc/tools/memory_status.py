from __future__ import annotations

from ..memory import MemoryRuntime
from .base import BaseTool, ToolCall, ToolResult, ToolSpec
from .context import ToolUseContext


class MemoryStatusTool(BaseTool):
    """Legacy alias for the unified ``MemoryTool(action="status")``.

    Kept callable for in-process consumers and existing tests; hidden from
    the LLM-facing schema via ``is_enabled``.
    """

    def __init__(self, *, memory_runtime: MemoryRuntime) -> None:
        super().__init__(
            ToolSpec(
                name="memory_status",
                description="Inspect the configured memory provider and its current availability.",
                input_schema={"type": "object", "properties": {}},
                is_read_only=True,
            )
        )
        self.memory_runtime = memory_runtime

    def is_enabled(self, ctx):  # noqa: D401
        del ctx
        return False

    async def execute(self, tool_call: ToolCall, ctx: ToolUseContext) -> ToolResult:
        del ctx
        status = self.memory_runtime.status()
        content = (
            f"provider={status.provider}\n"
            f"available={status.available}\n"
            f"message={status.message}"
        )
        return ToolResult(
            tool_use_id=tool_call.tool_use_id,
            tool_name=tool_call.tool_name,
            success=True,
            content=content,
            data={
                "provider": status.provider,
                "available": status.available,
                "message": status.message,
                "details": status.details,
            },
        )
