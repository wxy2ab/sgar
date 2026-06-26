from __future__ import annotations

from typing import Any

from ..memory import MemoryRuntime, MemoryWriteCandidate
from ..memory.policy import infer_room
from .base import BaseTool, ToolCall, ToolResult, ToolSpec, ValidationResult
from .context import ToolUseContext


class MemoryStoreTool(BaseTool):
    def __init__(self, *, memory_runtime: MemoryRuntime) -> None:
        super().__init__(
            ToolSpec(
                name="memory_store",
                description="Store a structured memory candidate in the configured provider.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "memory_kind": {"type": "string"},
                        "subject": {"type": "string"},
                        "summary": {"type": "string"},
                        "text": {"type": "string"},
                        "wing": {"type": "string"},
                        "room": {"type": "string"},
                        "sources": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["memory_kind", "subject", "summary", "text"],
                },
            )
        )
        self.memory_runtime = memory_runtime

    def is_enabled(self, ctx):
        del ctx
        return False

    def validate_input(self, arguments: dict[str, Any]) -> ValidationResult:
        for field in ("memory_kind", "subject", "summary", "text"):
            if not str(arguments.get(field) or "").strip():
                return ValidationResult(ok=False, message=f"{field} is required.")
        return ValidationResult(ok=True)

    async def execute(self, tool_call: ToolCall, ctx: ToolUseContext) -> ToolResult:
        candidate = MemoryWriteCandidate(
            memory_kind=str(tool_call.arguments["memory_kind"]),
            subject=str(tool_call.arguments["subject"]),
            summary=str(tool_call.arguments["summary"]),
            text=str(tool_call.arguments["text"]),
            wing=str(tool_call.arguments.get("wing") or "wing_code"),
            room=str(tool_call.arguments.get("room") or infer_room(str(tool_call.arguments["memory_kind"]))),
            sources=(
                [tool_call.arguments["sources"]]
                if isinstance(tool_call.arguments.get("sources"), str)
                else [str(item) for item in (tool_call.arguments.get("sources") or []) if item is not None]
            ),
            details={"session_id": ctx.session_id},
        )
        result = self.memory_runtime.explicit_store(candidate)
        return ToolResult(
            tool_use_id=tool_call.tool_use_id,
            tool_name=tool_call.tool_name,
            success=result.success,
            content=result.message,
            data={
                "stored": result.stored,
                "duplicate": result.duplicate,
                "memory_id": result.memory_id,
                "fact_ids": result.fact_ids,
            },
            error_code=None if result.success else "SC1002",
        )
