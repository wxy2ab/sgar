from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import BaseTool, ToolCall, ToolResult, ToolSpec, ValidationResult
from .context import ToolUseContext


@dataclass(slots=True)
class TodoItem:
    content: str
    status: str = "pending"

    def to_dict(self) -> dict[str, str]:
        return {"content": self.content, "status": self.status}


class TodoWriteTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            ToolSpec(
                name="todo_write",
                description="Update the session todo list used for planning and progress tracking.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "todos": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "content": {"type": "string"},
                                    "status": {"type": "string"},
                                },
                                "required": ["content"],
                            },
                        },
                    },
                    "required": ["todos"],
                },
                is_read_only=False,
            )
        )

    def validate_input(self, arguments: dict[str, Any]) -> ValidationResult:
        todos = arguments.get("todos")
        if not isinstance(todos, list):
            return ValidationResult(ok=False, message="todos must be a list.")
        for item in todos:
            if not isinstance(item, dict) or not item.get("content"):
                return ValidationResult(ok=False, message="each todo needs a content field.")
        return ValidationResult(ok=True)

    async def execute(self, tool_call: ToolCall, ctx: ToolUseContext) -> ToolResult:
        incoming = [
            TodoItem(content=str(item["content"]), status=str(item.get("status", "pending")))
            for item in tool_call.arguments["todos"]
        ]
        current_state = ctx.get_app_state()
        old_todos = list(current_state.get("todos", []))
        incoming_dicts = [item.to_dict() for item in incoming]

        def modify(current_ctx: ToolUseContext) -> ToolUseContext:
            next_state = current_ctx.get_app_state()
            existing = {t["content"]: t for t in next_state.get("todos", [])}
            for item in incoming_dicts:
                existing[item["content"]] = item
            next_state["todos"] = list(existing.values())
            return current_ctx.set_app_state(next_state)

        merged_preview = {t["content"]: t for t in old_todos}
        for item in incoming_dicts:
            merged_preview[item["content"]] = item
        merged_todos = list(merged_preview.values())

        return ToolResult(
            tool_use_id=tool_call.tool_use_id,
            tool_name=tool_call.tool_name,
            success=True,
            content=f"Todo list updated ({len(merged_todos)} items).",
            data={"old_todos": old_todos, "new_todos": merged_todos},
            context_modifiers=[modify],
        )
