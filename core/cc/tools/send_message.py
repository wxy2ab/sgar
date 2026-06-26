from __future__ import annotations

import uuid
from typing import Any

from ..errors import AgentTaskError
from ..agents.runtime import AgentMessage
from ..agents.runtime_registry import InProcessRuntimeRegistry, get_in_process_runtime_registry
from ..agents.task_model import is_terminal_status
from .base import BaseTool, ToolCall, ToolResult, ToolSpec, ValidationResult
from .context import ToolUseContext


class SendMessageTool(BaseTool):
    def __init__(self, runtime_registry: InProcessRuntimeRegistry | None = None) -> None:
        super().__init__(
            ToolSpec(
                name="send_message",
                description="Send a follow-up message to a running agent runtime.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "to": {"type": "string"},
                        "message": {"type": "string"},
                        "kind": {"type": "string"},
                    },
                    "required": ["to", "message"],
                },
            )
        )
        self.runtime_registry = runtime_registry

    def validate_input(self, arguments: dict[str, Any]) -> ValidationResult:
        if not arguments.get("to"):
            return ValidationResult(ok=False, message="to is required.")
        if not arguments.get("message"):
            return ValidationResult(ok=False, message="message is required.")
        return ValidationResult(ok=True)

    async def execute(self, tool_call: ToolCall, ctx: ToolUseContext) -> ToolResult:
        registry = self.runtime_registry or get_in_process_runtime_registry(ctx.config.runtime_root_path(ctx.cwd))
        target = str(tool_call.arguments["to"])
        runtime = registry.get(target) or registry.get_by_task_id(target)
        if runtime is None:
            return ToolResult(
                tool_use_id=tool_call.tool_use_id,
                tool_name=tool_call.tool_name,
                success=False,
                content=f"Runtime not found: {target}",
                error_code="AG1002",
            )
        current_task = runtime.task_manager.get(runtime.task.task_id) or runtime.task
        if is_terminal_status(current_task.status):
            return ToolResult(
                tool_use_id=tool_call.tool_use_id,
                tool_name=tool_call.tool_name,
                success=False,
                content=f"Task is already terminal: {current_task.status.value}",
                error_code="AG1003",
            )
        definition = getattr(getattr(runtime, "runtime", None), "definition", None)
        message = AgentMessage(
            message_id=f"msg_{uuid.uuid4().hex[:10]}",
            from_agent_id=ctx.metadata.get("agent_id") or "main",
            to_agent_id=getattr(definition, "agent_id", "worker"),
            team_id=ctx.metadata.get("team_id"),
            kind=str(tool_call.arguments.get("kind") or "follow_up"),
            content=str(tool_call.arguments["message"]),
            correlation_id=tool_call.tool_use_id,
        )
        try:
            result = await runtime.send_message(message)
        except AgentTaskError as exc:
            return ToolResult(
                tool_use_id=tool_call.tool_use_id,
                tool_name=tool_call.tool_name,
                success=False,
                content=exc.message,
                error_code=exc.error_code,
            )
        return ToolResult(
            tool_use_id=tool_call.tool_use_id,
            tool_name=tool_call.tool_name,
            success=True,
            content=result.get("final_text", ""),
            data=result,
        )
