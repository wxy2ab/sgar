from __future__ import annotations

from typing import Any

from ..errors import AgentTaskError
from ..agents.runtime_registry import InProcessRuntimeRegistry, get_in_process_runtime_registry
from ..agents.task_model import is_terminal_status
from .base import BaseTool, ToolCall, ToolResult, ToolSpec, ValidationResult
from .context import ToolUseContext


class TaskStopTool(BaseTool):
    def __init__(self, runtime_registry: InProcessRuntimeRegistry | None = None) -> None:
        super().__init__(
            ToolSpec(
                name="task_stop",
                description="Stop a running agent task or runtime.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "target": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["target"],
                },
            )
        )
        self.runtime_registry = runtime_registry

    def validate_input(self, arguments: dict[str, Any]) -> ValidationResult:
        if not arguments.get("target"):
            return ValidationResult(ok=False, message="target is required.")
        return ValidationResult(ok=True)

    async def execute(self, tool_call: ToolCall, ctx: ToolUseContext) -> ToolResult:
        registry = self.runtime_registry or get_in_process_runtime_registry(ctx.config.runtime_root_path(ctx.cwd))
        target = str(tool_call.arguments["target"])
        runtime = registry.get(target) or registry.get_by_task_id(target)
        if runtime is None:
            return ToolResult(
                tool_use_id=tool_call.tool_use_id,
                tool_name=tool_call.tool_name,
                success=False,
                content=f"Runtime not found: {target}",
                error_code="AG1002",
            )
        current_status = runtime.task_manager.get(runtime.task.task_id) or runtime.task
        if is_terminal_status(current_status.status):
            status = await runtime.collect_status()
            return ToolResult(
                tool_use_id=tool_call.tool_use_id,
                tool_name=tool_call.tool_name,
                success=True,
                content=f"Task already terminal: {current_status.status.value}",
                data=status,
            )
        background_task = registry.get_background_task(runtime.task.runtime_id)
        if background_task is not None and not background_task.done():
            background_task.cancel()
        try:
            await runtime.stop(str(tool_call.arguments.get("reason") or "Stopped by task_stop tool."))
        except AgentTaskError as exc:
            return ToolResult(
                tool_use_id=tool_call.tool_use_id,
                tool_name=tool_call.tool_name,
                success=False,
                content=exc.message,
                error_code=exc.error_code,
            )
        status = await runtime.collect_status()
        return ToolResult(
            tool_use_id=tool_call.tool_use_id,
            tool_name=tool_call.tool_name,
            success=True,
            content=f"Stopped {target}",
            data=status,
        )
