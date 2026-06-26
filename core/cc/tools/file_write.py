from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ..command_runner import default_shell_kind
from ..config import CCConfig
from ..editing import CodeEditFacade, FileEditRequest
from ..editing.requests import EditResult
from ..editing.rollback import RollbackManager
from ..safety import classify_command_permission, classify_file_permission
from ..safety.file_rules import resolve_under_cwd
from .base import BaseTool, ToolCall, ToolResult, ToolSpec, ValidationResult
from .context import ToolUseContext


class FileWriteTool(BaseTool):
    def __init__(self, config: CCConfig | None = None, facade: CodeEditFacade | None = None) -> None:
        super().__init__(
            ToolSpec(
                name="file_write",
                description="Create or overwrite a file with the provided content, with validation and rollback.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "content": {"type": "string"},
                        "expected_hash": {"type": "string"},
                        "runtime_command": {"type": "string"},
                        "runtime_shell": {"type": "string"},
                    },
                    "required": ["file_path", "content"],
                },
                is_read_only=False,
                needs_confirmation=False,
            )
        )
        self.config = config or CCConfig()
        if facade is not None:
            self.facade = facade
        else:
            checkpoint_root = self.config.runtime_root_path() / "checkpoints"
            self.facade = CodeEditFacade(
                rollback_manager=RollbackManager(checkpoint_root),
            )

    def validate_input(self, arguments: dict[str, Any]) -> ValidationResult:
        if not arguments.get("file_path"):
            return ValidationResult(ok=False, message="file_path is required.")
        if "content" not in arguments:
            return ValidationResult(ok=False, message="content is required.")
        return ValidationResult(ok=True)

    def check_permissions(self, ctx: ToolUseContext, arguments: dict[str, Any]):
        file_decision = classify_file_permission(
            file_path=arguments["file_path"],
            cwd=ctx.cwd,
            mode=ctx.permissions.mode,
            allowed_paths=ctx.permissions.allowed_paths,
            denied_paths=ctx.permissions.denied_paths,
            operation="write",
        )
        if file_decision.status != "allow":
            return file_decision
        runtime_cmd = arguments.get("runtime_command")
        if runtime_cmd:
            cmd_decision = classify_command_permission(
                command=str(runtime_cmd),
                shell_kind=str(arguments.get("runtime_shell") or default_shell_kind()),
                cwd=ctx.cwd,
                target_cwd=str(resolve_under_cwd(arguments["file_path"], ctx.cwd).parent),
                mode=ctx.permissions.mode,
                allowed_paths=ctx.permissions.allowed_paths,
                allow_dangerous_commands=ctx.permissions.allow_dangerous_commands,
            )
            if cmd_decision.status != "allow":
                return cmd_decision
        return file_decision

    async def execute(self, tool_call: ToolCall, ctx: ToolUseContext) -> ToolResult:
        request = FileEditRequest(
            file_path=str(resolve_under_cwd(tool_call.arguments["file_path"], ctx.cwd)),
            new_string=str(tool_call.arguments.get("content", "")),
            create_if_missing=True,
            expected_hash=tool_call.arguments.get("expected_hash"),
            runtime_command=tool_call.arguments.get("runtime_command"),
            runtime_shell=tool_call.arguments.get("runtime_shell"),
            metadata={"prompt_language": ctx.prompt_language, "environment": ctx.environment},
        )
        result = await asyncio.to_thread(self.facade.apply_precise_edit, request)
        return self._to_tool_result(tool_call, result)

    def _to_tool_result(self, tool_call: ToolCall, result: EditResult) -> ToolResult:
        content = "File write applied." if result.success else "File write failed."
        if result.preview is not None and result.preview.diff:
            content = f"{content}\n\n{result.preview.diff}"
        return ToolResult(
            tool_use_id=tool_call.tool_use_id,
            tool_name=tool_call.tool_name,
            success=result.success,
            content=content,
            data=result.to_dict(),
            error_code=result.error_code,
        )
