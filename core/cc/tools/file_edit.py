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


class FileEditTool(BaseTool):
    def __init__(self, config: CCConfig | None = None, facade: CodeEditFacade | None = None) -> None:
        super().__init__(
            ToolSpec(
                name="file_edit",
                description="Modify file contents in place using exact-match replacement with validation and rollback.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "old_string": {"type": "string"},
                        "new_string": {"type": "string"},
                        "replace_all": {"type": "boolean"},
                        "create_if_missing": {"type": "boolean"},
                        "expected_hash": {"type": "string"},
                        "runtime_command": {"type": "string"},
                        "runtime_shell": {"type": "string"},
                    },
                    "required": ["file_path", "new_string"],
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
        if "new_string" not in arguments:
            return ValidationResult(ok=False, message="new_string is required.")
        return ValidationResult(ok=True)

    def check_permissions(self, ctx: ToolUseContext, arguments: dict[str, Any]):
        file_decision = classify_file_permission(
            file_path=arguments["file_path"],
            cwd=ctx.cwd,
            mode=ctx.permissions.mode,
            allowed_paths=ctx.permissions.allowed_paths,
            denied_paths=ctx.permissions.denied_paths,
            operation="edit",
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
            old_string=str(tool_call.arguments.get("old_string", "")),
            new_string=str(tool_call.arguments.get("new_string", "")),
            replace_all=bool(tool_call.arguments.get("replace_all", False)),
            create_if_missing=bool(tool_call.arguments.get("create_if_missing", False)),
            expected_hash=tool_call.arguments.get("expected_hash"),
            runtime_command=tool_call.arguments.get("runtime_command"),
            runtime_shell=tool_call.arguments.get("runtime_shell"),
            metadata={"prompt_language": ctx.prompt_language, "environment": ctx.environment},
        )
        result = await asyncio.to_thread(self.facade.apply_precise_edit, request)
        return self._to_tool_result(tool_call, result)

    def build_patch_preview(self, arguments: dict[str, Any]) -> dict[str, Any]:
        request = FileEditRequest(
            file_path=str(Path(arguments["file_path"]).resolve()),
            old_string=str(arguments.get("old_string", "")),
            new_string=str(arguments.get("new_string", "")),
            replace_all=bool(arguments.get("replace_all", False)),
            create_if_missing=bool(arguments.get("create_if_missing", False)),
            expected_hash=arguments.get("expected_hash"),
            runtime_command=arguments.get("runtime_command"),
            runtime_shell=arguments.get("runtime_shell"),
        )
        return self.facade.preview_edit(request).to_dict()

    def _to_tool_result(self, tool_call: ToolCall, result: EditResult) -> ToolResult:
        content = "File edit applied." if result.success else "File edit failed."
        if result.preview is not None and result.preview.diff:
            content = f"{content}\n\n{result.preview.diff}"
        data = result.to_dict()
        return ToolResult(
            tool_use_id=tool_call.tool_use_id,
            tool_name=tool_call.tool_name,
            success=result.success,
            content=content,
            data=data,
            error_code=result.error_code,
        )
