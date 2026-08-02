from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ..command_runner import default_shell_kind
from ..config import CCConfig
from ..editing import CodeEditFacade
from ..editing.batch import apply_batch_edit_to_file
from ..editing.rollback import RollbackManager
from ..safety import classify_command_permission, classify_file_permission
from ..safety.file_rules import resolve_under_cwd
from .base import BaseTool, ToolCall, ToolResult, ToolSpec, ValidationResult
from .context import ToolUseContext


class FileEditBatchTool(BaseTool):
    """Transactional multi-edit on one file (single checkpoint / atomic commit)."""

    def __init__(self, config: CCConfig | None = None, facade: CodeEditFacade | None = None) -> None:
        super().__init__(
            ToolSpec(
                name="file_edit_batch",
                description=(
                    "Apply an ordered batch of exact-match edits to one file in a "
                    "single transaction: optional expected_hash check, one checkpoint, "
                    "all edits in memory, optional runtime_command validation, one "
                    "atomic commit. Any edit failure leaves the file unchanged. "
                    "Prefer this over multiple file_edit calls when changing several "
                    "regions of the same file."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "expected_hash": {"type": "string"},
                        "edits": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "address": {"type": "string"},
                                    "old_string": {"type": "string"},
                                    "new_string": {"type": "string"},
                                },
                                "required": ["old_string", "new_string"],
                            },
                        },
                        "runtime_command": {"type": "string"},
                        "runtime_shell": {"type": "string"},
                    },
                    "required": ["file_path", "edits"],
                },
                is_read_only=False,
                needs_confirmation=False,
            )
        )
        self.config = config or CCConfig()
        self._injected_facade = facade
        self._facade_by_cwd: dict[str, CodeEditFacade] = {}
        self.facade = facade

    def _facade_for(self, cwd: str) -> CodeEditFacade:
        if self._injected_facade is not None:
            return self._injected_facade
        key = str(Path(cwd).resolve())
        cached = self._facade_by_cwd.get(key)
        if cached is not None:
            return cached
        checkpoint_root = self.config.runtime_root_path(cwd) / "checkpoints"
        facade = CodeEditFacade(
            rollback_manager=RollbackManager(checkpoint_root),
        )
        self._facade_by_cwd[key] = facade
        self.facade = facade
        return facade

    def validate_input(self, arguments: dict[str, Any]) -> ValidationResult:
        if not arguments.get("file_path"):
            return ValidationResult(ok=False, message="file_path is required.")
        edits = arguments.get("edits")
        if not isinstance(edits, list) or not edits:
            return ValidationResult(ok=False, message="edits must be a non-empty list.")
        for i, edit in enumerate(edits):
            if not isinstance(edit, dict):
                return ValidationResult(ok=False, message=f"edits[{i}] must be an object.")
            if "old_string" not in edit or "new_string" not in edit:
                return ValidationResult(
                    ok=False,
                    message=f"edits[{i}] needs old_string and new_string.",
                )
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
        args = tool_call.arguments or {}
        resolved = str(resolve_under_cwd(args["file_path"], ctx.cwd))
        facade = self._facade_for(ctx.cwd)
        result = await asyncio.to_thread(
            apply_batch_edit_to_file,
            file_path=resolved,
            edits=list(args.get("edits") or []),
            expected_hash=args.get("expected_hash"),
            rollback_manager=facade.rollback_manager,
            validator=facade.validator,
            runtime_command=args.get("runtime_command"),
            runtime_shell=args.get("runtime_shell"),
        )
        content = "File edit batch applied." if result.success else "File edit batch failed."
        if result.diff:
            content = f"{content}\n\n{result.diff}"
        elif result.error_code:
            content = f"{content} ({result.error_code})"
        data = result.to_dict()
        return ToolResult(
            tool_use_id=tool_call.tool_use_id,
            tool_name=tool_call.tool_name,
            success=result.success,
            content=content,
            data=data,
            error_code=result.error_code,
        )
