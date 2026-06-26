from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any
import uuid

from .base import BaseTool, ToolCall, ToolResult, ToolSpec, ValidationResult
from .context import ToolPermissionSnapshot, ToolUseContext


def _worktree_root(base_cwd: Path) -> Path:
    return base_cwd.parent / f".cc_worktrees_{base_cwd.name}"


def _copy_workspace(src: Path, dst: Path) -> None:
    ignore = shutil.ignore_patterns("__pycache__", ".git", ".cc", ".pytest_cache")
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=ignore)


class EnterWorktreeTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            ToolSpec(
                name="enter_worktree",
                description="Create a lightweight isolated workspace copy and switch the session cwd into it.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                    },
                },
                is_read_only=False,
            )
        )

    def validate_input(self, arguments: dict[str, Any]) -> ValidationResult:
        name = arguments.get("name")
        if name is not None:
            if not isinstance(name, str) or not name.strip():
                return ValidationResult(ok=False, message="name must be a non-empty string.")
            if ".." in name or "/" in name or "\\" in name:
                return ValidationResult(ok=False, message="name must not contain path separators or '..'.")
        return ValidationResult(ok=True)

    def is_enabled(self, ctx: ToolUseContext) -> bool:
        return not bool(ctx.get_app_state().get("worktree_active"))

    async def execute(self, tool_call: ToolCall, ctx: ToolUseContext) -> ToolResult:
        state = ctx.get_app_state()
        if state.get("worktree_active"):
            return ToolResult(
                tool_use_id=tool_call.tool_use_id,
                tool_name=tool_call.tool_name,
                success=False,
                content="Already in a worktree session.",
                error_code="AG1004",
            )
        src = Path(ctx.cwd).resolve()
        slug = str(tool_call.arguments.get("name") or f"wt_{uuid.uuid4().hex[:8]}")
        dst = _worktree_root(src) / slug
        _copy_workspace(src, dst)

        def modify(current_ctx: ToolUseContext) -> ToolUseContext:
            next_state = current_ctx.get_app_state()
            next_state["worktree_active"] = True
            next_state["original_cwd"] = current_ctx.cwd
            next_state["worktree_path"] = str(dst)
            next_state["pre_worktree_allowed_paths"] = list(current_ctx.permissions.allowed_paths)
            permissions = ToolPermissionSnapshot(
                mode=current_ctx.permissions.mode,
                allow_dangerous_commands=current_ctx.permissions.allow_dangerous_commands,
                allowed_paths=list(dict.fromkeys([str(dst), *current_ctx.permissions.allowed_paths])),
                denied_paths=list(current_ctx.permissions.denied_paths),
            )
            return current_ctx.with_updates(
                cwd=str(dst),
                permissions=permissions,
                app_state=next_state,
            )

        return ToolResult(
            tool_use_id=tool_call.tool_use_id,
            tool_name=tool_call.tool_name,
            success=True,
            content=f"Entered worktree at {dst}",
            data={"worktree_path": str(dst), "name": slug},
            context_modifiers=[modify],
        )


class ExitWorktreeTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            ToolSpec(
                name="exit_worktree",
                description="Leave the active worktree and optionally remove its directory.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                    },
                },
                is_read_only=False,
            )
        )

    def is_enabled(self, ctx: ToolUseContext) -> bool:
        return bool(ctx.get_app_state().get("worktree_active"))

    async def execute(self, tool_call: ToolCall, ctx: ToolUseContext) -> ToolResult:
        state = ctx.get_app_state()
        if not state.get("worktree_active"):
            return ToolResult(
                tool_use_id=tool_call.tool_use_id,
                tool_name=tool_call.tool_name,
                success=False,
                content="No active worktree session.",
                error_code="AG1005",
            )
        action = str(tool_call.arguments.get("action") or "keep").lower()
        original_cwd = str(state.get("original_cwd") or ctx.cwd)
        worktree_path = str(state.get("worktree_path") or ctx.cwd)
        if action == "remove":
            shutil.rmtree(worktree_path, ignore_errors=True)

        def modify(current_ctx: ToolUseContext) -> ToolUseContext:
            next_state = current_ctx.get_app_state()
            next_state["worktree_active"] = False
            next_state["original_cwd"] = original_cwd
            next_state["worktree_path"] = None
            restored_allowed = next_state.pop("pre_worktree_allowed_paths", [original_cwd])
            permissions = ToolPermissionSnapshot(
                mode=current_ctx.permissions.mode,
                allow_dangerous_commands=current_ctx.permissions.allow_dangerous_commands,
                allowed_paths=restored_allowed,
                denied_paths=list(current_ctx.permissions.denied_paths),
            )
            return current_ctx.with_updates(
                cwd=original_cwd,
                permissions=permissions,
                app_state=next_state,
            )

        return ToolResult(
            tool_use_id=tool_call.tool_use_id,
            tool_name=tool_call.tool_name,
            success=True,
            content=f"Exited worktree and returned to {original_cwd}",
            data={"original_cwd": original_cwd, "worktree_path": worktree_path, "action": action},
            context_modifiers=[modify],
        )
