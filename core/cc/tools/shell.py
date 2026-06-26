from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BaseTool, ToolCall, ToolResult, ToolSpec, ValidationResult
from ..safety import classify_command_permission
from ..command_runner import default_shell_kind, execute_command, execute_command_async
from .context import ToolUseContext


_DEFAULT_COMMAND_TIMEOUT_MS = 120_000
_VALID_KINDS = ("auto", "shell", "powershell")


def _resolve_kind(raw: Any) -> str:
    kind = str(raw or "auto").lower()
    if kind == "auto":
        return default_shell_kind()
    return kind


class ShellTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            ToolSpec(
                name="shell",
                description=(
                    "Run a shell command in the current working directory. "
                    "Use kind='auto' (default) to pick the host's native shell — "
                    "bash/zsh on Linux/macOS, PowerShell on Windows. "
                    "Explicit kind='shell' or kind='powershell' overrides the default."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "kind": {
                            "type": "string",
                            "enum": list(_VALID_KINDS),
                            "default": "auto",
                            "description": (
                                "Shell flavor. 'auto' picks the host default; "
                                "'shell' forces bash/zsh; 'powershell' forces pwsh."
                            ),
                        },
                        "cwd": {"type": "string"},
                        "timeout_ms": {"type": "integer"},
                    },
                    "required": ["command"],
                },
                is_read_only=False,
                needs_confirmation=True,
            )
        )

    def validate_input(self, arguments: dict[str, Any]) -> ValidationResult:
        if not arguments.get("command"):
            return ValidationResult(ok=False, message="command is required.")
        raw_kind = arguments.get("kind")
        if raw_kind is not None and str(raw_kind).lower() not in _VALID_KINDS:
            return ValidationResult(
                ok=False,
                message=f"kind must be one of {_VALID_KINDS}, got {raw_kind!r}.",
            )
        return ValidationResult(ok=True)

    def check_permissions(self, ctx: ToolUseContext, arguments: dict[str, Any]):
        target_cwd = str(Path(arguments.get("cwd") or ctx.cwd).resolve())
        resolved_kind = _resolve_kind(arguments.get("kind"))
        return classify_command_permission(
            command=str(arguments["command"]),
            shell_kind=resolved_kind,
            cwd=ctx.cwd,
            target_cwd=target_cwd,
            mode=ctx.permissions.mode,
            allowed_paths=ctx.permissions.allowed_paths,
            allow_dangerous_commands=ctx.permissions.allow_dangerous_commands,
        )

    async def execute(self, tool_call: ToolCall, ctx: ToolUseContext) -> ToolResult:
        cwd = str(Path(tool_call.arguments.get("cwd") or ctx.cwd).resolve())
        effective_timeout = int(tool_call.arguments.get("timeout_ms") or _DEFAULT_COMMAND_TIMEOUT_MS)
        resolved_kind = _resolve_kind(tool_call.arguments.get("kind"))
        shell = ctx.get_shell()
        if shell is not None:
            result = await shell.exec(
                str(tool_call.arguments["command"]),
                cwd=cwd,
                timeout_ms=effective_timeout,
                shell_kind=resolved_kind,
            )
        else:
            result = await execute_command_async(
                command=str(tool_call.arguments["command"]),
                cwd=cwd,
                shell_kind=resolved_kind,
                timeout_ms=effective_timeout,
            )
        content_parts = []
        if result.stdout.strip():
            content_parts.append(result.stdout.strip())
        if result.stderr.strip():
            content_parts.append(result.stderr.strip())
        content = "\n\n".join(content_parts)
        return ToolResult(
            tool_use_id=tool_call.tool_use_id,
            tool_name=tool_call.tool_name,
            success=result.success,
            content=content,
            data=result.to_dict(),
            error_code="TL1005" if result.was_timeout else (None if result.success else "TL1007"),
        )
