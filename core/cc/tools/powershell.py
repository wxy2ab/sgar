"""PowerShellTool — back-compat alias for the unified ShellTool.

The standalone PowerShell tool collapsed into ``ShellTool(kind=...)``. This
module remains for one deprecation window: in-process Python callers and
existing tests that look up ``registry.get("powershell")`` continue to
resolve a working tool. The LLM no longer sees this name in its tool
schema (``is_enabled`` returns False).

Every call is routed to ``ShellTool.execute`` after injecting
``kind="powershell"`` into the arguments, so behavior is bit-identical to
the original ``PowerShellTool``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .base import BaseTool, ToolCall, ToolResult, ToolSpec, ValidationResult
from .context import ToolUseContext
from .shell import ShellTool


class PowerShellTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            ToolSpec(
                name="powershell",
                description="Run a PowerShell command in the current working directory.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "cwd": {"type": "string"},
                        "timeout_ms": {"type": "integer"},
                    },
                    "required": ["command"],
                },
                is_read_only=False,
                needs_confirmation=True,
            )
        )
        self._impl = ShellTool()

    def is_enabled(self, ctx: Any) -> bool:
        # Hidden from the LLM-facing schema; unified ShellTool replaces it.
        del ctx
        return False

    def validate_input(self, arguments: dict[str, Any]) -> ValidationResult:
        return self._impl.validate_input({**arguments, "kind": "powershell"})

    def check_permissions(self, ctx: ToolUseContext, arguments: dict[str, Any]):
        return self._impl.check_permissions(ctx, {**arguments, "kind": "powershell"})

    async def execute(self, tool_call: ToolCall, ctx: ToolUseContext) -> ToolResult:
        forwarded = replace(
            tool_call,
            arguments={**tool_call.arguments, "kind": "powershell"},
        )
        result = await self._impl.execute(forwarded, ctx)
        # Preserve the original wire name on the result so audit logs and
        # any name-keyed consumers don't see "shell" when the caller asked
        # for "powershell".
        return replace(result, tool_name=tool_call.tool_name)
