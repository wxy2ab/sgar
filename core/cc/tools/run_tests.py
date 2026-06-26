"""RunTestsTool — a first-class, exit-code-gated verification tool.

Unlike the generic ``shell`` tool (which returns raw stdout/stderr the model
must interpret), ``run_tests`` runs the project's verification command (tests /
lint / typecheck / build), gates on the exit code (0 = pass), and returns a
typed verdict in ``ToolResult.data`` (``passed`` / ``exit_code`` /
``unrunnable``). ``ToolResult.success`` reflects the pass/fail result, so a red
suite is unambiguous to the loop and to downstream consumers.

Default-OFF: ``is_enabled`` returns False unless the operator sets
``run_tests_tool_enabled`` on the config, so when off the model's exported tool
schema is byte-identical to before this tool existed.

The actual execution / exit-code gating is delegated to
``core.cc.command_runner.run_check_command_async`` so this tool and the
interactive loop's post-edit auto-verify step share ONE primitive (and so this
``cc`` tool never has to import ``core.ccx``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BaseTool, ToolCall, ToolResult, ToolSpec, ValidationResult
from .context import ToolUseContext
from ..command_runner import run_check_command_async
from ..safety import classify_command_permission


_DEFAULT_VERIFY_TIMEOUT_MS = 120_000


class RunTestsTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            ToolSpec(
                name="run_tests",
                description=(
                    "Run the project's verification command (tests / lint / "
                    "typecheck / build) and report a structured pass/fail "
                    "verdict. Exit code 0 means PASS; any non-zero means FAIL. "
                    "Use this to verify your edits before declaring a task "
                    "done — a green result is the evidence that a change works. "
                    "The command runs via shlex.split (NO shell), so wrap a "
                    "pipeline or redirection explicitly as sh -c \"...\"."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": (
                                "Verification command, e.g. 'pytest -q', "
                                "'ruff check .', or 'mypy pkg'."
                            ),
                        },
                        "cwd": {
                            "type": "string",
                            "description": "Working directory (defaults to the session cwd).",
                        },
                        "timeout_ms": {"type": "integer"},
                    },
                    "required": ["command"],
                },
                is_read_only=False,
                needs_confirmation=True,
            )
        )

    def is_enabled(self, ctx: Any) -> bool:
        # Default-OFF. Hidden from the LLM-facing schema unless the operator
        # opts in via ``run_tests_tool_enabled``; when off the exported tool
        # schema is byte-identical to before this tool existed.
        config = getattr(ctx, "config", None)
        return bool(getattr(config, "run_tests_tool_enabled", False))

    def validate_input(self, arguments: dict[str, Any]) -> ValidationResult:
        if not arguments.get("command"):
            return ValidationResult(ok=False, message="command is required.")
        return ValidationResult(ok=True)

    def check_permissions(self, ctx: ToolUseContext, arguments: dict[str, Any]):
        # A verification command is still an arbitrary command — route it
        # through the SAME permission classifier as the shell tool so this tool
        # can't become a way to bypass command permissions.
        target_cwd = str(Path(arguments.get("cwd") or ctx.cwd).resolve())
        return classify_command_permission(
            command=str(arguments.get("command") or ""),
            shell_kind="shell",
            cwd=ctx.cwd,
            target_cwd=target_cwd,
            mode=ctx.permissions.mode,
            allowed_paths=ctx.permissions.allowed_paths,
            allow_dangerous_commands=ctx.permissions.allow_dangerous_commands,
        )

    async def execute(self, tool_call: ToolCall, ctx: ToolUseContext) -> ToolResult:
        cwd = str(Path(tool_call.arguments.get("cwd") or ctx.cwd).resolve())
        timeout_ms = int(
            tool_call.arguments.get("timeout_ms") or _DEFAULT_VERIFY_TIMEOUT_MS
        )
        verdict = await run_check_command_async(
            command=str(tool_call.arguments["command"]),
            cwd=cwd,
            timeout_ms=timeout_ms,
        )
        error_code: str | None = None
        if not verdict.passed:
            error_code = "TL1005" if verdict.timed_out else "TL1007"
        return ToolResult(
            tool_use_id=tool_call.tool_use_id,
            tool_name=tool_call.tool_name,
            success=verdict.passed,
            content=verdict.evidence_line(),
            data=verdict.to_dict(),
            error_code=error_code,
        )
