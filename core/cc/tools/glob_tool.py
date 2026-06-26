from __future__ import annotations

from typing import Any

from ..safety import classify_file_permission
from ..safety.file_rules import resolve_under_cwd
from .base import BaseTool, ToolCall, ToolResult, ToolSpec, ValidationResult
from .context import ToolUseContext
from .grep_tool import _has_parent_traversal


_DEFAULT_MAX_RESULTS = 500


class GlobTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            ToolSpec(
                name="glob",
                description="Find files under the workspace using a glob pattern.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "cwd": {"type": "string"},
                        "max_results": {"type": "integer"},
                    },
                    "required": ["pattern"],
                },
                is_read_only=True,
            )
        )

    def validate_input(self, arguments: dict[str, Any]) -> ValidationResult:
        if not arguments.get("pattern"):
            return ValidationResult(ok=False, message="pattern is required.")
        if _has_parent_traversal(arguments["pattern"]):
            return ValidationResult(ok=False, message="pattern must not contain '..' segments.")
        return ValidationResult(ok=True)

    def check_permissions(self, ctx: ToolUseContext, arguments: dict[str, Any]):
        # Anchor the glob root to the workspace. glob is read-only, so an
        # unconstrained ``cwd`` (e.g. ``../..``) is an information-disclosure
        # surface. Same classifier file_read uses; operation="read" => a root
        # outside the allowed set returns "ask" (blocked by the executor).
        return classify_file_permission(
            file_path=arguments.get("cwd") or ctx.cwd,
            cwd=ctx.cwd,
            mode=ctx.permissions.mode,
            allowed_paths=ctx.permissions.allowed_paths,
            denied_paths=ctx.permissions.denied_paths,
            operation="read",
        )

    async def execute(self, tool_call: ToolCall, ctx: ToolUseContext) -> ToolResult:
        root = resolve_under_cwd(tool_call.arguments.get("cwd") or ctx.cwd, ctx.cwd)
        pattern = str(tool_call.arguments["pattern"])
        max_results = max(1, int(tool_call.arguments.get("max_results") or _DEFAULT_MAX_RESULTS))
        all_matches = sorted(str(path) for path in root.glob(pattern) if path.is_file())
        matches = all_matches[:max_results]
        truncated = len(all_matches) > len(matches)
        content = "\n".join(matches)
        if truncated:
            content = f"{content}\n\n[truncated to {max_results} results out of {len(all_matches)} total matches]"
        return ToolResult(
            tool_use_id=tool_call.tool_use_id,
            tool_name=tool_call.tool_name,
            success=True,
            content=content,
            data={
                "matches": matches,
                "count": len(matches),
                "total_count": len(all_matches),
                "cwd": str(root),
                "max_results": max_results,
                "truncated": truncated,
            },
            truncated=truncated,
        )
