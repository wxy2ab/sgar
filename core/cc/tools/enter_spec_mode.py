from __future__ import annotations

from ..specs import ensure_spec_state
from ..safety.permission_mode import normalize_execute_policy
from .base import BaseTool, ToolCall, ToolResult, ToolSpec
from .context import ToolPermissionSnapshot, ToolUseContext


class EnterSpecModeTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            ToolSpec(
                name="enter_spec_mode",
                description="Switch the session into spec-first mode and initialize spec artifacts.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "task_slug": {"type": "string"},
                        "spec_root": {"type": "string"},
                        "source_text": {"type": "string"},
                        "execute_policy": {"type": "string"},
                    },
                },
                is_read_only=False,
            )
        )

    def is_enabled(self, ctx: ToolUseContext) -> bool:
        return ctx.metadata.get("agent_id") in {None, "", "main"}

    async def execute(self, tool_call: ToolCall, ctx: ToolUseContext) -> ToolResult:
        task_slug = str(tool_call.arguments.get("task_slug", "")).strip() or None
        spec_root = str(tool_call.arguments.get("spec_root", "")).strip() or None
        source_text = str(tool_call.arguments.get("source_text", "")).strip() or None
        raw_policy = tool_call.arguments.get("execute_policy")
        execute_policy = normalize_execute_policy(raw_policy) if raw_policy else None

        def modify(current_ctx: ToolUseContext) -> ToolUseContext:
            next_state = ensure_spec_state(
                current_state=current_ctx.get_app_state(),
                cwd=current_ctx.cwd,
                config=current_ctx.config,
                task_slug=task_slug,
                spec_root=spec_root,
                source_text=source_text,
                enabled=True,
            )
            next_state["pre_spec_mode"] = current_ctx.permissions.mode
            if execute_policy is not None:
                next_state["execute_policy"] = execute_policy
            return current_ctx.with_updates(
                permissions=ToolPermissionSnapshot(
                    mode="spec",
                    allow_dangerous_commands=current_ctx.permissions.allow_dangerous_commands,
                    allowed_paths=list(current_ctx.permissions.allowed_paths),
                    denied_paths=list(current_ctx.permissions.denied_paths),
                ),
                app_state=next_state,
            )

        return ToolResult(
            tool_use_id=tool_call.tool_use_id,
            tool_name=tool_call.tool_name,
            success=True,
            content="Entered spec mode. Focus on tasks, checklist, and spec artifacts before implementation.",
            data={"mode": "spec", "execute_policy": execute_policy or "config_default", "spec_root": spec_root, "task_slug": task_slug},
            context_modifiers=[modify],
        )
