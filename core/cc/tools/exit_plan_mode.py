from __future__ import annotations

from ..plan import ensure_plan_state, plan_artifact_ready
from ..safety.permission_mode import normalize_permission_mode
from .base import BaseTool, ToolCall, ToolResult, ToolSpec
from .context import ToolPermissionSnapshot, ToolUseContext


class ExitPlanModeTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            ToolSpec(
                name="exit_plan_mode",
                description="Exit plan mode after the plan artifact is ready.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "approved": {"type": "boolean"},
                        "force": {"type": "boolean"},
                        "next_mode": {"type": "string"},
                    },
                },
                is_read_only=False,
            )
        )

    def is_enabled(self, ctx: ToolUseContext) -> bool:
        return bool(ctx.permissions.mode == "plan" or ctx.get_app_state().get("plan_mode"))

    async def execute(self, tool_call: ToolCall, ctx: ToolUseContext) -> ToolResult:
        summary = str(tool_call.arguments.get("summary", "")).strip()
        approved = bool(tool_call.arguments.get("approved", False))
        force = bool(tool_call.arguments.get("force", False))
        requested_next_mode = str(tool_call.arguments.get("next_mode", "")).strip()
        state = ensure_plan_state(
            current_state=ctx.get_app_state(),
            cwd=ctx.cwd,
            config=ctx.config,
            enabled=True,
        )
        ready = plan_artifact_ready(state)
        policy = str(state.get("execute_policy", "auto_execute"))
        if not ready and not force:
            return ToolResult(
                tool_use_id=tool_call.tool_use_id,
                tool_name=tool_call.tool_name,
                success=False,
                content="Plan artifacts are not ready. Write both plan.md and tasks.md using plan_artifact_write before exiting plan mode.",
                error_code="TL3001",
            )
        if policy == "approval_required" and not (approved or force):
            return ToolResult(
                tool_use_id=tool_call.tool_use_id,
                tool_name=tool_call.tool_name,
                success=False,
                content="Plan mode requires approval before switching into code implementation.",
                error_code="TL3002",
            )

        def modify(current_ctx: ToolUseContext) -> ToolUseContext:
            next_state = ensure_plan_state(
                current_state=current_ctx.get_app_state(),
                cwd=current_ctx.cwd,
                config=current_ctx.config,
                enabled=False,
            )
            next_state["plan_mode"] = False
            next_state["plan_phase"] = "implementation"
            if summary:
                next_state["last_plan_summary"] = summary
            restored_mode = normalize_permission_mode(
                requested_next_mode or str(next_state.pop("pre_plan_mode", "default"))
            )
            return current_ctx.with_updates(
                permissions=ToolPermissionSnapshot(
                    mode=restored_mode,
                    allow_dangerous_commands=current_ctx.permissions.allow_dangerous_commands,
                    allowed_paths=list(current_ctx.permissions.allowed_paths),
                    denied_paths=list(current_ctx.permissions.denied_paths),
                ),
                app_state=next_state,
            )

        next_mode = normalize_permission_mode(
            requested_next_mode or str(state.get("pre_plan_mode", "default"))
        )
        return ToolResult(
            tool_use_id=tool_call.tool_use_id,
            tool_name=tool_call.tool_name,
            success=True,
            content="Exited plan mode. Proceed with code implementation.",
            data={"mode": next_mode, "approved": approved, "ready": ready},
            context_modifiers=[modify],
        )
