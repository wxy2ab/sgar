from __future__ import annotations

from ..specs import ensure_spec_state, spec_artifacts_ready
from ..safety.permission_mode import normalize_permission_mode
from .base import BaseTool, ToolCall, ToolResult, ToolSpec
from .context import ToolPermissionSnapshot, ToolUseContext


class ExitSpecModeTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            ToolSpec(
                name="exit_spec_mode",
                description="Exit spec mode after required artifacts are ready.",
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
        return bool(ctx.permissions.mode == "spec" or ctx.get_app_state().get("spec_mode"))

    async def execute(self, tool_call: ToolCall, ctx: ToolUseContext) -> ToolResult:
        summary = str(tool_call.arguments.get("summary", "")).strip()
        approved = bool(tool_call.arguments.get("approved", False))
        force = bool(tool_call.arguments.get("force", False))
        requested_next_mode = str(tool_call.arguments.get("next_mode", "")).strip()
        state = ensure_spec_state(
            current_state=ctx.get_app_state(),
            cwd=ctx.cwd,
            config=ctx.config,
            enabled=True,
        )
        ready = spec_artifacts_ready(state)
        policy = str(state.get("execute_policy", "auto_execute"))
        if not ready and not force:
            return ToolResult(
                tool_use_id=tool_call.tool_use_id,
                tool_name=tool_call.tool_name,
                success=False,
                content="Spec artifacts are not ready. Complete tasks/checklist/spec before exiting spec mode.",
                error_code="TL2001",
            )
        if policy == "approval_required" and not (approved or force):
            return ToolResult(
                tool_use_id=tool_call.tool_use_id,
                tool_name=tool_call.tool_name,
                success=False,
                content="Spec mode requires approval before switching into render/edit execution.",
                error_code="TL2002",
            )

        def modify(current_ctx: ToolUseContext) -> ToolUseContext:
            next_state = ensure_spec_state(
                current_state=current_ctx.get_app_state(),
                cwd=current_ctx.cwd,
                config=current_ctx.config,
                enabled=False,
            )
            next_state["spec_mode"] = False
            next_state["spec_phase"] = "render"
            if summary:
                next_state["last_spec_summary"] = summary
            restored_mode = normalize_permission_mode(
                requested_next_mode or str(next_state.pop("pre_spec_mode", "default"))
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
            requested_next_mode or str(state.get("pre_spec_mode", "default"))
        )
        return ToolResult(
            tool_use_id=tool_call.tool_use_id,
            tool_name=tool_call.tool_name,
            success=True,
            content="Exited spec mode.",
            data={"mode": next_mode, "approved": approved, "ready": ready},
            context_modifiers=[modify],
        )
