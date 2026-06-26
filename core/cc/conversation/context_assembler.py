from __future__ import annotations

from pathlib import Path
from typing import Any

from ..providers import Environment, default_environment
from ..tools.context import ToolPermissionSnapshot, ToolUseContext
from .session import QuerySession


class ContextAssembler:
    def __init__(self, *, environment: Environment | None = None) -> None:
        self.environment = environment or default_environment()

    def _normalize_paths(self, paths: list[str] | None) -> list[str]:
        normalized: list[str] = []
        for path_value in paths or []:
            resolved = str(Path(path_value).resolve())
            if resolved not in normalized:
                normalized.append(resolved)
        return normalized

    def build_allowed_paths(self, session: QuerySession) -> list[str]:
        session_state = dict(session.metadata.state)
        allowed_paths = [str(Path(session.cwd).resolve())]
        for key in ("team_shared_allowed_paths", "allowed_paths"):
            for path_value in self._normalize_paths(session_state.get(key)):
                if path_value not in allowed_paths:
                    allowed_paths.append(path_value)
        return allowed_paths

    def build_denied_paths(self, session: QuerySession) -> list[str]:
        session_state = dict(session.metadata.state)
        return self._normalize_paths(session_state.get("denied_paths"))

    def build_tool_context(self, *, session: QuerySession, turn_id: str) -> ToolUseContext:
        session_state = dict(session.metadata.state)
        if session_state.get("plan_mode"):
            effective_mode = "plan"
        elif session_state.get("spec_mode"):
            effective_mode = "spec"
        else:
            effective_mode = session.permission_mode
        return ToolUseContext(
            session_id=session.session_id,
            turn_id=turn_id,
            cwd=str(Path(session.cwd).resolve()),
            prompt_language=session.prompt_language,
            config=session.config,
            permissions=ToolPermissionSnapshot(
                mode=effective_mode,
                allowed_paths=self.build_allowed_paths(session),
                denied_paths=self.build_denied_paths(session),
            ),
            app_state=session_state,
            metadata={
                "session": session,
                "agent_id": session.metadata.agent_id,
                "team_id": session.metadata.team_id,
            },
            environment=self.environment,
        )

    def build_prompt_context(
        self,
        *,
        session: QuerySession,
        tool_ctx: ToolUseContext,
        enabled_tools: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        session_state = dict(session.metadata.state)
        context = {
            "cwd": tool_ctx.cwd,
            "session_id": session.session_id,
            "enabled_tools": list(enabled_tools or []),
            "permission_mode": session.permission_mode,
            "permission_snapshot": tool_ctx.to_permission_context_snapshot(),
            "team_id": session.metadata.team_id,
            "team_shared_context": session_state.get("team_shared_context"),
            "team_shared_allowed_paths": session_state.get("team_shared_allowed_paths", []),
            "agent_mode": session.agent_mode,
            "execute_policy": session_state.get("execute_policy") or session.config.execute_policy,
            "plan_mode": session_state.get("plan_mode", session.agent_mode == "plan") is True,
            "plan_phase": session_state.get("plan_phase"),
            "plan_root": session_state.get("plan_root"),
            "plan_artifacts": session_state.get("plan_artifacts"),
            "plan_artifact_status": session_state.get("plan_artifact_status"),
            "plan_ready": session_state.get("plan_ready") is True,
            "spec_mode": session_state.get("spec_mode", session.agent_mode == "spec") is True,
            "spec_phase": session_state.get("spec_phase"),
            "spec_root": session_state.get("spec_root"),
            "spec_artifacts": session_state.get("spec_artifacts"),
            "spec_artifact_status": session_state.get("spec_artifact_status"),
            "render_ready": session_state.get("render_ready") is True,
            "agent_collaboration_strategy": (session_state.get("system_prompt_context") or {}).get(
                "agent_collaboration_strategy",
            ),
            "agent_collaboration_required": bool(
                (session_state.get("system_prompt_context") or {}).get("agent_collaboration_required"),
            ),
            "agent_collaboration_pattern": (session_state.get("system_prompt_context") or {}).get(
                "agent_collaboration_pattern",
            ),
            "agent_collaboration_completed": bool(
                (session_state.get("system_prompt_context") or {}).get("agent_collaboration_completed"),
            ),
            "agent_collaboration_count": int(
                (session_state.get("system_prompt_context") or {}).get("agent_collaboration_count", 0) or 0,
            ),
            "mode_strategy": (session_state.get("system_prompt_context") or {}).get("mode_strategy"),
            "repository_outline_enabled": bool(
                (session_state.get("system_prompt_context") or {}).get("repository_outline_enabled"),
            ),
            "todo_count": len(session_state.get("todos", [])),
            "todos_pending": [
                t.get("content", "") for t in session_state.get("todos", [])
                if str(t.get("status", "pending")).lower() not in ("completed", "cancelled")
            ],
            "todos_completed_count": sum(
                1 for t in session_state.get("todos", [])
                if str(t.get("status", "")).lower() == "completed"
            ),
            "latest_compact_summary": session_state.get("latest_compact_summary"),
        }
        if "memory_provider" in session_state:
            context["memory_provider"] = session_state.get("memory_provider")
        if "memory_context" in session_state:
            context["memory_context"] = session_state.get("memory_context")
            room_summaries = (session_state.get("memory_context") or {}).get("room_summaries")
            if room_summaries:
                context["memory_room_summaries"] = room_summaries
                context["memory_prompt_summary_max_chars"] = session.config.memory_prompt_summary_max_chars
        if "memory_status" in session_state:
            context["memory_status"] = session_state.get("memory_status")
        if extra:
            context.update(extra)
        return context

    def apply_tool_context(self, *, session: QuerySession, tool_ctx: ToolUseContext) -> None:
        session.cwd = tool_ctx.cwd
        session.metadata.state = tool_ctx.get_app_state()
