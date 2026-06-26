from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..plan import PLAN_ARTIFACT_NAMES, ensure_plan_state
from .base import BaseTool, ToolCall, ToolResult, ToolSpec, ValidationResult
from .context import ToolUseContext


def _count_task_markers(text: str) -> tuple[int, int]:
    """Return (completed_count, total_count) from markdown checkbox content."""
    markers = re.findall(r"^\s*-\s+\[([ xX])\]", text, re.MULTILINE)
    completed = sum(1 for m in markers if m.strip().lower() == "x")
    return (completed, len(markers))


class PlanArtifactWriteTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            ToolSpec(
                name="plan_artifact_write",
                description="Write a plan artifact (plan.md or tasks.md) under .cc/plans/<task-slug>/.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "artifact": {"type": "string"},
                        "content": {"type": "string"},
                        "status": {"type": "string"},
                        "merge_mode": {"type": "string"},
                        "section_heading": {"type": "string"},
                    },
                    "required": ["content"],
                },
                is_read_only=False,
            )
        )

    def is_enabled(self, ctx: ToolUseContext) -> bool:
        app_state = ctx.get_app_state()
        if ctx.permissions.mode == "plan" or app_state.get("plan_mode"):
            return True
        return app_state.get("plan_phase") == "implementation"

    def validate_input(self, arguments: dict[str, Any]) -> ValidationResult:
        if "content" not in arguments:
            return ValidationResult(ok=False, message="content is required.")
        artifact = str(arguments.get("artifact", "plan")).strip().lower() or "plan"
        if artifact not in PLAN_ARTIFACT_NAMES:
            return ValidationResult(
                ok=False,
                message=f"artifact must be one of: {', '.join(PLAN_ARTIFACT_NAMES)}.",
            )
        merge_mode = str(arguments.get("merge_mode", "replace")).strip().lower() or "replace"
        if merge_mode not in {"replace", "append", "prepend", "replace_section"}:
            return ValidationResult(
                ok=False,
                message="merge_mode must be replace, append, prepend, or replace_section.",
            )
        if merge_mode == "replace_section" and not str(arguments.get("section_heading", "")).strip():
            return ValidationResult(ok=False, message="section_heading is required for replace_section.")
        return ValidationResult(ok=True)

    async def execute(self, tool_call: ToolCall, ctx: ToolUseContext) -> ToolResult:
        artifact = str(tool_call.arguments.get("artifact", "plan")).strip().lower() or "plan"
        content = str(tool_call.arguments.get("content", ""))
        status = str(tool_call.arguments.get("status", "ready")).strip().lower() or "ready"
        merge_mode = str(tool_call.arguments.get("merge_mode", "replace")).strip().lower() or "replace"
        section_heading = str(tool_call.arguments.get("section_heading", "")).strip()

        is_impl = ctx.get_app_state().get("plan_phase") == "implementation"
        state = ensure_plan_state(
            current_state=ctx.get_app_state(),
            cwd=ctx.cwd,
            config=ctx.config,
            enabled=not is_impl,
        )
        target_path = Path(state["plan_artifacts"][artifact]).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        existing = target_path.read_text(encoding="utf-8") if target_path.exists() else ""
        merged = self._merge_content(
            existing,
            content,
            merge_mode=merge_mode,
            section_heading=section_heading,
        )

        if is_impl and artifact == "tasks" and existing.strip():
            rejection = self._validate_impl_tasks_update(existing, merged)
            if rejection is not None:
                return ToolResult(
                    tool_use_id=tool_call.tool_use_id,
                    tool_name=tool_call.tool_name,
                    success=False,
                    content=rejection,
                    error_code="TL3010",
                )

        target_path.write_text(merged, encoding="utf-8")

        def modify(current_ctx: ToolUseContext) -> ToolUseContext:
            next_state = ensure_plan_state(
                current_state=current_ctx.get_app_state(),
                cwd=current_ctx.cwd,
                config=current_ctx.config,
                enabled=not is_impl,
            )
            if not is_impl:
                next_state["plan_phase"] = "planning"
            statuses = dict(next_state.get("plan_artifact_status") or {})
            statuses[artifact] = status
            next_state["plan_artifact_status"] = statuses
            next_state["plan_ready"] = all(
                str(statuses.get(name, "pending")).lower() in {"ready", "completed"}
                for name in PLAN_ARTIFACT_NAMES
            )
            return current_ctx.set_app_state(next_state)

        return ToolResult(
            tool_use_id=tool_call.tool_use_id,
            tool_name=tool_call.tool_name,
            success=True,
            content=f"Updated {artifact} artifact at {target_path}.",
            data={
                "artifact": artifact,
                "path": str(target_path),
                "status": status,
                "merge_mode": merge_mode,
                "section_heading": section_heading,
            },
            context_modifiers=[modify],
        )

    @staticmethod
    def _validate_impl_tasks_update(existing: str, merged: str) -> str | None:
        """Validate tasks.md update during implementation phase.

        Returns an error message if the update is invalid, None if acceptable.
        """
        prev_completed, prev_total = _count_task_markers(existing)
        curr_completed, curr_total = _count_task_markers(merged)
        if prev_total == 0:
            return None
        if curr_completed < prev_completed:
            return (
                f"Cannot un-complete tasks during implementation. "
                f"Previously {prev_completed} completed, now {curr_completed}. "
                f"Only mark additional tasks as [x], do not rewrite the entire list."
            )
        if curr_total > prev_total * 2:
            return (
                f"Task list grew too much ({prev_total} -> {curr_total}). "
                f"Do not replace the task list with unrelated content. "
                f"Only update individual task checkboxes from [ ] to [x]."
            )
        return None

    def _merge_content(
        self,
        existing: str,
        incoming: str,
        *,
        merge_mode: str,
        section_heading: str = "",
    ) -> str:
        if merge_mode == "replace" or not existing:
            return incoming
        if merge_mode == "append":
            separator = "\n\n" if existing.strip() and incoming.strip() else ""
            return f"{existing.rstrip()}{separator}{incoming.lstrip()}"
        if merge_mode == "prepend":
            separator = "\n\n" if existing.strip() and incoming.strip() else ""
            return f"{incoming.rstrip()}{separator}{existing.lstrip()}"
        if merge_mode == "replace_section":
            return self._replace_markdown_section(existing, incoming, section_heading=section_heading)
        return incoming

    def _replace_markdown_section(self, existing: str, incoming: str, *, section_heading: str) -> str:
        heading = section_heading.strip()
        if not heading:
            return incoming
        incoming_block = incoming.strip()
        lines = existing.splitlines()
        start_index: int | None = None
        heading_level: int | None = None
        for index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped.startswith("#"):
                continue
            prefix, _, title = stripped.partition(" ")
            if prefix and set(prefix) == {"#"} and title.strip() == heading:
                start_index = index
                heading_level = len(prefix)
                break
        if start_index is not None and heading_level is not None:
            end_index = len(lines)
            for index in range(start_index + 1, len(lines)):
                stripped = lines[index].strip()
                if not stripped.startswith("#"):
                    continue
                prefix, _, _title = stripped.partition(" ")
                if prefix and set(prefix) == {"#"} and len(prefix) <= heading_level:
                    end_index = index
                    break
            replacement_lines = incoming_block.splitlines()
            merged_lines = lines[:start_index] + replacement_lines + lines[end_index:]
            return "\n".join(merged_lines).strip()
        separator = "\n\n" if existing.strip() and incoming_block else ""
        return f"{existing.rstrip()}{separator}{incoming_block}".strip()
