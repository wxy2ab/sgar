from __future__ import annotations

from pathlib import Path
from typing import Any

from ..specs import SPEC_ARTIFACT_NAMES, ensure_spec_state
from .base import BaseTool, ToolCall, ToolResult, ToolSpec, ValidationResult
from .context import ToolUseContext


class SpecArtifactWriteTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            ToolSpec(
                name="spec_artifact_write",
                description="Write the controlled spec artifacts under .cc/specs/<task-slug>/.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "artifact": {"type": "string"},
                        "content": {"type": "string"},
                        "status": {"type": "string"},
                        "merge_mode": {"type": "string"},
                        "section_heading": {"type": "string"},
                    },
                    "required": ["artifact", "content"],
                },
                is_read_only=False,
            )
        )

    def is_enabled(self, ctx: ToolUseContext) -> bool:
        return bool(ctx.permissions.mode == "spec" or ctx.get_app_state().get("spec_mode"))

    def validate_input(self, arguments: dict[str, Any]) -> ValidationResult:
        artifact = str(arguments.get("artifact", "")).strip().lower()
        if artifact not in SPEC_ARTIFACT_NAMES:
            return ValidationResult(ok=False, message=f"artifact must be one of {SPEC_ARTIFACT_NAMES}.")
        if "content" not in arguments:
            return ValidationResult(ok=False, message="content is required.")
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
        artifact = str(tool_call.arguments["artifact"]).strip().lower()
        content = str(tool_call.arguments.get("content", ""))
        status = str(tool_call.arguments.get("status", "ready")).strip().lower() or "ready"
        merge_mode = str(tool_call.arguments.get("merge_mode", "replace")).strip().lower() or "replace"
        section_heading = str(tool_call.arguments.get("section_heading", "")).strip()
        state = ensure_spec_state(
            current_state=ctx.get_app_state(),
            cwd=ctx.cwd,
            config=ctx.config,
            enabled=True,
        )
        target_path = Path(state["spec_artifacts"][artifact]).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        existing = target_path.read_text(encoding="utf-8") if target_path.exists() else ""
        target_path.write_text(
            self._merge_content(
                existing,
                content,
                merge_mode=merge_mode,
                section_heading=section_heading,
            ),
            encoding="utf-8",
        )

        def modify(current_ctx: ToolUseContext) -> ToolUseContext:
            next_state = ensure_spec_state(
                current_state=current_ctx.get_app_state(),
                cwd=current_ctx.cwd,
                config=current_ctx.config,
                enabled=True,
            )
            next_state["spec_phase"] = artifact
            statuses = dict(next_state.get("spec_artifact_status") or {})
            statuses[artifact] = status
            next_state["spec_artifact_status"] = statuses
            next_state["render_ready"] = all(
                str(statuses.get(name, "pending")).lower() in {"ready", "completed"}
                for name in SPEC_ARTIFACT_NAMES
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
