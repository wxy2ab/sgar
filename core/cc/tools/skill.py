"""SkillTool — load a discovered skill's instructions on demand.

A *skill* is a ``SKILL.md`` document (with ``name`` / ``description``
frontmatter) discovered from three roots (repo ``skills/``, user
``~/.<root>/skills/``, project ``.skills/`` — see :mod:`core.cc.skills.loader`).
When the model calls ``skill(name=...)`` this tool returns that skill's full body
plus its base directory, so the model can then follow the instructions and read
any assets bundled alongside the ``SKILL.md``.

The tool is constructed with an already-discovered :class:`SkillRegistry` (built
once at registry-build time, when the session cwd is known). Available skills are
listed in the tool *description* (not the system prompt) so the listing is local
to the tool and never touches the shared/cached prompt pipeline.

Default-ON, but self-disabling: ``is_enabled`` returns False when no skills were
discovered (or the operator clears ``skills_enabled``), so a checkout with zero
skill files exports a byte-identical tool schema — mirroring ``RunTestsTool``.
This ``cc`` tool never imports ``core.ccx``.
"""

from __future__ import annotations

from typing import Any

from ..skills import SkillRegistry
from .base import BaseTool, ToolCall, ToolResult, ToolSpec, ValidationResult
from .context import ToolUseContext


def _build_description(registry: SkillRegistry) -> str:
    lines = [
        "Load a skill to get specialized, prompt-facing instructions for a task. "
        "Returns the skill's full instructions plus its base directory (read "
        "files referenced by the skill relative to that directory). Call this "
        "when a task matches one of the available skills below.",
    ]
    skills = registry.list_skills()
    if skills:
        lines.append("")
        lines.append("<available_skills>")
        for skill in skills:
            lines.append(f"- {skill.name}: {skill.description}")
        lines.append("</available_skills>")
    return "\n".join(lines)


class SkillTool(BaseTool):
    def __init__(self, registry: SkillRegistry) -> None:
        self._registry = registry
        super().__init__(
            ToolSpec(
                name="skill",
                description=_build_description(registry),
                input_schema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the skill to load (see the available skills list).",
                        },
                    },
                    "required": ["name"],
                },
                is_read_only=True,
            )
        )

    def is_enabled(self, ctx: Any) -> bool:
        # Default-ON but self-disabling: hidden from the LLM schema when no
        # skills were discovered, so a zero-skill checkout's schema is
        # byte-identical to before this tool existed. ``skills_enabled`` is an
        # explicit operator escape hatch (defaults True).
        config = getattr(ctx, "config", None)
        if not getattr(config, "skills_enabled", True):
            return False
        return bool(self._registry)

    def validate_input(self, arguments: dict[str, Any]) -> ValidationResult:
        name = arguments.get("name")
        if not isinstance(name, str) or not name.strip():
            return ValidationResult(ok=False, message="name is required.")
        return ValidationResult(ok=True)

    async def execute(self, tool_call: ToolCall, ctx: ToolUseContext) -> ToolResult:
        del ctx
        name = str(tool_call.arguments["name"]).strip()
        skill = self._registry.get(name)
        if skill is None:
            available = ", ".join(self._registry.names()) or "(none)"
            return ToolResult(
                tool_use_id=tool_call.tool_use_id,
                tool_name=tool_call.tool_name,
                success=False,
                content=f"Unknown skill: {name}. Available skills: {available}.",
                error_code="SK1001",
            )
        return ToolResult(
            tool_use_id=tool_call.tool_use_id,
            tool_name=tool_call.tool_name,
            success=True,
            content=skill.content,
            data={
                "name": skill.name,
                "source": skill.source,
                "path": skill.path,
                "base_dir": skill.base_dir,
            },
        )
