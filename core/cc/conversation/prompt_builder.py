from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

from .models import SystemPromptParts
from .protocol import build_response_protocol
from .prompt_catalog import PromptCatalog


@dataclass(slots=True)
class PromptAssemblyResult:
    parts: SystemPromptParts
    resolved_keys: list[str] = field(default_factory=list)


def build_effective_prompt(
    *,
    prompt_catalog: PromptCatalog,
    prompt_language: str,
    prompt_key: str | None = None,
    main_thread_agent_definition: Any = None,
    custom_system_prompt: str | None = None,
    append_system_prompt: str | None = None,
    context: dict[str, Any] | None = None,
) -> SystemPromptParts:
    key = prompt_key or ("system.coordinator" if main_thread_agent_definition else "system.default")
    primary = custom_system_prompt or prompt_catalog.resolve(key, prompt_language)
    append: list[str] = []
    if append_system_prompt:
        append.append(append_system_prompt)
    if context:
        runtime_context = dict(context)
        collaboration_strategy = runtime_context.pop("agent_collaboration_strategy", None)
        collaboration_required = bool(runtime_context.pop("agent_collaboration_required", False))
        collaboration_pattern = str(runtime_context.pop("agent_collaboration_pattern", "") or "")
        collaboration_completed = bool(runtime_context.pop("agent_collaboration_completed", False))
        try:
            collaboration_count = int(runtime_context.pop("agent_collaboration_count", 0) or 0)
        except (ValueError, TypeError):
            collaboration_count = 0
        repository_outline_text = str(runtime_context.pop("repository_outline_text", "") or "")
        paths_in_request_text = str(runtime_context.pop("paths_in_request_text", "") or "")
        memory_room_summaries = dict(runtime_context.pop("memory_room_summaries", {}) or {})
        try:
            memory_prompt_summary_max_chars = int(runtime_context.pop("memory_prompt_summary_max_chars", 400) or 400)
        except (ValueError, TypeError):
            memory_prompt_summary_max_chars = 400
        runtime_context.pop("repository_outline", None)
        append.append(f"# Runtime Context\n{json.dumps(runtime_context, ensure_ascii=False, default=str)}")
        if memory_room_summaries:
            append.append(
                _render_memory_highlights(
                    memory_room_summaries,
                    max_chars=memory_prompt_summary_max_chars,
                )
            )
        if collaboration_required and isinstance(collaboration_strategy, dict):
            append.append(_render_agent_collaboration_strategy(
                strategy=collaboration_strategy,
                pattern=collaboration_pattern,
                completed=collaboration_completed,
                count=collaboration_count,
            ))
        # ``Paths in this task`` MUST land before the repository
        # outline so the LLM reads "these paths exist, regardless of
        # what the outline below shows" before forming any conclusions
        # from the (truncated) outline. See
        # ``mode_strategy.build_paths_in_request_block`` for the why.
        if paths_in_request_text.strip():
            append.append(f"# Paths in this task\n{paths_in_request_text}")
        if repository_outline_text.strip():
            append.append(
                "# Repository Outline (PARTIAL — truncated; trust paths above)\n"
                f"{repository_outline_text}"
            )
        if not runtime_context.get("native_tool_calling"):
            enabled_tools = runtime_context.get("enabled_tools")
            if isinstance(enabled_tools, list):
                append.append(build_response_protocol(prompt_language, enabled_tools))
    return SystemPromptParts(primary=primary, append=append, metadata={"prompt_key": key})


def _render_agent_collaboration_strategy(
    *,
    strategy: dict[str, Any],
    pattern: str,
    completed: bool,
    count: int,
) -> str:
    lines = ["# Agent Collaboration Strategy"]
    lines.append(f"- required: true")
    lines.append(f"- pattern: {pattern or strategy.get('pattern', '')}")
    lines.append(f"- completed: {str(completed).lower()}")
    lines.append(f"- child_agent_count: {count}")
    rationale = str(strategy.get("rationale", "") or "").strip()
    if rationale:
        lines.append(f"- rationale: {rationale}")
    roles = strategy.get("roles")
    if isinstance(roles, list) and roles:
        lines.append(f"- roles: {', '.join(str(item) for item in roles)}")
    plan = strategy.get("delegation_plan")
    if isinstance(plan, list) and plan:
        lines.append("- delegation_plan:")
        for step in plan:
            lines.append(f"  - {step}")
    return "\n".join(lines)


def _render_memory_highlights(room_summaries: dict[str, Any], *, max_chars: int) -> str:
    lines = ["# Memory Highlights"]
    remaining = max(40, int(max_chars))
    for room, summary in room_summaries.items():
        text = str(summary or "").strip()
        if not text:
            continue
        prefix = f"- {room}: "
        if remaining <= len(prefix):
            break
        available = remaining - len(prefix)
        if len(text) > available:
            text = f"{text[: max(0, available - 3)]}..."
        lines.append(f"{prefix}{text}")
        remaining -= len(prefix) + len(text) + 1
        if remaining <= 0:
            break
    return "\n".join(lines)


class UserContextBuilder:
    def build(self, *, cwd: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        context = {"cwd": cwd}
        if extra:
            context.update(extra)
        return context


class SystemPromptBuilder:
    def __init__(self, prompt_catalog: PromptCatalog) -> None:
        self.prompt_catalog = prompt_catalog

    def build(
        self,
        *,
        prompt_language: str,
        prompt_key: str | None = None,
        custom_system_prompt: str | None = None,
        append_system_prompt: str | None = None,
        context: dict[str, Any] | None = None,
        main_thread_agent_definition: Any = None,
    ) -> PromptAssemblyResult:
        parts = build_effective_prompt(
            prompt_catalog=self.prompt_catalog,
            prompt_language=prompt_language,
            prompt_key=prompt_key,
            main_thread_agent_definition=main_thread_agent_definition,
            custom_system_prompt=custom_system_prompt,
            append_system_prompt=append_system_prompt,
            context=context,
        )
        return PromptAssemblyResult(parts=parts, resolved_keys=[parts.metadata["prompt_key"]])
