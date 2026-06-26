from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..conversation.prompt_catalog import PromptCatalog


@dataclass(slots=True)
class AgentDefinition:
    agent_id: str
    name: str
    description: str
    prompt_key: str
    tools: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def resolve_agent_prompt(
    *,
    definition: AgentDefinition,
    prompt_catalog: PromptCatalog,
    prompt_language: str,
) -> str:
    return prompt_catalog.resolve(definition.prompt_key, prompt_language)
