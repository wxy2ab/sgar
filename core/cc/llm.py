from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import CCConfig

logger = logging.getLogger(__name__)


class LLMClientProvider(Protocol):
    def get_client(self, *, config: CCConfig, purpose: str, overrides: dict[str, Any] | None = None) -> Any:
        ...


#: Recommended purpose → client routing. Structured / tool-driving purposes go
#: to a fast no-thinking client (``SimpleDeepSeekClient``): it turns turns in
#: ~5-15s and is the better tool driver, and these purposes already get
#: API-level JSON mode (``CCConfig.json_mode_purposes``) so a non-reasoning
#: model can't "say it will emit JSON then stop". Synthesis / hard-reasoning
#: purposes (doc_synthesize, goal.judge, goal.report, query_engine.turn) are
#: deliberately ABSENT — they fall through to ``default_llm_client``
#: (``SimpleDeepSeekClientReasoning``). Opt in per run via
#: ``CCConfig.cc_use_recommended_routing`` (an explicit ``role_llm_clients``
#: entry still wins). See ``core/ccx/docs/role_based_llm_routing.md``.
RECOMMENDED_ROLE_LLM_CLIENTS: dict[str, str] = {
    "plan": "SimpleDeepSeekClient",
    "spec": "SimpleDeepSeekClient",
    "doc_decompose": "SimpleDeepSeekClient",
    "doc_prose_to_json": "SimpleDeepSeekClient",
    "agent": "SimpleDeepSeekClient",
    "structured_flow.planning": "SimpleDeepSeekClient",
    "structured_flow_planning": "SimpleDeepSeekClient",
    "swarm.goal_planner": "SimpleDeepSeekClient",
    "goal.plan": "SimpleDeepSeekClient",
    "goal.replan": "SimpleDeepSeekClient",
}


def _recommended_routing_enabled(config: CCConfig) -> bool:
    """Whether the recommended preset is active: CCConfig field primary,
    ``CCX_USE_RECOMMENDED_ROUTING`` env override (both directions). Read per
    call so a launch can flip it without rebuilding the config — matching the
    project's other ``CCX_*`` per-call env knobs. Default off → byte-identical.
    """
    raw = os.environ.get("CCX_USE_RECOMMENDED_ROUTING")
    if raw is not None and raw.strip():
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return bool(getattr(config, "cc_use_recommended_routing", False))


def _resolve_client_name(config: CCConfig, purpose: str) -> tuple[str, str]:
    """Resolve purpose -> (client_name, via).

    ``via`` is ``"role"`` when an explicit ``role_llm_clients`` entry matched,
    ``"recommended"`` when recommended routing is on and the preset matched,
    ``"default"`` otherwise. Explicit role entries always win over the preset.
    Empty/missing purpose always falls back to default.
    """
    role_table = config.role_llm_clients or {}
    if purpose and purpose in role_table:
        return role_table[purpose], "role"
    if (
        purpose
        and purpose in RECOMMENDED_ROLE_LLM_CLIENTS
        and _recommended_routing_enabled(config)
    ):
        return RECOMMENDED_ROLE_LLM_CLIENTS[purpose], "recommended"
    return config.default_llm_client, "default"


@dataclass(slots=True)
class DefaultLLMClientProvider:
    """All LLM access must go through LLMFactory."""

    default_kwargs: dict[str, Any] = field(default_factory=dict)

    def get_client(
        self,
        *,
        config: CCConfig,
        purpose: str,
        overrides: dict[str, Any] | None = None,
    ) -> Any:
        from core.llms.llm_factory import LLMFactory

        client_name, via = _resolve_client_name(config, purpose)
        logger.debug(
            "DefaultLLMClientProvider.get_client purpose=%s via=%s client=%s",
            purpose, via, client_name,
        )
        params = dict(self.default_kwargs)
        params.update(overrides or {})
        client = LLMFactory().get_instance(client_name, **params)
        if purpose in (config.json_mode_purposes or ()):
            if getattr(client, "supports_structured_output", False):
                client.set_response_format({"type": "json_object"})
                logger.debug(
                    "DefaultLLMClientProvider: enabled JSON mode for "
                    "purpose=%s client=%s",
                    purpose, client_name,
                )
            else:
                logger.info(
                    "DefaultLLMClientProvider: purpose=%s requested JSON "
                    "mode but client=%s does not support structured output; "
                    "falling back to prompt-level discipline only",
                    purpose, client_name,
                )
        return client
