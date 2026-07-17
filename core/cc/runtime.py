from __future__ import annotations

from .config import CCConfig
from .conversation.query_engine import QueryEngine
from .conversation.session import QuerySession
from .engine_factory import get_default_engine_factory
from .llm import LLMClientProvider
from .tools import ToolRegistry


def build_default_query_engine(
    *,
    cwd: str | None = None,
    config: CCConfig | None = None,
    llm_client_provider: LLMClientProvider | None = None,
    session: QuerySession | None = None,
    tool_registry: ToolRegistry | None = None,
) -> QueryEngine:
    """Build the default internal QueryEngine composition.

    This is an advanced entrypoint for callers that need direct session/turn
    control. For most external integrations, prefer ``CodeAgent`` from
    ``core.cc.api`` and treat this function as a lower-level composition helper.
    ``tool_registry`` is an opt-in replacement for the built-in code tools.
    """

    return get_default_engine_factory().build_query_engine(
        cwd=cwd,
        config=config,
        llm_client_provider=llm_client_provider,
        session=session,
        tool_registry=tool_registry,
    )
