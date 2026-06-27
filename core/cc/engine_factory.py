from __future__ import annotations

from pathlib import Path
import threading

from .agents.runtime_registry import get_in_process_runtime_registry
from .config import CCConfig, load_cc_config
from .conversation.compact import SessionCompactor
from .conversation.context_assembler import ContextAssembler
from .conversation.message_store import SessionMessageStore
from .conversation.prompt_builder import SystemPromptBuilder
from .conversation.prompt_catalog import PromptCatalog
from .conversation.query_engine import QueryEngine
from .conversation.session import QuerySession, SessionFactory
from .conversation.turn_pipeline import (
    SessionStatePreparer,
    TurnPersistenceCoordinator,
    TurnPromptAssembler,
)
from .llm import DefaultLLMClientProvider, LLMClientProvider
from .memory import MemoryRuntime, build_default_memory_provider_registry
from .providers import Environment, default_environment
from .tools import ToolOrchestrator
from .tools.builtin import build_builtin_tool_registry


class EngineFactory:
    """Internal composition root for QueryEngine defaults.

    The public integration surface is ``CodeAgent``. ``EngineFactory`` exists so
    runtime assembly can evolve without pushing external callers onto
    conversation-level implementation details.
    """

    def __init__(self, *, environment: Environment | None = None) -> None:
        self.environment = environment or default_environment()
        self._prompt_catalogs: dict[str, PromptCatalog] = {}

    def get_prompt_catalog(self, prompt_root: str | Path) -> PromptCatalog:
        key = str(Path(prompt_root).resolve())
        catalog = self._prompt_catalogs.get(key)
        if catalog is None:
            catalog = PromptCatalog.from_prompt_root(key)
            self._prompt_catalogs[key] = catalog
        return catalog

    def build_query_engine(
        self,
        *,
        cwd: str | None = None,
        config: CCConfig | None = None,
        llm_client_provider: LLMClientProvider | None = None,
        session: QuerySession | None = None,
    ) -> QueryEngine:
        """Assemble the default QueryEngine stack.

        This method is intentionally low-level. Keep external integrations on
        ``CodeAgent`` unless they explicitly need direct engine/session control.
        """

        resolved_config = config or load_cc_config()
        resolved_cwd = str(Path(cwd or Path.cwd()).resolve())
        resolved_session = session or SessionFactory(resolved_config).create(cwd=resolved_cwd)
        prompt_catalog = self.get_prompt_catalog(resolved_config.prompt_root_path(resolved_cwd))
        message_store = SessionMessageStore(resolved_session)
        provider = llm_client_provider or DefaultLLMClientProvider()
        runtime_registry = get_in_process_runtime_registry(resolved_config.runtime_root_path(resolved_cwd))
        memory_provider_registry = build_default_memory_provider_registry()
        memory_runtime = MemoryRuntime(
            config=resolved_config,
            provider=memory_provider_registry.resolve(resolved_config),
        )
        registry = build_builtin_tool_registry(
            resolved_config,
            llm_client_provider=provider,
            runtime_registry=runtime_registry,
            memory_runtime=memory_runtime,
            cwd=resolved_cwd,
        )
        orchestrator = ToolOrchestrator(registry)
        context_assembler = ContextAssembler(environment=self.environment)
        prompt_builder = SystemPromptBuilder(prompt_catalog)
        return QueryEngine(
            session=resolved_session,
            message_store=message_store,
            prompt_builder=prompt_builder,
            tool_orchestrator=orchestrator,
            llm_client_provider=provider,
            compactor=SessionCompactor(),
            context_assembler=context_assembler,
            state_preparer=SessionStatePreparer(),
            prompt_assembler=TurnPromptAssembler(
                prompt_builder=prompt_builder,
                context_assembler=context_assembler,
                tool_orchestrator=orchestrator,
                llm_client_provider=provider,
                memory_runtime=memory_runtime,
            ),
            persistence_coordinator=TurnPersistenceCoordinator(
                session=resolved_session,
                message_store=message_store,
            ),
            memory_runtime=memory_runtime,
        )


_DEFAULT_ENGINE_FACTORY: EngineFactory | None = None
_DEFAULT_ENGINE_FACTORY_LOCK = threading.Lock()


def get_default_engine_factory() -> EngineFactory:
    global _DEFAULT_ENGINE_FACTORY
    if _DEFAULT_ENGINE_FACTORY is None:
        with _DEFAULT_ENGINE_FACTORY_LOCK:
            if _DEFAULT_ENGINE_FACTORY is None:
                _DEFAULT_ENGINE_FACTORY = EngineFactory()
    return _DEFAULT_ENGINE_FACTORY
