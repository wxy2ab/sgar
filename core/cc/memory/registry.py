from __future__ import annotations

from collections.abc import Callable

from ..config import CCConfig
from .noop_provider import NoOpMemoryProvider


ProviderFactory = Callable[[CCConfig], object]


class MemoryProviderRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, ProviderFactory] = {}

    def register(self, name: str, factory: ProviderFactory) -> None:
        self._factories[str(name)] = factory

    def resolve(self, config: CCConfig) -> object:
        provider_name = config.memory_provider if config.memory_enabled else "noop"
        factory = self._factories.get(provider_name) or self._factories.get("noop")
        if factory is None:
            raise KeyError(f"Unknown memory provider: {provider_name}")
        return factory(config)


def build_default_memory_provider_registry() -> MemoryProviderRegistry:
    registry = MemoryProviderRegistry()
    registry.register("noop", lambda config: NoOpMemoryProvider())
    return registry
