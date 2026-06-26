from __future__ import annotations

from typing import Protocol

from .models import (
    MemoryFact,
    MemoryProviderStatus,
    MemoryQuery,
    MemoryRecallBundle,
    MemoryWriteCandidate,
    MemoryWriteResult,
)


class MemoryProvider(Protocol):
    name: str

    def supports(self, capability: str) -> bool:
        ...

    def status(self) -> MemoryProviderStatus:
        ...

    def recall(self, query: MemoryQuery) -> MemoryRecallBundle:
        ...

    def store(self, candidate: MemoryWriteCandidate) -> MemoryWriteResult:
        ...

    def query_facts(
        self,
        *,
        entity: str,
        as_of: str | None = None,
        direction: str = "both",
    ) -> list[MemoryFact]:
        ...

    def store_fact(self, fact: MemoryFact) -> MemoryWriteResult:
        ...
