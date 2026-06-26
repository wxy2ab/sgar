from __future__ import annotations

from .models import (
    MemoryFact,
    MemoryProviderStatus,
    MemoryQuery,
    MemoryRecallBundle,
    MemoryWriteCandidate,
    MemoryWriteResult,
)


class NoOpMemoryProvider:
    name = "noop"

    def supports(self, capability: str) -> bool:
        del capability
        return False

    def status(self) -> MemoryProviderStatus:
        return MemoryProviderStatus(
            provider=self.name,
            available=False,
            message="Memory is disabled.",
        )

    def recall(self, query: MemoryQuery) -> MemoryRecallBundle:
        return MemoryRecallBundle(
            provider=self.name,
            query=query.query,
            available=False,
            error="Memory is disabled.",
        )

    def store(self, candidate: MemoryWriteCandidate) -> MemoryWriteResult:
        del candidate
        return MemoryWriteResult(success=True, stored=False, message="Memory is disabled.")

    def query_facts(
        self,
        *,
        entity: str,
        as_of: str | None = None,
        direction: str = "both",
    ) -> list[MemoryFact]:
        del entity, as_of, direction
        return []

    def store_fact(self, fact: MemoryFact) -> MemoryWriteResult:
        del fact
        return MemoryWriteResult(success=True, stored=False, message="Memory is disabled.")
