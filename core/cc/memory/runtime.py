from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import CCConfig
from ..conversation.session import QuerySession
from .extractor import build_compaction_candidate, build_turn_write_candidates
from .models import (
    MemoryFact,
    MemoryHit,
    MemoryProviderStatus,
    MemoryQuery,
    MemoryRecallBundle,
    MemoryWriteCandidate,
    MemoryWriteResult,
)
from .policy import recall_enabled, should_store_candidate, store_enabled, trim_text


@dataclass(slots=True)
class MemoryRuntime:
    config: CCConfig
    provider: Any

    @property
    def provider_name(self) -> str:
        return str(getattr(self.provider, "name", "unknown"))

    def status(self) -> MemoryProviderStatus:
        return self.provider.status()

    def before_turn(
        self,
        *,
        session: QuerySession,
        user_input: str | list[dict[str, object]],
    ) -> MemoryRecallBundle | None:
        if not recall_enabled(self.config):
            return None
        query_text = self._query_text(user_input)
        if not query_text:
            return None
        if self.config.memory_structure_first or self._should_prefer_structural(query_text):
            bundle = self._structural_search(
                query=query_text,
                wing="wing_code",
                room=None,
                limit=self.config.memory_max_prompt_hits,
            )
        else:
            bundle = self.provider.recall(
                MemoryQuery(
                    query=query_text,
                    limit=max(1, self.config.memory_max_prompt_hits),
                )
            )
        session.metadata.state["memory_context"] = bundle.to_prompt_payload()
        session.metadata.state["memory_provider"] = self.provider_name
        session.metadata.state["memory_status"] = {
            "available": bundle.available,
            "error": bundle.error,
        }
        return bundle

    def after_turn(
        self,
        *,
        session: QuerySession,
        assistant_text: str,
    ) -> list[MemoryWriteResult]:
        if not store_enabled(self.config):
            return []
        candidates = build_turn_write_candidates(
            session=session,
            assistant_text=assistant_text,
            max_chars=self.config.memory_write_char_budget,
        )
        return self.store_candidates(candidates)

    def after_compaction(
        self,
        *,
        session: QuerySession,
        compact_summary: str,
    ) -> MemoryWriteResult | None:
        if not store_enabled(self.config):
            return None
        candidate = build_compaction_candidate(
            session=session,
            compact_summary=compact_summary,
            max_chars=self.config.memory_write_char_budget,
        )
        if candidate is None or not should_store_candidate(candidate, self.config):
            return None
        return self.provider.store(candidate)

    def explicit_search(
        self,
        *,
        query: str,
        wing: str | None = None,
        room: str | None = None,
        limit: int | None = None,
        mode: str = "semantic",
    ) -> MemoryRecallBundle:
        if mode == "semantic" and self.config.memory_structure_first and room is None:
            mode = "structure_first"
        if mode in {"structural", "structure_first"}:
            return self._structural_search(
                query=query,
                wing=wing or "wing_code",
                room=room,
                limit=limit,
            )
        return self.provider.recall(
            MemoryQuery(
                query=query,
                wing=wing,
                room=room,
                limit=max(1, int(limit or self.config.memory_max_prompt_hits)),
                mode=mode,
            )
        )

    def explicit_store(self, candidate: MemoryWriteCandidate) -> MemoryWriteResult:
        if not should_store_candidate(candidate, self.config):
            return MemoryWriteResult(success=True, stored=False, message="Candidate filtered by memory policy.")
        return self.provider.store(candidate)

    def explicit_query_facts(
        self,
        *,
        entity: str,
        as_of: str | None = None,
        direction: str = "both",
    ) -> list[MemoryFact]:
        return list(self.provider.query_facts(entity=entity, as_of=as_of, direction=direction))

    def explicit_store_fact(self, fact: MemoryFact) -> MemoryWriteResult:
        return self.provider.store_fact(fact)

    def store_candidates(self, candidates: list[MemoryWriteCandidate]) -> list[MemoryWriteResult]:
        results: list[MemoryWriteResult] = []
        for candidate in candidates:
            if not should_store_candidate(candidate, self.config):
                continue
            results.append(self.provider.store(candidate))
        return results

    @staticmethod
    def _query_text(user_input: str | list[dict[str, object]]) -> str:
        if isinstance(user_input, str):
            return trim_text(user_input, 400)
        return trim_text(str(user_input), 400)

    def _structural_search(
        self,
        *,
        query: str,
        wing: str,
        room: str | None,
        limit: int | None,
    ) -> MemoryRecallBundle:
        max_hits = max(1, int(limit or self.config.memory_max_prompt_hits))
        if room:
            return self.provider.recall(
                MemoryQuery(
                    query=query,
                    wing=wing,
                    room=room,
                    limit=max_hits,
                    mode="semantic",
                )
            )
        rooms = ["repo-structure", "module-map", "architecture", "design-decisions"]
        bundles: list[MemoryRecallBundle] = []
        for target_room in rooms:
            bundles.append(
                self.provider.recall(
                    MemoryQuery(
                        query=query,
                        wing=wing,
                        room=target_room,
                        limit=max_hits,
                        mode="semantic",
                    )
                )
            )
        hits = self._dedupe_hits([hit for bundle in bundles for hit in bundle.hits])[:max_hits]
        facts = self._dedupe_facts([fact for bundle in bundles for fact in bundle.facts])[:max_hits]
        available = any(bundle.available for bundle in bundles)
        errors = [bundle.error for bundle in bundles if bundle.error]
        return MemoryRecallBundle(
            provider=self.provider_name,
            query=query,
            hits=hits,
            facts=facts,
            summary=self._structural_summary(hits, facts),
            available=available,
            error="; ".join(errors) if (errors and not available) else None,
        )

    @staticmethod
    def _dedupe_hits(hits: list[MemoryHit]) -> list[MemoryHit]:
        deduped: list[MemoryHit] = []
        seen: set[tuple[str, str, str]] = set()
        for hit in hits:
            key = (hit.wing, hit.room, hit.summary)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(hit)
        return deduped

    @staticmethod
    def _dedupe_facts(facts: list[MemoryFact]) -> list[MemoryFact]:
        deduped: list[MemoryFact] = []
        seen: set[tuple[str, str, str]] = set()
        for fact in facts:
            key = (fact.subject, fact.predicate, fact.object)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(fact)
        return deduped

    @staticmethod
    def _structural_summary(hits: list[MemoryHit], facts: list[MemoryFact]) -> str:
        lines: list[str] = []
        grouped_hits: dict[str, list[MemoryHit]] = {}
        for hit in hits:
            grouped_hits.setdefault(hit.room, []).append(hit)
        for room, room_hits in grouped_hits.items():
            lines.append(f"{room}:")
            for index, hit in enumerate(room_hits, start=1):
                lines.append(f"  {index}. {hit.summary}")
        for fact in facts[:5]:
            lines.append(f"fact: {fact.subject} -> {fact.predicate} -> {fact.object}")
        return "\n".join(lines) or "No structural memory results."

    @staticmethod
    def _should_prefer_structural(query: str) -> bool:
        lowered = str(query or "").lower()
        structural_keywords = (
            "architecture",
            "component",
            "defined",
            "definition",
            "directory",
            "entrypoint",
            "handler",
            "implemented",
            "implements",
            "implementation",
            "layout",
            "location",
            "module",
            "outline",
            "owned by",
            "responsible",
            "relationship",
            "repo",
            "repository",
            "structure",
            "where does",
            "where is",
            "which module",
            "which file",
            "who handles",
            "目录",
            "在哪里",
            "处理",
            "哪段逻辑",
            "哪个文件",
            "哪个模块",
            "定义",
            "入口",
            "关系",
            "负责",
            "仓库",
            "实现",
            "文件",
            "逻辑",
            "模块",
            "架构",
            "结构",
        )
        return any(keyword in lowered for keyword in structural_keywords)
