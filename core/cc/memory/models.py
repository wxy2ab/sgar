from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class MemoryProviderStatus:
    provider: str
    available: bool
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MemoryQuery:
    query: str
    wing: str | None = None
    room: str | None = None
    limit: int = 5
    mode: str = "semantic"


@dataclass(slots=True)
class MemoryHit:
    text: str
    wing: str = "unknown"
    room: str = "unknown"
    source_file: str = "?"
    similarity: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        text = " ".join(self.text.strip().split())
        if len(text) <= 200:
            return text
        return f"{text[:197]}..."


@dataclass(slots=True)
class MemoryFact:
    subject: str
    predicate: str
    object: str
    valid_from: str | None = None
    valid_to: str | None = None
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MemoryRecallBundle:
    provider: str
    query: str
    hits: list[MemoryHit] = field(default_factory=list)
    facts: list[MemoryFact] = field(default_factory=list)
    summary: str = ""
    available: bool = True
    error: str | None = None

    def to_prompt_payload(self) -> dict[str, Any]:
        room_groups: dict[str, list[dict[str, Any]]] = {}
        for hit in self.hits:
            room_groups.setdefault(hit.room, []).append(
                {
                    "summary": hit.summary,
                    "wing": hit.wing,
                    "source_file": hit.source_file,
                    "similarity": hit.similarity,
                }
            )
        room_summaries = {
            room: " | ".join(item["summary"] for item in items[:2])
            for room, items in room_groups.items()
        }
        preferred_room_order = list(room_groups.keys())
        top_room = preferred_room_order[0] if preferred_room_order else None
        return {
            "provider": self.provider,
            "query": self.query,
            "available": self.available,
            "error": self.error,
            "summary": self.summary,
            "preferred_room_order": preferred_room_order,
            "top_room": top_room,
            "room_groups": room_groups,
            "room_summaries": room_summaries,
            "hits": [
                {
                    "summary": hit.summary,
                    "wing": hit.wing,
                    "room": hit.room,
                    "source_file": hit.source_file,
                    "similarity": hit.similarity,
                }
                for hit in self.hits
            ],
            "facts": [
                {
                    "subject": fact.subject,
                    "predicate": fact.predicate,
                    "object": fact.object,
                    "valid_from": fact.valid_from,
                    "valid_to": fact.valid_to,
                    "confidence": fact.confidence,
                }
                for fact in self.facts
            ],
        }


@dataclass(slots=True)
class MemoryWriteCandidate:
    memory_kind: str
    subject: str
    summary: str
    text: str
    details: dict[str, Any] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    wing: str = "wing_code"
    room: str = "general"
    facts: list[MemoryFact] = field(default_factory=list)


@dataclass(slots=True)
class MemoryWriteResult:
    success: bool
    stored: bool
    message: str = ""
    duplicate: bool = False
    memory_id: str | None = None
    fact_ids: list[str] = field(default_factory=list)
