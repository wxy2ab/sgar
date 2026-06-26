from __future__ import annotations

from dataclasses import dataclass
import time
import uuid

from .message_store import SessionMessageStore
from .models import SessionMessage
from .session import QuerySession


@dataclass(slots=True)
class CompactResult:
    applied: bool
    boundary_message: SessionMessage | None = None
    compacted_count: int = 0


class SessionCompactor:
    def __init__(self, *, preserve_recent_messages: int = 20, max_summary_chars: int = 4000) -> None:
        self.preserve_recent_messages = preserve_recent_messages
        self.max_summary_chars = max_summary_chars

    def should_compact(self, session: QuerySession, message_store: SessionMessageStore) -> bool:
        total_chars = message_store.total_char_count()
        if total_chars >= session.config.compact_soft_threshold:
            return True
        return len(message_store.snapshot()) > max(self.preserve_recent_messages + 10, 40)

    def compact(self, session: QuerySession, message_store: SessionMessageStore) -> CompactResult:
        slice_to_compact = message_store.get_compactable_slice(self.preserve_recent_messages)
        if not slice_to_compact:
            return CompactResult(applied=False)
        summary = self._summarize_messages(slice_to_compact)
        boundary = SessionMessage(
            message_id=f"msg_{uuid.uuid4().hex[:10]}",
            turn_id=f"compact_{uuid.uuid4().hex[:8]}",
            role="system",
            content=summary,
            kind="compact_boundary",
            metadata={
                "compacted_count": len(slice_to_compact),
                "created_at": time.time(),
            },
        )
        message_store.compact_with_boundary(
            boundary_message=boundary,
            preserve_recent_messages=self.preserve_recent_messages,
        )
        session.metadata.state["latest_compact_summary"] = summary
        session.metadata.state["compact_count"] = int(session.metadata.state.get("compact_count", 0)) + 1
        session.touch()
        return CompactResult(
            applied=True,
            boundary_message=boundary,
            compacted_count=len(slice_to_compact),
        )

    def _summarize_messages(self, messages: list[SessionMessage]) -> str:
        lines = ["# Compact Summary", "Earlier conversation has been compacted. Key retained context:"]
        for message in messages:
            snippet = " ".join(message.content.strip().split())
            if len(snippet) > 160:
                snippet = f"{snippet[:157]}..."
            lines.append(f"- [{message.role}/{message.kind}] {snippet}")
        text = "\n".join(lines)
        if len(text) <= self.max_summary_chars:
            return text
        return f"{text[: self.max_summary_chars - 3]}..."
