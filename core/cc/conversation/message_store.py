from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .models import SessionMessage
from .session_journal import SessionJournal
from .session import QuerySession


class SessionMessageStore:
    def __init__(self, session: QuerySession, *, journal: SessionJournal | None = None) -> None:
        self.session = session
        self._messages: list[SessionMessage] = []
        self._char_count = 0
        self._turn_index: dict[str, list[int]] = {}
        self._pending_ops: list[dict[str, object]] = []
        self._journal = journal or SessionJournal(session.session_dir)

    def _load_messages(self, messages: list[SessionMessage]) -> None:
        self._messages = []
        self._char_count = 0
        self._turn_index = {}
        self._pending_ops = []
        for message in messages:
            self._append_in_memory(message)

    def _append_in_memory(self, message: SessionMessage) -> None:
        self._turn_index.setdefault(message.turn_id, []).append(len(self._messages))
        self._messages.append(message)
        self._char_count += len(message.content)

    def append(self, message: SessionMessage) -> None:
        self._append_in_memory(message)
        self._pending_ops.append({"op": "append", "message": message.to_dict()})
        self.session.touch()

    def append_many(self, messages: Iterable[SessionMessage]) -> None:
        added = False
        for message in messages:
            self._append_in_memory(message)
            self._pending_ops.append({"op": "append", "message": message.to_dict()})
            added = True
        if added:
            self.session.touch()

    def get_turn_messages(self, turn_id: str) -> list[SessionMessage]:
        return [self._messages[index] for index in self._turn_index.get(turn_id, [])]

    def get_compactable_slice(self, preserve_recent_messages: int = 20) -> list[SessionMessage]:
        return list(self._messages[:-preserve_recent_messages]) if len(self._messages) > preserve_recent_messages else []

    def snapshot(self) -> list[SessionMessage]:
        return list(self._messages)

    def total_char_count(self) -> int:
        return self._char_count

    def compact_with_boundary(self, *, boundary_message: SessionMessage, preserve_recent_messages: int = 20) -> None:
        if self._pending_ops:
            stale_appends = [
                op["message"] for op in self._pending_ops if op.get("op") == "append"
            ]
        else:
            stale_appends = []
        recent = self._messages[-preserve_recent_messages:] if len(self._messages) > preserve_recent_messages else []
        merged_recent = [boundary_message, *recent]
        if stale_appends:
            existing_contents = {(m.role, m.content) for m in merged_recent}
            for raw in stale_appends:
                msg = SessionMessage.from_dict(raw) if isinstance(raw, dict) else raw
                if (msg.role, msg.content) not in existing_contents:
                    merged_recent.append(msg)
        self._load_messages(merged_recent)
        self._pending_ops = [{
            "op": "reset",
            "messages": [message.to_dict() for message in self._messages],
        }]
        self.session.touch()

    @property
    def session_dir(self) -> Path:
        return self.session.session_dir

    @property
    def messages_path(self) -> Path:
        return self.session_dir / "messages.jsonl"

    async def persist(self) -> None:
        pending_ops = list(self._pending_ops)
        if not pending_ops:
            return
        await self._journal.persist_ops(pending_ops)
        if self._pending_ops[:len(pending_ops)] == pending_ops:
            del self._pending_ops[:len(pending_ops)]

    async def load_from_disk(self) -> None:
        self._load_messages(await self._journal.load_messages())
