from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from ..errors import SessionPersistenceError
from ..jsonl import append_jsonl_many_sync
from .models import SessionMessage


class SessionJournal:
    def __init__(self, session_dir: str | Path) -> None:
        self.session_dir = Path(session_dir)
        self.messages_path = self.session_dir / "messages.jsonl"

    def _load_messages_sync(self) -> list[SessionMessage]:
        if not self.messages_path.exists():
            return []
        messages: list[SessionMessage] = []
        try:
            with self.messages_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        payload = json.loads(stripped)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(payload, dict):
                        continue
                    op = str(payload.get("op") or "append")
                    if op == "reset":
                        encoded_messages = payload.get("messages") or []
                        if not isinstance(encoded_messages, list):
                            encoded_messages = []
                        messages = [
                            SessionMessage.from_dict(item)
                            for item in encoded_messages
                            if isinstance(item, dict)
                        ]
                        continue
                    message_payload = payload.get("message", payload)
                    if isinstance(message_payload, dict):
                        messages.append(SessionMessage.from_dict(message_payload))
        except OSError as exc:
            raise SessionPersistenceError(
                f"Failed to load session messages: {self.messages_path}"
            ) from exc
        return messages

    async def load_messages(self) -> list[SessionMessage]:
        return await asyncio.to_thread(self._load_messages_sync)

    def _persist_ops_sync(self, ops: list[dict[str, Any]]) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        try:
            append_jsonl_many_sync(self.messages_path, ops)
        except OSError as exc:
            raise SessionPersistenceError(
                f"Failed to persist session messages: {self.messages_path}"
            ) from exc

    async def persist_ops(self, ops: list[dict[str, Any]]) -> None:
        if not ops:
            return
        await asyncio.to_thread(self._persist_ops_sync, ops)

