from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from pathlib import Path
import time
import threading
from collections.abc import AsyncIterator
from typing import Any

from ...jsonl import JsonlTailReader, append_jsonl_sync


@dataclass(slots=True)
class MailboxEnvelope:
    envelope_id: str
    team_id: str | None
    from_runtime_id: str
    to_runtime_id: str
    message_type: str
    payload: dict[str, Any]
    created_at: float = field(default_factory=time.time)
    acknowledged_at: float | None = None
    delivery_count: int = 0
    last_delivery_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MailboxEnvelope":
        return cls(
            envelope_id=str(payload["envelope_id"]),
            team_id=payload.get("team_id"),
            from_runtime_id=str(payload["from_runtime_id"]),
            to_runtime_id=str(payload["to_runtime_id"]),
            message_type=str(payload["message_type"]),
            payload=dict(payload.get("payload") or {}),
            created_at=float(payload.get("created_at") or time.time()),
            acknowledged_at=payload.get("acknowledged_at"),
            delivery_count=int(payload.get("delivery_count") or 0),
            last_delivery_at=payload.get("last_delivery_at"),
        )


@dataclass(slots=True)
class MailboxCursor:
    last_created_at: float = 0.0
    last_envelope_id: str = ""

    def advance(self, envelope: MailboxEnvelope) -> "MailboxCursor":
        return MailboxCursor(
            last_created_at=envelope.created_at,
            last_envelope_id=envelope.envelope_id,
        )


class MailboxStore:
    def __init__(self, mailbox_root: str | Path) -> None:
        self.mailbox_root = Path(mailbox_root)
        self.mailbox_root.mkdir(parents=True, exist_ok=True)
        self.envelopes_root = self.mailbox_root / "envelopes"
        self._envelopes: dict[str, MailboxEnvelope] = {}
        self._path = self.mailbox_root / "mailbox.json"
        self._events_path = self.mailbox_root / "mailbox_events.jsonl"
        self._reader = JsonlTailReader(self._events_path)
        self._lock = threading.RLock()
        self._pending_local_events: list[tuple[str, str, float | None]] = []
        self._load()

    def enqueue(self, envelope: MailboxEnvelope) -> None:
        with self._lock:
            self._sync()
            self._append_event("enqueue", envelope=envelope)
            self._envelopes[envelope.envelope_id] = envelope
            self._pending_local_events.append(("enqueue", envelope.envelope_id, None))

    def mark_delivered(self, envelope_id: str) -> None:
        with self._lock:
            self._sync()
            envelope = self._envelopes.get(envelope_id)
            if envelope is None:
                return
            timestamp = time.time()
            self._append_event("mark_delivered", envelope_id=envelope_id, timestamp=timestamp)
            envelope.delivery_count += 1
            envelope.last_delivery_at = timestamp
            self._pending_local_events.append(("mark_delivered", envelope_id, timestamp))

    def ack(self, envelope_id: str) -> None:
        with self._lock:
            self._sync()
            envelope = self._envelopes.get(envelope_id)
            if envelope is None:
                return
            timestamp = time.time()
            self._append_event("ack", envelope_id=envelope_id, timestamp=timestamp)
            envelope.acknowledged_at = timestamp
            self._pending_local_events.append(("ack", envelope_id, timestamp))

    def redeliver_unacked(self, *, stale_seconds: int) -> list[MailboxEnvelope]:
        self._sync()
        now = time.time()
        return [
            envelope
            for envelope in self.pending()
            if envelope.acknowledged_at is None
            and (now - (envelope.last_delivery_at or envelope.created_at)) >= stale_seconds
        ]

    def list_for_runtime(self, runtime_id: str) -> list[MailboxEnvelope]:
        self._sync()
        return sorted(
            [item for item in self._envelopes.values() if item.to_runtime_id == runtime_id],
            key=lambda item: (item.created_at, item.envelope_id),
        )

    def pending(self) -> list[MailboxEnvelope]:
        self._sync()
        return sorted(
            [item for item in self._envelopes.values() if item.acknowledged_at is None],
            key=lambda item: (item.created_at, item.envelope_id),
        )

    def pending_for_runtime(self, runtime_id: str) -> list[MailboxEnvelope]:
        return [item for item in self.list_for_runtime(runtime_id) if item.acknowledged_at is None]

    def all(self) -> list[MailboxEnvelope]:
        self._sync()
        return sorted(
            self._envelopes.values(),
            key=lambda item: (item.created_at, item.envelope_id),
        )

    def latest_cursor(
        self,
        runtime_id: str,
        *,
        message_types: set[str] | None = None,
        include_acked: bool = True,
    ) -> MailboxCursor:
        envelopes = self.list_for_runtime(runtime_id)
        if not include_acked:
            envelopes = [item for item in envelopes if item.acknowledged_at is None]
        if message_types is not None:
            envelopes = [item for item in envelopes if item.message_type in message_types]
        if not envelopes:
            return MailboxCursor()
        return MailboxCursor(
            last_created_at=envelopes[-1].created_at,
            last_envelope_id=envelopes[-1].envelope_id,
        )

    def list_since(
        self,
        runtime_id: str,
        *,
        cursor: MailboxCursor | None = None,
        message_types: set[str] | None = None,
        include_acked: bool = False,
    ) -> tuple[list[MailboxEnvelope], MailboxCursor]:
        current_cursor = cursor or MailboxCursor()
        envelopes = self.list_for_runtime(runtime_id)
        if not include_acked:
            envelopes = [item for item in envelopes if item.acknowledged_at is None]
        if message_types is not None:
            envelopes = [item for item in envelopes if item.message_type in message_types]
        unseen = [
            item
            for item in envelopes
            if (item.created_at, item.envelope_id)
            > (current_cursor.last_created_at, current_cursor.last_envelope_id)
        ]
        next_cursor = current_cursor
        if unseen:
            next_cursor = current_cursor.advance(unseen[-1])
        return unseen, next_cursor

    async def wait_for_runtime(
        self,
        runtime_id: str,
        *,
        message_types: set[str] | None = None,
        include_acked: bool = False,
        ack: bool = False,
        poll_interval: float = 0.05,
        timeout: float | None = None,
        start_cursor: MailboxCursor | None = None,
    ) -> tuple[list[MailboxEnvelope], MailboxCursor]:
        cursor = start_cursor or MailboxCursor()
        deadline = (time.monotonic() + timeout) if timeout is not None else None
        while True:
            envelopes, cursor = await asyncio.to_thread(
                self.list_since,
                runtime_id,
                cursor=cursor,
                message_types=message_types,
                include_acked=include_acked,
            )
            if envelopes:
                if ack:
                    for envelope in envelopes:
                        await asyncio.to_thread(self.ack, envelope.envelope_id)
                return envelopes, cursor
            if deadline is not None and time.monotonic() >= deadline:
                return [], cursor
            await asyncio.sleep(poll_interval)

    async def watch_for_runtime(
        self,
        runtime_id: str,
        *,
        message_types: set[str] | None = None,
        include_acked: bool = False,
        ack: bool = False,
        poll_interval: float = 0.05,
        start_cursor: MailboxCursor | None = None,
    ) -> AsyncIterator[MailboxEnvelope]:
        cursor = start_cursor or MailboxCursor()
        while True:
            envelopes, cursor = await self.wait_for_runtime(
                runtime_id,
                message_types=message_types,
                include_acked=include_acked,
                ack=ack,
                poll_interval=poll_interval,
                start_cursor=cursor,
            )
            for envelope in envelopes:
                yield envelope

    def _load(self) -> None:
        with self._lock:
            self._envelopes = {}
            self._reader.reset()
            if self._events_path.exists():
                self._sync()
                return
            if not self.envelopes_root.exists():
                return
            import json

            for file_path in sorted(self.envelopes_root.glob("*.json")):
                raw = file_path.read_text(encoding="utf-8")
                if not raw.strip():
                    continue
                payload = json.loads(raw)
                envelope = MailboxEnvelope.from_dict(payload)
                self._envelopes[envelope.envelope_id] = envelope

    def _sync(self) -> None:
        with self._lock:
            for event in self._reader.read_new():
                if self._consume_local_event(event):
                    continue
                self._apply_event(event)

    def _consume_local_event(self, event: dict[str, Any]) -> bool:
        kind = str(event.get("kind") or "")
        if kind == "enqueue":
            payload = event.get("envelope")
            envelope_id = str(payload.get("envelope_id") or "") if isinstance(payload, dict) else ""
            timestamp = None
        else:
            envelope_id = str(event.get("envelope_id") or "")
            raw_timestamp = event.get("timestamp")
            timestamp = float(raw_timestamp) if raw_timestamp is not None else None
        key = (kind, envelope_id, timestamp)
        try:
            index = self._pending_local_events.index(key)
        except ValueError:
            return False
        del self._pending_local_events[index]
        return True

    def _append_event(
        self,
        kind: str,
        *,
        envelope: MailboxEnvelope | None = None,
        envelope_id: str | None = None,
        timestamp: float | None = None,
    ) -> None:
        payload: dict[str, Any] = {"kind": kind, "created_at": time.time()}
        if envelope is not None:
            payload["envelope"] = envelope.to_dict()
        if envelope_id is not None:
            payload["envelope_id"] = envelope_id
        if timestamp is not None:
            payload["timestamp"] = timestamp
        append_jsonl_sync(self._events_path, payload)

    def _apply_event(self, event: dict[str, Any]) -> None:
        kind = str(event.get("kind") or "")
        if kind == "enqueue":
            payload = event.get("envelope")
            if isinstance(payload, dict):
                envelope = MailboxEnvelope.from_dict(payload)
                self._envelopes[envelope.envelope_id] = envelope
            return
        envelope_id = str(event.get("envelope_id") or "")
        envelope = self._envelopes.get(envelope_id)
        if envelope is None:
            return
        if kind == "mark_delivered":
            envelope.delivery_count += 1
            envelope.last_delivery_at = float(event.get("timestamp") or time.time())
            return
        if kind == "ack":
            envelope.acknowledged_at = float(event.get("timestamp") or time.time())
