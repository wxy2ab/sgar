from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Protocol

from ..jsonl import JsonlTailReader
from .swarm.mailbox import MailboxEnvelope, MailboxStore


class RuntimeTransport(Protocol):
    async def enqueue_message(self, envelope: MailboxEnvelope) -> None: ...
    async def wait_for_response(self, message_id: str, *, timeout: float) -> dict[str, Any]: ...
    def read_status_payload(self) -> dict[str, Any]: ...
    def request_stop(self, reason: str) -> None: ...


class FileRuntimeTransport:
    def __init__(
        self,
        *,
        control_dir: str | Path,
        mailbox: MailboxStore | None = None,
        poll_interval: float = 0.02,
    ) -> None:
        self.control_dir = Path(control_dir)
        self.control_dir.mkdir(parents=True, exist_ok=True)
        self.mailbox = mailbox or MailboxStore(self.control_dir / "mailbox")
        self.responses_path = self.control_dir / "responses.jsonl"
        self.status_path = self.control_dir / "status.json"
        self.stop_path = self.control_dir / "stop.flag"
        self.poll_interval = poll_interval
        self._response_reader = JsonlTailReader(self.responses_path)
        self._responses: dict[str, dict[str, Any]] = {}

    async def enqueue_message(self, envelope: MailboxEnvelope) -> None:
        self.mailbox.enqueue(envelope)

    async def wait_for_response(self, message_id: str, *, timeout: float) -> dict[str, Any]:
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            self._sync_responses()
            response = self._responses.get(message_id)
            if response is not None:
                return dict(response)
            await asyncio.sleep(self.poll_interval)
        raise TimeoutError(f"Timed out waiting for response: {message_id}")

    def read_status_payload(self) -> dict[str, Any]:
        if not self.status_path.exists():
            return {}
        try:
            import json

            return json.loads(self.status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def request_stop(self, reason: str) -> None:
        self.stop_path.write_text(reason, encoding="utf-8")

    def _sync_responses(self) -> None:
        for payload in self._response_reader.read_new():
            message_id = str(payload.get("message_id") or "")
            if message_id:
                self._responses[message_id] = payload
