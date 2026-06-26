from __future__ import annotations

import asyncio
from pathlib import Path
import time

from ..swarm.mailbox import MailboxEnvelope, MailboxStore
from .base import BackendHandle, RuntimeBackend, RuntimeController


class InProcessController:
    def __init__(self, runtime, runtime_root: Path, *, keep_alive: bool) -> None:
        self.runtime = runtime
        self.task = runtime.task
        self.task_manager = runtime.task_manager
        self.keep_alive = keep_alive
        session = runtime.query_engine.session
        team_id = session.metadata.team_id
        lead_runtime_id = session.metadata.state.get("team_lead_runtime_id")
        self.team_mailbox = (
            MailboxStore(runtime_root / "teams" / str(team_id) / "mailbox")
            if team_id is not None and lead_runtime_id
            else None
        )
        self.lead_runtime_id = str(lead_runtime_id) if lead_runtime_id else None
        self.handle = BackendHandle(
            runtime_id=runtime.task.runtime_id,
            backend_name="in_process",
            process_id=None,
            output_path=None,
        )

    async def start(self, prompt: str) -> dict[str, object]:
        return await self.runtime.start(prompt, keep_alive=self.keep_alive)

    async def send_message(self, message, *, timeout_seconds: float | None = None) -> dict[str, object]:
        del timeout_seconds
        self._publish_team_event(
            "status_update",
            {
                "message_id": message.message_id,
                "status": "running",
                "description": message.metadata.get("description"),
            },
        )
        self._publish_team_event(
            "partial_result",
            {
                "message_id": message.message_id,
                "partial_text": f"Worker {self.task.runtime_id} started task {message.message_id}",
                "description": message.metadata.get("description"),
            },
        )
        result = await self.runtime.send_message(message)
        self._publish_team_event(
            "final_result",
            {
                "message_id": message.message_id,
                "description": message.metadata.get("description"),
                "final_text": result.get("final_text", ""),
                "status": result.get("status"),
            },
        )
        self._publish_team_event(
            "assignment_completed",
            {
                "message_id": message.message_id,
                "description": message.metadata.get("description"),
                "final_text": result.get("final_text", ""),
                "status": result.get("status"),
            },
        )
        return result

    async def stop(self, reason: str) -> None:
        await self.runtime.stop(reason)

    async def collect_status(self) -> dict[str, object]:
        return await self.runtime.collect_status()

    async def apply_shared_state(
        self,
        *,
        shared_context: dict[str, object],
        shared_allowed_paths: list[str],
        timeout_seconds: float | None = None,
    ) -> dict[str, object]:
        del timeout_seconds
        session = self.runtime.query_engine.session
        session.metadata.state["team_shared_context"] = dict(shared_context)
        session.metadata.state["team_shared_allowed_paths"] = list(shared_allowed_paths)
        session.metadata.state["allowed_paths"] = list(shared_allowed_paths)
        return {
            "runtime_id": self.task.runtime_id,
            "backend": self.handle.backend_name,
            "shared_context": dict(shared_context),
            "shared_allowed_paths": list(shared_allowed_paths),
        }

    def _publish_team_event(self, message_type: str, payload: dict[str, object]) -> None:
        if self.team_mailbox is None or self.lead_runtime_id is None:
            return
        message_token = str(payload.get("message_id") or "evt")
        self.team_mailbox.enqueue(
            MailboxEnvelope(
                envelope_id=f"env_{message_type}_{self.task.runtime_id}_{message_token}_{int(time.time() * 1000)}",
                team_id=self.runtime.query_engine.session.metadata.team_id,
                from_runtime_id=self.task.runtime_id,
                to_runtime_id=self.lead_runtime_id,
                message_type=message_type,
                payload=dict(payload),
            )
        )


class InProcessBackend(RuntimeBackend):
    name = "in_process"

    async def create_controller(
        self,
        *,
        runtime,
        run_in_background: bool,
        runtime_root: Path,
    ) -> RuntimeController:
        return InProcessController(runtime, runtime_root, keep_alive=run_in_background)
