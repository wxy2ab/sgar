from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import time
from typing import Any
import asyncio
from typing import TYPE_CHECKING

from ..conversation.models import SessionEvent
from ..observability import EventRecord, JsonlAuditLogger
from .definitions import AgentDefinition
from .task_manager import TaskManager
from .task_model import AgentTask, AgentTaskStatus

if TYPE_CHECKING:
    from ..conversation.query_engine import QueryEngine


@dataclass(slots=True)
class AgentMessage:
    message_id: str
    from_agent_id: str
    to_agent_id: str
    team_id: str | None
    kind: str
    content: Any
    created_at: float = field(default_factory=time.time)
    correlation_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentRuntime:
    definition: AgentDefinition
    task: AgentTask
    query_engine: "QueryEngine"
    task_manager: TaskManager
    events: deque[SessionEvent] = field(init=False)
    final_text: str | None = None
    event_count_total: int = 0
    event_count_dropped: int = 0
    last_run_event_count: int = 0
    audit_logger: JsonlAuditLogger = field(init=False)
    _run_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        self.events = deque(maxlen=self.query_engine.session.config.agent_runtime_event_buffer_size)
        self.audit_logger = JsonlAuditLogger(
            self.query_engine.session.config.runtime_root_path(self.query_engine.session.cwd)
            / "audit"
            / "agent_events.jsonl"
        )

    async def start(self, prompt: str, *, keep_alive: bool = False) -> dict[str, Any]:
        async with self._run_lock:
            self.final_text = None
            self.last_run_event_count = 0
            running_payload = {"waiting_reason": "waiting_llm_response"} if keep_alive else None
            self.task_manager.update_task_status(
                self.task.task_id,
                AgentTaskStatus.RUNNING,
                result_payload=running_payload,
            )
            self._record_audit_event(
                "agent_run_started",
                prompt_preview=prompt[:200],
            )
            timeout = getattr(
                self.query_engine.session.config, "max_turn_timeout_seconds", None,
            )
            try:
                # Wall-clock guard around the entire submit_message iteration.
                # QueryEngine.submit_message also tracks turn elapsed time,
                # but its check only fires when an event arrives; if a child
                # tool hangs (e.g. a subprocess that never emits output), no
                # events flow and the elapsed-time guard never trips. This
                # asyncio.wait_for forces a hard cancellation deadline so a
                # hung helper agent surfaces as a failure rather than blocking
                # the parent ccx.agent dispatch indefinitely.
                if timeout is not None and timeout > 0:
                    await asyncio.wait_for(
                        self._consume_query_events(prompt),
                        timeout=float(timeout),
                    )
                else:
                    await self._consume_query_events(prompt)
            except asyncio.TimeoutError as exc:
                error_message = (
                    f"Agent run wall-clock timeout after {float(timeout):g}s "
                    f"(max_turn_timeout_seconds); inner tool may be hung."
                )
                self._record_audit_event(
                    "agent_run_failed",
                    success=False,
                    error_code="QE1008",
                    error=error_message,
                    last_run_event_count=self.last_run_event_count,
                    timeout_seconds=float(timeout) if timeout else None,
                )
                self.task_manager.update_task_status(
                    self.task.task_id,
                    AgentTaskStatus.FAILED,
                    result_payload={
                        "error": error_message,
                        "error_code": "QE1008",
                    },
                )
                raise
            except Exception as exc:
                self._record_audit_event(
                    "agent_run_failed",
                    success=False,
                    error_code=str(getattr(exc, "error_code", "") or "") or None,
                    error=str(exc),
                    last_run_event_count=self.last_run_event_count,
                )
                self.task_manager.update_task_status(
                    self.task.task_id,
                    AgentTaskStatus.FAILED,
                    result_payload={"error": str(exc)},
                )
                raise
            completed_status = AgentTaskStatus.WAITING_MESSAGE if keep_alive else AgentTaskStatus.COMPLETED
            completed_payload = (
                {"final_text": self.final_text or "", "waiting_reason": "waiting_message"}
                if keep_alive
                else {"final_text": self.final_text or ""}
            )
            self.task_manager.update_task_status(
                self.task.task_id,
                completed_status,
                result_payload=completed_payload,
            )
            self._record_audit_event(
                "agent_run_completed",
                success=True,
                final_text=self.final_text or "",
                last_run_event_count=self.last_run_event_count,
            )
            return {
                "task_id": self.task.task_id,
                "runtime_id": self.task.runtime_id,
                "status": self.task.status.value,
                "final_text": self.final_text or "",
            }

    async def _consume_query_events(self, prompt: str) -> None:
        async for event in self.query_engine.submit_message(prompt):
            self._append_event(event)
            if event.message and event.message.role == "assistant" and event.message.kind == "assistant_text":
                self.final_text = event.message.content

    async def send_message(self, message: AgentMessage) -> dict[str, Any]:
        return await self.start(str(message.content), keep_alive=True)

    async def stop(self, reason: str) -> None:
        self.task_manager.update_task_status(
            self.task.task_id,
            AgentTaskStatus.KILLED,
            result_payload={"reason": reason},
        )

    async def collect_status(self) -> dict[str, Any]:
        task = self.task_manager.get(self.task.task_id) or self.task
        return {
            "task_id": task.task_id,
            "runtime_id": task.runtime_id,
            "status": task.status.value,
            "agent_id": self.definition.agent_id,
            "final_text": self.final_text,
            "waiting_reason": task.result_payload.get("waiting_reason"),
            "event_count_total": self.event_count_total,
            "event_count_dropped": self.event_count_dropped,
            "recent_event_count": len(self.events),
            "last_run_event_count": self.last_run_event_count,
        }

    def _append_event(self, event: SessionEvent) -> None:
        if self.events.maxlen is not None and len(self.events) >= self.events.maxlen:
            self.event_count_dropped += 1
        self.events.append(event)
        self.event_count_total += 1
        self.last_run_event_count += 1

    def _record_audit_event(
        self,
        event_type: str,
        *,
        success: bool | None = None,
        error_code: str | None = None,
        **details: Any,
    ) -> None:
        self.audit_logger.append(
            EventRecord(
                event_type=event_type,
                session_id=self.query_engine.session.session_id,
                task_id=self.task.task_id,
                success=success,
                error_code=error_code,
                details={
                    "runtime_id": self.task.runtime_id,
                    "agent_id": self.definition.agent_id,
                    "team_id": self.query_engine.session.metadata.team_id,
                    **details,
                },
            )
        )
