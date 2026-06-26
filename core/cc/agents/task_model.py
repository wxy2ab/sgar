from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import time
from typing import Any
import uuid


class AgentTaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_MESSAGE = "waiting_message"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"


_ALLOWED_STATUS_TRANSITIONS: dict[AgentTaskStatus, set[AgentTaskStatus]] = {
    AgentTaskStatus.PENDING: {
        AgentTaskStatus.PENDING,
        AgentTaskStatus.RUNNING,
        AgentTaskStatus.FAILED,
        AgentTaskStatus.KILLED,
    },
    AgentTaskStatus.RUNNING: {
        AgentTaskStatus.RUNNING,
        AgentTaskStatus.WAITING_MESSAGE,
        AgentTaskStatus.COMPLETED,
        AgentTaskStatus.FAILED,
        AgentTaskStatus.KILLED,
    },
    AgentTaskStatus.WAITING_MESSAGE: {
        AgentTaskStatus.WAITING_MESSAGE,
        AgentTaskStatus.RUNNING,
        AgentTaskStatus.COMPLETED,
        AgentTaskStatus.FAILED,
        AgentTaskStatus.KILLED,
    },
    AgentTaskStatus.COMPLETED: {AgentTaskStatus.COMPLETED},
    AgentTaskStatus.FAILED: {AgentTaskStatus.FAILED},
    AgentTaskStatus.KILLED: {AgentTaskStatus.KILLED},
}


@dataclass(slots=True)
class AgentTask:
    task_id: str
    runtime_id: str
    agent_type: str
    backend: str
    status: AgentTaskStatus
    prompt_language: str
    title: str = ""
    input_payload: dict[str, Any] = field(default_factory=dict)
    result_payload: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload

    @classmethod
    def create(
        cls,
        *,
        agent_type: str,
        backend: str,
        prompt_language: str,
        title: str = "",
        input_payload: dict[str, Any] | None = None,
    ) -> "AgentTask":
        token = uuid.uuid4().hex[:10]
        now = time.time()
        return cls(
            task_id=f"task_{token}",
            runtime_id=f"rt_{token}",
            agent_type=agent_type,
            backend=backend,
            status=AgentTaskStatus.PENDING,
            prompt_language=prompt_language,
            title=title,
            input_payload=input_payload or {},
            created_at=now,
            updated_at=now,
        )


def is_terminal_status(status: AgentTaskStatus | str) -> bool:
    value = status.value if isinstance(status, AgentTaskStatus) else status
    return value in {
        AgentTaskStatus.COMPLETED.value,
        AgentTaskStatus.FAILED.value,
        AgentTaskStatus.KILLED.value,
    }


def can_transition_status(
    current: AgentTaskStatus | str,
    target: AgentTaskStatus | str,
) -> bool:
    current_status = current if isinstance(current, AgentTaskStatus) else AgentTaskStatus(str(current))
    target_status = target if isinstance(target, AgentTaskStatus) else AgentTaskStatus(str(target))
    return target_status in _ALLOWED_STATUS_TRANSITIONS[current_status]
