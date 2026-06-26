from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
import time
from typing import Any
import uuid

from ..config import CCConfig


class TurnState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    WAITING_TOOL = "waiting_tool"
    COMPACTING = "compacting"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass(slots=True)
class SessionMetadata:
    agent_id: str | None = None
    team_id: str | None = None
    parent_task_id: str | None = None
    labels: list[str] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None = None) -> "SessionMetadata":
        data = dict(payload or {})
        return cls(
            agent_id=data.get("agent_id"),
            team_id=data.get("team_id"),
            parent_task_id=data.get("parent_task_id"),
            labels=list(data.get("labels") or []),
            state=dict(data.get("state") or {}),
        )


@dataclass(slots=True)
class QuerySession:
    session_id: str
    cwd: str
    config: CCConfig
    model_name: str
    permission_mode: str
    prompt_language: str
    agent_mode: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: SessionMetadata = field(default_factory=SessionMetadata)
    active_turn_id: str | None = None

    def touch(self) -> None:
        self.updated_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["config"] = self.config.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "QuerySession":
        data = dict(payload)
        config_payload = data.get("config") or {}
        return cls(
            session_id=str(data["session_id"]),
            cwd=str(Path(data["cwd"]).resolve()),
            config=CCConfig.from_mapping(config_payload),
            model_name=str(data["model_name"]),
            permission_mode=str(data["permission_mode"]),
            agent_mode=str(data.get("agent_mode") or ""),
            prompt_language=str(data["prompt_language"]),
            created_at=float(data.get("created_at") or time.time()),
            updated_at=float(data.get("updated_at") or time.time()),
            metadata=SessionMetadata.from_dict(data.get("metadata")),
            active_turn_id=data.get("active_turn_id"),
        )

    @property
    def session_dir(self) -> Path:
        return self.config.session_root_path(self.cwd) / self.session_id


class SessionFactory:
    def __init__(self, config: CCConfig) -> None:
        self.config = config

    def create(
        self,
        *,
        cwd: str | None = None,
        model_name: str | None = None,
        agent_id: str | None = None,
        team_id: str | None = None,
        parent_task_id: str | None = None,
    ) -> QuerySession:
        now = time.time()
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        metadata = SessionMetadata(
            agent_id=agent_id,
            team_id=team_id,
            parent_task_id=parent_task_id,
        )
        return QuerySession(
            session_id=session_id,
            cwd=str(Path(cwd or Path.cwd()).resolve()),
            config=self.config,
            model_name=model_name or self.config.default_llm_client,
            permission_mode=self.config.permission_mode,
            agent_mode=self.config.agent_mode,
            prompt_language=self.config.prompt_language,
            created_at=now,
            updated_at=now,
            metadata=metadata,
        )
