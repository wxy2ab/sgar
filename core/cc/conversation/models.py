from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import time
from typing import Any


@dataclass(slots=True)
class SessionMessage:
    message_id: str
    turn_id: str
    role: str
    content: str
    kind: str = "text"
    name: str | None = None
    tool_name: str | None = None
    tool_use_id: str | None = None
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SessionMessage":
        _required_defaults = {
            "message_id": "",
            "turn_id": "",
            "role": "unknown",
            "content": "",
        }
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in payload.items() if k in known}
        for key, default in _required_defaults.items():
            if key not in filtered:
                filtered[key] = default
        return cls(**filtered)


@dataclass(slots=True)
class SystemPromptParts:
    primary: str
    append: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def combined(self) -> str:
        blocks = [self.primary, *self.append]
        return "\n\n".join(block.strip() for block in blocks if block and block.strip())


@dataclass(slots=True)
class SessionEvent:
    event_type: str
    turn_id: str
    message: SessionMessage | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass(slots=True)
class TurnRecord:
    turn_id: str
    state: str
    tool_call_count: int = 0
    compact_applied: bool = False
    continue_count: int = 0
    error_code: str | None = None
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
