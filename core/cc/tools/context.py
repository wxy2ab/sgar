from __future__ import annotations

from dataclasses import replace
from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING
from typing import Any

from ..config import CCConfig

if TYPE_CHECKING:
    from ..providers import Environment


@dataclass(slots=True)
class ToolPermissionSnapshot:
    mode: str
    allow_dangerous_commands: bool = False
    allowed_paths: list[str] = field(default_factory=list)
    denied_paths: list[str] = field(default_factory=list)


@dataclass(slots=True)
class InProgressToolTracker:
    active_tool_use_ids: set[str] = field(default_factory=set)

    def start(self, tool_use_id: str) -> None:
        self.active_tool_use_ids.add(tool_use_id)

    def finish(self, tool_use_id: str) -> None:
        self.active_tool_use_ids.discard(tool_use_id)


@dataclass(slots=True)
class ToolUseContext:
    session_id: str
    turn_id: str
    cwd: str
    prompt_language: str
    config: CCConfig
    permissions: ToolPermissionSnapshot
    app_state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    tracker: InProgressToolTracker = field(default_factory=InProgressToolTracker)
    environment: Environment | None = None

    def get_app_state(self) -> dict[str, Any]:
        return dict(self.app_state)

    def to_permission_context_snapshot(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "cwd": self.cwd,
            "prompt_language": self.prompt_language,
            "permission_mode": self.permissions.mode,
            "allowed_paths": list(self.permissions.allowed_paths),
            "denied_paths": list(self.permissions.denied_paths),
        }

    def set_app_state(self, updater: Callable[[dict[str, Any]], dict[str, Any]] | dict[str, Any]) -> "ToolUseContext":
        next_state = updater(self.get_app_state()) if callable(updater) else dict(updater)
        return replace(self, app_state=next_state)

    def with_updates(self, **changes: Any) -> "ToolUseContext":
        return replace(self, **changes)

    def get_fs(self) -> Any:
        """Return the ``FileSystemProvider`` from the environment, or ``None``."""
        return self.environment.fs if self.environment is not None else None

    def get_shell(self) -> Any:
        """Return the ``CommandProvider`` from the environment, or ``None``."""
        return self.environment.shell if self.environment is not None else None
