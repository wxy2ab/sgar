from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from ..runtime import AgentMessage, AgentRuntime
    from ..task_model import AgentTask


@dataclass(slots=True)
class BackendHandle:
    runtime_id: str
    backend_name: str
    process_id: int | None = None
    endpoint: str | None = None
    output_path: str | None = None


class RuntimeController(Protocol):
    task: "AgentTask"
    handle: BackendHandle

    async def start(self, prompt: str) -> dict[str, Any]:
        ...

    async def send_message(
        self,
        message: "AgentMessage",
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        ...

    async def stop(self, reason: str) -> None:
        ...

    async def collect_status(self) -> dict[str, Any]:
        ...

    async def apply_shared_state(
        self,
        *,
        shared_context: dict[str, Any],
        shared_allowed_paths: list[str],
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        ...


class RuntimeBackend(Protocol):
    name: str

    async def create_controller(
        self,
        *,
        runtime: "AgentRuntime",
        run_in_background: bool,
        runtime_root: Path,
    ) -> RuntimeController:
        ...
