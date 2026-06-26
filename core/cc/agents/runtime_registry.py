from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Protocol


class RuntimeControllerLike(Protocol):
    task: object

    async def send_message(self, message) -> dict: ...
    async def stop(self, reason: str) -> None: ...
    async def collect_status(self) -> dict: ...


class InProcessRuntimeRegistry:
    def __init__(self, runtime_root: str | Path) -> None:
        self.runtime_root = Path(runtime_root)
        self.registry_key = str(self.runtime_root.resolve())
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self._runtimes: dict[str, RuntimeControllerLike] = {}
        self._background_tasks: dict[str, asyncio.Task] = {}

    def register(self, runtime: RuntimeControllerLike) -> None:
        self._runtimes[runtime.task.runtime_id] = runtime

    def get(self, runtime_id: str) -> RuntimeControllerLike | None:
        return self._runtimes.get(runtime_id)

    def unregister(self, runtime_id: str) -> None:
        self._runtimes.pop(runtime_id, None)
        self._background_tasks.pop(runtime_id, None)
        _remove_registry_if_idle(self.registry_key, self)

    def get_by_task_id(self, task_id: str) -> RuntimeControllerLike | None:
        for runtime in self._runtimes.values():
            if runtime.task.task_id == task_id:
                return runtime
        return None

    def list_runtime_ids(self) -> list[str]:
        return sorted(self._runtimes)

    def register_background_task(self, runtime_id: str, task: asyncio.Task) -> None:
        self._background_tasks[runtime_id] = task
        task.add_done_callback(self._done_callback(runtime_id))

    def get_background_task(self, runtime_id: str) -> asyncio.Task | None:
        return self._background_tasks.get(runtime_id)

    def _done_callback(self, runtime_id: str):
        def _callback(_: asyncio.Task) -> None:
            self._background_tasks.pop(runtime_id, None)
            _remove_registry_if_idle(self.registry_key, self)

        return _callback


_REGISTRIES: dict[str, InProcessRuntimeRegistry] = {}


def _remove_registry_if_idle(key: str, registry: InProcessRuntimeRegistry) -> None:
    if registry._runtimes or registry._background_tasks:
        return
    if _REGISTRIES.get(key) is registry:
        _REGISTRIES.pop(key, None)


def get_in_process_runtime_registry(runtime_root: str | Path) -> InProcessRuntimeRegistry:
    key = str(Path(runtime_root).resolve())
    registry = _REGISTRIES.get(key)
    if registry is None:
        registry = InProcessRuntimeRegistry(key)
        _REGISTRIES[key] = registry
    return registry


def shutdown_all_runtime_registries() -> None:
    for registry in list(_REGISTRIES.values()):
        for runtime_id, runtime in list(registry._runtimes.items()):
            close_sync = getattr(runtime, "close_sync", None)
            if callable(close_sync):
                close_sync()
            registry.unregister(runtime_id)
        for task in list(registry._background_tasks.values()):
            task.cancel()
        registry._background_tasks.clear()
    _REGISTRIES.clear()
