from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from typing import Any

from ...errors import AgentTaskError
from ..runtime_transport import FileRuntimeTransport
from ..swarm.mailbox import MailboxEnvelope, MailboxStore
from ..task_model import AgentTaskStatus, is_terminal_status
from .base import BackendHandle, RuntimeBackend, RuntimeController


class LocalSubprocessController:
    def __init__(
        self,
        *,
        runtime,
        runtime_root: Path,
        run_in_background: bool,
    ) -> None:
        self.runtime = runtime
        self.task = runtime.task
        self.runtime_root = runtime_root
        self.run_in_background = run_in_background
        self.task_manager = runtime.task_manager
        self.control_dir = runtime_root / "subprocess" / self.task.runtime_id
        self.control_dir.mkdir(parents=True, exist_ok=True)
        self.launch_path = self.control_dir / "launch.json"
        self.transport = FileRuntimeTransport(
            control_dir=self.control_dir,
            mailbox=MailboxStore(self.control_dir / "mailbox"),
        )
        self.mailbox = self.transport.mailbox
        self.responses_path = self.transport.responses_path
        self.status_path = self.transport.status_path
        self.stop_path = self.transport.stop_path
        self.stdout_path = self.control_dir / "stdout.log"
        self.stderr_path = self.control_dir / "stderr.log"
        self.process: asyncio.subprocess.Process | None = None
        self._started = False
        self._initial_message_id = f"msg_{uuid.uuid4().hex[:10]}"
        self._stdout_handle = None
        self._stderr_handle = None
        self.handle = BackendHandle(
            runtime_id=self.task.runtime_id,
            backend_name="local_subprocess",
            process_id=None,
            output_path=str(self.responses_path),
        )

    async def start(self, prompt: str) -> dict[str, Any]:
        if not self._started:
            await self._spawn_process(prompt)
        if self.run_in_background:
            await self._wait_for_startup()
            return {
                "task_id": self.task.task_id,
                "runtime_id": self.task.runtime_id,
                "status": self._current_status(),
                "background": True,
                "backend": self.handle.backend_name,
                "process_id": self.handle.process_id,
            }
        result = await self._wait_for_response(
            self._initial_message_id,
            timeout=self.runtime.query_engine.session.config.swarm_assignment_response_timeout_seconds,
        )
        self._close_streams()
        return result

    async def send_message(self, message, *, timeout_seconds: float | None = None) -> dict[str, Any]:
        if not self._started:
            raise AgentTaskError("Subprocess runtime is not started.", error_code="AG1002")
        self.task_manager.load_tasks_from_disk()
        current_task = self.task_manager.get(self.task.task_id) or self.task
        if is_terminal_status(current_task.status):
            raise AgentTaskError("Task is already terminal.", error_code="AG1003")
        payload = {
            "message_id": message.message_id,
            "from_agent_id": message.from_agent_id,
            "to_agent_id": message.to_agent_id,
            "team_id": message.team_id,
            "kind": message.kind,
            "content": str(message.content),
            "created_at": message.created_at,
            "correlation_id": message.correlation_id,
            "metadata": dict(message.metadata),
        }
        await self.transport.enqueue_message(
            MailboxEnvelope(
                envelope_id=f"env_{message.message_id}",
                team_id=message.team_id,
                from_runtime_id=self.runtime.query_engine.session.metadata.agent_id or "main",
                to_runtime_id=self.task.runtime_id,
                message_type=message.kind,
                payload=payload,
            )
        )
        response_timeout = timeout_seconds or self.runtime.query_engine.session.config.swarm_assignment_response_timeout_seconds
        return await self._wait_for_response(message.message_id, timeout=response_timeout)

    async def stop(self, reason: str) -> None:
        self.transport.request_stop(reason)
        if self.process is not None and self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
        self._close_streams()
        self.task_manager.load_tasks_from_disk()
        current_task = self.task_manager.get(self.task.task_id)
        if current_task is not None and not is_terminal_status(current_task.status):
            self.task_manager.update_task_status(
                self.task.task_id,
                AgentTaskStatus.KILLED,
                result_payload={"reason": reason, "waiting_reason": "stopped"},
            )

    async def collect_status(self) -> dict[str, Any]:
        self.task_manager.load_tasks_from_disk()
        task = self.task_manager.get(self.task.task_id) or self.task
        status_payload = self._read_status_payload()
        return {
            "task_id": task.task_id,
            "runtime_id": task.runtime_id,
            "status": task.status.value,
            "agent_id": self.runtime.definition.agent_id,
            "backend": self.handle.backend_name,
            "process_id": self.handle.process_id,
            "worker_state": status_payload.get("worker_state"),
            "waiting_reason": task.result_payload.get("waiting_reason") or status_payload.get("waiting_reason"),
            "last_message_id": status_payload.get("last_message_id"),
            "pending_message_count": len(self.mailbox.pending_for_runtime(self.task.runtime_id)),
            "team_shared_context": status_payload.get("team_shared_context"),
            "team_shared_allowed_paths": status_payload.get("team_shared_allowed_paths"),
            "final_text": task.result_payload.get("final_text"),
        }

    async def apply_shared_state(
        self,
        *,
        shared_context: dict[str, Any],
        shared_allowed_paths: list[str],
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        if not self._started:
            raise AgentTaskError("Subprocess runtime is not started.", error_code="AG1002")
        message_id = f"sync_{uuid.uuid4().hex[:10]}"
        await self.transport.enqueue_message(
            MailboxEnvelope(
                envelope_id=f"env_{message_id}",
                team_id=self.runtime.query_engine.session.metadata.team_id,
                from_runtime_id="team_runtime",
                to_runtime_id=self.task.runtime_id,
                message_type="permission_sync",
                payload={
                    "message_id": message_id,
                    "kind": "permission_sync",
                    "content": "",
                    "shared_context": dict(shared_context),
                    "shared_allowed_paths": list(shared_allowed_paths),
                },
            )
        )
        response_timeout = timeout_seconds or self.runtime.query_engine.session.config.swarm_assignment_response_timeout_seconds
        return await self._wait_for_response(message_id, timeout=response_timeout)

    async def _spawn_process(self, prompt: str) -> None:
        session = self.runtime.query_engine.session
        launch_payload = {
            "task": self.task.to_dict(),
            "definition": asdict(self.runtime.definition),
            "session": session.to_dict(),
            "runtime_root": str(self.runtime_root),
            "initial_prompt": prompt,
            "initial_message_id": self._initial_message_id,
            "keep_alive": self.run_in_background,
        }
        self.launch_path.write_text(
            json.dumps(launch_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        repo_root = Path(__file__).resolve().parents[4]
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(repo_root) if not existing_pythonpath else f"{repo_root}{os.pathsep}{existing_pythonpath}"
        self._stdout_handle = self.stdout_path.open("a", encoding="utf-8")
        self._stderr_handle = self.stderr_path.open("a", encoding="utf-8")
        self.process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "core.cc.agents.backends.local_worker",
            str(self.launch_path),
            cwd=session.cwd,
            env=env,
            stdout=self._stdout_handle,
            stderr=self._stderr_handle,
        )
        self.handle.process_id = self.process.pid
        self._started = True

    async def _wait_for_startup(self, timeout: float = 10.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.process is not None and self.process.returncode is not None:
                stderr_tail = ""
                if self.stderr_path.exists():
                    stderr_tail = self.stderr_path.read_text(encoding="utf-8")[-1000:]
                self._close_streams()
                raise AgentTaskError(
                    f"Subprocess agent exited during startup. {stderr_tail}".strip(),
                    error_code="AG1002",
                )
            status_payload = self._read_status_payload()
            if status_payload.get("worker_state") in {"running", "waiting_message"}:
                return
            await asyncio.sleep(0.1)
        raise AgentTaskError("Timed out waiting for subprocess agent startup.", error_code="AG1002")

    async def _wait_for_response(self, message_id: str, *, timeout: float) -> dict[str, Any]:
        response_task = asyncio.create_task(
            self.transport.wait_for_response(message_id, timeout=timeout)
        )
        process_wait_task: asyncio.Task[int] | None = None
        if self.process is not None and self.process.returncode is None:
            process_wait_task = asyncio.create_task(self.process.wait())
        try:
            wait_set = {response_task}
            if process_wait_task is not None:
                wait_set.add(process_wait_task)
            done, pending = await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)
            if response_task in done:
                response = response_task.result()
                result = {
                    "task_id": self.task.task_id,
                    "runtime_id": self.task.runtime_id,
                    "status": response.get("status", self._current_status()),
                    "backend": self.handle.backend_name,
                    "final_text": response.get("final_text", ""),
                    "message_id": message_id,
                }
                for key in ("synced", "shared_context", "shared_allowed_paths", "kind"):
                    if key in response:
                        result[key] = response[key]
                return result
            if process_wait_task is not None and process_wait_task in done:
                self.task_manager.load_tasks_from_disk()
                task = self.task_manager.get(self.task.task_id) or self.task
                self._close_streams()
                if is_terminal_status(task.status):
                    return {
                        "task_id": task.task_id,
                        "runtime_id": task.runtime_id,
                        "status": task.status.value,
                        "backend": self.handle.backend_name,
                        "final_text": task.result_payload.get("final_text", ""),
                        "message_id": message_id,
                        "error_code": task.result_payload.get("error_code"),
                        "error": task.result_payload.get("error"),
                    }
                raise AgentTaskError("Subprocess agent exited unexpectedly.", error_code="AG1002")
            raise AgentTaskError(f"Timed out waiting for message response: {message_id}", error_code="AG1003")
        except TimeoutError as exc:
            raise AgentTaskError(f"Timed out waiting for message response: {message_id}", error_code="AG1003") from exc
        finally:
            response_task.cancel()
            if process_wait_task is not None:
                process_wait_task.cancel()

    def _read_status_payload(self) -> dict[str, Any]:
        return self.transport.read_status_payload()

    def _current_status(self) -> str:
        self.task_manager.load_tasks_from_disk()
        current = self.task_manager.get(self.task.task_id) or self.task
        return current.status.value

    def _close_streams(self) -> None:
        if self._stdout_handle is not None and not self._stdout_handle.closed:
            self._stdout_handle.close()
        if self._stderr_handle is not None and not self._stderr_handle.closed:
            self._stderr_handle.close()

    def close_sync(self) -> None:
        if self.process is not None and self.process.returncode is None:
            try:
                self.process.terminate()
            except ProcessLookupError:
                pass
        elif self.handle.process_id:
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(self.handle.process_id), "/T", "/F"],
                    capture_output=True,
                    check=False,
                )
            except Exception:
                pass
        self._close_streams()

    def __del__(self) -> None:
        try:
            self.close_sync()
        except Exception:
            pass


class LocalSubprocessBackend(RuntimeBackend):
    name = "local_subprocess"

    async def create_controller(
        self,
        *,
        runtime,
        run_in_background: bool,
        runtime_root: Path,
    ) -> RuntimeController:
        return LocalSubprocessController(
            runtime=runtime,
            runtime_root=runtime_root,
            run_in_background=run_in_background,
        )
