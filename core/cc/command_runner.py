from __future__ import annotations

import asyncio
import base64
from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
import platform
import shlex
import shutil
import signal
import subprocess
import time
from typing import Any


@dataclass(slots=True)
class CommandExecutionResult:
    success: bool
    command: str
    shell_kind: str
    cwd: str
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_ms: int = 0
    was_timeout: bool = False
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def default_shell_kind() -> str:
    return "powershell" if _is_windows() else "shell"


def _is_windows() -> bool:
    return platform.system().lower().startswith("win")


def resolve_powershell_executable(env: dict[str, str] | None = None) -> str | None:
    """Resolve PowerShell 7 first, then Windows PowerShell 5.1."""
    search_path = env.get("PATH") if env is not None else None
    for candidate in ("pwsh", "powershell.exe", "powershell"):
        executable = shutil.which(candidate, path=search_path)
        if executable:
            return executable
    return None


@dataclass(slots=True)
class _CommandInvocation:
    argv: list[str]
    metadata: dict[str, object] = field(default_factory=dict)


def _powershell_script(command: str) -> str:
    # -EncodedCommand removes the host shell/CRT quoting layer. The user
    # command is carried as opaque UTF-8 base64 so an unbalanced `}` (or a
    # `<#` comment) cannot close/comment-out the wrapper and skip the exit
    # trailer. ScriptBlock::Create parses the payload in isolation.
    # Preamble: deterministic UTF-8 redirected output on pwsh and Windows
    # PowerShell 5.1. Trailer: propagate native-command and cmdlet failures.
    payload = base64.b64encode(command.encode("utf-8")).decode("ascii")
    return (
        "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); "
        "$OutputEncoding = [Console]::OutputEncoding; "
        f"$__cc_cmd = [System.Text.Encoding]::UTF8.GetString("
        f"[System.Convert]::FromBase64String('{payload}')); "
        "& ([ScriptBlock]::Create($__cc_cmd)); "
        "$__cc_success = $?; $__cc_native_exit = $LASTEXITCODE; "
        "if ($null -ne $__cc_native_exit) { exit $__cc_native_exit }; "
        "if (-not $__cc_success) { exit 1 }"
    )


def _normalize_shell_argv(argv: list[str]) -> list[str]:
    if not _is_windows():
        return argv
    normalized: list[str] = []
    for token in argv:
        if len(token) >= 2 and token[0] == token[-1] and token[0] in "'\"":
            normalized.append(token[1:-1])
        else:
            normalized.append(token)
    return normalized


def format_shell_command(argv: list[str]) -> str:
    """Serialize argv for later parsing by ``kind='shell'`` on the current OS."""
    if _is_windows():
        return subprocess.list2cmdline(argv)
    return shlex.join(argv)


def _build_invocation(
    command: str,
    shell_kind: str,
    *,
    env: dict[str, str] | None = None,
) -> _CommandInvocation:
    if not command.strip():
        raise ValueError("Empty command.")
    if shell_kind == "powershell":
        executable = resolve_powershell_executable(env)
        if executable is None:
            raise FileNotFoundError(
                "PowerShell is not installed or not on PATH "
                "(tried pwsh, powershell.exe, powershell)."
            )
        encoded = base64.b64encode(_powershell_script(command).encode("utf-16-le")).decode("ascii")
        family = "pwsh" if Path(executable).name.lower().startswith("pwsh") else "windows_powershell"
        return _CommandInvocation(
            argv=[
                executable,
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-EncodedCommand",
                encoded,
            ],
            metadata={"executable": executable, "powershell_family": family},
        )
    if shell_kind == "shell":
        try:
            argv = shlex.split(command, posix=not _is_windows())
        except ValueError as exc:
            raise ValueError(f"Invalid command syntax: {exc}") from exc
        if not argv:
            raise ValueError("Empty command.")
        argv = _normalize_shell_argv(argv)
        return _CommandInvocation(argv=argv, metadata={"executable": argv[0]})
    raise ValueError(f"Unsupported shell_kind: {shell_kind}")


def _decode_output(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return value.decode("utf-8", errors="replace")


def _spawn_group_kwargs() -> dict[str, Any]:
    if _is_windows():
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    return {"start_new_session": True}


def _failure_result(
    *,
    command: str,
    shell_kind: str,
    cwd: str,
    error: Exception,
    metadata: dict[str, object] | None = None,
) -> CommandExecutionResult:
    return CommandExecutionResult(
        success=False,
        command=command,
        shell_kind=shell_kind,
        cwd=cwd,
        exit_code=-1 if isinstance(error, ValueError) else 127,
        stderr=str(error),
        metadata=dict(metadata or {}),
    )


def _terminate_process_sync(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    if _is_windows():
        try:
            completed = subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                check=False,
                timeout=5,
            )
            if completed.returncode == 0:
                return
        except (OSError, subprocess.SubprocessError):
            pass
    elif hasattr(os, "killpg"):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass
    try:
        proc.kill()
    except (ProcessLookupError, OSError):
        pass


def execute_command(
    *,
    command: str,
    cwd: str | Path,
    shell_kind: str | None = None,
    timeout_ms: int | None = None,
    env: dict[str, str] | None = None,
) -> CommandExecutionResult:
    resolved_shell = shell_kind or default_shell_kind()
    resolved_cwd = str(Path(cwd).resolve())
    timeout_s = None if timeout_ms is None else timeout_ms / 1000

    try:
        invocation = _build_invocation(command, resolved_shell, env=env)
    except (OSError, ValueError) as exc:
        return _failure_result(
            command=command,
            shell_kind=resolved_shell,
            cwd=resolved_cwd,
            error=exc,
        )
    started_at = time.perf_counter()
    try:
        proc = subprocess.Popen(
            invocation.argv,
            cwd=resolved_cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **_spawn_group_kwargs(),
        )
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_sync(proc)
        try:
            stdout, stderr = proc.communicate(timeout=1)
        except subprocess.TimeoutExpired:
            stdout, stderr = exc.output, exc.stderr
        return CommandExecutionResult(
            success=False,
            command=command,
            shell_kind=resolved_shell,
            cwd=resolved_cwd,
            exit_code=-1,
            stdout=_decode_output(stdout),
            stderr=_decode_output(stderr),
            duration_ms=int(timeout_ms or 0),
            was_timeout=True,
            metadata=invocation.metadata,
        )
    except OSError as exc:
        return _failure_result(
            command=command,
            shell_kind=resolved_shell,
            cwd=resolved_cwd,
            error=exc,
            metadata=invocation.metadata,
        )

    duration_ms = int((time.perf_counter() - started_at) * 1000)
    return CommandExecutionResult(
        success=proc.returncode == 0,
        command=command,
        shell_kind=resolved_shell,
        cwd=resolved_cwd,
        exit_code=proc.returncode or 0,
        stdout=_decode_output(stdout),
        stderr=_decode_output(stderr),
        duration_ms=duration_ms,
        metadata=invocation.metadata,
    )


async def _terminate_process(proc: "asyncio.subprocess.Process") -> None:
    """Terminate the complete subprocess tree on the current platform."""
    if proc.returncode is not None:
        return
    if _is_windows() and proc.pid is not None:
        try:
            killer = await asyncio.create_subprocess_exec(
                "taskkill",
                "/PID",
                str(proc.pid),
                "/T",
                "/F",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(killer.wait(), timeout=5)
            if killer.returncode == 0:
                return
        except (OSError, asyncio.TimeoutError):
            pass
    elif proc.pid is not None and hasattr(os, "killpg"):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass
    try:
        proc.kill()
    except (ProcessLookupError, OSError):
        pass


async def _pump_stream(stream: "asyncio.StreamReader | None", chunks: list[bytes]) -> None:
    """Continuously read a stream into ``chunks`` until EOF.

    Runs concurrently with the process so output is accumulated as it arrives —
    on timeout the already-read chunks are the salvaged partial output, and the
    pipes never fill (which would deadlock a producer). ``communicate()`` cannot
    be used for this: it buffers into a local that is discarded when the
    surrounding ``wait_for`` cancels it, losing the partial output.
    """
    if stream is None:
        return
    while True:
        chunk = await stream.read(65536)
        if not chunk:
            break
        chunks.append(chunk)


async def execute_command_async(
    *,
    command: str,
    cwd: str | Path,
    shell_kind: str | None = None,
    timeout_ms: int | None = None,
    env: dict[str, str] | None = None,
) -> CommandExecutionResult:
    resolved_shell = shell_kind or default_shell_kind()
    resolved_cwd = str(Path(cwd).resolve())
    timeout_s = None if timeout_ms is None else timeout_ms / 1000

    try:
        invocation = _build_invocation(command, resolved_shell, env=env)
    except (OSError, ValueError) as exc:
        return _failure_result(
            command=command,
            shell_kind=resolved_shell,
            cwd=resolved_cwd,
            error=exc,
        )
    started_at = time.perf_counter()
    try:
        proc = await asyncio.create_subprocess_exec(
            *invocation.argv,
            cwd=resolved_cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **_spawn_group_kwargs(),
        )
    except OSError as exc:
        result = _failure_result(
            command=command,
            shell_kind=resolved_shell,
            cwd=resolved_cwd,
            error=exc,
            metadata=invocation.metadata,
        )
        result.duration_ms = int((time.perf_counter() - started_at) * 1000)
        return result
    # Pump stdout/stderr concurrently so partial output is preserved on timeout
    # and the pipes can't fill and deadlock the child.
    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []
    pump_stdout = asyncio.ensure_future(_pump_stream(proc.stdout, stdout_chunks))
    pump_stderr = asyncio.ensure_future(_pump_stream(proc.stderr, stderr_chunks))
    try:
        await asyncio.wait_for(proc.wait(), timeout=timeout_s)
    except asyncio.TimeoutError:
        # Kill the whole process group (not just the direct child); the pumps
        # then drain any remaining buffered output before the pipes close, so
        # the [check:]/run_tests evidence tail survives (matching the sync path).
        await _terminate_process(proc)
        try:
            await asyncio.wait_for(asyncio.gather(pump_stdout, pump_stderr), timeout=1.0)
        except (asyncio.TimeoutError, Exception):
            pump_stdout.cancel()
            pump_stderr.cancel()
        return CommandExecutionResult(
            success=False,
            command=command,
            shell_kind=resolved_shell,
            cwd=resolved_cwd,
            exit_code=-1,
            stdout=b"".join(stdout_chunks).decode("utf-8", errors="replace"),
            stderr=b"".join(stderr_chunks).decode("utf-8", errors="replace"),
            duration_ms=int(timeout_ms or 0),
            was_timeout=True,
            metadata=invocation.metadata,
        )
    except asyncio.CancelledError:
        await _terminate_process(proc)
        await asyncio.gather(pump_stdout, pump_stderr, return_exceptions=True)
        raise

    # Process exited within the budget — let the pumps finish reading to EOF.
    await asyncio.gather(pump_stdout, pump_stderr, return_exceptions=True)
    duration_ms = int((time.perf_counter() - started_at) * 1000)
    return CommandExecutionResult(
        success=proc.returncode == 0,
        command=command,
        shell_kind=resolved_shell,
        cwd=resolved_cwd,
        exit_code=proc.returncode or 0,
        stdout=b"".join(stdout_chunks).decode("utf-8", errors="replace"),
        stderr=b"".join(stderr_chunks).decode("utf-8", errors="replace"),
        duration_ms=duration_ms,
        metadata=invocation.metadata,
    )


# ---------------------------------------------------------------------------
# Check verdict — exit-code-gated verification for the interactive cc loop.
#
# Mirrors the *semantics* of ``core.ccx.sgar.checks`` (exit 0 = pass, bounded
# evidence tail, an unrunnable-vs-genuine-failure distinction) but lives in the
# ``cc`` layer so the interactive ``run_tests`` tool and the post-edit
# auto-verify step can reuse it WITHOUT importing ``core.ccx``. The dependency
# direction in this repo is ``ccx -> cc`` and never the reverse, so the check
# primitive a cc tool needs has to live here.
#
# Deliberate divergence from ccx's *hermetic* checks: this runs the command in
# the operator's normal environment, NOT a sanitized
# ``PYTHONSAFEPATH``/``PYTHONNOUSERSITE`` one. ccx's hermetic env is a
# governance trust-root defense against an agent that actively poisons its
# interpreter to fake a green check; the interactive cc path is cooperative (a
# human drives and reviews it), and a hermetic env would surprise users by
# breaking ordinary ``python -c "import my_module"`` / cwd-relative imports.
# Adversarial, gate-bearing verification belongs to the governed ccx paths.
# ---------------------------------------------------------------------------

_CHECK_EVIDENCE_TAIL_LINES = 20
_CHECK_EVIDENCE_MAX_CHARS = 2000


@dataclass(slots=True)
class CheckVerdict:
    """Exit-code-gated verdict of one verification command run."""

    passed: bool
    command: str
    exit_code: int | None  # None on timeout / spawn failure
    output_tail: str = ""
    timed_out: bool = False
    unrunnable: bool = False  # could not execute at all (no verification signal)
    error: str | None = None  # spawn-time error (bad command, not found, …)
    shell_kind: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def evidence_line(self) -> str:
        """One-line (+ optional output tail) machine-evidence summary."""
        if self.timed_out:
            status = "TIMEOUT"
        elif self.error is not None:
            status = f"ERROR: {self.error}"
        else:
            status = f"exit={self.exit_code}"
        verdict = "PASS" if self.passed else "FAIL"
        head = f"check `{self.command}` -> {verdict} ({status})"
        if self.output_tail:
            return f"{head}\n{self.output_tail}"
        return head


def _check_tail(text: str) -> str:
    if not text:
        return ""
    lines = text.strip().splitlines()
    tail = "\n".join(lines[-_CHECK_EVIDENCE_TAIL_LINES:])
    if len(tail) > _CHECK_EVIDENCE_MAX_CHARS:
        tail = tail[-_CHECK_EVIDENCE_MAX_CHARS:]
    return tail


def _result_unrunnable(result: CommandExecutionResult) -> bool:
    """Did the command FAIL TO EXECUTE (vs run and report non-zero)?

    Mirrors ``core.ccx.sgar.checks.check_unrunnable`` so a harness/config defect
    (missing binary, bad command syntax, timeout) is not mistaken for a genuine
    red result. Conservative: a command that ran and exited non-zero for a real
    reason (e.g. a failing test suite at rc=1) is NOT flagged.
    """
    if result.was_timeout:
        return True
    rc = result.exit_code
    # ``command_runner`` marks parse-failure / empty-command with exit_code -1.
    if rc is None or rc == -1:
        return True
    if rc == 127:  # command not found (e.g. via ``sh -c``)
        return True
    if rc == 2 and "syntax error" in (result.stderr or "").lower():
        return True
    return False


async def run_check_command_async(
    *,
    command: str,
    cwd: str | Path,
    shell_kind: str | None = None,
    timeout_ms: int | None = None,
) -> CheckVerdict:
    """Run ``command`` and return an exit-code-gated :class:`CheckVerdict`.

    Exit 0 (and no timeout) => ``passed``. Never raises for an ordinary
    failed / timed-out / unspawnable command — those become ``passed=False``
    verdicts (``unrunnable=True`` when the command could not execute at all) so
    the caller can attach evidence and decide policy.
    """
    resolved_shell = shell_kind or default_shell_kind()
    try:
        result = await execute_command_async(
            command=command,
            cwd=cwd,
            shell_kind=resolved_shell,
            timeout_ms=timeout_ms,
        )
    except (OSError, ValueError) as exc:  # e.g. binary not found on spawn
        return CheckVerdict(
            passed=False,
            command=command,
            exit_code=None,
            unrunnable=True,
            error=str(exc),
            shell_kind=resolved_shell,
        )
    tail = _check_tail(f"{result.stdout}\n{result.stderr}")
    return CheckVerdict(
        passed=result.success and not result.was_timeout,
        command=command,
        exit_code=None if result.was_timeout else result.exit_code,
        output_tail=tail,
        timed_out=result.was_timeout,
        unrunnable=_result_unrunnable(result),
        error=None,
        shell_kind=result.shell_kind,
    )
