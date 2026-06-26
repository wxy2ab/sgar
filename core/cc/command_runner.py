from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from pathlib import Path
import platform
import shlex
import subprocess
import time
from typing import Sequence


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
    return "powershell" if platform.system().lower().startswith("win") else "shell"


def execute_command(
    *,
    command: str,
    cwd: str | Path,
    shell_kind: str | None = None,
    timeout_ms: int | None = None,
) -> CommandExecutionResult:
    resolved_shell = shell_kind or default_shell_kind()
    resolved_cwd = str(Path(cwd).resolve())
    timeout_s = None if timeout_ms is None else timeout_ms / 1000

    if resolved_shell == "powershell":
        cmd: Sequence[str] = [
            "powershell",
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            command,
        ]
        shell = False
    elif resolved_shell == "shell":
        try:
            cmd = shlex.split(command)
        except ValueError as exc:
            return CommandExecutionResult(
                success=False,
                command=command,
                shell_kind=resolved_shell,
                cwd=resolved_cwd,
                exit_code=-1,
                stderr=f"Invalid command syntax: {exc}",
            )
        if not cmd:
            return CommandExecutionResult(
                success=False,
                command=command,
                shell_kind=resolved_shell,
                cwd=resolved_cwd,
                exit_code=-1,
                stderr="Empty command.",
            )
        shell = False
    else:
        raise ValueError(f"Unsupported shell_kind: {resolved_shell}")

    try:
        started_at = time.perf_counter()
        completed = subprocess.run(
            cmd,
            cwd=resolved_cwd,
            shell=shell,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandExecutionResult(
            success=False,
            command=command,
            shell_kind=resolved_shell,
            cwd=resolved_cwd,
            exit_code=-1,
            stdout=(exc.stdout or ""),
            stderr=(exc.stderr or ""),
            duration_ms=int(timeout_ms or 0),
            was_timeout=True,
        )

    duration_ms = int((time.perf_counter() - started_at) * 1000)
    return CommandExecutionResult(
        success=completed.returncode == 0,
        command=command,
        shell_kind=resolved_shell,
        cwd=resolved_cwd,
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        duration_ms=duration_ms,
    )


async def execute_command_async(
    *,
    command: str,
    cwd: str | Path,
    shell_kind: str | None = None,
    timeout_ms: int | None = None,
) -> CommandExecutionResult:
    resolved_shell = shell_kind or default_shell_kind()
    resolved_cwd = str(Path(cwd).resolve())
    timeout_s = None if timeout_ms is None else timeout_ms / 1000

    if resolved_shell == "powershell":
        args = [
            "powershell",
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            command,
        ]
        create = asyncio.create_subprocess_exec(
            *args,
            cwd=resolved_cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    elif resolved_shell == "shell":
        try:
            args = shlex.split(command)
        except ValueError as exc:
            return CommandExecutionResult(
                success=False,
                command=command,
                shell_kind=resolved_shell,
                cwd=resolved_cwd,
                exit_code=-1,
                stderr=f"Invalid command syntax: {exc}",
            )
        if not args:
            return CommandExecutionResult(
                success=False,
                command=command,
                shell_kind=resolved_shell,
                cwd=resolved_cwd,
                exit_code=-1,
                stderr="Empty command.",
            )
        create = asyncio.create_subprocess_exec(
            *args,
            cwd=resolved_cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    else:
        raise ValueError(f"Unsupported shell_kind: {resolved_shell}")

    started_at = time.perf_counter()
    proc = await create
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return CommandExecutionResult(
            success=False,
            command=command,
            shell_kind=resolved_shell,
            cwd=resolved_cwd,
            exit_code=-1,
            stdout="",
            stderr="",
            duration_ms=int(timeout_ms or 0),
            was_timeout=True,
        )

    duration_ms = int((time.perf_counter() - started_at) * 1000)
    return CommandExecutionResult(
        success=proc.returncode == 0,
        command=command,
        shell_kind=resolved_shell,
        cwd=resolved_cwd,
        exit_code=proc.returncode or 0,
        stdout=stdout_bytes.decode("utf-8", errors="replace"),
        stderr=stderr_bytes.decode("utf-8", errors="replace"),
        duration_ms=duration_ms,
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
    try:
        result = await execute_command_async(
            command=command,
            cwd=cwd,
            shell_kind=shell_kind,
            timeout_ms=timeout_ms,
        )
    except (OSError, ValueError) as exc:  # e.g. binary not found on spawn
        return CheckVerdict(
            passed=False,
            command=command,
            exit_code=None,
            unrunnable=True,
            error=str(exc),
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
    )
