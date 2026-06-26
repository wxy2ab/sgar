"""Sgarx mode runner — parallel of :mod:`core.ccx.modes.blueprint`.

The runner behind ``agent_mode='sgarx'``. Identical command surface to
``BlueprintModeRunner`` plus the sgarx-only stage recovery commands
(``reopen-stage`` / ``abandon-stage``); the runtime backing each command
is :class:`SgarxRuntime` (writes to ``.sgarx/``) instead of
:class:`SgarRuntime` (writes to ``.sgar/``).

The dispatch body is NOT duplicated from blueprint.py anymore: both
runners call ``_sgar_command_helpers.run_sgar_instruction``, so a
dispatch-level fix lands once. This runner only differs in the runtime
class it constructs and ``supports_reopen_abandon=True``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..agents.subagent import ModeRunner, SubagentInvocation, SubagentResult
from ..sgar import SgarError
from ..sgarx import SgarxRuntime
from ._sgar_command_helpers import (
    _session_id,
    governance_error_extras,
    resolve_sgar_command,
    run_sgar_instruction,
)
from .llm_client import LLMCallable


@dataclass(slots=True)
class BlueprintxModeRunner(ModeRunner):
    """Thin deterministic bridge from ``agent_mode='sgarx'`` to SgarxRuntime."""

    cwd: str | None = None
    mode_name: str = "sgarx"
    llm: LLMCallable | None = None
    # P2: opt-in machine-checkable exit criteria (default off). See
    # BlueprintModeRunner — operator config, forwarded to SgarxRuntime.
    run_criterion_checks: bool = False
    criterion_check_timeout_s: float = 120.0

    def run(self, invocation: SubagentInvocation) -> SubagentResult:
        request_metadata = invocation.metadata.get("request_metadata")
        if not isinstance(request_metadata, dict):
            request_metadata = {}
        cwd = str(invocation.metadata.get("cwd") or self.cwd or ".")
        # Resolve --session from the command text only — resume/steer
        # context prepended to the goal must not hijack session routing.
        _, command_text = resolve_sgar_command(
            invocation.goal,
            metadata=invocation.metadata,
        )
        session_id = (
            _session_id(command_text)
            or invocation.metadata.get("sgar_session")
            or invocation.metadata.get("session_id")
            or request_metadata.get("sgar_session")
            or request_metadata.get("session_id")
        )
        runtime = SgarxRuntime(
            Path(cwd), session_id=str(session_id) if session_id else None,
            run_criterion_checks=self.run_criterion_checks,
            criterion_check_timeout_s=self.criterion_check_timeout_s,
        )
        try:
            text, extras = run_sgar_instruction(
                runtime,
                invocation.goal,
                llm=self.llm,
                supports_reopen_abandon=True,
                metadata=invocation.metadata,
            )
        except SgarError as exc:
            text = f"ERROR: {exc}"
            extras = governance_error_extras(
                exc, invocation.goal, metadata=invocation.metadata,
            )
        return SubagentResult(final_text=text, subtasks=[], extras=extras)


__all__ = ["BlueprintxModeRunner"]
