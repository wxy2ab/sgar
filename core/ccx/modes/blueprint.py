"""Blueprint mode runner for SGAR governance operations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..agents.subagent import ModeRunner, SubagentInvocation, SubagentResult
from ..sgar import SgarError, SgarRuntime
from ._sgar_command_helpers import (
    _session_id,
    governance_error_extras,
    resolve_sgar_command,
    run_sgar_instruction,
)
from .llm_client import LLMCallable


@dataclass(slots=True)
class BlueprintModeRunner(ModeRunner):
    """Thin deterministic bridge from ccx agent_mode to SGAR.

    This runner intentionally does not ask the LLM to interpret governance
    rules. It maps a small command vocabulary onto the SGAR runtime so hard
    transitions remain enforced by code. Command resolution is ANCHORED
    token matching (see ``resolve_sgar_command``) — substring dispatch
    used to let words like "definition" trigger a destructive ``init``.

    The dispatch body lives in ``_sgar_command_helpers.run_sgar_instruction``
    and is shared with ``BlueprintxModeRunner`` (sgarx), so a dispatch fix
    lands once instead of being ported manually between the two modes.
    """

    cwd: str | None = None
    mode_name: str = "blueprint"
    llm: LLMCallable | None = None
    # P2: opt-in machine-checkable exit criteria (default off). Operator
    # config, threaded from build_runtime — never derived from the LLM
    # instruction. Forwarded to the SgarRuntime this runner constructs.
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
        runtime = SgarRuntime(
            Path(cwd), session_id=str(session_id) if session_id else None,
            run_criterion_checks=self.run_criterion_checks,
            criterion_check_timeout_s=self.criterion_check_timeout_s,
        )
        try:
            text, extras = run_sgar_instruction(
                runtime,
                invocation.goal,
                llm=self.llm,
                supports_reopen_abandon=False,
                metadata=invocation.metadata,
            )
        except SgarError as exc:
            text = f"ERROR: {exc}"
            extras = governance_error_extras(
                exc, invocation.goal, metadata=invocation.metadata,
            )
        return SubagentResult(final_text=text, subtasks=[], extras=extras)


__all__ = ["BlueprintModeRunner"]
