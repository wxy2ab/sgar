"""Ask mode runner — focused, single-shot, read-only Q&A.

Mirrors cc's ``ask`` agent_mode but runs through ccx's ModeRunner
contract so the run goes through v5's NodeSpec dispatcher (consistent
event stream / persistence / cancellation with plan/spec/agent).

Key differences from cc's ask:

* Tool-layer read-only enforcement. cc only tells the LLM "prefer read-
  only tools"; this runner mutates the cc tool registry to physically
  remove ``file_edit`` / ``shell`` / writers via
  ``read_only_runner.restrict_tool_registry``.
* ``has_tools=False`` (lite agent_runner_kind) degrades to a single
  LLM call with no tool access — still useful for general questions
  but cannot inspect the codebase. Documented limitation.

Always returns a terminal ``SubagentResult`` with ``final_text`` set
and no subtasks. Never spawns children.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..agents.cc_agent import (
    _apply_needs_model_marker_result,
    _emit_provider_cost_event,
)
from ..agents.read_only_runner import restrict_tool_registry
from ..agents.subagent import (
    ModeRunner,
    SubagentInvocation,
    SubagentResult,
)
from ..prompts import load_cc_system_prompt
from ..services.cost_events import report_cost_to_budget
from ..services.repository_outline import RepositoryOutlineCache
from .llm_client import text_of
from ._paths import extract_path_tokens
from .llm_client import LLMCallable


logger = logging.getLogger(__name__)


def _should_inject_outline(text: str) -> bool:
    """Mirror cc's ask-mode outline heuristic through its public helper.

    """
    from core.cc.conversation.mode_strategy import decide_mode_strategy

    return bool(decide_mode_strategy("ask", text)["use_repository_outline"])

@dataclass(slots=True)
class AskModeRunner(ModeRunner):
    """Single-shot read-only Q&A runner.

    ``has_tools`` toggles between two execution paths:

    * ``True``  — build a cc QueryEngine, restrict its tool registry to
                  read-only, and run a multi-round LLM↔tool turn.
    * ``False`` — call ``self.llm(system, user)`` once, no tools.

    The two paths share the same system prompt (cc's
    ``system.ask_mode``) and the same outline-injection heuristic.
    """
    llm: LLMCallable
    cwd: str
    cc_config: Any | None = None
    llm_provider: Any | None = None  # LLMClientProvider; required when has_tools=True
    language: str = "en"
    outline_cache: RepositoryOutlineCache | None = None
    has_tools: bool = True
    max_tool_rounds: int | None = None
    mode_name: str = "ask"

    def run(self, invocation: SubagentInvocation) -> SubagentResult:
        if self.has_tools:
            context = None
            token = None
            if self.llm_provider is not None and hasattr(self.llm_provider, "begin_invocation"):
                context, token = self.llm_provider.begin_invocation(
                    mode=self.mode_name,
                    metadata=invocation.metadata,
                )
            try:
                result = self._run_with_tools(invocation)
            finally:
                if context is not None and context.cost_accumulator:
                    cost_usd = sum(context.cost_accumulator)
                    _emit_provider_cost_event(
                        mode=self.mode_name,
                        cost_usd=cost_usd,
                        call_count=len(context.cost_accumulator),
                        tokens=sum(context.token_accumulator),
                    )
                    report_cost_to_budget(
                        cost_usd=cost_usd,
                        tokens=sum(context.token_accumulator),
                    )
                if token is not None and hasattr(self.llm_provider, "end_invocation"):
                    self.llm_provider.end_invocation(token)
            if context is not None and context.needs_accumulator:
                result = _apply_needs_model_marker_result(
                    result, context.needs_accumulator[-1],
                )
            return result
        return self._run_no_tools(invocation)

    # ------------------------------------------------------------------ #
    # Tool path (cc_query_loop)
    # ------------------------------------------------------------------ #

    def _run_with_tools(self, invocation: SubagentInvocation) -> SubagentResult:
        from ..agents.cc_agent import _run_in_fresh_loop
        return _run_in_fresh_loop(self._run_with_tools_async(invocation))

    async def _run_with_tools_async(
        self, invocation: SubagentInvocation,
    ) -> SubagentResult:
        if self.llm_provider is None or self.cc_config is None:
            raise RuntimeError(
                "AskModeRunner with has_tools=True requires "
                "`cc_config` and `llm_provider`."
            )
        from core.cc.runtime import build_default_query_engine

        engine = build_default_query_engine(
            cwd=self.cwd,
            config=self.cc_config,
            llm_client_provider=self.llm_provider,
        )
        removed, kept = restrict_tool_registry(engine)
        logger.debug(
            "ask runner: filtered cc registry, removed %d non-read-only "
            "tools (kept %s)",
            removed, kept,
        )

        system_prompt = load_cc_system_prompt("ask_mode", self.language)
        user_prompt = self._build_user_prompt(invocation)
        framed = f"<system>\n{system_prompt}\n</system>\n\n{user_prompt}"

        final_text = ""
        tool_call_count = 0
        event_count = 0
        try:
            async for event in engine.submit_message(
                framed, max_tool_rounds=self.max_tool_rounds,
            ):
                event_count += 1
                if event.event_type == "assistant_tool_use":
                    tool_call_count += 1
                msg = getattr(event, "message", None)
                if msg is None:
                    continue
                if (
                    getattr(msg, "role", "") == "assistant"
                    and getattr(msg, "kind", "") == "assistant_text"
                ):
                    final_text = str(getattr(msg, "content", ""))
        finally:
            engine.close()

        return SubagentResult(
            final_text=final_text,
            subtasks=[],
            sequential=False,
            extras={
                "tool_call_count": tool_call_count,
                "event_count": event_count,
                "via": "ccx_ask_with_tools",
                "goal": invocation.goal,
                "outline_injected": self._wants_outline(invocation.goal),
            },
        )

    # ------------------------------------------------------------------ #
    # No-tools path (lite)
    # ------------------------------------------------------------------ #

    def _run_no_tools(self, invocation: SubagentInvocation) -> SubagentResult:
        system_prompt = load_cc_system_prompt("ask_mode", self.language)
        user_prompt = self._build_user_prompt(invocation)
        response = text_of(
            self.llm(system=system_prompt, user=user_prompt, purpose="ask")
        )
        return SubagentResult(
            final_text=response.strip(),
            subtasks=[],
            sequential=False,
            extras={
                "via": "ccx_ask_lite",
                "goal": invocation.goal,
                "outline_injected": self._wants_outline(invocation.goal),
            },
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _wants_outline(self, goal: str) -> bool:
        return _should_inject_outline(goal)

    def _build_user_prompt(self, invocation: SubagentInvocation) -> str:
        parts: list[str] = [f"## Question\n{invocation.goal}"]
        path_block = self._paths_context_block(invocation.goal)
        if path_block:
            parts.append(path_block)
        if self._wants_outline(invocation.goal) and self.outline_cache is not None:
            try:
                outline_text = self.outline_cache.get_text(deep=False)
            except Exception as exc:  # noqa: BLE001
                logger.debug("ask runner: outline build failed: %s", exc)
                outline_text = ""
            if outline_text:
                parts.append(
                    "## Repository Outline (PARTIAL — truncated; trust paths above even if missing here)\n"
                    "Use this as a structural starting point; verify with tools "
                    "before drawing conclusions.\n\n"
                    f"```\n{outline_text}\n```"
                )
        parts.append(
            "Answer directly. If you cite files, give relative paths and "
            "line ranges where possible."
        )
        return "\n\n".join(parts)

    def _paths_context_block(self, *texts: str) -> str:
        """Same idea as DocModeRunner._paths_context_block but lighter:
        no focused subtree expansion (ask is single-shot and the LLM
        can `glob` itself), only existence verification."""
        merged = " ".join(t for t in texts if t)
        tokens = extract_path_tokens(merged)
        if not tokens:
            return ""
        cwd_path = Path(self.cwd) if self.cwd else None
        lines = ["## Paths in this task"]
        for tok in tokens:
            status = "[?]"
            if cwd_path is not None:
                cand = (cwd_path / tok) if not Path(tok).is_absolute() else Path(tok)
                try:
                    resolved = cand.resolve()
                except OSError:
                    resolved = cand
                status = "[verified]" if resolved.exists() else "[missing in cwd — verify with `glob`]"
            lines.append(f"- {status} `{tok}`")
        return "\n".join(lines)


__all__ = ["AskModeRunner", "_should_inject_outline"]
