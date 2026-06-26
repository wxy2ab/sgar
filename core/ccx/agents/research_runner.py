"""ResearchRunner — drives a read-only investigative cc turn.

Sister of ``CcAgentRunner`` but with a hard tool whitelist and an
investigation-shaped system prompt. The v5 runtime wires
``mode_name="research"`` to this runner so that ``ccx_research``-buffered
requests, after being drained into ``SubagentInvocation(mode="research")``,
land here.

Design choices for v1:

* Whitelist enforcement is done by mutating the cc ``ToolRegistry``
  in-place after building the QueryEngine — only tools with
  ``is_read_only=True`` survive. Concrete result: ``Read``, ``Grep``,
  ``Glob``, ``memory_search``, ``memory_status``. Edits, shell, file
  writes are gone for this turn even if cc's default registry exposes
  them.
* The system prompt is engineered for the **two-phase search pattern**:
  first narrow with ``Grep --files-with-matches`` / ``Glob`` to identify
  candidate files (cheap), then read targeted line ranges. This is the
  key to handling large codebases without context blow-up.
* Output schema: SubagentResult.final_text is a structured findings
  paragraph; ``extras["evidence"]`` holds a list of ``{path, lines,
  excerpt}`` references. Downstream nodes can consume either.

Recursive research (a research subagent spawning more research) is
intentionally NOT supported in v1: the buffer / spawn tooling is not
registered into the research engine. If a question is too big, the LLM
should narrow the question and the parent agent can fan out. v2 may
relax this if real workloads demand it.
"""

from __future__ import annotations

import contextvars
import json
import logging
from dataclasses import dataclass
from typing import Any

from core.cc.config import CCConfig
from core.cc.runtime import build_default_query_engine

from .cc_agent import (
    _LLMProviderInvocationContext,
    _apply_needs_model_marker_result,
    _emit_provider_cost_event,
    _is_turn_timeout_message,
    _run_in_fresh_loop,
)
from .read_only_runner import (
    DEFAULT_READ_ONLY_WHITELIST as _RESEARCH_TOOL_WHITELIST,
    restrict_tool_registry,
)
from .subagent import (
    ModeRunner,
    SubagentInvocation,
    SubagentResult,
)
from ..services.cost_events import report_cost_to_budget


logger = logging.getLogger(__name__)


# ``_RESEARCH_TOOL_WHITELIST`` is re-exported above for back-compat;
# callers should prefer ``DEFAULT_READ_ONLY_WHITELIST`` from
# ``read_only_runner`` going forward.


_RESEARCH_SYSTEM_PROMPT = """\
You are a READ-ONLY research subagent. Your job is to answer ONE specific \
question about the codebase by reading code — never by editing it.

You have ONLY these tools (case-sensitive, call them exactly as listed):
  * `file_read` — args: ``file_path`` (required), ``max_bytes`` \
(optional cap, default 100_000). Reads UTF-8 text. There is no \
offset/limit; use ``max_bytes`` if you only need a file head.
  * `glob` — args: ``pattern`` (required), ``cwd`` (search root, \
default workspace root), ``max_results``. Returns matching FILE \
paths. There is NO separate ``list_files`` tool; `glob` with \
``cwd=<dir>`` and ``pattern="**/*"`` enumerates that directory.
  * `grep` — args: ``pattern`` (required), ``cwd`` (search root), \
``glob`` (file glob filter), ``files_only`` (bool — list only \
filenames), ``file_type`` (e.g. ``"py"``), ``context_lines``, \
``max_results``.
  * `memory_search` / `memory_status` — when memory is enabled.

You cannot write files, run shell commands, or modify state.

CRITICAL — scope every search to the question's path:
  * If the question or scope mentions a specific directory or file, \
EVERY `grep` / `glob` / `file_read` call MUST use it as the search \
root via ``cwd=<that-path>`` (or ``file_path=<that-path>/...``). \
Calls without scope hit the entire repository, return irrelevant \
matches, blow up your context, and waste tool rounds — DO NOT do \
that.
  * If the path doesn't appear in the outline, that just means the \
outline is a truncated sample — the path still exists. Use \
``glob(pattern="**/*", cwd="<path>")`` to verify and enumerate it; \
never conclude "directory not found" for a path the caller named.

Use the TWO-PHASE SEARCH PATTERN. The codebase may be very large; you \
must not dump entire directories.

Phase 1 — narrow:
  * Start with `grep` ``files_only=true`` (always with ``cwd`` set to \
the question's scope) to find candidate files. Or use `glob` to \
enumerate by filename pattern.
  * If the first pass returns too many matches, refine the pattern; do \
not proceed to phase 2 until the candidate set is small (≤ ~10 files).

Phase 2 — read targeted:
  * For each candidate, use `grep` (``context_lines>=2``) to locate the \
relevant lines with surrounding context.
  * Use `file_read` (with ``max_bytes`` if the file might be large) on \
the most relevant files. Don't conclude based only on filenames or \
zero-context grep hits.

Output format — when you have your answer, emit a final text response \
with THIS exact JSON shape and no surrounding prose:

{
  "summary": "<1-3 sentence direct answer>",
  "evidence": [
    {"path": "<repo-relative path>", "lines": "<start-end>", "excerpt": "<≤300 chars>"},
    ...
  ],
  "confidence": "high" | "medium" | "low",
  "limits": "<what you couldn't determine, or empty string>"
}

Rules:
* `summary` answers the question directly. No "I looked at..." preamble.
* `evidence` cites specific lines that support the summary (≤ 6 entries).
* If the question is unanswerable from the code, emit summary="<reason>", \
confidence="low", and explain in `limits`.
* Do NOT speculate beyond what the code shows.
"""


def _restrict_tool_registry(engine: Any) -> int:
    """Back-compat shim around ``restrict_tool_registry``.

    Existing callers / tests use this thin wrapper which returns only
    the removed count; new code should call ``restrict_tool_registry``
    directly to also receive the kept-names list.
    """
    removed, _kept = restrict_tool_registry(engine)
    return removed


def _build_user_prompt(invocation: SubagentInvocation) -> str:
    """Assemble the user-side prompt for a research turn."""
    md = invocation.metadata or {}
    scope = md.get("scope") or md.get("scope_dir") or ""
    focus = md.get("focus_paths") or []
    parts: list[str] = []
    parts.append(f"## Research question\n{invocation.goal}")
    if scope:
        parts.append(f"## Scope\nLook within: `{scope}` (do not read outside).")
    if focus:
        focus_lines = "\n".join(f"  - `{p}`" for p in focus)
        parts.append(f"## Suggested starting points\n{focus_lines}")
    parts.append(
        "Begin with phase 1 (narrow). Emit the JSON-shaped final answer "
        "when you have enough evidence."
    )
    return "\n\n".join(parts)


def _parse_findings(final_text: str) -> tuple[str, list[dict[str, Any]], str]:
    """Parse a researcher's final text into (summary, evidence, confidence).

    Robust to LLMs that wrap the JSON in fenced markdown or include a
    short prose preamble. Falls back to {summary=<raw text>, evidence=[],
    confidence="low"} on any parse failure so the parent always gets
    SOMETHING usable.
    """
    text = final_text.strip()
    # Use the same robust extractor that doc-mode uses for investigator
    # output (``_robust_json_object``). Local import avoids creating a
    # circular dep at module load — doc.py and research_runner are in
    # parallel package trees.
    from ..modes.doc import _robust_json_object

    parsed = _robust_json_object(text)
    if parsed is None:
        return final_text.strip(), [], "low"
    summary = str(parsed.get("summary", "")).strip()
    evidence_raw = parsed.get("evidence") or []
    evidence: list[dict[str, Any]] = []
    if isinstance(evidence_raw, list):
        for item in evidence_raw:
            if not isinstance(item, dict):
                continue
            evidence.append({
                "path": str(item.get("path", "")),
                "lines": str(item.get("lines", "")),
                "excerpt": str(item.get("excerpt", ""))[:300],
            })
    confidence = str(parsed.get("confidence", "")).lower() or "medium"
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"
    if not summary:
        summary = final_text.strip()
    return summary, evidence, confidence


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class ResearchRunner(ModeRunner):
    """Run a single read-only research turn through cc's QueryEngine."""
    cc_config: CCConfig
    llm_provider: Any  # LLMClientProvider
    cwd: str
    max_tool_rounds: int | None = None
    mode_name: str = "research"

    def run(self, invocation: SubagentInvocation) -> SubagentResult:
        context: _LLMProviderInvocationContext | None = None
        token: contextvars.Token[_LLMProviderInvocationContext | None] | None = None
        if hasattr(self.llm_provider, "begin_invocation"):
            context, token = self.llm_provider.begin_invocation(
                mode=self.mode_name,
                metadata=invocation.metadata,
            )
        try:
            result = _run_in_fresh_loop(self._run_async(invocation))
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

    async def _run_async(
        self, invocation: SubagentInvocation,
    ) -> SubagentResult:
        engine = build_default_query_engine(
            cwd=self.cwd,
            config=self.cc_config,
            llm_client_provider=self.llm_provider,
        )
        removed = _restrict_tool_registry(engine)
        logger.debug(
            "research runner: filtered cc tool registry, removed %d "
            "non-read-only tools (kept whitelist: %s)",
            removed, sorted(_RESEARCH_TOOL_WHITELIST),
        )

        # Bridge cc SessionEvents into the v5 events table so a
        # research subagent's tool activity (grep / file_read / glob)
        # surfaces via ``watch --tail`` instead of staying invisible
        # between the v5 node.created and node.completed pair.
        from .event_bridge import make_event_sink
        bridge_sink = make_event_sink()

        user_prompt = _build_user_prompt(invocation)
        final_text = ""
        turn_timed_out = False
        tool_call_count = 0
        event_count = 0

        # Inject the research system prompt as a leading user-system
        # message via cc's existing system_prompt_context channel. cc's
        # build_default_query_engine accepts a system prompt addition
        # through the engine's chat config, but the cleanest universal
        # path is to prepend the system prompt into the user message —
        # cc's _CallableBackedClient already supports system+user shape.
        # For maximum compatibility we wrap in a single submit_message
        # call and rely on the LLMCallable adapter to surface the system
        # part.
        framed_goal = (
            f"<system>\n{_RESEARCH_SYSTEM_PROMPT}\n</system>\n\n{user_prompt}"
        )
        try:
            async for event in engine.submit_message(
                framed_goal,
                max_tool_rounds=self.max_tool_rounds,
                purpose="research",
            ):
                bridge_sink(event)
                event_count += 1
                if event.event_type == "assistant_tool_use":
                    tool_call_count += 1
                msg = getattr(event, "message", None)
                if msg is None:
                    continue
                role = getattr(msg, "role", "")
                kind = getattr(msg, "kind", "")
                if _is_turn_timeout_message(msg):
                    turn_timed_out = True
                if role == "assistant" and kind == "assistant_text":
                    final_text = str(getattr(msg, "content", ""))
        finally:
            engine.close()
        if turn_timed_out:
            raise TimeoutError(final_text or "research turn timed out")

        summary, evidence, confidence = _parse_findings(final_text)
        return SubagentResult(
            final_text=summary,
            subtasks=[],  # research is terminal — no recursive spawning
            sequential=False,
            extras={
                "tool_call_count": tool_call_count,
                "event_count": event_count,
                "via": "ccx_research",
                "question": invocation.goal,
                "evidence": evidence,
                "confidence": confidence,
                "raw_final_text": final_text,
            },
        )


__all__ = [
    "ResearchRunner",
    "_RESEARCH_TOOL_WHITELIST",
    "_RESEARCH_SYSTEM_PROMPT",
    "_parse_findings",
    "_restrict_tool_registry",
]
