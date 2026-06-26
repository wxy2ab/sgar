"""CcAgentRunner — agent-mode runner that drives cc's full QueryEngine.

When ccx is configured with this runner, each ``ccx.agent`` v5 node
executes a complete cc multi-tool-round turn:

* full LLM↔tool loop (multi-round; honouring ``max_tool_rounds``)
* access to cc's default tool registry (editing, safety, memory,
  bash, file IO, …)
* event stream from cc is collapsed to a terminal ``SubagentResult``
  with the final assistant text and tool-call count

The recursive subagent path (an agent decomposing into sibling agents)
is preserved by having a special ``ccx_spawn`` cc tool that returns
spawn requests; that lives in ``ccx_spawn_tool.py``. When the LLM
calls it, requests buffer; the runner converts them into
``SubagentInvocation`` and returns SpawnResult via the v5 layer.
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import logging
import re
import threading
from dataclasses import dataclass, replace
from functools import wraps
from typing import Any, Mapping

from core.cc.config import CCConfig
from core.cc.runtime import build_default_query_engine
from core.deepstack_v5.memory import ContentStore

from ..modes.llm_client import LLMCallable, LLMResult
from ..modes.llm_client import llm_result_tokens, text_of
from ..modes._text_masking import mask_fenced_segments
from ..services.cost_events import emit_cost_event, report_cost_to_budget
from .governed_spawn import parse_contract, run_governed_spawn
from .subagent import (
    ModeRunner,
    SubagentInvocation,
    SubagentResult,
)

logger = logging.getLogger(__name__)


# Parent invocation metadata keys that are meaningful to spawned
# children and so are auto-propagated when the child does not supply
# its own value. Single source of truth lives in
# ``metadata_inheritance.py``; this module re-exports it under the
# pre-C4 name for backward compatibility with callers that imported
# the private alias (notably the metadata-propagation test).
from .metadata_inheritance import (
    DEFAULT_MAX_SPAWN_DEPTH,
    DEFAULT_MAX_SPAWN_FANOUT,
    INHERITABLE_METADATA_KEYS,
    SPAWN_DEPTH_METADATA_KEY,
    coerce_spawn_depth,
)

# Pre-C4 alias kept for back-compat: tests / docs that imported
# ``_INHERITABLE_PARENT_KEYS`` continue to work. New code should
# import ``INHERITABLE_METADATA_KEYS`` from
# ``core.ccx.agents.metadata_inheritance`` directly.
_INHERITABLE_PARENT_KEYS: tuple[str, ...] = tuple(sorted(INHERITABLE_METADATA_KEYS))


# Generic anti-repeat guardrails appended to caller-supplied system prompts
# so the multi-tool-round LLM loop doesn't burn rounds re-reading the same
# path or re-grepping the same pattern. Lifted from doc.py's investigator
# system prompts (the DEDUPLICATION RULES section); the doc-mode "WHEN TO
# STOP / enter Stage 3" exit conditions are deliberately omitted — that
# logic is investigator-specific and too prescriptive for a general agent.
_GENERIC_DEDUP_TAIL_EN = """\
==========================================================================
DEDUPLICATION RULES — never repeat the same call (violation → shallow)
==========================================================================
The runtime keeps your full tool history in conversation messages but \
does NOT warn you about duplicates. Apply these rules yourself:

* DO NOT `file_read` the same path twice with the same or smaller \
  ``max_bytes``. If a prior read returned ``[truncated to N bytes]`` \
  and you need more content, DOUBLE the ``max_bytes`` \
  (20_000 → 60_000 → 100_000) on the next read, then stop. **A single \
  path must never be `file_read` more than 3 times in one turn** — \
  beyond that you are wasting rounds.
* DO NOT issue the same `grep` ``pattern`` + ``cwd`` combination \
  twice. If you already have the hits, `file_read` the file(s) you \
  saw — do NOT re-grep to "double-check".
* DO NOT use `grep` to enumerate files. Use `glob` for listings. \
  `grep` matches content; using it as a directory enumerator wastes a \
  round and, without ``cwd``, scans the whole repo.
* Before each tool call, mentally check: have I already called this? \
  If yes → either widen the `file_read` ``max_bytes``, or proceed to \
  Stage 3 with the evidence you have.
"""

_GENERIC_DEDUP_TAIL_ZH = """\
==========================================================================
去重规则——同样的调用不要发第二次（关键，违反会被判 shallow）
==========================================================================
运行时把你完整的工具历史保留在 conversation messages 里，但**不会**\
主动提醒你重复了。请自己执行以下规则：

* **同一文件不要用同样或更小的 ``max_bytes`` file_read 两次以上。** \
  如果上一次返回了 ``[truncated to N bytes]`` 且你需要更多内容，\
  下一次把 ``max_bytes`` **翻倍**（20_000 → 60_000 → 100_000）\
  再 read，然后到此为止。**同一路径在一个 turn 内总共最多 file_read \
  3 次**——再多就是浪费 round。
* **同一 grep ``pattern`` + ``cwd`` 组合不要发第二次。** 已经有命中\
  结果，就直接 file_read 看到的那些文件——**不要**再 grep "复核"。
* **不要把 grep 当目录枚举器。** 要列文件用 glob。grep 是匹配内容的，\
  拿去列文件名一是浪费一轮、二是没有 ``cwd`` 限定时会扫全仓。
* **每次发起工具调用前默念一遍**：这个我刚才是不是已经调过了？\
  是 → 要么加大 ``file_read`` 的 ``max_bytes`` 看更多、要么停下来\
  进阶段 3 用已经收集到的证据 emit JSON。
"""


def _compose_system_prompt_with_dedup_tail(
    system_prompt: str,
    *,
    language: str,
) -> str:
    """Append the generic dedup tail to ``system_prompt`` if the caller's
    text doesn't already cover that ground.

    Skip detection is intentionally cheap (case-insensitive substring on
    "DEDUPLICATION" / "去重") — callers that already include their own
    dedup section, or who copy-paste the investigator prompt verbatim,
    won't get the rules twice.
    """
    lowered = system_prompt.lower()
    if "deduplication" in lowered or "去重" in system_prompt:
        return system_prompt
    tail = (
        _GENERIC_DEDUP_TAIL_ZH
        if str(language or "en").lower().startswith("zh")
        else _GENERIC_DEDUP_TAIL_EN
    )
    return f"{system_prompt}\n{tail}"


def _is_turn_timeout_message(msg: Any) -> bool:
    if getattr(msg, "role", "") != "assistant":
        return False
    if getattr(msg, "kind", "") != "assistant_text":
        return False
    metadata = getattr(msg, "metadata", None) or {}
    if metadata.get("turn_timeout_reached") or metadata.get("error_code") == "QE1008":
        return True
    content = str(getattr(msg, "content", "") or "")
    return content.startswith("Turn global timeout reached")


def _child_metadata(raw_metadata: Any, *, depends_on_previous: bool) -> dict[str, Any]:
    metadata = dict(raw_metadata or {})
    metadata.pop("ccx_depends_on", None)
    metadata.pop("ccx_depends_on_previous", None)
    metadata["ccx_depends_on_previous"] = depends_on_previous
    return metadata


# cc's default tool registry includes an ``agent`` tool (cc's own
# sub-agent launcher, ``core.cc.agents.agent_tool.AgentTool``). Inside a
# ccx cc_query_loop turn that tool is both redundant and a governance
# hole: ``ccx_spawn`` is ccx's governed replacement — every child it
# creates becomes a v5 node with inherited metadata (``sgar_session`` …),
# a spawn-depth budget, and audit lineage — whereas a child launched
# through cc's ``agent`` tool escapes ALL of that: no v5 node, no SGAR
# session inheritance, and no ``max_spawn_depth`` ceiling (so it silently
# bypasses the recursion guard). Observed in an agent-driven SGAR run:
# the supervisor burned three ``agent`` calls (~128 s) wandering instead
# of issuing the governed ccx_sgar verify/close ops. We drop these
# redundant cc-native orchestration tools so the LLM is steered onto the
# governed ccx surface.
_REDUNDANT_CC_ORCHESTRATION_TOOLS: tuple[str, ...] = ("agent",)


def _drop_cc_native_orchestration_tools(registry: Any) -> list[str]:
    """Remove cc's ungoverned sub-agent launcher(s) from a turn registry.

    Returns the names actually removed. No-ops (returns ``[]``) when the
    registry shape isn't the expected ``_tools`` dict so a cc-internal
    refactor degrades to "tool still present" rather than crashing the
    turn — unlike research's read-only filter, leaving these tools in is a
    governance/quality regression, not a safety hole, so failing closed is
    not warranted here.
    """
    tools = getattr(registry, "_tools", None)
    if not isinstance(tools, dict):
        return []
    removed: list[str] = []
    for name in _REDUNDANT_CC_ORCHESTRATION_TOOLS:
        if name in tools:
            del tools[name]
            removed.append(name)
    return removed


_NEEDS_MODEL_MARKER_RE = re.compile(
    r"(?m)^[ \t]*<<<NEEDS_([A-Z][A-Z0-9_]*)>>>[ \t]*$",
)


def _extract_needs_model_marker(text: str) -> str | None:
    matches = _NEEDS_MODEL_MARKER_RE.findall(
        mask_fenced_segments(text or "", logger=logger),
    )
    if not matches:
        return None
    return matches[-1].lower()


def _coerce_model_override(value: Any) -> dict[str, str] | None:
    if isinstance(value, str) and value:
        return {"model": value}
    if isinstance(value, Mapping):
        model = value.get("model")
        if isinstance(model, str) and model:
            return {"model": model}
    return None


def _coerce_preferred_model_overrides(
    value: Mapping[str, Any] | None,
) -> dict[str, dict[str, str]]:
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, dict[str, str]] = {}
    for alias, raw_override in value.items():
        if not isinstance(alias, str) or not alias:
            continue
        override = _coerce_model_override(raw_override)
        if override is not None:
            out[alias] = override
    return out


def _resolve_preferred_model_overrides(
    preferred_model: str | None,
    preferred_model_overrides: Mapping[str, dict[str, str]],
) -> dict[str, str] | None:
    if not preferred_model:
        return None
    mapped = preferred_model_overrides.get(preferred_model)
    if mapped is not None:
        return dict(mapped)
    if preferred_model_overrides:
        logger.info(
            "ccx agent: ignoring preferred_model=%r for production provider; "
            "no configured model override exists for that alias",
            preferred_model,
        )
    return None


def _wrap_tracking_client(
    client: Any,
    context: _LLMProviderInvocationContext | None,
) -> Any:
    if callable(client):
        return _CallableTrackingClient(client, context)
    return _TrackingClient(client, context)


def _log_missing_preferred_model_mapping(preferred_model: str) -> None:
    logger.info(
        "ccx agent: ignoring preferred_model=%r for callable shim; "
        "no configured llm_routes entry exists for that alias",
        preferred_model,
    )


def _response_text(response: Any) -> str:
    if isinstance(response, LLMResult):
        return response.text
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        content = response.get("content")
    else:
        content = getattr(response, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    pieces.append(text)
        return "\n".join(pieces)
    return text_of(response)


def _response_cost(response: Any) -> float | None:
    cost = getattr(response, "cost_usd", None)
    if cost is None and isinstance(response, Mapping):
        cost = response.get("cost_usd")
    if cost is None:
        return None
    try:
        return float(cost)
    except (TypeError, ValueError):
        return None


def _client_total_tokens(client: Any) -> int | None:
    """Cumulative token counter a chat client exposes, or ``None``.

    cc's production DeepSeek clients tally ``token_count``
    (prompt + completion + reasoning) across every call inside
    ``_update_stats``. A bare ``str`` / tool-dict response carries no
    per-call usage, so the wrapper reads this running counter's delta to
    recover the real per-call token spend. Returns ``None`` when the client
    has no such counter (e.g. the callable shim used by tests).
    """
    raw = getattr(client, "token_count", None)
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _estimate_tokens_from_text(text: str) -> int:
    """Coarse char-based token estimate (~4 chars/token).

    Used only as a fallback when no real token counter is available, so that
    per-call cost telemetry reflects "tokens were spent" instead of a
    misleading literal 0. Counts the response text only (prompt tokens are
    not visible on a bare ``str`` / dict return), so it is a rough lower
    bound, not an exact figure.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


@dataclass(slots=True)
class _LLMProviderInvocationContext:
    mode: str
    metadata: dict[str, Any]
    preferred_model: str | None
    cost_accumulator: list[float]
    token_accumulator: list[int]
    needs_accumulator: list[str]


def _emit_provider_cost_event(
    *, mode: str, cost_usd: float, call_count: int, tokens: int = 0,
) -> None:
    emit_cost_event(
        mode=mode, cost_usd=cost_usd, call_count=call_count, tokens=tokens,
    )


class _TrackingClient:
    _TRACKED_METHODS = {"tool_invoke", "one_chat", "chat", "text_chat"}

    def __init__(
        self,
        client: Any,
        context: _LLMProviderInvocationContext | None,
    ) -> None:
        self._client = client
        self._context = context
        # Baseline for the per-call ``token_count`` delta (see
        # ``_consume_token_delta``). Captured at wrap time so the first
        # tracked call attributes only its own tokens, not the client's
        # whole history. ``None`` ⇒ this client exposes no counter.
        self._token_baseline = _client_total_tokens(client)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._client, name)
        if name in self._TRACKED_METHODS and callable(attr):
            if inspect.iscoroutinefunction(attr):
                @wraps(attr)
                async def _async_tracked(*args: Any, **kwargs: Any) -> Any:
                    return self._track(await attr(*args, **kwargs))
                return _async_tracked

            @wraps(attr)
            def _tracked(*args: Any, **kwargs: Any) -> Any:
                response = attr(*args, **kwargs)
                if inspect.isawaitable(response):
                    async def _await_and_track() -> Any:
                        return self._track(await response)
                    return _await_and_track()
                return self._track(response)
            return _tracked
        return attr

    def _track(self, response: Any) -> Any:
        context = self._context
        if context is None:
            return text_of(response) if isinstance(response, LLMResult) else response
        if isinstance(response, LLMResult):
            context.cost_accumulator.append(float(response.cost_usd))
            context.token_accumulator.append(llm_result_tokens(response))
            text = response.text
            returned: Any = text
        else:
            context.cost_accumulator.append(_response_cost(response) or 0.0)
            text = _response_text(response)
            context.token_accumulator.append(self._consume_token_delta(text))
            returned = response
        marker = _extract_needs_model_marker(text)
        if marker is not None:
            context.needs_accumulator.append(marker)
        return returned

    def _consume_token_delta(self, text: str) -> int:
        """Real per-call token count for a bare (non-LLMResult) response.

        Reads the client's cumulative ``token_count`` delta since the
        previous tracked call — the production DeepSeek clients advance it in
        ``_update_stats`` on every completion, so the delta is that call's
        true token spend. Falls back to a coarse char estimate when the
        client exposes no counter (test shims) or it didn't advance, so the
        accumulator never records a misleading literal 0. Calls within a turn
        are sequential, and each turn wraps its own client instance, so the
        running baseline is race-free.
        """
        current = _client_total_tokens(self._client)
        if current is None:
            return _estimate_tokens_from_text(text)
        baseline = self._token_baseline if self._token_baseline is not None else 0
        self._token_baseline = current
        delta = current - baseline
        if delta > 0:
            return delta
        return _estimate_tokens_from_text(text)


class _CallableTrackingClient(_TrackingClient):
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        response = self._client(*args, **kwargs)
        if inspect.isawaitable(response):
            async def _await_and_track() -> Any:
                return self._track(await response)
            return _await_and_track()
        return self._track(response)


# --------------------------------------------------------------------------- #
# LLMCallable -> cc LLMClientProvider adapter
# --------------------------------------------------------------------------- #

class _CallableBackedClient:
    """A chat client cc's machinery will accept. Maps `one_chat([msgs])` to
    our `LLMCallable(system, user, purpose)` shape.

    cc's ``llm_adapter.invoke()`` (where this client is consumed) passes a
    list of role/content dicts — system + user (+ tool messages mid-loop).
    We collapse the conversation by joining all non-system roles into the
    user prompt; this is fine for stubs, and for real LLMs it falls
    through to the provider that knows the right call shape.
    """

    def __init__(self, llm: LLMCallable, *, purpose: str) -> None:
        self._llm = llm
        self._purpose = purpose

    def one_chat(self, messages: list[dict[str, Any]] | str, **_kw: Any) -> Any:
        if isinstance(messages, str):
            return self._llm(system="", user=messages, purpose=self._purpose)
        system_parts = [
            str(m.get("content", "")) for m in messages
            if m.get("role") == "system"
        ]
        non_system = [
            m for m in messages if m.get("role") != "system"
        ]
        user_parts: list[str] = []
        for m in non_system:
            content = m.get("content", "")
            if isinstance(content, list):
                # Anthropic-style content blocks; flatten text only.
                text_pieces = [
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                content = "\n".join(text_pieces)
            user_parts.append(f"[{m.get('role', 'user')}] {content}")
        return self._llm(
            system="\n\n".join(system_parts),
            user="\n\n".join(user_parts),
            purpose=self._purpose,
        )

    # cc's LLMAdapter may probe these alternative shapes; map them through.
    def chat(self, messages: list[dict[str, Any]], **kw: Any) -> Any:
        return self.one_chat(messages, **kw)

    def __call__(self, messages: list[dict[str, Any]], **kw: Any) -> Any:
        return self.one_chat(messages, **kw)


class LLMCallableProvider:
    """Wrap an ``LLMCallable`` as cc's ``LLMClientProvider`` protocol.

    When ``cc_provider`` is supplied (production path), ``get_client`` delegates
    to it directly so cc's QueryEngine receives the *real* LLM client object —
    preserving structured methods like ``tool_invoke`` that LLMAdapter needs to
    advertise tool schemas to the model. Without this, ``LLMAdapter.invoke``
    falls through to the text-only ``one_chat`` branch and silently drops the
    tool list, leaving the model unable to call ``file_write``/``file_edit``.

    The plain ``LLMCallable`` shim is retained as a fallback for tests that
    inject a stub callable (no real provider).
    """

    def __init__(
        self,
        llm: LLMCallable,
        *,
        cc_provider: Any | None = None,
        llm_routes: Mapping[str, LLMCallable] | None = None,
        preferred_model_overrides: Mapping[str, Any] | None = None,
    ) -> None:
        self._llm = llm
        self._cc_provider = cc_provider
        self._llm_routes = dict(llm_routes or {})
        self._preferred_model_overrides = _coerce_preferred_model_overrides(
            preferred_model_overrides,
        )
        if self._cc_provider is None and self._preferred_model_overrides:
            logger.info(
                "ccx agent: preferred_model_overrides configured without a "
                "production provider; callable shim routing uses llm_routes, "
                "so model overrides will be ignored"
            )
        self._context_var: contextvars.ContextVar[
            _LLMProviderInvocationContext | None
        ] = contextvars.ContextVar("ccx_llm_provider_context", default=None)

    def begin_invocation(
        self,
        *,
        mode: str,
        metadata: Mapping[str, Any] | None,
    ) -> tuple[
        _LLMProviderInvocationContext,
        contextvars.Token[_LLMProviderInvocationContext | None],
    ]:
        meta = dict(metadata or {})
        preferred_model = meta.get("preferred_model")
        context = _LLMProviderInvocationContext(
            mode=mode,
            metadata=meta,
            preferred_model=(
                preferred_model
                if isinstance(preferred_model, str) and preferred_model
                else None
            ),
            cost_accumulator=[],
            token_accumulator=[],
            needs_accumulator=[],
        )
        return context, self._context_var.set(context)

    def end_invocation(
        self,
        token: contextvars.Token[_LLMProviderInvocationContext | None],
    ) -> None:
        self._context_var.reset(token)

    def get_client(
        self,
        *,
        config: CCConfig,
        purpose: str,
        overrides: dict[str, Any] | None = None,
    ) -> Any:
        context = self._context_var.get()
        effective_overrides = dict(overrides or {})
        if self._cc_provider is not None:
            model_overrides = (
                _resolve_preferred_model_overrides(
                    context.preferred_model,
                    self._preferred_model_overrides,
                )
                if context is not None
                else None
            )
            if model_overrides:
                for key, value in model_overrides.items():
                    effective_overrides.setdefault(key, value)
            client = self._cc_provider.get_client(
                config=config,
                purpose=purpose,
                overrides=effective_overrides or None,
            )
            return _wrap_tracking_client(client, context)
        llm = self._llm
        if context is not None and context.preferred_model:
            routed = self._llm_routes.get(context.preferred_model)
            if routed is not None:
                llm = routed
            elif self._llm_routes:
                _log_missing_preferred_model_mapping(context.preferred_model)
        return _wrap_tracking_client(
            _CallableBackedClient(llm, purpose=purpose),
            context,
        )


def _apply_needs_model_marker_result(
    result: SubagentResult,
    marker: str,
) -> SubagentResult:
    extras = dict(result.extras) if result.extras else {}
    extras.setdefault("needs_model", marker)
    subtasks = list(result.subtasks)
    if subtasks:
        subtasks = [
            replace(sub, preferred_model=marker)
            if sub.preferred_model is None
            else sub
            for sub in subtasks
        ]
    return replace(result, extras=extras, subtasks=subtasks)


# --------------------------------------------------------------------------- #
# CcAgentRunner
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class CcAgentRunner(ModeRunner):
    """Run an agent invocation through cc's full QueryEngine.

    ``cc_config`` and ``llm_provider`` are passed through to
    ``build_default_query_engine``; ``cwd`` is the project working
    directory cc operates against (session/audit files land under
    ``cwd/.cc/...``).

    Recursive subagents: when ``enable_ccx_spawn=True`` (the default),
    the ``ccx_spawn`` cc tool is registered into the cc tool registry
    for this turn. The LLM may call it to enqueue child agents; after
    the turn finishes the buffer is drained into SubagentInvocations
    and surfaced via SubagentResult.subtasks → v5 SpawnResult.

    Recursion is depth-limited: every drained child carries
    ``ccx_spawn_depth = parent_depth + 1`` in its metadata, and a turn
    whose own depth is >= ``max_spawn_depth`` has ordinary spawn modes
    refused — the ccx_spawn tool returns a clear policy error to the
    LLM, and any spawn entries that reached a caller-supplied buffer
    anyway are dropped at drain time. Without this, agent → child →
    grandchild chains recurse unboundedly (each generation is a full
    multi-tool-round cc turn) with only ConfigV5.max_loop_iterations
    (10k nodes) as a backstop. Terminal modes (research / sgar) stay
    available at the limit — they cannot recurse.
    """
    cc_config: CCConfig
    llm_provider: Any  # LLMClientProvider
    cwd: str
    max_tool_rounds: int | None = None
    max_spawn_depth: int = DEFAULT_MAX_SPAWN_DEPTH
    # Per-turn ceiling on ordinary-spawn fan-out WIDTH (siblings enqueued
    # across all ccx_spawn calls in one turn). Mirrors the depth guard but
    # bounds breadth; ``None`` disables it. Enforced inside CcxUnifiedTool.
    max_spawn_fanout: int | None = DEFAULT_MAX_SPAWN_FANOUT
    # When True, ccx_research spawn entries count toward the fan-out WIDTH cap
    # alongside ordinary spawn modes (they are exempt by default). Only takes
    # effect on mixed turns that also enqueue spawn modes — see CcxUnifiedTool.
    count_research_in_fanout: bool = False
    spawn_buffer: Any | None = None  # see ccx_spawn_tool
    research_buffer: Any | None = None  # see ccx_research_tool
    sgar_buffer: Any | None = None  # see ccx_sgar_tool
    enable_ccx_spawn: bool = True
    enable_ccx_research: bool = True
    enable_ccx_sgar: bool = True
    # Strip cc's ungoverned ``agent`` sub-agent launcher from each turn's
    # registry so the LLM uses the governed ``ccx_spawn`` surface instead
    # (see ``_drop_cc_native_orchestration_tools``). Default on; a caller
    # that genuinely wants cc's native sub-agent tool can flip it off.
    drop_cc_native_orchestration_tools: bool = True
    mode_name: str = "agent"
    # Phase 2: optional ContentStore forwarded to the event bridge so
    # large ``tool_result`` bodies (>4 KB) land in an FTS5-indexed
    # store rather than being truncated to the v5 event preview.
    content_store: ContentStore | None = None
    # Spawn contract (default OFF). When enabled AND the invocation carries a
    # ``metadata["ccx_contract"]``, this agent's turn runs inside a bounded
    # verification-driven repair loop that gates on independent ``[check:]``
    # commands (see governed_spawn.py). Reuses the SGAR ``[check:]`` executor
    # + timeout. Default off + no contract present ⇒ ``run`` is byte-identical
    # to the pre-contract single-turn path. Wired from
    # ``build_runtime(sgar_run_criterion_checks=..., sgar_criterion_check_timeout_s=...)``
    # so one operator switch governs every machine-check capability in ccx.
    enable_spawn_contract: bool = False
    contract_check_timeout_s: float = 120.0

    def run(self, invocation: SubagentInvocation) -> SubagentResult:
        """Run one agent invocation, optionally under a spawn contract.

        Default path (no contract, or the feature disabled): a single
        ``_run_once`` turn — byte-identical to the pre-contract behaviour.
        With a contract present and ``enable_spawn_contract=True``: the
        bounded verification-repair loop in ``governed_spawn`` owns the
        turn(s), re-running ``_run_once`` with failing-check evidence until
        the independent checks pass or a hard bound trips. A malformed
        contract raises ``ContractError`` (fail loud — never silently drop
        the gate).
        """
        contract = None
        if self.enable_spawn_contract:
            contract = parse_contract(invocation.metadata)
        if contract is None:
            return self._run_once(invocation)
        return run_governed_spawn(
            self._run_once,
            invocation,
            contract,
            cwd=self.cwd,
            check_timeout_s=self.contract_check_timeout_s,
            log=lambda message: logger.info("ccx spawn-contract: %s", message),
        )

    def _run_once(self, invocation: SubagentInvocation) -> SubagentResult:
        # cc's submit_message is async; v5 dispatcher invokes us
        # synchronously. Run inside a fresh asyncio loop in the calling
        # thread (v5 may be running us via ThreadPoolExecutor — each
        # worker thread can have its own loop).
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
        # Register the unified ccx_spawn tool into THIS engine's
        # registry. One tool (wire name "ccx_spawn") replaces the
        # previous trio (ccx_spawn / ccx_research / ccx_sgar). The three
        # buffer dataclasses remain because the drain logic below still
        # walks each buffer separately to produce the right
        # SubagentInvocation shape per mode.
        registry = getattr(engine.tool_orchestrator, "registry", None)

        # Strip cc's ungoverned ``agent`` sub-agent launcher so the LLM is
        # steered onto the governed ``ccx_spawn`` surface — children that
        # bypass it get no v5 node, no metadata inheritance, and no
        # spawn-depth ceiling. See _drop_cc_native_orchestration_tools.
        if self.drop_cc_native_orchestration_tools and registry is not None:
            dropped = _drop_cc_native_orchestration_tools(registry)
            if dropped:
                logger.debug(
                    "ccx cc_query_loop: dropped redundant cc-native "
                    "orchestration tool(s) from turn registry: %s",
                    dropped,
                )

        # Recursion guard: how many ccx_spawn generations sit above this
        # turn. The parent's drain stamped this authoritatively; missing
        # key (root turn, or a hand-built invocation) counts as 0. At or
        # past the ceiling, ordinary spawn modes are refused for the
        # whole turn — the unified tool carries the reason so the LLM
        # gets policy text instead of a wiring error, and the drain
        # below drops anything that landed in a caller-supplied buffer.
        spawn_depth = coerce_spawn_depth(
            (invocation.metadata or {}).get(SPAWN_DEPTH_METADATA_KEY)
        )
        spawn_depth_exceeded = spawn_depth >= self.max_spawn_depth
        spawn_refusal_reason: str | None = None
        if spawn_depth_exceeded and self.enable_ccx_spawn:
            spawn_refusal_reason = (
                f"recursive spawn depth limit reached (depth="
                f"{spawn_depth}, max={self.max_spawn_depth}). Child-agent "
                f"spawning is disabled for this turn; complete the goal "
                f"directly with your own tools instead of delegating. "
                f"mode='research' and mode='sgar' remain available."
            )

        spawn_buffer = self.spawn_buffer
        if (
            self.enable_ccx_spawn
            and spawn_buffer is None
            and not spawn_depth_exceeded
        ):
            from .ccx_spawn_tool import SpawnBuffer
            spawn_buffer = SpawnBuffer()

        research_buffer = self.research_buffer
        if self.enable_ccx_research and research_buffer is None:
            from .ccx_research_tool import ResearchBuffer
            research_buffer = ResearchBuffer()

        sgar_buffer = self.sgar_buffer
        if self.enable_ccx_sgar and sgar_buffer is None:
            from .ccx_sgar_tool import SgarBuffer
            sgar_buffer = SgarBuffer()

        if (
            self.enable_ccx_spawn
            or self.enable_ccx_research
            or self.enable_ccx_sgar
        ) and registry is not None and hasattr(registry, "register"):
            from .ccx_tool import CcxUnifiedTool
            registry.register(
                CcxUnifiedTool(
                    spawn_buffer=(
                        spawn_buffer
                        if self.enable_ccx_spawn and not spawn_depth_exceeded
                        else None
                    ),
                    research_buffer=research_buffer if self.enable_ccx_research else None,
                    sgar_buffer=sgar_buffer if self.enable_ccx_sgar else None,
                    spawn_unavailable_reason=spawn_refusal_reason,
                    max_fanout=self.max_spawn_fanout,
                    count_research_in_fanout=self.count_research_in_fanout,
                )
            )
            # Register the legacy ``ccx_research`` and ``ccx_sgar`` wire
            # names as hidden aliases (is_enabled=False) so e2e tests
            # that hardcode those names in fake LLM tool_calls still
            # dispatch correctly. The legacy ``ccx_spawn`` class is NOT
            # re-registered — its wire name is owned by the unified tool
            # already, and re-registering would overwrite the visible
            # unified tool in the registry's name → instance dict.
            if self.enable_ccx_research and research_buffer is not None:
                from .ccx_research_tool import CcxResearchTool
                registry.register(CcxResearchTool(buffer=research_buffer))
            if self.enable_ccx_sgar and sgar_buffer is not None:
                from .ccx_sgar_tool import CcxSgarTool
                registry.register(CcxSgarTool(buffer=sgar_buffer))

        # Phase 5: register ctx_search whenever a ContentStore is attached.
        # The tool lets the LLM recall the full body of a prior large
        # tool_result that was offloaded by event_bridge.event_to_v5 — the
        # outlet to Phase 2's inlet.
        if (
            self.content_store is not None
            and registry is not None
            and hasattr(registry, "register")
        ):
            from .ctx_search_tool import CcxCtxSearchTool
            registry.register(CcxCtxSearchTool(content_store=self.content_store))

        # Build the cc→v5 event bridge sink. Inside a real v5 dispatch
        # the sink resolves to the dispatcher's emit callback (so cc
        # SessionEvents become persisted v5 events tagged with this
        # run_id/node_id). Outside a dispatch — e.g. when CcAgentRunner
        # runs in a unit test that calls .run() directly — the sink is
        # a no-op and the loop behaves exactly as before.
        from .event_bridge import make_event_sink
        bridge_sink = make_event_sink(content_store=self.content_store)

        # When the caller stamped ``system_prompt`` into invocation
        # metadata, append the generic dedup tail (unless the caller's
        # text already covers it) and frame the result as a leading
        # ``<system>...</system>`` block in the goal. cc's own default
        # system prompt is still applied by the QueryEngine; this is an
        # additive supplement, mirroring research_runner's pattern.
        # Without ``system_prompt`` the goal is passed unchanged so the
        # default cc loop and existing tests behave exactly as before.
        caller_meta = invocation.metadata or {}
        caller_system_prompt = str(caller_meta.get("system_prompt") or "")
        caller_language = str(caller_meta.get("language") or "en")
        effective_goal = invocation.goal
        if caller_system_prompt:
            framed_system = _compose_system_prompt_with_dedup_tail(
                caller_system_prompt, language=caller_language,
            )
            effective_goal = (
                f"<system>\n{framed_system}\n</system>\n\n{invocation.goal}"
            )

        final_text = ""
        turn_timed_out = False
        tool_call_count = 0
        event_count = 0
        try:
            async for event in engine.submit_message(
                effective_goal,
                max_tool_rounds=self.max_tool_rounds,
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
            # A timed-out turn has no trustworthy final tool plan. Drop
            # buffered child requests so the parent sees a hard failure
            # and v5 retry policy can rerun the whole turn from scratch.
            for buffer in (research_buffer, spawn_buffer, sgar_buffer):
                if buffer is not None and hasattr(buffer, "drain"):
                    buffer.drain()
            raise TimeoutError(final_text or "cc turn timed out")

        # Drain spawn / research / sgar buffers (the ones we created above
        # OR the explicitly supplied ones). Order: research first, then
        # ordinary spawns, then sgar ops — siblings unless a per-item
        # ``sequential_with_previous`` flag chains them. Parent
        # invocation metadata that is meaningful to children (sgar
        # session, cwd, request metadata, stage / mission ids) is
        # auto-propagated via ``_inherit`` so the LLM doesn't have to
        # restate it on every spawn; child-supplied keys win.
        subtasks: list[SubagentInvocation] = []
        research_count = 0
        sgar_count = 0

        parent_meta = invocation.metadata or {}
        # Children are one spawn generation below this turn. Stamped
        # AFTER the child-supplied metadata merge so an LLM that passes
        # its own ccx_spawn_depth on a spawn request cannot reset the
        # counter.
        child_spawn_depth = spawn_depth + 1

        def _inherit(child_meta: dict[str, Any]) -> dict[str, Any]:
            out = dict(child_meta)
            for key in INHERITABLE_METADATA_KEYS:
                if key in parent_meta and key not in out:
                    out[key] = parent_meta[key]
            out[SPAWN_DEPTH_METADATA_KEY] = child_spawn_depth
            return out

        if research_buffer is not None and hasattr(research_buffer, "drain"):
            from .ccx_research_tool import normalize_focus_paths
            for raw in research_buffer.drain():
                depends_on_previous = bool(raw.get("sequential_with_previous")) and research_count > 0
                research_count += 1
                metadata = {
                    "ccx_parent_mode": "agent",
                    "ccx_recursive": False,
                    "ccx_spawn_origin": "ccx_research_tool",
                    "scope": raw.get("scope", ""),
                    "focus_paths": normalize_focus_paths(raw.get("focus_paths")),
                    **_child_metadata(
                        raw.get("metadata"),
                        depends_on_previous=depends_on_previous,
                    ),
                }
                subtasks.append(SubagentInvocation(
                    goal=raw["question"],
                    mode="research",
                    metadata=_inherit(metadata),
                ))

        spawn_depth_refused = 0
        if spawn_buffer is not None and hasattr(spawn_buffer, "drain"):
            raw_entries = spawn_buffer.drain()
            if spawn_depth_exceeded and raw_entries:
                # Caller-supplied buffer carrying entries despite the
                # depth limit (the tool-registration refusal above only
                # covers buffers the LLM can reach). Drop them — past
                # the ceiling nothing may spawn through this path.
                spawn_depth_refused = len(raw_entries)
                raw_entries = []
            spawn_index = 0
            for raw in raw_entries:
                depends_on_previous = bool(raw.get("sequential_with_previous")) and spawn_index > 0
                metadata = {
                    "ccx_parent_mode": "agent",
                    "ccx_recursive": True,
                    "ccx_spawn_origin": "ccx_spawn_tool",
                    **_child_metadata(
                        raw.get("metadata"),
                        depends_on_previous=depends_on_previous,
                    ),
                }
                subtasks.append(SubagentInvocation(
                    goal=raw["goal"],
                    mode=raw.get("mode", "agent"),
                    metadata=_inherit(metadata),
                ))
                spawn_index += 1

        if sgar_buffer is not None and hasattr(sgar_buffer, "drain"):
            for raw in sgar_buffer.drain():
                # Chain EVERY sgar op to its predecessor in emission order,
                # not just when the LLM passed sequential=true. SGAR ops
                # mutate a shared, precondition-gated state machine; two
                # ops from one turn running as parallel siblings race — a
                # close-stage that starts before its record-verification
                # fails "stage cannot close without a verification report"
                # and aborts the workflow. (Reproduced in an agent-driven
                # run where the LLM split verify + close into two ccx_sgar
                # calls, so neither carried sequential_with_previous.)
                # Sequential-by-default is the only safe ordering for a
                # hard governance DAG; the ``sequential_with_previous``
                # flag stays on the buffered record for trace fidelity but
                # no longer gates the dependency.
                depends_on_previous = sgar_count > 0
                sgar_count += 1
                metadata = {
                    "ccx_parent_mode": "agent",
                    "ccx_recursive": False,
                    "ccx_spawn_origin": "ccx_sgar_tool",
                    **_child_metadata(
                        raw.get("metadata"),
                        depends_on_previous=depends_on_previous,
                    ),
                }
                subtasks.append(SubagentInvocation(
                    # ``mode`` is "sgar" (→ ccx.sgar / .sgar/) or "sgarx"
                    # (→ ccx.sgarx / .sgarx/, with reopen/abandon). Both
                    # share this buffer; the unified tool stamps the variant.
                    goal=raw["instruction"],
                    mode=raw.get("mode", "sgar"),
                    metadata=_inherit(metadata),
                ))

        extras: dict[str, Any] = {
            "tool_call_count": tool_call_count,
            "event_count": event_count,
            "via": "cc_query_loop",
            "goal": invocation.goal,
            "spawn_count": len(subtasks),
            "research_count": research_count,
            "sgar_count": sgar_count,
        }
        if spawn_depth_exceeded:
            extras["spawn_depth"] = spawn_depth
            extras["spawn_depth_limit"] = self.max_spawn_depth
            extras["spawn_depth_refused"] = spawn_depth_refused
        # Diagnostic for the "queued children then quit" signature: an agent
        # that emits spawns as its terminal action with no substantive
        # final_text produced no output of its own — the drained children run
        # AFTER this turn and their results are NOT fed back here (the root
        # cannot harvest them). Flag it (additive extras key + warning, only
        # on this specific shape, so other runs stay byte-identical) so callers
        # don't mistake the empty/placeholder final_text for the real output,
        # which lives in the child nodes (see api session_snapshot.child_artifacts).
        if subtasks and not final_text.strip():
            extras["spawned_without_harvest"] = len(subtasks)
            logger.warning(
                "ccx cc_query_loop: agent emitted %d spawn(s) as its terminal "
                "action with empty final_text; spawned children run after this "
                "turn and their results are not returned to this agent. Read "
                "their output from the child nodes / "
                "session_snapshot['child_artifacts'], not this run's final_text.",
                len(subtasks),
            )
        return SubagentResult(
            final_text=final_text,
            subtasks=subtasks,
            sequential=False,
            extras=extras,
        )


# --------------------------------------------------------------------------- #
# Event-loop helpers
# --------------------------------------------------------------------------- #

def _run_in_fresh_loop(coro: Any) -> Any:
    """Run a coroutine in a brand-new event loop on the *current* thread.

    ``asyncio.run()`` is the natural choice but raises if a loop already
    runs in the thread. In v5 each worker thread is independent so this
    is fine; in tests where the runner is invoked from a thread that may
    have a loop already running, we offload to a sub-thread instead.
    """
    try:
        asyncio.get_running_loop()
        # We're inside a live loop — run the coroutine in a side thread.
        return _run_in_thread(coro)
    except RuntimeError:
        # No running loop on this thread; safe to start one fresh.
        return asyncio.run(coro)


def _run_in_thread(coro: Any) -> Any:
    box: dict[str, Any] = {}
    ctx = contextvars.copy_context()

    def target() -> None:
        try:
            box["result"] = ctx.run(asyncio.run, coro)
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box.get("result")


# ``_run_in_fresh_loop`` is exported with the leading underscore
# preserved to mark it as "intentional sibling-runner helper":
# ``agents/research_runner.py`` reuses this asyncio-loop-isolation
# trampoline to keep its own QueryEngine off the caller's running
# loop, mirroring what CcAgentRunner does in cc_query_loop mode.
# Listing it in ``__all__`` keeps the C2 boundary scanner happy
# while the underscore continues to signal "not part of the
# public CcAgentRunner surface".
__all__ = [
    "CcAgentRunner",
    "LLMCallableProvider",
    "_GENERIC_DEDUP_TAIL_EN",
    "_GENERIC_DEDUP_TAIL_ZH",
    "_LLMProviderInvocationContext",
    "_apply_needs_model_marker_result",
    "_compose_system_prompt_with_dedup_tail",
    "_emit_provider_cost_event",
    "_is_turn_timeout_message",
    "_run_in_fresh_loop",
]
