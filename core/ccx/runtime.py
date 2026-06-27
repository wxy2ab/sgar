"""ccx runtime wiring — translates ccx mode runners into v5 ToolSpecs
and exposes a single `build_runtime()` factory.

Lives between cc's CCConfig / LLMClientProvider and v5's RuntimeV5.

Three v5 tools are registered, one per mode:
* `ccx.plan`  — runs a PlanModeRunner; spawns spec children.
* `ccx.spec`  — runs a SpecModeRunner; spawns agent children.
* `ccx.agent` — runs an AgentModeRunner; either terminal or recursive.

The tool fn signature is ``(*, goal, metadata)`` — both keys come from the
NodeSpec.params constructed by ``to_spawn_result``. v5 dispatcher unpacks
params via ``**``.
"""

from __future__ import annotations

import dataclasses
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from core.deepstack_v5 import (
    Budget,
    ConfigV5,
    NodeSpec,
    RuntimeV5,
    SpawnResult,
    ToolSpec,
)

from .agents.metadata_inheritance import (
    SPAWN_DEPTH_METADATA_KEY,
    coerce_spawn_depth,
)
from .agents.subagent import (
    ModeRunner,
    SubagentInvocation,
    SubagentResult,
    to_spawn_result,
)
from .modes import (
    AgentModeRunner,
    AskModeRunner,
    BlueprintModeRunner,
    BlueprintxModeRunner,
    DocModeRunner,
    LLMCallable,
    LLMResult,
    llm_result_tokens,
    text_of,
    PlanDiagnosticsTracer,
    PlanModeRunner,
    SpecModeRunner,
    from_provider,
)
from .services import (
    FindingsCollector,
    RepositoryOutlineCache,
    SteerInbox,
    format_steer_block,
    steer_payload_hash,
)
from .services.cost_events import emit_cost_event, report_cost_to_budget
from .modes._sgar_command_helpers import CCX_GOAL_OFFSET_METADATA_KEY
from .modes._text_masking import mask_fenced_segments
from .memory.inject import read_memory_block


logger = logging.getLogger(__name__)

RESUME_EVENT_METADATA_KEY = "ccx.resume.pending_event"


def _build_cc_agent_runner(
    *, llm: LLMCallable, cwd: str, cc_config: Any | None,
    max_tool_rounds: int | None,
    cc_provider: Any | None = None,
    llm_routes: Mapping[str, LLMCallable] | None = None,
    preferred_model_overrides: Mapping[str, Any] | None = None,
    content_store: Any | None = None,
    max_spawn_depth: int | None = None,
    max_spawn_fanout: int | None = None,
    count_research_in_fanout: bool = False,
    enable_spawn_contract: bool = False,
    contract_check_timeout_s: float = 120.0,
    enable_ask_human: bool = False,
    interaction_timeout_s: float = 300.0,
):
    """Lazy import of CcAgentRunner so ccx works without cc's full stack
    when the lite agent runner is sufficient.

    ``cc_provider`` is the project's real ``LLMClientProvider`` (e.g.
    ``DefaultLLMClientProvider``). When supplied, the cc QueryEngine
    receives the underlying LLM client unwrapped — so structured
    capabilities like ``tool_invoke`` reach the LLMAdapter. Otherwise
    we fall back to a text-only callable shim (test path).

    ``content_store`` is forwarded to the cc → v5 event bridge so
    large ``tool_result`` bodies are FTS5-indexed for later retrieval
    instead of truncated to a 240-char preview. ``None`` disables the
    indexing path (Phase 1 behaviour).
    """
    from core.cc.config import CCConfig
    from .agents.cc_agent import CcAgentRunner, LLMCallableProvider
    extra_kwargs: dict[str, Any] = {}
    if max_spawn_depth is not None:
        extra_kwargs["max_spawn_depth"] = max_spawn_depth
    if max_spawn_fanout is not None:
        extra_kwargs["max_spawn_fanout"] = max_spawn_fanout
    return CcAgentRunner(
        cc_config=cc_config or CCConfig(),
        llm_provider=LLMCallableProvider(
            llm,
            cc_provider=cc_provider,
            llm_routes=llm_routes,
            preferred_model_overrides=preferred_model_overrides,
        ),
        cwd=cwd,
        max_tool_rounds=max_tool_rounds,
        content_store=content_store,
        count_research_in_fanout=count_research_in_fanout,
        enable_spawn_contract=enable_spawn_contract,
        contract_check_timeout_s=contract_check_timeout_s,
        enable_ask_human=enable_ask_human,
        interaction_timeout_s=interaction_timeout_s,
        **extra_kwargs,
    )


def _build_research_runner(
    *, llm: LLMCallable, cwd: str, cc_config: Any | None,
    max_tool_rounds: int | None,
    cc_provider: Any | None = None,
    llm_routes: Mapping[str, LLMCallable] | None = None,
    preferred_model_overrides: Mapping[str, Any] | None = None,
):
    """Lazy import of ResearchRunner — same import-deferral reasoning as
    ``_build_cc_agent_runner``. Returns None on import failure so a build
    that doesn't have cc available still produces a working runtime
    (just without ``ccx.research``).

    ``cc_provider`` is forwarded into ``LLMCallableProvider`` so the
    research runner's QueryEngine receives a real client with
    ``tool_invoke``; see ``_build_cc_agent_runner`` for the full why.
    """
    from core.cc.config import CCConfig
    from .agents.cc_agent import LLMCallableProvider
    from .agents.research_runner import ResearchRunner
    return ResearchRunner(
        cc_config=cc_config or CCConfig(),
        llm_provider=LLMCallableProvider(
            llm,
            cc_provider=cc_provider,
            llm_routes=llm_routes,
            preferred_model_overrides=preferred_model_overrides,
        ),
        cwd=cwd,
        max_tool_rounds=max_tool_rounds,
    )


CCX_MODE_TOOL_MAP = {
    "plan": "ccx.plan",
    "spec": "ccx.spec",
    "agent": "ccx.agent",
    # "research" is an INTERNAL mode — not exposed as a top-level
    # ``CodeAgent.agent_mode`` value. It exists only so that buffered
    # ``ccx_research`` requests drained from a cc turn can be turned
    # into v5 NodeSpecs via to_spawn_result. Plan/spec/agent runners
    # may also emit research subtasks if they decide investigation is
    # needed before further decomposition.
    "research": "ccx.research",
    # "doc" / "ask" are public top-level modes. The v5 tool used for
    # all three doc phases (planner / investigator / synthesizer) is
    # the same — DocModeRunner reads the phase from
    # ``invocation.metadata["ccx_doc_phase"]``.
    "doc": "ccx.doc",
    "ask": "ccx.ask",
    "blueprint": "ccx.blueprint",
    "sgar": "ccx.sgar",
    "sgarx": "ccx.sgarx",
}


@dataclass(slots=True)
class CcxRuntimeBundle:
    """Holds the assembled v5 RuntimeV5 plus the LLMCallable so the
    caller can introspect / override later.
    """
    runtime: RuntimeV5
    llm: LLMCallable
    plan_runner: PlanModeRunner
    spec_runner: SpecModeRunner
    agent_runner: ModeRunner   # AgentModeRunner OR CcAgentRunner
    research_runner: ModeRunner | None = None  # ResearchRunner OR None
    doc_runner: DocModeRunner | None = None
    ask_runner: AskModeRunner | None = None
    blueprint_runner: BlueprintModeRunner | None = None
    sgar_runner: BlueprintModeRunner | None = None
    sgarx_runner: BlueprintxModeRunner | None = None
    outline_cache: RepositoryOutlineCache | None = None
    findings_collector: FindingsCollector | None = None
    engine: Any | None = None
    # Phase 2: optional FTS5-indexed store for large tool_result bodies.
    # Owned by the bundle so ``shutdown`` flushes pending background
    # writes and closes the connection — leaking it would keep the
    # writer thread alive after a run completes.
    content_store: Any | None = None  # ContentStore (untyped to avoid hard import)
    _shutdown_started: bool = field(default=False, init=False, repr=False)
    _shutdown_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False,
    )
    _marked_deletable_run_ids: set[str] = field(
        default_factory=set, init=False, repr=False,
    )

    def shutdown(
        self,
        *,
        run_id: str | None = None,
        content_retain_for_ms: int | None = None,
    ) -> None:
        with self._shutdown_lock:
            run_runtime_shutdown = not self._shutdown_started
            if run_runtime_shutdown:
                self._shutdown_started = True
            mark_run_id = None
            if run_id is not None and run_id not in self._marked_deletable_run_ids:
                self._marked_deletable_run_ids.add(run_id)
                mark_run_id = run_id
        if not run_runtime_shutdown and mark_run_id is None:
            return
        if run_runtime_shutdown and self.engine is not None:
            # Cancel any in-flight parallel-dispatch executors before we flush
            # downstream stores, so no worker is still trying to write to the
            # content_store / DB while we close them. Engine.shutdown never
            # blocks (wait=False, cancel_futures=True). Best-effort: a teardown
            # error must not mask the rest of shutdown.
            engine_shutdown = getattr(self.engine, "shutdown", None)
            if callable(engine_shutdown):
                try:
                    engine_shutdown()
                except Exception:
                    logger.warning(
                        "ccx: engine.shutdown() failed during bundle teardown",
                        exc_info=True,
                    )
        if self.content_store is not None:
            if run_runtime_shutdown and hasattr(self.content_store, "flush"):
                try:
                    self.content_store.flush(timeout_s=1.0)
                except Exception:
                    logger.warning(
                        "ccx content store flush failed during shutdown",
                        exc_info=True,
                    )
            if run_runtime_shutdown:
                try:
                    self.content_store.close()
                except Exception:
                    logger.warning(
                        "ccx content store close failed during shutdown",
                        exc_info=True,
                    )
            if mark_run_id is not None:
                try:
                    kwargs: dict[str, Any] = {}
                    if content_retain_for_ms is not None:
                        kwargs["retain_for_ms"] = content_retain_for_ms
                    self.content_store.mark_run_deletable(mark_run_id, **kwargs)
                except Exception:
                    logger.warning(
                        "ccx content store mark_run_deletable failed during shutdown",
                        exc_info=True,
                    )
                try:
                    self.content_store.close()
                except Exception:
                    logger.warning(
                        "ccx content store close after GC mark failed during shutdown",
                        exc_info=True,
                    )
        if run_runtime_shutdown:
            self.runtime.shutdown()


# --------------------------------------------------------------------------- #
# ToolSpec wrapping
# --------------------------------------------------------------------------- #

def _emit_steer_event(*, items: list[str], mode: str) -> None:
    """Publish ``ccx.steer.injected`` on the active v5 dispatch context.

    Uses ``current_dispatch_context()`` instead of holding an event_bus
    reference so we don't need to thread it through ``_make_mode_tool``
    or solve the chicken-and-egg of ``RuntimeV5.create`` not existing
    yet at capability-construction time. Outside a dispatch (e.g. unit
    tests that call fn directly) the context is None and we no-op.
    """
    from core.deepstack_v5.execution.dispatch_context import (
        current_dispatch_context,
    )
    ctx = current_dispatch_context()
    if ctx is None:
        return
    ctx.emit("ccx.steer.injected", {
        "run_id": ctx.run_id,
        "node_id": ctx.node_id,
        "attempt_id": ctx.attempt_id,
        "mode": mode,
        "steer_count": len(items),
        "steer_hash": steer_payload_hash(items),
    })


def _emit_pending_resume_event(payload: Mapping[str, Any], *, mode: str) -> None:
    from core.deepstack_v5.execution.dispatch_context import (
        current_dispatch_context,
    )

    ctx = current_dispatch_context()
    if ctx is None:
        return
    if ctx.attempt_ordinal != 1:
        return
    kind = str(payload.get("kind") or "")
    if kind not in {"ccx.resume.injected", "ccx.resume.skipped"}:
        return
    body = dict(payload.get("payload") or {})
    body.update({
        "run_id": ctx.run_id,
        "node_id": ctx.node_id,
        "attempt_id": ctx.attempt_id,
        "mode": mode,
    })
    ctx.emit(kind, body)


def _emit_cost_event(
    *, mode: str, cost_usd: float, call_count: int, tokens: int = 0,
) -> None:
    """Publish ``ccx.cost.node`` on the active v5 dispatch context (R4).

    Fires once per node, after the runner finishes. ``cost_usd`` is
    the sum of all per-LLM-call costs reported during this node's
    run; ``call_count`` is the number of LLM calls (regardless of
    whether each one carried a cost). A node with ``call_count > 0``
    and ``cost_usd == 0`` means "the LLMCallable returns bare strings
    so cost is unknown / unmeasured (no core/llms client exposes a
    per-call USD cost) — NOT that the node was free" — distinguishable
    from "the runner made no LLM calls" (``call_count == 0`` ⇒ no event
    emitted). ``tokens`` is still populated in the cost-unknown case (a
    coarse char-based estimate of the response text; see
    ``_estimate_tokens_from_text``), so the figure is never a silent
    literal 0.

    Same dispatch-context dance as ``_emit_steer_event``: silently
    no-op outside a dispatch so unit tests calling fn directly don't
    fail.
    """
    emit_cost_event(
        mode=mode, cost_usd=cost_usd, call_count=call_count, tokens=tokens,
    )


# R1 Step B: marker the LLM can emit on its own line to
# self-report "this task is too hard for my current model; route the
# spawned children (and signal the caller) to a stronger model". The
# captured group is the model key, lower-cased before being routed
# (``<<<NEEDS_PRO>>>`` → ``"pro"``; the caller's ``llm_routes`` decides
# what ``"pro"`` actually maps to). Marker is left in the response text
# so the runner's parser sees the same string it always has — the
# marker just rides alongside as a side-channel signal.
_NEEDS_MODEL_MARKER_RE = re.compile(
    r"(?m)^[ \t]*<<<NEEDS_([A-Z][A-Z0-9_]*)>>>[ \t]*$",
)


def _extract_needs_model_marker(text: str) -> str | None:
    """Return the last ``<<<NEEDS_X>>>`` marker as a lower-case route
    key, or ``None`` if the text carries no standalone marker. Last-wins so a
    multi-call runner that escalates partway through (e.g. tried
    flash then escalated to pro) gets the most recent signal.
    """
    if not text:
        return None
    matches = _NEEDS_MODEL_MARKER_RE.findall(
        mask_fenced_segments(text, logger=logger)
    )
    if not matches:
        return None
    return matches[-1].lower()


def _estimate_tokens_from_text(text: str) -> int:
    """Coarse char-based token estimate (~4 chars/token).

    Fallback used when no real token counter is reachable, so the lite
    path's cost telemetry reflects "tokens were spent" instead of a
    misleading literal 0. Mirrors ``agents/cc_agent.py``'s helper of the
    same name (the cc path's own fallback), keeping the two paths
    symmetric. Counts the response text only (prompt tokens are not
    visible on a bare return), so it is a rough lower bound, not exact.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def _wrap_llm_for_cost_and_needs_model(
    inner: LLMCallable,
    cost_acc: list[float],
    needs_acc: list[str],
    token_acc: list[int] | None = None,
) -> LLMCallable:
    """Wrap an ``LLMCallable`` so each call's reported cost lands in
    ``cost_acc`` and any standalone ``<<<NEEDS_X>>>`` marker in the response text
    lands in ``needs_acc``. The runner sees a bare ``str`` either way.

    R4 (cost) and R1 Step B (needs_model) ride on the same wrapper:
    a single pass over the response text covers both signals, and
    they share the "always-on, zero-cost when unused" contract — a
    pre-R4 stub that returns plain strings without markers will
    populate ``cost_acc=[0.0]`` and ``needs_acc=[]``, both indicating
    "nothing reported, no event needed".

    The marker is intentionally not stripped from the returned text:
    if the LLM wrote ``"<<<NEEDS_PRO>>>"`` on its own line and the runner's
    parser is whitespace-tolerant, the marker character bytes are
    benign payload. Stripping would risk breaking JSON parsers in
    edge cases where the LLM embedded the marker inside a string
    field. The marker is detected by regex on the same text — no
    mutation required.
    """
    def _tracked(*, system: str, user: str, purpose: str) -> str:
        out = inner(system=system, user=user, purpose=purpose)
        text = text_of(out)
        if isinstance(out, LLMResult):
            cost_acc.append(float(out.cost_usd))
            if token_acc is not None:
                token_acc.append(llm_result_tokens(out))
        else:
            # Bare str/dict return. This layer wraps an *LLMCallable*, not a
            # chat client, so the production adapter's real ``token_count`` is
            # not reachable here — estimate tokens from the response text, the
            # same char-based fallback the cc path uses
            # (``_estimate_tokens_from_text`` in agents/cc_agent.py). This
            # keeps the lite path symmetric with cc: it reports "tokens were
            # spent" instead of a misleading literal 0.
            #
            # cost stays 0.0 = UNMEASURED, not free: no client in core/llms
            # exposes a per-call USD cost (verified 2026-06-16), so a bare
            # return carries none. The emitted ``ccx.cost.node`` event keeps
            # this honest rather than silently zeroed — ``call_count > 0`` with
            # ``cost_usd == 0`` and ``tokens > 0`` is the documented
            # "cost unknown" signal (see ``_emit_cost_event``).
            cost_acc.append(0.0)
            if token_acc is not None:
                token_acc.append(_estimate_tokens_from_text(text))
        marker = _extract_needs_model_marker(text)
        if marker is not None:
            needs_acc.append(marker)
        return text
    return _tracked


# Back-compat shim for callers (and tests) that still import the R4
# wrapper by its original name. The two-list version is the canonical
# implementation; this thin wrapper drops the needs_model output.
def _wrap_llm_for_cost_tracking(
    inner: LLMCallable, accumulator: list[float],
) -> LLMCallable:
    """Pre-R1B wrapper kept for API stability. Records cost into
    ``accumulator``; needs_model markers are silently ignored. New
    code should use ``_wrap_llm_for_cost_and_needs_model`` so the
    R1B routing signal isn't dropped on the floor.
    """
    sink: list[str] = []
    return _wrap_llm_for_cost_and_needs_model(inner, accumulator, sink)


def _make_mode_tool(
    *,
    name: str,
    runner: ModeRunner,
    parent_id_resolver,
    timeout_s: float | None = None,
    steer_inbox: SteerInbox | None = None,
    language: str = "en",
    llm_routes: Mapping[str, LLMCallable] | None = None,
) -> ToolSpec:
    """Wrap a ModeRunner as a v5 ToolSpec.

    `parent_id_resolver` is a callable returning the current node_id; we
    can't introspect it from inside the tool fn cleanly because v5
    invokes ``fn(**params)``. Instead the engine sets a marker on each
    NodeSpec.metadata before dispatch, and we read it from params via the
    metadata key (we already echo metadata into params).

    ``timeout_s`` is forwarded to ``ToolSpec.timeout_s`` so the v5
    dispatcher's ``_run_tool`` wraps each invocation in
    ``_call_with_timeout``. Without it a hung synchronous LLM call inside
    a mode runner (plan/spec/agent-lite call ``llm(...)`` directly with no
    asyncio context) blocks the dispatcher thread indefinitely; the v5
    engine has no way to surface progress while a single tool invocation
    sits inside ``cap.fn(**tc.params)``. ``None`` preserves the previous
    "block forever" semantics for callers that opt out.

    ``steer_inbox`` enables mid-turn steer injection: before each
    runner invocation, the fn drains the inbox and prepends a wrapped
    block to ``goal`` so the LLM sees the user's supplemental guidance
    as an additional constraint. ``None`` (the default) preserves the
    pre-R2 behaviour exactly — no steer code path is exercised.

    ``language`` controls the steer wrapper's marker text (``en`` or
    ``zh``); body content is never translated. Defaulting to ``en``
    matches ``build_runtime``'s own default.

    ``llm_routes`` enables R1 per-invocation model routing. Lite
    runners swap to the matched callable for that invocation, leaving
    the shared ``runner`` instance unchanged. Cc query-loop runners
    apply routing at the provider boundary so tool-aware clients keep
    their native methods. ``None`` (the default) skips routing entirely;
    behaviour is identical to pre-R1.
    """
    def fn(*, goal: str, metadata: Mapping[str, Any] | None = None,
           **_unused: Any) -> Any:
        meta_dict = dict(metadata or {})
        meta_dict.pop(CCX_GOAL_OFFSET_METADATA_KEY, None)
        pending_resume_event = meta_dict.pop(RESUME_EVENT_METADATA_KEY, None)
        if isinstance(pending_resume_event, Mapping):
            _emit_pending_resume_event(
                pending_resume_event,
                mode=runner.mode_name,
            )
        # Prefix injection: ccx.api may stamp rendered Markdown blocks
        # into root metadata. Persistent memory is project background;
        # resume is single-chain prior-run context. When resume is
        # present, record the full opaque prefix length so SGAR command
        # parsing can recover the true current goal without re-scanning
        # marker text.
        from core.deepstack_v5.memory import read_resume_block
        memory_block = read_memory_block(meta_dict)
        resume_block = read_resume_block(meta_dict)
        prefix_parts: list[str] = []
        if memory_block:
            prefix_parts.append(f"{memory_block}\n")
        if resume_block:
            prefix_parts.append(f"{resume_block}\n## Current goal\n")
        if prefix_parts:
            prefix = "".join(prefix_parts)
            goal = f"{prefix}{goal}"
            meta_dict[CCX_GOAL_OFFSET_METADATA_KEY] = len(prefix)
        # Mid-turn steer injection (R2). Drain happens here — once per
        # dispatch — so the next node about to run sees the constraint
        # in its goal prompt. Drain is FIFO + atomic: parallel siblings
        # do not double-consume the same steer.
        if steer_inbox is not None:
            steer_items = steer_inbox.drain()
            if steer_items:
                steer_block = format_steer_block(steer_items, language=language)
                meta_dict[CCX_GOAL_OFFSET_METADATA_KEY] = int(
                    meta_dict.get(CCX_GOAL_OFFSET_METADATA_KEY) or 0
                ) + len(steer_block)
                goal = f"{steer_block}{goal}"
                _emit_steer_event(
                    items=steer_items, mode=runner.mode_name,
                )
        preferred_model = meta_dict.get("preferred_model")
        invocation = SubagentInvocation(
            goal=goal,
            mode=runner.mode_name,
            metadata=meta_dict,
            preferred_model=preferred_model if isinstance(preferred_model, str)
            and preferred_model else None,
        )
        # R1 routing: pick the LLMCallable for this invocation. If the
        # invocation requests a model that's present in llm_routes,
        # use a shallow-replaced copy of the runner so concurrent
        # dispatches don't fight over a shared ``runner.llm`` slot.
        # ``dataclasses.replace`` skips ``__init__`` for plain
        # dataclasses (no ``__post_init__``); ccx mode runners are
        # all field-only dataclasses so this is cheap.
        effective_runner = runner
        if llm_routes and invocation.preferred_model:
            routed = llm_routes.get(invocation.preferred_model)
            if routed is not None and routed is not getattr(runner, "llm", None):
                try:
                    effective_runner = dataclasses.replace(runner, llm=routed)
                except (TypeError, ValueError):
                    # Runner isn't a dataclass with an ``llm`` field
                    # (e.g. a wrapped non-dataclass runner). Fall back
                    # silently to the default — the alternative would
                    # be a hard error in production for a feature the
                    # caller opted in to.
                    effective_runner = runner
        # R4 cost tracking + R1B needs_model detection share one
        # wrapper pass. Always-on; both lists end up empty / [0.0]
        # for legacy str-returning callables without markers, so the
        # wrapping has no observable effect when nothing's reported.
        cost_accumulator: list[float] = []
        token_accumulator: list[int] = []
        needs_accumulator: list[str] = []
        base_llm = getattr(effective_runner, "llm", None)
        if base_llm is not None:
            try:
                tracked = _wrap_llm_for_cost_and_needs_model(
                    base_llm, cost_accumulator, needs_accumulator,
                    token_accumulator,
                )
                effective_runner = dataclasses.replace(
                    effective_runner, llm=tracked,
                )
            except (TypeError, ValueError):
                # Runner without an ``llm`` dataclass field — both
                # signals silently no-op for it.
                pass
        try:
            result: SubagentResult = effective_runner.run(invocation)
        finally:
            # Emit cost event even on runner exceptions so a partial
            # failure still surfaces what it spent before crashing.
            if cost_accumulator:
                cost_usd = sum(cost_accumulator)
                _emit_cost_event(
                    mode=runner.mode_name,
                    cost_usd=cost_usd,
                    call_count=len(cost_accumulator),
                    tokens=sum(token_accumulator),
                )
                report_cost_to_budget(
                    cost_usd=cost_usd,
                    tokens=sum(token_accumulator),
                )
        # R1B: if any LLM call in this dispatch emitted
        # ``<<<NEEDS_X>>>``, record the last marker on
        # ``result.extras["needs_model"]`` (signal to upstream
        # observers) and use it as the default ``preferred_model``
        # for every spawned child (any child that already declared
        # its own preferred_model wins, mirroring metadata-inherit
        # semantics elsewhere in ccx).
        if needs_accumulator:
            last_marker = needs_accumulator[-1]
            extras = dict(result.extras) if result.extras else {}
            extras.setdefault("needs_model", last_marker)
            result = dataclasses.replace(result, extras=extras)
            if result.subtasks:
                new_subtasks = []
                for sub in result.subtasks:
                    if sub.preferred_model is None:
                        new_subtasks.append(
                            dataclasses.replace(sub, preferred_model=last_marker)
                        )
                    else:
                        new_subtasks.append(sub)
                result = dataclasses.replace(result, subtasks=new_subtasks)
        if not result.subtasks:
            # Preserve extras (e.g. doc-synth's ``artifact_path``) even
            # when ``final_text`` is non-empty. The previous form
            # ``return result.final_text or {...}`` silently dropped
            # extras whenever a runner produced a non-empty final_text,
            # so downstream consumers like ``CodeAgent._build_result``
            # had no way to surface artifact paths to the caller.
            if result.extras:
                return {
                    "final_text": result.final_text or "",
                    "extras": dict(result.extras),
                }
            return result.final_text or ""
        # Spawn-depth propagation: CcAgentRunner stamps an incremented
        # ``ccx_spawn_depth`` on the children it drains, but other mode
        # runners (plan / spec / doc) build child metadata from scratch,
        # which would silently zero the counter — letting a spawn chain
        # launder its depth through an intermediate plan/spec hop
        # (agent → plan → spec → agent restarts at depth 0). Carry the
        # parent's value through unchanged here (no increment: only the
        # cc-agent spawn point consumes depth budget). setdefault keeps
        # CcAgentRunner's authoritative stamp intact.
        if SPAWN_DEPTH_METADATA_KEY in meta_dict:
            inherited_depth = coerce_spawn_depth(
                meta_dict.get(SPAWN_DEPTH_METADATA_KEY)
            )
            for sub in result.subtasks:
                sub.metadata.setdefault(
                    SPAWN_DEPTH_METADATA_KEY, inherited_depth,
                )
        # Children to spawn.
        return to_spawn_result(
            result,
            parent_id=parent_id_resolver(),
            tool_for_mode=CCX_MODE_TOOL_MAP,
        )

    return ToolSpec(
        name=name,
        fn=fn,
        description=f"ccx {runner.mode_name} mode subagent",
        concurrent_safe=True,
        idempotent=False,
        timeout_s=timeout_s,
    )


# --------------------------------------------------------------------------- #
# Build factory
# --------------------------------------------------------------------------- #

def build_runtime(
    *,
    workspace: Path | str,
    llm: LLMCallable | None = None,
    llm_client_provider: Any | None = None,
    cc_config: Any | None = None,
    language: str = "en",
    extra_capabilities: Mapping[str, ToolSpec] | None = None,
    budget: Budget | None = None,
    parallelism: int = 4,
    config_overrides: Mapping[str, Any] | None = None,
    propose_initial: Any | None = None,
    agent_runner_kind: str = "lite",
    cc_cwd: str | None = None,
    cc_max_tool_rounds: int | None = None,
    cc_max_spawn_depth: int | None = None,
    cc_max_spawn_fanout: int | None = None,
    cc_count_research_in_fanout: bool = False,
    artifact_cwd: str | None = None,
    artifact_root: str | None = None,
    docs_artifact_root: str | None = None,
    docs_write_artifact: bool = True,
    docs_output_path: str | None = None,
    tracer: PlanDiagnosticsTracer | None = None,
    node_timeout_s: float | None = None,
    content_store: Any | None = None,
    steer_inbox: SteerInbox | None = None,
    llm_routes: Mapping[str, LLMCallable] | None = None,
    preferred_model_overrides: Mapping[str, Any] | None = None,
    sgar_run_criterion_checks: bool = False,
    sgar_criterion_check_timeout_s: float = 120.0,
    interaction_fn: Callable[[Any], Any] | None = None,
    interaction_timeout_s: float = 300.0,
) -> CcxRuntimeBundle:
    """Build a v5 RuntimeV5 wired with ccx plan/spec/agent tools.

    `llm` and `llm_client_provider` are mutually exclusive — pass one or
    the other. If neither is supplied, raises ValueError. (For tests:
    inject a callable. For production: supply an LLMClientProvider +
    CCConfig and the adapter is built from them.)

    `extra_capabilities` adds non-mode tools (e.g. file-read) the agent
    runner can call indirectly.

    `node_timeout_s` bounds each ccx mode tool invocation via
    ``ToolSpec.timeout_s``. Without it the v5 dispatcher blocks
    indefinitely on a hung LLM call inside plan/spec/agent runners (these
    call ``llm(...)`` synchronously — there's no async wait_for guard like
    the cc QueryEngine uses). ``None`` keeps the legacy unbounded
    behaviour (used by tests).

    `cc_max_spawn_depth` caps how many generations of recursive
    ccx_spawn children a cc_query_loop agent chain may create (see
    ``CcAgentRunner.max_spawn_depth``). ``None`` keeps the runner's
    default (``DEFAULT_MAX_SPAWN_DEPTH``, currently 3).

    `cc_max_spawn_fanout` caps the per-turn fan-out WIDTH (total ordinary
    spawn-mode children one cc_query_loop turn may enqueue across all its
    ccx_spawn calls; see ``CcAgentRunner.max_spawn_fanout``). ``None`` keeps
    the runner's default (``DEFAULT_MAX_SPAWN_FANOUT``, currently 32). Pass a
    runner with ``max_spawn_fanout=None`` to disable the cap entirely.
    """
    if llm is None and llm_client_provider is None:
        raise ValueError(
            "build_runtime requires either `llm` or `llm_client_provider`"
        )
    if llm is None:
        if cc_config is None:
            raise ValueError(
                "passing llm_client_provider also requires cc_config"
            )
        llm = from_provider(llm_client_provider, cc_config)

    plan_runner = PlanModeRunner(
        llm=llm, language=language,
        cwd=artifact_cwd, artifact_root=artifact_root,
        tracer=tracer,
    )
    spec_runner = SpecModeRunner(
        llm=llm, language=language,
        cwd=artifact_cwd, artifact_root=artifact_root,
        tracer=tracer,
    )
    research_runner: ModeRunner | None = None
    if agent_runner_kind == "cc_query_loop":
        agent_runner = _build_cc_agent_runner(
            llm=llm,
            cwd=cc_cwd or str(Path(workspace)),
            cc_config=cc_config,
            max_tool_rounds=cc_max_tool_rounds,
            cc_provider=llm_client_provider,
            llm_routes=llm_routes,
            preferred_model_overrides=preferred_model_overrides,
            content_store=content_store,
            max_spawn_depth=cc_max_spawn_depth,
            max_spawn_fanout=cc_max_spawn_fanout,
            count_research_in_fanout=cc_count_research_in_fanout,
            # Reuse the SGAR machine-check opt-in as the single operator
            # switch for "ccx may run [check:] commands". The spawn contract
            # only engages when this is True AND the invocation carries a
            # ``metadata["ccx_contract"]`` — so existing run_checks=True runs
            # stay byte-equivalent until a contract is actually attached.
            enable_spawn_contract=sgar_run_criterion_checks,
            contract_check_timeout_s=sgar_criterion_check_timeout_s,
            # Enable the ask_human tool iff a host interaction handler is
            # present; off ⇒ the tool is never registered (byte-equivalent).
            enable_ask_human=interaction_fn is not None,
            interaction_timeout_s=interaction_timeout_s,
        )
        # ResearchRunner requires the cc QueryEngine path (it reuses cc's
        # Read/Grep/Glob tools). Only register ccx.research when we have
        # cc available; lite agent mode skips it (no cc tools to call).
        research_runner = _build_research_runner(
            llm=llm,
            cwd=cc_cwd or str(Path(workspace)),
            cc_config=cc_config,
            max_tool_rounds=cc_max_tool_rounds,
            cc_provider=llm_client_provider,
            llm_routes=llm_routes,
            preferred_model_overrides=preferred_model_overrides,
        )
    elif agent_runner_kind == "lite":
        agent_runner = AgentModeRunner(
            llm=llm, language=language, tracer=tracer,
        )
    else:
        raise ValueError(
            f"unknown agent_runner_kind={agent_runner_kind!r}; "
            f"expected 'lite' or 'cc_query_loop'"
        )

    # Run-scoped services for doc / ask. The outline cache is shared so
    # parallel doc investigators don't each rescan the filesystem; the
    # findings collector is the sidecar v5 lacks for predecessor-result
    # propagation (see services/findings_collector.py).
    has_tools = (agent_runner_kind == "cc_query_loop")
    doc_ask_cwd = cc_cwd or str(Path(workspace))
    outline_cache = RepositoryOutlineCache(cwd=doc_ask_cwd)
    findings_collector = FindingsCollector()

    # Lazy provider import — only needed when has_tools. For lite we pass
    # llm_provider=None and the runners take their no-tools branches.
    llm_provider_for_modes: Any | None = None
    cc_config_for_modes: Any | None = cc_config
    if has_tools:
        from core.cc.config import CCConfig
        from .agents.cc_agent import LLMCallableProvider
        llm_provider_for_modes = LLMCallableProvider(
            llm,
            cc_provider=llm_client_provider,
            llm_routes=llm_routes,
            preferred_model_overrides=preferred_model_overrides,
        )
        if cc_config_for_modes is None:
            cc_config_for_modes = CCConfig()

    doc_runner = DocModeRunner(
        llm=llm,
        cwd=doc_ask_cwd,
        cc_config=cc_config_for_modes,
        llm_provider=llm_provider_for_modes,
        language=language,
        parallelism=parallelism,
        outline_cache=outline_cache,
        findings_collector=findings_collector,
        docs_artifact_root=docs_artifact_root,
        output_path=docs_output_path,
        write_artifact=docs_write_artifact,
        has_tools=has_tools,
        max_tool_rounds=cc_max_tool_rounds,
    )
    ask_runner = AskModeRunner(
        llm=llm,
        cwd=doc_ask_cwd,
        cc_config=cc_config_for_modes,
        llm_provider=llm_provider_for_modes,
        language=language,
        outline_cache=outline_cache,
        has_tools=has_tools,
        max_tool_rounds=cc_max_tool_rounds,
    )
    blueprint_runner = BlueprintModeRunner(
        cwd=doc_ask_cwd, llm=llm,
        run_criterion_checks=sgar_run_criterion_checks,
        criterion_check_timeout_s=sgar_criterion_check_timeout_s,
    )
    sgar_runner = BlueprintModeRunner(
        cwd=doc_ask_cwd, mode_name="sgar", llm=llm,
        run_criterion_checks=sgar_run_criterion_checks,
        criterion_check_timeout_s=sgar_criterion_check_timeout_s,
    )
    sgarx_runner = BlueprintxModeRunner(
        cwd=doc_ask_cwd, llm=llm,
        run_criterion_checks=sgar_run_criterion_checks,
        criterion_check_timeout_s=sgar_criterion_check_timeout_s,
    )

    # parent_id_resolver is a placeholder — v5 dispatcher already stamps
    # `parent_node_id` automatically into spawned children's metadata,
    # so we don't actually need to know the parent at tool-fn time.
    def _no_op_resolver() -> str:
        return ""

    # Shared kwargs forwarded into every _make_mode_tool wrapping below.
    # Keeping them in one dict ensures plan / spec / agent / doc / ask /
    # blueprint / sgar / sgarx / research all observe identical steer,
    # language, and llm-routing semantics — adding a 10th call site
    # cannot drift.
    mode_tool_kwargs: dict[str, Any] = {
        "parent_id_resolver": _no_op_resolver,
        "timeout_s": node_timeout_s,
        "steer_inbox": steer_inbox,
        "language": language,
        "llm_routes": llm_routes,
    }
    capabilities: dict[str, ToolSpec] = {
        "ccx.plan": _make_mode_tool(
            name="ccx.plan", runner=plan_runner, **mode_tool_kwargs,
        ),
        "ccx.spec": _make_mode_tool(
            name="ccx.spec", runner=spec_runner, **mode_tool_kwargs,
        ),
        "ccx.agent": _make_mode_tool(
            name="ccx.agent", runner=agent_runner, **mode_tool_kwargs,
        ),
        "ccx.doc": _make_mode_tool(
            name="ccx.doc", runner=doc_runner, **mode_tool_kwargs,
        ),
        "ccx.ask": _make_mode_tool(
            name="ccx.ask", runner=ask_runner, **mode_tool_kwargs,
        ),
        "ccx.blueprint": _make_mode_tool(
            name="ccx.blueprint", runner=blueprint_runner, **mode_tool_kwargs,
        ),
        "ccx.sgar": _make_mode_tool(
            name="ccx.sgar", runner=sgar_runner, **mode_tool_kwargs,
        ),
        "ccx.sgarx": _make_mode_tool(
            name="ccx.sgarx", runner=sgarx_runner, **mode_tool_kwargs,
        ),
    }
    if research_runner is not None:
        capabilities["ccx.research"] = _make_mode_tool(
            name="ccx.research", runner=research_runner, **mode_tool_kwargs,
        )
    if extra_capabilities:
        capabilities.update(extra_capabilities)

    cfg = ConfigV5()
    cfg.parallelism = parallelism
    if config_overrides:
        allowed_config_keys = {field.name for field in dataclasses.fields(ConfigV5)}
        unknown = sorted(set(config_overrides) - allowed_config_keys)
        if unknown:
            allowed = ", ".join(sorted(allowed_config_keys))
            raise ValueError(
                "unknown config_overrides key(s): "
                f"{', '.join(unknown)}; expected one of: {allowed}"
            )
        for key, value in config_overrides.items():
            setattr(cfg, key, value)

    runtime = RuntimeV5.create(
        capabilities=capabilities,
        workspace=Path(workspace),
        budget=budget,
        config=cfg,
        propose_initial=propose_initial,
        interaction_fn=interaction_fn,
    )
    if tracer is not None:
        tracer.attach_event_bus(runtime.event_bus)
    return CcxRuntimeBundle(
        runtime=runtime,
        llm=llm,
        plan_runner=plan_runner,
        spec_runner=spec_runner,
        agent_runner=agent_runner,
        research_runner=research_runner,
        doc_runner=doc_runner,
        ask_runner=ask_runner,
        blueprint_runner=blueprint_runner,
        sgar_runner=sgar_runner,
        sgarx_runner=sgarx_runner,
        outline_cache=outline_cache,
        findings_collector=findings_collector,
        content_store=content_store,
    )


def root_node_for(
    *,
    goal: str,
    mode: str = "plan",
    metadata: Mapping[str, Any] | None = None,
) -> NodeSpec:
    """Build the root NodeSpec a CodeAgent run will start from."""
    tool = CCX_MODE_TOOL_MAP.get(mode)
    if tool is None:
        raise ValueError(f"unknown ccx mode: {mode!r}")
    meta = {"ccx_mode": mode, "ccx_root": True, **dict(metadata or {})}
    return NodeSpec(
        node_id=f"root-{mode}",
        tool=tool,
        params={"goal": goal, "metadata": dict(meta)},
        metadata=meta,
    )


__all__ = [
    "CCX_MODE_TOOL_MAP",
    "CcxRuntimeBundle",
    "build_runtime",
    "root_node_for",
]
