"""ccx CodeAgent — drop-in replacement for ``core.cc.api.CodeAgent``.

Same constructor signature, same ``AgentRunRequest`` / ``AgentRunResult``
types, same ``run`` / ``run_sync`` / ``stream`` methods. Internally the run
is driven by a deepstack_v5 EngineV5 with ccx plan/spec/agent tools,
giving:

* parallel sibling subagents (v5 DAG with independent siblings)
* recursive subagents (v5 SpawnResult from agent mode)
* persistent intent / lineage (v5 SQLite DB + ClaimStore)
* DAG ordering & dependencies (NodeSpec.depends_on)

The cc-side entrypoint that does single-turn LLM-with-tools chat is *not*
replaced — for ``agent_mode in ("plan", "spec")`` ccx drives v5; for
unrecognised modes ``run`` / ``run_sync`` return a ``failed=True`` result
carrying ``error_code="CCX_UNSUPPORTED_MODE"`` (drop-in parity with cc's
"never raise out of run, always return a result" contract) so callers can
detect the gap and fall back to ``core.cc.api.CodeAgent`` themselves. The
streaming entrypoint ``stream`` still raises ``NotImplementedError`` for an
unrecognised mode, matching cc's own streaming contract (which also raises).
(Future work: full single-turn parity.)

``agent_mode="structured"`` is a TEXT-ONLY pipeline: no phase reads files
or runs commands, and "execution" dispatches each task to a tool-less LLM
call (see ``structured_flow.py``). It therefore returns an ANALYSIS /
PROPOSAL document, never a record of applied changes. ``run`` surfaces
this at the API boundary so a caller does not mistake ``success=True`` for
"the task was executed": the result's ``session_snapshot["execution"]`` is
stamped ``"text_only_no_tools"`` and a ``logger.warning`` is emitted on
every structured run. Callers that need real tool-backed edits should use
``agent_mode in ("plan", "spec", "agent")`` with
``agent_runner_kind="cc_query_loop"`` (the default ``"auto"`` already
selects it when cc is importable).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
import tempfile
from typing import Any, AsyncIterator


logger = logging.getLogger(__name__)
_BLOCKING_CANCEL_GRACE_S = 5.0
_CANCEL_CLEANUP_TIMEOUT_S = 5.0
_STREAM_TEARDOWN_GRACE_S = 30.0


def _retrieve_task_exception(task: asyncio.Task[Any]) -> None:
    try:
        task.exception()
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.debug("ccx: background task ended after cleanup", exc_info=True)


# Re-export cc's request/result/event types so callers can import either
# from core.ccx.api or core.cc.api interchangeably.
from core.cc.api import AgentRunRequest, AgentRunResult, CodeBuildRequest
from core.cc.config import CCConfig
from core.cc.conversation.models import SessionEvent, SessionMessage
from core.cc.llm import DefaultLLMClientProvider, LLMClientProvider

from core.deepstack_v5 import Budget, NodeState, RunStatus
from core.deepstack_v5.memory import (
    RESUME_PREVIOUS_RUN_METADATA_KEY,
    ResumeContext,
    install_resume_metadata,
)

from .modes import LLMCallable, from_provider
from .memory import (
    JsonlMemoryStore,
    MemoryOptions,
    install_memory_metadata,
    memory_disabled,
    normalize_tags,
    render_memory_block,
    request_memory_tags,
    select_entries,
    summarize_run,
)
from .runtime import (
    RESUME_EVENT_METADATA_KEY,
    build_runtime,
    CcxRuntimeBundle,
    root_node_for,
)
from .services import SteerInbox
from .agents.governed_run import (
    CCX_RUN_CONTRACT_METADATA_KEY,
    parse_run_audit_contract,
    run_run_audit_loop,
)
from .agents.governed_goal import (
    CCX_GOAL_REQUEST_METADATA_KEY,
    CCX_GOAL_VERDICT_SNAPSHOT_KEY,
    _DEFAULT_GOAL_MAX_ITERS as _DEFAULT_GOAL_MAX_ITERS_API,
    run_goal_loop,
)
from .agents.governed_spawn import ContractError
from .services.governance_events import (
    emission_enabled as _governance_emission_enabled,
    emit_governance_verdict,
)


# Default mode for ccx CodeAgent when caller doesn't specify one. We
# intentionally pick "plan" so the v5 DAG demonstrates its full
# decomposition path; callers can pass agent_mode="agent" for a flat run.
DEFAULT_AGENT_MODE = "plan"
MEMORY_RECALL_MODES = {"plan", "spec", "agent", "doc", "ask"}

# Agent modes the v5-backed run path implements directly. Anything outside
# this set is not yet ported, so callers fall back to ``core.cc.api.CodeAgent``.
# Single source of truth for the ``_run_streaming`` / ``_run_blocking`` accept
# checks so the two can't drift. NOTE: deliberately distinct from
# ``MEMORY_RECALL_MODES`` (a subset gating memory recall) and from
# ``ccx_tool._ALL_MODES`` (which also lists the internal "research" spawn
# mode) — do not merge them.
SUPPORTED_AGENT_MODES = frozenset(
    {"plan", "spec", "agent", "doc", "ask", "blueprint", "sgar", "sgarx", "goal",
     "debug"}
)

#: Operator params for ``agent_mode="debug"`` (route / max_iters / heavy_stimulus
#: / docs_output_path), read from ``request.metadata``. Distinct from the goal
#: key so a caller can carry both; the producer/DAG never writes it.
CCX_DEBUG_REQUEST_METADATA_KEY = "ccx_debug"


def _debug_advisories(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """Anti-theater advisories over a finished debug result — INFORM-only.

    Reuses the existing detectors (``llm_monitor`` performative-completion +
    ``watch`` degraded-completion) against the result snapshot. These never
    decide the verdict — the ``[check:]``-driven ``goal_verdict`` is the floor;
    they only surface "node view reads clean but the run-level truth isn't" so an
    operator inspecting a debug run isn't misled. Returns a (possibly empty) list.
    """
    from .llm_monitor import _heuristic_performative_completion
    from .watch import degraded_completion

    snap = snapshot or {}
    counts = {
        "succeeded": int(snap.get("succeeded", 0) or 0),
        "failed": int(snap.get("failed", 0) or 0),
        "abandoned": int(snap.get("abandoned", 0) or 0),
    }
    advisories: list[dict[str, Any]] = []

    goal_verdict = snap.get(CCX_GOAL_VERDICT_SNAPSHOT_KEY)
    governance_verdict: dict[str, Any] | None = None
    if isinstance(goal_verdict, dict):
        # In debug/goal the goal_verdict IS the run-level verdict.
        governance_verdict = {
            "passed": goal_verdict.get("passed"),
            "goal_verdict": goal_verdict,
            "run_audit_verdict": snap.get("run_audit_verdict"),
            "contract_verdict": snap.get("contract_verdict"),
        }
    perf = _heuristic_performative_completion(counts, governance_verdict)
    if perf:
        advisories.append(perf)

    degraded = degraded_completion(
        snap.get("status"),
        {"abandoned": counts["abandoned"], "failed": counts["failed"]},
    )
    if degraded:
        advisories.append(
            {"rule": "degraded_completion", "severity": "warn", "detail": degraded}
        )
    return advisories

# Modes whose whole-DAG result is auto-wrapped in the code-task definition-of-done
# audit (a run-level verify gate) when ``CCX_CODE_TASK_AUDIT`` is enabled. These
# are the "ungated aggregation" modes (plan/spec) plus plain agent — exactly the
# ones with no acceptance gate today. doc/ask/blueprint aren't code tasks; sgar /
# sgarx inject the same criterion into their own close gate; goal injects it into
# its verification spec.
_CODE_TASK_AUDIT_RUN_MODES = frozenset({"plan", "spec", "agent"})


@dataclass(slots=True)
class ContentStoreOptions:
    db_path: Path | str | None = None
    retain_for_ms: int = 7 * 24 * 60 * 60 * 1000
    purge_on_start: bool = True


def _merge_root_metadata(root: Any, updates: dict[str, Any]) -> None:
    base_meta: dict[str, Any] = dict(root.metadata) if root.metadata else {}
    base_meta.update(updates)
    root.metadata = base_meta
    params = dict(root.params) if root.params else {}
    inner = dict(params.get("metadata") or {})
    inner.update(updates)
    params["metadata"] = inner
    root.params = params


def _maybe_inject_resume(
    *,
    root: Any,
    bundle: CcxRuntimeBundle,
    request: AgentRunRequest,
) -> None:
    """If ``request.metadata`` names a previous run, stamp a resume
    block onto the root node so the LLM sees prior-run context.

    Mutates ``root.metadata`` and ``root.params["metadata"]`` in place.
    The two must stay in sync because v5's ``_make_mode_tool`` invokes
    the mode runner with ``fn(*, goal, metadata=...)``, reading the
    ``metadata`` key from ``NodeSpec.params`` (not ``NodeSpec.metadata``
    directly). ``root_node_for`` deep-copies the original metadata into
    params at construction, so updating one without the other would
    silently leave the runner blind.

    Stamps a pending resume event onto root metadata so the root node can
    emit it on the *new* run once a run_id exists. If no previous run is
    supplied, returns silently.
    """
    request_meta = request.metadata or {}
    previous_run_id = request_meta.get(RESUME_PREVIOUS_RUN_METADATA_KEY)
    if not previous_run_id:
        return
    ctx = ResumeContext.from_event_store(
        bundle.runtime.event_store, str(previous_run_id),
    )
    # Skip stamping (and the bus event) for an empty snapshot — no
    # point telling the LLM "your prior run did nothing" when we could
    # just behave like a fresh run.
    if ctx.snapshot.is_empty:
        _merge_root_metadata(root, {
            RESUME_EVENT_METADATA_KEY: {
                "kind": "ccx.resume.skipped",
                "payload": {
                    "previous_run_id": str(previous_run_id),
                    "reason": "empty_snapshot",
                },
            },
        })
        return
    base_meta: dict[str, Any] = dict(root.metadata) if root.metadata else {}
    new_meta = install_resume_metadata(base_meta, ctx)
    new_meta[RESUME_EVENT_METADATA_KEY] = {
        "kind": "ccx.resume.injected",
        "payload": {
            "previous_run_id": str(previous_run_id),
            "snapshot_events": len(ctx.snapshot.events),
            "highwater_sequence": ctx.snapshot.highwater_sequence,
        },
    }
    _merge_root_metadata(root, new_meta)


def _memory_root(*, request: AgentRunRequest, options: MemoryOptions) -> Path:
    if options.root is not None:
        return Path(options.root)
    try:
        cwd = Path(request.cwd or ".").resolve()
    except (OSError, ValueError):
        cwd = Path(request.cwd or ".")
    return cwd / ".ccx" / "memory"


def _maybe_inject_memory(
    *,
    root: Any,
    request: AgentRunRequest,
    mode: str,
    options: MemoryOptions | None,
) -> None:
    if options is None or not options.enabled or not options.auto_recall:
        return
    if memory_disabled(request.metadata):
        return
    if mode not in MEMORY_RECALL_MODES:
        return
    try:
        store = JsonlMemoryStore(_memory_root(request=request, options=options))
        entries, _skipped = store.load()
        if not entries:
            return
        query_tags = set(normalize_tags((
            *options.tags,
            *request_memory_tags(request.metadata),
        )))
        selected = select_entries(
            entries,
            goal=request.instruction,
            query_tags=query_tags,
            now=datetime.now().astimezone(),
            options=options,
        )
        block = render_memory_block(selected)
        if not block:
            return
        base_meta: dict[str, Any] = (
            dict(root.metadata) if root.metadata else {}
        )
        new_meta = install_memory_metadata(base_meta, block)
        root.metadata = new_meta
        params = dict(root.params) if root.params else {}
        inner = dict(params.get("metadata") or {})
        inner.update(new_meta)
        params["metadata"] = inner
        root.params = params
        logger.info(
            "ccx memory: injected %d entries, %d chars",
            len(selected), len(block),
        )
    except Exception:
        logger.warning("ccx memory: injection failed", exc_info=True)


def _node_timeout_from_config(config: CCConfig | None) -> float | None:
    """Pick a per-tool timeout for v5 ToolSpecs from CCConfig.

    Mirrors ``cc_config.max_turn_timeout_seconds`` so a single config
    value bounds both the cc QueryEngine wall-clock guard and the v5
    dispatcher's per-tool ``_call_with_timeout`` deadline. Returning
    ``None`` keeps the legacy "block forever" behaviour (used when a
    caller has explicitly disabled the cc-side timeout).
    """
    if config is None:
        return None
    raw = getattr(config, "max_turn_timeout_seconds", None)
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return value


def _resolve_policy_knob(
    config: CCConfig | None,
    attr: str,
    env_name: str,
    coerce: Any,
) -> Any:
    """Resolve an operator policy knob: CCConfig field primary, env override.

    The ``CCConfig`` field (``attr``) is the primary value; a ``CCX_*`` env
    var (``env_name``) overrides it when set and parseable, so an operator can
    tune a single run without editing config. Returns ``None`` when neither is
    set — leaving the caller's default (byte-identical) path intact. A
    malformed env value is ignored (falls back to the config value), never
    raising into the run path.
    """
    value = getattr(config, attr, None) if config is not None else None
    raw = os.environ.get(env_name)
    if raw is not None and raw.strip():
        try:
            value = coerce(raw.strip())
        except (TypeError, ValueError):
            pass
    return value


def _budget_from_config(config: CCConfig | None) -> Budget | None:
    """Build a run Budget from CCConfig + ``CCX_*`` env, or ``None``.

    Returns ``None`` — keeping ``build_runtime``'s default unbounded budget,
    byte-identical — when no cap or price is configured. When a cost cap,
    token cap, or token→cost price is set, returns a ``Budget`` carrying them
    so a configured cap actually halts the run; the price additionally makes a
    cost cap bite for token-only reasoning clients (see
    ``BudgetTracker.consume``).
    """
    max_cost = _resolve_policy_knob(
        config, "ccx_max_cost_usd", "CCX_MAX_COST_USD", float,
    )
    max_tokens = _resolve_policy_knob(
        config, "ccx_max_tokens", "CCX_MAX_TOKENS", int,
    )
    price = _resolve_policy_knob(
        config, "ccx_cost_per_1k_tokens", "CCX_COST_PER_1K_TOKENS", float,
    )
    if max_cost is None and max_tokens is None and price is None:
        return None
    return Budget(
        max_cost=max_cost,
        max_tokens=max_tokens,
        cost_per_1k_tokens=price,
    )


def _env_truthy(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _spawn_policy_from_config(config: CCConfig | None) -> dict[str, Any]:
    """Operator fan-out caps as ``build_runtime`` kwargs (CCConfig primary,
    ``CCX_*`` env override).

    ``None`` fanout/depth keep the runner's defaults (32 / 3);
    ``cc_count_research_in_fanout`` defaults False so ccx_research stays
    exempt — all byte-identical when nothing is configured.
    """
    return {
        "cc_max_spawn_fanout": _resolve_policy_knob(
            config, "ccx_max_spawn_fanout", "CCX_MAX_SPAWN_FANOUT", int,
        ),
        "cc_max_spawn_depth": _resolve_policy_knob(
            config, "ccx_max_spawn_depth", "CCX_MAX_SPAWN_DEPTH", int,
        ),
        "cc_count_research_in_fanout": bool(
            _resolve_policy_knob(
                config, "ccx_count_research_in_fanout",
                "CCX_COUNT_RESEARCH_FANOUT", _env_truthy,
            )
        ),
    }


def _create_content_store(
    options: ContentStoreOptions | None,
    *,
    cwd: str,
) -> Any | None:
    if options is None:
        return None
    try:
        from core.deepstack_v5.memory import ContentStore, default_db_path

        db_path = Path(options.db_path) if options.db_path else default_db_path(cwd)
        store = ContentStore(db_path=db_path)
        if options.purge_on_start:
            store.purge_expired()
        return store
    except Exception:
        logger.warning(
            "ccx content store unavailable; continuing without it",
            exc_info=True,
        )
        return None


def _close_content_store_on_setup_failure(content_store: Any | None) -> None:
    if content_store is None:
        return
    try:
        content_store.close()
    except Exception:
        logger.debug(
            "ccx content store close failed during runtime setup cleanup",
            exc_info=True,
        )


def _translate_runtime_setup_error(
    exc: BaseException,
    *,
    workspace: Path,
) -> BaseException:
    message = str(exc)
    lowered = message.lower()
    if "schema version" in lowered and "newer" in lowered:
        db_path = Path(workspace) / "runtime.db"
        return RuntimeError(
            "CCX_RUNTIME_SCHEMA_NEWER: the ccx runtime DB schema is newer "
            f"than this checkout supports: {db_path}. Upgrade this checkout "
            "to the version that created the DB, or move/delete that runtime "
            "DB to let ccx rebuild it for this checkout."
        )
    return exc


def _session_event_from_v5(
    event: dict[str, Any],
    *,
    turn_id: str,
) -> SessionEvent | None:
    kind = event.get("kind", "")
    if not kind.startswith("node."):
        return None
    payload = event.get("payload") or {}
    return SessionEvent(
        event_type=f"ccx.{kind}",
        turn_id=turn_id,
        payload={
            "run_id": event.get("run_id"),
            "node_id": payload.get("node_id"),
            "spawned": payload.get("spawned") or [],
            "result_summary": payload.get("result_summary"),
            "kind_message": payload.get("message"),
            "sequence": event.get("sequence"),
        },
    )


class CodeAgent:
    """Same surface as ``core.cc.api.CodeAgent``, v5-backed.

    Limitation in this milestone: ``stream`` returns events *after* the run
    completes (not streaming during execution). True streaming requires
    routing v5 EventBus through an asyncio queue — straightforward but
    out of scope for the first iteration.
    """

    def __init__(
        self,
        *,
        config: CCConfig | None = None,
        llm_client_provider: LLMClientProvider | None = None,
        llm: LLMCallable | None = None,
        llm_routes: dict[str, LLMCallable] | None = None,
        preferred_model_overrides: dict[str, object] | None = None,
        agent_runner_kind: str = "auto",
        max_tool_rounds: int | None = None,
        memory: MemoryOptions | None = None,
        content_store: ContentStoreOptions | None = None,
        sgar_run_criterion_checks: bool = False,
        sgar_criterion_check_timeout_s: float = 120.0,
        run_audit_contract: dict[str, Any] | None = None,
        run_audit_check_timeout_s: float = 120.0,
        goal_llm_timeout_s: float = 600.0,
        goal_check_timeout_s: float = 120.0,
    ) -> None:
        self.config = config
        self.llm_client_provider = llm_client_provider or (
            DefaultLLMClientProvider() if llm is None else None
        )
        # Optional direct LLMCallable override (preferred for tests so we
        # don't have to fake the LLMClientProvider protocol).
        self._llm_callable = llm
        # R1: optional per-invocation LLM routing. When set, a child
        # node whose ``invocation.preferred_model`` matches a key in
        # this dict runs against that LLMCallable instead of the
        # default. ``None`` (default) preserves pre-R1 behaviour.
        self._llm_routes = llm_routes
        # Optional provider-bound alias -> model mapping. Unlike per-run
        # metadata, this is caller-owned configuration and is safe to use
        # when forwarding overrides into a production LLMClientProvider.
        self._preferred_model_overrides = preferred_model_overrides
        # Three valid values:
        # * 'auto'           — pick at run time: cc_query_loop when cc is
        #                      importable, else lite. This is the default
        #                      because the lite AgentModeRunner has no tool
        #                      access (file_write/shell/grep are out of
        #                      scope) so spawned agent nodes under lite
        #                      can return text but never produce real
        #                      artifacts. Most production callers want
        #                      cc_query_loop; tests that pass a stub
        #                      LLMCallable should opt into 'lite'.
        # * 'lite'           — each agent node = one LLM call, no tools.
        #                      Used by tests with stub LLMs.
        # * 'cc_query_loop'  — each agent node drives the full cc
        #                      QueryEngine (multi-tool-round turn with cc's
        #                      tool registry).
        self._agent_runner_kind = agent_runner_kind
        # Per cc-agent-turn tool-round budget for cc_query_loop nodes. Default
        # ``None`` means the ROUND COUNT is unbounded — a single agent turn may
        # loop LLM↔tool indefinitely so long as it keeps making progress. It is
        # bounded only by (a) the wall-clock guard
        # ``CCConfig.max_turn_timeout_seconds`` (default 7200s, also drives the
        # v5 per-tool ``node_timeout_s``) and (b) cc's read-only stall guard,
        # which aborts after ``_MAX_GENERIC_READ_ONLY_STALL_ROUNDS`` (30; 60 if
        # todos remain) consecutive rounds with no successful write/progress
        # tool — a successful write resets that counter, so a long but
        # genuinely-progressing turn is NOT capped by round count. We keep the
        # default unbounded (cc-parity, no truncated work) and leave bounding
        # opt-in: pass ``max_tool_rounds=N`` here, per request via
        # ``AgentRunRequest.max_tool_rounds`` (request wins), or set
        # ``cc_max_tool_rounds`` in config to add a hard round ceiling.
        self._max_tool_rounds = max_tool_rounds
        self._memory = memory
        self._content_store = content_store
        # P2: opt-in machine-checkable exit criteria. Operator config — when
        # True, SGAR verify/close run any ``[check: ...]`` declared on a
        # criterion and refuse a pass the command contradicts. Default OFF
        # preserves "trust the agent's self-report" for every existing spec.
        self._sgar_run_criterion_checks = sgar_run_criterion_checks
        self._sgar_criterion_check_timeout_s = sgar_criterion_check_timeout_s
        # Run-level externalized hard audit (default OFF). When a contract is
        # present (here, or per-request via ``metadata["ccx_run_contract"]``),
        # ``run`` wraps the whole-DAG drive in a bounded verify-repair loop
        # gated on independent ``[check:]`` commands run against the workspace
        # after the run quiesces (see agents/governed_run.py). Same JSON shape
        # as ``ccx_contract`` but a distinct, non-inheritable key. Default off +
        # no contract ⇒ the DAG is driven once and the only snapshot change is a
        # ``run_audit_verdict=None`` key.
        self._run_audit_contract = run_audit_contract
        self._run_audit_check_timeout_s = run_audit_check_timeout_s
        # Goal mode (agent_mode="goal", default-OFF unless selected). Per-meta-
        # LLM-call (planner / judge / replanner / reporter) wall clock, enforced
        # via a daemon-thread join inside the goal loop so a hung reasoning
        # client can't make the loop un-killable. Not engaged for any other mode.
        self._goal_llm_timeout_s = goal_llm_timeout_s
        # Per-``[check:]`` command wall clock for goal-mode verification. Kept
        # distinct from ``run_audit_check_timeout_s`` so goal and run-audit can
        # be tuned independently; the CLI wires its ``check_timeout`` flag here.
        self._goal_check_timeout_s = goal_check_timeout_s
        # Mid-turn steer routing (R2). ``_steer_inbox`` remains the
        # pre-run/default queue for backwards compatibility; active runs
        # get their own inbox registered once the v5 run_id is known.
        self._steer_inbox = SteerInbox()
        self._steer_lock = threading.Lock()
        self._active_steer_inboxes: dict[str, SteerInbox] = {}

    # -- public API ---------------------------------------------------------

    def push_steer(self, text: str, *, run_id: str | None = None) -> None:
        """Inject mid-turn supplemental guidance for this agent.

        Thread-safe. The next subagent node v5 dispatches drains and
        prepends the wrapped text to its goal so the LLM sees a clear
        "this is an additional constraint, not a new task" marker.
        Empty / whitespace-only strings are dropped silently. Texts
        over ``MAX_STEER_BODY_BYTES`` are truncated with a warning.

        Has no effect on ``agent_mode="structured"`` runs — that flow
        is out-of-band and does not pass through ``_make_mode_tool``.
        Pushing a steer before / during a structured run emits a
        ``UserWarning`` so the caller knows the steer was ignored.
        """
        with self._steer_lock:
            if run_id is not None:
                inbox = self._active_steer_inboxes.get(str(run_id))
                if inbox is None:
                    raise ValueError(
                        f"CCX_STEER_TARGET_NOT_ACTIVE: run_id={run_id!r} "
                        "is not an active ccx run"
                    )
                inbox.push(text)
                return
            active = list(self._active_steer_inboxes.values())
            if len(active) == 1:
                active[0].push(text)
                return
            if len(active) > 1:
                raise ValueError(
                    "CCX_STEER_TARGET_REQUIRED: multiple ccx runs are "
                    "active; pass run_id=..."
                )
            self._steer_inbox.push(text)

    def _new_run_steer_inbox(self) -> SteerInbox:
        inbox = SteerInbox()
        with self._steer_lock:
            for item in self._steer_inbox.drain():
                inbox.push(item)
        return inbox

    def _register_steer_run(
        self,
        run_id: str | None,
        inbox: SteerInbox,
    ) -> None:
        if not run_id:
            return
        with self._steer_lock:
            if str(run_id) not in self._active_steer_inboxes:
                self._active_steer_inboxes[str(run_id)] = inbox
                for item in self._steer_inbox.drain():
                    inbox.push(item)

    def _unregister_steer_run(
        self,
        run_id: str | None,
        inbox: SteerInbox,
    ) -> None:
        with self._steer_lock:
            if run_id is not None:
                active = self._active_steer_inboxes.get(str(run_id))
                if active is inbox:
                    self._active_steer_inboxes.pop(str(run_id), None)
            inbox.clear()

    def _failed_result(
        self,
        request: AgentRunRequest,
        *,
        error_code: str,
        error_message: str,
    ) -> AgentRunResult:
        """Build a drop-in failure result, mirroring ``cc.api.CodeAgent.run``.

        cc never raises out of ``run`` for an operational failure — it
        returns ``AgentRunResult(failed=True, error_code=..., error_message=...)``
        with empty session fields. ccx matches that shape so a caller can
        swap ``cc.CodeAgent`` → ``ccx.CodeAgent`` and rely on the same
        ``.failed`` / ``.error_code`` contract instead of having to catch
        exceptions on some paths and inspect a result on others.
        """
        try:
            resolved_cwd = str(Path(request.cwd or ".").resolve())
        except (OSError, ValueError):
            resolved_cwd = request.cwd or ""
        return AgentRunResult(
            final_text="",
            session_id="",
            turn_id=None,
            cwd=resolved_cwd,
            session_snapshot={},
            failed=True,
            error_code=error_code,
            error_message=error_message,
        )

    async def run(self, request: AgentRunRequest) -> AgentRunResult:
        # Structured mode → delegate to StructuredFlowRunner (4-phase
        # analyze → plan → execute (v5 parallel) → summarize).
        config = self._resolve_config(request)
        mode = self._resolve_mode(request, config)
        if mode == "structured":
            # Run-level audit does not cover the structured path: that flow
            # runs its own SwarmCoordinator out-of-band and never reaches the
            # run-contract parse below, so a contract supplied here would be
            # silently dropped. Warn (non-breaking) instead of staying mute so
            # a caller who attached one isn't misled into thinking the run was
            # audited. Detection mirrors parse_run_audit_contract's source
            # resolution (request metadata key OR the ctor default) but never
            # raises on a malformed contract — structured ignores it anyway.
            has_run_contract = (
                isinstance(request.metadata, dict)
                and request.metadata.get(CCX_RUN_CONTRACT_METADATA_KEY) is not None
            ) or self._run_audit_contract is not None
            if has_run_contract:
                logger.warning(
                    "ccx run-level audit contract is ignored for "
                    "agent_mode='structured': the structured flow runs a "
                    "separate out-of-band pipeline and is not wrapped in the "
                    "verify-repair loop. Use agent_mode plan/spec/agent for a "
                    "run-audited execution.",
                )
            return await self._run_structured(request, config)
        # Unknown mode: drop-in parity with cc — return a failed result the
        # caller can branch on (and fall back to cc with) rather than raising.
        # A distinct error_code keeps "this mode isn't ported" detectable.
        # (``stream`` still raises NotImplementedError for unknown modes,
        # matching cc's streaming contract, which also raises.)
        if mode not in SUPPORTED_AGENT_MODES:
            return self._failed_result(
                request,
                error_code="CCX_UNSUPPORTED_MODE",
                error_message=(
                    f"ccx.CodeAgent does not implement agent_mode={mode!r}; "
                    f"use core.cc.api.CodeAgent for that mode"
                ),
            )

        # Goal mode → the decompose → execute-DAG → verify → iterate-until-met
        # orchestration loop. Intercepted here (before the run-audit parse): goal
        # mode runs its OWN independent verifier and re-drives the whole run per
        # iteration via ``_drive_run_once`` (driving plan mode under the hood), so
        # it never becomes a v5 root tool itself. A separately-supplied run-audit
        # contract is therefore ignored — warn so the caller isn't misled.
        if mode == "goal":
            result = await self._run_goal_loop(request, config)
            await self._maybe_emit_governance_verdict(request, result)
            return result

        # Debug mode → the goal loop re-parameterized for monitored adversarial
        # debugging. Same orchestration (decompose → drive → verify → iterate),
        # same authoritative [check:] floor; ``_run_debug_loop`` adds, only when
        # ``CCX_DEBUG_MODE`` is on, a run-once heavy-stimulus confirmation and
        # consumption of the anti-theater advisories. ``"debug"`` is permanently
        # in SUPPORTED_AGENT_MODES so a flag-off debug request degrades to a
        # plain goal loop here instead of silently falling through to cc.
        if mode == "debug":
            result = await self._run_debug_loop(request, config)
            await self._maybe_emit_governance_verdict(request, result)
            return result

        # Code-task definition-of-done auto-injection (default OFF). When
        # ``CCX_CODE_TASK_AUDIT`` is enabled and the caller supplied no run-audit
        # contract of their own, synthesize one whose single ``[check:]`` is the
        # code-task audit (wiring + scoped tests green). This gives the otherwise
        # ungated plan/spec aggregation — and plain agent — a real whole-DAG
        # acceptance gate. Pinned to ``max_iters=1`` (gate-once: verify after the
        # DAG, never auto re-drive the non-idempotent run). An explicit
        # caller-supplied contract always wins. Flag unset ⇒ this whole block is
        # inert and ``request`` is untouched (byte-equivalent).
        if mode in _CODE_TASK_AUDIT_RUN_MODES:
            from .audit import code_task_audit_enabled
            if code_task_audit_enabled():
                caller_has_contract = (
                    (
                        isinstance(request.metadata, dict)
                        and request.metadata.get(CCX_RUN_CONTRACT_METADATA_KEY)
                        is not None
                    )
                    or self._run_audit_contract is not None
                )
                if not caller_has_contract:
                    from .audit import build_code_task_contract
                    request = replace(
                        request,
                        metadata={
                            **(request.metadata or {}),
                            CCX_RUN_CONTRACT_METADATA_KEY: build_code_task_contract(
                                "contract", cwd=request.cwd or ".",
                            ),
                        },
                    )

        # Run-level externalized hard audit (default OFF). A parsed contract
        # wraps the whole-DAG drive in a bounded, externally-judged verify-repair
        # loop; absent (the default) the DAG is driven exactly once. A malformed
        # contract is a caller/author error — parse fails loud, and we surface it
        # as a distinctly-coded failed result (drop-in parity: ``run`` returns a
        # failed result rather than raising for bad input, exactly as it does for
        # an unsupported mode above).
        try:
            run_audit_contract = parse_run_audit_contract(
                request.metadata, self._run_audit_contract,
            )
        except ContractError as exc:
            return self._failed_result(
                request,
                error_code="CCX_RUN_CONTRACT_INVALID",
                error_message=str(exc),
            )
        if run_audit_contract is None:
            result = await self._drive_run_once(request)
        else:
            result = await self._run_with_run_audit(request, run_audit_contract)
        await self._maybe_emit_governance_verdict(request, result)
        return result

    async def _drive_run_once(self, request: AgentRunRequest) -> AgentRunResult:
        """Drive the whole v5 DAG exactly once → a terminal ``AgentRunResult``.

        This is the original ``run`` body, extracted verbatim so the run-level
        audit loop can re-invoke it per iteration. Owns the ``asyncio.shield`` +
        ``_cancel_blocking_worker`` discipline, so cancelling the awaiting caller
        reaches THIS iteration's worker/run_id (each call gets a fresh ``state``).
        Returns a failed result for an operational engine-thread failure (cc
        parity); re-raises ``CancelledError`` fail-loud.
        """
        state: dict[str, Any] = {}
        worker = asyncio.create_task(
            asyncio.to_thread(self._run_blocking, request, state)
        )
        try:
            bundle, verdict, events = await asyncio.shield(worker)
        except asyncio.CancelledError:
            await self._cancel_blocking_worker(worker, state)
            raise
        except Exception as exc:
            # Operational failure raised out of the engine thread (most
            # commonly a runtime-startup error translated by
            # ``_translate_runtime_setup_error``). ``_run_blocking`` already
            # shut its own bundle down before re-raising, so there is nothing
            # to clean up here — convert to a failed result for cc parity.
            # Cancellation is handled above and stays fail-loud; programming
            # errors in run_sync (called from a running loop) also stay
            # fail-loud since they surface before ``run`` is awaited.
            return self._failed_result(
                request,
                error_code=(
                    getattr(exc, "error_code", None)
                    or getattr(exc, "code", None)
                    or "CCX_RUN_FAILED"
                ),
                error_message=str(exc),
            )
        try:
            result = self._build_result(request, bundle, verdict, events)
        finally:
            await asyncio.to_thread(
                bundle.shutdown,
                run_id=verdict.run_id,
                content_retain_for_ms=(
                    self._content_store.retain_for_ms
                    if self._content_store is not None
                    else None
                ),
            )
        await asyncio.to_thread(self._maybe_finalize_memory, request, result)
        return result

    async def _run_with_run_audit(
        self, request: AgentRunRequest, contract: Any,
    ) -> AgentRunResult:
        """Wrap ``_drive_run_once`` in the run-level verify-repair loop.

        The whole DAG is re-driven per iteration and an independent judge
        (``run_criterion_check`` over the post-run workspace) decides pass/fail;
        the producer's ``final_text`` is never trusted. See
        ``agents/governed_run.run_run_audit_loop``.
        """
        cwd_resolved = str(Path(request.cwd or ".").resolve())
        return await run_run_audit_loop(
            self._drive_run_once,
            request,
            contract,
            cwd=cwd_resolved,
            check_timeout_s=self._run_audit_check_timeout_s,
            log=lambda message: logger.info("ccx run-audit: %s", message),
        )

    async def _run_goal_loop(
        self, request: AgentRunRequest, config: CCConfig,
    ) -> AgentRunResult:
        """Drive ``agent_mode="goal"`` through the goal orchestration loop.

        Resolves the goal-mode params from ``metadata["ccx_goal"]`` (route /
        max_iters), hands ``_drive_run_once`` to the loop as its whole-DAG drive
        (so per-iteration ``asyncio.shield`` + cancellation is preserved), and
        threads ``docs_output_path`` through for the summary report. See
        ``agents/governed_goal.run_goal_loop``.
        """
        meta = dict(request.metadata or {})
        params = meta.get(CCX_GOAL_REQUEST_METADATA_KEY)
        params = params if isinstance(params, dict) else {}
        route_override = str(params.get("route") or "auto").strip().lower()
        if route_override not in {"auto", "explicit", "plan"}:
            route_override = "auto"
        raw_max_iters = params.get("max_iters")
        try:
            max_iters = int(raw_max_iters) if raw_max_iters else _DEFAULT_GOAL_MAX_ITERS_API
        except (TypeError, ValueError):
            max_iters = _DEFAULT_GOAL_MAX_ITERS_API
        docs_output_path = str(meta.get("docs_output_path") or "").strip() or None

        # A run-audit contract is ignored under goal mode (goal runs its own
        # independent verifier). Detection mirrors parse_run_audit_contract's
        # source resolution; warn (non-breaking) so a caller isn't misled.
        has_run_contract = (
            meta.get(CCX_RUN_CONTRACT_METADATA_KEY) is not None
            or self._run_audit_contract is not None
        )
        if has_run_contract:
            logger.warning(
                "ccx run-level audit contract is ignored for agent_mode='goal': "
                "goal mode runs its own decompose→verify→iterate loop with an "
                "independent verifier. The goal_verdict carries the outcome.",
            )

        llm = self._resolve_llm(config)
        cwd_resolved = str(Path(request.cwd or ".").resolve())
        return await run_goal_loop(
            self._drive_run_once,
            request,
            llm=llm,
            language=config.prompt_language if config else "en",
            cwd=cwd_resolved,
            check_timeout_s=self._goal_check_timeout_s,
            llm_timeout_s=self._goal_llm_timeout_s,
            route_override=route_override,
            max_iters=max_iters,
            docs_output_path=docs_output_path,
            log=lambda message: logger.info("ccx goal: %s", message),
        )

    async def _run_debug_loop(
        self, request: AgentRunRequest, config: CCConfig,
    ) -> AgentRunResult:
        """Drive ``agent_mode="debug"`` — the goal loop re-parameterized for
        monitored adversarial debugging.

        The loop machinery is REUSED verbatim (``run_goal_loop``): the cheap
        repro is the loop's per-iteration ``[check:]`` (re-driven each round, the
        authoritative exit-code floor); the adversarial judge manages residual;
        ``max_iters`` is the same hard-clamped bound; LLM calls keep the
        daemon-thread timeout. We do NOT write a new loop.

        When ``CCX_DEBUG_MODE`` is on we add two things AROUND the loop, never
        inside it:

        * a HEAVY re-stimulus (e.g. a soak / big run) supplied by the operator
          in ``metadata['ccx_debug']['heavy_stimulus']`` — run **ONCE** after the
          loop converges (a post-fix confirmation gate), never re-driven per
          iteration. It is a ``[check:]`` too, so a genuine FAIL **downgrades**
          the verdict (never upgrades — the bar can't move to pass); an
          *unrunnable* heavy check is surfaced as an advisory, not a verdict.
        * consumption of the anti-theater advisories (``llm_monitor`` performative
          completion + ``watch`` degraded completion) — INFORM-only, attached to
          the snapshot; they never override the ``[check:]`` floor.

        Flag-off ⇒ this is byte-equivalent to a plain goal run.
        """
        from dataclasses import replace as _replace

        from .audit import debug_mode_enabled
        from .audit.check_template import validate_check_command  # lazy, flag-cheap

        meta = dict(request.metadata or {})
        params = meta.get(CCX_DEBUG_REQUEST_METADATA_KEY)
        params = params if isinstance(params, dict) else {}
        route_override = str(params.get("route") or "auto").strip().lower()
        if route_override not in {"auto", "explicit", "plan"}:
            route_override = "auto"
        raw_max_iters = params.get("max_iters")
        try:
            max_iters = int(raw_max_iters) if raw_max_iters else _DEFAULT_GOAL_MAX_ITERS_API
        except (TypeError, ValueError):
            max_iters = _DEFAULT_GOAL_MAX_ITERS_API
        docs_output_path = str(meta.get("docs_output_path") or "").strip() or None

        llm = self._resolve_llm(config)
        cwd_resolved = str(Path(request.cwd or ".").resolve())
        result = await run_goal_loop(
            self._drive_run_once,
            request,
            llm=llm,
            language=config.prompt_language if config else "en",
            cwd=cwd_resolved,
            check_timeout_s=self._goal_check_timeout_s,
            llm_timeout_s=self._goal_llm_timeout_s,
            route_override=route_override,
            max_iters=max_iters,
            docs_output_path=docs_output_path,
            log=lambda message: logger.info("ccx debug: %s", message),
        )

        # Flag-off: debug ≡ goal, byte-equivalent.
        if not debug_mode_enabled():
            return result

        snapshot = dict(result.session_snapshot or {})

        # --- run-once HEAVY re-stimulus (cached; never re-driven per iter) ---
        heavy_cmd = str(params.get("heavy_stimulus") or "").strip()
        if heavy_cmd:
            try:
                validate_check_command(heavy_cmd)
                from .sgar.checks import (
                    check_unrunnable as _unrunnable,
                    run_criterion_check as _run_check,
                )
                from .sgar.models import ExitCriterion as _Crit

                outcome = await asyncio.to_thread(
                    _run_check,
                    _Crit(
                        criterion_id="debug_heavy_stimulus",
                        description="run-once post-fix heavy confirmation",
                        blocking=True,
                        check=heavy_cmd,
                    ),
                    cwd=cwd_resolved,
                    timeout_s=self._goal_check_timeout_s,
                )
                snapshot["debug_heavy_stimulus"] = {
                    "command": heavy_cmd,
                    "passed": bool(outcome.passed),
                    "unrunnable": bool(_unrunnable(outcome)),
                    "evidence": outcome.evidence_line(),
                }
                if not outcome.passed and not _unrunnable(outcome):
                    # A real heavy [check:] failure DOWNGRADES the verdict (never
                    # upgrades) — the exit code is the floor, applied once here.
                    gv = dict(snapshot.get(CCX_GOAL_VERDICT_SNAPSHOT_KEY) or {})
                    gv["passed"] = False
                    gv["heavy_stimulus_failed"] = True
                    snapshot[CCX_GOAL_VERDICT_SNAPSHOT_KEY] = gv
            except ValueError as exc:
                snapshot["debug_heavy_stimulus"] = {
                    "command": heavy_cmd, "passed": None,
                    "error": f"rejected: {exc}",
                }

        # --- anti-theater advisories (INFORM-only; never override the floor) ---
        snapshot["debug_advisories"] = _debug_advisories(snapshot)

        return _replace(result, session_snapshot=snapshot)

    async def _maybe_emit_governance_verdict(
        self, request: AgentRunRequest, result: AgentRunResult,
    ) -> None:
        """Emit one ``ccx.governance.verdict`` event over the finished run.

        Default OFF (``CCX_EMIT_GOVERNANCE_EVENTS`` unset) ⇒ returns before
        touching the DB at all, so the event stream stays byte-identical — the
        byte-equivalence anchor. When ON, this surfaces the run-level governance
        verdict (which until now lived only in the in-memory
        ``result.session_snapshot``) into the ``runtime.db`` event stream the
        operator renderers read.

        Emission happens HERE, at the outermost ``run()`` boundary, rather than
        in ``_build_result``: ``run_audit_verdict`` and ``goal_verdict`` are
        stamped by the outer loops AFTER ``_build_result`` returns (and after the
        producing bundle is shut down), so only here is the snapshot fully
        stamped. The producing bundle's event_bus is already closed by this
        point, so we reopen a short-lived bus on the run's ``runtime.db`` (a
        shared, persistent SQLite file under the workspace). Best-effort: any
        failure is swallowed and never disturbs the returned result.
        """
        if not _governance_emission_enabled():
            return
        # Defence-in-depth for Iron Law ③: the worker swallows its own errors,
        # but the ``asyncio.to_thread`` hand-off itself could still raise (e.g. a
        # rare executor / thread-creation failure). Catch it here too so an
        # emission attempt can never turn an already-computed result into a
        # crashed run. ``CancelledError`` (a ``BaseException``) is intentionally
        # NOT caught — cancellation must stay fail-loud, as elsewhere in run().
        try:
            await asyncio.to_thread(
                self._emit_governance_verdict_blocking, request, result,
            )
        except Exception:
            logger.debug(
                "ccx: governance verdict emission dispatch failed", exc_info=True,
            )

    def _emit_governance_verdict_blocking(
        self, request: AgentRunRequest, result: AgentRunResult,
    ) -> None:
        try:
            run_id = result.session_id or (
                (result.session_snapshot or {}).get("run_id")
            )
            if not run_id:
                return
            workspace = self._resolve_workspace(request)
            db_path = Path(workspace) / "runtime.db"
            if not db_path.exists():
                # The producing run wrote its events elsewhere (e.g. the rare
                # tempdir fallback in _resolve_workspace) or never persisted —
                # nothing for a renderer to attach to. Skip silently.
                return
            from core.deepstack_v5.events import EventBus
            from core.deepstack_v5.persistence.db import SQLiteRuntimeDB
            from core.deepstack_v5.persistence.stores import EventStore

            db = SQLiteRuntimeDB(db_path)
            try:
                # No outbox: this is a one-shot append to the events table the
                # renderers query; there are no in-process subscribers to replay
                # to on resume.
                bus = EventBus(db=db, event_store=EventStore(db))
                emit_governance_verdict(
                    bus, str(run_id), result.session_snapshot,
                )
            finally:
                db.close()
        except Exception:
            logger.debug(
                "ccx: governance verdict emission failed", exc_info=True,
            )

    async def _wait_for_state_run_id(
        self, state: dict[str, Any], *, timeout_s: float = 0.5,
    ) -> str | None:
        deadline = asyncio.get_running_loop().time() + timeout_s
        while True:
            run_id = state.get("run_id")
            if run_id:
                return str(run_id)
            if asyncio.get_running_loop().time() >= deadline:
                return None
            await asyncio.sleep(0.02)

    async def _await_cleanup(self, awaitable: Any) -> Any:
        task = asyncio.ensure_future(awaitable)
        was_cancelled = False
        while True:
            try:
                result = await asyncio.shield(task)
                if was_cancelled:
                    raise asyncio.CancelledError
                return result
            except asyncio.CancelledError:
                if task.done():
                    if was_cancelled:
                        try:
                            task.result()
                        except Exception:
                            logger.debug(
                                "ccx: cleanup task failed after caller "
                                "cancellation",
                                exc_info=True,
                            )
                        raise
                    return task.result()
                was_cancelled = True
                continue
            except Exception:
                if was_cancelled:
                    logger.debug(
                        "ccx: cleanup task failed after caller cancellation",
                        exc_info=True,
                    )
                    raise asyncio.CancelledError
                raise

    async def _cancel_blocking_worker(
        self,
        worker: asyncio.Task[tuple[CcxRuntimeBundle, Any, list[SessionEvent]]],
        state: dict[str, Any],
    ) -> None:
        run_id: str | None = None
        try:
            run_id = await self._wait_for_state_run_id(state)
            cancel_sent = False
            deadline = asyncio.get_running_loop().time() + _BLOCKING_CANCEL_GRACE_S
            while not worker.done() and asyncio.get_running_loop().time() < deadline:
                run_id = str(state["run_id"]) if state.get("run_id") else run_id
                engine = state.get("engine")
                if engine is not None and run_id and not cancel_sent:
                    try:
                        await self._await_cleanup(
                            asyncio.wait_for(
                                asyncio.to_thread(engine.cancel, run_id),
                                timeout=_CANCEL_CLEANUP_TIMEOUT_S,
                            )
                        )
                    except Exception:
                        logger.warning(
                            "ccx: engine.cancel failed during run cancellation",
                            exc_info=True,
                        )
                    cancel_sent = True
                remaining = max(0.0, deadline - asyncio.get_running_loop().time())
                try:
                    await self._await_cleanup(
                        asyncio.wait_for(
                            asyncio.shield(worker),
                            timeout=min(0.1, remaining),
                        )
                    )
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    logger.debug(
                        "ccx: worker cancellation observed during cleanup",
                        exc_info=True,
                    )
                    break
                except Exception:
                    logger.debug(
                        "ccx: worker ended with an exception during "
                        "cancellation cleanup",
                        exc_info=True,
                    )
                    break
            if not worker.done():
                worker.add_done_callback(_retrieve_task_exception)
                logger.warning(
                    "ccx: worker did not finish within cancellation grace period; "
                    "continuing with bounded bundle shutdown."
                )
        finally:
            run_id = str(state["run_id"]) if state.get("run_id") else run_id
            bundle = state.get("bundle")
            if bundle is not None:
                try:
                    await self._await_cleanup(asyncio.to_thread(
                        bundle.shutdown,
                        run_id=run_id,
                        content_retain_for_ms=(
                            self._content_store.retain_for_ms
                            if self._content_store is not None
                            else None
                        ),
                    ))
                except Exception:
                    logger.warning(
                        "ccx: bundle.shutdown failed during run cancellation",
                        exc_info=True,
                    )

    async def _run_structured(
        self, request: AgentRunRequest, config: CCConfig,
    ) -> AgentRunResult:
        # Steer is not supported on the structured path — that flow
        # bypasses ``_make_mode_tool`` and assembles its own per-phase
        # prompts. Warn loudly if the caller pushed any so they aren't
        # silently lost; drop them so subsequent runs aren't affected.
        # ccx persistent memory v1 also does not read or write on this
        # out-of-band structured path.
        if len(self._steer_inbox) > 0:
            import warnings as _warnings
            pending = len(self._steer_inbox)
            self._steer_inbox.clear()
            _warnings.warn(
                f"structured agent_mode does not consume mid-turn steer; "
                f"discarding {pending} pending steer item(s)",
                UserWarning,
                stacklevel=2,
            )
        from .structured_flow import StructuredFlowRunner
        from core.deepstack_v5 import new_id
        llm = self._resolve_llm(config)
        runner = StructuredFlowRunner(
            config=config,
            llm=llm,
        )
        result = await runner.run(
            request.instruction,
            cwd=request.cwd or ".",
            prompt_language=request.prompt_language,
            permission_mode=request.permission_mode,
            event_sink=request.event_sink,
        )
        try:
            resolved_cwd = str(Path(request.cwd or ".").resolve())
        except (OSError, ValueError):
            resolved_cwd = request.cwd or ""
        # Make the "nothing was executed" fact visible at the API boundary.
        # StructuredFlowRunner is text-only on EVERY path (analyze / plan /
        # execute / summarize are all tool-less LLM calls), so the marker is
        # authoritative regardless of which phase the result stopped at —
        # default it when an early-return PhaseResult didn't carry one. We
        # lift it to a top-level snapshot key (not just the nested metadata
        # copy) and warn unconditionally so a caller treating success=True as
        # "task executed" gets corrected: the output is analysis/proposals,
        # not applied changes.
        execution = str(result.metadata.get("execution") or "text_only_no_tools")
        logger.warning(
            "ccx agent_mode='structured' is analysis-only (execution=%s): no "
            "files were read or written and no commands were run. final_text "
            "holds proposed changes, not applied ones — treat success as "
            "'analysis produced', not 'task executed'. Use agent_mode "
            "plan/spec/agent for tool-backed execution.",
            execution,
        )
        # Synthesize a run-* session_id so callers can uniformly check
        # ``result.session_id.startswith("run-")`` across plan/spec/agent
        # /structured modes. Structured flow doesn't run a single v5
        # engine pass (it runs a SwarmCoordinator inside); we mint a
        # surrogate id here.
        return AgentRunResult(
            final_text=result.output,
            session_id=new_id("run"),
            turn_id=None,
            cwd=resolved_cwd,
            session_snapshot={
                "structured_flow": True,
                "phase": result.phase,
                "execution": execution,
                "metadata": dict(result.metadata),
            },
            failed=not result.success,
            error_code="SF1001" if not result.success else None,
            error_message=result.error,
        )

    def run_sync(self, request: AgentRunRequest) -> AgentRunResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run(request))
        raise RuntimeError(
            "CodeAgent.run_sync() cannot be called from a running event loop; "
            "use await CodeAgent.run(...) instead."
        )

    async def build_code(self, request: CodeBuildRequest) -> AgentRunResult:
        """ccx equivalent of ``cc.api.CodeAgent.build_code``.

        The CodeBuildRequest fields (goal / context_paths / constraints /
        acceptance_criteria) are serialised into a structured instruction
        string and forwarded to ``run()``. Mode (plan/spec/agent) is
        preserved end-to-end so the v5 DAG decomposes the build the same
        way it would for an ad-hoc instruction.
        """
        run_request = self._build_request_from_code_build(request)
        return await self.run(run_request)

    def build_code_sync(self, request: CodeBuildRequest) -> AgentRunResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.build_code(request))
        raise RuntimeError(
            "CodeAgent.build_code_sync() cannot be called from a running "
            "event loop; use await CodeAgent.build_code(...) instead."
        )

    async def stream_build_code(
        self, request: CodeBuildRequest,
    ) -> AsyncIterator[SessionEvent]:
        run_request = self._build_request_from_code_build(request)
        async for event in self.stream(run_request):
            yield event

    def _build_request_from_code_build(
        self, request: CodeBuildRequest,
    ) -> AgentRunRequest:
        return AgentRunRequest(
            instruction=self._serialize_code_build_request(request),
            cwd=request.cwd,
            config=request.config,
            session=request.session,
            max_tool_rounds=request.max_tool_rounds,
            prompt_language=request.prompt_language,
            permission_mode=request.permission_mode,
            agent_mode=request.agent_mode,
            metadata=dict(request.metadata or {}),
            system_prompt_key=None,
            system_prompt_context={
                "build_mode": True,
                "agent_mode": request.agent_mode == "agent",
                "spec_mode": request.agent_mode == "spec",
                "plan_mode": request.agent_mode == "plan",
            },
            event_sink=request.event_sink,
        )

    @staticmethod
    def _serialize_code_build_request(request: CodeBuildRequest) -> str:
        try:
            cwd_path = Path(request.cwd).resolve()
            cwd_resolved = str(cwd_path)
        except (OSError, ValueError):
            cwd_path = Path(request.cwd or ".")
            cwd_resolved = request.cwd or ""
        return json.dumps(
            {
                "goal": request.goal,
                "cwd": cwd_resolved,
                "context_paths": [
                    str(((cwd_path / p) if not Path(p).is_absolute() else Path(p)).resolve())
                    if p else p
                    for p in (request.context_paths or [])
                ],
                "constraints": list(request.constraints or []),
                "acceptance_criteria": list(request.acceptance_criteria or []),
            },
            ensure_ascii=False,
            indent=2,
        )

    def _run_blocking_build(
        self, request: CodeBuildRequest,
    ) -> AgentRunResult:
        run_request = self._build_request_from_code_build(request)
        return self.run_sync(run_request)

    async def stream(self, request: AgentRunRequest) -> AsyncIterator[SessionEvent]:
        """Yield SessionEvents as they arrive from v5 (true streaming).

        Engine runs on a background thread; v5's EventBus pushes events
        into an asyncio.Queue we await here. When the engine finishes, a
        sentinel terminates the iterator and the run result is folded
        into a final SessionEvent (event_type='ccx.run.completed') for
        callers that need terminal info without separate ``run()``.
        Persistent memory v1 can be recalled on this path, but does not
        summarize/write streamed runs.
        """
        agen = self._run_streaming(request)
        try:
            async for ev in agen:
                yield ev
        finally:
            await agen.aclose()

    async def _run_streaming(
        self, request: AgentRunRequest,
    ) -> AsyncIterator[SessionEvent]:
        config = self._resolve_config(request)
        mode = self._resolve_mode(request, config)
        if mode not in SUPPORTED_AGENT_MODES:
            raise NotImplementedError(
                f"ccx.CodeAgent does not yet implement agent_mode={mode!r}; "
                f"use core.cc.api.CodeAgent for that mode"
            )
        # Goal mode is a multi-iteration meta-orchestration loop with no single
        # streaming chain to yield; it is non-streaming by design. Reject here
        # (it IS in SUPPORTED, so the check above won't) so callers get a clear
        # signal rather than a half-streamed goal run.
        if mode == "goal":
            raise NotImplementedError(
                "ccx.CodeAgent does not support streaming for agent_mode='goal'; "
                "use run()/run_sync() (goal mode is a non-streaming loop)"
            )
        llm = self._resolve_llm(config)
        workspace = self._resolve_workspace(request)
        cwd_resolved = str(Path(request.cwd or ".").resolve())
        # R1: lift ``preferred_model`` to the top level so the
        # _make_mode_tool fn finds it without having to spelunk the
        # nested ``request_metadata`` dict. The original key stays
        # inside request_metadata so other consumers see the same
        # request shape as before.
        _root_meta: dict[str, Any] = {
            "request_metadata": dict(request.metadata or {}),
            "cwd": request.cwd,
        }
        _request_pref = (request.metadata or {}).get("preferred_model")
        if isinstance(_request_pref, str) and _request_pref:
            _root_meta["preferred_model"] = _request_pref
        # Lift a spawn ``ccx_contract`` to the top level too (same reason as
        # preferred_model): the agent runner's contract parser reads top-level
        # metadata, not the nested request_metadata copy. Only present when a
        # caller actually attached one, so a root agent run under
        # ``sgar_run_criterion_checks`` (run_checks) can be governed from the
        # CLI via ``metadata_json={"ccx_contract": {...}}``. Absent ⇒ inert.
        _request_contract = (request.metadata or {}).get("ccx_contract")
        if isinstance(_request_contract, dict):
            _root_meta["ccx_contract"] = _request_contract
        # Lift a goal-mode explicit DAG to the top level too: ``PlanModeRunner``'s
        # short-circuit reads top-level invocation metadata, not the nested
        # request_metadata copy. Only present when the goal loop drives an
        # explicit-route iteration (which stamps ``ccx_goal_dag`` into the request
        # metadata); absent for every other run ⇒ inert.
        _request_goal_dag = (request.metadata or {}).get("ccx_goal_dag")
        if _request_goal_dag is not None:
            _root_meta["ccx_goal_dag"] = _request_goal_dag
        root = root_node_for(
            goal=request.instruction,
            mode=mode,
            metadata=_root_meta,
        )
        # ``docs_output_path`` may be supplied via request metadata when
        # the caller wants the report to land at a specific location
        # (e.g. inside the project's actual ``docs/`` directory) rather
        # than the auto-generated ``.ccx/docs/doc-<id>.md``. Empty
        # string and missing key both fall through to the default.
        _meta = dict(request.metadata or {})
        _docs_output_path = str(_meta.get("docs_output_path") or "").strip() or None
        # Bound each ccx mode tool invocation. Without a per-tool timeout
        # the v5 dispatcher blocks indefinitely on a hung LLM call inside
        # plan/spec/agent runners (those call ``llm(...)`` synchronously
        # — there is no async wait_for guard like cc QueryEngine has).
        # Mirror cc's existing wall-clock budget knob so a single config
        # value drives both layers.
        node_timeout_s = _node_timeout_from_config(config)
        content_store = _create_content_store(
            self._content_store,
            cwd=cwd_resolved,
        )
        run_steer_inbox = self._new_run_steer_inbox()
        try:
            bundle = build_runtime(
                workspace=workspace,
                llm=llm,
                llm_client_provider=self.llm_client_provider,
                language=config.prompt_language if config else "en",
                parallelism=max(1, getattr(config, "spec_max_parallel_agents", 4)),
                propose_initial=lambda _goal: [root],
                agent_runner_kind=self._effective_runner_kind(mode),
                cc_config=config,
                cc_cwd=cwd_resolved,
                cc_max_tool_rounds=(
                    request.max_tool_rounds
                    if request.max_tool_rounds is not None
                    else self._max_tool_rounds
                ),
                artifact_cwd=cwd_resolved,
                docs_artifact_root=str(Path(cwd_resolved) / ".ccx" / "docs"),
                docs_write_artifact=True,
                docs_output_path=_docs_output_path,
                budget=_budget_from_config(config),
                **_spawn_policy_from_config(config),
                node_timeout_s=node_timeout_s,
                steer_inbox=run_steer_inbox,
                llm_routes=self._llm_routes,
                preferred_model_overrides=self._preferred_model_overrides,
                content_store=content_store,
                sgar_run_criterion_checks=self._sgar_run_criterion_checks,
                sgar_criterion_check_timeout_s=(
                    self._sgar_criterion_check_timeout_s
                ),
            )
        except BaseException as exc:
            self._unregister_steer_run(None, run_steer_inbox)
            _close_content_store_on_setup_failure(content_store)
            translated = _translate_runtime_setup_error(exc, workspace=workspace)
            if translated is exc:
                raise
            raise translated from exc
        # Setup between bundle construction and engine-thread start must
        # shut the bundle down on failure: the try/finally below only
        # owns shutdown once the engine thread is running, so an
        # exception here (resume injection, subscribe, thread start)
        # would otherwise leak the SQLite runtime DB connection.
        try:
            # Resume injection: must happen after the bundle is built
            # (we need bundle.runtime.event_store to read the prior run)
            # but before engine.run() fires (the root NodeSpec is
            # captured by propose_initial's closure; mutating it after
            # run-start is too late). Safe to call unconditionally —
            # no-op when the caller didn't ask for resume.
            _maybe_inject_resume(root=root, bundle=bundle, request=request)
            _maybe_inject_memory(
                root=root, request=request, mode=mode, options=self._memory,
            )

            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[SessionEvent | None] = asyncio.Queue()
            turn_id = f"turn-{mode}"
            run_id_box: dict[str, str] = {}

            def _on_event(event: dict[str, Any]) -> None:
                if event.get("run_id"):
                    run_id_box.setdefault("run_id", str(event.get("run_id")))
                    self._register_steer_run(
                        run_id_box.get("run_id"),
                        run_steer_inbox,
                    )
                session_event = _session_event_from_v5(event, turn_id=turn_id)
                if session_event is None:
                    return
                try:
                    loop.call_soon_threadsafe(queue.put_nowait, session_event)
                except RuntimeError:
                    # Loop may already be closed if caller cancelled; drop
                    # silently — at-least-once delivery semantics.
                    pass

            bundle.runtime.event_bus.subscribe(_on_event, kind="node.")

            verdict_box: dict[str, Any] = {}
            engine = bundle.runtime.engine()
            bundle.engine = engine

            def _drive_engine() -> None:
                try:
                    verdict = engine.run(goal=request.instruction)
                    verdict_box["verdict"] = verdict
                    run_id_box["run_id"] = verdict.run_id
                except BaseException as exc:  # noqa: BLE001
                    verdict_box["error"] = exc
                finally:
                    # Sentinel — caller's await on queue.get() will
                    # unblock.
                    try:
                        loop.call_soon_threadsafe(queue.put_nowait, None)
                    except RuntimeError:
                        pass

            thread = threading.Thread(target=_drive_engine, daemon=True)
            thread.start()
        except BaseException:
            self._unregister_steer_run(
                run_id_box.get("run_id") if "run_id_box" in locals() else None,
                run_steer_inbox,
            )
            try:
                await asyncio.shield(asyncio.to_thread(bundle.shutdown))
            except BaseException:
                # Never mask the original failure with a teardown error.
                logger.warning(
                    "ccx stream: bundle.shutdown() failed while handling "
                    "a setup failure; bundle teardown may be incomplete",
                    exc_info=True,
                )
            raise
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
            # Surface a terminal event with summary for callers that want it.
            verdict = verdict_box.get("verdict")
            if verdict is not None:
                yield SessionEvent(
                    event_type="ccx.run.completed",
                    turn_id=turn_id,
                    payload={
                        "run_id": verdict.run_id,
                        "status": verdict.status.value,
                        "succeeded": verdict.succeeded,
                        "node_count": verdict.node_count,
                        "abandoned": verdict.abandoned,
                    },
                )
            err = verdict_box.get("error")
            if err is not None:
                yield SessionEvent(
                    event_type="ccx.run.failed",
                    turn_id=turn_id,
                    payload={"error": str(err), "kind": type(err).__name__},
                )
        finally:
            try:
                run_id = run_id_box.get("run_id")
                cancel_sent = False
                deadline = (
                    asyncio.get_running_loop().time()
                    + _STREAM_TEARDOWN_GRACE_S
                )
                while (
                    thread.is_alive()
                    and asyncio.get_running_loop().time() < deadline
                ):
                    run_id = run_id_box.get("run_id") or run_id
                    if run_id and not cancel_sent:
                        try:
                            await self._await_cleanup(
                                asyncio.wait_for(
                                    asyncio.to_thread(engine.cancel, run_id),
                                    timeout=_CANCEL_CLEANUP_TIMEOUT_S,
                                )
                            )
                        except Exception:
                            logger.warning(
                                "ccx stream: engine.cancel failed during teardown",
                                exc_info=True,
                            )
                        cancel_sent = True
                    remaining = max(
                        0.0,
                        deadline - asyncio.get_running_loop().time(),
                    )
                    await self._await_cleanup(
                        asyncio.to_thread(thread.join, min(0.1, remaining))
                    )
                if thread.is_alive():
                    logger.warning(
                        "ccx stream: engine thread still alive after 30s; "
                        "continuing with bounded bundle shutdown."
                    )
                run_id = run_id_box.get("run_id") or run_id
                await self._await_cleanup(
                    asyncio.to_thread(
                        bundle.shutdown,
                        run_id=run_id,
                        content_retain_for_ms=(
                            self._content_store.retain_for_ms
                            if self._content_store is not None
                            else None
                        ),
                    )
                )
            finally:
                self._unregister_steer_run(
                    run_id_box.get("run_id"),
                    run_steer_inbox,
                )

    # -- internals ---------------------------------------------------------

    def _resolve_config(self, request: AgentRunRequest) -> CCConfig:
        if request.config is not None:
            return request.config
        if self.config is not None:
            return self.config
        return CCConfig()

    def _resolve_mode(self, request: AgentRunRequest, config: CCConfig) -> str:
        explicit = (
            request.agent_mode
            if request.agent_mode is not None
            else (config.agent_mode if config else "")
        )
        if explicit:
            return explicit
        return DEFAULT_AGENT_MODE

    def _resolve_llm(self, config: CCConfig) -> LLMCallable:
        if self._llm_callable is not None:
            return self._llm_callable
        if self.llm_client_provider is None:
            raise RuntimeError(
                "CodeAgent has no LLM source: pass `llm` or "
                "`llm_client_provider` to the constructor"
            )
        return from_provider(self.llm_client_provider, config)

    def _resolve_workspace(self, request: AgentRunRequest) -> Path:
        cwd = Path(request.cwd or ".")
        # Use a hidden subdirectory of cwd if it exists, otherwise a temp dir.
        candidate = cwd / ".ccx" / "runtime"
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except OSError:
            return Path(tempfile.mkdtemp(prefix="ccx-runtime-"))

    def _effective_runner_kind(self, mode: str) -> str:
        """Pick the agent_runner_kind to actually use for this run.

        Resolution rules:

        * ``configured == "auto"`` (the default) → ``cc_query_loop`` when
          cc's QueryEngine is importable, else ``lite``. This makes the
          common case ("agent nodes should be able to call file_write")
          work without forcing every caller to pass the runner kind.
        * ``configured == "lite"`` and ``mode`` ∈ {sgar, sgarx, blueprint}
          → upgrade to cc_query_loop. SGAR / blueprint workflows spawn
          agent children that MUST be able to read/write .sgar/ artifacts;
          a literal "lite" here was almost certainly an oversight rather
          than an intent to disable tools, so we silently fix it (with a
          warning when cc isn't available).
        * Anything else → return ``configured`` unchanged. Callers that
          explicitly pass "lite" (typically test stubs that supply a
          text-only LLMCallable) get exactly what they asked for.
        """
        configured = self._agent_runner_kind

        def _cc_available() -> bool:
            try:
                from core.cc.config import CCConfig  # noqa: F401
                from .agents.cc_agent import CcAgentRunner  # noqa: F401
            except ImportError:
                return False
            return True

        if configured == "auto":
            if _cc_available():
                return "cc_query_loop"
            logger.warning(
                "agent_runner_kind='auto' but cc is not importable; falling "
                "back to 'lite' — this path has NO tools: agent nodes can "
                "only return text and will not read/write files or run "
                "commands. Install/repair cc to get tool-backed execution.",
            )
            return "lite"

        if mode in {"sgar", "sgarx", "blueprint"} and configured == "lite":
            if _cc_available():
                logger.debug(
                    "agent_mode=%r: auto-upgrading agent_runner_kind from "
                    "'lite' to 'cc_query_loop' so child agents can drive "
                    "SGAR ops.",
                    mode,
                )
                return "cc_query_loop"
            logger.warning(
                "agent_mode=%r needs cc_query_loop for child agents to "
                "read/write .sgar/ artifacts; cc is not importable, "
                "staying on lite — child agents spawned by this run will "
                "not be able to interact with SGAR files.",
                mode,
            )
            return configured

        return configured

    def _run_blocking(
        self,
        request: AgentRunRequest,
        state: dict[str, Any] | None = None,
    ) -> tuple[CcxRuntimeBundle, Any, list[SessionEvent]]:
        config = self._resolve_config(request)
        mode = self._resolve_mode(request, config)
        if mode not in SUPPORTED_AGENT_MODES:
            raise NotImplementedError(
                f"ccx.CodeAgent does not yet implement agent_mode={mode!r}; "
                f"use core.cc.api.CodeAgent for that mode"
            )

        llm = self._resolve_llm(config)
        workspace = self._resolve_workspace(request)
        cwd_resolved = str(Path(request.cwd or ".").resolve())
        # Pre-build the root NodeSpec so propose_initial can return it.
        # R1: lift ``preferred_model`` to the top level so the
        # _make_mode_tool fn finds it without having to spelunk the
        # nested ``request_metadata`` dict. The original key stays
        # inside request_metadata so other consumers see the same
        # request shape as before.
        _root_meta: dict[str, Any] = {
            "request_metadata": dict(request.metadata or {}),
            "cwd": request.cwd,
        }
        _request_pref = (request.metadata or {}).get("preferred_model")
        if isinstance(_request_pref, str) and _request_pref:
            _root_meta["preferred_model"] = _request_pref
        # Lift a spawn ``ccx_contract`` to the top level too (same reason as
        # preferred_model): the agent runner's contract parser reads top-level
        # metadata, not the nested request_metadata copy. Only present when a
        # caller actually attached one, so a root agent run under
        # ``sgar_run_criterion_checks`` (run_checks) can be governed from the
        # CLI via ``metadata_json={"ccx_contract": {...}}``. Absent ⇒ inert.
        _request_contract = (request.metadata or {}).get("ccx_contract")
        if isinstance(_request_contract, dict):
            _root_meta["ccx_contract"] = _request_contract
        # Lift a goal-mode explicit DAG to the top level too: ``PlanModeRunner``'s
        # short-circuit reads top-level invocation metadata, not the nested
        # request_metadata copy. Only present when the goal loop drives an
        # explicit-route iteration (which stamps ``ccx_goal_dag`` into the request
        # metadata); absent for every other run ⇒ inert.
        _request_goal_dag = (request.metadata or {}).get("ccx_goal_dag")
        if _request_goal_dag is not None:
            _root_meta["ccx_goal_dag"] = _request_goal_dag
        root = root_node_for(
            goal=request.instruction,
            mode=mode,
            metadata=_root_meta,
        )
        # ``docs_output_path`` may be supplied via request metadata when
        # the caller wants the report to land at a specific location
        # (e.g. inside the project's actual ``docs/`` directory) rather
        # than the auto-generated ``.ccx/docs/doc-<id>.md``. Empty
        # string and missing key both fall through to the default.
        _meta = dict(request.metadata or {})
        _docs_output_path = str(_meta.get("docs_output_path") or "").strip() or None
        node_timeout_s = _node_timeout_from_config(config)
        content_store = _create_content_store(
            self._content_store,
            cwd=cwd_resolved,
        )
        run_steer_inbox = self._new_run_steer_inbox()
        try:
            bundle = build_runtime(
                workspace=workspace,
                llm=llm,
                llm_client_provider=self.llm_client_provider,
                language=config.prompt_language if config else "en",
                parallelism=max(1, getattr(config, "spec_max_parallel_agents", 4)),
                propose_initial=lambda _goal: [root],
                agent_runner_kind=self._effective_runner_kind(mode),
                cc_config=config,
                cc_cwd=cwd_resolved,
                cc_max_tool_rounds=(
                    request.max_tool_rounds
                    if request.max_tool_rounds is not None
                    else self._max_tool_rounds
                ),
                artifact_cwd=cwd_resolved,
                docs_artifact_root=str(Path(cwd_resolved) / ".ccx" / "docs"),
                docs_write_artifact=True,
                docs_output_path=_docs_output_path,
                budget=_budget_from_config(config),
                **_spawn_policy_from_config(config),
                node_timeout_s=node_timeout_s,
                steer_inbox=run_steer_inbox,
                llm_routes=self._llm_routes,
                preferred_model_overrides=self._preferred_model_overrides,
                content_store=content_store,
                sgar_run_criterion_checks=self._sgar_run_criterion_checks,
                sgar_criterion_check_timeout_s=(
                    self._sgar_criterion_check_timeout_s
                ),
            )
        except BaseException as exc:
            self._unregister_steer_run(None, run_steer_inbox)
            _close_content_store_on_setup_failure(content_store)
            translated = _translate_runtime_setup_error(exc, workspace=workspace)
            if translated is exc:
                raise
            raise translated from exc
        if state is not None:
            state["bundle"] = bundle
        # Everything from here until the bundle is returned must shut
        # the bundle down on failure: the caller's
        # ``finally: bundle.shutdown()`` only guards the bundle *after*
        # it is handed over, so an exception out of resume injection /
        # subscribe / engine.run() would otherwise leak the SQLite
        # runtime DB connection (.ccx/runtime/runtime.db) — fatal in a
        # long-lived host process where failed runs accumulate.
        try:
            # See _run_streaming for why this lives between
            # build_runtime and engine.run().
            _maybe_inject_resume(root=root, bundle=bundle, request=request)
            _maybe_inject_memory(
                root=root, request=request, mode=mode, options=self._memory,
            )

            return self._drive_blocking_engine(
                request,
                bundle,
                mode,
                state,
                run_steer_inbox=run_steer_inbox,
            )
        except BaseException:
            self._unregister_steer_run(
                str(state.get("run_id")) if state and state.get("run_id") else None,
                run_steer_inbox,
            )
            try:
                bundle.shutdown(
                    run_id=str(state.get("run_id")) if state and state.get("run_id") else None,
                    content_retain_for_ms=(
                        self._content_store.retain_for_ms
                        if self._content_store is not None
                        else None
                    ),
                )
            except Exception:
                # Never mask the original failure with a teardown error.
                logger.warning(
                    "ccx: bundle.shutdown() failed while handling a run "
                    "failure; bundle teardown may be incomplete",
                    exc_info=True,
                )
            raise

    def _drive_blocking_engine(
        self,
        request: AgentRunRequest,
        bundle: CcxRuntimeBundle,
        mode: str,
        state: dict[str, Any] | None = None,
        *,
        run_steer_inbox: SteerInbox,
    ) -> tuple[CcxRuntimeBundle, Any, list[SessionEvent]]:
        events: list[SessionEvent] = []
        turn_id = f"turn-{mode}"
        run_id_box: dict[str, str] = {}
        # Honour ``request.event_sink`` so callers (notably
        # ``task/deep/ccx.py``) can stream node events live instead of
        # waiting for ``run_sync`` to return its full ``events`` list.
        # During a long blocking node (e.g. a hung LLM call) live events
        # are the only signal that the engine started at all.
        live_sink = request.event_sink

        def _on_event(event: dict[str, Any]) -> None:
            if event.get("run_id"):
                run_id = str(event.get("run_id"))
                if state is not None:
                    state.setdefault("run_id", run_id)
                run_id_box.setdefault("run_id", run_id)
                self._register_steer_run(run_id, run_steer_inbox)
            session_event = _session_event_from_v5(event, turn_id=turn_id)
            if session_event is None:
                return
            events.append(session_event)
            if live_sink is not None:
                try:
                    live_sink(session_event)
                except Exception:
                    # Event delivery is best-effort; never let an
                    # observer fault interrupt engine progress.
                    pass

        bundle.runtime.event_bus.subscribe(_on_event, kind="node.")

        engine = bundle.runtime.engine()
        bundle.engine = engine
        if state is not None:
            state["engine"] = engine
        try:
            verdict = engine.run(goal=request.instruction)
            run_id_box["run_id"] = verdict.run_id
            if state is not None:
                state["run_id"] = verdict.run_id
        finally:
            self._unregister_steer_run(
                run_id_box.get("run_id"),
                run_steer_inbox,
            )
        return bundle, verdict, events

    def _maybe_finalize_memory(
        self,
        request: AgentRunRequest,
        result: AgentRunResult,
    ) -> None:
        options = self._memory
        if options is None or not options.enabled or not options.auto_summarize:
            return
        if memory_disabled(request.metadata):
            return
        try:
            config = self._resolve_config(request)
            mode = self._resolve_mode(request, config)
            store = JsonlMemoryStore(_memory_root(request=request, options=options))
            route = options.summary_route
            routed = self._llm_routes.get(route) if route and self._llm_routes else None
            llm = routed if routed is not None else self._resolve_llm(config)
            entries = summarize_run(
                llm=llm,
                request=request,
                result=result,
                options=options,
                existing_tags=store.tag_vocabulary(),
                mode=mode,
            )
            if not entries:
                return
            append_result = store.append(
                entries,
                max_total_entries=options.max_total_entries,
                entry_text_max_chars=options.entry_text_max_chars,
            )
            logger.info(
                "ccx memory: stored %d entries (run %s)",
                append_result.stored,
                result.session_id,
            )
        except Exception:
            logger.warning("ccx memory: summarize failed", exc_info=True)

    def _build_result(
        self,
        request: AgentRunRequest,
        bundle: CcxRuntimeBundle,
        verdict: Any,
        events: list[SessionEvent],
    ) -> AgentRunResult:
        run_id = verdict.run_id
        # Root result is the terminal value of the root node, whose ID
        # is ``root-{mode}`` (see runtime.root_node_for). Resolve the
        # mode from the request so a future fourth mode is found
        # automatically.
        mode = self._resolve_mode(request, self._resolve_config(request))
        root_node_id = f"root-{mode}"
        engine = getattr(bundle, "engine", None)
        try:
            memory_results = (
                engine.list_node_results(run_id)
                if engine is not None and hasattr(engine, "list_node_results")
                else {}
            )
        except Exception:
            memory_results = {}
        root_row = bundle.runtime.graph_store.get_node(run_id, root_node_id)
        final_text = ""
        res = memory_results.get(root_node_id)
        if res is not None:
            if isinstance(res, dict):
                final_text = res.get("final_text", "") or ""
            else:
                final_text = str(res)
        elif root_row is not None:
            res = root_row.get("result")
            if isinstance(res, dict):
                final_text = res.get("final_text", "") or ""
            elif res is not None:
                final_text = str(res)

        # Append leaf agent texts.
        all_nodes = bundle.runtime.graph_store.list_nodes(run_id)
        leaf_texts: list[str] = []
        for row in all_nodes:
            if row["node_id"] == root_node_id:
                continue
            if row["spec"]["tool"] != "ccx.agent":
                continue
            if row["state"] != NodeState.SUCCEEDED.value:
                continue
            res = memory_results.get(row["node_id"])
            if res is None:
                res = row.get("result")
            if isinstance(res, dict):
                t = res.get("final_text", "") or ""
            elif isinstance(res, str):
                t = res
            else:
                t = ""
            if t:
                leaf_texts.append(t)

        if leaf_texts:
            joined = "\n".join(leaf_texts)
            final_text = f"{final_text}\n\n{joined}".strip() if final_text else joined

        # Walk all node rows for any ``extras.artifact_path`` produced by
        # a terminal mode runner (e.g. doc-synth, blueprint write). The
        # last non-empty value wins because doc mode emits exactly one
        # synth node per run. Surface it on the snapshot so the CLI can
        # show ``artifact: <path>`` without callers having to dig into
        # the event stream.
        artifact_path: str | None = None
        for res in memory_results.values():
            if isinstance(res, dict):
                extras = res.get("extras")
                if isinstance(extras, dict):
                    ap = extras.get("artifact_path")
                    if ap:
                        artifact_path = str(ap)
        for row in all_nodes:
            res = memory_results.get(row["node_id"])
            if res is None:
                res = row.get("result")
            if isinstance(res, dict):
                extras = res.get("extras")
                if isinstance(extras, dict):
                    ap = extras.get("artifact_path")
                    if ap:
                        artifact_path = str(ap)

        # Surface ALL spawned-child terminal artifacts, not just the last one.
        # When an agent root fans out via ``ccx_spawn`` into doc/research/agent
        # children, each runs as a separate v5 node AFTER the root node returns,
        # so the root cannot harvest them and the leaf-text aggregation above
        # only covers ``ccx.agent`` nodes. Without this, a fan-out run's real
        # output (e.g. N doc-synth artifacts) is stranded: ``artifact_path``
        # keeps only the last and ``final_text`` keeps only the root's (often a
        # "waiting for children" non-answer). Additive: new snapshot key; an
        # empty list when nothing carried an artifact, so non-fanout runs keep a
        # byte-identical snapshot apart from this always-present (empty) key.
        child_artifacts: list[dict[str, Any]] = []
        for row in all_nodes:
            if row["node_id"] == root_node_id:
                continue
            res = memory_results.get(row["node_id"])
            if res is None:
                res = row.get("result")
            if not isinstance(res, dict):
                continue
            extras = res.get("extras")
            if not isinstance(extras, dict):
                continue
            ap = extras.get("artifact_path")
            if ap:
                spec = row.get("spec")
                child_artifacts.append({
                    "node_id": row["node_id"],
                    "tool": spec.get("tool") if isinstance(spec, dict) else None,
                    "state": row.get("state"),
                    "artifact_path": str(ap),
                })

        # Aggregate governance rejections. A BlueprintModeRunner /
        # BlueprintxModeRunner node that hit an SgarError stays SUCCEEDED on
        # the v5 graph (the node ran and returned a deterministic "ERROR:"
        # text — we deliberately do NOT fail the node, to keep the
        # node-success + return-text contract and retry semantics intact).
        # That makes the rejection invisible to the orchestration layer
        # unless it digs into per-node extras, which conflicts with SGAR's
        # "hard governance" promise. Surface every ``sgar_failed`` node here
        # so a caller can detect the refusal directly instead of inferring
        # it from a state machine that quietly failed to advance.
        governance_errors: list[dict[str, Any]] = []
        for row in all_nodes:
            res = memory_results.get(row["node_id"])
            if res is None:
                res = row.get("result")
            if not isinstance(res, dict):
                continue
            extras = res.get("extras")
            if not isinstance(extras, dict) or not extras.get("sgar_failed"):
                continue
            governance_errors.append({
                "node_id": row["node_id"],
                "error": extras.get("sgar_error") or extras.get("error"),
                "error_code": extras.get("sgar_error_code"),
                "command": extras.get("sgar_command"),
            })

        # Lift the spawn-contract verdict (governed_spawn) the same way as
        # ``artifact_path`` above. It lives only in per-node ``extras``; the
        # v5-graph -> AgentRunResult projection drops per-node extras and the
        # ``ccx.node.completed`` event truncates ``result_summary`` at 200
        # chars, so without this the machine-derived verdict is invisible to
        # the CLI / output_json path. Last non-empty wins (mirroring
        # artifact_path); stays ``None`` when no node carried a contract, so
        # the no-contract path keeps a byte-identical snapshot.
        contract_verdict: dict[str, Any] | None = None
        for row in all_nodes:
            res = memory_results.get(row["node_id"])
            if res is None:
                res = row.get("result")
            if not isinstance(res, dict):
                continue
            extras = res.get("extras")
            if not isinstance(extras, dict):
                continue
            cv = extras.get("contract_verdict")
            if cv:
                contract_verdict = cv

        session_snapshot = {
            "run_id": run_id,
            "status": verdict.status.value,
            "node_count": verdict.node_count,
            "succeeded": verdict.succeeded,
            "failed": verdict.failed,
            "abandoned": verdict.abandoned,
            "iterations": verdict.iterations,
            "elapsed_s": verdict.elapsed_s,
            "artifact_path": artifact_path,
            "child_artifacts": child_artifacts,
            "governance_errors": governance_errors,
            "contract_verdict": contract_verdict,
            # Run-level audit verdict. Default ``None`` (no run-level contract);
            # set once from the outer verify-repair loop (``_run_with_run_audit``
            # → ``_stamp``), NOT scavenged from node extras — kept distinct from
            # the per-node ``contract_verdict`` above. A no-contract snapshot
            # therefore carries this key as ``None``; behaviour is unchanged.
            "run_audit_verdict": None,
            # Goal-mode verdict. Default ``None`` (not a goal run); set once by
            # the goal loop's ``_stamp_goal`` for ``agent_mode="goal"``. Additive
            # like ``run_audit_verdict`` — a non-goal snapshot carries it as
            # ``None`` and behaviour is unchanged.
            CCX_GOAL_VERDICT_SNAPSHOT_KEY: None,
        }

        # v5 reports "some succeeded, some abandoned" as COMPLETED on purpose
        # (best-effort completion; the abandoned count stays on the Verdict).
        # That is a partial/degraded outcome a caller can easily miss, so we
        # surface it: a warning log + an additive snapshot flag. This is
        # observe-only — we deliberately do NOT flip ``failed`` (preserving
        # v5's best-effort contract and keeping the result byte-equivalent
        # for callers that already inspect ``abandoned``).
        if verdict.status == RunStatus.COMPLETED and verdict.abandoned > 0:
            logger.warning(
                "ccx run %s COMPLETED with %d abandoned node(s); "
                "treating as a partial/degraded run",
                run_id, verdict.abandoned,
            )
            session_snapshot["abandoned_warning"] = True

        degraded_result = (
            verdict.status == RunStatus.COMPLETED
            and root_row is None
            and memory_results.get(root_node_id) is None
        )
        failed = verdict.status not in (RunStatus.COMPLETED,) or degraded_result
        tool_call_count = 0
        try:
            tool_call_count = sum(
                1
                for event in bundle.runtime.event_store.read_last(
                    run_id, limit=100_000,
                )
                if event.get("kind") == "cc.tool_use"
            )
        except Exception:
            logger.debug("ccx: failed to count cc.tool_use events", exc_info=True)
        return AgentRunResult(
            final_text=final_text,
            session_id=run_id,
            turn_id=events[0].turn_id if events else None,
            cwd=str(Path(request.cwd or ".").resolve()),
            session_snapshot=session_snapshot,
            events=events,
            messages=[],
            tool_call_count=tool_call_count,
            failed=failed,
            error_code=(
                None
                if not failed
                else (
                    "CCX_RESULT_DEGRADED"
                    if degraded_result
                    else "CCX_VERDICT_NOT_COMPLETED"
                )
            ),
            error_message=(
                None
                if not failed
                else (
                    "verdict completed but root result is unavailable"
                    if degraded_result
                    else f"verdict status: {verdict.status.value}"
                )
            ),
        )


# --------------------------------------------------------------------------- #
# Public re-exports — same surface as core.cc.api
# --------------------------------------------------------------------------- #

__all__ = [
    "AgentRunRequest",
    "AgentRunResult",
    "CodeAgent",
    "CodeBuildRequest",
    "ContentStoreOptions",
    "MemoryOptions",
]
