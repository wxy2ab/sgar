"""Unified ccx spawn tool — absorbs ccx_research and ccx_sgar.

A single LLM-facing tool (wire name ``ccx_spawn``) replaces the previous
trio of ``ccx_spawn`` / ``ccx_research`` / ``ccx_sgar``. The three did the
same thing — buffer a deferred ``SubagentInvocation`` for the cc-agent
turn to drain — and only differed in payload shape and which buffer the
record landed in.

The unified tool routes on ``mode``:

* ``mode in {"plan","spec","agent","doc","ask","blueprint"}`` → SpawnBuffer
  with payload ``{"goal": ..., "metadata": ...}``.
* ``mode == "research"`` → ResearchBuffer with payload
  ``{"question": ..., "scope": ..., "focus_paths": [...]}``.
* ``mode == "sgar"`` → SgarBuffer with payload
  ``{"instruction": ..., "metadata": ...}``.

Single ``spawns[]`` accepts mixed-mode entries; each entry routes
independently. ``sequential=true`` chains entries within their own
buffer (research items run in parallel by design — sequential applies
to spawn/sgar entries).

Backward compatibility: legacy callers passing ``{goal, mode}`` (the old
``ccx_spawn`` shape) still work — when ``payload`` is absent the tool
synthesises one from the top-level field that matches the chosen
mode. Legacy ``spawns[]`` entries also work unchanged.
"""

from __future__ import annotations

from typing import Any

from core.cc.tools.base import (
    BaseTool,
    ToolCall,
    ToolResult,
    ToolSpec as CcToolSpec,
    ValidationResult,
)

from .ccx_research_tool import (
    ResearchBuffer,
    ResearchRequest,
    normalize_focus_paths,
)
from .ccx_sgar_tool import SgarBuffer, SgarRequest
from .ccx_spawn_tool import SpawnBuffer, SpawnRequest
from .subagent import CCX_REQUIRES_APPROVAL_UNSUPPORTED


_TOOL_NAME = "ccx_spawn"

_SPAWN_MODES: frozenset[str] = frozenset(
    {"plan", "spec", "agent", "doc", "ask", "blueprint"}
)
_ALL_MODES: tuple[str, ...] = (
    "plan", "spec", "agent", "doc", "ask", "blueprint", "sgar", "sgarx",
    "research",
)
# Governance modes share the SgarBuffer + ``{instruction, metadata}``
# payload shape; only the target runtime differs (.sgar/ vs .sgarx/).
_SGAR_MODES: frozenset[str] = frozenset({"sgar", "sgarx"})

_TOOL_DESCRIPTION = (
    "Spawn one or more subagents that run after this turn finishes. Pick a "
    "mode and pass a payload that matches it:\n"
    "- mode in {plan, spec, agent, doc, ask, blueprint}: payload={goal, "
    "metadata?, contract?}. The child runs as that mode (use 'agent' for "
    "ordinary decomposition, 'doc'/'ask' for documentation or read-only Q&A, "
    "'plan'/'spec'/'blueprint' to drive structured planning). Optional "
    "'contract' attaches machine-verified acceptance checks the child must "
    "satisfy (honored by 'agent' mode under cc_query_loop).\n"
    "- mode='research': payload={question, scope?, focus_paths?}. The "
    "child runs read-only with Grep/Glob/Read and returns structured "
    "findings (summary + evidence). Use when N independent investigative "
    "questions can run in parallel.\n"
    "- mode='sgar': payload={instruction, metadata?}. The child drives an "
    "SGAR governance op (init, set-blueprint, validate, start-stage, "
    "verify, close-stage, doctor, trace) against .sgar/. Use "
    "sequential=true to chain dependent ops.\n"
    "- mode='sgarx': same payload as 'sgar' but targets the extended "
    ".sgarx/ runtime, which also offers reopen-stage / abandon-stage "
    "recovery ops. Use one governance runtime per workflow.\n"
    "Use spawns=[{mode, payload}, ...] to enqueue several at once; modes "
    "may be mixed. Each becomes its own node in the orchestration DAG."
)

_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "mode": {
            "type": "string",
            "enum": list(_ALL_MODES),
            "default": "agent",
            "description": (
                "Per-entry mode. Determines which payload shape is "
                "expected. Default 'agent' for ordinary decomposition."
            ),
        },
        "payload": {
            "type": "object",
            "description": (
                "Mode-specific data. {goal, metadata?} for ordinary "
                "spawn modes; {question, scope?, focus_paths?} for "
                "'research'; {instruction, metadata?} for 'sgar'. "
                "If omitted on a single-spawn call the tool falls back "
                "to legacy top-level fields ('goal', 'question', "
                "'instruction')."
            ),
        },
        # Top-level legacy convenience fields (kept for back-compat with
        # callers that don't wrap their args in `payload`).
        "goal": {"type": "string"},
        "question": {"type": "string"},
        "scope": {"type": "string"},
        "focus_paths": {"type": "array", "items": {"type": "string"}},
        "instruction": {"type": "string"},
        "metadata": {"type": "object"},
        "contract": {
            "type": "object",
            "description": (
                "Optional machine-verified acceptance contract for a spawn "
                "mode (honored by 'agent' under cc_query_loop). Spawner-"
                "authored — do NOT ask the child to emit it. Shape: "
                "{acceptance:[{id,text,check?}], verify:'check'|'none', "
                "loop:{max_iters,no_progress_stop}}. The child's turn re-runs "
                "with failing-check evidence until each [check:] command "
                "passes or a bound trips."
            ),
        },
        "spawns": {
            "type": "array",
            "description": (
                "Bulk form — list of {mode, payload} entries. Each entry "
                "routes independently; mixed modes are allowed."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": list(_ALL_MODES)},
                    "payload": {"type": "object"},
                    # Legacy per-entry shortcuts.
                    "goal": {"type": "string"},
                    "question": {"type": "string"},
                    "scope": {"type": "string"},
                    "focus_paths": {"type": "array", "items": {"type": "string"}},
                    "instruction": {"type": "string"},
                    "metadata": {"type": "object"},
                    "contract": {"type": "object"},
                },
            },
        },
        "sequential": {
            "type": "boolean",
            "default": False,
            "description": (
                "When true and 'spawns' has multiple entries of the same "
                "buffer (spawn or sgar), chain them left-to-right. "
                "Research entries always run in parallel."
            ),
        },
    },
}


def _normalise_entry(
    entry: dict[str, Any],
    *,
    default_mode: str,
) -> tuple[str, dict[str, Any]]:
    """Resolve (mode, payload) from a raw call argument dict.

    Accepts the new ``{mode, payload}`` shape and the legacy flat shape
    (``{goal, mode}`` / ``{question, scope}`` / ``{instruction}``). The
    returned payload is always a dict with the mode-specific fields the
    underlying buffer expects.
    """
    mode = str(entry.get("mode") or default_mode)
    raw_payload = entry.get("payload")
    if isinstance(raw_payload, dict):
        payload = dict(raw_payload)
    # Legacy fall-back: collect the relevant top-level fields.
    elif mode == "research":
        payload = {
            "question": entry.get("question") or entry.get("goal") or "",
            "scope": entry.get("scope") or "",
            "focus_paths": entry.get("focus_paths"),
            "metadata": dict(entry.get("metadata") or {}),
        }
    elif mode in _SGAR_MODES:
        payload = {
            "instruction": entry.get("instruction") or entry.get("goal") or "",
            "metadata": dict(entry.get("metadata") or {}),
        }
    else:
        payload = {
            "goal": entry.get("goal") or "",
            "metadata": dict(entry.get("metadata") or {}),
        }
    # Fold an optional spawn contract (payload-level or entry-level) into the
    # child's ``metadata["ccx_contract"]`` so it rides through the spawn buffer
    # → drain → child invocation exactly like any other metadata key. Only for
    # spawn modes; ``setdefault`` lets an author who wrote ccx_contract
    # directly into metadata win over the convenience 'contract' field. The
    # contract is opaque here — structural validation happens at parse time in
    # governed_spawn.parse_contract.
    if mode in _SPAWN_MODES:
        contract = payload.pop("contract", None)
        if contract is None:
            contract = entry.get("contract")
        if contract is not None:
            md = dict(payload.get("metadata") or {})
            md.setdefault("ccx_contract", contract)
            payload["metadata"] = md
    return mode, payload


def _validate_payload(mode: str, payload: dict[str, Any]) -> str | None:
    if payload.get("requires_approval") is True:
        return (
            f"{CCX_REQUIRES_APPROVAL_UNSUPPORTED}: "
            "ccx does not support requires_approval=True"
        )
    if mode == "research":
        if not str(payload.get("question") or "").strip():
            return "research payload requires non-empty 'question'"
        try:
            normalize_focus_paths(payload.get("focus_paths"))
        except TypeError as exc:
            return str(exc)
        return None
    if mode in _SGAR_MODES:
        if not str(payload.get("instruction") or "").strip():
            return f"{mode} payload requires non-empty 'instruction'"
        return None
    if mode in _SPAWN_MODES:
        if not str(payload.get("goal") or "").strip():
            return f"{mode} payload requires non-empty 'goal'"
        # ``_normalise_entry`` has already folded any 'contract' field into
        # metadata. A grossly malformed (non-object) contract is worth
        # rejecting early; deep structural validation is parse-time.
        contract = (payload.get("metadata") or {}).get("ccx_contract")
        if contract is not None and not isinstance(contract, dict):
            return "spawn 'contract' must be a JSON object"
        return None
    return f"unknown mode: {mode!r}"


def _normalise_goal(goal: Any) -> str:
    """Collapse whitespace + lowercase a spawn goal for obligation matching.

    Two spawns whose goals differ only in casing / whitespace describe the
    same obligation. Kept intentionally simple — semantic near-duplicates
    (paraphrases) are NOT collapsed; the dedup guard only refuses *exact*
    re-spawns, so a genuinely different sub-task is never blocked.
    """
    return " ".join(str(goal or "").split()).lower()


def _spawn_obligation_key(mode: str, payload: dict[str, Any]) -> tuple[Any, ...]:
    """Obligation identity of a spawn-mode entry: (mode, goal, check-set).

    The ``[check:]`` set of an attached ``ccx_contract`` is folded in so two
    same-goal spawns carrying *different* acceptance contracts stay distinct
    obligations (they discharge different work); same goal + same checks (or
    both contract-less) collapse to one obligation.
    """
    goal = _normalise_goal(payload.get("goal"))
    contract = (payload.get("metadata") or {}).get("ccx_contract")
    checks: tuple[str, ...] = ()
    if isinstance(contract, dict):
        acceptance = contract.get("acceptance")
        if isinstance(acceptance, list):
            checks = tuple(sorted(
                str(c.get("check")).strip()
                for c in acceptance
                if isinstance(c, dict) and c.get("check")
            ))
    return (mode, goal, checks)


class CcxUnifiedTool(BaseTool):
    """Unified ccx_spawn tool — routes by ``mode`` to one of three buffers.

    Holds references to up to three buffers (spawn / research / sgar).
    When a mode is invoked whose corresponding buffer is missing, the
    tool returns a clear error instead of silently dropping the request.
    """

    def __init__(
        self,
        *,
        spawn_buffer: SpawnBuffer | None = None,
        research_buffer: ResearchBuffer | None = None,
        sgar_buffer: SgarBuffer | None = None,
        spawn_unavailable_reason: str | None = None,
        max_fanout: int | None = None,
        count_research_in_fanout: bool = False,
        dedup_spawns: bool = False,
    ) -> None:
        super().__init__(
            spec=CcToolSpec(
                name=_TOOL_NAME,
                description=_TOOL_DESCRIPTION,
                input_schema=_INPUT_SCHEMA,
                is_read_only=True,
                needs_confirmation=False,
                metadata={"ccx": True, "ccx_unified": True},
            )
        )
        self.spawn_buffer = spawn_buffer
        self.research_buffer = research_buffer
        self.sgar_buffer = sgar_buffer
        # Human/LLM-readable explanation for why ordinary spawn modes
        # are deliberately unavailable on this turn (e.g. the recursion
        # depth limit was reached). When set, a spawn-mode entry hitting
        # a missing spawn_buffer returns this text instead of the
        # generic "buffer was not provided" wiring error, so the LLM
        # understands the refusal is policy, not a bug, and finishes
        # the task itself.
        self.spawn_unavailable_reason = spawn_unavailable_reason
        # Ceiling on cumulative ordinary-spawn fan-out WIDTH for the turn
        # this tool serves (the spawn_buffer is per-turn). Mirrors the depth
        # guard: when buffering the spawn-mode entries on this call would push
        # the buffer past the cap, those entries are refused with policy text
        # (the LLM finishes more itself) rather than silently buffered. None
        # disables the cap. research/sgar entries are terminal and uncounted.
        self.max_fanout = max_fanout
        # Opt-in: when True, ccx_research entries count toward the WIDTH cap
        # alongside ordinary spawn modes (they are exempt by default). sgar/
        # sgarx stay exempt regardless. Only affects mixed turns that also
        # enqueue spawn modes (the guard below is gated on spawn_buffer).
        self.count_research_in_fanout = count_research_in_fanout
        # Opt-in (default OFF): refuse an ordinary-spawn entry whose obligation
        # — (mode, normalized-goal, contract [check:] set) — already matches one
        # buffered earlier this turn or emitted earlier in this same call. Kills
        # redundant lateral expansion (spawn A to verify X, then spawn A again
        # to "double-check") that no depth/width/budget cap catches, because a
        # count cap penalizes ALL expansion uniformly while this penalizes only
        # the exact duplicate. research/sgar are terminal and never deduped.
        self.dedup_spawns = dedup_spawns

    def is_concurrency_safe(self, arguments: dict[str, Any]) -> bool:
        del arguments
        return True

    def validate_input(self, arguments: dict[str, Any]) -> ValidationResult:
        args = arguments or {}
        spawns = args.get("spawns")
        has_top_level = any(
            k in args for k in ("payload", "goal", "question", "instruction")
        )
        if spawns is not None and not isinstance(spawns, list):
            return ValidationResult(
                ok=False,
                message="ccx_spawn: 'spawns' must be an array",
            )
        if isinstance(spawns, list) and not spawns:
            return ValidationResult(
                ok=False,
                message="ccx_spawn: 'spawns' must not be empty",
            )
        if isinstance(spawns, list) and has_top_level:
            return ValidationResult(
                ok=False,
                message=(
                    "ccx_spawn: pass either a top-level payload/goal/question/"
                    "instruction OR 'spawns', not both"
                ),
            )
        if spawns is None and not has_top_level:
            return ValidationResult(
                ok=False,
                message=(
                    "ccx_spawn requires a payload (with goal/question/"
                    "instruction depending on mode) or a 'spawns' array"
                ),
            )
        if isinstance(spawns, list):
            for i, entry in enumerate(spawns):
                if not isinstance(entry, dict):
                    return ValidationResult(
                        ok=False,
                        message=f"ccx_spawn.spawns[{i}] must be an object",
                    )
                if entry.get("requires_approval") is True:
                    return ValidationResult(
                        ok=False,
                        message=(
                            f"ccx_spawn.spawns[{i}]: "
                            f"{CCX_REQUIRES_APPROVAL_UNSUPPORTED}: "
                            "ccx does not support requires_approval=True"
                        ),
                    )
                mode, payload = _normalise_entry(entry, default_mode="agent")
                if mode not in _ALL_MODES:
                    return ValidationResult(
                        ok=False,
                        message=(
                            f"ccx_spawn.spawns[{i}].mode={mode!r} not allowed; "
                            f"choose one of {list(_ALL_MODES)}"
                        ),
                    )
                err = _validate_payload(mode, payload)
                if err is not None:
                    return ValidationResult(
                        ok=False, message=f"ccx_spawn.spawns[{i}]: {err}"
                    )
        else:
            if args.get("requires_approval") is True:
                return ValidationResult(
                    ok=False,
                    message=(
                        f"ccx_spawn: {CCX_REQUIRES_APPROVAL_UNSUPPORTED}: "
                        "ccx does not support requires_approval=True"
                    ),
                )
            mode, payload = _normalise_entry(args, default_mode="agent")
            if mode not in _ALL_MODES:
                return ValidationResult(
                    ok=False,
                    message=(
                        f"ccx_spawn: mode={mode!r} not allowed; "
                        f"choose one of {list(_ALL_MODES)}"
                    ),
                )
            err = _validate_payload(mode, payload)
            if err is not None:
                return ValidationResult(ok=False, message=f"ccx_spawn: {err}")
        return ValidationResult(ok=True)

    async def execute(self, tool_call: ToolCall, ctx: Any) -> ToolResult:
        del ctx
        args = dict(tool_call.arguments or {})
        entries: list[tuple[str, dict[str, Any]]] = []
        sequential = bool(args.get("sequential", False))

        spawns = args.get("spawns")
        if isinstance(spawns, list) and spawns:
            for entry in spawns:
                if not isinstance(entry, dict):
                    continue
                entries.append(_normalise_entry(entry, default_mode="agent"))
        else:
            entries.append(_normalise_entry(args, default_mode="agent"))

        queued: list[dict[str, Any]] = []
        missing_buffer: list[str] = []
        refused: list[str] = []
        fanout_refused: list[str] = []
        spawn_idx_in_buffer = 0  # for sequential chaining inside spawn_buffer
        sgar_idx_in_buffer = 0
        # Fan-out width guard (mirrors the depth guard, but for WIDTH). The
        # spawn_buffer is per-turn, so its current length plus the spawn-mode
        # entries on THIS call is the cumulative width the turn would reach.
        # When that exceeds the cap we refuse every spawn-mode entry on this
        # call (policy text, not a buffered request) so the LLM finishes more
        # itself. Skipped when the cap is disabled or there's no spawn_buffer
        # (the depth-refusal path already owns that case). research/sgar are
        # terminal and never counted toward the width.
        fanout_over_limit = False
        if self.max_fanout is not None and self.spawn_buffer is not None:
            counted_modes = _SPAWN_MODES
            if self.count_research_in_fanout:
                counted_modes = _SPAWN_MODES | {"research"}
            pending_spawns = sum(
                1 for mode, _payload in entries if mode in counted_modes
            )
            if pending_spawns and (
                len(self.spawn_buffer) + pending_spawns > self.max_fanout
            ):
                fanout_over_limit = True
        # Obligation-dedup guard (opt-in, default OFF). Seed the seen-set with
        # the obligations already buffered this turn (earlier ccx_spawn calls),
        # then refuse any spawn-mode entry whose obligation repeats — whether it
        # collides with the buffer or with an earlier entry in THIS call. Off ⇒
        # the set stays empty, the per-entry check below is skipped, and control
        # flow is byte-identical.
        dup_refused: list[str] = []
        seen_obligations: set[tuple[Any, ...]] = set()
        if self.dedup_spawns and self.spawn_buffer is not None:
            for r in self.spawn_buffer.snapshot():
                seen_obligations.add(
                    _spawn_obligation_key(
                        r.mode, {"goal": r.goal, "metadata": r.metadata}
                    )
                )
        for i, (mode, payload) in enumerate(entries):
            if mode == "research":
                if self.research_buffer is None:
                    missing_buffer.append(f"spawns[{i}] mode=research")
                    continue
                if self.count_research_in_fanout and fanout_over_limit:
                    # Opt-in: research counts toward (and is refused by) the
                    # width cap, mirroring the spawn-mode guard below. Default
                    # off → this branch is dead → research always buffers.
                    fanout_refused.append(f"spawns[{i}] mode=research")
                    continue
                self.research_buffer.append(
                    ResearchRequest(
                        question=str(payload.get("question") or ""),
                        scope=str(payload.get("scope") or ""),
                        focus_paths=normalize_focus_paths(
                            payload.get("focus_paths")
                        ),
                        metadata=dict(payload.get("metadata") or {}),
                    )
                )
                queued.append({"mode": mode, "question": payload.get("question")})
            elif mode in _SGAR_MODES:
                if self.sgar_buffer is None:
                    missing_buffer.append(f"spawns[{i}] mode={mode}")
                    continue
                self.sgar_buffer.append(
                    SgarRequest(
                        instruction=str(payload.get("instruction") or ""),
                        metadata=dict(payload.get("metadata") or {}),
                        sequential_with_previous=(sequential and sgar_idx_in_buffer > 0),
                        mode=mode,
                    )
                )
                sgar_idx_in_buffer += 1
                queued.append({"mode": mode, "instruction": payload.get("instruction")})
            else:
                if self.spawn_buffer is None:
                    if self.spawn_unavailable_reason:
                        refused.append(f"spawns[{i}] mode={mode}")
                    else:
                        missing_buffer.append(f"spawns[{i}] mode={mode}")
                    continue
                if fanout_over_limit:
                    fanout_refused.append(f"spawns[{i}] mode={mode}")
                    continue
                if self.dedup_spawns:
                    key = _spawn_obligation_key(mode, payload)
                    if key in seen_obligations:
                        dup_refused.append(f"spawns[{i}] mode={mode}")
                        continue
                    seen_obligations.add(key)
                self.spawn_buffer.append(
                    SpawnRequest(
                        goal=str(payload.get("goal") or ""),
                        mode=mode,
                        metadata=dict(payload.get("metadata") or {}),
                        sequential_with_previous=(sequential and spawn_idx_in_buffer > 0),
                    )
                )
                spawn_idx_in_buffer += 1
                queued.append({"mode": mode, "goal": payload.get("goal")})

        if refused:
            return ToolResult(
                tool_use_id=tool_call.tool_use_id,
                tool_name=tool_call.tool_name,
                success=False,
                content=(
                    f"ccx_spawn refused ({', '.join(refused)}): "
                    f"{self.spawn_unavailable_reason}"
                ),
                data={"queued": queued, "refused": refused},
                error_code="TL1008",
            )
        if fanout_refused:
            return ToolResult(
                tool_use_id=tool_call.tool_use_id,
                tool_name=tool_call.tool_name,
                success=False,
                content=(
                    f"ccx_spawn refused ({', '.join(fanout_refused)}): per-turn "
                    f"fan-out width limit reached (already buffered "
                    f"{len(self.spawn_buffer) if self.spawn_buffer else 0}, "
                    f"max={self.max_fanout}). Spawn fewer child agents this "
                    f"turn — enqueue only the highest-value work, or complete "
                    f"more of it directly with your own tools. "
                    + (
                        "sgar entries are not affected by this limit."
                        if self.count_research_in_fanout
                        else "research/sgar entries are not affected by this limit."
                    )
                ),
                data={
                    "queued": queued,
                    "fanout_refused": fanout_refused,
                    "max_fanout": self.max_fanout,
                },
                error_code="TL1009",
            )
        if dup_refused:
            return ToolResult(
                tool_use_id=tool_call.tool_use_id,
                tool_name=tool_call.tool_name,
                success=False,
                content=(
                    f"ccx_spawn refused ({', '.join(dup_refused)}): duplicate "
                    f"obligation — a spawn with the same goal + mode (+ contract "
                    f"checks) is already queued this turn. Redundant re-spawns "
                    f"(e.g. delegating the same sub-task twice to 'double-check') "
                    f"are dropped; vary the goal for a genuinely different "
                    f"sub-task, or complete the re-check yourself. Non-duplicate "
                    f"entries on this call were still queued."
                ),
                data={"queued": queued, "dup_refused": dup_refused},
                error_code="TL1010",
            )
        if missing_buffer:
            return ToolResult(
                tool_use_id=tool_call.tool_use_id,
                tool_name=tool_call.tool_name,
                success=False,
                content=(
                    "ccx_spawn: the following entries could not be buffered "
                    "because their buffer was not provided to this tool: "
                    + ", ".join(missing_buffer)
                ),
                data={"queued": queued, "skipped": missing_buffer},
                error_code="TL1004",
            )
        return ToolResult(
            tool_use_id=tool_call.tool_use_id,
            tool_name=tool_call.tool_name,
            success=True,
            content=(
                f"Queued {len(queued)} entry/entries; they will run after "
                f"this turn finishes."
            ),
            data={"queued": queued},
        )


def make_ccx_unified_tool(
    *,
    spawn_buffer: SpawnBuffer | None = None,
    research_buffer: ResearchBuffer | None = None,
    sgar_buffer: SgarBuffer | None = None,
    spawn_unavailable_reason: str | None = None,
    max_fanout: int | None = None,
    dedup_spawns: bool = False,
) -> CcxUnifiedTool:
    """Factory returning the unified ccx tool wired to the given buffers."""
    return CcxUnifiedTool(
        spawn_buffer=spawn_buffer,
        research_buffer=research_buffer,
        sgar_buffer=sgar_buffer,
        spawn_unavailable_reason=spawn_unavailable_reason,
        max_fanout=max_fanout,
        dedup_spawns=dedup_spawns,
    )


__all__ = [
    "CcxUnifiedTool",
    "make_ccx_unified_tool",
]
