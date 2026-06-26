from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

from core.utils.config_setting import Config
from core.utils.prompt_language import DEFAULT_PROMPT_LANGUAGE, normalize_prompt_language

from .errors import ConfigError


# Agent modes cc itself implements.
_CC_NATIVE_AGENT_MODES = frozenset(
    {"", "plan", "spec", "agent", "ask", "doc", "structured"}
)
# ccx Stage-Governed Agent Runtime modes. CCConfig is shared between cc and
# ccx; the ccx layer (core.ccx.api) is what actually dispatches these, but the
# config validation lives here, so the whitelist must admit them. This is a
# deliberately-retained cc<->ccx seam (see the cc-ccx rearchitecture decision):
# true ccx-side registration cannot be byte-equivalent because cc must not
# import ccx, so cc would reject these modes whenever ccx is not loaded.
_CCX_AGENT_MODES = frozenset({"blueprint", "sgar", "sgarx", "goal"})
_VALID_AGENT_MODES = _CC_NATIVE_AGENT_MODES | _CCX_AGENT_MODES


def _validate_positive(name: str, value: int | float | None, *, optional: bool = False) -> None:
    """Raise CF1004 when ``value`` is not positive.

    ``optional=True`` mirrors the "must be positive when provided" guards:
    ``None`` is accepted and the message gains the " when provided" suffix.
    """

    if optional and value is None:
        return
    if value <= 0:
        suffix = " when provided" if optional else ""
        raise ConfigError(f"{name} must be positive{suffix}.", error_code="CF1004")


@dataclass(slots=True)
class CCConfig:
    prompt_language: str = DEFAULT_PROMPT_LANGUAGE
    default_llm_client: str = "SimpleDeepSeekClientReasoning"
    # Optional purpose -> LLMFactory client-name override. Empty (default)
    # routes every call through ``default_llm_client``. See
    # ``core/ccx/docs/role_based_llm_routing.md``. Names must match
    # ``LLMFactory.list_available_llms()`` or ``get_client`` raises.
    role_llm_clients: dict[str, str] = field(default_factory=dict)
    # Purposes for which ``DefaultLLMClientProvider`` enables API-level
    # JSON mode (``response_format={"type":"json_object"}``) on clients
    # that ``supports_structured_output``. Clients that don't support it
    # silently degrade to prompt-level discipline (Layer 1). See
    # ``core/ccx/docs/role_based_llm_routing.md`` §7.
    #
    # Each purpose in the default tuple was audited to confirm its
    # prompt strings contain the literal "JSON" keyword (DeepSeek and
    # most OpenAI-compatible APIs reject ``response_format`` when the
    # prompt doesn't mention "JSON", flooding whitespace instead). When
    # adding more purposes per project via ``CCConfig`` JSON config,
    # verify the prompt has "JSON" before opting in.
    json_mode_purposes: tuple[str, ...] = (
        "watch_phase0",
        "plan",
        "spec",
        "doc_decompose",
        "doc_prose_to_json",
        "doc_investigate_lite",
        "watch_analyze_lite",
        "research",
        "agent",
        "structured_flow.planning",
        "structured_flow_planning",
        "swarm.goal_planner",
    )
    permission_mode: str = "default"
    default_backend: str = "in_process"
    max_tool_rounds: int | None = None
    llm_request_timeout_seconds: float | None = 7200.0
    max_turn_timeout_seconds: float | None = 7200.0
    compact_soft_threshold: int = 120_000
    compact_hard_threshold: int = 160_000
    auto_compact_enabled: bool = True
    runtime_validation_enabled: bool = True
    rollback_on_runtime_failure: bool = True
    # --- Interactive post-edit verification (all default OFF; byte-equivalent
    # when off). ``run_tests_tool_enabled`` exposes the first-class ``run_tests``
    # verification tool to the model. ``auto_post_edit_verify`` runs
    # ``post_edit_verify_command`` automatically after any tool round that
    # mutated code, feeds the exit-code-gated verdict back into the loop, and
    # blocks implementation task auto-completion while the suite is red. With
    # both off the model's tool schema and the loop's control flow are
    # unchanged. ---
    run_tests_tool_enabled: bool = False
    auto_post_edit_verify: bool = False
    post_edit_verify_command: str | None = None
    post_edit_verify_timeout_ms: int = 120_000
    persist_sessions: bool = True
    concurrency_enabled: bool = True
    session_root: str = ".cc/sessions"
    runtime_root: str = ".cc/runtime"
    prompt_root: str = "core/cc/prompts"
    spec_root: str = ".cc/specs"
    spec_max_parallel_agents: int = 4
    # --- ccx run policy knobs (operator-facing). CCConfig field is primary;
    # a CCX_* env var overrides it per run (resolved in core/ccx/api.py).
    # All default to "unset" so build_runtime's existing defaults apply
    # unchanged — byte-identical when none is configured. ---
    ccx_max_cost_usd: float | None = None          # run cost cap (USD)
    ccx_max_tokens: int | None = None              # run token cap
    ccx_cost_per_1k_tokens: float | None = None    # token→cost price for cost cap
    ccx_max_spawn_fanout: int | None = None        # per-turn spawn WIDTH cap
    ccx_max_spawn_depth: int | None = None         # recursive spawn DEPTH cap
    ccx_count_research_in_fanout: bool = False      # count ccx_research toward width
    # When True, structured / tool-driving purposes route to a fast client via
    # RECOMMENDED_ROLE_LLM_CLIENTS, unless an explicit role_llm_clients entry
    # overrides them. Default False → routing is byte-identical.
    cc_use_recommended_routing: bool = False
    plan_root: str = ".cc/plans"
    execute_policy: str = "auto_execute"
    agent_mode: str = ""
    agent_runtime_event_buffer_size: int = 500
    swarm_assignment_event_buffer_size: int = 200
    swarm_retry_backoff_base_ms: int = 100
    swarm_retry_backoff_max_ms: int = 1000
    swarm_default_assignment_timeout_seconds: float = 300.0
    swarm_assignment_response_timeout_seconds: float = 300.0
    memory_enabled: bool = False
    memory_provider: str = "noop"
    memory_auto_recall: bool = True
    memory_auto_store: bool = True
    memory_store_structural_only: bool = True
    memory_structure_first: bool = False
    memory_max_prompt_hits: int = 5
    memory_prompt_summary_max_chars: int = 400
    memory_recall_char_budget: int = 4000
    memory_write_char_budget: int = 4000
    durable_runtime_enabled: bool = False
    durable_runtime_mode: str = "shadow"
    durable_runtime_root: str | None = None
    durable_recover_on_startup: bool = True
    durable_wrap_side_effect_tools: bool = True
    durable_wrap_readonly_tools: bool = False
    durable_tool_flush_outbox: bool = True
    durable_checkpoint_interval: int = 1
    durable_default_lease_ttl_seconds: int = 30
    durable_default_max_attempts: int = 1
    durable_cognition_mode: str = "deterministic"
    durable_cognition_model: str | None = None
    durable_cognition_reasoning_effort: str | None = None
    durable_emitter_threshold_events: int | None = None
    durable_emitter_overflow_token_ratio: float | None = None
    # Round B: CC main-loop emitter switch.
    # ``"auto"`` -> shadow mode (prompt section injected, candidates
    # collected & logged, store left untouched).
    # ``"on"`` -> shadow + persist via DurableRuntimeBridge emitter.
    # ``"off"`` -> no prompt section, no buffer flush.
    cc_emitter_enabled: str = "auto"
    cc_emitter_max_candidates_per_turn: int = 8
    # Optional explicit override for prompt-section injection. ``None``
    # follows ``cc_emitter_enabled`` (off-> hide, auto/on -> show).
    cc_emitter_prompt_section_enabled: bool | None = None
    # Round C: tool-side ContextAsset emitter switch. Controls whether
    # ``ToolResult.asset_candidates`` produced by the cc tool chain is
    # flushed to ``ContextAssetStoreV4`` via ``DurableToolBridge``.
    # Three states match ``cc_emitter_enabled`` (``auto`` shadow / ``on``
    # persist / ``off`` skip). Independent knob from the main-loop
    # emitter so ops can stage rollouts. Under ``deterministic`` cognition
    # mode this knob (and ``cc_emitter_enabled``) is forced to ``"off"`` at
    # engine-build time -- see ``engine_factory`` where it is clamped when
    # ``mode_profile.requires_emitter()`` is false (NOT in ``__post_init__``,
    # so the stored config value is left as configured until then).
    tool_emitter_enabled: str = "auto"
    tool_emitter_max_candidates_per_call: int = 4
    # Reserved for follow-up phases that may inject a tool-side emit
    # protocol section into the system prompt (today the candidates are
    # heuristically populated by tool authors, no LLM training needed).
    tool_emitter_prompt_section_enabled: bool | None = None

    # P8-核心: cc-side recall of previously-emitted ContextAssetV4 rows.
    # Three states mirror ``cc_emitter_enabled``:
    # ``"off"``   -> no recall (default; preserves pre-P8 behaviour)
    # ``"shadow"`` -> recall + log + skip prompt injection (for staged
    #                rollout / A/B observation)
    # ``"on"``    -> recall + inject as a ``# Recalled Context Assets``
    #                section in the system prompt
    # ``deterministic`` cognition mode forces this to ``"off"`` regardless
    # of user config (no asset_store -> nothing to recall).
    cc_consume_enabled: str = "off"
    # Maximum number of recalled assets surfaced into the prompt section.
    cc_consume_max_assets: int = 5
    # Hard cap on the recalled-section character length.
    cc_consume_max_chars: int = 3000
    # Round-2 M1: ``importance`` threshold (0-100) above which a recalled
    # asset is *pinned*: it is reserved a slot ahead of
    # ``cc_consume_max_assets`` so high-priority observations / decisions
    # / verifications cannot be silently evicted by lower-priority rows.
    # Set to ``None`` to disable the policy and fall back to the
    # straight ``importance desc -> [:max_assets]`` selection.
    cc_consume_pin_threshold: int | None = 90

    def __post_init__(self) -> None:
        if self.agent_mode not in _VALID_AGENT_MODES:
            raise ConfigError(
                f"Unsupported agent_mode: {self.agent_mode}",
                error_code="CF1003",
            )
        self.prompt_language = normalize_prompt_language(self.prompt_language)
        if self.default_backend not in {"in_process", "local_subprocess", "remote_backend"}:
            raise ConfigError(
                f"Unsupported default_backend: {self.default_backend}",
                error_code="CF1003",
            )
        if self.compact_soft_threshold <= 0 or self.compact_hard_threshold <= 0:
            raise ConfigError("Compact thresholds must be positive.", error_code="CF1004")
        if self.compact_soft_threshold > self.compact_hard_threshold:
            raise ConfigError(
                "compact_soft_threshold must be <= compact_hard_threshold",
                error_code="CF1004",
            )
        _validate_positive("max_tool_rounds", self.max_tool_rounds, optional=True)
        _validate_positive(
            "llm_request_timeout_seconds", self.llm_request_timeout_seconds, optional=True
        )
        _validate_positive(
            "max_turn_timeout_seconds", self.max_turn_timeout_seconds, optional=True
        )
        _validate_positive("spec_max_parallel_agents", self.spec_max_parallel_agents)
        if self.execute_policy not in {"approval_required", "auto_execute"}:
            raise ConfigError(
                f"Unsupported execute_policy: {self.execute_policy}",
                error_code="CF1003",
            )
        _validate_positive(
            "agent_runtime_event_buffer_size", self.agent_runtime_event_buffer_size
        )
        _validate_positive(
            "swarm_assignment_event_buffer_size", self.swarm_assignment_event_buffer_size
        )
        if self.swarm_retry_backoff_base_ms < 0:
            raise ConfigError("swarm_retry_backoff_base_ms must be >= 0.", error_code="CF1004")
        if self.swarm_retry_backoff_max_ms < self.swarm_retry_backoff_base_ms:
            raise ConfigError(
                "swarm_retry_backoff_max_ms must be >= swarm_retry_backoff_base_ms",
                error_code="CF1004",
            )
        _validate_positive(
            "swarm_default_assignment_timeout_seconds",
            self.swarm_default_assignment_timeout_seconds,
        )
        _validate_positive(
            "swarm_assignment_response_timeout_seconds",
            self.swarm_assignment_response_timeout_seconds,
        )
        if not str(self.memory_provider or "").strip():
            raise ConfigError("memory_provider must not be empty.", error_code="CF1004")
        _validate_positive("memory_max_prompt_hits", self.memory_max_prompt_hits)
        _validate_positive(
            "memory_prompt_summary_max_chars", self.memory_prompt_summary_max_chars
        )
        _validate_positive("memory_recall_char_budget", self.memory_recall_char_budget)
        _validate_positive("memory_write_char_budget", self.memory_write_char_budget)
        if self.durable_runtime_mode not in {"shadow", "tools", "tasks", "turns"}:
            raise ConfigError(
                f"Unsupported durable_runtime_mode: {self.durable_runtime_mode}",
                error_code="CF1003",
            )
        _validate_positive("durable_checkpoint_interval", self.durable_checkpoint_interval)
        _validate_positive(
            "durable_default_lease_ttl_seconds", self.durable_default_lease_ttl_seconds
        )
        _validate_positive(
            "durable_default_max_attempts", self.durable_default_max_attempts
        )
        if self.durable_cognition_mode not in {"full", "half", "simple", "deterministic"}:
            raise ConfigError(
                f"Unsupported durable_cognition_mode: {self.durable_cognition_mode}",
                error_code="CF1003",
            )
        _validate_positive(
            "durable_emitter_threshold_events",
            self.durable_emitter_threshold_events,
            optional=True,
        )
        if self.durable_emitter_overflow_token_ratio is not None and not (
            0.0 < self.durable_emitter_overflow_token_ratio <= 1.0
        ):
            raise ConfigError(
                "durable_emitter_overflow_token_ratio must be in (0.0, 1.0] when provided.",
                error_code="CF1004",
            )
        normalized_emitter = str(self.cc_emitter_enabled or "auto").strip().lower()
        if normalized_emitter not in {"auto", "on", "off"}:
            raise ConfigError(
                "cc_emitter_enabled must be one of 'auto' | 'on' | 'off'.",
                error_code="CF1003",
            )
        self.cc_emitter_enabled = normalized_emitter
        _validate_positive(
            "cc_emitter_max_candidates_per_turn", self.cc_emitter_max_candidates_per_turn
        )
        normalized_tool_emitter = str(self.tool_emitter_enabled or "auto").strip().lower()
        if normalized_tool_emitter not in {"auto", "on", "off"}:
            raise ConfigError(
                "tool_emitter_enabled must be one of 'auto' | 'on' | 'off'.",
                error_code="CF1003",
            )
        self.tool_emitter_enabled = normalized_tool_emitter
        # P8-核心: cc_consume normalisation + deterministic-mode clamp.
        normalized_cc_consume = str(
            self.cc_consume_enabled or "off"
        ).strip().lower()
        # ``auto`` is accepted as an alias for ``shadow`` so the same
        # ``cc_emitter_enabled``-style spelling works on both knobs.
        if normalized_cc_consume == "auto":
            normalized_cc_consume = "shadow"
        if normalized_cc_consume not in {"off", "shadow", "on"}:
            raise ConfigError(
                "cc_consume_enabled must be one of 'off' | 'shadow' | 'on'.",
                error_code="CF1003",
            )
        if self.durable_cognition_mode == "deterministic" and normalized_cc_consume != "off":
            normalized_cc_consume = "off"
        self.cc_consume_enabled = normalized_cc_consume
        _validate_positive("cc_consume_max_assets", self.cc_consume_max_assets)
        _validate_positive("cc_consume_max_chars", self.cc_consume_max_chars)
        if self.cc_consume_pin_threshold is not None and not (
            0 <= int(self.cc_consume_pin_threshold) <= 100
        ):
            raise ConfigError(
                "cc_consume_pin_threshold must be in [0, 100] or None.",
                error_code="CF1004",
            )
        _validate_positive(
            "tool_emitter_max_candidates_per_call",
            self.tool_emitter_max_candidates_per_call,
        )
        if not isinstance(self.role_llm_clients, dict):
            raise ConfigError(
                "role_llm_clients must be a dict of purpose -> client_name strings.",
                error_code="CF1003",
            )
        for key, value in self.role_llm_clients.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ConfigError(
                    "role_llm_clients keys and values must both be strings.",
                    error_code="CF1003",
                )
        if not isinstance(self.json_mode_purposes, (tuple, list)):
            raise ConfigError(
                "json_mode_purposes must be a tuple of strings.",
                error_code="CF1003",
            )
        for item in self.json_mode_purposes:
            if not isinstance(item, str):
                raise ConfigError(
                    "json_mode_purposes items must be strings.",
                    error_code="CF1003",
                )
        self.json_mode_purposes = tuple(self.json_mode_purposes)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def session_root_path(self, cwd: str | Path | None = None) -> Path:
        base = Path(cwd) if cwd is not None else Path.cwd()
        return (base / self.session_root).resolve()

    def runtime_root_path(self, cwd: str | Path | None = None) -> Path:
        base = Path(cwd) if cwd is not None else Path.cwd()
        return (base / self.runtime_root).resolve()

    def prompt_root_path(self, cwd: str | Path | None = None) -> Path:
        base = Path(cwd) if cwd is not None else Path.cwd()
        root = Path(self.prompt_root)
        if root.is_absolute():
            return root
        candidate = (base / root).resolve()
        if candidate.exists():
            return candidate
        package_candidate = (Path(__file__).resolve().parent / "prompts").resolve()
        if package_candidate.exists():
            return package_candidate
        return candidate

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None = None) -> "CCConfig":
        payload = dict(data or {})
        if "prompt_language" in payload:
            payload["prompt_language"] = normalize_prompt_language(payload["prompt_language"])
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in payload.items() if k in known_fields}
        unknown = set(payload.keys()) - known_fields
        if unknown:
            import warnings
            warnings.warn(
                f"Ignored unknown config keys: {', '.join(sorted(unknown))}",
                UserWarning,
                stacklevel=2,
            )
        return cls(**filtered)


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _coerce_int(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Expected integer value, got {value!r}", error_code="CF1004") from exc


def _coerce_optional_int(value: Any, default: int | None) -> int | None:
    if value is None or value == "":
        return default
    text = str(value).strip().lower()
    if text in {"none", "null", "unlimited", "infinite", "inf"}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Expected integer or null value, got {value!r}", error_code="CF1004") from exc


def _coerce_float(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Expected float value, got {value!r}", error_code="CF1004") from exc


def _coerce_optional_float(value: Any, default: float | None) -> float | None:
    if value is None or value == "":
        return default
    text = str(value).strip().lower()
    if text in {"none", "null", "unlimited", "infinite", "inf"}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Expected float or null value, got {value!r}", error_code="CF1004") from exc


def load_cc_config(
    *,
    cli_overrides: Mapping[str, Any] | None = None,
    session_overrides: Mapping[str, Any] | None = None,
    project_config_path: str | Path | None = None,
    user_config_path: str | Path | None = None,
) -> CCConfig:
    values: dict[str, Any] = {}
    defaults = CCConfig()

    if user_config_path:
        values.update(_load_json_file(user_config_path))
    if project_config_path:
        values.update(_load_json_file(project_config_path))
    if session_overrides:
        values.update(dict(session_overrides))
    if cli_overrides:
        values.update(dict(cli_overrides))

    settings = Config()
    values.setdefault("prompt_language", settings.get("prompt_language") or DEFAULT_PROMPT_LANGUAGE)
    values.setdefault(
        "default_llm_client",
        settings.get("cc_default_llm_client") or settings.get("llm_api") or defaults.default_llm_client,
    )
    values.setdefault("permission_mode", settings.get("cc_permission_mode") or defaults.permission_mode)
    values.setdefault("default_backend", settings.get("cc_default_backend") or defaults.default_backend)
    values.setdefault(
        "max_tool_rounds",
        _coerce_optional_int(settings.get("cc_max_tool_rounds"), defaults.max_tool_rounds),
    )
    values.setdefault(
        "llm_request_timeout_seconds",
        _coerce_optional_float(
            settings.get("cc_llm_request_timeout_seconds"),
            defaults.llm_request_timeout_seconds,
        ),
    )
    values.setdefault(
        "max_turn_timeout_seconds",
        _coerce_optional_float(settings.get("cc_max_turn_timeout_seconds"), defaults.max_turn_timeout_seconds),
    )
    values.setdefault(
        "compact_soft_threshold",
        _coerce_int(settings.get("cc_compact_soft_threshold"), defaults.compact_soft_threshold),
    )
    values.setdefault(
        "compact_hard_threshold",
        _coerce_int(settings.get("cc_compact_hard_threshold"), defaults.compact_hard_threshold),
    )
    values.setdefault(
        "auto_compact_enabled",
        _coerce_bool(settings.get("cc_auto_compact_enabled"), defaults.auto_compact_enabled),
    )
    values.setdefault(
        "runtime_validation_enabled",
        _coerce_bool(settings.get("cc_runtime_validation_enabled"), defaults.runtime_validation_enabled),
    )
    values.setdefault(
        "rollback_on_runtime_failure",
        _coerce_bool(settings.get("cc_rollback_on_runtime_failure"), defaults.rollback_on_runtime_failure),
    )
    values.setdefault(
        "run_tests_tool_enabled",
        _coerce_bool(settings.get("cc_run_tests_tool_enabled"), defaults.run_tests_tool_enabled),
    )
    values.setdefault(
        "auto_post_edit_verify",
        _coerce_bool(settings.get("cc_auto_post_edit_verify"), defaults.auto_post_edit_verify),
    )
    values.setdefault(
        "post_edit_verify_command",
        settings.get("cc_post_edit_verify_command") or defaults.post_edit_verify_command,
    )
    values.setdefault(
        "post_edit_verify_timeout_ms",
        _coerce_int(settings.get("cc_post_edit_verify_timeout_ms"), defaults.post_edit_verify_timeout_ms),
    )
    values.setdefault(
        "persist_sessions",
        _coerce_bool(settings.get("cc_persist_sessions"), defaults.persist_sessions),
    )
    values.setdefault(
        "concurrency_enabled",
        _coerce_bool(settings.get("cc_concurrency_enabled"), defaults.concurrency_enabled),
    )
    values.setdefault("spec_root", settings.get("cc_spec_root") or defaults.spec_root)
    values.setdefault(
        "spec_max_parallel_agents",
        _coerce_int(settings.get("cc_spec_max_parallel_agents"), defaults.spec_max_parallel_agents),
    )
    values.setdefault(
        "ccx_max_cost_usd",
        _coerce_optional_float(settings.get("cc_ccx_max_cost_usd"), defaults.ccx_max_cost_usd),
    )
    values.setdefault(
        "ccx_max_tokens",
        _coerce_optional_int(settings.get("cc_ccx_max_tokens"), defaults.ccx_max_tokens),
    )
    values.setdefault(
        "ccx_cost_per_1k_tokens",
        _coerce_optional_float(settings.get("cc_ccx_cost_per_1k_tokens"), defaults.ccx_cost_per_1k_tokens),
    )
    values.setdefault(
        "ccx_max_spawn_fanout",
        _coerce_optional_int(settings.get("cc_ccx_max_spawn_fanout"), defaults.ccx_max_spawn_fanout),
    )
    values.setdefault(
        "ccx_max_spawn_depth",
        _coerce_optional_int(settings.get("cc_ccx_max_spawn_depth"), defaults.ccx_max_spawn_depth),
    )
    values.setdefault(
        "ccx_count_research_in_fanout",
        _coerce_bool(settings.get("cc_ccx_count_research_in_fanout"), defaults.ccx_count_research_in_fanout),
    )
    values.setdefault(
        "cc_use_recommended_routing",
        _coerce_bool(settings.get("cc_use_recommended_routing"), defaults.cc_use_recommended_routing),
    )
    values.setdefault("plan_root", settings.get("cc_plan_root") or defaults.plan_root)
    values.setdefault(
        "execute_policy",
        settings.get("cc_execute_policy") or defaults.execute_policy,
    )
    values.setdefault("agent_mode", settings.get("cc_agent_mode") or defaults.agent_mode)
    values.setdefault(
        "agent_runtime_event_buffer_size",
        _coerce_int(settings.get("cc_agent_runtime_event_buffer_size"), defaults.agent_runtime_event_buffer_size),
    )
    values.setdefault(
        "swarm_assignment_event_buffer_size",
        _coerce_int(
            settings.get("cc_swarm_assignment_event_buffer_size"),
            defaults.swarm_assignment_event_buffer_size,
        ),
    )
    values.setdefault(
        "swarm_retry_backoff_base_ms",
        _coerce_int(settings.get("cc_swarm_retry_backoff_base_ms"), defaults.swarm_retry_backoff_base_ms),
    )
    values.setdefault(
        "swarm_retry_backoff_max_ms",
        _coerce_int(settings.get("cc_swarm_retry_backoff_max_ms"), defaults.swarm_retry_backoff_max_ms),
    )
    values.setdefault(
        "swarm_default_assignment_timeout_seconds",
        _coerce_float(
            settings.get("cc_swarm_default_assignment_timeout_seconds"),
            defaults.swarm_default_assignment_timeout_seconds,
        ),
    )
    values.setdefault(
        "swarm_assignment_response_timeout_seconds",
        _coerce_float(
            settings.get("cc_swarm_assignment_response_timeout_seconds"),
            defaults.swarm_assignment_response_timeout_seconds,
        ),
    )
    values.setdefault(
        "memory_enabled",
        _coerce_bool(settings.get("cc_memory_enabled"), defaults.memory_enabled),
    )
    values.setdefault("memory_provider", settings.get("cc_memory_provider") or defaults.memory_provider)
    values.setdefault(
        "memory_auto_recall",
        _coerce_bool(settings.get("cc_memory_auto_recall"), defaults.memory_auto_recall),
    )
    values.setdefault(
        "memory_auto_store",
        _coerce_bool(settings.get("cc_memory_auto_store"), defaults.memory_auto_store),
    )
    values.setdefault(
        "memory_store_structural_only",
        _coerce_bool(
            settings.get("cc_memory_store_structural_only"),
            defaults.memory_store_structural_only,
        ),
    )
    values.setdefault(
        "memory_structure_first",
        _coerce_bool(
            settings.get("cc_memory_structure_first"),
            defaults.memory_structure_first,
        ),
    )
    values.setdefault(
        "memory_max_prompt_hits",
        _coerce_int(settings.get("cc_memory_max_prompt_hits"), defaults.memory_max_prompt_hits),
    )
    values.setdefault(
        "memory_prompt_summary_max_chars",
        _coerce_int(
            settings.get("cc_memory_prompt_summary_max_chars"),
            defaults.memory_prompt_summary_max_chars,
        ),
    )
    values.setdefault(
        "memory_recall_char_budget",
        _coerce_int(
            settings.get("cc_memory_recall_char_budget"),
            defaults.memory_recall_char_budget,
        ),
    )
    values.setdefault(
        "memory_write_char_budget",
        _coerce_int(
            settings.get("cc_memory_write_char_budget"),
            defaults.memory_write_char_budget,
        ),
    )
    values.setdefault(
        "durable_runtime_enabled",
        _coerce_bool(settings.get("cc_durable_runtime_enabled"), defaults.durable_runtime_enabled),
    )
    values.setdefault(
        "durable_runtime_mode",
        settings.get("cc_durable_runtime_mode") or defaults.durable_runtime_mode,
    )
    values.setdefault(
        "durable_runtime_root",
        settings.get("cc_durable_runtime_root") or defaults.durable_runtime_root,
    )
    values.setdefault(
        "durable_recover_on_startup",
        _coerce_bool(settings.get("cc_durable_recover_on_startup"), defaults.durable_recover_on_startup),
    )
    values.setdefault(
        "durable_wrap_side_effect_tools",
        _coerce_bool(settings.get("cc_durable_wrap_side_effect_tools"), defaults.durable_wrap_side_effect_tools),
    )
    values.setdefault(
        "durable_wrap_readonly_tools",
        _coerce_bool(settings.get("cc_durable_wrap_readonly_tools"), defaults.durable_wrap_readonly_tools),
    )
    values.setdefault(
        "durable_tool_flush_outbox",
        _coerce_bool(settings.get("cc_durable_tool_flush_outbox"), defaults.durable_tool_flush_outbox),
    )
    values.setdefault(
        "durable_checkpoint_interval",
        _coerce_int(settings.get("cc_durable_checkpoint_interval"), defaults.durable_checkpoint_interval),
    )
    values.setdefault(
        "durable_default_lease_ttl_seconds",
        _coerce_int(
            settings.get("cc_durable_default_lease_ttl_seconds"),
            defaults.durable_default_lease_ttl_seconds,
        ),
    )
    values.setdefault(
        "durable_default_max_attempts",
        _coerce_int(settings.get("cc_durable_default_max_attempts"), defaults.durable_default_max_attempts),
    )
    values.setdefault(
        "durable_cognition_mode",
        settings.get("cc_durable_cognition_mode") or defaults.durable_cognition_mode,
    )
    values.setdefault(
        "durable_cognition_model",
        settings.get("cc_durable_cognition_model") or defaults.durable_cognition_model,
    )
    values.setdefault(
        "durable_cognition_reasoning_effort",
        settings.get("cc_durable_cognition_reasoning_effort")
        or defaults.durable_cognition_reasoning_effort,
    )
    raw_threshold = settings.get("cc_durable_emitter_threshold_events")
    values.setdefault(
        "durable_emitter_threshold_events",
        _coerce_int(raw_threshold, defaults.durable_emitter_threshold_events or 0)
        if raw_threshold not in (None, "")
        else defaults.durable_emitter_threshold_events,
    )
    raw_overflow = settings.get("cc_durable_emitter_overflow_token_ratio")
    values.setdefault(
        "durable_emitter_overflow_token_ratio",
        _coerce_float(raw_overflow, defaults.durable_emitter_overflow_token_ratio or 0.0)
        if raw_overflow not in (None, "")
        else defaults.durable_emitter_overflow_token_ratio,
    )
    values.setdefault(
        "cc_emitter_enabled",
        settings.get("cc_emitter_enabled") or defaults.cc_emitter_enabled,
    )
    values.setdefault(
        "cc_emitter_max_candidates_per_turn",
        _coerce_int(
            settings.get("cc_emitter_max_candidates_per_turn"),
            defaults.cc_emitter_max_candidates_per_turn,
        ),
    )
    raw_section = settings.get("cc_emitter_prompt_section_enabled")
    if raw_section in (None, ""):
        values.setdefault(
            "cc_emitter_prompt_section_enabled",
            defaults.cc_emitter_prompt_section_enabled,
        )
    else:
        values.setdefault(
            "cc_emitter_prompt_section_enabled",
            _coerce_bool(raw_section, False),
        )
    values.setdefault(
        "tool_emitter_enabled",
        settings.get("tool_emitter_enabled") or defaults.tool_emitter_enabled,
    )
    values.setdefault(
        "tool_emitter_max_candidates_per_call",
        _coerce_int(
            settings.get("tool_emitter_max_candidates_per_call"),
            defaults.tool_emitter_max_candidates_per_call,
        ),
    )
    raw_tool_section = settings.get("tool_emitter_prompt_section_enabled")
    if raw_tool_section in (None, ""):
        values.setdefault(
            "tool_emitter_prompt_section_enabled",
            defaults.tool_emitter_prompt_section_enabled,
        )
    else:
        values.setdefault(
            "tool_emitter_prompt_section_enabled",
            _coerce_bool(raw_tool_section, False),
        )
    values.setdefault(
        "cc_consume_enabled",
        settings.get("cc_consume_enabled") or defaults.cc_consume_enabled,
    )
    values.setdefault(
        "cc_consume_max_assets",
        _coerce_int(
            settings.get("cc_consume_max_assets"),
            defaults.cc_consume_max_assets,
        ),
    )
    values.setdefault(
        "cc_consume_max_chars",
        _coerce_int(
            settings.get("cc_consume_max_chars"),
            defaults.cc_consume_max_chars,
        ),
    )
    raw_pin_threshold = settings.get("cc_consume_pin_threshold")
    if raw_pin_threshold in (None, ""):
        values.setdefault(
            "cc_consume_pin_threshold", defaults.cc_consume_pin_threshold
        )
    else:
        # Allow explicit ``"none"``/``"off"`` / ``"-1"`` to disable.
        flag = str(raw_pin_threshold).strip().lower()
        if flag in {"none", "off", "disable", "disabled", "-1"}:
            values.setdefault("cc_consume_pin_threshold", None)
        else:
            values.setdefault(
                "cc_consume_pin_threshold",
                _coerce_int(
                    raw_pin_threshold,
                    defaults.cc_consume_pin_threshold or 90,
                ),
            )
    return CCConfig.from_mapping(values)


def _load_json_file(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid config JSON: {file_path}", error_code="CF1001") from exc
