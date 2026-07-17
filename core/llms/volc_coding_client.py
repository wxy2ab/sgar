import os
from typing import Any, Dict, List, Optional

from tenacity import retry, stop_after_attempt, wait_fixed

from ..utils.config_setting import Config
from ..utils.log import logger
from ._empty_content_retry import is_blank
from ._llm_api_client import LLMApiClient
from .openai_chat_client import OpenAIChatClient


def _blank_and_no_tool_calls(resp: Any) -> bool:
    """A tool_invoke response is *pathologically* empty when its visible content
    is blank AND it carries no tool_calls (the model finished the turn but said
    nothing). Empty content *with* tool_calls is legitimate and must not retry.
    """
    if not isinstance(resp, dict):
        return False
    content = resp.get("content")
    if content is not None and str(content).strip():
        return False
    return not resp.get("tool_calls")


def _resolve_enable_thinking() -> bool:
    """doubao-seed-code's hidden ``reasoning_content`` mode.

    Defaults to ``True`` (the historical behavior for every existing caller).
    Structured / tool-loop workloads — notably ccx doc-mode investigators that
    must emit strict JSON after a tool loop — are hurt by it: the reasoning
    trace can consume an entire turn and return empty visible content
    (``status=empty``) or narrate-then-truncate the JSON (``status=unparseable``),
    and it roughly triples per-call latency. An operator can force non-thinking
    for such a run via ``VOLC_CODING_ENABLE_THINKING=0`` (also accepts
    ``false`` / ``no`` / ``off``), which yields reliable strict JSON and much
    lower latency without swapping the client class or model.
    """
    raw = os.getenv("VOLC_CODING_ENABLE_THINKING")
    if raw is not None and raw.strip().lower() in {"0", "false", "no", "off"}:
        return False
    return True


def _resolve_reasoning_effort() -> str:
    """Reasoning-effort budget for the coding endpoint. Defaults to ``"high"``
    (historical behavior). This is the real lever for structured / tool-loop
    workloads: at ``"high"``/``"low"`` the hidden reasoning trace on
    ark-code-latest can run for minutes per call and, in a tool loop, eat the
    final turn (empty / unparseable JSON); ``"minimal"`` bounds the trace to
    ~zero, giving ~5s reliable strict-JSON completions. Operators set
    ``VOLC_CODING_REASONING_EFFORT=minimal`` (or low/medium/high) for such runs.
    """
    raw = os.getenv("VOLC_CODING_REASONING_EFFORT")
    if raw and raw.strip():
        return raw.strip().lower()
    return "high"


class VolcCodingClient(LLMApiClient):
    pass



class VolcCodingClient(OpenAIChatClient):
    # The ark-code-latest model rejects the STANDARD OpenAI
    # response_format={"type": "json_object"} request field outright: 400
    # InvalidParameter — "json_object is not supported by this model"
    # (confirmed live, bypassing any client wrapper — a model/endpoint
    # restriction, not a client bug). It DOES support forced JSON output via
    # a Volc-specific extra_body field named "format" with the same
    # {"type": "json_object"} shape — also confirmed live. set_response_format
    # (below) routes into that field instead of the standard one.
    supports_structured_output = True

    #: model="auto" hands routing to Volc's own "intelligent scheduling"
    #: gateway, which picks the backend model (observed: glm-5.2) per
    #: request. Confirmed live: that routing layer itself adds ~8-35s of
    #: overhead on top of the actual model call, for IDENTICAL tiny
    #: (~124-token) requests with reasoning_tokens=0 (i.e. not spent on
    #: hidden thinking — pure gateway/queueing latency). Targeting a
    #: concrete backend model directly bypasses that layer entirely: the
    #: same requests complete in 1.9-4.5s, matching SimpleDeepSeekClient.
    #: doubao-seed-code-preview-251028 is Volc's own coding-tuned model for
    #: this coding-plan endpoint — the natural default for a "coding" client.
    DEFAULT_MODEL = "ark-code-latest"

    def __init__(self, model: str = ""):
        base_url = "https://ark.cn-beijing.volces.com/api/coding/v3"
        config = Config()

        api_key = config.get_with_fallback(("ark_coding_key", "volcengine_api_key"))
        # This endpoint's implicit completion budget (no max_tokens sent) is
        # ~1000 tokens. With enable_thinking=True + reasoning_effort="high"
        # the hidden reasoning trace alone can consume all 1000 of those,
        # leaving 0 tokens for the visible answer (finish_reason="length",
        # content="") — confirmed against the live API. 16000 leaves ample
        # room for both a deep reasoning trace and a full-length answer/tool
        # call. Unlike MoonShotClient, OpenAIChatClient wires max_tokens live
        # into every request, so passing it here is enough.
        resolved_model = config.resolve_value(
            model, ("volc_coding_client_model",), self.DEFAULT_MODEL,
        )
        super().__init__(
            api_key, base_url, max_tokens=16000,
            enable_thinking=_resolve_enable_thinking(),
            reasoning_effort=_resolve_reasoning_effort(),
            model=resolved_model,
            # doubao-seed-code's documented spec: 256k max context, 224k max
            # input. Used by cc's query_loop (getattr(..., "context_window_
            # tokens", None)) for token-budget-aware compaction instead of a
            # fixed message-count trim — previously DeepSeek-only since only
            # SimpleDeepSeekClient set this attribute, but the mechanism
            # itself is client-agnostic. Calibrated for DEFAULT_MODEL; if
            # ``model``/``volc_coding_client_model`` overrides to a different
            # backend with a different real window, recalibrate this too.
            context_window_tokens=(
                224_000 if resolved_model == self.DEFAULT_MODEL else None
            ),
        )

    def set_response_format(self, fmt: Optional[Dict[str, Any]]) -> None:
        if fmt is not None and not isinstance(fmt, dict):
            raise TypeError("response_format must be a dict or None")
        # Deliberately do NOT call super()/set self._response_format: that
        # would make _create_chat_completion send the standard
        # response_format field, which this endpoint rejects (see class
        # docstring). Route through extra_body["format"] instead.
        if fmt is None:
            self.extra_body.pop("format", None)
        else:
            self.extra_body["format"] = fmt

    def _create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        is_stream: bool,
        tools: List[Dict[str, Any]] = None,
        raw_response: bool = False,
    ):
        """Prefer streaming for durable text/JSON against ark idle-close.

        The coding endpoint idle-closes a *non-stream* request that stays silent
        for ~60s while the model works. Story Insight architect packets routinely
        exceed that window, so ``one_chat``'s non-stream path used to hang and
        burn the base client's 12 connection retries. Tool calls already stream
        via ``_streaming_tool_completion``; text/JSON now matches that contract.
        ``tools`` / ``raw_response`` keep the base non-stream shape for callers
        that inspect Completion objects directly.
        """

        if tools or raw_response:
            return super()._create_chat_completion(
                messages, is_stream, tools, raw_response
            )
        if is_stream:
            return super()._create_chat_completion(
                messages, True, tools, raw_response
            )
        text = self._streaming_text_completion(messages)
        retries = self._EMPTY_CONTENT_RETRIES
        attempt = 0
        while attempt < retries and is_blank(text):
            force_no_thinking = attempt == retries - 1
            logger.warning(
                "VolcCodingClient.one_chat/text: empty visible content "
                "(reasoning_effort=%s); re-issuing (attempt %d/%d%s)",
                self.reasoning_effort,
                attempt + 1,
                retries,
                ", thinking disabled" if force_no_thinking else "",
            )
            if force_no_thinking and self.enable_thinking is not False:
                prev = self.enable_thinking
                self.enable_thinking = False
                try:
                    text = self._streaming_text_completion(messages)
                finally:
                    self.enable_thinking = prev
            else:
                text = self._streaming_text_completion(messages)
            attempt += 1
        return text or ""

    @retry(stop=stop_after_attempt(4), wait=wait_fixed(5))
    def _streaming_text_completion(
        self, messages: List[Dict[str, str]]
    ) -> str:
        """Stream a plain chat completion and return the visible content."""

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": True,
            "timeout": 600,
            "stream_options": {"include_usage": True},
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        extra_body = dict(self.extra_body) if self.extra_body else {}
        if self.enable_thinking is not None:
            extra_body["enable_thinking"] = self.enable_thinking
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.verbosity is not None:
            kwargs["verbosity"] = self.verbosity
        if extra_body:
            kwargs["extra_body"] = extra_body
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.presence_penalty is not None:
            kwargs["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty is not None:
            kwargs["frequency_penalty"] = self.frequency_penalty
        if self.stop is not None:
            kwargs["stop"] = self.stop
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if self.service_tier is not None:
            kwargs["service_tier"] = self.service_tier

        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        usage = None
        stream = self.client.chat.completions.create(**kwargs)
        for chunk in stream:
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            model_id = getattr(chunk, "model", None)
            if model_id:
                self.observed_backend_model = model_id
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta
            reasoning_delta = getattr(delta, "reasoning_content", None)
            if reasoning_delta:
                reasoning_parts.append(reasoning_delta)
            piece = getattr(delta, "content", None)
            if piece:
                content_parts.append(piece)
        if reasoning_parts:
            self._last_reasoning_content = "".join(reasoning_parts)
        if usage is not None:
            try:
                self._update_stats(usage)
            except Exception:  # noqa: BLE001 — stats are best-effort
                pass
        else:
            self._update_stats(None)
        return "".join(content_parts)

    def tool_invoke(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Mirror the base classes' "no forced-JSON-mode on tool paths" rule
        # (tools + forced JSON output is an ambiguous combination for
        # OpenAI-compatible APIs — see _create_chat_completion's `not tools`
        # gate on response_format). self.extra_body["format"] must not leak
        # into tool-calling requests just because a prior call on this same
        # instance turned on JSON mode.
        saved_format = self.extra_body.pop("format", None)
        try:
            result = self._streaming_tool_completion(messages, tools)
            # At reasoning_effort="high"/"low", doubao-seed-code can spend an
            # ENTIRE turn on hidden reasoning and return empty visible content
            # with NO tool_calls (finish_reason "stop") — a pathological
            # "finished but said nothing" turn. Empty AND no tool_calls is never
            # legitimate, so re-issue. The first re-issue keeps full reasoning
            # (preserve depth); the LAST forces enable_thinking=False so a
            # visible answer must appear. ``get_client`` builds a fresh client
            # per turn (core/cc/llm.py:96), so this instance is not shared across
            # threads and the temporary enable_thinking toggle is race-free.
            retries = self._TOOL_EMPTY_CONTENT_RETRIES
            attempt = 0
            while attempt < retries and _blank_and_no_tool_calls(result):
                force_no_thinking = attempt == retries - 1
                logger.warning(
                    "VolcCodingClient.tool_invoke: empty content + no tool_calls "
                    "(reasoning_effort=%s); re-issuing (attempt %d/%d%s)",
                    self.reasoning_effort, attempt + 1, retries,
                    ", thinking disabled" if force_no_thinking else "",
                )
                if force_no_thinking and self.enable_thinking is not False:
                    prev = self.enable_thinking
                    self.enable_thinking = False
                    try:
                        result = self._streaming_tool_completion(messages, tools)
                    finally:
                        self.enable_thinking = prev
                else:
                    result = self._streaming_tool_completion(messages, tools)
                attempt += 1
            return result
        finally:
            if saved_format is not None:
                self.extra_body["format"] = saved_format

    @retry(stop=stop_after_attempt(4), wait=wait_fixed(5))
    def _streaming_tool_completion(
        self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Tool-calling completion issued with ``stream=True``.

        The ark coding endpoint idle-closes a *non-stream* request that goes
        silent for ~60s while the model reasons — and at ``reasoning_effort``
        "high" the hidden reasoning trace routinely runs several minutes, so
        the base (stream=False) ``tool_invoke`` reliably fails deep-reasoning
        turns with a connection reset -> retry storm -> bridge timeout.
        Streaming makes the reasoning_content tokens flow, keeping the
        connection alive; we accumulate the visible content and the tool_call
        deltas (id / name / arguments arrive incrementally) and return the same
        normalized ``{content, tool_calls}`` shape the base produces. The
        ``@retry`` covers a genuine mid-stream drop (rare once streaming keeps
        the socket warm). Mirrors the kwargs the base ``tool_invoke`` builds.
        """
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": True,
            "timeout": 600,
            "tools": tools,
            "stream_options": {"include_usage": True},
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        extra_body = dict(self.extra_body) if self.extra_body else {}
        if self.enable_thinking is not None:
            extra_body["enable_thinking"] = self.enable_thinking
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.verbosity is not None:
            kwargs["verbosity"] = self.verbosity
        if extra_body:
            kwargs["extra_body"] = extra_body
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.presence_penalty is not None:
            kwargs["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty is not None:
            kwargs["frequency_penalty"] = self.frequency_penalty
        if self.stop is not None:
            kwargs["stop"] = self.stop
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if self.service_tier is not None:
            kwargs["service_tier"] = self.service_tier
        if self.parallel_tool_calls is not None:
            kwargs["parallel_tool_calls"] = self.parallel_tool_calls

        content_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        usage = None
        stream = self.client.chat.completions.create(**kwargs)
        for chunk in stream:
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta
            piece = getattr(delta, "content", None)
            if piece:
                content_parts.append(piece)
            deltas = getattr(delta, "tool_calls", None)
            if deltas:
                for tc in deltas:
                    idx = getattr(tc, "index", 0) or 0
                    while len(tool_calls) <= idx:
                        tool_calls.append(
                            {"id": None, "type": "function",
                             "function": {"name": "", "arguments": ""}}
                        )
                    slot = tool_calls[idx]
                    if getattr(tc, "id", None):
                        slot["id"] = tc.id
                    fn = getattr(tc, "function", None)
                    if fn is not None:
                        if getattr(fn, "name", None):
                            slot["function"]["name"] = fn.name
                        if getattr(fn, "arguments", None):
                            slot["function"]["arguments"] += fn.arguments
        if usage is not None:
            try:
                self._update_stats(usage)
            except Exception:  # noqa: BLE001 — stats are best-effort
                pass
        return self._normalize_tool_invoke_response(
            "".join(content_parts), tool_calls or None
        )

    #: Re-issues on a pathological empty-content-with-no-tool-calls turn. Kept
    #: to 2 (vs the base ``_EMPTY_CONTENT_RETRIES=3``) because each high-effort
    #: re-issue is expensive; the 2nd (final) one disables thinking and is fast.
    _TOOL_EMPTY_CONTENT_RETRIES = 2

    # NOTE on "ccx occasionally emits empty output" with this client:
    # doubao-seed-code runs with enable_thinking=True and can spend an entire
    # turn on hidden reasoning_content, returning empty visible content with
    # finish_reason="stop" (no exception -> one_chat's @retry never fires). That
    # empty propagates to ccx as an empty final_text. Setting max_tokens does NOT
    # bound the reasoning trace on this endpoint, so the token budget in the
    # class docstring cannot prevent it. Text/JSON completions stream and then
    # re-issue on empty in ``_create_chat_completion`` (mirroring the shared
    # ``_empty_content_retry`` contract); tool_invoke keeps its own empty+no-
    # tool-calls guard above.
