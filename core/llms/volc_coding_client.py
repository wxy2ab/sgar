from typing import Any, Dict, List, Optional

from ..utils.config_setting import Config
from ._llm_api_client import LLMApiClient
from .openai_chat_client import OpenAIChatClient


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
    DEFAULT_MODEL = "doubao-seed-code-preview-251028"

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
            api_key, base_url, max_tokens=16000, enable_thinking=True,
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

    def tool_invoke(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Mirror the base classes' "no forced-JSON-mode on tool paths" rule
        # (tools + forced JSON output is an ambiguous combination for
        # OpenAI-compatible APIs — see _create_chat_completion's `not tools`
        # gate on response_format). self.extra_body["format"] must not leak
        # into tool-calling requests just because a prior call on this same
        # instance turned on JSON mode.
        saved_format = self.extra_body.pop("format", None)
        try:
            return super().tool_invoke(messages, tools)
        finally:
            if saved_format is not None:
                self.extra_body["format"] = saved_format
