"""DeepSeek reasoning client (``deepseek-v4-pro`` + ``thinking=True``).

This is the default ``llm`` for ``task.py ccx``. The reasoning phase is
valuable for genuine multi-step planning, but the long *first-token gap* it
introduces is fragile over a real network: ccx issues these as non-streaming
calls, so one HTTP connection sits idle for minutes while the model thinks,
and a proxy/gateway routinely idle-closes it (connection reset / unexpected
EOF / read timeout). The whole expensive call is then retried from scratch.

To make ccx tasks complete reliably, this client runs every non-streaming
request as a **resilient internal stream** — bytes keep flowing during the
think phase, a per-chunk idle watchdog reclaims dead connections, and a
connection drop before completion is replayed and re-accumulated. The caller's
API is unchanged: ``one_chat`` / ``text_chat`` / ``tool_invoke`` /
``tool_chat`` still return the same shapes. See ``_deepseek_stream.py`` for the
techniques, ported from DeepSeek-Reasonix.

Pass ``robust_streaming=False`` to fall back to the plain base-class transport.
"""

from typing import Any, Dict, Optional

from core.llms._llm_api_client import LLMApiClient
from core.llms.simple_deep_seek_client import SimpleDeepSeekClient
from core.llms._deepseek_stream import (
    DEFAULT_MAX_RECONNECTS,
    DEFAULT_STREAM_IDLE_TIMEOUT,
    wrap_client_for_robust_streaming,
)
from ..utils.config_setting import Config


# Forward-declaration shim: keeps ``LLMFactory`` discovery happy when a type
# checker walks this module before the real class is defined below.
class SimpleDeepSeekClientReasoning(LLMApiClient):
    pass


class SimpleDeepSeekClientReasoning(SimpleDeepSeekClient):
    MODEL_CONFIG_KEYS = ("simple_deep_seek_reasoning_model", "simple_deep_seek_model")
    DEFAULT_MODEL = "deepseek-v4-pro"

    def __init__(
        self,
        model: Optional[str] = None,
        max_tokens: int = 64000,
        temperature: float = 1.0,
        top_p: float = 1,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
        stop=None,
        reasoning_effort: Optional[str] = "high",
        extra_body: Optional[Dict[str, Any]] = None,
        thinking: bool = True,
        context_window_tokens: Optional[int] = None,
        # --- reliability knobs (ported from DeepSeek-Reasonix) ---
        robust_streaming: bool = True,
        stream_idle_timeout: float = DEFAULT_STREAM_IDLE_TIMEOUT,
        max_reconnects: int = DEFAULT_MAX_RECONNECTS,
    ):
        config = Config()
        api_key = config.get("deep_seek_api_key")
        super().__init__(
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            stop=stop,
            reasoning_effort=reasoning_effort,
            extra_body=extra_body,
            thinking=thinking,
            context_window_tokens=context_window_tokens,
        )

        # Read at call time by the robust transport (model can change via
        # ``set_task``), so store them as plain attributes.
        self._stream_idle_timeout = stream_idle_timeout
        self._max_reconnects = max_reconnects
        self._robust_streaming = robust_streaming

        if robust_streaming:
            # Keep a handle to the genuine SDK client and swap in a proxy that
            # turns non-streaming create() calls into resilient streams. Every
            # base-class path funnels through ``self.client.chat.completions
            # .create``, so this single seam covers one_chat / text_chat /
            # tool_chat / tool_invoke / the second tool-result round.
            self._raw_client = self.client
            self.client = wrap_client_for_robust_streaming(self.client, self)
