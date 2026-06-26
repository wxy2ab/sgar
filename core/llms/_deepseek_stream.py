"""Connection-resilient streaming transport for DeepSeek *reasoning* calls.

WHY THIS EXISTS
---------------
``SimpleDeepSeekClientReasoning`` drives ``deepseek-v4-pro`` with
``thinking=True`` + ``reasoning_effort=high``. ccx (doc-mode investigators,
sgar drivers, cc_query_loop) issues these as **non-streaming** calls
(``one_chat`` / ``tool_invoke`` with ``is_stream=False``). A non-streaming
reasoning request holds one HTTP connection open with *no bytes flowing* for
the entire reasoning gap — often minutes for a high-effort answer. That idle
window is exactly what kills the request:

  * a local proxy (v2rayN / sing-box / clash) or the gateway idle-closes the
    long-lived connection during the first-token gap, surfacing as a
    connection reset / unexpected EOF / read timeout, and
  * the whole expensive call is then retried *from scratch* by tenacity,
    usually hitting the same wall again.

This module ports the techniques DeepSeek-Reasonix uses to make the same
endpoint reliable (``internal/provider/openai/openai.go`` +
``internal/provider/retry.go``):

  1. **Always stream under the hood**, even when the caller wants one
     complete object. Streaming makes the model emit ``reasoning_content``
     deltas continuously, so bytes keep flowing and nothing idle-closes the
     connection during the think phase.
  2. **Per-chunk idle watchdog.** A half-open TCP connection (proxy switched
     mid-stream) sends no RST, so a naive read blocks forever. We cap the
     gap-between-chunks with an httpx read timeout, turning a silent hang into
     a recoverable error.
  3. **Reconnect-and-replay on connection drops.** Because we accumulate the
     whole answer internally and hand back a single object, we can replay on
     *any* connection reset that happens before completion — unlike Reasonix,
     which streams live to a UI and can only replay before the first forwarded
     token.
  4. **Treat an early-ended stream as incomplete.** A proxy that idle-closes
     with a clean FIN ends iteration with no error and no ``finish_reason``;
     committing that half-streamed turn as complete is the
     truncated-output / unparseable-JSON failure mode. We detect a missing
     ``finish_reason`` and replay instead of returning a partial answer.

The accumulated result is rebuilt into a real ``openai`` ``ChatCompletion``
(via ``model_construct``) so every downstream consumer in
``simple_deep_seek_client.py`` — ``_create_chat_completion``, ``tool_invoke``,
``_process_tool_response`` — sees an object byte-compatible with a genuine
non-streaming response, including ``reasoning_content`` and ``usage``.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import httpx
import openai
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from openai.types.completion_usage import CompletionUsage

logger = logging.getLogger(__name__)

# How long a started stream may go without *any* chunk before it's treated as a
# dropped connection. Mirrors Reasonix's defaultStreamIdleTimeout (120s):
# generous on purpose — a live reasoner emits reasoning deltas far more often —
# but far below the base client's 600s read timeout so a dead connection is
# reclaimed in ~2min instead of ~10min.
DEFAULT_STREAM_IDLE_TIMEOUT = 120.0

# Times a pre-completion connection drop is replayed before the error surfaces.
DEFAULT_MAX_RECONNECTS = 3

# Cap for the exponential backoff between reconnects (seconds).
MAX_BACKOFF = 15.0

# HTTP statuses worth retrying: request-timeout, conflict/too-early, rate-limit
# and any 5xx. Every other 4xx (400/401/403/404/422 …) is a caller/config
# problem a replay can't fix — fail fast. Matches provider.RetryableStatus.
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429})

# Substrings that mark a connection-level drop when no typed error is available
# (some proxies surface resets as plain OSError/str).
_CONN_RESET_HINTS = (
    "connection reset",
    "econnreset",
    "connection aborted",
    "broken pipe",
    "peer closed",
    "server disconnected",
    "unexpected eof",
    "incomplete chunked read",
    "remoteprotocolerror",
    "connection closed",
)


class IncompleteStreamError(Exception):
    """A stream ended with no ``finish_reason`` — a proxy idle-close with a
    clean FIN, not a real completion. Retryable: replay rather than commit a
    half-streamed turn (Reasonix #3953)."""


def is_retryable_stream_error(exc: BaseException) -> bool:
    """Report whether ``exc`` is a transient transport failure that a fresh
    request can plausibly recover from, vs a caller/config error that will
    fail identically on replay."""
    if isinstance(exc, IncompleteStreamError):
        return True

    # Typed HTTP status from the openai SDK (APIStatusError subclasses carry it).
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in _RETRYABLE_STATUS or 500 <= status <= 599

    # Network-level openai errors have no status_code.
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return True

    # httpx transport/timeout failures (ReadTimeout, ConnectError,
    # RemoteProtocolError, PoolTimeout, …) — TransportError is the common base.
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True

    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True

    msg = str(exc).lower()
    return any(hint in msg for hint in _CONN_RESET_HINTS)


def _backoff_delay(attempt: int) -> float:
    """Capped exponential backoff with jitter: 0.5·2^(n-1) up to MAX_BACKOFF,
    plus up to 250ms of jitter. ``attempt`` is the 1-based retry number."""
    base = 0.5 * (2 ** (attempt - 1))
    if base > MAX_BACKOFF:
        base = MAX_BACKOFF
    return base + random.uniform(0.0, 0.25)


@dataclass
class _StreamResult:
    content: str = ""
    reasoning: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    usage: Optional[Any] = None
    finish_reason: Optional[str] = None


def _consume_stream(stream: Any) -> _StreamResult:
    """Accumulate one SSE response into a ``_StreamResult``. Tolerant of both
    SDK ``ChatCompletionChunk`` objects and duck-typed test doubles (every field
    read via ``getattr``). May raise mid-iteration on a transport drop — the
    caller's reconnect loop handles that."""
    content: List[str] = []
    reasoning: List[str] = []
    tool_acc: Dict[int, Dict[str, Any]] = {}
    order: List[int] = []
    usage: Optional[Any] = None
    finish_reason: Optional[str] = None

    for chunk in stream:
        u = getattr(chunk, "usage", None)
        if u is not None:
            usage = u
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        choice = choices[0]
        fr = getattr(choice, "finish_reason", None)
        if fr:
            finish_reason = fr
        delta = getattr(choice, "delta", None)
        if delta is None:
            continue
        rc = getattr(delta, "reasoning_content", None)
        if rc:
            reasoning.append(rc)
        dc = getattr(delta, "content", None)
        if dc:
            content.append(dc)
        for tc in getattr(delta, "tool_calls", None) or []:
            idx = getattr(tc, "index", 0) or 0
            slot = tool_acc.get(idx)
            if slot is None:
                slot = {"index": idx, "id": None, "name": None, "arguments": ""}
                tool_acc[idx] = slot
                order.append(idx)
            tc_id = getattr(tc, "id", None)
            if tc_id:
                slot["id"] = tc_id
            fn = getattr(tc, "function", None)
            if fn is not None:
                fn_name = getattr(fn, "name", None)
                if fn_name:
                    slot["name"] = fn_name
                fn_args = getattr(fn, "arguments", None)
                if fn_args:
                    slot["arguments"] += fn_args

    return _StreamResult(
        content="".join(content),
        reasoning="".join(reasoning),
        # Order by stream index, not arrival order: a gateway may interleave
        # deltas for multiple calls out of order (Reasonix sorts the same way,
        # openai.go sort.Ints(order)).
        tool_calls=[tool_acc[i] for i in sorted(order)],
        usage=usage,
        finish_reason=finish_reason,
    )


def _build_completion(result: _StreamResult, model: str) -> ChatCompletion:
    """Rebuild the accumulated stream into a real ``ChatCompletion`` so the base
    client's non-streaming consumers see a genuine-looking response object."""
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None
    if result.tool_calls:
        tool_calls = []
        for slot in result.tool_calls:
            # Synthesize a stable id keyed on the stream index when the gateway
            # omits one — an empty id collapses multi-tool turns downstream
            # (mirrors Reasonix's call_%d-by-index).
            tool_calls.append(
                ChatCompletionMessageToolCall.model_construct(
                    id=slot.get("id") or f"call_{slot.get('index', 0)}",
                    type="function",
                    function=Function.model_construct(
                        name=slot.get("name") or "",
                        arguments=slot.get("arguments") or "",
                    ),
                )
            )

    # A pure tool-call turn carries content=null upstream; a text turn carries
    # the (possibly empty) string. Match that so re-serialised history is valid.
    msg_content: Optional[str]
    if tool_calls and not result.content:
        msg_content = None
    else:
        msg_content = result.content

    message = ChatCompletionMessage.model_construct(
        role="assistant",
        content=msg_content,
        tool_calls=tool_calls,
        reasoning_content=result.reasoning or None,
    )

    usage = result.usage
    if usage is None:
        usage = CompletionUsage.model_construct(
            prompt_tokens=0, completion_tokens=0, total_tokens=0
        )

    choice = Choice.model_construct(
        index=0,
        finish_reason=result.finish_reason or "stop",
        message=message,
        logprobs=None,
    )
    return ChatCompletion.model_construct(
        id="chatcmpl-reasoning-stream",
        object="chat.completion",
        created=0,
        model=model,
        choices=[choice],
        usage=usage,
    )


def robust_stream_create(
    create_fn: Callable[..., Any],
    kwargs: Dict[str, Any],
    *,
    model: str,
    idle_timeout: float = DEFAULT_STREAM_IDLE_TIMEOUT,
    max_reconnects: int = DEFAULT_MAX_RECONNECTS,
    sleep: Callable[[float], None] = time.sleep,
) -> ChatCompletion:
    """Run a non-streaming ``chat.completions.create`` as a resilient internal
    stream and return an equivalent ``ChatCompletion``.

    ``create_fn`` is the underlying ``completions.create`` (SDK or test double).
    ``kwargs`` is the fully-built non-streaming request; this function flips it
    to streaming, adds ``stream_options.include_usage`` and a per-chunk idle
    timeout, then accumulates with reconnect-on-drop. Non-retryable errors
    (4xx, auth) propagate immediately; transient drops replay up to
    ``max_reconnects`` times with exponential backoff.
    """
    stream_kwargs = dict(kwargs)
    stream_kwargs["stream"] = True
    stream_options = dict(stream_kwargs.get("stream_options") or {})
    stream_options.setdefault("include_usage", True)
    stream_kwargs["stream_options"] = stream_options
    # Per-request timeout override: ``read`` is the gap-between-chunks watchdog.
    stream_kwargs["timeout"] = httpx.Timeout(
        connect=60.0, read=idle_timeout, write=120.0, pool=60.0
    )

    last_exc: Optional[BaseException] = None
    for attempt in range(max_reconnects + 1):
        if attempt > 0:
            delay = _backoff_delay(attempt)
            logger.warning(
                "DeepSeek reasoning stream interrupted; reconnecting "
                "(%d/%d) after %.2fs: %s",
                attempt,
                max_reconnects,
                delay,
                last_exc,
            )
            sleep(delay)
        try:
            stream = create_fn(**stream_kwargs)
            result = _consume_stream(stream)
        except Exception as exc:  # noqa: BLE001 — classify, don't blanket-swallow
            if is_retryable_stream_error(exc) and attempt < max_reconnects:
                last_exc = exc
                continue
            raise

        if result.finish_reason is None:
            # Clean-FIN / idle-close before completion: replay rather than
            # commit a truncated turn.
            last_exc = IncompleteStreamError(
                "stream ended before completion (no finish_reason); "
                "connection likely idle-closed mid-think"
            )
            if attempt < max_reconnects:
                continue
            raise last_exc

        return _build_completion(result, model)

    # Unreachable: the loop returns or raises on the final attempt.
    raise last_exc if last_exc is not None else RuntimeError(
        "robust_stream_create exhausted retries without an error"
    )


class _RobustCompletions:
    """Proxy for ``client.chat.completions`` that routes non-streaming
    ``create`` calls through :func:`robust_stream_create`. Explicit streaming
    calls (``stream=True``) pass straight through unchanged."""

    def __init__(self, real: Any, owner: Any) -> None:
        self._real = real
        self._owner = owner

    def create(self, **kwargs: Any) -> Any:
        if kwargs.get("stream"):
            return self._real.create(**kwargs)
        return robust_stream_create(
            self._real.create,
            kwargs,
            model=kwargs.get("model") or getattr(self._owner, "model", ""),
            idle_timeout=getattr(
                self._owner, "_stream_idle_timeout", DEFAULT_STREAM_IDLE_TIMEOUT
            ),
            max_reconnects=getattr(
                self._owner, "_max_reconnects", DEFAULT_MAX_RECONNECTS
            ),
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class _RobustChat:
    def __init__(self, real: Any, owner: Any) -> None:
        self._real = real
        self.completions = _RobustCompletions(real.completions, owner)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class _RobustOpenAIClient:
    """Thin proxy over an ``openai.OpenAI`` client that swaps in robust
    streaming for ``chat.completions.create`` while delegating everything else
    to the real client."""

    def __init__(self, real: Any, owner: Any) -> None:
        self._real = real
        self.chat = _RobustChat(real.chat, owner)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def wrap_client_for_robust_streaming(real_client: Any, owner: Any) -> _RobustOpenAIClient:
    """Wrap ``real_client`` so non-streaming ``chat.completions.create`` calls
    become resilient internal streams. ``owner`` supplies the live ``model``,
    ``_stream_idle_timeout`` and ``_max_reconnects`` at call time."""
    return _RobustOpenAIClient(real_client, owner)
