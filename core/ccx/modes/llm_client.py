"""LLM client wrapper with a single callable signature.

ccx mode runners don't talk to `LLMClientProvider` directly. Instead a thin
adapter `LLMCallable(system, user, purpose) -> str | LLMResult` is injected.
This:

* Decouples mode runners from cc's `LLMClientProvider` protocol (which
  returns "Any") so tests can inject a stub function.
* Makes the per-call purpose explicit (used by cc for routing / pricing).
* Lets callers who can compute per-call cost (R4) wrap the response in
  ``LLMResult`` so ``_make_mode_tool`` can accumulate cost and emit
  ``ccx.cost.node`` events. Callers without cost info just return a
  bare ``str`` (cost defaults to 0). Pre-R4 stubs and providers keep
  working unchanged.

Production wiring builds an `LLMCallable` from a real `LLMClientProvider`
+ a CCConfig via `from_provider()`.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Union

logger = logging.getLogger(__name__)


def _default_llm_timeout_s() -> float:
    try:
        return float(os.environ.get("CCX_LLM_TIMEOUT_S", "600") or 600)
    except (TypeError, ValueError):
        return 600.0


#: Default per-LLM-call wall clock for the production provider adapter. A real
#: reasoning client that drops its connection can otherwise hang the calling
#: thread indefinitely (this project has been bitten by a 7-hour stall). 600s
#: is generous enough that a healthy call never trips it; it only bounds true
#: hangs. Override via ``CCX_LLM_TIMEOUT_S`` (``0`` / negative disables).
_DEFAULT_LLM_TIMEOUT_S = _default_llm_timeout_s()


@dataclass(frozen=True, slots=True)
class LLMResult:
    """Optional structured return type for an ``LLMCallable``.

    ``text`` is the LLM's response — the same string a bare-``str``
    return would have produced.

    ``cost_usd`` is the dollar cost of producing that response (input
    + output tokens, after any provider-side discount). ``0.0`` means
    "unknown / free / not measured" — ``_make_mode_tool`` accumulates
    it into the ``ccx.cost.node`` event payload, so a watcher can
    distinguish "this node was cheap" from "we don't track cost on
    this LLMCallable yet" only by looking at adjacent calls'
    ``call_count`` vs ``cost_usd`` distribution.

    ``extras`` is a free-form dict for provider-specific telemetry
    (input_tokens, output_tokens, model_id, cache_hit, ...). ccx
    doesn't interpret it; consumers of the cost event are free to
    surface it.
    """

    text: str
    cost_usd: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)


# An LLMCallable may return either a bare string (pre-R4 contract) or
# an ``LLMResult`` (post-R4, for cost-aware providers). Union'd at the
# protocol level so type-checkers see both forms as valid.
LLMResponse = Union[str, LLMResult]


class LLMCallable(Protocol):
    def __call__(self, *, system: str, user: str, purpose: str) -> LLMResponse: ...


def from_callable(fn: Callable[..., LLMResponse]) -> LLMCallable:
    """Wrap a plain Callable into the LLMCallable shape (kw-only)."""
    def _adapter(*, system: str, user: str, purpose: str) -> LLMResponse:
        return fn(system=system, user=user, purpose=purpose)
    return _adapter


def text_of(response: LLMResponse | Any) -> str:
    if response is None:
        return ""
    if isinstance(response, LLMResult):
        return response.text
    return str(response)


def llm_result_tokens(result: LLMResult) -> int:
    extras = result.extras or {}
    total = 0
    for key in ("input_tokens", "output_tokens"):
        try:
            total += int(extras.get(key) or 0)
        except (TypeError, ValueError):
            continue
    return total


def _merged_prompt(system: str, user: str) -> str:
    system = str(system or "").strip()
    user = str(user or "")
    if not system:
        return user
    return f"{system}\n\n{user}"


def _run_with_timeout(
    work: Callable[[], str], *, label: str, timeout_s: float,
) -> str:
    """Run ``work`` in a daemon thread, joined with ``timeout_s``.

    Returns ``work()``'s text, or ``""`` on timeout / error. An empty string
    is treated by every downstream ``parse_llm_json`` as a parse failure → the
    caller's safe fallback, so a hung provider degrades the run instead of
    blocking it forever. A daemon thread (not ``ThreadPoolExecutor``) is used
    deliberately: a hung reasoning client thread is then abandoned at process
    exit rather than blocking ``atexit`` join forever. Mirrors
    ``governed_goal._call_llm_bounded`` — the project's established pattern for
    bounding real LLM calls.

    A non-positive ``timeout_s`` disables bounding and calls ``work`` inline.
    """
    if not timeout_s or timeout_s <= 0:
        return work()

    box: dict[str, Any] = {"text": "", "done": False}

    def _worker() -> None:
        try:
            box["text"] = work()
        except Exception:  # noqa: BLE001 — never let a worker crash the caller
            logger.warning("ccx llm: call %s raised", label, exc_info=True)
        finally:
            box["done"] = True

    thread = threading.Thread(
        target=_worker, name=f"ccx-llm-{label}", daemon=True,
    )
    thread.start()
    thread.join(timeout_s)
    if not box["done"]:
        logger.warning(
            "ccx llm: call %s did not return within %.0fs; "
            "abandoning the thread and falling back", label, timeout_s,
        )
        return ""
    return str(box["text"] or "")


@dataclass(slots=True)
class _ProviderAdapter:
    """Bridges core.cc.llm.LLMClientProvider to LLMCallable.

    Production providers return a chat-capable client; we attempt
    common-shaped chat / one_chat methods. Tests should bypass this entirely
    and inject a callable directly.

    ``timeout_s`` bounds the actual chat call so a hung provider can never
    stall the calling thread forever; see ``_run_with_timeout``. Only the
    network call is bounded — an unknown-client ``TypeError`` still raises
    loudly (it fires before any bounded call).
    """
    provider: Any
    config: Any
    timeout_s: float = _DEFAULT_LLM_TIMEOUT_S

    def __call__(self, *, system: str, user: str, purpose: str) -> str:
        client = self.provider.get_client(config=self.config, purpose=purpose)
        # Try the common one_chat(messages) shape used in this repo.
        if hasattr(client, "one_chat"):
            prompt = _merged_prompt(system, user)
            return _run_with_timeout(
                lambda: text_of(client.one_chat(prompt)),
                label=purpose, timeout_s=self.timeout_s,
            )
        # Fallback: chat(messages) returning a structured response.
        if hasattr(client, "chat"):
            messages = [
                {"role": "user", "content": _merged_prompt(system, user)},
            ]

            def _chat() -> str:
                result = client.chat(messages)
                if hasattr(result, "content"):
                    return str(result.content)
                return text_of(result)

            return _run_with_timeout(
                _chat, label=purpose, timeout_s=self.timeout_s,
            )
        raise TypeError(
            f"LLM client {type(client).__name__} has no chat / one_chat method"
        )


def from_provider(
    provider: Any, config: Any, *, timeout_s: float | None = None,
) -> LLMCallable:
    if timeout_s is None:
        return _ProviderAdapter(provider=provider, config=config)
    return _ProviderAdapter(
        provider=provider, config=config, timeout_s=timeout_s,
    )


__all__ = [
    "LLMCallable",
    "LLMResponse",
    "LLMResult",
    "from_callable",
    "from_provider",
    "llm_result_tokens",
    "text_of",
]
