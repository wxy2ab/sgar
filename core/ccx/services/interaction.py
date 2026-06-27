"""Human-in-the-loop interaction primitives for the ccx ``ask_human`` tool.

ccx is an autonomous code agent, but a host can opt into a request/response
seam so the agent can put a question / decision / approval to a human mid-run
and continue with the answer. The seam is deliberately *not* the forbidden
``requires_approval`` (which parks a v5 DAG node forever): instead a host
callback is threaded onto every per-call ``DispatchContext`` (next to
``report_cost_fn``) and invoked from inside an already-running, wall-clock
bounded tool dispatch.

This module owns the data contract and the bounded-wait helper; the cc tool
that surfaces it lives in ``core.ccx.agents.ask_human_tool``. Nothing here
imports the engine, the runtime, or the scheduler — the dependency arrow points
only *into* this module.

Safety model (the reason ``run_interaction`` looks the way it does):

* The host handler may block (a real human takes seconds-to-minutes to answer)
  or hang. We bound it with a daemon-thread join — the same discipline as
  ``llm_client._run_with_timeout`` (daemon, not ``ThreadPoolExecutor``, so a
  hung handler is abandoned at process exit rather than wedging ``atexit``).
* While blocked, the dispatcher's idle-abandon watchdog (``CCX_NODE_IDLE_TIMEOUT_S``)
  would otherwise see "no progress" and kill the turn. The watchdog heartbeat
  is bumped on every emitted event, so we *poll* the join in heartbeat slices
  and emit ``ccx.interaction.waiting`` each slice to keep the turn alive.
* On timeout / handler error we return a ``timeout`` sentinel — never raise,
  never abort the run. An autonomous agent then proceeds on its own judgment.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

logger = logging.getLogger(__name__)


#: Default bound on a single human answer. Human-paced but not infinite. Keep
#: it below the node wall-clock / idle timeout so the inner bound always fires
#: first; configurable per ``CodeAgent`` via ``interaction_timeout_s``.
DEFAULT_INTERACTION_TIMEOUT_S = 300.0

#: How often the bounded wait wakes to emit a ``ccx.interaction.waiting``
#: heartbeat (which bumps the dispatcher activity watchdog). Must be shorter
#: than any sane ``CCX_NODE_IDLE_TIMEOUT_S`` so an enabled idle-abandon never
#: trips while a human is genuinely deliberating.
INTERACTION_HEARTBEAT_S = 10.0

#: Defense-in-depth env flag. The *real* gate is handler presence (the tool is
#: not even registered without one); this lets an operator force the tool off
#: regardless. Default ON when a handler exists.
_ENABLE_FLAG_ENV = "CCX_ENABLE_ASK_HUMAN"
_FALSEY = frozenset({"0", "false", "off", "no"})

# Interaction status values surfaced back to the agent.
STATUS_ANSWERED = "answered"
STATUS_TIMEOUT = "timeout"
STATUS_REFUSED = "refused"
STATUS_NO_HANDLER = "no_handler"

# Severity hints the agent may attach to a request.
SEVERITY_INFO = "info"
SEVERITY_DECISION = "decision"
SEVERITY_BLOCKING = "blocking"
_VALID_SEVERITIES = frozenset({SEVERITY_INFO, SEVERITY_DECISION, SEVERITY_BLOCKING})

# Event kinds (namespaced under ccx. like ccx.steer.injected / ccx.cost.node).
EVENT_REQUESTED = "ccx.interaction.requested"
EVENT_WAITING = "ccx.interaction.waiting"
EVENT_ANSWERED = "ccx.interaction.answered"


def ask_human_enabled() -> bool:
    """Whether the ``ask_human`` tool may be offered (default ON).

    Returns ``False`` only when an operator explicitly sets
    ``CCX_ENABLE_ASK_HUMAN`` to a falsey value. The decisive gate remains
    handler presence — this is a kill switch, not the primary control.
    """
    raw = os.environ.get(_ENABLE_FLAG_ENV)
    if raw is None:
        return True
    return raw.strip().lower() not in _FALSEY


@dataclass(frozen=True, slots=True)
class InteractionRequest:
    """A question the agent puts to a human."""

    question: str
    options: tuple[str, ...] = ()
    context: str = ""
    severity: str = SEVERITY_DECISION
    run_id: str | None = None
    node_id: str | None = None


@dataclass(frozen=True, slots=True)
class InteractionResponse:
    """A human's answer (or a non-answer)."""

    status: str
    answer: str = ""
    selected: str | None = None


class InteractionHandler(Protocol):
    """The host-side contract. Registered via ``CodeAgent``.

    Implementations may block while a human decides; ``run_interaction`` bounds
    the wait and converts a hang/error into a ``timeout`` response, so a handler
    never has to implement its own watchdog (though it may, and return a
    ``timeout`` response itself when it knows it cannot answer).
    """

    def __call__(self, request: InteractionRequest) -> InteractionResponse: ...


def normalize_severity(value: Any) -> str:
    """Coerce an arbitrary severity to one of the known values."""
    text = str(value or "").strip().lower()
    return text if text in _VALID_SEVERITIES else SEVERITY_DECISION


def run_interaction(
    fn: Callable[[InteractionRequest], Any],
    request: InteractionRequest,
    *,
    timeout_s: float,
    emit: Callable[[str, dict[str, Any]], None] | None = None,
    heartbeat_s: float = INTERACTION_HEARTBEAT_S,
) -> InteractionResponse:
    """Invoke the host handler under a bounded, heartbeated daemon-thread wait.

    Returns the handler's :class:`InteractionResponse` on success, or a
    ``status="timeout"`` sentinel if the handler does not return within
    ``timeout_s`` (or raises). Never raises. ``emit`` (when provided) is called
    with ``ccx.interaction.waiting`` once per ``heartbeat_s`` slice so an enabled
    idle-abandon watchdog keeps the turn alive while a human deliberates.

    A non-positive ``timeout_s`` falls back to :data:`DEFAULT_INTERACTION_TIMEOUT_S`
    rather than blocking unbounded — an autonomous runtime must never wait
    forever for a human.
    """
    if not timeout_s or timeout_s <= 0:
        timeout_s = DEFAULT_INTERACTION_TIMEOUT_S
    slice_s = heartbeat_s if heartbeat_s and heartbeat_s > 0 else timeout_s
    slice_s = min(slice_s, timeout_s)

    box: dict[str, Any] = {"resp": None, "done": False}

    def _worker() -> None:
        try:
            box["resp"] = fn(request)
        except Exception:  # noqa: BLE001 — a handler must never crash the run
            logger.warning("ask_human handler raised", exc_info=True)
        finally:
            box["done"] = True

    thread = threading.Thread(target=_worker, name="ccx-ask-human", daemon=True)
    thread.start()

    deadline = time.monotonic() + timeout_s
    beats = 0
    while True:
        thread.join(timeout=slice_s)
        if box["done"]:
            break
        if time.monotonic() >= deadline:
            logger.warning(
                "ask_human: no answer within %.0fs; abandoning the wait", timeout_s,
            )
            return InteractionResponse(status=STATUS_TIMEOUT)
        beats += 1
        if emit is not None:
            try:
                emit(EVENT_WAITING, {
                    "run_id": request.run_id,
                    "node_id": request.node_id,
                    "beat": beats,
                    "elapsed_s": round(beats * slice_s, 1),
                })
            except Exception:  # noqa: BLE001 — observability is best-effort
                logger.debug("ask_human: waiting-heartbeat emit failed", exc_info=True)

    resp = box["resp"]
    if isinstance(resp, InteractionResponse):
        return resp
    # A handler that returned None or a malformed value is treated as a
    # no-answer; degrade to autonomy rather than trusting a bad shape.
    return InteractionResponse(status=STATUS_TIMEOUT)


__all__ = [
    "DEFAULT_INTERACTION_TIMEOUT_S",
    "INTERACTION_HEARTBEAT_S",
    "EVENT_REQUESTED",
    "EVENT_WAITING",
    "EVENT_ANSWERED",
    "STATUS_ANSWERED",
    "STATUS_TIMEOUT",
    "STATUS_REFUSED",
    "STATUS_NO_HANDLER",
    "SEVERITY_INFO",
    "SEVERITY_DECISION",
    "SEVERITY_BLOCKING",
    "InteractionRequest",
    "InteractionResponse",
    "InteractionHandler",
    "ask_human_enabled",
    "normalize_severity",
    "run_interaction",
]
