"""Dispatch-time context for tools that need run-aware observability.

The v5 ``Dispatcher`` calls ``capability.fn(**params)`` synchronously. By
design the tool callable only sees ``params`` — there is no way for a
tool to ask "what run am I inside?" or "what node am I attached to?"
without either threading those ids through ``params`` (which breaks the
declared schema of every tool) or reaching into dispatcher internals.

This module exposes a tiny ``ContextVar`` that the dispatcher sets right
before invoking the capability and resets right after. Tools that want
observability — most notably the ccx → cc bridge that re-publishes cc
SessionEvents as v5 events so they show up in ``watch --tail`` — read
the contextvar through :func:`current_dispatch_context`.

Tools that don't care simply ignore it. The contextvar default is
``None``, so any code that runs *outside* a v5 dispatch (e.g. a unit
test that calls a tool fn directly) sees ``None`` and behaves as before.

The emit callable matches the dispatcher's existing ``EventEmitter``
signature ``(kind, payload) -> None``. The dispatcher already prepends
``run_id`` into payload at its own emit sites, so we mirror that here:
the context exposes ``run_id`` as a separate field and we recommend
tools include it in the payload they emit, but the emit function itself
is the same one the dispatcher uses internally.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import Event
from typing import Any, Callable, Iterator


EventEmitter = Callable[[str, dict[str, Any]], None]


@dataclass(slots=True, frozen=True)
class DispatchContext:
    """The run / node / emit triple a tool sees during a v5 dispatch."""

    run_id: str
    node_id: str
    attempt_id: str
    emit: EventEmitter
    report_cost_fn: Callable[[int, float], None] | None = None
    cancel_event: Event | None = None
    attempt_ordinal: int = 1

    def is_cancelled(self) -> bool:
        return self.cancel_event is not None and self.cancel_event.is_set()

    def report_cost(self, *, tokens: int = 0, cost: float = 0.0) -> None:
        """Report token/cost usage against the active run budget."""
        if self.report_cost_fn is not None:
            self.report_cost_fn(int(tokens or 0), float(cost or 0.0))


_current: ContextVar[DispatchContext | None] = ContextVar(
    "_v5_dispatch_context", default=None
)


def current_dispatch_context() -> DispatchContext | None:
    """Return the active context, or ``None`` if not inside a dispatch.

    Tools should always handle ``None`` — tests and direct calls do not
    set the context.
    """
    return _current.get()


@contextmanager
def set_dispatch_context(ctx: DispatchContext) -> Iterator[DispatchContext]:
    """Install ``ctx`` for the duration of a ``with`` block.

    Uses ``ContextVar.set`` + ``reset`` so that concurrent dispatches in
    different async tasks / threads remain isolated. Safe to nest; the
    inner context wins until the ``with`` exits.
    """
    token = _current.set(ctx)
    try:
        yield ctx
    finally:
        _current.reset(token)


__all__ = [
    "DispatchContext",
    "EventEmitter",
    "current_dispatch_context",
    "set_dispatch_context",
]
