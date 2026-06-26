"""Composable turn-level middleware for the query pipeline.

Inspired by the ``Runner`` / ``Middleware`` pattern in *open-harness*, adapted
for Python ``AsyncIterator[SessionEvent]``.

The key abstractions:

* **TurnRunner** — a callable that accepts turn parameters and yields
  ``SessionEvent`` objects.  ``run_single_turn`` is the canonical inner
  runner.
* **TurnMiddleware** — a callable that wraps one ``TurnRunner`` and returns
  another.  Each middleware adds a single cross-cutting concern (compaction,
  persistence, retry, hooks …).
* **pipe / apply** — compose middleware in the familiar "outermost first"
  order so that ``pipe(a, b, c)(runner)`` equals ``a(b(c(runner)))``.

Usage example::

    from core.cc.conversation.middleware import (
        apply, with_compaction, with_hooks, with_persistence,
    )

    base_runner = functools.partial(run_single_turn, ...)
    runner = apply(
        base_runner,
        with_compaction(compactor, session, store),
        with_hooks(hooks),
        with_persistence(session, store),
    )
    async for event in runner(...):
        ...
"""

from __future__ import annotations

import functools
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, ParamSpec, Protocol

from .compact import SessionCompactor
from .message_store import SessionMessageStore
from .models import SessionEvent, SessionMessage
from .session import QuerySession

# ---------------------------------------------------------------------------
# Core type aliases
# ---------------------------------------------------------------------------

P = ParamSpec("P")


class TurnRunner(Protocol):
    """Anything that produces a stream of ``SessionEvent`` for a single turn."""

    def __call__(self, **kwargs: Any) -> AsyncIterator[SessionEvent]: ...


TurnMiddleware = Callable[[TurnRunner], TurnRunner]


# ---------------------------------------------------------------------------
# Composition helpers
# ---------------------------------------------------------------------------


def pipe(*middleware: TurnMiddleware) -> TurnMiddleware:
    """Compose middleware (outermost listed first).

    ``pipe(a, b, c)(runner)`` is equivalent to ``a(b(c(runner)))``.
    """
    return functools.reduce(
        lambda inner, outer: lambda runner: outer(inner(runner)),
        reversed(middleware),
        lambda runner: runner,
    )


def apply(runner: TurnRunner, *middleware: TurnMiddleware) -> TurnRunner:
    """Shorthand for ``pipe(*middleware)(runner)``."""
    return pipe(*middleware)(runner)


# ---------------------------------------------------------------------------
# with_compaction
# ---------------------------------------------------------------------------


def with_compaction(
    compactor: SessionCompactor,
    session: QuerySession,
    message_store: SessionMessageStore,
) -> TurnMiddleware:
    """Check and apply compaction *before* each turn."""

    def middleware(runner: TurnRunner) -> TurnRunner:
        async def wrapped(**kwargs: Any) -> AsyncIterator[SessionEvent]:
            turn_id = kwargs.get("turn_id", "")

            if compactor.should_compact(session, message_store):
                result = compactor.compact(session, message_store)
                if result.applied and result.boundary_message is not None:
                    yield SessionEvent(
                        event_type="compact_applied",
                        turn_id=turn_id,
                        message=result.boundary_message,
                        payload={"compacted_count": result.compacted_count},
                    )

            async for event in runner(**kwargs):
                yield event

        return wrapped  # type: ignore[return-value]

    return middleware


# ---------------------------------------------------------------------------
# with_persistence
# ---------------------------------------------------------------------------


def with_persistence(
    session: QuerySession,
    message_store: SessionMessageStore,
) -> TurnMiddleware:
    """Persist every emitted message into the ``SessionMessageStore``."""

    def middleware(runner: TurnRunner) -> TurnRunner:
        async def wrapped(**kwargs: Any) -> AsyncIterator[SessionEvent]:
            async for event in runner(**kwargs):
                if event.message is not None:
                    message_store.append(event.message)
                yield event

        return wrapped  # type: ignore[return-value]

    return middleware


# ---------------------------------------------------------------------------
# with_hooks
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TurnHooks:
    """Lifecycle hooks injected around a turn.

    All callbacks are optional.  Async and sync callables are both accepted.
    """

    on_before_turn: Callable[..., Any] | None = None
    on_after_turn: Callable[..., Any] | None = None
    on_error: Callable[[Exception], Any] | None = None
    on_before_tool_batch: Callable[..., Any] | None = None
    on_after_tool_batch: Callable[..., Any] | None = None


async def _maybe_await(result: Any) -> None:
    if isinstance(result, Awaitable):
        await result


def with_hooks(hooks: TurnHooks) -> TurnMiddleware:
    """Invoke lifecycle hooks around the inner runner."""

    def middleware(runner: TurnRunner) -> TurnRunner:
        async def wrapped(**kwargs: Any) -> AsyncIterator[SessionEvent]:
            if hooks.on_before_turn:
                await _maybe_await(hooks.on_before_turn(kwargs))

            error_occurred: Exception | None = None
            try:
                async for event in runner(**kwargs):
                    if event.event_type == "tool_started" and hooks.on_before_tool_batch:
                        await _maybe_await(hooks.on_before_tool_batch(event))
                    if event.event_type in ("tool_completed", "tool_failed") and hooks.on_after_tool_batch:
                        await _maybe_await(hooks.on_after_tool_batch(event))
                    yield event
            except Exception as exc:
                error_occurred = exc
                if hooks.on_error:
                    await _maybe_await(hooks.on_error(exc))
                raise
            finally:
                if hooks.on_after_turn:
                    await _maybe_await(hooks.on_after_turn(kwargs, error_occurred))

        return wrapped  # type: ignore[return-value]

    return middleware


# ---------------------------------------------------------------------------
# with_retry
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RetryPolicy:
    """Configuration for the retry middleware."""

    max_retries: int = 2
    is_retryable: Callable[[Exception], bool] = field(
        default_factory=lambda: _default_is_retryable
    )
    delay_seconds: Callable[[int], float] = field(
        default_factory=lambda: _default_delay
    )


def _default_is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    retryable_keywords = ("rate limit", "timeout", "429", "503", "overloaded")
    return any(keyword in msg for keyword in retryable_keywords)


def _default_delay(attempt: int) -> float:
    return min(2 ** attempt, 30.0)


def with_retry(policy: RetryPolicy) -> TurnMiddleware:
    """Retry the inner runner on transient errors.

    If the inner runner has already yielded content-bearing events the turn is
    **not** retried (partial output must not be discarded).
    """

    def middleware(runner: TurnRunner) -> TurnRunner:
        async def wrapped(**kwargs: Any) -> AsyncIterator[SessionEvent]:
            import asyncio

            last_exc: Exception | None = None
            for attempt in range(1 + policy.max_retries):
                if attempt > 0:
                    delay = policy.delay_seconds(attempt)
                    yield SessionEvent(
                        event_type="retry",
                        turn_id=kwargs.get("turn_id", ""),
                        payload={"attempt": attempt, "delay_seconds": delay},
                    )
                    await asyncio.sleep(delay)

                has_content = False
                try:
                    async for event in runner(**kwargs):
                        if event.message is not None:
                            has_content = True
                        yield event
                    return
                except Exception as exc:
                    last_exc = exc
                    if has_content or not policy.is_retryable(exc):
                        raise
                    if attempt == policy.max_retries:
                        raise

            if last_exc is not None:
                raise last_exc

        return wrapped  # type: ignore[return-value]

    return middleware


# ---------------------------------------------------------------------------
# with_turn_tracking
# ---------------------------------------------------------------------------


def with_turn_tracking(session: QuerySession) -> TurnMiddleware:
    """Track turn state on the session (sets/clears active_turn_id)."""

    def middleware(runner: TurnRunner) -> TurnRunner:
        async def wrapped(**kwargs: Any) -> AsyncIterator[SessionEvent]:
            turn_id = kwargs.get("turn_id", "")
            session.active_turn_id = turn_id
            try:
                async for event in runner(**kwargs):
                    yield event
            finally:
                session.active_turn_id = None

        return wrapped  # type: ignore[return-value]

    return middleware
