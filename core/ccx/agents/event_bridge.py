"""Bridge cc ``SessionEvent``s into the v5 event bus.

When ``CcAgentRunner`` (or any other runner that drives cc's full
``QueryEngine``) is invoked as a v5 capability, the cc side emits its
own stream of ``SessionEvent``s describing what the LLM is doing
right now — tool calls, tool results, assistant text, turn completion.
Until this module those events stayed inside the cc layer: callers
could subscribe via ``event_sink``, but nothing was persisted to the
v5 events table, so the out-of-process watcher (``python -m
core.ccx.watch --tail``) only saw the coarse v5 ``node.*`` lifecycle
events. From the outside, a long-running agent looked frozen between
``node.created`` and ``node.completed`` even when the LLM was busy
making real tool calls.

The bridge closes that gap. It reads :func:`current_dispatch_context`
to learn the active v5 ``run_id``/``node_id``/``emit`` triple set by
the dispatcher right before invoking the capability, and converts each
interesting ``SessionEvent`` into a v5 event with kind ``cc.*``. The
v5 EventBus persists those into the same ``events`` table the watcher
tails by sequence cursor, so a single ``watch --tail`` session
interleaves v5 node lifecycle and cc tool-call activity.

Why a separate kind namespace (``cc.``): keeping cc events distinct
from native v5 events (``node.``, ``replan.``, ``budget.``) lets the
watcher filter them with ``--kind cc.`` or ``--kind node.``
independently, and makes it obvious in the event stream which layer a
given row came from.

The bridge is opt-in by design — runners must build a sink and pass it
to the cc QueryEngine. Runners that don't care (or run outside a v5
dispatch context) keep working unchanged; without a dispatch context
the sink is a no-op.
"""

from __future__ import annotations

from typing import Any, Callable

from core.cc.conversation.models import SessionEvent
from core.deepstack_v5.execution.dispatch_context import (
    DispatchContext,
    current_dispatch_context,
)
from core.deepstack_v5.memory import ContentStore, priority_for


# tool_result bodies larger than this go to the ContentStore (full
# content, FTS5-indexed) and only a short preview rides in the v5
# event payload. The cap is tuned so a typical multi-file Read /
# small Bash result still inlines, but a giant grep / file_read /
# log dump doesn't bloat every event row.
_FULL_CONTENT_THRESHOLD = 4_096


# cc SessionEvent.event_type → v5 kind. Anything not in this map is
# dropped — we deliberately filter to events the LLM/agent observer
# actually cares about (tool activity, terminal text, errors). High
# volume internal events like ``tool_context_updated`` and progress
# heartbeats are skipped.
_KIND_MAP: dict[str, str] = {
    "assistant_tool_use": "cc.tool_use",
    "tool_completed": "cc.tool_completed",
    "tool_failed": "cc.tool_failed",
    "tool_result": "cc.tool_result",
    "assistant_text": "cc.assistant_text",
    "assistant_completed": "cc.assistant_text",
    "assistant_followup_completed": "cc.assistant_text",
    "turn_completed": "cc.turn_completed",
    "turn_failed": "cc.turn_failed",
}


# Truncate large free-text content fields when re-publishing so a single
# verbose tool result doesn't bloat the events table. The watcher reads
# payload_json fully, so the cap keeps the row narrow.
_TEXT_TRUNCATE = 240


def _truncate(text: str | None, limit: int = _TEXT_TRUNCATE) -> str:
    if not text:
        return ""
    s = str(text)
    if len(s) <= limit:
        return s
    return s[: limit - 1] + "…"


def _finalize(
    v5_kind: str, payload: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Attach the computed priority and return the (kind, payload) pair.

    Computing priority here (rather than at each per-branch return)
    keeps the rule in one place and applies it uniformly to every
    bridged event. Downstream consumers — ResumeSnapshot, ctx_stats —
    rely on this field being present on every ``cc.*`` payload.
    """
    payload["priority"] = priority_for(v5_kind, payload)
    return v5_kind, payload


def event_to_v5(
    event: SessionEvent,
    *,
    run_id: str,
    node_id: str,
    attempt_id: str | None = None,
    content_store: ContentStore | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """Convert a cc SessionEvent to ``(v5_kind, payload)`` or ``None``.

    Returning ``None`` means "skip this event"; the caller is expected
    to ignore it. The payload always carries ``run_id`` and ``node_id``
    so the watcher's filters (``--run-id``, ``--node-id``) work the
    same way for ``cc.*`` events as they do for native ``node.*``
    events. ``source="cc"`` makes it easy for downstream consumers to
    tell at a glance which layer emitted the row.
    """
    v5_kind = _KIND_MAP.get(event.event_type)
    if v5_kind is None:
        return None

    payload: dict[str, Any] = {
        "run_id": run_id,
        "node_id": node_id,
        "attempt_id": attempt_id or "",
        "source": "cc",
        "cc_event_type": event.event_type,
        "turn_id": getattr(event, "turn_id", ""),
    }
    msg = getattr(event, "message", None)

    if event.event_type == "assistant_tool_use":
        # Capture the tool name + a compact view of which arg keys were
        # supplied. Argument *values* can be huge (file content, long
        # shell commands) so we summarise rather than copy.
        tool_call = (event.payload or {}).get("tool_call") or {}
        payload["tool_name"] = tool_call.get("tool_name") or (
            getattr(msg, "tool_name", None) if msg else None
        )
        payload["tool_use_id"] = tool_call.get("tool_use_id") or (
            getattr(msg, "tool_use_id", None) if msg else None
        )
        args = tool_call.get("arguments")
        if isinstance(args, dict):
            payload["arg_keys"] = list(args.keys())[:8]
            # Pull a couple of distinctive values for the common tools
            # so the watcher line shows useful context.
            for hint in ("command", "pattern", "path", "query", "action", "mode"):
                if hint in args:
                    payload[hint] = _truncate(str(args[hint]), 80)
                    break
        return _finalize(v5_kind, payload)

    if event.event_type in {"tool_completed", "tool_failed"}:
        progress = (event.payload or {}).get("tool_progress") or {}
        payload["tool_name"] = progress.get("tool_name") or (
            getattr(msg, "tool_name", None) if msg else None
        )
        payload["tool_use_id"] = progress.get("tool_use_id") or (
            getattr(msg, "tool_use_id", None) if msg else None
        )
        payload["success"] = progress.get("success")
        payload["error_code"] = progress.get("error_code")
        payload["duration_ms"] = progress.get("duration_ms")
        return _finalize(v5_kind, payload)

    if event.event_type == "tool_result":
        # The result content can be very large (e.g. a file read). The
        # v5 event row always keeps a 240-char preview for the
        # watcher; the full body — when above _FULL_CONTENT_THRESHOLD
        # — is forwarded to the ContentStore so a later turn can grep
        # it via FTS5 instead of dumping the whole thing back into the
        # LLM context. A ``full_content_ref`` link on the payload lets
        # the watcher dereference if the user wants to inspect.
        content = getattr(msg, "content", "") if msg else ""
        tool_name = getattr(msg, "tool_name", None) if msg else None
        tool_use_id = getattr(msg, "tool_use_id", None) if msg else None
        payload["tool_name"] = tool_name
        payload["tool_use_id"] = tool_use_id
        payload["preview"] = _truncate(content)
        content_bytes = len(content.encode("utf-8")) if content else 0
        if (
            content_store is not None
            and content
            and content_bytes > _FULL_CONTENT_THRESHOLD
            and tool_use_id
        ):
            source_id = f"cc.{node_id}.{tool_use_id}"
            try:
                accepted = content_store.enqueue(
                    run_id,
                    source_id,
                    content,
                    label=str(tool_name) if tool_name else None,
                    content_type="prose",
                )
                if accepted:
                    payload["full_content_ref"] = f"content://{source_id}"
                    payload["full_content_bytes"] = content_bytes
            except Exception:
                # ContentStore failure must never break a cc turn.
                pass
        return _finalize(v5_kind, payload)

    if event.event_type in {
        "assistant_text",
        "assistant_completed",
        "assistant_followup_completed",
    }:
        content = getattr(msg, "content", "") if msg else ""
        payload["preview"] = _truncate(content)
        payload["chars"] = len(content) if content else 0
        metadata = getattr(msg, "metadata", None) or {}
        if metadata.get("turn_timeout_reached"):
            payload["turn_timeout_reached"] = True
            payload["success"] = False
            payload["error_code"] = metadata.get("error_code") or "QE1008"
        return _finalize(v5_kind, payload)

    if event.event_type == "turn_failed":
        # The cc query engine attaches details on the message; pull a
        # short reason string for the watcher line.
        if msg is not None:
            payload["reason"] = _truncate(getattr(msg, "content", ""))
            payload["error_code"] = (msg.metadata or {}).get("error_code")
        return _finalize(v5_kind, payload)

    if event.event_type == "turn_completed":
        # Final marker. Some agents emit it explicitly; if not, the v5
        # node.completed event already covers the boundary.
        return _finalize(v5_kind, payload)

    return _finalize(v5_kind, payload)  # pragma: no cover - covered by _KIND_MAP keys


CcEventSink = Callable[[SessionEvent], Any]


def make_event_sink(
    *,
    ctx: DispatchContext | None = None,
    content_store: ContentStore | None = None,
) -> CcEventSink:
    """Return a cc ``event_sink``-compatible callable that publishes to v5.

    ``ctx`` defaults to :func:`current_dispatch_context`; pass an
    explicit context for tests that aren't inside a real v5 dispatch.
    If no context is available the returned sink is a no-op (the
    runner can install it unconditionally without checking).

    ``content_store`` is the Phase 2 plumbing: when supplied, any
    ``tool_result`` SessionEvent whose body exceeds
    ``_FULL_CONTENT_THRESHOLD`` is enqueued into it for later FTS5
    retrieval, and the v5 event payload picks up a
    ``full_content_ref`` link. When ``None``, only the 240-char
    preview is persisted (Phase 1a/1b behaviour preserved).

    The cc QueryEngine calls ``event_sink(event)`` for every emitted
    ``SessionEvent``. The return value is forwarded back to the engine
    — we return ``None`` so cc treats it as a passive observer that
    shouldn't influence event streaming.
    """
    if ctx is None:
        ctx = current_dispatch_context()
    if ctx is None:
        def _noop(_event: SessionEvent) -> None:
            return None
        return _noop

    run_id = ctx.run_id
    node_id = ctx.node_id
    attempt_id = ctx.attempt_id
    emit = ctx.emit

    def _sink(event: SessionEvent) -> None:
        if ctx.is_cancelled():
            return None
        try:
            out = event_to_v5(
                event,
                run_id=run_id,
                node_id=node_id,
                attempt_id=attempt_id,
                content_store=content_store,
            )
            if out is None or ctx.is_cancelled():
                return None
            kind, payload = out
            emit(kind, payload)
        except Exception:
            # The bridge is best-effort observability. Conversion or emit
            # failures (bad payloads, corrupt SQLite, a closed bus during
            # shutdown) must not break the cc turn.
            return None
        return None

    return _sink


__all__ = [
    "CcEventSink",
    "event_to_v5",
    "make_event_sink",
]
