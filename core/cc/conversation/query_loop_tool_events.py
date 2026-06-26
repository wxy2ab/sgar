from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
import uuid

from ..tools.base import ToolCall
from ..tools.context import ToolUseContext
from ..tools.result_mapper import ToolResultMapper
from .context_assembler import ContextAssembler
from .models import SessionEvent, SessionMessage
from .session import QuerySession


def normalize_tool_call(payload: dict[str, Any]) -> ToolCall:
    tool_name = str(payload.get("tool_name") or payload.get("name") or "")
    tool_use_id = str(payload.get("tool_use_id") or payload.get("id") or f"toolu_{uuid.uuid4().hex[:8]}")
    arguments = payload.get("arguments") or payload.get("input") or {}
    if not isinstance(arguments, dict):
        arguments = {"value": arguments}
    return ToolCall(
        tool_use_id=tool_use_id,
        tool_name=tool_name,
        arguments=arguments,
    )


async def run_additional_tool_calls(
    *,
    turn_id: str,
    tool_calls: list[ToolCall],
    tool_orchestrator: Any,
    tool_ctx: ToolUseContext,
    tool_mapper: ToolResultMapper,
    context_assembler: ContextAssembler,
    session: QuerySession,
) -> AsyncIterator[SessionEvent]:
    current_ctx = tool_ctx
    for call in tool_calls:
        assistant_tool_message = SessionMessage(
            message_id=f"msg_{uuid.uuid4().hex[:10]}",
            turn_id=turn_id,
            role="assistant",
            content=f"tool:{call.tool_name}",
            kind="assistant_tool_use",
            tool_name=call.tool_name,
            tool_use_id=call.tool_use_id,
            metadata={
                "structured_payload": {"tool_name": call.tool_name, "arguments": call.arguments},
                "auto_generated": True,
            },
        )
        yield SessionEvent(
            event_type="assistant_tool_use",
            turn_id=turn_id,
            message=assistant_tool_message,
            payload={"tool_call": {"tool_name": call.tool_name, "arguments": call.arguments}},
        )

    async for tool_event in tool_orchestrator.run_tool_calls(tool_calls, current_ctx):
        if tool_event.event_type == "tool_context_updated":
            maybe_context = tool_event.payload.get("tool_context")
            if isinstance(maybe_context, ToolUseContext):
                current_ctx = maybe_context
                context_assembler.apply_tool_context(session=session, tool_ctx=current_ctx)
            yield tool_event
            continue

        progress_message = SessionMessage(
            message_id=f"msg_{uuid.uuid4().hex[:10]}",
            turn_id=turn_id,
            role="system",
            content=f"tool_progress:{tool_event.tool_name}:{tool_event.event_type}",
            kind="progress_message",
            tool_name=tool_event.tool_name,
            tool_use_id=tool_event.tool_use_id,
            metadata={"structured_payload": tool_mapper.to_progress_message(tool_event)},
        )
        yield SessionEvent(
            event_type=tool_event.event_type,
            turn_id=turn_id,
            message=progress_message,
            payload={
                "tool_progress": tool_mapper.to_progress_message(tool_event),
            },
        )
        result = tool_event.payload.get("result")
        if result is not None:
            yield SessionEvent(
                event_type="tool_result",
                turn_id=turn_id,
                message=tool_mapper.to_session_message(turn_id, result),
                payload={"result": result},
            )
