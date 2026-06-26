from __future__ import annotations

import uuid

from ..conversation.models import SessionMessage
from .base import ToolExecutionEvent, ToolResult


class ToolResultMapper:
    def to_session_message(self, turn_id: str, result: ToolResult) -> SessionMessage:
        metadata = dict(result.data)
        metadata["structured_payload"] = {
            "tool_use_id": result.tool_use_id,
            "tool_name": result.tool_name,
            "success": result.success,
            "error_code": result.error_code,
        }
        return SessionMessage(
            message_id=f"msg_{uuid.uuid4().hex[:10]}",
            turn_id=turn_id,
            role="tool",
            content=result.content,
            kind="tool_result",
            tool_name=result.tool_name,
            tool_use_id=result.tool_use_id,
            is_error=not result.success,
            metadata=metadata,
        )

    def to_progress_message(self, event: ToolExecutionEvent) -> dict[str, object]:
        return {
            "tool_use_id": event.tool_use_id,
            "tool_name": event.tool_name,
            "event_type": event.event_type,
            "success": event.success,
            "error_code": event.error_code,
            "duration_ms": event.duration_ms,
        }

    def to_error_result(self, tool_use_id: str, tool_name: str, error_code: str, message: str) -> ToolResult:
        return ToolResult(
            tool_use_id=tool_use_id,
            tool_name=tool_name,
            success=False,
            content=message,
            error_code=error_code,
        )
