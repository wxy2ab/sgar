from __future__ import annotations

import asyncio
from typing import Any

from ..errors import ToolExecutionError, ToolValidationError
from ..safety.decision import PermissionDecision
from .base import BaseTool, ToolCall, ToolResult
from .context import ToolUseContext


async def execute_single_tool(
    *,
    tool: BaseTool,
    tool_call: ToolCall,
    ctx: ToolUseContext,
    timeout_ms: int | None = None,
) -> ToolResult:
    # Disabled tools are hidden from the LLM schema (``is_enabled``), but a
    # by-name dispatch (older client, hardcoded tool_call) would otherwise still
    # execute. Reject it here — the single choke point both orchestrator paths
    # funnel through — so schema visibility and dispatch stay consistent.
    if not tool.is_enabled(ctx):
        return ToolResult(
            tool_use_id=tool_call.tool_use_id,
            tool_name=tool_call.tool_name,
            success=False,
            content=f"Tool '{tool_call.tool_name}' is disabled.",
            error_code="TL1009",
        )

    validation = tool.validate_input(tool_call.arguments)
    if not validation.ok:
        raise ToolValidationError(validation.message or "Invalid tool input.")

    permission = tool.check_permissions(ctx, tool_call.arguments)
    if isinstance(permission, PermissionDecision):
        snapshot = permission.context_snapshot or ctx.to_permission_context_snapshot()
        if permission.status == "deny":
            return ToolResult(
                tool_use_id=tool_call.tool_use_id,
                tool_name=tool_call.tool_name,
                success=False,
                content=f"Permission denied. {permission.reason}",
                error_code="PM1002",
                data={
                    "permission": {
                        "status": permission.status,
                        "reason": permission.reason,
                        "source": permission.source,
                        "context_snapshot": snapshot,
                    }
                },
            )
        if permission.status == "ask":
            return ToolResult(
                tool_use_id=tool_call.tool_use_id,
                tool_name=tool_call.tool_name,
                success=False,
                content=f"Permission requires approval. {permission.reason}",
                error_code="PM1003",
                data={
                    "permission": {
                        "status": permission.status,
                        "reason": permission.reason,
                        "source": permission.source,
                        "context_snapshot": snapshot,
                    }
                },
            )
    elif permission == "deny":
        return ToolResult(
            tool_use_id=tool_call.tool_use_id,
            tool_name=tool_call.tool_name,
            success=False,
            content="Permission denied.",
            error_code="PM1002",
        )
    elif permission == "ask":
        return ToolResult(
            tool_use_id=tool_call.tool_use_id,
            tool_name=tool_call.tool_name,
            success=False,
            content="Permission requires approval.",
            error_code="PM1003",
        )

    ctx.tracker.start(tool_call.tool_use_id)
    try:
        if timeout_ms is None:
            return await tool.execute(tool_call, ctx)
        return await asyncio.wait_for(tool.execute(tool_call, ctx), timeout_ms / 1000)
    except asyncio.TimeoutError as exc:
        raise ToolExecutionError("Tool execution timed out.", error_code="TL1004") from exc
    except ToolExecutionError:
        raise
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # pragma: no cover
        raise ToolExecutionError(str(exc), error_code="TL1006") from exc
    finally:
        ctx.tracker.finish(tool_call.tool_use_id)


def execute_with_timeout(coro: Any, timeout_ms: int | None) -> Any:
    if timeout_ms is None:
        return coro
    return asyncio.wait_for(coro, timeout_ms / 1000)


def map_exception_to_tool_error(exc: Exception) -> ToolResult:
    if isinstance(exc, ToolExecutionError):
        return ToolResult(
            tool_use_id="",
            tool_name="",
            success=False,
            content=exc.message,
            error_code=exc.error_code,
        )
    return ToolResult(tool_use_id="", tool_name="", success=False, content=str(exc), error_code="TL1006")
