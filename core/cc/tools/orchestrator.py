from __future__ import annotations

from collections.abc import AsyncIterator, Callable
import asyncio
import time

from ..observability import EventRecord, JsonlAuditLogger
from .base import ToolCall, ToolExecutionEvent, ToolResult
from .context import ToolUseContext
from .executor import execute_single_tool
from .registry import ToolRegistry
from .result_mapper import ToolResultMapper


def merge_context_modifiers(
    modifiers: list[Callable[[ToolUseContext], ToolUseContext]],
    ctx: ToolUseContext,
) -> ToolUseContext:
    updated = ctx
    for modifier in modifiers:
        updated = modifier(updated)
    return updated


class ToolOrchestrator:
    def __init__(
        self,
        registry: ToolRegistry,
        mapper: ToolResultMapper | None = None,
        *,
        max_read_concurrency: int = 8,
    ) -> None:
        self.registry = registry
        self.mapper = mapper or ToolResultMapper()
        self.max_read_concurrency = max(1, max_read_concurrency)

    def partition_tool_calls(self, tool_calls: list[ToolCall]) -> list[list[ToolCall]]:
        if not tool_calls:
            return []
        batches: list[list[ToolCall]] = []
        current_batch: list[ToolCall] = []
        current_safe: bool | None = None
        for tool_call in tool_calls:
            tool = self.registry.get(tool_call.tool_name)
            is_safe = False
            if tool is not None:
                try:
                    is_safe = bool(tool.is_concurrency_safe(tool_call.arguments))
                except asyncio.CancelledError:
                    raise
                except Exception:
                    is_safe = False
            if current_batch and (not is_safe or current_safe is False):
                batches.append(current_batch)
                current_batch = []
                current_safe = None
            if is_safe and current_safe is True:
                current_batch.append(tool_call)
                continue
            if current_batch:
                batches.append(current_batch)
                current_batch = []
            current_batch = [tool_call]
            current_safe = is_safe
        if current_batch:
            batches.append(current_batch)
        return batches

    async def run_tool_calls(
        self,
        tool_calls: list[ToolCall],
        ctx: ToolUseContext,
    ) -> AsyncIterator[ToolExecutionEvent]:
        current_ctx = ctx
        audit_logger = JsonlAuditLogger(ctx.config.runtime_root_path(ctx.cwd) / "audit" / "tool_events.jsonl")
        invocation_index = 0
        for batch in self.partition_tool_calls(tool_calls):
            tools = [self.registry.get(tool_call.tool_name) for tool_call in batch]
            try:
                is_concurrency_safe_batch = (
                    len(batch) > 1
                    and all(tool is not None and tool.is_concurrency_safe(call.arguments) for tool, call in zip(tools, batch))
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                is_concurrency_safe_batch = False
            started_events: list[ToolExecutionEvent] = []
            for tool_call, tool in zip(batch, tools):
                if tool is not None:
                    event = ToolExecutionEvent(
                        tool_use_id=tool_call.tool_use_id,
                        tool_name=tool_call.tool_name,
                        event_type="tool_started",
                        success=None,
                    )
                    started_events.append(event)
                    yield event
            self._record_tool_events(audit_logger, ctx, started_events)

            batch_indices = list(range(invocation_index + 1, invocation_index + len(batch) + 1))
            invocation_index += len(batch)
            if is_concurrency_safe_batch:
                results = await self._run_batch_concurrently(batch, tools, current_ctx, batch_indices)
            else:
                results = []
                for tool_call, tool, call_index in zip(batch, tools, batch_indices):
                    if tool is None:
                        results.append(
                            (
                                self.mapper.to_error_result(
                                    tool_use_id=tool_call.tool_use_id,
                                    tool_name=tool_call.tool_name,
                                    error_code="TL1001",
                                    message=f"Unknown tool: {tool_call.tool_name}",
                                ),
                                0,
                                {},
                            )
                        )
                        continue
                    results.append(await self._run_one(tool_call, tool, current_ctx, invocation_index=call_index))

            queued_modifiers: list[Callable[[ToolUseContext], ToolUseContext]] = []
            completion_events: list[ToolExecutionEvent] = []
            for result, duration_ms, tool_refs in results:
                queued_modifiers.extend(result.context_modifiers)
                payload: dict[str, object] = {"result": result}
                if tool_refs:
                    payload.update(tool_refs)
                event = ToolExecutionEvent(
                    tool_use_id=result.tool_use_id,
                    tool_name=result.tool_name,
                    event_type="tool_completed" if result.success else "tool_failed",
                    success=result.success,
                    error_code=result.error_code,
                    duration_ms=duration_ms,
                    payload=payload,
                )
                completion_events.append(event)
                yield event
            self._record_tool_events(audit_logger, ctx, completion_events)
            if queued_modifiers:
                current_ctx = merge_context_modifiers(queued_modifiers, current_ctx)
                event = ToolExecutionEvent(
                    tool_use_id="",
                    tool_name="",
                    event_type="tool_context_updated",
                    success=True,
                    payload={"tool_context": current_ctx},
                )
                self._record_tool_events(audit_logger, ctx, [event])
                yield event

    async def _run_one(
        self,
        tool_call: ToolCall,
        tool: object,
        ctx: ToolUseContext,
        *,
        invocation_index: int = 1,
    ) -> tuple[ToolResult, int, dict[str, object]]:
        started_at = time.time()
        try:
            result = await execute_single_tool(tool=tool, tool_call=tool_call, ctx=ctx)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            result = self.mapper.to_error_result(
                tool_use_id=tool_call.tool_use_id,
                tool_name=tool_call.tool_name,
                error_code=getattr(exc, "error_code", "TL1006"),
                message=str(exc),
            )
        duration_ms = int((time.time() - started_at) * 1000)
        return result, duration_ms, {}

    async def _run_batch_concurrently(
        self,
        batch: list[ToolCall],
        tools: list[object | None],
        ctx: ToolUseContext,
        invocation_indices: list[int],
    ) -> list[tuple[ToolResult, int, dict[str, object]]]:
        semaphore = asyncio.Semaphore(self.max_read_concurrency)

        async def run_guarded(
            tool_call: ToolCall,
            tool: object | None,
            invocation_index: int,
        ) -> tuple[ToolResult, int, dict[str, object]]:
            if tool is None:
                return (
                    self.mapper.to_error_result(
                        tool_use_id=tool_call.tool_use_id,
                        tool_name=tool_call.tool_name,
                        error_code="TL1001",
                        message=f"Unknown tool: {tool_call.tool_name}",
                    ),
                    0,
                    {},
                )
            async with semaphore:
                return await self._run_one(tool_call, tool, ctx, invocation_index=invocation_index)

        return list(
            await asyncio.gather(
                *(
                    run_guarded(tool_call, tool, invocation_index)
                    for tool_call, tool, invocation_index in zip(batch, tools, invocation_indices)
                )
            )
        )

    @staticmethod
    def _record_tool_events(
        audit_logger: JsonlAuditLogger,
        ctx: ToolUseContext,
        events: list[ToolExecutionEvent],
    ) -> None:
        records: list[EventRecord] = []
        for event in events:
            details: dict[str, object] = dict(event.payload)
            details.setdefault("tool_use_id", event.tool_use_id)
            details.setdefault("tool_name", event.tool_name)
            if event.duration_ms is not None:
                details.setdefault("duration_ms", event.duration_ms)
            result = event.payload.get("result")
            if isinstance(result, ToolResult):
                details["result"] = {
                    "tool_use_id": result.tool_use_id,
                    "tool_name": result.tool_name,
                    "success": result.success,
                    "content": result.content,
                    "data": dict(result.data),
                    "error_code": result.error_code,
                }
            records.append(
                EventRecord(
                    event_type=event.event_type,
                    session_id=ctx.session_id,
                    turn_id=ctx.turn_id,
                    tool_name=event.tool_name or None,
                    success=event.success,
                    error_code=event.error_code,
                    details=details,
                )
            )
        audit_logger.append_many(records)
