from __future__ import annotations

from collections.abc import AsyncIterator
import time
import uuid

from .context_assembler import ContextAssembler
from ..llm import LLMClientProvider
from ..memory import MemoryRuntime
from .compact import SessionCompactor
from .message_store import SessionMessageStore
from .models import SessionEvent, SessionMessage, TurnRecord
from ..observability import EventRecord, JsonlAuditLogger
from .prompt_builder import SystemPromptBuilder
from .query_loop import run_single_turn
from .session import QuerySession, TurnState
from .turn_pipeline import (
    SessionStatePreparer,
    TurnPersistenceCoordinator,
    TurnPromptAssembler,
)


class QueryEngine:
    _SESSION_SNAPSHOT_INTERVAL = 5

    def __init__(
        self,
        session: QuerySession,
        message_store: SessionMessageStore,
        prompt_builder: SystemPromptBuilder,
        tool_orchestrator: object,
        llm_client_provider: LLMClientProvider,
        compactor: SessionCompactor | None = None,
        context_assembler: ContextAssembler | None = None,
        state_preparer: SessionStatePreparer | None = None,
        prompt_assembler: TurnPromptAssembler | None = None,
        persistence_coordinator: TurnPersistenceCoordinator | None = None,
        memory_runtime: MemoryRuntime | None = None,
    ) -> None:
        self.session = session
        self.message_store = message_store
        self.prompt_builder = prompt_builder
        self.tool_orchestrator = tool_orchestrator
        self.llm_client_provider = llm_client_provider
        self.compactor = compactor or SessionCompactor()
        self.context_assembler = context_assembler or ContextAssembler()
        self.state_preparer = state_preparer or SessionStatePreparer()
        self.prompt_assembler = prompt_assembler or TurnPromptAssembler(
            prompt_builder=self.prompt_builder,
            context_assembler=self.context_assembler,
            tool_orchestrator=self.tool_orchestrator,
            llm_client_provider=self.llm_client_provider,
        )
        self.persistence = persistence_coordinator or TurnPersistenceCoordinator(
            session=self.session,
            message_store=self.message_store,
            session_snapshot_interval=self._SESSION_SNAPSHOT_INTERVAL,
        )
        self.memory_runtime = memory_runtime
        self.audit_logger = JsonlAuditLogger(
            self.session.config.runtime_root_path(self.session.cwd) / "audit" / "session_events.jsonl"
        )
        self.turn_state = TurnState.IDLE

    def close(self) -> None:
        """No-op retained for lifecycle-API compatibility with callers."""

    async def submit_message(
        self,
        user_input: str | list[dict[str, object]],
        *,
        max_tool_rounds: int | None = None,
        purpose: str = "query_engine.turn",
    ) -> AsyncIterator[SessionEvent]:
        turn_id = f"turn_{uuid.uuid4().hex[:10]}"
        self.session.active_turn_id = turn_id
        self.turn_state = TurnState.RUNNING
        tool_call_count = 0
        continue_count = 0
        final_assistant_text = ""
        compact_applied = False
        timed_out = False
        try:
            compact_event = self._build_compaction_event(turn_id)
            compact_applied = compact_event is not None
            if compact_event is not None:
                self._store_compaction_memory(turn_id, compact_event)
                self._record_session_event(compact_event)
                yield compact_event
            prepared_turn = self._prepare_turn(
                turn_id=turn_id,
                user_input=user_input,
                max_tool_rounds=max_tool_rounds,
                purpose=purpose,
            )
            self._record_memory_recall_event(turn_id)
            turn_timeout = self.session.config.max_turn_timeout_seconds
            turn_start = time.monotonic()
            async for event in run_single_turn(
                session=self.session,
                turn_id=turn_id,
                user_input=user_input,
                llm_adapter=prepared_turn.llm_adapter,
                prompt_parts=prepared_turn.prompt_parts,
                tool_orchestrator=self.tool_orchestrator,
                tool_ctx=prepared_turn.tool_ctx,
                prompt_catalog=self.prompt_builder.prompt_catalog,
                context_assembler=self.context_assembler,
                prompt_builder=self.prompt_builder,
                max_tool_rounds=prepared_turn.effective_max_tool_rounds,
            ):
                if event.message is not None:
                    self.message_store.append(event.message)
                    try:
                        continue_count = max(continue_count, int(event.message.metadata.get("continue_count", 0) or 0))
                    except (ValueError, TypeError):
                        pass
                    if event.message.role == "assistant" and event.message.kind == "assistant_text":
                        final_assistant_text = str(event.message.content)
                if event.event_type == "assistant_tool_use":
                    tool_call_count += 1
                self._record_session_event(event)
                yield event
                if turn_timeout is not None and (time.monotonic() - turn_start) >= turn_timeout:
                    elapsed = int(time.monotonic() - turn_start)
                    timeout_event = self._build_turn_timeout_event(
                        turn_id=turn_id, elapsed_seconds=elapsed, limit_seconds=int(turn_timeout),
                    )
                    if timeout_event.message is not None:
                        self.message_store.append(timeout_event.message)
                    self._record_session_event(timeout_event)
                    yield timeout_event
                    timed_out = True
                    break
        except Exception as exc:
            failed_event = self._build_turn_failed_event(turn_id=turn_id, exc=exc)
            self._record_session_event(failed_event)
            yield failed_event
            await self._finalize_turn_failure(
                turn_id=turn_id,
                compact_applied=compact_applied,
                tool_call_count=tool_call_count,
                continue_count=continue_count,
                exc=exc,
            )
            raise
        else:
            if timed_out:
                self.turn_state = TurnState.FAILED
                self.session.active_turn_id = None
                if self.session.config.persist_sessions:
                    record = TurnRecord(
                        turn_id=turn_id,
                        state=self.turn_state.value,
                        tool_call_count=tool_call_count,
                        compact_applied=compact_applied,
                        continue_count=continue_count,
                        error_code="QE1008",
                    )
                    await self.persistence.persist_turn_end(
                        record,
                        extra={
                            "last_error": "turn_timeout",
                            "last_error_code": "QE1008",
                            "last_error_message": "Turn timed out",
                        },
                    )
            else:
                await self._finalize_turn_success(
                    turn_id=turn_id,
                    compact_applied=compact_applied,
                    tool_call_count=tool_call_count,
                    continue_count=continue_count,
                    final_assistant_text=final_assistant_text,
                )

    async def abort_active_turn(self, reason: str) -> None:
        active_turn_id = self.session.active_turn_id
        self.turn_state = TurnState.ABORTED
        self.session.active_turn_id = None
        self.session.touch()
        if self.session.config.persist_sessions:
            if active_turn_id:
                await self.persistence.append_turn_record(
                    TurnRecord(
                        turn_id=active_turn_id,
                        state=self.turn_state.value,
                        error_code="QE1007",
                    )
                )
            await self.persistence.persist_session(extra={"abort_reason": reason})

    async def restore_from_disk(self) -> None:
        await self.message_store.load_from_disk()

    def export_session_snapshot(self) -> dict[str, object]:
        return {
            "session": self.session.to_dict(),
            "message_count": len(self.message_store.snapshot()),
            "latest_compact_summary": self.session.metadata.state.get("latest_compact_summary"),
            "turn_state": self.turn_state.value,
        }

    def _resolve_default_prompt_key(self) -> str | None:
        if self.session.agent_mode == "spec":
            return "system.spec_mode"
        if self.session.agent_mode == "plan":
            return "system.plan_mode"
        if self.session.agent_mode == "agent":
            return "system.agent_mode"
        if self.session.agent_mode == "ask":
            return "system.ask_mode"
        if self.session.agent_mode == "doc":
            return "system.doc_mode"
        return None

    def _build_compaction_event(self, turn_id: str) -> SessionEvent | None:
        if not self.compactor or not self.compactor.should_compact(self.session, self.message_store):
            return None
        compact_result = self.compactor.compact(self.session, self.message_store)
        if not compact_result.applied or compact_result.boundary_message is None:
            return None
        return SessionEvent(
            event_type="compact_applied",
            turn_id=turn_id,
            message=compact_result.boundary_message,
            payload={"compacted_count": compact_result.compacted_count},
        )

    def _prepare_turn(
        self,
        *,
        turn_id: str,
        user_input: str | list[dict[str, object]],
        max_tool_rounds: int | None,
        purpose: str = "query_engine.turn",
    ):
        session_state = self.state_preparer.prepare(
            session=self.session,
            user_input=user_input,
        )
        return self.prompt_assembler.prepare_turn(
            session=self.session,
            turn_id=turn_id,
            session_state=session_state,
            max_tool_rounds=max_tool_rounds,
            default_prompt_key=self._resolve_default_prompt_key(),
            user_input=user_input,
            extra_prompt_context=None,
            purpose=purpose,
        )

    async def _finalize_turn_failure(
        self,
        *,
        turn_id: str,
        compact_applied: bool,
        tool_call_count: int,
        continue_count: int,
        exc: BaseException,
    ) -> None:
        self.turn_state = TurnState.FAILED
        self.session.active_turn_id = None
        error_code, error_message = self._error_details(exc)
        if self.session.config.persist_sessions:
            record = TurnRecord(
                turn_id=turn_id,
                state=self.turn_state.value,
                tool_call_count=tool_call_count,
                compact_applied=compact_applied,
                continue_count=continue_count,
                error_code=error_code,
            )
            await self.persistence.persist_turn_end(
                record,
                extra={
                    "last_error": "turn_failed",
                    "last_error_code": error_code,
                    "last_error_message": error_message,
                },
            )

    async def _finalize_turn_success(
        self,
        *,
        turn_id: str,
        compact_applied: bool,
        tool_call_count: int,
        continue_count: int,
        final_assistant_text: str,
    ) -> None:
        self.turn_state = TurnState.COMPLETED
        self.session.active_turn_id = None
        self._store_turn_memory(turn_id, final_assistant_text)
        if self.session.config.persist_sessions:
            record = TurnRecord(
                turn_id=turn_id,
                state=self.turn_state.value,
                tool_call_count=tool_call_count,
                compact_applied=compact_applied,
                continue_count=continue_count,
            )
            await self.persistence.persist_turn_end(record)

    @staticmethod
    def _build_turn_timeout_event(*, turn_id: str, elapsed_seconds: int, limit_seconds: int) -> SessionEvent:
        return SessionEvent(
            event_type="assistant_followup_completed",
            turn_id=turn_id,
            message=SessionMessage(
                message_id=f"msg_{uuid.uuid4().hex[:10]}",
                turn_id=turn_id,
                role="assistant",
                content=(
                    f"Turn global timeout reached: {elapsed_seconds}s elapsed "
                    f"(limit {limit_seconds}s). Stopping."
                ),
                kind="assistant_text",
                metadata={
                    "turn_timeout_reached": True,
                    "elapsed_seconds": elapsed_seconds,
                    "limit_seconds": limit_seconds,
                    "exit_reason": "turn_timeout",
                },
            ),
        )

    def _build_turn_failed_event(self, *, turn_id: str, exc: BaseException) -> SessionEvent:
        error_code, error_message = self._error_details(exc)
        return SessionEvent(
            event_type="turn_failed",
            turn_id=turn_id,
            payload={
                "error_code": error_code,
                "error_message": error_message,
                "turn_state": TurnState.FAILED.value,
            },
        )

    @staticmethod
    def _error_details(exc: BaseException) -> tuple[str, str]:
        error_code = getattr(exc, "error_code", None) or getattr(exc, "code", "QE1001")
        return str(error_code), str(exc)

    def _record_session_event(self, event: SessionEvent) -> None:
        details: dict[str, object] = dict(event.payload)
        if event.message is not None:
            details.setdefault("message_id", event.message.message_id)
            details.setdefault("message_role", event.message.role)
            details.setdefault("message_kind", event.message.kind)
            if event.message.tool_name:
                details.setdefault("tool_name", event.message.tool_name)
            if event.message.metadata:
                details.setdefault("message_metadata", dict(event.message.metadata))
        success: bool | None = None
        error_code = None
        if event.event_type in {"assistant_completed", "assistant_followup_completed"}:
            success = True
        elif event.event_type == "turn_failed":
            success = False
            error_code = str(event.payload.get("error_code") or "QE1001")
        elif event.event_type in {
            "memory_recall_completed",
            "memory_provider_unavailable",
            "memory_store_completed",
            "memory_store_failed",
        }:
            payload_success = event.payload.get("success")
            success = bool(payload_success) if payload_success is not None else None
            payload_error_code = event.payload.get("error_code")
            error_code = str(payload_error_code) if payload_error_code else None
        self.audit_logger.append(
            EventRecord(
                event_type=event.event_type,
                session_id=self.session.session_id,
                turn_id=event.turn_id,
                tool_name=str(details.get("tool_name")) if details.get("tool_name") else None,
                success=success,
                error_code=error_code,
                details=details,
            )
        )

    def _store_turn_memory(self, turn_id: str, final_assistant_text: str) -> None:
        if self.memory_runtime is None or not final_assistant_text.strip():
            return
        results = self.memory_runtime.after_turn(
            session=self.session,
            assistant_text=final_assistant_text,
        )
        self.session.metadata.state["last_memory_store_results"] = [
            {
                "success": item.success,
                "stored": item.stored,
                "duplicate": item.duplicate,
                "memory_id": item.memory_id,
                "message": item.message,
            }
            for item in results
        ]
        self._record_memory_store_event(turn_id, results, source="turn")

    def _store_compaction_memory(self, turn_id: str, compact_event: SessionEvent) -> None:
        if self.memory_runtime is None or compact_event.message is None:
            return
        result = self.memory_runtime.after_compaction(
            session=self.session,
            compact_summary=str(compact_event.message.content),
        )
        if result is None:
            return
        self.session.metadata.state["last_compaction_memory_result"] = {
            "success": result.success,
            "stored": result.stored,
            "duplicate": result.duplicate,
            "memory_id": result.memory_id,
            "message": result.message,
        }
        self._record_memory_store_event(turn_id, [result], source="compaction")

    def _record_memory_recall_event(self, turn_id: str) -> None:
        if self.memory_runtime is None:
            return
        memory_status = dict(self.session.metadata.state.get("memory_status") or {})
        memory_context = dict(self.session.metadata.state.get("memory_context") or {})
        if not memory_status and not memory_context:
            return
        available = bool(memory_status.get("available", memory_context.get("available", False)))
        details = {
            "provider": self.session.metadata.state.get("memory_provider"),
            "query": memory_context.get("query"),
            "hit_count": len(memory_context.get("hits") or []),
            "fact_count": len(memory_context.get("facts") or []),
            "summary": memory_context.get("summary"),
            "error": memory_status.get("error") or memory_context.get("error"),
        }
        event = SessionEvent(
            event_type="memory_recall_completed" if available else "memory_provider_unavailable",
            turn_id=turn_id,
            payload={
                "success": available,
                "error_code": None if available else "MM1001",
                **details,
            },
        )
        self._record_session_event(event)

    def _record_memory_store_event(self, turn_id: str, results: list[object], *, source: str) -> None:
        if self.memory_runtime is None:
            return
        if not results:
            event = SessionEvent(
                event_type="memory_store_completed",
                turn_id=turn_id,
                payload={
                    "success": True,
                    "provider": self.session.metadata.state.get("memory_provider"),
                    "source": source,
                    "result_count": 0,
                    "stored_count": 0,
                    "duplicate_count": 0,
                },
            )
            self._record_session_event(event)
            return
        success = all(bool(getattr(item, "success", False)) for item in results)
        event = SessionEvent(
            event_type="memory_store_completed" if success else "memory_store_failed",
            turn_id=turn_id,
            payload={
                "success": success,
                "error_code": None if success else "MM1002",
                "provider": self.session.metadata.state.get("memory_provider"),
                "source": source,
                "result_count": len(results),
                "stored_count": sum(1 for item in results if getattr(item, "stored", False)),
                "duplicate_count": sum(1 for item in results if getattr(item, "duplicate", False)),
                "memory_ids": [getattr(item, "memory_id", None) for item in results if getattr(item, "memory_id", None)],
                "messages": [getattr(item, "message", "") for item in results if getattr(item, "message", "")],
            },
        )
        self._record_session_event(event)
