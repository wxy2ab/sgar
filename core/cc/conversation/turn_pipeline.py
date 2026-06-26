from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from typing import Any

from .agent_mode_strategy import decide_agent_collaboration_strategy
from .context_assembler import ContextAssembler
from .llm_adapter import LLMAdapter
from .message_store import SessionMessageStore
from .mode_strategy import (
    build_paths_in_request_block,
    build_repository_outline,
    decide_mode_strategy,
)
from .models import TurnRecord
from .prompt_builder import SystemPromptBuilder
from .session import QuerySession
from ..llm import LLMClientProvider
from ..memory import MemoryRuntime
from ..plan import ensure_plan_state
from ..specs import ensure_spec_state


@dataclass(slots=True)
class PreparedTurn:
    session_state: dict[str, Any]
    tool_ctx: Any
    prompt_parts: Any
    llm_adapter: LLMAdapter
    effective_max_tool_rounds: int | None


class SessionStatePreparer:
    def prepare(
        self,
        *,
        session: QuerySession,
        user_input: str | list[dict[str, object]],
    ) -> dict[str, Any]:
        session_state = dict(session.metadata.state)
        if session.agent_mode == "spec":
            session_state = self._prepare_spec_state(session, session_state, user_input)
        elif session.agent_mode == "plan":
            session_state = self._prepare_plan_state(session, session_state, user_input)
        elif session.agent_mode == "agent":
            session_state = self._prepare_agent_mode_state(session, session_state, user_input)
        elif session.agent_mode in {"ask", "doc"}:
            session_state = self._prepare_repository_mode_state(session, session_state, user_input)
        session.metadata.state = dict(session_state)
        return session_state

    def _prepare_spec_state(
        self,
        session: QuerySession,
        session_state: dict[str, Any],
        user_input: str | list[dict[str, object]],
    ) -> dict[str, Any]:
        already_exited = session_state.get("spec_phase") in ("render",)
        source_text = user_input if isinstance(user_input, str) else session.session_id
        return ensure_spec_state(
            current_state=session_state,
            cwd=session.cwd,
            config=session.config,
            source_text=source_text,
            enabled=not already_exited,
        )

    def _prepare_plan_state(
        self,
        session: QuerySession,
        session_state: dict[str, Any],
        user_input: str | list[dict[str, object]],
    ) -> dict[str, Any]:
        already_exited = session_state.get("plan_phase") == "implementation"
        source_text = user_input if isinstance(user_input, str) else session.session_id
        return ensure_plan_state(
            current_state=session_state,
            cwd=session.cwd,
            config=session.config,
            source_text=source_text,
            enabled=not already_exited,
        )

    def _prepare_agent_mode_state(
        self,
        session: QuerySession,
        session_state: dict[str, Any],
        user_input: str | list[dict[str, object]],
    ) -> dict[str, Any]:
        prompt_context = dict(session_state.get("system_prompt_context") or {})
        for key in (
            "agent_collaboration_strategy",
            "agent_collaboration_required",
            "agent_collaboration_pattern",
            "agent_collaboration_roles",
            "agent_collaboration_plan",
            "agent_collaboration_completed",
            "agent_collaboration_count",
        ):
            prompt_context.pop(key, None)
        strategy = decide_agent_collaboration_strategy(
            session.agent_mode,
            user_input,
            build_mode=bool(prompt_context.get("build_mode")),
        )
        prompt_context["agent_collaboration_strategy"] = strategy
        prompt_context["agent_collaboration_required"] = bool(strategy.get("must_delegate"))
        prompt_context["agent_collaboration_pattern"] = strategy.get("pattern")
        prompt_context["agent_collaboration_roles"] = list(strategy.get("roles", []))
        prompt_context["agent_collaboration_plan"] = list(strategy.get("delegation_plan", []))
        prompt_context["agent_collaboration_completed"] = bool(
            session_state.get("agent_collaboration_completed"),
        )
        prompt_context["agent_collaboration_count"] = int(session_state.get("agent_collaboration_count", 0) or 0)
        session_state["system_prompt_context"] = prompt_context
        return session_state

    def _prepare_repository_mode_state(
        self,
        session: QuerySession,
        session_state: dict[str, Any],
        user_input: str | list[dict[str, object]],
    ) -> dict[str, Any]:
        prompt_context = dict(session_state.get("system_prompt_context") or {})
        for key in (
            "mode_strategy",
            "repository_outline",
            "repository_outline_text",
            "repository_outline_enabled",
            "paths_in_request_text",
        ):
            prompt_context.pop(key, None)
        strategy = decide_mode_strategy(session.agent_mode, user_input)
        prompt_context["mode_strategy"] = strategy
        prompt_context["repository_outline_enabled"] = strategy["use_repository_outline"]
        if strategy["use_repository_outline"]:
            outline = build_repository_outline(
                session.cwd,
                max_depth=4 if session.agent_mode == "doc" else 3,
                max_entries_per_dir=8 if session.agent_mode == "doc" else 6,
            )
            prompt_context["repository_outline"] = outline
            prompt_context["repository_outline_text"] = outline["text"]
        # ``paths_in_request_text`` is independent of outline injection:
        # the LLM benefits from a verified-paths block whether or not we
        # also expose the truncated outline. When the user didn't name
        # any path tokens this returns ``""`` and prompt_builder skips
        # the block.
        paths_block = build_paths_in_request_block(user_input, session.cwd)
        if paths_block:
            prompt_context["paths_in_request_text"] = paths_block
        session_state["system_prompt_context"] = prompt_context
        return session_state


class TurnPromptAssembler:
    def __init__(
        self,
        *,
        prompt_builder: SystemPromptBuilder,
        context_assembler: ContextAssembler,
        tool_orchestrator: object,
        llm_client_provider: LLMClientProvider,
        memory_runtime: MemoryRuntime | None = None,
    ) -> None:
        self.prompt_builder = prompt_builder
        self.context_assembler = context_assembler
        self.tool_orchestrator = tool_orchestrator
        self.llm_client_provider = llm_client_provider
        self.memory_runtime = memory_runtime

    def prepare_turn(
        self,
        *,
        session: QuerySession,
        turn_id: str,
        session_state: dict[str, Any],
        max_tool_rounds: int | None,
        default_prompt_key: str | None,
        user_input: str | list[dict[str, object]] | None = None,
        extra_prompt_context: dict[str, Any] | None = None,
        purpose: str = "query_engine.turn",
    ) -> PreparedTurn:
        tool_ctx = self.context_assembler.build_tool_context(session=session, turn_id=turn_id)
        llm_client = self.llm_client_provider.get_client(
            config=session.config,
            purpose=purpose,
        )
        if self.memory_runtime is not None and user_input is not None:
            self.memory_runtime.before_turn(session=session, user_input=user_input)
        prompt_parts = self._build_prompt_parts(
            session=session,
            session_state=session_state,
            tool_ctx=tool_ctx,
            llm_client=llm_client,
            default_prompt_key=default_prompt_key,
            extra_prompt_context=extra_prompt_context,
        )
        return PreparedTurn(
            session_state=session_state,
            tool_ctx=tool_ctx,
            prompt_parts=prompt_parts,
            llm_adapter=LLMAdapter(
                llm_client,
                timeout_seconds=session.config.llm_request_timeout_seconds,
            ),
            effective_max_tool_rounds=(
                session.config.max_tool_rounds if max_tool_rounds is None else max_tool_rounds
            ),
        )

    def _build_prompt_parts(
        self,
        *,
        session: QuerySession,
        session_state: dict[str, Any],
        tool_ctx: Any,
        llm_client: Any,
        default_prompt_key: str | None,
        extra_prompt_context: dict[str, Any] | None = None,
    ) -> Any:
        enabled_tools = self._list_enabled_tools(tool_ctx)
        prompt_extra = dict(session_state.get("system_prompt_context") or {})
        if hasattr(llm_client, "tool_invoke"):
            prompt_extra["native_tool_calling"] = True
        if extra_prompt_context:
            prompt_extra.update(extra_prompt_context)
        return self.prompt_builder.build(
            prompt_language=session.prompt_language,
            prompt_key=session_state.get("system_prompt_key") or default_prompt_key,
            context=self.context_assembler.build_prompt_context(
                session=session,
                tool_ctx=tool_ctx,
                enabled_tools=enabled_tools,
                extra=prompt_extra,
            ),
        ).parts

    def _list_enabled_tools(self, tool_ctx: Any) -> list[str]:
        registry = getattr(self.tool_orchestrator, "registry", None)
        if registry is None:
            return []
        return [tool.spec.name for tool in registry.list_visible(tool_ctx)]


class TurnPersistenceCoordinator:
    def __init__(
        self,
        *,
        session: QuerySession,
        message_store: SessionMessageStore,
        session_snapshot_interval: int = 5,
    ) -> None:
        self.session = session
        self.message_store = message_store
        self.session_snapshot_interval = max(1, session_snapshot_interval)

    def _write_session_json_sync(self, payload: dict[str, object]) -> None:
        session_dir = self.session.session_dir
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "session.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    async def persist_session(self, *, extra: dict[str, object] | None = None) -> None:
        payload = self.session.to_dict()
        if extra:
            payload.update(extra)
        await asyncio.gather(
            asyncio.to_thread(self._write_session_json_sync, payload),
            self.message_store.persist(),
        )

    def _append_turn_record_sync(self, record: TurnRecord) -> None:
        turns_path = self.session.session_dir / "turns.jsonl"
        turns_path.parent.mkdir(parents=True, exist_ok=True)
        with turns_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False))
            handle.write("\n")

    async def append_turn_record(self, record: TurnRecord) -> None:
        await asyncio.to_thread(self._append_turn_record_sync, record)

    async def persist_turn_end(
        self,
        record: TurnRecord,
        *,
        extra: dict[str, object] | None = None,
    ) -> None:
        await asyncio.gather(
            self.append_turn_record(record),
            self.message_store.persist(),
        )
        if self._should_persist_session_snapshot(record, extra=extra):
            payload = self.session.to_dict()
            if extra:
                payload.update(extra)
            await asyncio.to_thread(self._write_session_json_sync, payload)

    def _should_persist_session_snapshot(
        self,
        record: TurnRecord,
        *,
        extra: dict[str, object] | None = None,
    ) -> bool:
        if extra:
            return True
        if record.state != "completed":
            return True
        counter = int(self.session.metadata.state.get("session_snapshot_counter", 0) or 0) + 1
        self.session.metadata.state["session_snapshot_counter"] = counter
        return counter % self.session_snapshot_interval == 0
