from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
from pathlib import Path
import threading
from typing import Any
from typing import AsyncIterator

from .config import CCConfig, load_cc_config
from .conversation.models import SessionEvent, SessionMessage
from .conversation.session import QuerySession
from .llm import DefaultLLMClientProvider, LLMClientProvider
from .runtime import build_default_query_engine


@dataclass(slots=True)
class AgentRunRequest:
    instruction: str
    cwd: str = "."
    config: CCConfig | None = None
    session: QuerySession | None = None
    max_tool_rounds: int | None = None
    prompt_language: str | None = None
    permission_mode: str | None = None
    agent_mode: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    system_prompt_key: str | None = None
    system_prompt_context: dict[str, Any] = field(default_factory=dict)
    event_sink: Any | None = None


@dataclass(slots=True)
class AgentRunResult:
    final_text: str
    session_id: str
    turn_id: str | None
    cwd: str
    session_snapshot: dict[str, Any]
    events: list[SessionEvent] = field(default_factory=list)
    messages: list[SessionMessage] = field(default_factory=list)
    tool_call_count: int = 0
    failed: bool = False
    error_code: str | None = None
    error_message: str | None = None


@dataclass(slots=True)
class CodeBuildRequest:
    goal: str
    cwd: str = "."
    context_paths: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    config: CCConfig | None = None
    session: QuerySession | None = None
    max_tool_rounds: int | None = None
    prompt_language: str | None = None
    permission_mode: str | None = None
    agent_mode: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    event_sink: Any | None = None


class CodeAgent:
    """Primary public API for host integrations.

    External projects should prefer ``CodeAgent`` plus the request/result models
    in this module. Lower-level engine construction remains available for
    advanced control, but it is not the recommended default integration surface.
    """

    def __init__(
        self,
        *,
        config: CCConfig | None = None,
        llm_client_provider: LLMClientProvider | None = None,
    ) -> None:
        self.config = config
        self.llm_client_provider = llm_client_provider or DefaultLLMClientProvider()

    async def stream(self, request: AgentRunRequest) -> AsyncIterator[SessionEvent]:
        resolved_agent_mode = (
            request.agent_mode
            if request.agent_mode is not None
            else (self.config.agent_mode if self.config else "")
        )
        if resolved_agent_mode == "structured":
            raise NotImplementedError("Structured mode does not support streaming. Use run() instead.")
        engine = self._build_engine(request)
        try:
            async for event in self._stream_with_engine(engine, request):
                yield event
        finally:
            self._close_engine(engine)

    async def run(self, request: AgentRunRequest) -> AgentRunResult:
        resolved_agent_mode = (
            request.agent_mode
            if request.agent_mode is not None
            else (self.config.agent_mode if self.config else "")
        )
        if resolved_agent_mode == "structured":
            return await self._run_structured(request)

        events: list[SessionEvent] = []
        messages: list[SessionMessage] = []
        final_text = ""
        tool_call_count = 0
        turn_id: str | None = None
        session_id = ""
        resolved_cwd = request.cwd or ""
        session_snapshot: dict[str, Any] = {}
        engine = None

        try:
            resolved_cwd = str(Path(request.cwd).resolve())
            engine = self._build_engine(request)
            session_id = engine.session.session_id
            resolved_cwd = engine.session.cwd

            async for event in self._stream_with_engine(engine, request):
                events.append(event)
                if turn_id is None:
                    turn_id = event.turn_id
                if event.message is not None:
                    messages.append(event.message)
                    if event.message.role == "assistant" and event.message.kind == "assistant_text":
                        final_text = event.message.content
                if event.event_type == "assistant_tool_use":
                    tool_call_count += 1

            session_snapshot = engine.export_session_snapshot()
            return AgentRunResult(
                final_text=final_text,
                session_id=session_id,
                turn_id=turn_id,
                cwd=resolved_cwd,
                session_snapshot=session_snapshot,
                events=events,
                messages=messages,
                tool_call_count=tool_call_count,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if engine is not None:
                session_snapshot = engine.export_session_snapshot()
            return AgentRunResult(
                final_text=final_text,
                session_id=session_id,
                turn_id=turn_id,
                cwd=resolved_cwd,
                session_snapshot=session_snapshot,
                events=events,
                messages=messages,
                tool_call_count=tool_call_count,
                failed=True,
                error_code=getattr(exc, "error_code", None) or getattr(exc, "code", "CC1000"),
                error_message=str(exc),
            )
        finally:
            if engine is not None:
                self._close_engine(engine)

    def run_sync(self, request: AgentRunRequest) -> AgentRunResult:
        return _run_coro_sync(self.run(request))

    async def build_code(self, request: CodeBuildRequest) -> AgentRunResult:
        config = request.config or self.config or load_cc_config()
        resolved_agent_mode = (
            request.agent_mode
            if request.agent_mode is not None
            else config.agent_mode
        )
        prompt_key = self._resolve_build_prompt_key(resolved_agent_mode)
        return await self.run(
            AgentRunRequest(
                instruction=self._serialize_code_build_request(request),
                cwd=request.cwd,
                config=config,
                session=request.session,
                max_tool_rounds=request.max_tool_rounds,
                prompt_language=request.prompt_language,
                permission_mode=request.permission_mode,
                agent_mode=resolved_agent_mode,
                metadata=request.metadata,
                system_prompt_key=prompt_key,
                system_prompt_context={
                    "build_mode": True,
                    "agent_mode": resolved_agent_mode == "agent",
                    "spec_mode": resolved_agent_mode == "spec",
                    "plan_mode": resolved_agent_mode == "plan",
                },
                event_sink=request.event_sink,
            )
        )

    def build_code_sync(self, request: CodeBuildRequest) -> AgentRunResult:
        return _run_coro_sync(self.build_code(request))

    async def stream_build_code(self, request: CodeBuildRequest) -> AsyncIterator[SessionEvent]:
        config = request.config or self.config or load_cc_config()
        resolved_agent_mode = (
            request.agent_mode
            if request.agent_mode is not None
            else config.agent_mode
        )
        prompt_key = self._resolve_build_prompt_key(resolved_agent_mode)
        async for event in self.stream(
            AgentRunRequest(
                instruction=self._serialize_code_build_request(request),
                cwd=request.cwd,
                config=config,
                session=request.session,
                max_tool_rounds=request.max_tool_rounds,
                prompt_language=request.prompt_language,
                permission_mode=request.permission_mode,
                agent_mode=resolved_agent_mode,
                metadata=request.metadata,
                system_prompt_key=prompt_key,
                system_prompt_context={
                    "build_mode": True,
                    "agent_mode": resolved_agent_mode == "agent",
                    "spec_mode": resolved_agent_mode == "spec",
                    "plan_mode": resolved_agent_mode == "plan",
                },
                event_sink=request.event_sink,
            )
        ):
            yield event

    async def _run_structured(self, request: AgentRunRequest) -> AgentRunResult:
        from .structured_flow import StructuredFlowRunner

        config = request.config or self.config or load_cc_config()
        runner = StructuredFlowRunner(
            config=config,
            llm_client_provider=self.llm_client_provider,
        )
        result = await runner.run(
            instruction=request.instruction,
            cwd=request.cwd,
            prompt_language=request.prompt_language,
            permission_mode=request.permission_mode,
            event_sink=request.event_sink,
        )
        try:
            resolved_cwd = str(Path(request.cwd).resolve())
        except (OSError, ValueError):
            resolved_cwd = request.cwd or ""
        return AgentRunResult(
            final_text=result.output,
            session_id="",
            turn_id=None,
            cwd=resolved_cwd,
            session_snapshot={"structured_flow": True, "phase": result.phase},
            failed=not result.success,
            error_message=result.error,
            error_code="SF1001" if not result.success else None,
        )

    def _build_engine(self, request: AgentRunRequest):
        config = request.config or self.config or load_cc_config()
        session = request.session
        cwd = session.cwd if session is not None else str(Path(request.cwd).resolve())
        engine = build_default_query_engine(
            cwd=cwd,
            config=config,
            llm_client_provider=self.llm_client_provider,
            session=session,
        )
        if request.prompt_language:
            engine.session.prompt_language = request.prompt_language
        if request.permission_mode:
            engine.session.permission_mode = request.permission_mode
        if request.agent_mode is not None:
            engine.session.agent_mode = request.agent_mode
        return engine

    @staticmethod
    def _close_engine(engine: Any) -> None:
        close = getattr(engine, "close", None)
        if callable(close):
            close()

    async def _stream_with_engine(self, engine: Any, request: AgentRunRequest) -> AsyncIterator[SessionEvent]:
        state = engine.session.metadata.state
        if request.metadata:
            state.update(dict(request.metadata))
        if request.system_prompt_key:
            state["system_prompt_key"] = request.system_prompt_key
        if request.system_prompt_context:
            state["system_prompt_context"] = dict(request.system_prompt_context)

        async for event in engine.submit_message(request.instruction, max_tool_rounds=request.max_tool_rounds):
            if request.event_sink is not None:
                sink_result = request.event_sink(event)
                if asyncio.iscoroutine(sink_result):
                    await sink_result
            yield event

    @staticmethod
    def _resolve_build_prompt_key(agent_mode: str) -> str:
        # build_code keeps a construction-oriented prompt family; workflow modes override it explicitly
        if agent_mode == "spec":
            return "system.spec_mode"
        if agent_mode == "plan":
            return "system.plan_mode"
        if agent_mode == "agent":
            return "system.agent_mode"
        return "system.code_build"

    def _serialize_code_build_request(self, request: CodeBuildRequest) -> str:
        return json.dumps(
            {
                "goal": request.goal,
                "cwd": str(Path(request.cwd).resolve()),
                "context_paths": [str(Path(item).resolve()) for item in request.context_paths],
                "constraints": list(request.constraints),
                "acceptance_criteria": list(request.acceptance_criteria),
            },
            ensure_ascii=False,
            indent=2,
        )


def _run_coro_sync(coro: Any) -> Any:
    """Compatibility sync wrapper for async-first APIs.

    Prefer the async methods when integrating into applications that already own
    an event loop. This helper keeps backwards compatibility for scripts and
    simple synchronous callers.
    """

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    outcome: dict[str, Any] = {}

    def run_in_thread() -> None:
        try:
            outcome["result"] = asyncio.run(coro)
        except BaseException as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()
    thread.join(timeout=300)
    if thread.is_alive():
        raise RuntimeError("_run_coro_sync timed out after 300s — possible deadlock in nested event loop")
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("result")


def run_code_agent(
    instruction: str,
    *,
    cwd: str = ".",
    config: CCConfig | None = None,
    llm_client_provider: LLMClientProvider | None = None,
    prompt_language: str | None = None,
    permission_mode: str | None = None,
    agent_mode: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentRunResult:
    agent = CodeAgent(config=config, llm_client_provider=llm_client_provider)
    return agent.run_sync(
        AgentRunRequest(
            instruction=instruction,
            cwd=cwd,
            config=config,
            prompt_language=prompt_language,
            permission_mode=permission_mode,
            agent_mode=agent_mode,
            metadata=dict(metadata or {}),
        )
    )


def build_code_with_agent(
    goal: str,
    *,
    cwd: str = ".",
    context_paths: list[str] | None = None,
    constraints: list[str] | None = None,
    acceptance_criteria: list[str] | None = None,
    config: CCConfig | None = None,
    llm_client_provider: LLMClientProvider | None = None,
    prompt_language: str | None = None,
    permission_mode: str | None = None,
    agent_mode: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentRunResult:
    agent = CodeAgent(config=config, llm_client_provider=llm_client_provider)
    return agent.build_code_sync(
        CodeBuildRequest(
            goal=goal,
            cwd=cwd,
            context_paths=list(context_paths or []),
            constraints=list(constraints or []),
            acceptance_criteria=list(acceptance_criteria or []),
            config=config,
            prompt_language=prompt_language,
            permission_mode=permission_mode,
            agent_mode=agent_mode,
            metadata=dict(metadata or {}),
        )
    )
