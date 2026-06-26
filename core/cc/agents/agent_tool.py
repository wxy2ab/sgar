from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Any
import asyncio

from ..agents.backends import InProcessBackend, LocalSubprocessBackend, RuntimeBackend
from ..config import CCConfig
from ..conversation.session import QuerySession, SessionFactory
from ..llm import DefaultLLMClientProvider, LLMClientProvider
from ..observability import EventRecord, JsonlAuditLogger
from ..tools.base import BaseTool, ToolCall, ToolResult, ToolSpec, ValidationResult
from ..tools.context import ToolUseContext
from .definitions import AgentDefinition
from .runtime import AgentRuntime
from .runtime_registry import InProcessRuntimeRegistry, get_in_process_runtime_registry
from .task_manager import TaskManager
from .task_model import AgentTask, AgentTaskStatus


logger = logging.getLogger(__name__)


# Maximum recursion depth for the ``agent`` tool. With cc_query_loop as the
# default ccx runner kind, every agent node can call ``agent`` to spawn a
# child runtime, which itself can call ``agent``, and so on. Each level
# multiplies the cost by the per-node fan-out, so an unconstrained recursion
# of depth 4 with fan-out 3 is already 81 leaves. The cap here is the only
# mechanical (non-prompt) backstop — LLMs can't reliably stay within a
# nominal budget via prompt instructions, but the runtime can refuse the
# call.
#
# Depth 0 = the top-level cc engine driven by ccx's CcAgentRunner; depth 1
# = its first helper; depth 2 = a helper-of-helper. The default of 3 keeps
# room for a real "lead → researcher → reviewer" pattern without permitting
# unbounded chains. Override via the ``CC_MAX_AGENT_RECURSION_DEPTH``
# environment variable for unusual workloads.
_DEFAULT_MAX_AGENT_RECURSION_DEPTH = 3
_RECURSION_DEPTH_STATE_KEY = "agent_recursion_depth"
_AGENT_SPAWN_REFUSED_ERROR_CODE = "AT1100"


def _resolve_max_recursion_depth() -> int:
    raw = os.environ.get("CC_MAX_AGENT_RECURSION_DEPTH")
    if not raw:
        return _DEFAULT_MAX_AGENT_RECURSION_DEPTH
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "CC_MAX_AGENT_RECURSION_DEPTH=%r is not an int; using default %d",
            raw, _DEFAULT_MAX_AGENT_RECURSION_DEPTH,
        )
        return _DEFAULT_MAX_AGENT_RECURSION_DEPTH
    if value < 0:
        logger.warning(
            "CC_MAX_AGENT_RECURSION_DEPTH=%d is negative; using default %d",
            value, _DEFAULT_MAX_AGENT_RECURSION_DEPTH,
        )
        return _DEFAULT_MAX_AGENT_RECURSION_DEPTH
    return value


@dataclass(slots=True)
class AgentToolRequest:
    description: str
    prompt: str
    subagent_type: str | None = None
    backend: str | None = None
    model: str | None = None
    run_in_background: bool = False
    name: str | None = None
    team_name: str | None = None
    mode: str | None = None
    isolation: str | None = None
    cwd: str | None = None


class AgentTool(BaseTool):
    def __init__(
        self,
        llm_client_provider: LLMClientProvider | None = None,
        runtime_registry: InProcessRuntimeRegistry | None = None,
    ) -> None:
        super().__init__(
            ToolSpec(
                name="agent",
                description="Spawn a child agent to work on a sub-task and return its result.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "prompt": {"type": "string"},
                        "subagent_type": {"type": "string"},
                        "backend": {"type": "string"},
                        "run_in_background": {"type": "boolean"},
                        "cwd": {"type": "string"},
                    },
                    "required": ["description", "prompt"],
                },
            )
        )
        self.llm_client_provider = llm_client_provider or DefaultLLMClientProvider()
        self.runtime_registry = runtime_registry

    def validate_input(self, arguments: dict[str, Any]) -> ValidationResult:
        if not arguments.get("description"):
            return ValidationResult(ok=False, message="description is required.")
        if not arguments.get("prompt"):
            return ValidationResult(ok=False, message="prompt is required.")
        backend = arguments.get("backend")
        if backend and str(backend) not in {"in_process", "local_subprocess"}:
            return ValidationResult(ok=False, message=f"Unsupported backend: {backend}")
        return ValidationResult(ok=True)

    async def execute(self, tool_call: ToolCall, ctx: ToolUseContext) -> ToolResult:
        request = AgentToolRequest(
            description=str(tool_call.arguments["description"]),
            prompt=str(tool_call.arguments["prompt"]),
            subagent_type=tool_call.arguments.get("subagent_type"),
            backend=tool_call.arguments.get("backend"),
            run_in_background=bool(tool_call.arguments.get("run_in_background", False)),
            cwd=tool_call.arguments.get("cwd"),
        )
        # Recursion depth check (Tier 1 backstop against runaway sub-agent
        # spawning when cc_query_loop is the default ccx runner). Read the
        # parent's depth from session metadata.state, refuse if we'd exceed
        # the cap, and emit an audit event either way so deep chains are
        # debuggable post-hoc.
        parent_session = ctx.metadata.get("session")
        parent_depth = 0
        if parent_session is not None:
            parent_depth = int(
                (parent_session.metadata.state or {}).get(
                    _RECURSION_DEPTH_STATE_KEY, 0,
                ) or 0
            )
        max_depth = _resolve_max_recursion_depth()
        runtime_root_for_audit = ctx.config.runtime_root_path(ctx.cwd)
        agent_audit_logger = JsonlAuditLogger(
            runtime_root_for_audit / "audit" / "agent_events.jsonl"
        )
        prospective_child_depth = parent_depth + 1
        if prospective_child_depth > max_depth:
            agent_audit_logger.append(EventRecord(
                event_type="agent_spawn_refused_recursion_cap",
                session_id=getattr(parent_session, "session_id", None),
                tool_name="agent",
                success=False,
                error_code=_AGENT_SPAWN_REFUSED_ERROR_CODE,
                details={
                    "parent_depth": parent_depth,
                    "prospective_child_depth": prospective_child_depth,
                    "max_depth": max_depth,
                    "description": (request.description or "")[:200],
                    "tool_use_id": tool_call.tool_use_id,
                },
            ))
            logger.warning(
                "agent tool refused: parent_depth=%d would exceed max_depth=%d "
                "(description=%r). Set CC_MAX_AGENT_RECURSION_DEPTH to raise the cap.",
                parent_depth, max_depth, (request.description or "")[:120],
            )
            return ToolResult(
                tool_use_id=tool_call.tool_use_id,
                tool_name=tool_call.tool_name,
                success=False,
                content=(
                    f"agent spawn refused: recursion depth {prospective_child_depth} "
                    f"would exceed cap {max_depth}. Complete this task with the "
                    f"context you already have, or finalize and return."
                ),
                error_code=_AGENT_SPAWN_REFUSED_ERROR_CODE,
                data={
                    "parent_depth": parent_depth,
                    "max_depth": max_depth,
                    "refused": True,
                },
            )
        # Approaching-cap warning (one less than max). Distinct event type
        # so dashboards can flag chains that get close without firing on
        # every spawn.
        if prospective_child_depth == max_depth:
            agent_audit_logger.append(EventRecord(
                event_type="agent_collaboration_depth_warning",
                session_id=getattr(parent_session, "session_id", None),
                tool_name="agent",
                success=None,
                details={
                    "parent_depth": parent_depth,
                    "child_depth": prospective_child_depth,
                    "max_depth": max_depth,
                    "description": (request.description or "")[:200],
                    "tool_use_id": tool_call.tool_use_id,
                },
            ))

        definition = self.resolve_agent_definition(request)
        backend_name = self.resolve_backend_name(request, ctx)
        backend = self.resolve_backend(backend_name)
        runtime_root = ctx.config.runtime_root_path(ctx.cwd)
        runtime_registry = self.runtime_registry or get_in_process_runtime_registry(runtime_root)
        task_manager = TaskManager(runtime_root / "tasks")
        task = task_manager.create_task(
            AgentTask.create(
                agent_type=definition.agent_id,
                backend=backend_name,
                prompt_language=ctx.prompt_language,
                title=request.description,
                input_payload={
                    "description": request.description,
                    "prompt": request.prompt,
                },
            )
        )
        child_session = self.build_child_session(
            parent_session=ctx.metadata["session"],
            task=task,
            agent_definition=definition,
            cwd=request.cwd or ctx.cwd,
        )
        # Stamp the child's depth so its own ``agent`` tool calls see a
        # bumped value and the cap composes across levels.
        child_session.metadata.state[_RECURSION_DEPTH_STATE_KEY] = prospective_child_depth
        from ..runtime import build_default_query_engine

        runtime = AgentRuntime(
            definition=definition,
            task=task,
            query_engine=build_default_query_engine(
                cwd=child_session.cwd,
                config=child_session.config,
                llm_client_provider=self.llm_client_provider,
                session=child_session,
            ),
            task_manager=task_manager,
        )
        controller = await backend.create_controller(
            runtime=runtime,
            run_in_background=request.run_in_background,
            runtime_root=runtime_root,
        )
        runtime_registry.register(controller)
        if request.run_in_background:
            if backend_name == "in_process":
                task_manager.update_task_status(task.task_id, AgentTaskStatus.RUNNING)
                background_task = asyncio.create_task(controller.start(request.prompt))
                runtime_registry.register_background_task(task.runtime_id, background_task)
                launch_result = {
                    "task_id": task.task_id,
                    "runtime_id": task.runtime_id,
                    "status": task.status.value,
                    "background": True,
                    "backend": backend_name,
                }
            else:
                launch_result = await controller.start(request.prompt)
            return ToolResult(
                tool_use_id=tool_call.tool_use_id,
                tool_name=tool_call.tool_name,
                success=True,
                content="Agent launched in background mode.",
                data=launch_result,
            )
        result = await controller.start(request.prompt)
        return ToolResult(
            tool_use_id=tool_call.tool_use_id,
            tool_name=tool_call.tool_name,
            success=True,
            content=result["final_text"],
            data=result,
        )

    def resolve_backend_name(self, request: AgentToolRequest, ctx: ToolUseContext) -> str:
        return str(request.backend or request.mode or ctx.config.default_backend or "in_process")

    def resolve_backend(self, backend_name: str) -> RuntimeBackend:
        if backend_name == "in_process":
            return InProcessBackend()
        if backend_name == "local_subprocess":
            return LocalSubprocessBackend()
        raise ValueError(f"Unsupported backend: {backend_name}")

    def resolve_agent_definition(self, request: AgentToolRequest) -> AgentDefinition:
        agent_id = request.subagent_type or "worker"
        return AgentDefinition(
            agent_id=agent_id,
            name=request.name or agent_id,
            description=request.description,
            prompt_key="agents.worker",
        )

    def build_child_session(
        self,
        *,
        parent_session: QuerySession,
        task: AgentTask,
        agent_definition: AgentDefinition,
        cwd: str,
    ) -> QuerySession:
        child_config = CCConfig.from_mapping(
            {
                **parent_session.config.to_dict(),
                "agent_mode": "",
            }
        )
        child_factory = SessionFactory(child_config)
        child_session = child_factory.create(
            cwd=cwd,
            model_name=parent_session.model_name,
            agent_id=agent_definition.agent_id,
            parent_task_id=task.task_id,
        )
        child_session.prompt_language = parent_session.prompt_language
        child_session.permission_mode = parent_session.permission_mode
        child_session.agent_mode = ""
        child_session.metadata.state["spawned_by_agent_mode"] = parent_session.agent_mode
        return child_session
