from __future__ import annotations

from ..agents.agent_tool import AgentTool
from ..agents.runtime_registry import (
    InProcessRuntimeRegistry,
    get_in_process_runtime_registry,
)
from ..config import CCConfig
from ..llm import LLMClientProvider
from ..memory import MemoryRuntime
from .enter_plan_mode import EnterPlanModeTool
from .enter_spec_mode import EnterSpecModeTool
from .exit_plan_mode import ExitPlanModeTool
from .exit_spec_mode import ExitSpecModeTool
from .file_read import FileReadTool
from .file_edit import FileEditTool
from .file_write import FileWriteTool
from .plan_artifact_write import PlanArtifactWriteTool
from .glob_tool import GlobTool
from .grep_tool import GrepTool
from .memory import MemoryTool
from .memory_fact import MemoryFactTool
from .memory_search import MemorySearchTool
from .memory_status import MemoryStatusTool
from .memory_store import MemoryStoreTool
from .powershell import PowerShellTool
from .registry import ToolRegistry
from .run_tests import RunTestsTool
from .send_message import SendMessageTool
from .shell import ShellTool
from .spec_artifact_write import SpecArtifactWriteTool
from .task_stop import TaskStopTool
from .todo_write import TodoWriteTool
from .worktree import EnterWorktreeTool, ExitWorktreeTool


def build_builtin_tool_registry(
    config: CCConfig | None = None,
    *,
    llm_client_provider: LLMClientProvider | None = None,
    runtime_registry: InProcessRuntimeRegistry | None = None,
    memory_runtime: MemoryRuntime | None = None,
) -> ToolRegistry:
    resolved_config = config or CCConfig()
    registry_handle = runtime_registry or get_in_process_runtime_registry(resolved_config.runtime_root_path())
    registry = ToolRegistry()
    registry.register(FileReadTool())
    registry.register(FileEditTool(config=resolved_config))
    registry.register(FileWriteTool(config=resolved_config))
    registry.register(TodoWriteTool())
    registry.register(GlobTool())
    registry.register(GrepTool())
    if memory_runtime is not None:
        registry.register(MemoryTool(memory_runtime=memory_runtime))
        # Legacy aliases — hidden from the LLM schema (is_enabled=False) but
        # still resolvable by name for in-process callers and tests during
        # the deprecation window.
        registry.register(MemoryStatusTool(memory_runtime=memory_runtime))
        registry.register(MemorySearchTool(memory_runtime=memory_runtime))
        registry.register(MemoryStoreTool(memory_runtime=memory_runtime))
        registry.register(MemoryFactTool(memory_runtime=memory_runtime))
    registry.register(EnterPlanModeTool())
    registry.register(ExitPlanModeTool())
    registry.register(EnterSpecModeTool())
    registry.register(ExitSpecModeTool())
    registry.register(SpecArtifactWriteTool())
    registry.register(PlanArtifactWriteTool())
    registry.register(EnterWorktreeTool())
    registry.register(ExitWorktreeTool())
    registry.register(ShellTool())
    registry.register(PowerShellTool())
    # Default-OFF (hidden from the model schema unless ``run_tests_tool_enabled``
    # is set on the config) — registering it unconditionally keeps the tool
    # name resolvable for in-process callers/tests while ``is_enabled`` gates
    # its visibility, so the exported schema is byte-identical when off.
    registry.register(RunTestsTool())
    registry.register(AgentTool(llm_client_provider=llm_client_provider, runtime_registry=registry_handle))
    registry.register(SendMessageTool(runtime_registry=registry_handle))
    registry.register(TaskStopTool(runtime_registry=registry_handle))
    return registry
