"""ccx — cc on top of deepstack_v5.

Same public API as ``core.cc.api`` (CodeAgent, AgentRunRequest,
AgentRunResult) so callers can flip imports to ccx without touching call
sites. Internally, runs are driven by ``deepstack_v5.EngineV5`` with
plan/spec/agent ToolSpecs, giving:

* parallel sibling subagents (independent siblings dispatched concurrently)
* recursive subagents (agent mode can SpawnResult more agents)
* persistent intent across restart (v5 SQLite checkpoint)
* DAG ordering & dependencies (NodeSpec.depends_on)

The cc subsystems that don't need to know about agent orchestration —
editing, safety, prompts, providers, memory, audit, observability — are
imported directly from ``core.cc`` and not duplicated here.
"""

from .api import (
    AgentRunRequest,
    AgentRunResult,
    CodeAgent,
    CodeBuildRequest,
    ContentStoreOptions,
    MemoryOptions,
)
from .runtime import (
    CCX_MODE_TOOL_MAP,
    CcxRuntimeBundle,
    build_runtime,
    root_node_for,
)
from .services import SteerInbox

__all__ = [
    "AgentRunRequest",
    "AgentRunResult",
    "CCX_MODE_TOOL_MAP",
    "CcxRuntimeBundle",
    "CodeAgent",
    "CodeBuildRequest",
    "ContentStoreOptions",
    "MemoryOptions",
    "SteerInbox",
    "build_runtime",
    "root_node_for",
]
