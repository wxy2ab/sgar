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
from .fixed_dag import (
    RunSpecNode,
    RunSpecResult,
    run_spec,
)
from .fixed_dag_export import (
    DraftDag,
    DraftDagError,
    export_draft_dag,
    load_draft_dag,
    write_draft_dag,
)
from .fixed_dag_guards import (
    NamedSpecRegistry,
    OnceGuard,
    SchemaContract,
    SchemaContractError,
    check_schema,
    mark_requires_approval,
    once_per_period,
    schema_preflight_capability,
)
from .services import SteerInbox

# Skill subsystem — defined in core.cc (cc owns the tool registry the ccx modes
# drive), re-exported here for ergonomics so sgar/ccx callers can
# ``from core.ccx import load_skill_registry, SkillRegistry`` and register skills
# dynamically. Skills already work in every ccx mode via the cc tool wiring.
from core.cc.skills import (
    SkillDefinition,
    SkillRegistry,
    load_skill_registry,
    skill_roots,
)
from core.cc.tools.skill import SkillTool

__all__ = [
    "AgentRunRequest",
    "AgentRunResult",
    "CCX_MODE_TOOL_MAP",
    "CcxRuntimeBundle",
    "CodeAgent",
    "CodeBuildRequest",
    "ContentStoreOptions",
    "DraftDag",
    "DraftDagError",
    "MemoryOptions",
    "NamedSpecRegistry",
    "OnceGuard",
    "RunSpecNode",
    "RunSpecResult",
    "SchemaContract",
    "SchemaContractError",
    "SkillDefinition",
    "SkillRegistry",
    "SkillTool",
    "SteerInbox",
    "build_runtime",
    "check_schema",
    "export_draft_dag",
    "load_draft_dag",
    "load_skill_registry",
    "mark_requires_approval",
    "once_per_period",
    "root_node_for",
    "run_spec",
    "schema_preflight_capability",
    "skill_roots",
    "write_draft_dag",
]
