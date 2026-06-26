# ---------------------------------------------------------------------------
# Public API (stable, recommended for external integrations)
# ---------------------------------------------------------------------------
from .audit import (
    RuntimeAuditQuery,
    RuntimeAuditSnapshot,
    RuntimeAuditSummary,
    query_runtime_audit,
    read_runtime_audit,
    summarize_runtime_audit,
)
from .api import (
    AgentRunRequest,
    AgentRunResult,
    CodeAgent,
    CodeBuildRequest,
    build_code_with_agent,
    run_code_agent,
)
from .config import CCConfig, load_cc_config

# ---------------------------------------------------------------------------
# Advanced API (stable but lower-level; for fine-grained session control)
# ---------------------------------------------------------------------------
from .llm import DefaultLLMClientProvider, LLMClientProvider
from .runtime import build_default_query_engine
from .conversation.query_engine import QueryEngine
from .conversation.session import QuerySession, SessionFactory
from .conversation.models import SessionEvent, SessionMessage

# ---------------------------------------------------------------------------
# Middleware pipeline (composable cross-cutting concerns)
# ---------------------------------------------------------------------------
from .conversation.middleware import (
    RetryPolicy,
    TurnHooks,
    TurnMiddleware,
    TurnRunner,
    apply as apply_middleware,
    pipe as pipe_middleware,
    with_compaction,
    with_hooks,
    with_persistence,
    with_retry,
    with_turn_tracking,
)

# ---------------------------------------------------------------------------
# Environment providers (injectable filesystem / shell abstractions)
# ---------------------------------------------------------------------------
from .providers import (
    CommandProvider,
    Environment,
    FileSystemProvider,
    LocalCommandProvider,
    LocalFileSystemProvider,
    default_environment,
)

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
from .errors import (
    CCError,
    ConfigError,
    ToolExecutionError,
    ToolValidationError,
)

__all__ = [
    # -- Public API --
    "AgentRunRequest",
    "AgentRunResult",
    "CCConfig",
    "CodeAgent",
    "CodeBuildRequest",
    "RuntimeAuditQuery",
    "RuntimeAuditSnapshot",
    "RuntimeAuditSummary",
    "build_code_with_agent",
    "load_cc_config",
    "query_runtime_audit",
    "read_runtime_audit",
    "run_code_agent",
    "summarize_runtime_audit",
    # -- Advanced API --
    "DefaultLLMClientProvider",
    "LLMClientProvider",
    "QueryEngine",
    "QuerySession",
    "SessionEvent",
    "SessionFactory",
    "SessionMessage",
    "build_default_query_engine",
    # -- Middleware pipeline --
    "RetryPolicy",
    "TurnHooks",
    "TurnMiddleware",
    "TurnRunner",
    "apply_middleware",
    "pipe_middleware",
    "with_compaction",
    "with_hooks",
    "with_persistence",
    "with_retry",
    "with_turn_tracking",
    # -- Environment providers --
    "CommandProvider",
    "Environment",
    "FileSystemProvider",
    "LocalCommandProvider",
    "LocalFileSystemProvider",
    "default_environment",
    # -- Errors --
    "CCError",
    "ConfigError",
    "ToolExecutionError",
    "ToolValidationError",
]
