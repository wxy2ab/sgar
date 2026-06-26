"""Memory subpackage — cross-run / cross-compaction state.

Distinct from ``knowledge/`` (which holds run-scoped claims with
confidence/evidence semantics): the modules here are derived views and
indices, useful when a later run needs to know what an earlier run did
or learned. Lifecycle: append-only or read-only over the existing v5
event store.
Current modules:
* ``priority`` classifies events into P1..P4 snapshot priority.
* ``snapshot`` builds bounded ResumeSnapshot views from event history.
* ``content_store`` is an optional explicit FTS5 index for large outputs.
"""

from .content_store import (
    ChunkHit,
    ContentStore,
    ContentStoreStats,
    chunk_markdown,
    compute_workspace_hash,
    default_db_path,
)
from .priority import priority_for
from .resume import (
    RESUME_PREVIOUS_RUN_METADATA_KEY,
    RESUME_PROMPT_METADATA_KEY,
    ResumeContext,
    install_resume_metadata,
    read_resume_block,
)
from .snapshot import EventRef, ResumeSnapshot, build_snapshot

__all__ = [
    "ChunkHit",
    "ContentStore",
    "ContentStoreStats",
    "EventRef",
    "RESUME_PREVIOUS_RUN_METADATA_KEY",
    "RESUME_PROMPT_METADATA_KEY",
    "ResumeContext",
    "ResumeSnapshot",
    "build_snapshot",
    "chunk_markdown",
    "compute_workspace_hash",
    "default_db_path",
    "install_resume_metadata",
    "priority_for",
    "read_resume_block",
]
