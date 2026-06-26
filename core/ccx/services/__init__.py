"""Run-scoped services shared by ccx mode runners.

These services are constructed once per ``build_runtime()`` call and
shared across all subagent runners that take a reference. They exist so
parallel subagents can amortize expensive shared work (repository
outline scanning) and exchange structured data through a sidecar
channel (findings collection — v5 does not auto-propagate dependency
results into dependent params).
"""

from .findings_collector import FindingsCollector
from .repository_outline import RepositoryOutlineCache
from .steer_inbox import (
    MAX_STEER_BODY_BYTES,
    SteerInbox,
    format_steer_block,
    steer_payload_hash,
)

__all__ = [
    "FindingsCollector",
    "MAX_STEER_BODY_BYTES",
    "RepositoryOutlineCache",
    "SteerInbox",
    "format_steer_block",
    "steer_payload_hash",
]
