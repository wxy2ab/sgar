"""Helpers for separating injected run context from the current goal."""

from __future__ import annotations

from typing import Any, Mapping

from ._sgar_command_helpers import CCX_GOAL_OFFSET_METADATA_KEY


def current_goal_text(goal: str, metadata: Mapping[str, Any] | None) -> str:
    """Return the caller's goal after any ccx-injected prefix."""
    raw_offset = (metadata or {}).get(CCX_GOAL_OFFSET_METADATA_KEY)
    try:
        offset = int(raw_offset)
    except (TypeError, ValueError):
        return goal
    if offset <= 0 or offset > len(goal):
        return goal
    return goal[offset:]
