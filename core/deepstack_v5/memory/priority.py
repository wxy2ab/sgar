"""Event priority classifier — pure function of (kind, payload).

Used by the ccx → cc event bridge to tag every republished event so
later passes (ResumeSnapshot, ctx_stats) can pick the load-bearing rows
out of a long run without re-deriving "what matters". Event priority is
persisted on the event payload when available and recomputed as a
fallback for older rows, so both write-time tagging and read-time
snapshotting share this one classifier.

Scale: 1 = low (chatter / status), 2 = normal (observation tool calls),
3 = high (mutating tool calls), 4 = critical (failures / errors).

Inputs:
* ``kind`` is the v5 event kind, e.g. ``"node.failed"``, ``"cc.tool_use"``.
* ``payload`` is the v5 event payload as the bridge or dispatcher built
  it. The function reads only well-known fields (``tool_name``,
  ``success``) and never mutates the input.
"""

from __future__ import annotations

from typing import Any


# Tool names that *change* the workspace. A failed mutation gets
# priority 4 via the success=False branch; a successful mutation is 3
# because "we definitely touched X" is the kind of thing a resume needs
# to remember.
_MUTATING_TOOLS: frozenset[str] = frozenset({
    "Edit",
    "Write",
    "MultiEdit",
    "NotebookEdit",
    "file_write",
    "file_edit",
    "apply_patch",
})

# Tool names that *observe* without mutating. A successful observation
# is normal-priority context; a failed one bubbles to 4 via the
# success=False branch.
_OBSERVATION_TOOLS: frozenset[str] = frozenset({
    "Read",
    "Grep",
    "Glob",
    "Bash",
    "WebFetch",
    "WebSearch",
    "file_read",
    "shell",
    "grep",
    "glob",
})


# Kinds that always indicate failure regardless of payload.
_ALWAYS_CRITICAL: frozenset[str] = frozenset({
    "node.failed",
    "cc.tool_failed",
    "cc.turn_failed",
    "run.failed",
})


# Kinds that are low-signal chatter unless payload says otherwise.
_LOW_BY_DEFAULT: frozenset[str] = frozenset({
    "cc.assistant_text",
    "cc.turn_completed",
    "node.running",
    "node.completed",
})

_LOW_PREFIXES: tuple[str, ...] = (
    "ccx.cost.",
    "ccx.steer.",
    "ccx.resume.",
)


def priority_for(kind: str, payload: dict[str, Any] | None = None) -> int:
    """Return the 1..4 priority for a v5 event.

    The classification cascade:

    1. **Critical kinds** (``node.failed`` family) → 4.
    2. **Explicit failure in payload** (``success`` is exactly ``False``)
       → 4. Covers ``cc.tool_completed`` rows that report failure via
       the progress object.
    3. **``cc.tool_use``** — branch on ``tool_name``:
       * mutating → 3
       * observation → 2
       * unknown → 2 (conservative: treat unrecognised tools as
         observation rather than chatter, so a new tool that we haven't
         classified still surfaces in resume snapshots).
    4. **Low-by-default kinds** (``cc.assistant_text``, ``cc.turn_completed``)
       → 1.
    5. **Anything else** → 2 (normal).
    """
    if kind in _ALWAYS_CRITICAL:
        return 4

    p = payload or {}
    if p.get("success") is False:
        return 4

    if kind == "cc.tool_use":
        tool_name = p.get("tool_name") or ""
        if tool_name in _MUTATING_TOOLS:
            return 3
        if tool_name in _OBSERVATION_TOOLS:
            return 2
        return 2

    if kind in _LOW_BY_DEFAULT or any(
        kind.startswith(prefix) for prefix in _LOW_PREFIXES
    ):
        return 1

    return 2


__all__ = ["priority_for"]
