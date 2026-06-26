"""ResumeContext — render a :class:`ResumeSnapshot` for prompt injection.

Kept separate from ``snapshot.py`` so the snapshot module stays free of
formatting concerns; the renderer can evolve (Markdown today, JSON-LD
tomorrow) without touching the priority/budget logic.

A :class:`ResumeContext` wraps a snapshot plus the previous run id and
provides :meth:`format_for_prompt`. ccx splices the formatted block at
the top of each mode runner's user prompt so the LLM sees prior failures
before it sees the new goal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .snapshot import EventRef, ResumeSnapshot


# Stable metadata key used to thread the rendered block from the ccx api
# layer into each ccx mode runner. Centralised here so api/runtime/modes
# all agree on one spelling.
RESUME_PROMPT_METADATA_KEY = "ccx.resume.prompt_block"
RESUME_PREVIOUS_RUN_METADATA_KEY = "ccx.resume.previous_run_id"


@dataclass(slots=True)
class ResumeContext:
    previous_run_id: str
    snapshot: ResumeSnapshot

    @classmethod
    def from_event_store(
        cls,
        event_store: Any,
        previous_run_id: str,
        *,
        token_budget_chars: int = 12_000,
    ) -> "ResumeContext":
        from .snapshot import build_snapshot
        snap = build_snapshot(
            event_store, previous_run_id, token_budget_chars=token_budget_chars
        )
        return cls(previous_run_id=previous_run_id, snapshot=snap)

    def format_for_prompt(self) -> str:
        """Render a Markdown block suitable for prepending to a user prompt.

        Empty snapshot → empty string (no block is rendered when there
        is nothing to say; callers can splice unconditionally).
        """
        if self.snapshot.is_empty:
            return ""
        lines: list[str] = []
        lines.append(f"## Prior session context ({self.previous_run_id})")
        lines.append("")
        lines.append("### Summary")
        lines.append(self.snapshot.summary)
        if self.snapshot.events:
            lines.append("")
            lines.append(f"### Key events ({len(self.snapshot.events)})")
            for ref in self.snapshot.events:
                lines.append(_format_event_line(ref))
        lines.append("")
        return "\n".join(lines)


def _format_event_line(ref: EventRef) -> str:
    """Format one event into a compact bullet.

    Format: ``- [P{priority}] {kind} {distinguishing-detail}``
    """
    bits: list[str] = []
    p = ref.payload_excerpt
    tool = p.get("tool_name")
    if tool:
        bits.append(str(tool))
    for hint in ("error_code", "command", "pattern", "path", "query"):
        v = p.get(hint)
        if v:
            bits.append(f"{hint}={_truncate(str(v))}")
            break
    if "reason" in p and p["reason"]:
        bits.append(f"reason={_truncate(str(p['reason']))}")
    elif "preview" in p and p["preview"]:
        bits.append(f"preview={_truncate(str(p['preview']))}")
    suffix = " — " + ", ".join(bits) if bits else ""
    return f"- [P{ref.priority}] {ref.kind}{suffix}"


def _truncate(s: str, limit: int = 80) -> str:
    if len(s) <= limit:
        return s
    return s[: limit - 1] + "…"


def install_resume_metadata(
    metadata: dict[str, Any], context: ResumeContext
) -> dict[str, Any]:
    """Stamp ``metadata`` with the rendered block and previous-run pointer.

    Returns a *new* dict (does not mutate the input). The output is what
    callers pass to ``root_node_for(metadata=...)``.
    """
    rendered = context.format_for_prompt()
    out = dict(metadata)
    if rendered:
        out[RESUME_PROMPT_METADATA_KEY] = rendered
    out[RESUME_PREVIOUS_RUN_METADATA_KEY] = context.previous_run_id
    return out


def read_resume_block(metadata: dict[str, Any] | None) -> str:
    """Helper for mode runners: pull the rendered block, or empty string."""
    if not metadata:
        return ""
    block = metadata.get(RESUME_PROMPT_METADATA_KEY)
    return str(block) if block else ""


__all__ = [
    "RESUME_PREVIOUS_RUN_METADATA_KEY",
    "RESUME_PROMPT_METADATA_KEY",
    "ResumeContext",
    "install_resume_metadata",
    "read_resume_block",
]
