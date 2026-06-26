from __future__ import annotations

from pathlib import Path
import re

from ..tools.base import ToolResult
from .session import QuerySession


MAX_IMPLEMENTATION_STALL_ROUNDS = 6
MAX_CODE_ONLY_GRACE_ROUNDS = 8


def implementation_tasks_incomplete(session: QuerySession) -> bool:
    text = implementation_tasks_snapshot(session)
    if text is None:
        return False
    task_lines = re.findall(r"^\s*-\s+\[([ xX])\]", text, re.MULTILINE)
    if not task_lines:
        return False
    return any(marker.strip().lower() != "x" for marker in task_lines)


def implementation_tasks_snapshot(session: QuerySession) -> str | None:
    state = dict(session.metadata.state)
    if state.get("plan_phase") != "implementation":
        return None
    artifacts = dict(state.get("plan_artifacts") or {})
    tasks_path = str(artifacts.get("tasks") or "").strip()
    if not tasks_path:
        return None
    try:
        return Path(tasks_path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def auto_complete_tasks(session: QuerySession) -> None:
    state = dict(session.metadata.state)
    artifacts = dict(state.get("plan_artifacts") or {})
    tasks_path = str(artifacts.get("tasks") or "").strip()
    if not tasks_path:
        return
    path = Path(tasks_path)
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return
    updated = re.sub(r"^(\s*-\s+)\[ \]", r"\1[x]", content, flags=re.MULTILINE)
    if updated != content:
        try:
            path.write_text(updated, encoding="utf-8")
        except OSError:
            return


def count_task_markers(text: str | None) -> tuple[int, int]:
    if not text:
        return (0, 0)
    markers = re.findall(r"^\s*-\s+\[([ xX])\]", text, re.MULTILINE)
    completed = sum(1 for m in markers if m.strip().lower() == "x")
    return (completed, len(markers))


def implementation_round_made_progress(
    *,
    previous_tasks_snapshot: str | None,
    current_tasks_snapshot: str | None,
    tool_results: list[ToolResult],
) -> bool:
    del tool_results
    if current_tasks_snapshot and current_tasks_snapshot != previous_tasks_snapshot:
        prev_completed, prev_total = count_task_markers(previous_tasks_snapshot)
        curr_completed, curr_total = count_task_markers(current_tasks_snapshot)
        if curr_completed < prev_completed:
            return False
        if prev_total > 0 and curr_total > prev_total + 2:
            return False
        return True
    return False


def implementation_requires_task_sync(
    *,
    previous_tasks_snapshot: str | None,
    current_tasks_snapshot: str | None,
    tool_results: list[ToolResult],
) -> bool:
    if current_tasks_snapshot and current_tasks_snapshot != previous_tasks_snapshot:
        return False
    code_mutation_tools = {"file_write", "file_edit", "delete_file"}
    for result in tool_results:
        if result.success and result.tool_name in code_mutation_tools:
            return True
    return False
