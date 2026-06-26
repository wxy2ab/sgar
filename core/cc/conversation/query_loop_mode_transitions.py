from __future__ import annotations

from ..plan import plan_artifact_ready
from ..specs import SPEC_ARTIFACT_NAMES, SPEC_READY_STATUSES, spec_artifacts_ready
from .session import QuerySession


def mode_exited(old_state: dict[str, object], new_state: dict[str, object]) -> str | None:
    if old_state.get("plan_mode") and not new_state.get("plan_mode"):
        return "plan"
    if old_state.get("spec_mode") and not new_state.get("spec_mode"):
        return "spec"
    return None


def should_auto_exit_spec(session: QuerySession) -> bool:
    state = dict(session.metadata.state)
    if not state.get("spec_mode"):
        return False
    if str(state.get("execute_policy") or "").lower() != "auto_execute":
        return False
    return spec_artifacts_ready(state)


def should_auto_exit_plan(session: QuerySession) -> bool:
    state = dict(session.metadata.state)
    if not state.get("plan_mode"):
        return False
    if str(state.get("execute_policy") or "").lower() != "auto_execute":
        return False
    return plan_artifact_ready(state)


def plan_artifacts_incomplete(session: QuerySession) -> bool:
    state = dict(session.metadata.state)
    if not state.get("plan_mode"):
        return False
    return not plan_artifact_ready(state)


def plan_incomplete_instruction(session: QuerySession) -> str:
    state = dict(session.metadata.state)
    artifacts = dict(state.get("plan_artifact_status") or {})
    missing: list[str] = []
    if str(artifacts.get("plan", "")).lower() not in {"ready", "completed"}:
        missing.append("plan.md")
    if str(artifacts.get("tasks", "")).lower() not in {"ready", "completed"}:
        missing.append("tasks.md")
    names = "、".join(missing) if missing else "artifacts"
    return (
        f"计划尚未完成，{names} 还未写入。\n"
        f"请使用 plan_artifact_write 工具完成所有计划产物的编写。\n"
        f"tasks.md 必须包含结构化的任务列表，格式为 `- [ ] 任务描述`。\n"
        f"不要输出总结文字，直接调用工具。"
    )


def spec_artifacts_incomplete(session: QuerySession) -> bool:
    state = dict(session.metadata.state)
    if not state.get("spec_mode"):
        return False
    return not spec_artifacts_ready(state)


def spec_incomplete_instruction(session: QuerySession) -> str:
    state = dict(session.metadata.state)
    statuses = dict(state.get("spec_artifact_status") or {})
    missing: list[str] = []
    for name in SPEC_ARTIFACT_NAMES:
        if str(statuses.get(name, "pending")).lower() not in SPEC_READY_STATUSES:
            missing.append(f"{name}.md")
    names = "、".join(missing) if missing else "artifacts"
    return (
        f"规范尚未完成，{names} 还未写入。\n"
        f"请使用 spec_artifact_write 工具完成所有规范产物的编写。\n"
        f"不要输出总结文字，直接调用工具。"
    )


_TODO_TERMINAL_STATUSES = {"completed", "cancelled"}


def todos_incomplete(session: QuerySession) -> bool:
    """Return True when the session has todos with non-terminal status."""
    todos = list(session.metadata.state.get("todos") or [])
    if not todos:
        return False
    return any(
        str(t.get("status", "pending")).lower() not in _TODO_TERMINAL_STATUSES
        for t in todos if isinstance(t, dict)
    )


def todos_incomplete_instruction(session: QuerySession) -> str:
    """Build a reprompt instruction listing remaining incomplete todos."""
    todos = list(session.metadata.state.get("todos") or [])
    pending = [
        str(t.get("content", ""))
        for t in todos
        if isinstance(t, dict) and str(t.get("status", "pending")).lower() not in _TODO_TERMINAL_STATUSES
    ]
    items = "\n".join(f"- {item}" for item in pending[:8])
    return (
        f"你的任务清单中还有 {len(pending)} 项未完成：\n{items}\n"
        f"请继续执行下一个待完成的任务。完成后用 todo_write 更新状态为 completed。\n"
        f"不要输出总结文字，直接调用工具。"
    )


def todos_stall_rescue_instruction(session: QuerySession) -> str:
    """Force the agent to transition from read-only exploration to action."""
    todos = list(session.metadata.state.get("todos") or [])
    pending = [
        str(t.get("content", ""))
        for t in todos
        if isinstance(t, dict) and str(t.get("status", "pending")).lower() not in _TODO_TERMINAL_STATUSES
    ]
    items = "\n".join(f"- {item}" for item in pending[:5])
    return (
        f"警告：你已经进行了大量的只读操作（读取文件、搜索等），但没有产生任何实际输出。\n"
        f"你的任务清单中还有 {len(pending)} 项未完成：\n{items}\n\n"
        f"你必须立即采取行动：\n"
        f"1. 如果分析已经足够，立即使用 file_write 或 file_edit 创建/修改文件\n"
        f"2. 如果某些任务已经完成，使用 todo_write 更新状态为 completed\n"
        f"3. 如果某些任务无法完成，使用 todo_write 更新状态为 cancelled\n\n"
        f"不要再继续只读探索，直接开始产出。"
    )
