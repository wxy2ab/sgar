from __future__ import annotations

from typing import Any

from .strategy_common import coerce_text, contains_target_path, count_hint_hits, summarize_source_text


_ROLE_SPLIT_HINTS = (
    "architecture",
    "cross-module",
    "cross module",
    "end-to-end",
    "multi-file",
    "multi module",
    "pipeline",
    "refactor",
    "workflow",
    "全链路",
    "全流程",
    "多文件",
    "多模块",
    "架构",
    "流程",
    "重构",
    "跨模块",
)
_ADVERSARIAL_HINTS = (
    "correctness",
    "critic",
    "debate",
    "design tradeoff",
    "edge case",
    "risk",
    "review",
    "safety",
    "审查",
    "对抗",
    "方案比较",
    "正确性",
    "风险",
    "边界",
)
_IMPLEMENTATION_HINTS = (
    "build",
    "bug",
    "fix",
    "implement",
    "tests",
    "优化",
    "修复",
    "实现",
    "构建",
    "测试",
)
def decide_agent_collaboration_strategy(
    agent_mode: str,
    user_input: str | list[dict[str, object]],
    *,
    build_mode: bool = False,
) -> dict[str, Any]:
    text = coerce_text(user_input)
    lowered = text.lower()
    has_paths = contains_target_path(text)
    role_split_score = count_hint_hits(lowered, _ROLE_SPLIT_HINTS)
    adversarial_score = count_hint_hits(lowered, _ADVERSARIAL_HINTS)
    implementation_score = count_hint_hits(lowered, _IMPLEMENTATION_HINTS)

    if build_mode:
        role_split_score += 1
        implementation_score += 1
    if has_paths:
        implementation_score += 1

    if role_split_score >= max(adversarial_score, implementation_score) and role_split_score > 0:
        pattern = "role_split"
        roles = ["researcher", "implementer", "reviewer"]
        suggested_agent_count = 3
        rationale = "任务涉及跨模块拆解或多阶段执行，优先采用角色分工。"
        delegation_plan = [
            "researcher: 先梳理现有实现、边界和风险",
            "implementer: 基于研究结论推进主实现方案",
            "reviewer: 审查实现缺口、回归风险和验证完整性",
        ]
    elif adversarial_score > max(role_split_score, implementation_score):
        pattern = "adversarial_iteration"
        roles = ["proposer", "critic"]
        suggested_agent_count = 2
        rationale = "任务更依赖方案正确性、边界检查或风险审查，适合生成-质疑式对抗。"
        delegation_plan = [
            "proposer: 先给出主要方案或实现路径",
            "critic: 审查边界条件、风险点和替代方案",
            "lead: 综合冲突点后再决定最终执行方案",
        ]
    else:
        pattern = "leader_helper"
        roles = ["helper"]
        suggested_agent_count = 1
        rationale = "任务更适合由主代理主导推进，并让辅助代理做补充分析或验证。"
        delegation_plan = [
            "helper: 先做补充分析、实现定位或验证检查",
            "lead: 结合 helper 结果继续推进并负责最终结论",
        ]

    # ``must_delegate`` previously hard-coded ``agent_mode == "agent"``, which
    # made the system prompt mandate child-agent spawning for every turn.
    # That blocked single-agent tasks (e.g. "read these files and write a
    # report") from ever reaching the Write call, because the loop kept
    # reprompting the model to spawn. We now default to ``False`` and let the
    # caller opt in via env var when they actually need enforced delegation.
    import os as _os
    _mandate_env = _os.environ.get("CC_AGENT_DELEGATION_MANDATORY", "")
    _mandate = _mandate_env.strip().lower() in ("1", "true", "yes", "on")
    return {
        "mode": agent_mode,
        "must_delegate": _mandate and agent_mode == "agent",
        "pattern": pattern,
        "rationale": rationale,
        "suggested_agent_count": suggested_agent_count,
        "roles": roles,
        "delegation_plan": delegation_plan,
        "signals": {
            "build_mode": build_mode,
            "has_paths": has_paths,
            "role_split_score": role_split_score,
            "adversarial_score": adversarial_score,
            "implementation_score": implementation_score,
        },
        "source_text": summarize_source_text(text),
    }
