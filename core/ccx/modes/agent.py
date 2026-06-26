"""Agent mode runner — terminal task execution.

An agent invocation does the actual work. Two response shapes are accepted:

1. Terminal:
   {"final_text": "...", "actions_taken": [...]}
   Returns SubagentResult with `final_text` set, no subtasks.

2. Recursive (when the LLM judges the task too large):
   {"subtasks": [{"goal": "...", "mode": "agent"}, ...]}
   Returns SubagentResult with subtasks; ccx will spawn child agent nodes.
   This is how recursive subagents emerge — an agent can decompose itself
   without going back through plan/spec.

Tool calls (e.g. file reads, shell) are out of scope for this milestone — a
real implementation would wire `core.cc.tools.ToolOrchestrator` through here.
For testing we lean on the LLM stub returning either terminal or recursive
responses, which is enough to demonstrate orchestration.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from .parsing import parse_llm_json
from ._goal import current_goal_text
from .prompts import PromptLoadError, fallback_system_prompt, load_mode_prompts
from ..agents.subagent import (
    ModeRunner,
    SubagentInvocation,
    SubagentResult,
)
from .diagnostics import ModeStepRecord, PlanDiagnosticsTracer
from .llm_client import LLMCallable, text_of


logger = logging.getLogger(__name__)


# Fallback constants — authoritative source is
# ``core/ccx/modes/prompts/agent.toml``. See ``plan.py`` for the
# rationale.
SYSTEM_PROMPT_EN = (
    "You are an execution agent. Given a single concrete task, either "
    "execute it and return final_text, or — if you judge it too large — "
    "decompose it into 1-3 sibling agent subtasks. Return strict JSON."
)


SYSTEM_PROMPT_ZH = (
    "你是执行 agent。给定一个具体任务：要么执行并在 final_text 中返回结果；"
    "要么——若你判断任务过大——将其分解为 1-3 个并列的 agent 子任务。"
    "返回严格的 JSON。"
)


def build_agent_prompt(
    task_goal: str,
    *,
    language: str = "en",
    root_goal: str = "",
    parent_plan_goal: str = "",
) -> tuple[str, str]:
    """Build the agent system+user prompt.

    ``root_goal`` and ``parent_plan_goal`` add lineage context so the
    LLM sees how this task fits into the overall plan. Both are
    optional — they are skipped when not provided so the agent prompt
    stays minimal for ad-hoc invocations.
    """
    try:
        system = load_mode_prompts("agent").system_for(language)
    except PromptLoadError as exc:
        system = fallback_system_prompt(
            "agent",
            language,
            exc,
            fallback_en=SYSTEM_PROMPT_EN,
            fallback_zh=SYSTEM_PROMPT_ZH,
            logger=logger,
        )
    parts: list[str] = []
    if root_goal:
        parts.append(f"Project goal (context): {root_goal}")
    if parent_plan_goal and parent_plan_goal != task_goal:
        parts.append(f"Parent plan item: {parent_plan_goal}")
    parts.append(f"Task: {task_goal}")
    parts.append(
        "Respond with EITHER:\n"
        '  {"final_text": "...", "actions_taken": [...]}\n'
        "OR\n"
        '  {"subtasks": [{"goal": "...", "mode": "agent"}, ...]}'
    )
    user = "\n\n".join(parts)
    return system, user


@dataclass(slots=True)
class AgentModeRunner(ModeRunner):
    llm: LLMCallable
    language: str = "en"
    mode_name: str = "agent"
    tracer: PlanDiagnosticsTracer | None = None

    def run(self, invocation: SubagentInvocation) -> SubagentResult:
        started_at = time.time()
        current_goal = current_goal_text(invocation.goal, invocation.metadata)
        root_goal = str(invocation.metadata.get("ccx_root_goal") or current_goal)
        parent_plan_goal = str(
            invocation.metadata.get("ccx_parent_plan_goal")
            or invocation.metadata.get("ccx_parent_goal")
            or ""
        )
        prompt_root_goal = str(invocation.metadata.get("ccx_root_goal") or "")
        system, user = build_agent_prompt(
            invocation.goal,
            language=self.language,
            root_goal=prompt_root_goal,
            parent_plan_goal=parent_plan_goal,
        )
        response = text_of(self.llm(system=system, user=user, purpose="agent"))
        parsed = _parse_agent_response(response)

        if parsed.get("subtasks"):
            result = SubagentResult(
                final_text=parsed.get("final_text", ""),
                subtasks=[
                    SubagentInvocation(
                        goal=item["goal"],
                        mode=item.get("mode", "agent"),
                        metadata={
                            "ccx_parent_mode": "agent",
                            "ccx_recursive": True,
                            "ccx_parent_goal": current_goal,
                            "ccx_root_goal": root_goal,
                            "ccx_parent_plan_goal": parent_plan_goal,
                        },
                    )
                    for item in parsed["subtasks"]
                ],
                sequential=parsed.get("sequential", False),
                extras={
                    "goal": current_goal,
                    "actions_taken": parsed.get("actions_taken", []),
                },
            )
            self._record(
                started_at=started_at, invocation=invocation,
                system=system, user=user, response=response,
                result=result, parse_status="recursive",
            )
            return result

        result = SubagentResult(
            final_text=parsed.get("final_text", "") or response.strip(),
            subtasks=[],
            extras={
                "goal": current_goal,
                "actions_taken": parsed.get("actions_taken", []),
            },
        )
        self._record(
            started_at=started_at, invocation=invocation,
            system=system, user=user, response=response,
            result=result, parse_status="terminal",
        )
        return result

    def _record(
        self, *, started_at: float, invocation: SubagentInvocation,
        system: str, user: str, response: str,
        result: SubagentResult, parse_status: str,
    ) -> None:
        if self.tracer is None:
            return
        issues: list[str] = []
        if parse_status == "recursive" and len(result.subtasks) > 3:
            issues.append(
                f"agent returned {len(result.subtasks)} recursive subtasks "
                "(>3) — likely should have been a spec instead"
            )
        if parse_status == "terminal" and not result.final_text.strip():
            issues.append("agent terminal with empty final_text")

        self.tracer.record_step(ModeStepRecord(
            mode="agent",
            invocation_goal=invocation.goal,
            parent_goal=str(invocation.metadata.get(
                "ccx_parent_plan_goal",
                invocation.metadata.get("ccx_parent_goal", ""),
            )),
            metadata=dict(invocation.metadata),
            started_at=started_at,
            finished_at=time.time(),
            system_prompt=system,
            user_prompt=user,
            raw_response=response,
            parse_status=parse_status,
            parsed_items=[],
            dropped_items=0,
            rationale="",
            sequential=result.sequential,
            sequential_reason="recursive" if parse_status == "recursive" else "",
            final_text=result.final_text,
            spawned_subtasks=[
                {"mode": st.mode, "goal": st.goal,
                 "depends_on_previous": False}
                for st in result.subtasks
            ],
            issues=issues,
        ))


def _parse_agent_response(response: str) -> dict[str, Any]:
    # Smart fallback: unparseable response becomes terminal text so
    # the LLM's prose is surfaced to the caller instead of being
    # dropped. ``parse_llm_json`` returns the fallback shape directly,
    # which the coerce below then treats as "no subtasks, no actions".
    data = parse_llm_json(
        response,
        schema_name="agent",
        fallback_factory=lambda raw: {"final_text": raw.strip()},
        expected_type=dict,
    )

    subtasks: list[dict[str, Any]] = []
    for raw in data.get("subtasks") or []:
        if isinstance(raw, dict) and raw.get("goal"):
            subtasks.append({
                "goal": str(raw["goal"]),
                "mode": raw.get("mode", "agent"),
            })

    return {
        "final_text": str(data.get("final_text", "") or ""),
        "subtasks": subtasks,
        "actions_taken": data.get("actions_taken") or [],
        "sequential": bool(data.get("sequential", False)),
    }


__all__ = ["AgentModeRunner", "build_agent_prompt"]
