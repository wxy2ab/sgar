"""Spec mode runner.

Refines a single plan item into 1-N agent tasks (the actual unit of work).
Specs may declare ordering (e.g. "first read, then write") via
`depends_on_previous` on the response items.

LLM I/O contract mirrors plan mode but produces "spec_items":

  {"spec_items": [
      {"goal": "...", "depends_on_previous": true},
      ...
    ],
    "rationale": "..."
  }
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from ..agents.subagent import (
    ModeRunner,
    SubagentInvocation,
    SubagentResult,
)
from .diagnostics import ModeStepRecord, PlanDiagnosticsTracer
from ._goal import current_goal_text
from .llm_client import LLMCallable, text_of
from .parsing import parse_llm_json
from .prompts import PromptLoadError, fallback_system_prompt, load_mode_prompts


logger = logging.getLogger(__name__)


# Fallback constants — authoritative source is
# ``core/ccx/modes/prompts/spec.toml``. See ``plan.py`` for the
# rationale (TOML loader degrades to these constants if the data file
# is missing, and ``test_mode_prompts.py`` asserts byte-equivalence
# between the two so an out-of-sync edit fails fast).
SYSTEM_PROMPT_EN = (
    "You are a specification assistant. Given a plan item, refine it into "
    "1-4 concrete agent-executable tasks.\n\n"
    "Dependency rules — these matter for parallelism:\n"
    "- Default to PARALLEL siblings. Set ``depends_on_previous: true`` "
    "  only on items that need the immediately-preceding item's output.\n"
    "- For DAG ordering, set ``depends_on: [<zero-based indices>]``.\n"
    "- Mixing parallel and chained items in the same spec is encouraged "
    "  whenever the work allows it.\n\n"
    "Return strict JSON only — no preamble, no fences."
)


SYSTEM_PROMPT_ZH = (
    "你是规格细化助手。给定一个计划项，将其细化为 1-4 个可由 agent 直接"
    "执行的任务。\n\n"
    "依赖规则——这影响并行度：\n"
    "- 默认并行。仅当后一项必须等前一项产出时，才设置\n"
    "  depends_on_previous: true。\n"
    "- DAG：在该项上设置 depends_on: [<0 起的下标列表>]。\n"
    "- 鼓励在同一个 spec 里混合使用并行项和链式项。\n\n"
    "只返回严格的 JSON。"
)


def build_spec_prompt(
    plan_item_goal: str,
    *,
    language: str = "en",
    root_goal: str = "",
    chained_to_previous: bool = False,
) -> tuple[str, str]:
    """Build the spec system+user prompt.

    ``root_goal`` (when known) gives the spec LLM the project-level
    context it would otherwise be blind to. ``chained_to_previous``
    tells it that the plan-mode parent declared this item depends on
    its predecessor — useful for the LLM to scope work appropriately
    without assuming artifacts already exist on disk.
    """
    try:
        system = load_mode_prompts("spec").system_for(language)
    except PromptLoadError as exc:
        system = fallback_system_prompt(
            "spec",
            language,
            exc,
            fallback_en=SYSTEM_PROMPT_EN,
            fallback_zh=SYSTEM_PROMPT_ZH,
            logger=logger,
        )
    parts = [f"Plan item: {plan_item_goal}"]
    if root_goal:
        parts.insert(0, f"Project goal (context): {root_goal}")
    if chained_to_previous:
        parts.append(
            "Note: plan-mode declared this item depends on its predecessor; "
            "preserve that ordering in any generated subtasks."
        )
    parts.append(
        'Respond with: {"spec_items": [{"goal": "...", '
        '"depends_on_previous": false, "depends_on": [<optional indices>]}'
        ', ...], "rationale": "..."}'
    )
    user = "\n\n".join(parts)
    return system, user


@dataclass(slots=True)
class SpecModeRunner(ModeRunner):
    llm: LLMCallable
    language: str = "en"
    mode_name: str = "spec"
    cwd: str | None = None
    artifact_root: str | None = None
    tracer: PlanDiagnosticsTracer | None = None

    def run(self, invocation: SubagentInvocation) -> SubagentResult:
        started_at = time.time()
        current_goal = current_goal_text(invocation.goal, invocation.metadata)
        root_goal = str(invocation.metadata.get("ccx_root_goal") or current_goal)
        prompt_root_goal = str(invocation.metadata.get("ccx_root_goal") or "")
        chained = bool(invocation.metadata.get("ccx_depends_on_previous"))
        system, user = build_spec_prompt(
            invocation.goal,
            language=self.language,
            root_goal=prompt_root_goal,
            chained_to_previous=chained,
        )
        response = text_of(self.llm(system=system, user=user, purpose="spec"))
        spec = _parse_spec_response(response)

        artifacts: dict[str, str] | None = None
        if self.cwd and spec.get("items"):
            from .artifacts import write_spec_artifacts
            paths = write_spec_artifacts(
                cwd=self.cwd,
                plan_item_goal=invocation.goal,
                items=spec["items"],
                rationale=spec.get("rationale", ""),
                artifact_root=self.artifact_root,
            )
            artifacts = paths.to_dict()

        if not spec.get("items"):
            # Fallback: single agent task with the spec goal verbatim.
            result = SubagentResult(
                final_text=spec.get("rationale", "")
                or "(no spec items returned; falling back to single agent)",
                subtasks=[
                    SubagentInvocation(
                        goal=invocation.goal,
                        mode="agent",
                        metadata={
                            "ccx_parent_mode": "spec",
                            "ccx_fallback": True,
                            "ccx_root_goal": root_goal,
                            "ccx_parent_plan_goal": current_goal,
                        },
                    ),
                ],
                extras={"goal": current_goal, "raw": response[:1000]},
            )
            self._record(
                started_at=started_at, invocation=invocation,
                system=system, user=user, response=response,
                spec=spec, result=result,
                parse_status="fallback",
                sequential_reason="fallback to single agent (no items parsed)",
            )
            return result

        flags = [bool(item.get("depends_on_previous")) for item in spec["items"]]
        any_explicit_dag = any(item.get("depends_on") for item in spec["items"])
        if any_explicit_dag:
            sequential_reason = "explicit depends_on indices used"
        elif all(flags) and flags:
            sequential_reason = "all items chained via depends_on_previous"
        elif any(flags):
            sequential_reason = (
                f"mixed: {sum(flags)}/{len(flags)} items chained, "
                "rest run in parallel"
            )
        else:
            sequential_reason = "all items independent (parallel siblings)"

        subtasks = [
            SubagentInvocation(
                goal=item["goal"],
                mode="agent",
                metadata={
                    "ccx_parent_mode": "spec",
                    "ccx_spec_index": i,
                    "ccx_parent_plan_goal": current_goal,
                    # Propagate root goal down the chain so agents
                    # aren't blind to the project-level intent.
                    "ccx_root_goal": root_goal,
                    "ccx_depends_on_previous": bool(item.get("depends_on_previous")),
                    "ccx_depends_on": list(item.get("depends_on") or []),
                },
            )
            for i, item in enumerate(spec["items"])
        ]
        extras: dict[str, Any] = {
            "goal": current_goal,
            "items": spec["items"],
        }
        if artifacts:
            extras["artifact_paths"] = artifacts
        # Per-item metadata drives dependencies in to_spawn_result; keep
        # the global flag off so partial-chain plans don't lose parallel
        # siblings.
        result = SubagentResult(
            final_text=spec.get("rationale", ""),
            subtasks=subtasks,
            sequential=False,
            extras=extras,
        )
        self._record(
            started_at=started_at, invocation=invocation,
            system=system, user=user, response=response,
            spec=spec, result=result,
            parse_status="ok",
            sequential_reason=sequential_reason,
        )
        return result

    def _record(
        self, *, started_at: float, invocation: SubagentInvocation,
        system: str, user: str, response: str,
        spec: dict[str, Any], result: SubagentResult,
        parse_status: str, sequential_reason: str,
    ) -> None:
        if self.tracer is None:
            return
        issues: list[str] = []
        item_count = len(spec.get("items") or [])
        if parse_status == "ok" and item_count > 4:
            issues.append(
                f"spec returned {item_count} items (>4) — over-decomposition")
        if parse_status == "ok" and item_count == 1:
            issues.append(
                "spec returned only 1 item — decomposition adds no value; "
                "consider routing the plan item directly to an agent"
            )
        if parse_status == "ok" and not spec.get("rationale"):
            issues.append("spec returned no rationale — hard to audit")
        if not invocation.metadata.get("ccx_root_goal"):
            issues.append(
                "spec invoked without ccx_root_goal context — LLM cannot see "
                "the project-level intent"
            )

        issues.extend(str(item) for item in spec.get("dependency_issues") or [])
        parent_goal = str(
            invocation.metadata.get("ccx_root_goal")
            or invocation.metadata.get("ccx_parent_plan_goal")
            or current_goal_text(invocation.goal, invocation.metadata)
        )

        self.tracer.record_step(ModeStepRecord(
            mode="spec",
            invocation_goal=invocation.goal,
            parent_goal=parent_goal,
            metadata=dict(invocation.metadata),
            started_at=started_at,
            finished_at=time.time(),
            system_prompt=system,
            user_prompt=user,
            raw_response=response,
            parse_status=parse_status,
            parsed_items=list(spec.get("items") or []),
            dropped_items=int(spec.get("dropped") or 0),
            rationale=str(spec.get("rationale") or ""),
            sequential=result.sequential,
            sequential_reason=sequential_reason,
            final_text=result.final_text,
            spawned_subtasks=[
                {
                    "mode": st.mode,
                    "goal": st.goal,
                    "depends_on_previous": st.metadata.get(
                        "ccx_depends_on_previous", False),
                }
                for st in result.subtasks
            ],
            issues=issues,
        ))


def _parse_spec_response(response: str) -> dict[str, Any]:
    data = parse_llm_json(
        response,
        schema_name="spec",
        fallback_factory=lambda _raw: {},
        expected_type=dict,
    )
    items = data.get("spec_items") or data.get("items") or []
    rationale = data.get("rationale") or data.get("summary") or ""
    pending_items: list[tuple[int, dict[str, Any], list[int]]] = []
    dependency_issues: list[str] = []
    dropped = 0
    if not isinstance(items, list):
        # A bare string would otherwise iterate per-character, turning
        # one malformed response into dozens of bogus subtask nodes.
        items = []
        dropped = 1
    for original_index, raw in enumerate(items):
        if isinstance(raw, str) and raw.strip():
            pending_items.append((original_index, {
                "goal": raw,
                "depends_on_previous": False,
                "depends_on": [],
            }, []))
        elif isinstance(raw, dict) and raw.get("goal"):
            depends_on_raw = raw.get("depends_on") or []
            depends_on: list[int] = []
            if isinstance(depends_on_raw, list):
                for v in depends_on_raw:
                    try:
                        depends_on.append(int(v))
                    except (TypeError, ValueError):
                        pass
            pending_items.append((original_index, {
                "goal": str(raw["goal"]),
                "depends_on_previous": bool(raw.get("depends_on_previous", False)),
                "depends_on": [],
            }, depends_on))
        else:
            dropped += 1
    old_to_new = {
        original_index: new_index
        for new_index, (original_index, _item, _deps) in enumerate(pending_items)
    }
    cleaned_items: list[dict[str, Any]] = []
    for original_index, item, depends_on in pending_items:
        new_index = old_to_new[original_index]
        remapped: list[int] = []
        seen: set[int] = set()
        for idx in depends_on:
            if idx not in old_to_new:
                dependency_issues.append(
                    f"spec item {original_index} ignored dangling depends_on "
                    f"index {idx}"
                )
                continue
            mapped = old_to_new[idx]
            if mapped == new_index:
                dependency_issues.append(
                    f"spec item {original_index} ignored self depends_on "
                    f"index {idx}"
                )
                continue
            if mapped in seen:
                dependency_issues.append(
                    f"spec item {original_index} ignored duplicate depends_on "
                    f"index {idx}"
                )
                continue
            seen.add(mapped)
            remapped.append(mapped)
        item["depends_on"] = remapped
        cleaned_items.append(item)
    return {
        "items": cleaned_items,
        "rationale": rationale,
        "dropped": dropped,
        "dependency_issues": dependency_issues,
    }


__all__ = ["SpecModeRunner", "build_spec_prompt"]
