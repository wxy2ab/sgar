"""Plan mode runner.

Decomposes a high-level goal into spec invocations. Each plan item becomes
one spec subtask; spec subtasks default to *parallel* (no implicit
ordering) — callers pass `sequential=True` via the LLM response when they
want a chain.

LLM I/O contract:

  prompt: build_plan_prompt(goal, language="en")
  expected response: JSON like
    {"plan_items": [
        {"goal": "<step description>", "depends_on_previous": false},
        ...
     ],
     "rationale": "..."
    }

If the response can't be parsed, the runner falls back to a single
invocation with the original goal in agent mode (terminal recovery).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from .parsing import parse_llm_json
from ._dag_helpers import _order_for_backward_deps
from ._goal import current_goal_text
from ..agents.metadata_inheritance import (
    INVARIANTS_METADATA_KEY,
    format_invariants_block,
)
from ..agents.subagent import (
    ModeRunner,
    SubagentInvocation,
    SubagentResult,
)
from .diagnostics import ModeStepRecord, PlanDiagnosticsTracer
from .llm_client import LLMCallable, text_of
from .prompts import PromptLoadError, fallback_system_prompt, load_mode_prompts


logger = logging.getLogger(__name__)


def _stamp_invariants_on_subtasks(
    subtasks: list[SubagentInvocation],
    invariants: list[str],
) -> list[SubagentInvocation]:
    """Inject plan invariants into every subtask goal + metadata."""
    clean = [str(x).strip() for x in invariants if str(x).strip()]
    if not clean:
        return subtasks
    block = format_invariants_block(clean)
    out: list[SubagentInvocation] = []
    for st in subtasks:
        meta = dict(st.metadata or {})
        meta[INVARIANTS_METADATA_KEY] = list(clean)
        out.append(SubagentInvocation(
            goal=(st.goal or "") + block,
            mode=st.mode,
            metadata=meta,
            requires_approval=st.requires_approval,
            max_attempts=st.max_attempts,
            timeout_s=st.timeout_s,
            preferred_model=st.preferred_model,
        ))
    return out


# Fallback constants — authoritative source is
# ``core/ccx/modes/prompts/plan.toml``. These are kept for two reasons:
#
# 1. If the TOML file is missing on disk (e.g. a packaging hiccup,
#    a stale wheel install, an unusual deployment that excludes data
#    files), ``build_plan_prompt`` degrades gracefully to these
#    constants instead of crashing the run.
# 2. ``test_mode_prompts.py::test_plan_toml_matches_fallback_constants``
#    asserts byte-equivalence between the TOML payload and these
#    constants, so a prompt edit in either place that drifts from the
#    other is caught immediately.
#
# When editing prompts: change the TOML, then run the test, then copy
# the TOML content into the constants below to keep the fallback in
# sync. (Future cleanup: drop the fallbacks once packaging guarantees
# the TOMLs ship reliably.)
SYSTEM_PROMPT_EN = (
    "You are a planning assistant. Given a high-level goal, decompose it "
    "into 2-5 plan items. Each item should be concrete enough that a "
    "specification step could elaborate it.\n\n"
    "Dependency rules — these matter for parallelism:\n"
    "- Default to PARALLEL siblings. Only declare a dependency when item B "
    "  truly cannot start until item A's output exists.\n"
    "- For chain-style ordering, set per-item ``depends_on_previous: true`` "
    "  on the items that need it. Mixing parallel and chained items in the "
    "  same plan is allowed and encouraged.\n"
    "- For DAG ordering (item N depends on items 1 AND 3 but not 2), set "
    "  ``depends_on: [<zero-based indices>]`` on item N.\n"
    "- If the work has cyclic / closed-loop dependencies, you MUST also "
    "  emit ``cutset_anchors``: a list of interface/schema/directory/"
    "  decision anchors that cut the cycle into a deterministic order. "
    "  A cycle without cutset_anchors is rejected.\n"
    "- Declare each item's file footprint when you know it: "
    "  ``write_scope`` / ``read_scope`` as lists of path prefixes. Items "
    "  whose write scopes overlap are run in sequence instead of racing on "
    "  the same files; omit the field when the footprint is unknown.\n"
    "- Always include a short ``rationale`` explaining why these items, in "
    "  this dependency shape.\n\n"
    "Return strict JSON only — no preamble, no fences."
)


SYSTEM_PROMPT_ZH = (
    "你是规划助手。给定一个高层目标，将其分解为 2-5 个计划项。"
    "每一项要具体到可由后续 spec 步骤继续展开。\n\n"
    "依赖规则——这影响并行度：\n"
    "- 默认并行执行。仅当 B 必须等 A 的产出才可以开始时，才声明依赖。\n"
    "- 链式顺序：在需要按序执行的项上设置 depends_on_previous: true。\n"
    "  允许（鼓励）在同一个计划里混合使用并行项和链式项。\n"
    "- DAG 顺序（第 N 项同时依赖 1 和 3 但不依赖 2）：在该项上设置\n"
    "  depends_on: [<0 起的下标列表>]。\n"
    "- 若存在闭环依赖，必须同时给出 cutset_anchors（接口/schema/目录/"
    "  关键决策等切集锚点）；无锚点的环会被拒绝。\n"
    "- 已知时请声明每一项的文件范围：write_scope / read_scope，取值为"
    "路径前缀列表。写范围重叠的项会被串行执行而不是争用同一批文件；"
    "范围未知时省略该字段。\n"
    "- 始终给出 rationale，解释为什么是这些项以及这种依赖形状。\n\n"
    "只返回严格的 JSON——不要有前导说明，也不要代码块围栏。"
)


_USER_TEMPLATE_FALLBACK = (
    "Goal: {goal}\n\n"
    'Respond with: {{"plan_items": [{{"goal": "...", '
    '"depends_on_previous": false, "depends_on": [<optional indices>], '
    '"write_scope": [<optional paths>], "read_scope": [<optional paths>]}}'
    ', ...], "cutset_anchors": [<optional anchors>], '
    '"invariants": [<optional>], "rationale": "..."}}'
)


def build_plan_prompt(goal: str, *, language: str = "en") -> tuple[str, str]:
    try:
        prompts = load_mode_prompts("plan")
        system = prompts.system_for(language)
        user = prompts.user_template.format(goal=goal)
    except PromptLoadError as exc:
        # TOML not on disk or malformed — fall back to the compiled-in
        # constants so the run still produces a result. The shared
        # warn-and-select-constant shell lives in ``fallback_system_prompt``;
        # the ``{goal}`` user template is plan-specific so it stays here.
        system = fallback_system_prompt(
            "plan",
            language,
            exc,
            fallback_en=SYSTEM_PROMPT_EN,
            fallback_zh=SYSTEM_PROMPT_ZH,
            logger=logger,
        )
        user = _USER_TEMPLATE_FALLBACK.format(goal=goal)
    return system, user


@dataclass(slots=True)
class PlanModeRunner(ModeRunner):
    llm: LLMCallable
    language: str = "en"
    mode_name: str = "plan"
    # Optional artifact emission. When `cwd` is set, plan.md / tasks.md
    # are written to ``<cwd>/.cc/plans/<id>/``. When `artifact_root` is
    # also set, that overrides the default location.
    cwd: str | None = None
    artifact_root: str | None = None
    # Optional diagnostic tracer.
    tracer: PlanDiagnosticsTracer | None = None

    def run(self, invocation: SubagentInvocation) -> SubagentResult:
        started_at = time.time()
        root_goal = current_goal_text(invocation.goal, invocation.metadata)
        # Goal mode (agent_mode="goal") deterministic short-circuit: when the
        # goal loop stamps an explicit DAG into metadata, materialize it verbatim
        # as agent subtasks instead of calling the LLM to re-derive a plan. The
        # key is the canonical ``CCX_GOAL_DAG_METADATA_KEY`` defined in
        # ``agents/governed_goal.py`` (kept as a literal here to avoid importing
        # that module at plan-runner load). Absent (every non-goal run) ⇒ this
        # branch is inert and behaviour is byte-equivalent.
        explicit_dag = invocation.metadata.get("ccx_goal_dag")
        if explicit_dag is not None:
            return self._run_explicit_dag(
                invocation, explicit_dag,
                root_goal=root_goal, started_at=started_at,
            )
        system, user = build_plan_prompt(invocation.goal, language=self.language)
        response = text_of(self.llm(system=system, user=user, purpose="plan"))
        plan = _parse_plan_response(response)

        artifacts: dict[str, str] | None = None
        if self.cwd and plan.get("items"):
            from .artifacts import write_plan_artifacts
            paths = write_plan_artifacts(
                cwd=self.cwd,
                goal=invocation.goal,
                items=plan["items"],
                rationale=plan.get("rationale", ""),
                artifact_root=self.artifact_root,
            )
            artifacts = paths.to_dict()

        if not plan.get("items"):
            # Fallback: hand the whole goal to a single agent.
            result = SubagentResult(
                final_text=plan.get("rationale", "")
                or "(no plan items returned; running as single agent)",
                subtasks=[
                    SubagentInvocation(
                        goal=invocation.goal,
                        mode="agent",
                        metadata={
                            "ccx_parent_mode": "plan",
                            "ccx_fallback": True,
                            "ccx_root_goal": root_goal,
                        },
                    ),
                ],
                extras={"goal": root_goal, "raw": response[:1000]},
            )
            self._record(
                started_at=started_at,
                invocation=invocation,
                system=system, user=user, response=response,
                plan=plan, result=result,
                parse_status="fallback",
                sequential_reason="fallback to single agent (no items parsed)",
            )
            return result

        # Principle 3: a dependency cycle without cutset anchors is not a
        # silently-broken DAG — degrade to a single sequential agent.
        cycle_issues = [
            i for i in (plan.get("dependency_issues") or [])
            if "cycle" in str(i).lower()
        ]
        cutset = list(plan.get("cutset_anchors") or [])
        if cycle_issues and not cutset:
            result = SubagentResult(
                final_text=(
                    "Plan declared a dependency cycle without cutset_anchors; "
                    "refusing to silently break the cycle. Falling back to a "
                    "single agent. Issues: " + "; ".join(cycle_issues)
                ),
                subtasks=[
                    SubagentInvocation(
                        goal=invocation.goal,
                        mode="agent",
                        metadata={
                            "ccx_parent_mode": "plan",
                            "ccx_fallback": True,
                            "ccx_cycle_without_cutset": True,
                            "ccx_root_goal": root_goal,
                        },
                    ),
                ],
                extras={
                    "goal": root_goal,
                    "cutset_required": True,
                    "dependency_issues": cycle_issues,
                },
            )
            self._record(
                started_at=started_at,
                invocation=invocation,
                system=system, user=user, response=response,
                plan=plan, result=result,
                parse_status="cycle_without_cutset",
                sequential_reason="cycle without cutset_anchors → single agent",
            )
            return result

        cutset_applied = False
        if cycle_issues and cutset:
            # Materialize boundary-first DAG instead of the silently-broken order.
            dag = _items_to_id_dag(plan["items"])
            _ordered0, _issues0, residual = _topo_order_ex(dag, break_cycles=True)
            ordered, apply_issues = _apply_cutset(dag, cutset, cycle_ids=residual)
            plan = dict(plan)
            plan["dependency_issues"] = list(plan.get("dependency_issues") or []) + apply_issues
            plan["items"] = [
                {
                    "goal": n["goal"],
                    "depends_on_previous": False,
                    "depends_on": [],  # filled as indices below
                    **({"write_scope": list(n["write_scope"])} if n.get("write_scope") else {}),
                    **({"read_scope": list(n["read_scope"])} if n.get("read_scope") else {}),
                    **({"ccx_cutset_node": True} if n.get("ccx_cutset_node") else {}),
                    "_node_id": n["id"],
                    "_depends_on_ids": list(n.get("depends_on") or []),
                }
                for n in ordered
            ]
            id_to_index = {n["id"]: i for i, n in enumerate(ordered)}
            for item, node in zip(plan["items"], ordered):
                item["depends_on"] = [
                    id_to_index[d] for d in (node.get("depends_on") or [])
                    if d in id_to_index
                ]
            cutset_applied = True

        # Per-item dependency model: each subtask carries its own
        # depends_on_previous flag and (optionally) an explicit
        # depends_on index list. ``to_spawn_result`` honours these
        # individually so a mixed plan keeps independent items running
        # in parallel.
        flags = [bool(item.get("depends_on_previous")) for item in plan["items"]]
        any_explicit_dag = any(item.get("depends_on") for item in plan["items"])
        if cutset_applied:
            sequential_reason = "cycle rewritten via cutset_anchors (boundary-first)"
        elif any_explicit_dag:
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

        subtasks = []
        for i, item in enumerate(plan["items"]):
            meta: dict[str, Any] = {
                "ccx_parent_mode": "plan",
                "ccx_plan_index": i,
                "ccx_root_goal": root_goal,
                "ccx_depends_on_previous": bool(item.get("depends_on_previous")),
                "ccx_depends_on": list(item.get("depends_on") or []),
            }
            if item.get("write_scope"):
                meta["ccx_write_scope"] = list(item["write_scope"])
            if item.get("read_scope"):
                meta["ccx_read_scope"] = list(item["read_scope"])
            if item.get("ccx_cutset_node"):
                meta["ccx_cutset_node"] = True
            # Cutset boundary nodes run as agents (establish the freeze);
            # ordinary plan items still go through spec.
            mode = "agent" if item.get("ccx_cutset_node") else "spec"
            subtasks.append(SubagentInvocation(
                goal=item["goal"], mode=mode, metadata=meta,
            ))
        invariants = list(plan.get("invariants") or [])
        if invariants:
            subtasks = _stamp_invariants_on_subtasks(subtasks, invariants)
        extras: dict[str, Any] = {
            "goal": root_goal,
            "items": plan["items"],
        }
        if cutset:
            extras["cutset_anchors"] = cutset
        if cutset_applied:
            extras["cutset_applied"] = True
        if invariants:
            extras["invariants"] = list(invariants)
        if artifacts:
            extras["artifact_paths"] = artifacts
        # We deliberately keep `sequential=False` at the result level —
        # per-item metadata drives dependencies in to_spawn_result.
        result = SubagentResult(
            final_text=plan.get("rationale", ""),
            subtasks=subtasks,
            sequential=False,
            extras=extras,
        )
        self._record(
            started_at=started_at,
            invocation=invocation,
            system=system, user=user, response=response,
            plan=plan, result=result,
            parse_status="ok",
            sequential_reason=sequential_reason,
        )
        return result

    def _run_explicit_dag(
        self, invocation: SubagentInvocation, raw_dag: Any, *,
        root_goal: str, started_at: float,
    ) -> SubagentResult:
        """Materialize a goal-mode explicit DAG into agent subtasks (no LLM).

        ``raw_dag`` is the JSON-shaped list the goal loop stamped into
        ``metadata["ccx_goal_dag"]``: ``[{"id", "goal", "depends_on": [ids]}]``.
        Nodes are **topologically sorted** first so every declared edge becomes
        a backward reference that survives ``to_spawn_result``'s ``0 <= idx < i``
        wiring (a planner that lists nodes out of order would otherwise have its
        forward edges silently dropped). Duplicate ids and true cycles are
        handled deterministically and recorded as ``dependency_issues`` rather
        than silently lost. An empty / malformed DAG degrades to a single agent
        task on the goal, mirroring the no-items LLM fallback above.
        """
        nodes = raw_dag if isinstance(raw_dag, list) else []
        norm: list[dict[str, Any]] = []
        seen_ids: dict[str, int] = {}
        dependency_issues: list[str] = []
        dropped = 0
        for i, node in enumerate(nodes):
            if not isinstance(node, dict):
                dropped += 1
                continue
            node_goal = str(node.get("goal") or "").strip()
            if not node_goal:
                dropped += 1
                continue
            node_id = str(node.get("id") or node.get("node_id") or f"n{i}")
            if node_id in seen_ids:
                # Keep the first occurrence; a later dup id would otherwise make
                # dependency resolution ambiguous (last-write-wins).
                dependency_issues.append(
                    f"explicit DAG dropped duplicate node id {node_id!r}"
                )
                dropped += 1
                continue
            deps_raw = node.get("depends_on") or []
            deps = [str(d) for d in deps_raw] if isinstance(deps_raw, list) else []
            seen_ids[node_id] = len(norm)
            entry: dict[str, Any] = {
                "id": node_id, "goal": node_goal, "depends_on": deps,
            }
            ws = _as_str_list(node.get("write_scope") or node.get("ccx_write_scope"))
            rs = _as_str_list(node.get("read_scope") or node.get("ccx_read_scope"))
            if ws:
                entry["write_scope"] = ws
            if rs:
                entry["read_scope"] = rs
            norm.append(entry)
        if not norm:
            norm = [{"id": "n0", "goal": invocation.goal, "depends_on": []}]

        cutset = list(
            (invocation.metadata or {}).get("ccx_cutset_anchors")
            or (invocation.metadata or {}).get("cutset_anchors")
            or []
        )
        # Don't silently break cycles here — either fall back or rewrite via cutset.
        ordered, order_issues, residual = _topo_order_ex(
            norm, break_cycles=False,
        )
        dependency_issues.extend(order_issues)

        # Principle 3: fail-loud on cycles without cutset anchors.
        cycle_issues = [i for i in order_issues if "cycle" in str(i).lower()]
        cutset_applied = False
        if residual and not cutset:
            msg = (
                "explicit DAG has a dependency cycle without cutset_anchors; "
                "refusing to execute a silently broken topology. "
                + "; ".join(cycle_issues)
            )
            result = SubagentResult(
                final_text=msg,
                subtasks=[
                    SubagentInvocation(
                        goal=invocation.goal,
                        mode="agent",
                        metadata={
                            "ccx_parent_mode": "plan",
                            "ccx_goal_explicit": True,
                            "ccx_cycle_without_cutset": True,
                            "ccx_fallback": True,
                            "ccx_root_goal": root_goal,
                        },
                    ),
                ],
                sequential=False,
                extras={
                    "goal": root_goal,
                    "ccx_goal_explicit": True,
                    "cutset_required": True,
                    "dependency_issues": dependency_issues,
                },
            )
            self._record(
                started_at=started_at,
                invocation=invocation,
                system="(goal-mode explicit DAG — cycle without cutset)",
                user="",
                response="",
                plan={
                    "items": ordered, "rationale": "", "dropped": dropped,
                    "dependency_issues": dependency_issues,
                },
                result=result,
                parse_status="cycle_without_cutset",
                sequential_reason="explicit DAG cycle without cutset → single agent",
            )
            return result

        if residual and cutset:
            # Rewrite into boundary-first DAG instead of accepting the silent break.
            ordered, apply_issues = _apply_cutset(
                norm, cutset, cycle_ids=residual,
            )
            dependency_issues.extend(apply_issues)
            cutset_applied = True

        id_to_index = {node["id"]: idx for idx, node in enumerate(ordered)}
        subtasks: list[SubagentInvocation] = []
        for idx, node in enumerate(ordered):
            dep_indices: list[int] = []
            for dep in node["depends_on"]:
                j = id_to_index.get(str(dep))
                if j is None:
                    dependency_issues.append(
                        f"node {node['id']!r} ignored dangling depends_on "
                        f"{dep!r}"
                    )
                    continue
                if j == idx:
                    dependency_issues.append(
                        f"node {node['id']!r} ignored self-dependency"
                    )
                    continue
                if j not in dep_indices:
                    dep_indices.append(j)
            meta: dict[str, Any] = {
                "ccx_parent_mode": "plan",
                "ccx_plan_index": idx,
                "ccx_root_goal": root_goal,
                "ccx_goal_explicit": True,
                "ccx_depends_on": dep_indices,
            }
            if node.get("ccx_cutset_node"):
                meta["ccx_cutset_node"] = True
            if node.get("write_scope"):
                meta["ccx_write_scope"] = list(node["write_scope"])
            if node.get("read_scope"):
                meta["ccx_read_scope"] = list(node["read_scope"])
            subtasks.append(SubagentInvocation(
                goal=node["goal"],
                mode="agent",
                metadata=meta,
            ))

        extras: dict[str, Any] = {
            "goal": root_goal,
            "ccx_goal_explicit": True,
        }
        if cutset:
            extras["cutset_anchors"] = list(cutset)
        if cutset_applied:
            extras["cutset_applied"] = True
        if dependency_issues:
            extras["dependency_issues"] = dependency_issues

        result = SubagentResult(
            final_text=(
                f"(goal mode: materialized {len(subtasks)} explicit DAG node(s))"
            ),
            subtasks=subtasks,
            sequential=False,
            extras=extras,
        )
        self._record(
            started_at=started_at,
            invocation=invocation,
            system="(goal-mode explicit DAG — no LLM call)",
            user="",
            response="",
            plan={
                "items": ordered, "rationale": "", "dropped": dropped,
                "dependency_issues": dependency_issues,
                **({"cutset_applied": True} if cutset_applied else {}),
            },
            result=result,
            parse_status="explicit",
            sequential_reason=(
                "goal-mode explicit DAG + cutset rewrite"
                if cutset_applied
                else "goal-mode explicit DAG materialized deterministically"
            ),
        )
        return result

    def _record(
        self, *, started_at: float, invocation: SubagentInvocation,
        system: str, user: str, response: str,
        plan: dict[str, Any], result: SubagentResult,
        parse_status: str, sequential_reason: str,
    ) -> None:
        if self.tracer is None:
            return
        issues: list[str] = []
        item_count = len(plan.get("items") or [])
        if parse_status == "ok" and item_count == 1:
            issues.append(
                "plan returned only 1 item — consider whether decomposition "
                "is adding value here"
            )
        if parse_status == "ok" and item_count > 5:
            issues.append(
                f"plan returned {item_count} items (>5) — over-decomposition"
            )
        if parse_status == "ok" and not plan.get("rationale"):
            issues.append("plan returned no rationale — hard to audit")

        issues.extend(str(item) for item in plan.get("dependency_issues") or [])
        root_goal = str(
            invocation.metadata.get("ccx_root_goal")
            or current_goal_text(invocation.goal, invocation.metadata)
        )

        record = ModeStepRecord(
            mode="plan",
            invocation_goal=invocation.goal,
            parent_goal=root_goal,
            metadata=dict(invocation.metadata),
            started_at=started_at,
            finished_at=time.time(),
            system_prompt=system,
            user_prompt=user,
            raw_response=response,
            parse_status=parse_status,
            parsed_items=list(plan.get("items") or []),
            dropped_items=int(plan.get("dropped") or 0),
            rationale=str(plan.get("rationale") or ""),
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
        )
        self.tracer.record_step(record)


def _topo_order(
    norm: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Kahn topological sort of explicit-DAG nodes by their ``depends_on`` ids.

    Returns ``(ordered_nodes, issues)``. Only edges between known ids constrain
    ordering (dangling refs are left for the caller to record). Ties break by
    original index so the result is deterministic. A residual cycle is broken
    deterministically (the still-blocked nodes are appended in original order)
    and reported in ``issues`` — never an infinite loop, never a dropped node.

    Prefer :func:`_apply_cutset` when a cycle has declared anchors — that path
    rewrites the graph instead of silently appending residual nodes.
    """
    ordered, issues, _residual = _topo_order_ex(norm, break_cycles=True)
    return ordered, issues


def _topo_order_ex(
    norm: list[dict[str, Any]],
    *,
    break_cycles: bool = True,
) -> tuple[list[dict[str, Any]], list[str], set[str]]:
    """Like :func:`_topo_order`, also returning residual cycle member ids."""
    ids = {node["id"] for node in norm}
    index_of = {node["id"]: i for i, node in enumerate(norm)}
    indeg: dict[str, int] = {}
    children: dict[str, list[str]] = {nid: [] for nid in index_of}
    for node in norm:
        deps = {d for d in node["depends_on"] if d in ids and d != node["id"]}
        indeg[node["id"]] = len(deps)
        for d in deps:
            children[d].append(node["id"])

    ready = sorted((nid for nid, d in indeg.items() if d == 0),
                   key=lambda n: index_of[n])
    ordered_ids: list[str] = []
    while ready:
        nid = ready.pop(0)
        ordered_ids.append(nid)
        for child in children[nid]:
            indeg[child] -= 1
            if indeg[child] == 0:
                ready.append(child)
                ready.sort(key=lambda n: index_of[n])

    issues: list[str] = []
    residual = {
        n["id"] for n in norm if n["id"] not in set(ordered_ids)
    }
    if residual:
        remaining = [n["id"] for n in norm if n["id"] in residual]
        if break_cycles:
            issues.append(
                "explicit DAG has a dependency cycle among "
                f"{remaining}; broke it by appending those nodes in original order"
            )
            ordered_ids.extend(remaining)
        else:
            issues.append(
                "explicit DAG has a dependency cycle among "
                f"{remaining}"
            )
    return (
        [norm[index_of[nid]] for nid in ordered_ids if nid in index_of],
        issues,
        residual,
    )


def _apply_cutset(
    norm: list[dict[str, Any]],
    cutset_anchors: list[str],
    *,
    cycle_ids: set[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Rewrite a cyclic DAG into cutset-first then fan-out (principle 3).

    Injects synthetic ``cutset:{i}`` boundary nodes, strips intra-cycle edges,
    and makes every cycle member depend on all cutset nodes. Re-runs Kahn so
    the result is acyclic: boundary wave first, then the former cycle nodes.
    """
    anchors = [str(a).strip() for a in cutset_anchors if str(a).strip()]
    if not anchors or not cycle_ids:
        ordered, issues = _topo_order(norm)
        return ordered, issues

    cutset_nodes: list[dict[str, Any]] = []
    cutset_ids: list[str] = []
    existing = {n["id"] for n in norm}
    for i, anchor in enumerate(anchors):
        cid = f"cutset:{i}"
        # Avoid colliding with a planner-supplied id.
        n = 0
        while cid in existing or cid in cutset_ids:
            n += 1
            cid = f"cutset:{i}.{n}"
        cutset_ids.append(cid)
        cutset_nodes.append({
            "id": cid,
            "goal": f"Establish boundary: {anchor}",
            "depends_on": [],
            "ccx_cutset_node": True,
        })

    rewritten: list[dict[str, Any]] = []
    for node in norm:
        nid = str(node["id"])
        deps = [str(d) for d in (node.get("depends_on") or [])]
        if nid in cycle_ids:
            deps = [d for d in deps if d not in cycle_ids]
            for cid in cutset_ids:
                if cid not in deps:
                    deps.append(cid)
        new_node = dict(node)
        new_node["depends_on"] = deps
        rewritten.append(new_node)

    combined = cutset_nodes + rewritten
    ordered, order_issues, residual = _topo_order_ex(combined, break_cycles=True)
    issues = [
        f"cutset applied: injected {len(cutset_ids)} boundary node(s); "
        f"stripped intra-cycle edges among {sorted(cycle_ids)}",
    ]
    issues.extend(order_issues)
    if residual:
        issues.append(
            f"cutset rewrite still residual cycle among {sorted(residual)} "
            "(appended in original order)"
        )
    return ordered, issues


def _items_to_id_dag(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert index-based plan items into an id-keyed DAG for cutset rewrite."""
    out: list[dict[str, Any]] = []
    for i, item in enumerate(items):
        entry: dict[str, Any] = {
            "id": f"n{i}",
            "goal": str(item.get("goal") or ""),
            "depends_on": [f"n{d}" for d in (item.get("depends_on") or [])],
        }
        if item.get("write_scope"):
            entry["write_scope"] = list(item["write_scope"])
        if item.get("read_scope"):
            entry["read_scope"] = list(item["read_scope"])
        out.append(entry)
    return out


def _parse_plan_response(response: str) -> dict[str, Any]:
    """Best-effort parser.

    Accepts either bare JSON or JSON wrapped in ```json``` fences. If parsing
    fails entirely, returns an empty plan so the caller can fall back.

    Returned shape: ``{"items": list[dict], "rationale": str, "dropped": int}``
    where ``dropped`` is the count of raw entries that could not be coerced
    into a usable item (missing goal, wrong type, etc.).
    """
    data = parse_llm_json(
        response,
        schema_name="plan",
        fallback_factory=lambda _raw: {},
        expected_type=dict,
    )
    items = data.get("plan_items") or data.get("items") or []
    rationale = data.get("rationale") or data.get("summary") or ""
    cutset_anchors = _as_str_list(data.get("cutset_anchors"))
    invariants = _as_str_list(data.get("invariants"))
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
            item_dict: dict[str, Any] = {
                "goal": str(raw["goal"]),
                "depends_on_previous": bool(raw.get("depends_on_previous", False)),
                "depends_on": [],
            }
            ws = _as_str_list(raw.get("write_scope") or raw.get("ccx_write_scope"))
            rs = _as_str_list(raw.get("read_scope") or raw.get("ccx_read_scope"))
            if ws:
                item_dict["write_scope"] = ws
            if rs:
                item_dict["read_scope"] = rs
            pending_items.append((original_index, item_dict, depends_on))
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
                    f"plan item {original_index} ignored dangling depends_on "
                    f"index {idx}"
                )
                continue
            mapped = old_to_new[idx]
            if mapped == new_index:
                dependency_issues.append(
                    f"plan item {original_index} ignored self depends_on "
                    f"index {idx}"
                )
                continue
            if mapped in seen:
                dependency_issues.append(
                    f"plan item {original_index} ignored duplicate depends_on "
                    f"index {idx}"
                )
                continue
            seen.add(mapped)
            remapped.append(mapped)
        item["depends_on"] = remapped
        cleaned_items.append(item)
    cleaned_items, order_issues = _order_for_backward_deps(cleaned_items, "plan")
    dependency_issues.extend(order_issues)
    return {
        "items": cleaned_items,
        "rationale": rationale,
        "dropped": dropped,
        "dependency_issues": dependency_issues,
        "cutset_anchors": cutset_anchors,
        "invariants": invariants,
    }


def _as_str_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw] if raw.strip() else []
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw if str(x).strip()]
    return []


__all__ = ["PlanModeRunner", "build_plan_prompt"]
