"""Plan/spec/agent diagnostic tracer.

Captures every ModeRunner invocation in fine-grained detail so callers can
later evaluate planning quality:

* Was the planner's prompt + raw LLM response well-formed?
* How many items came back? How many were dropped during parsing?
* Did the planner correctly use ``depends_on_previous`` to express ordering,
  or did it lose information by collapsing partial dependency into a chain?
* What execution order was actually realised on the v5 DAG, and does it
  match the planned order?

The tracer is intentionally a side channel — runners stay correct without
it. Pass ``tracer=PlanDiagnosticsTracer()`` to a runner to opt in.

Thread-safe by virtue of a single ``Lock`` around the records list — v5
runs sibling subagents in parallel so multiple runners may write at once.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ModeStepRecord:
    """One invocation of a ModeRunner.run."""
    mode: str
    invocation_goal: str
    parent_goal: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at: float = 0.0
    finished_at: float = 0.0
    system_prompt: str = ""
    user_prompt: str = ""
    raw_response: str = ""
    parse_status: str = "ok"
    parsed_items: list[dict[str, Any]] = field(default_factory=list)
    dropped_items: int = 0
    rationale: str = ""
    sequential: bool = False
    sequential_reason: str = ""
    final_text: str = ""
    spawned_subtasks: list[dict[str, Any]] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return max(0.0, self.finished_at - self.started_at)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class NodeEventRecord:
    """One v5 event of interest (node lifecycle)."""
    sequence: int
    kind: str
    node_id: str
    timestamp: float
    payload: dict[str, Any] = field(default_factory=dict)


class PlanDiagnosticsTracer:
    """Collects ModeStepRecords and v5 NodeEventRecords for one run."""

    def __init__(self) -> None:
        # Reentrant — to_json/to_markdown acquire the lock and then call
        # summary() / execution_order() which also acquire it.
        self._lock = threading.RLock()
        self.records: list[ModeStepRecord] = []
        self.node_events: list[NodeEventRecord] = []

    # -- runner-side recording ------------------------------------------------

    def record_step(self, record: ModeStepRecord) -> None:
        with self._lock:
            self.records.append(record)

    # -- v5 event bus subscription -------------------------------------------

    def attach_event_bus(self, event_bus: Any) -> None:
        """Subscribe to v5 ``node.*`` events. Safe to call once per run."""
        def on_event(event: dict[str, Any]) -> None:
            kind = event.get("kind", "")
            if not kind.startswith("node."):
                return
            payload = event.get("payload") or {}
            with self._lock:
                self.node_events.append(NodeEventRecord(
                    sequence=int(event.get("sequence") or 0),
                    kind=kind,
                    node_id=str(payload.get("node_id", "")),
                    timestamp=time.time(),
                    payload=dict(payload),
                ))
        event_bus.subscribe(on_event, kind="node.")

    # -- analysis -------------------------------------------------------------

    def by_mode(self, mode: str) -> list[ModeStepRecord]:
        with self._lock:
            return [r for r in self.records if r.mode == mode]

    def execution_order(self) -> list[str]:
        """Order in which nodes reached SUCCEEDED state."""
        with self._lock:
            return [
                e.node_id for e in self.node_events
                if e.kind == "node.succeeded"
            ]

    def dispatch_order(self) -> list[str]:
        """Order in which nodes were dispatched (claimed/started)."""
        with self._lock:
            return [
                e.node_id for e in self.node_events
                if e.kind in {"node.claimed", "node.running",
                              "node.dispatched"}
            ]

    def summary(self) -> dict[str, Any]:
        """Aggregate planning-quality metrics."""
        with self._lock:
            records = list(self.records)
        plan_records = [r for r in records if r.mode == "plan"]
        spec_records = [r for r in records if r.mode == "spec"]
        agent_records = [r for r in records if r.mode == "agent"]

        all_issues: list[dict[str, Any]] = []
        for r in records:
            for issue in r.issues:
                all_issues.append({
                    "mode": r.mode,
                    "goal": r.invocation_goal,
                    "issue": issue,
                })

        plan_total_items = sum(len(r.parsed_items) for r in plan_records)
        spec_total_items = sum(len(r.parsed_items) for r in spec_records)

        # Dependency-shape signals: how many plan/spec records have
        # truly mixed (parallel + chained), all-chained, all-parallel,
        # or use explicit depends_on indices. Mixed and explicit are
        # *good* — they preserve parallelism.
        mixed_records = 0
        all_chain_records = 0
        all_parallel_records = 0
        explicit_dag_records = 0
        for r in plan_records + spec_records:
            flags = [bool(item.get("depends_on_previous"))
                     for item in r.parsed_items]
            has_explicit = any(item.get("depends_on")
                               for item in r.parsed_items)
            if has_explicit:
                explicit_dag_records += 1
            elif any(flags) and not all(flags):
                mixed_records += 1
            elif all(flags) and flags:
                all_chain_records += 1
            elif r.parsed_items:
                all_parallel_records += 1

        return {
            "step_counts": {
                "plan": len(plan_records),
                "spec": len(spec_records),
                "agent": len(agent_records),
                "total": len(records),
            },
            "items": {
                "plan_items": plan_total_items,
                "spec_items": spec_total_items,
                "dropped_items_total": sum(r.dropped_items for r in records),
            },
            "dependency_shapes": {
                "mixed_records": mixed_records,
                "all_chain_records": all_chain_records,
                "all_parallel_records": all_parallel_records,
                "explicit_dag_records": explicit_dag_records,
            },
            "issues": all_issues,
            "node_event_count": len(self.node_events),
            "execution_order": self.execution_order(),
        }

    # -- pretty-printers ------------------------------------------------------

    def to_markdown(self) -> str:
        lines: list[str] = ["# Plan-mode diagnostic trace", ""]
        with self._lock:
            records = list(self.records)
            events = list(self.node_events)

        lines.append(f"Total mode-runner invocations: **{len(records)}**")
        lines.append(f"Total v5 node events captured: **{len(events)}**")
        lines.append("")

        for i, r in enumerate(records, 1):
            lines += [
                f"## Step {i}: `{r.mode}` — {r.invocation_goal!r}",
                "",
                f"- duration: {r.duration_s * 1000:.1f} ms",
                f"- parent_goal: {r.parent_goal!r}",
                f"- parse_status: `{r.parse_status}`",
                f"- parsed_items: {len(r.parsed_items)}"
                f" (dropped: {r.dropped_items})",
                f"- sequential: {r.sequential}"
                f" — _{r.sequential_reason or 'n/a'}_",
                "",
            ]
            if r.parsed_items:
                lines.append("**Parsed items**:")
                for j, it in enumerate(r.parsed_items):
                    dep = "→" if it.get("depends_on_previous") else "•"
                    lines.append(f"  {j + 1}. {dep} {it.get('goal', '')!r}")
                lines.append("")
            if r.spawned_subtasks:
                lines.append("**Spawned subtasks**:")
                for st in r.spawned_subtasks:
                    lines.append(
                        f"  - mode={st.get('mode', '')} "
                        f"goal={st.get('goal', '')!r} "
                        f"depends_on_previous={st.get('depends_on_previous', False)}"
                    )
                lines.append("")
            if r.issues:
                lines.append("**Issues**:")
                for issue in r.issues:
                    lines.append(f"  - {issue}")
                lines.append("")

        order = self.execution_order()
        if order:
            lines += ["## Execution order (succeeded events)", ""]
            for nid in order:
                lines.append(f"- {nid}")
            lines.append("")
        return "\n".join(lines)

    def to_json(self) -> str:
        with self._lock:
            return json.dumps({
                "records": [r.to_dict() for r in self.records],
                "node_events": [asdict(e) for e in self.node_events],
                "summary": self.summary(),
            }, indent=2, ensure_ascii=False, default=str)


__all__ = [
    "ModeStepRecord",
    "NodeEventRecord",
    "PlanDiagnosticsTracer",
]
