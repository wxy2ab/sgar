"""ccx fixed declarative-DAG entry — run a *pre-built* ``list[NodeSpec]``.

For mature, shape-stable repeat tasks (e.g. a nightly sales-conversion
report: read DB → build report → analyse → conclude) there is no reason to
pay an LLM to re-decompose the same work every night. The decomposition is
already known; only the DAG needs running. ``run_spec`` is the public entry
that executes a caller-authored DAG directly, skipping the LLM planner.

This is deliberately thin: it is the same v5 assembly the ccx
``SwarmCoordinator`` already uses (``build_runtime`` with a fixed
``propose_initial`` + one ``engine.run``), lifted into a public function so a
caller does not have to reach through the swarm dataclasses to get at it. It
changes nothing in the engine / controller / dispatcher.

What it does NOT do (by design — see reports/ccx_fixed_dag_phaseA_ab.md):

* It is not a "workflow engine": there is no named-recipe table, no
  task→recipe matcher, no fuzzy dispatch. The caller passes the exact specs.
* Determinism comes from the *content* of the nodes the caller writes (SQL /
  templates inline in ``NodeSpec.params``, side effects behind
  ``requires_approval`` / an external once-guard), not from this function. A
  DAG whose nodes are all ``ccx.agent`` LLM tools is topology-frozen but still
  non-deterministic — the shape is fixed, the node behaviour is not.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from core.deepstack_v5 import NodeSpec, NodeState, RunStatus, ToolSpec, Verdict

from .modes.llm_client import LLMCallable
from .runtime import build_runtime


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RunSpecNode:
    """Terminal view of one node after a ``run_spec`` run.

    ``result`` is the tool's raw return value (a dict for mode tools, whatever
    a custom capability returned otherwise); ``error`` is populated only when
    the node did not reach SUCCEEDED.
    """

    node_id: str
    tool: str
    state: str
    success: bool
    result: Any | None = None
    error: str | None = None


@dataclass(slots=True)
class RunSpecResult:
    """Outcome of :func:`run_spec`.

    ``verdict`` is the v5 ``Verdict`` (run-level status + counts) exactly as
    ``engine.run`` produced it. ``nodes`` is read back from the live
    GraphStore *before* the runtime is shut down, so per-node results survive
    the DB close.
    """

    verdict: Verdict
    run_id: str
    nodes: dict[str, RunSpecNode] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        """True iff the run terminated COMPLETED with EVERY node SUCCEEDED.

        v5 treats *partial* completion as a COMPLETED run: a node that
        exhausts its retries goes to ABANDONED (not FAILED), and a node whose
        dependency did not succeed is SKIPPED — in both cases ``verdict.failed``
        stays 0 while the run status is COMPLETED. Checking status + failed
        alone would therefore call a half-broken pipeline a success, so we
        require no node to have ended in any non-SUCCEEDED terminal state.
        """
        v = self.verdict
        return (
            v.status == RunStatus.COMPLETED
            and v.failed == 0
            and v.abandoned == 0
            and v.skipped == 0
            and v.cancelled == 0
        )

    def node(self, node_id: str) -> RunSpecNode | None:
        return self.nodes.get(node_id)

    @property
    def final_texts(self) -> list[str]:
        """``final_text`` of every succeeded node that carried one, in node
        creation order (``list_nodes`` orders by ``created_at_ms``; under
        parallel dispatch this is not strictly topological run order)."""
        out: list[str] = []
        for node in self.nodes.values():
            if node.success and isinstance(node.result, dict):
                text = node.result.get("final_text")
                if text:
                    out.append(str(text))
        return out


def _topo_order(specs: list[NodeSpec]) -> list[NodeSpec]:
    """Return ``specs`` in a dependency-respecting order, or fail loud.

    ``WorkGraph.add`` rejects forward references (a dep must be added before
    its dependent), so the engine needs specs pre-ordered. Rather than force
    the caller to hand-order a declared DAG, we Kahn-sort here and fail loud on
    the three ways a hand-written DAG goes wrong: duplicate ``node_id``, a
    ``depends_on`` pointing at an unknown node, or a cycle. Each raises a
    distinctly-coded ``ValueError`` so a caller sees which one, not a generic
    engine FAILED verdict three layers down.
    """
    if not specs:
        raise ValueError("CCX_FIXED_DAG_EMPTY: run_spec requires at least one NodeSpec")

    by_id: dict[str, NodeSpec] = {}
    for spec in specs:
        if spec.node_id in by_id:
            raise ValueError(
                f"CCX_FIXED_DAG_DUPLICATE_NODE: node_id {spec.node_id!r} "
                "appears more than once"
            )
        by_id[spec.node_id] = spec

    indegree: dict[str, int] = {node_id: 0 for node_id in by_id}
    dependents: dict[str, list[str]] = {node_id: [] for node_id in by_id}
    for spec in specs:
        for dep in spec.depends_on:
            if dep not in by_id:
                raise ValueError(
                    f"CCX_FIXED_DAG_UNKNOWN_DEP: node {spec.node_id!r} "
                    f"depends on unknown node {dep!r}"
                )
            indegree[spec.node_id] += 1
            dependents[dep].append(spec.node_id)

    # Kahn, draining in the caller's declared order among ready nodes so the
    # output is stable (not set-iteration dependent).
    ready = [spec.node_id for spec in specs if indegree[spec.node_id] == 0]
    ordered: list[NodeSpec] = []
    while ready:
        node_id = ready.pop(0)
        ordered.append(by_id[node_id])
        for dependent in dependents[node_id]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                ready.append(dependent)

    if len(ordered) != len(specs):
        remaining = sorted(set(by_id) - {spec.node_id for spec in ordered})
        raise ValueError(
            f"CCX_FIXED_DAG_CYCLE: dependency cycle among nodes {remaining}"
        )
    return ordered


def run_spec(
    specs: list[NodeSpec],
    *,
    workspace: Path | str,
    llm: LLMCallable | None = None,
    llm_client_provider: Any | None = None,
    cc_config: Any | None = None,
    label: str = "ccx-fixed-dag",
    language: str = "en",
    parallelism: int = 4,
    agent_runner_kind: str = "lite",
    extra_capabilities: Mapping[str, ToolSpec] | None = None,
    node_timeout_s: float | None = None,
    budget: Any | None = None,
    cc_cwd: str | None = None,
) -> RunSpecResult:
    """Execute a pre-built declarative DAG, skipping the LLM planner.

    ``specs`` is the exact node set to run. Edges are declared via
    ``NodeSpec.depends_on``; the specs are topologically sorted here (and
    validated — duplicate ids / unknown deps / cycles fail loud). Custom
    deterministic tools (a fixed SQL query, a report template, a schema
    preflight) are registered via ``extra_capabilities`` and referenced by
    ``NodeSpec.tool``; the built-in ccx mode tools (``ccx.agent`` etc.) remain
    available for the nodes that genuinely need an LLM.

    Mirrors the v5 assembly in ``agents/swarm/coordinator.py``: one
    ``build_runtime`` with ``propose_initial`` fixed to the sorted specs, one
    synchronous ``engine.run``. Per-node results are read from the live
    GraphStore before shutdown. Either ``llm`` or ``llm_client_provider`` must
    be supplied (``build_runtime`` builds the mode-tool runners either way,
    even for an all-deterministic DAG).

    Returns a :class:`RunSpecResult` carrying the ``Verdict`` plus per-node
    terminal state. Raises ``ValueError`` for a malformed DAG (before any
    runtime is built); an operational engine failure surfaces on the Verdict's
    status, not as an exception (v5's own contract).
    """
    ordered = _topo_order(specs)

    bundle = build_runtime(
        workspace=workspace,
        llm=llm,
        llm_client_provider=llm_client_provider,
        cc_config=cc_config,
        language=language,
        parallelism=parallelism,
        propose_initial=lambda _goal: list(ordered),
        agent_runner_kind=agent_runner_kind,
        extra_capabilities=extra_capabilities,
        node_timeout_s=node_timeout_s,
        budget=budget,
        cc_cwd=cc_cwd,
    )

    run_id: str | None = None
    try:
        engine = bundle.runtime.engine()
        bundle.engine = engine
        verdict = engine.run(goal=label)
        run_id = verdict.run_id
        # Read node rows from the LIVE bundle's GraphStore before shutdown, so
        # per-node results survive the DB close (same discipline as the swarm
        # coordinator — reopening after shutdown can race worker connections).
        nodes: dict[str, RunSpecNode] = {}
        for row in bundle.runtime.graph_store.list_nodes(run_id):
            node_id = row["node_id"]
            state = row["state"]
            success = state == NodeState.SUCCEEDED.value
            error: str | None = None
            if not success:
                failure = row.get("failure")
                if failure:
                    error = failure.get("message") or str(failure)
                else:
                    error = f"node ended in state {state!r}"
            nodes[node_id] = RunSpecNode(
                node_id=node_id,
                tool=row["spec"].get("tool", ""),
                state=state,
                success=success,
                result=row.get("result"),
                error=error,
            )
        return RunSpecResult(verdict=verdict, run_id=run_id, nodes=nodes)
    finally:
        bundle.shutdown(run_id=run_id)


__all__ = [
    "RunSpecNode",
    "RunSpecResult",
    "run_spec",
]
