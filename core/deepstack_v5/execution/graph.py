"""WorkGraph — the in-memory DAG that tracks all NodeExecutions for a run.

Public surface kept small:
* `add(spec)` — append a node; rejects cycles and duplicate IDs.
* `mark(node_id, state, reason)` — drive the state machine for one node.
* `ready()` — node IDs whose dependencies have all SUCCEEDED and which are
  themselves PENDING/READY/BLOCKED-with-deps-now-ok/TIMER_HANG with timer
  cleared.
* `snapshot()` — read-only summary for Controller / persistence.

Persistence is the caller's responsibility: WorkGraph itself stays in
memory; engine writes through GraphStore.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Iterable, Mapping

from ..types import (
    NodeSpec,
    NodeState,
    TERMINAL_NODE_STATES,
)
from .node import NodeExecution


class CycleError(ValueError):
    pass


class DuplicateNodeError(ValueError):
    pass


class UnknownNodeError(KeyError):
    pass


@dataclass(slots=True)
class GraphSnapshot:
    nodes: Mapping[str, NodeExecution]
    edges: Mapping[str, tuple[str, ...]]  # node_id -> deps
    ready: tuple[str, ...]
    counts: Mapping[str, int]


class WorkGraph:
    def __init__(self) -> None:
        self._nodes: dict[str, NodeExecution] = {}
        # node_id -> tuple of dependency node_ids
        self._edges: dict[str, tuple[str, ...]] = {}
        self._lock = threading.RLock()

    # -- mutation ------------------------------------------------------------

    def add(self, spec: NodeSpec) -> NodeExecution:
        with self._lock:
            if spec.node_id in self._nodes:
                raise DuplicateNodeError(spec.node_id)
            # Validate deps exist (forward references not allowed: caller must
            # add deps before dependents).
            for dep in spec.depends_on:
                if dep not in self._nodes:
                    raise UnknownNodeError(
                        f"node {spec.node_id} depends on unknown node {dep}"
                    )
            node = NodeExecution(spec=spec)
            self._nodes[spec.node_id] = node
            self._edges[spec.node_id] = tuple(spec.depends_on)
            if self._has_cycle():
                # Roll back.
                del self._nodes[spec.node_id]
                del self._edges[spec.node_id]
                raise CycleError(f"adding {spec.node_id} introduces a cycle")
            return node

    def add_execution(
        self,
        node: NodeExecution,
        *,
        validate_deps: bool = True,
    ) -> NodeExecution:
        """Insert a rehydrated NodeExecution.

        WorkerHarness uses ``validate_deps=False`` because a READY row from
        the DB already encodes dependency satisfaction and may be dispatched
        without reconstructing the whole DAG.
        """
        with self._lock:
            if node.node_id in self._nodes:
                raise DuplicateNodeError(node.node_id)
            if validate_deps:
                for dep in node.spec.depends_on:
                    if dep not in self._nodes:
                        raise UnknownNodeError(
                            f"node {node.node_id} depends on unknown node {dep}"
                        )
            self._nodes[node.node_id] = node
            self._edges[node.node_id] = tuple(node.spec.depends_on)
            if validate_deps and self._has_cycle():
                del self._nodes[node.node_id]
                del self._edges[node.node_id]
                raise CycleError(f"adding {node.node_id} introduces a cycle")
            return node

    def replace_spec(self, node_id: str, spec: NodeSpec) -> NodeExecution:
        with self._lock:
            node = self._require(node_id)
            missing = [dep for dep in spec.depends_on if dep not in self._nodes]
            if missing:
                raise UnknownNodeError(
                    f"node {node_id} depends on unknown node {missing[0]}"
                )
            old_spec = node.spec
            old_edges = self._edges.get(node_id, ())
            node.spec = spec
            self._edges[node_id] = tuple(spec.depends_on)
            if self._has_cycle():
                node.spec = old_spec
                self._edges[node_id] = old_edges
                raise CycleError(f"replacing {node_id} introduces a cycle")
            return node

    def replace_execution(
        self,
        node: NodeExecution,
        *,
        validate_deps: bool = False,
    ) -> NodeExecution:
        """Replace a rehydrated node while holding the graph lock."""
        with self._lock:
            if node.node_id not in self._nodes:
                raise UnknownNodeError(node.node_id)
            if validate_deps:
                for dep in node.spec.depends_on:
                    if dep not in self._nodes:
                        raise UnknownNodeError(
                            f"node {node.node_id} depends on unknown node {dep}"
                        )
            old_node = self._nodes[node.node_id]
            old_edges = self._edges.get(node.node_id, ())
            self._nodes[node.node_id] = node
            self._edges[node.node_id] = tuple(node.spec.depends_on)
            if validate_deps and self._has_cycle():
                self._nodes[node.node_id] = old_node
                self._edges[node.node_id] = old_edges
                raise CycleError(f"replacing {node.node_id} introduces a cycle")
            return node

    def add_many(self, specs: Iterable[NodeSpec]) -> list[NodeExecution]:
        return [self.add(s) for s in specs]

    def mark(
        self,
        node_id: str,
        state: NodeState,
        *,
        reason: str = "",
    ) -> NodeExecution:
        with self._lock:
            node = self._require(node_id)
            node.transition(state, reason=reason)
            # Propagate SKIPPED downstream when a node enters a terminal failure
            # state that should cascade. We only auto-skip for ABANDONED
            # (terminal failure); other terminal states (CANCELLED) cascade only
            # via explicit caller action.
            if state == NodeState.ABANDONED:
                self._cascade_skip(node_id)
            return node

    def cascade_skip_from(self, node_id: str, *, reason: str = "") -> None:
        """Skip non-terminal descendants of an explicitly terminal node."""
        with self._lock:
            if node_id not in self._nodes:
                raise UnknownNodeError(node_id)
            for desc in self._descendants(node_id):
                n = self._nodes.get(desc)
                if n is None or n.is_terminal():
                    continue
                if n.state in (NodeState.PENDING, NodeState.READY, NodeState.BLOCKED):
                    n.transition(
                        NodeState.SKIPPED,
                        reason=reason or f"upstream {node_id} skipped",
                    )

    def _cascade_skip(self, failed_id: str) -> None:
        # Any node that depends transitively on failed_id, if not yet
        # terminal, transitions PENDING/READY/BLOCKED -> SKIPPED.
        descendants = self._descendants(failed_id)
        for desc in descendants:
            n = self._nodes.get(desc)
            if n is None:
                continue
            if n.is_terminal():
                continue
            # Allowed transitions to SKIPPED: from PENDING, READY (not from
                # RUNNING — running ones must be cancelled by dispatcher).
            if n.state in (NodeState.PENDING, NodeState.READY,
                           NodeState.BLOCKED):
                n.transition(NodeState.SKIPPED,
                             reason=f"upstream {failed_id} abandoned")

    # -- queries -------------------------------------------------------------

    def get(self, node_id: str) -> NodeExecution:
        with self._lock:
            return self._require(node_id)

    def has(self, node_id: str) -> bool:
        with self._lock:
            return node_id in self._nodes

    def nodes(self) -> Mapping[str, NodeExecution]:
        with self._lock:
            return dict(self._nodes)

    def edges(self) -> Mapping[str, tuple[str, ...]]:
        with self._lock:
            return dict(self._edges)

    def ready(self) -> list[str]:
        """Nodes currently in READY state (eligible for dispatch).

        Note: this does NOT include PENDING nodes whose deps just became
        satisfied — those need explicit promotion via
        `transition_pending_to_ready()`. Keeping the two operations
        separate makes the engine loop's ordering explicit.
        """
        with self._lock:
            return [
                nid for nid, n in self._nodes.items()
                if n.state == NodeState.READY
            ]

    def _dep_satisfied(self, dep_id: str) -> bool:
        dep = self._nodes.get(dep_id)
        if dep is None:
            return False
        return dep.state == NodeState.SUCCEEDED

    def transition_pending_to_ready(self) -> list[str]:
        """Promote PENDING nodes whose deps are satisfied to READY.

        Engine calls this between dispatch cycles. Returns the list of
        node IDs that were promoted.
        """
        with self._lock:
            promoted: list[str] = []
            for node_id, node in self._nodes.items():
                if node.state not in (NodeState.PENDING, NodeState.BLOCKED):
                    continue
                deps = self._edges.get(node_id, ())
                if all(self._dep_satisfied(d) for d in deps):
                    node.transition(NodeState.READY, reason="deps satisfied")
                    promoted.append(node_id)
            return promoted

    def all_terminal(self) -> bool:
        with self._lock:
            return all(n.is_terminal() for n in self._nodes.values())

    def has_failed(self) -> bool:
        with self._lock:
            return any(
                n.state == NodeState.ABANDONED for n in self._nodes.values()
            )

    def counts_by_state(self) -> dict[str, int]:
        with self._lock:
            out: dict[str, int] = {}
            for n in self._nodes.values():
                out[n.state.value] = out.get(n.state.value, 0) + 1
            return out

    def snapshot(self) -> GraphSnapshot:
        with self._lock:
            return GraphSnapshot(
                nodes=dict(self._nodes),
                edges=dict(self._edges),
                ready=tuple(
                    nid for nid, n in self._nodes.items()
                    if n.state == NodeState.READY
                ),
                counts=self.counts_by_state(),
            )

    # -- internal ------------------------------------------------------------

    def _require(self, node_id: str) -> NodeExecution:
        n = self._nodes.get(node_id)
        if n is None:
            raise UnknownNodeError(node_id)
        return n

    def _has_cycle(self) -> bool:
        # Iterative DFS; nodes have small fanout so this is fine.
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {nid: WHITE for nid in self._nodes}

        for start in self._nodes:
            if color[start] != WHITE:
                continue
            stack: list[tuple[str, int]] = [(start, 0)]
            while stack:
                node_id, idx = stack.pop()
                if idx == 0:
                    if color[node_id] == GRAY:
                        return True
                    if color[node_id] == BLACK:
                        continue
                    color[node_id] = GRAY
                deps = self._edges.get(node_id, ())
                if idx < len(deps):
                    stack.append((node_id, idx + 1))
                    dep = deps[idx]
                    if color.get(dep, WHITE) == GRAY:
                        return True
                    if color.get(dep, WHITE) == WHITE:
                        stack.append((dep, 0))
                else:
                    color[node_id] = BLACK
        return False

    def _descendants(self, node_id: str) -> set[str]:
        # Forward edges = inverse of self._edges. Compute the set of nodes
        # that have node_id transitively as a dependency.
        children: dict[str, set[str]] = {nid: set() for nid in self._nodes}
        for nid, deps in self._edges.items():
            for d in deps:
                if d in children:
                    children[d].add(nid)
        out: set[str] = set()
        stack = list(children.get(node_id, ()))
        while stack:
            cur = stack.pop()
            if cur in out:
                continue
            out.add(cur)
            stack.extend(children.get(cur, ()))
        return out


__all__ = ["CycleError", "DuplicateNodeError", "GraphSnapshot",
           "UnknownNodeError", "WorkGraph"]
