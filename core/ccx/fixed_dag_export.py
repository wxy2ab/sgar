"""P2 — distil a *draft* declarative DAG from observed runtime.db runs.

This reads what a run actually executed and hands back a `list[NodeSpec]` you
can seed a fixed DAG (P1 `run_spec`) from — so you don't hand-transcribe the
topology of a task the runtime has already run many times.

What it is NOT, and cannot be (read before using):

* **Not a runnable deterministic pipeline.** A persisted `ccx.agent` node's
  `params` is only ``{goal, metadata}`` — the SQL / template it *wrote at
  runtime* is nowhere in the DB. So the export gives you a **topology-frozen
  skeleton whose LLM nodes are still LLM nodes**; a human must rewrite each into
  a deterministic tool-node (real SQL / template) and review it. That authoring
  is where determinism comes from — the machine cannot supply it. Nodes needing
  this are flagged ``ccx_needs_authoring`` and surfaced in ``warnings``.
* **Not auto-trusted.** Every export is stamped ``ccx_draft=True`` and carries a
  hard expiry; it is an inert artifact until a person fleshes it out and wires
  it explicitly. There is no recipe table, no task→DAG matcher, no fuzzy
  dispatch — the caller passes exact run_ids and later passes exact specs.

Exported nodes carry **no execution state** — they are bare `NodeSpec`s, so they
can only ever start PENDING when re-run. We deliberately read ``row["spec"]``
directly (never `_reconstruct_graph` / `NodeExecution.from_dict`, which would
carry a SUCCEEDED state and launder a one-time pass into apparent authority).

Promotion guard: a single observation proves nothing about shape stability, so
export **refuses n<2 run_ids** and refuses runs whose topology is not identical
across every observation.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.deepstack_v5 import NodeSpec
from core.deepstack_v5.execution.node import _spec_from_dict, _spec_to_dict
from core.deepstack_v5.persistence import GraphStore, SQLiteRuntimeDB


logger = logging.getLogger(__name__)

#: Draft artifacts default to a 30-day shelf life. A stale draft distilled from
#: runs whose shape has since drifted is worse than none, so we expire loudly.
DEFAULT_DRAFT_TTL_MS = 30 * 24 * 60 * 60 * 1000

#: JSON envelope schema version. Bumped only on an incompatible shape change so
#: ``load_draft_dag`` can refuse a format it doesn't understand.
_ENVELOPE_VERSION = 1


class DraftDagError(RuntimeError):
    """Raised when an export is refused (too few / unstable observations)."""


@dataclass(slots=True)
class DraftDag:
    """A distilled draft DAG seed. Inert until a human authors + wires it.

    ``specs`` are bare `NodeSpec`s (no execution state). ``warnings`` lists the
    nodes that still need deterministic content authored before the DAG is
    anything more than a topology sketch.
    """

    specs: list[NodeSpec]
    created_from_run_ids: list[str]
    created_at_ms: int
    expires_at_ms: int
    draft: bool = True
    warnings: list[str] = field(default_factory=list)

    def is_expired(self, *, now_ms: int | None = None) -> bool:
        return _now_ms(now_ms) >= self.expires_at_ms

    @property
    def needs_authoring(self) -> list[str]:
        """Node ids whose deterministic content a human must still write."""
        return [
            s.node_id for s in self.specs
            if s.metadata.get("ccx_needs_authoring")
        ]


def _now_ms(now_ms: int | None) -> int:
    return int(time.time() * 1000) if now_ms is None else int(now_ms)


def _needs_authoring(tool: str) -> bool:
    """A ``ccx.*`` mode tool ran via an LLM — its persisted params hold only the
    goal, never the deterministic content it produced — so it must be re-authored
    as a real tool-node. A custom (non-``ccx.``) tool already carries its content
    (SQL / template) in params, so it is exportable as-is (still human-reviewed)."""
    return tool.startswith("ccx.")


def _canonical_shape(rows: list[dict[str, Any]]) -> tuple:
    """A node-id-independent signature of a run's topology.

    Agentic runs assign random node ids, so we label each node by
    ``(tool, goal)`` and express edges over those labels. Two runs share a shape
    iff they have the same multiset of node labels and the same set of labelled
    edges. This is a heuristic (two nodes with an identical ``(tool, goal)``
    collapse) — good enough as a *stability guard*, not a correctness oracle.
    """
    by_id = {r["node_id"]: r for r in rows}

    def label(r: dict[str, Any]) -> tuple[str, str]:
        spec = r["spec"]
        return (
            str(spec.get("tool", "")),
            str((spec.get("params") or {}).get("goal", "")),
        )

    nodes = sorted(label(r) for r in rows)
    edges = sorted(
        (label(by_id[dep]), label(r))
        for r in rows
        for dep in (r["spec"].get("depends_on") or [])
        if dep in by_id
    )
    return (tuple(nodes), tuple(edges))


def export_draft_dag(
    db_path: Path | str,
    run_ids: list[str],
    *,
    now_ms: int | None = None,
    ttl_ms: int = DEFAULT_DRAFT_TTL_MS,
) -> DraftDag:
    """Distil a draft DAG from ≥2 runs of the same task in ``db_path``.

    Reads ``row["spec"]`` only (no state), verifies the topology is identical
    across every ``run_ids`` observation, and returns a state-free `DraftDag`
    stamped DRAFT + provenance + expiry. Raises `DraftDagError` on n<2 or an
    unstable shape.
    """
    if len(run_ids) < 2:
        raise DraftDagError(
            "CCX_DRAFT_REFUSE_N1: export requires >=2 run_ids observed with a "
            f"stable shape (got {len(run_ids)}); one run proves nothing about "
            "whether the topology is actually stable."
        )

    db = SQLiteRuntimeDB(Path(db_path))
    try:
        store = GraphStore(db)
        per_run_rows: list[list[dict[str, Any]]] = []
        for run_id in run_ids:
            rows = store.list_nodes(run_id)
            if not rows:
                raise DraftDagError(
                    f"CCX_DRAFT_UNKNOWN_RUN: run_id {run_id!r} has no nodes in "
                    f"{db_path}"
                )
            per_run_rows.append(rows)
    finally:
        db.close()

    shapes = {_canonical_shape(rows) for rows in per_run_rows}
    if len(shapes) != 1:
        raise DraftDagError(
            "CCX_DRAFT_UNSTABLE_SHAPE: the observed runs do not share one "
            f"topology ({len(shapes)} distinct shapes across {len(run_ids)} "
            "runs); refusing to promote an unstable shape to a draft."
        )

    created = _now_ms(now_ms)
    specs: list[NodeSpec] = []
    warnings: list[str] = []
    # Distil from the first observation (all share a shape by the check above).
    for row in per_run_rows[0]:
        spec = _spec_from_dict(row["spec"])
        needs = _needs_authoring(spec.tool)
        # A fresh draft carries NONE of the original run's node metadata. That
        # bookkeeping (parent_node_id, cwd, session_id, sgar_session, spawn
        # depth, resume/memory blocks, …) is read at RE-RUN time and would bind
        # the seed to the dead run's directory / session / lineage. A denylist
        # would silently rot as new keys are added, so we keep only the DRAFT
        # stamp; ``created_from_run_id`` is the pointer back if a human needs
        # the rest.
        meta: dict[str, Any] = {
            "ccx_draft": True,
            "ccx_created_from_run_id": run_ids[0],
        }
        if needs:
            # An LLM node's deterministic content was never persisted, and its
            # ``params`` carry the same run-specific metadata blob (mode tools
            # read ``params["metadata"]``). Reduce it to the bare goal a human
            # reads while authoring the replacement tool-node.
            meta["ccx_needs_authoring"] = True
            goal = str((spec.params or {}).get("goal", ""))
            params: Mapping[str, Any] = {"goal": goal} if goal else {}
            warnings.append(
                f"node {spec.node_id!r} ran as LLM tool {spec.tool!r}; its "
                "deterministic content was NOT persisted — rewrite it as a "
                "tool-node with real SQL/template before use."
            )
        else:
            # A custom tool already carries its deterministic content (SQL /
            # template) in params — keep it verbatim; it IS the useful part (a
            # human still reviews any run-specific paths in it).
            params = dict(spec.params)
        specs.append(NodeSpec(
            node_id=spec.node_id,
            tool=spec.tool,
            params=dict(params),
            depends_on=tuple(spec.depends_on),
            max_attempts=spec.max_attempts,
            timeout_s=spec.timeout_s,
            requires_approval=spec.requires_approval,
            priority=spec.priority,
            metadata=meta,
        ))

    return DraftDag(
        specs=specs,
        created_from_run_ids=list(run_ids),
        created_at_ms=created,
        expires_at_ms=created + int(ttl_ms),
        warnings=warnings,
    )


# --------------------------------------------------------------------------- #
# JSON round-trip — a draft is a passive artifact a human edits, not a live
# object. The envelope makes the DRAFT flag + provenance + expiry impossible to
# miss when the file is opened.
# --------------------------------------------------------------------------- #

def draft_to_dict(draft: DraftDag) -> dict[str, Any]:
    return {
        "ccx_draft_dag": True,
        "version": _ENVELOPE_VERSION,
        "created_from_run_ids": list(draft.created_from_run_ids),
        "created_at_ms": draft.created_at_ms,
        "expires_at_ms": draft.expires_at_ms,
        "warnings": list(draft.warnings),
        "nodes": [_spec_to_dict(s) for s in draft.specs],
    }


def write_draft_dag(draft: DraftDag, path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(draft_to_dict(draft), indent=2), encoding="utf-8")
    return out


def load_draft_dag(path: Path | str) -> DraftDag:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not data.get("ccx_draft_dag"):
        raise DraftDagError(
            f"CCX_DRAFT_BAD_ENVELOPE: {path} is not a ccx draft-dag artifact"
        )
    version = data.get("version")
    if version != _ENVELOPE_VERSION:
        raise DraftDagError(
            f"CCX_DRAFT_BAD_VERSION: {path} has envelope version {version!r}, "
            f"this build understands {_ENVELOPE_VERSION}"
        )
    return DraftDag(
        specs=[_spec_from_dict(n) for n in data.get("nodes") or []],
        created_from_run_ids=list(data.get("created_from_run_ids") or []),
        created_at_ms=int(data.get("created_at_ms") or 0),
        expires_at_ms=int(data.get("expires_at_ms") or 0),
        warnings=list(data.get("warnings") or []),
    )


__all__ = [
    "DEFAULT_DRAFT_TTL_MS",
    "DraftDag",
    "DraftDagError",
    "draft_to_dict",
    "export_draft_dag",
    "load_draft_dag",
    "write_draft_dag",
]
