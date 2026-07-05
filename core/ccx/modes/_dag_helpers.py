"""Shared DAG-ordering helper for the plan / spec LLM parsers.

``_order_for_backward_deps`` is used by BOTH ``plan._parse_plan_response`` and
``spec._parse_spec_response`` (near-identical parsers). It lives here — a
leading-underscore ``modes/`` module, the shared-private namespace the
module-boundary contract sanctions — rather than being imported cross-sibling.

It reuses ``plan._topo_order`` (the same Kahn sort the explicit-DAG path uses)
via a function-scope import to avoid a ``plan`` <-> helper import cycle.
"""

from __future__ import annotations

from typing import Any


def _order_for_backward_deps(
    cleaned_items: list[dict[str, Any]], label: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Reorder parsed items so every ``depends_on`` edge points BACKWARD.

    The caller's remap drops dangling/self/dup indices but leaves FORWARD edges
    intact — and ``to_spawn_result`` then silently discards a forward index (its
    ``0 <= idx < i`` guard reads it as dangling), so items the planner meant to
    chain would run in parallel. The explicit-DAG path already avoids this via
    :func:`plan._topo_order`; reuse the very same sort here (keyed by position
    id) so the LLM path orders edges identically. Items with no explicit deps
    keep their original relative order (ties break by index), so a pure
    ``depends_on_previous`` chain is left untouched. A reorder is recorded as a
    ``dependency_issue``.
    """
    if not any(item["depends_on"] for item in cleaned_items):
        return cleaned_items, []
    from .plan import _topo_order  # lazy: avoid a plan <-> helper import cycle

    dag_nodes = [
        {
            "id": str(i),
            "goal": item["goal"],
            "depends_on": [str(d) for d in item["depends_on"]],
        }
        for i, item in enumerate(cleaned_items)
    ]
    ordered_nodes, issues = _topo_order(dag_nodes)
    final_order = [int(node["id"]) for node in ordered_nodes]
    if final_order == list(range(len(cleaned_items))):
        return cleaned_items, issues  # already backward-only; no reorder
    issues.append(
        f"{label} items reordered to satisfy forward depends_on edges "
        f"(new order of original positions: {final_order})"
    )
    old_to_final = {old: pos for pos, old in enumerate(final_order)}
    reordered = [
        {
            **cleaned_items[old],
            "depends_on": [old_to_final[d] for d in cleaned_items[old]["depends_on"]],
        }
        for old in final_order
    ]
    return reordered, issues


__all__ = ["_order_for_backward_deps"]
