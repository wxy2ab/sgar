"""Read/write set collection and conflict serialization (principle 8).

Sibling agents that declare overlapping write scopes (or a write that
intersects another's read/write set) must not run as free-parallel siblings:
conflicting patches are serialized by injecting ``ccx_depends_on`` edges.
Conflict-free patches keep their declared parallelism.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Iterable, Sequence

from .subagent import SubagentInvocation

WRITE_SCOPE_METADATA_KEY = "ccx_write_scope"
READ_SCOPE_METADATA_KEY = "ccx_read_scope"

_WRITE_TOOLS = frozenset({
    "file_write", "file_edit", "delete_file", "apply_patch", "Edit", "Write",
})
_READ_TOOLS = frozenset({
    "file_read", "Read", "grep", "Glob", "glob", "grep_tool",
})


@dataclass
class RwSet:
    reads: set[str] = field(default_factory=set)
    writes: set[str] = field(default_factory=set)

    def observe_tool(self, tool_name: str, arguments: dict[str, Any] | None) -> None:
        args = arguments or {}
        path = _extract_path(args)
        if not path:
            return
        name = str(tool_name or "")
        if name in _WRITE_TOOLS or name.endswith("_write") or name.endswith("_edit"):
            self.writes.add(path)
        elif name in _READ_TOOLS or name.endswith("_read"):
            self.reads.add(path)

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "read_set": sorted(self.reads),
            "write_set": sorted(self.writes),
        }


def scopes_from_metadata(metadata: Any) -> RwSet:
    """Build an RwSet from declared ``ccx_write_scope`` / ``ccx_read_scope``."""
    rw = RwSet()
    if not isinstance(metadata, dict):
        return rw
    for p in _as_path_list(metadata.get(WRITE_SCOPE_METADATA_KEY)):
        rw.writes.add(_normalize_path(p))
    for p in _as_path_list(metadata.get(READ_SCOPE_METADATA_KEY)):
        rw.reads.add(_normalize_path(p))
    return rw


def sets_conflict(a: RwSet, b: RwSet) -> bool:
    """True when two patches are unsafe to apply in either order / in parallel."""
    if a.writes & b.writes:
        return True
    if a.writes & b.reads:
        return True
    if b.writes & a.reads:
        return True
    return False


def rwset_from_extras(extras: Any) -> RwSet:
    """Rebuild an :class:`RwSet` from a node's recorded extras dict."""
    rw = RwSet()
    if not isinstance(extras, dict):
        return rw
    for p in extras.get("write_set") or []:
        if p:
            rw.writes.add(_normalize_path(str(p)))
    for p in extras.get("read_set") or []:
        if p:
            rw.reads.add(_normalize_path(str(p)))
    return rw


def _reachable_closure(
    node_ids: Sequence[str],
    direct_deps: dict[str, set[str]],
) -> dict[str, set[str]]:
    """Transitive closure of ``depends_on`` edges (``id → ancestors``)."""
    known = set(node_ids)
    memo: dict[str, set[str]] = {}

    def _walk(nid: str) -> set[str]:
        if nid in memo:
            return memo[nid]
        # Placeholder breaks cycles in malformed graphs.
        memo[nid] = set()
        reached: set[str] = set()
        for dep in direct_deps.get(nid, ()):
            if dep not in known:
                continue
            reached.add(dep)
            reached |= _walk(dep)
        memo[nid] = reached
        return reached

    for nid in node_ids:
        _walk(nid)
    return memo


def detect_parallel_observed_conflicts(
    nodes: Sequence[tuple[str, Any, Sequence[str]]],
) -> list[dict[str, Any]]:
    """Fail-loud detector for undeclared races among concurrent siblings.

    ``nodes`` is ``(node_id, extras, depends_on_ids)``. Two nodes conflict
    when their observed rwsets conflict AND neither depends on the other
    (transitively). Serialized pairs are allowed to touch the same paths.
    """
    items = [
        (str(nid), rwset_from_extras(extras), {str(d) for d in (deps or [])})
        for nid, extras, deps in nodes
    ]
    node_ids = [nid for nid, _, _ in items]
    direct = {nid: deps for nid, _, deps in items}
    ancestors = _reachable_closure(node_ids, direct)
    out: list[dict[str, Any]] = []
    for i in range(len(items)):
        id_a, rw_a, _deps_a = items[i]
        if not rw_a.writes and not rw_a.reads:
            continue
        for j in range(i + 1, len(items)):
            id_b, rw_b, _deps_b = items[j]
            if not rw_b.writes and not rw_b.reads:
                continue
            if id_b in ancestors[id_a] or id_a in ancestors[id_b]:
                continue  # serialized (possibly via a chain) — intentional
            if not sets_conflict(rw_a, rw_b):
                continue
            overlap = sorted(
                (rw_a.writes & rw_b.writes)
                | (rw_a.writes & rw_b.reads)
                | (rw_b.writes & rw_a.reads)
            )
            out.append({
                "node_a": id_a,
                "node_b": id_b,
                "paths": overlap,
            })
    return out


def detect_result_write_conflicts(
    results: Sequence[tuple[str, RwSet]],
) -> list[tuple[str, str]]:
    """Thin wrapper: conflict pairs with empty depends_on (no serialization).

    Prefer :func:`detect_parallel_observed_conflicts` when dependency edges
    are known.
    """
    nodes = [
        (str(nid), {"write_set": sorted(rw.writes), "read_set": sorted(rw.reads)}, ())
        for nid, rw in results
    ]
    return [
        (c["node_a"], c["node_b"])
        for c in detect_parallel_observed_conflicts(nodes)
    ]


def serialize_conflicting_invocations(
    invocations: Sequence[SubagentInvocation],
) -> tuple[list[SubagentInvocation], list[str]]:
    """Inject ``ccx_depends_on`` so conflicting siblings become a chain.

    Only considers *declared* scopes (``ccx_write_scope`` / ``ccx_read_scope``).
    Invocations without scopes are left untouched (unknown ⇒ assume no
    conflict a priori). Returns ``(new_invocations, issues)``.
    """
    if len(invocations) < 2:
        return list(invocations), []

    scopes = [scopes_from_metadata(inv.metadata) for inv in invocations]
    # Build conflict edges i -> j (j must wait for i) for i < j when they conflict.
    wait_for: dict[int, set[int]] = {i: set() for i in range(len(invocations))}
    issues: list[str] = []
    for i in range(len(invocations)):
        if not scopes[i].writes and not scopes[i].reads:
            continue
        for j in range(i + 1, len(invocations)):
            if not scopes[j].writes and not scopes[j].reads:
                continue
            if sets_conflict(scopes[i], scopes[j]):
                wait_for[j].add(i)
                issues.append(
                    f"subtask {j} serialized after {i}: write/read scope conflict "
                    f"(writes={sorted(scopes[i].writes | scopes[j].writes)})"
                )

    out: list[SubagentInvocation] = []
    for i, inv in enumerate(invocations):
        meta = dict(inv.metadata or {})
        preds = sorted(wait_for[i])
        if preds:
            existing = meta.get("ccx_depends_on")
            merged: list[int] = []
            if isinstance(existing, (list, tuple)):
                for v in existing:
                    try:
                        merged.append(int(v))
                    except (TypeError, ValueError):
                        pass
            for p in preds:
                if p not in merged:
                    merged.append(p)
            meta["ccx_depends_on"] = merged
            # Prefer explicit depends_on over the boolean previous flag.
            meta.pop("ccx_depends_on_previous", None)
        out.append(SubagentInvocation(
            goal=inv.goal,
            mode=inv.mode,
            metadata=meta,
            requires_approval=inv.requires_approval,
            max_attempts=inv.max_attempts,
            timeout_s=inv.timeout_s,
            preferred_model=inv.preferred_model,
        ))
    return out, issues


def merge_rwset_into_extras(extras: dict[str, Any], rw: RwSet) -> dict[str, Any]:
    out = dict(extras)
    out.update(rw.to_dict())
    return out


def _extract_path(arguments: dict[str, Any]) -> str | None:
    for key in ("file_path", "path", "file", "target"):
        val = arguments.get(key)
        if isinstance(val, str) and val.strip():
            return _normalize_path(val)
    paths = arguments.get("paths")
    if isinstance(paths, list) and paths:
        first = paths[0]
        if isinstance(first, str) and first.strip():
            return _normalize_path(first)
    return None


def _as_path_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw] if raw.strip() else []
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw if str(x).strip()]
    return []


def _normalize_path(path: str) -> str:
    text = path.strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    try:
        return str(PurePosixPath(text))
    except Exception:  # noqa: BLE001
        return text


__all__ = [
    "READ_SCOPE_METADATA_KEY",
    "WRITE_SCOPE_METADATA_KEY",
    "RwSet",
    "detect_parallel_observed_conflicts",
    "detect_result_write_conflicts",
    "merge_rwset_into_extras",
    "rwset_from_extras",
    "scopes_from_metadata",
    "serialize_conflicting_invocations",
    "sets_conflict",
]
