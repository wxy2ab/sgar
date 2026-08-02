"""Incremental verification by dependency / path intersection (principle 7).

A local patch only invalidates checks whose declared ``scope`` intersects the
changed path set. Checks without a scope remain global (always re-run). At
merge / close points callers still pass ``changed_paths=None`` (or empty with
``force_full=True``) to run the full suite.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Iterable, Sequence


def normalize_scope(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        item = raw.strip().replace("\\", "/")
        return (item,) if item else ()
    if isinstance(raw, (list, tuple)):
        out: list[str] = []
        for v in raw:
            s = str(v).strip().replace("\\", "/")
            if s:
                out.append(s)
        return tuple(out)
    return ()


def paths_intersect(scope: Sequence[str], changed: Sequence[str]) -> bool:
    """True if any scope entry covers (or is covered by) a changed path."""
    if not scope:
        return True  # no scope ⇒ global
    if not changed:
        return True  # unknown change set ⇒ conservative full run
    for s in scope:
        sn = _norm(s)
        for c in changed:
            cn = _norm(c)
            if sn == cn or cn.startswith(sn.rstrip("/") + "/") or sn.startswith(
                cn.rstrip("/") + "/"
            ):
                return True
            # basename / glob-ish suffix match: "tests/" covers "tests/foo.py"
            if sn.endswith("/") and cn.startswith(sn):
                return True
    return False


def select_criteria_for_changes(
    criteria: Sequence[Any],
    changed_paths: Iterable[str] | None,
    *,
    force_full: bool = False,
) -> list[Any]:
    """Filter criteria to those intersecting ``changed_paths``.

    Criteria expose optional ``scope`` attribute (tuple/list of path prefixes).
    When ``force_full`` or ``changed_paths`` is None, return all criteria.
    """
    if force_full or changed_paths is None:
        return list(criteria)
    changed = [str(p) for p in changed_paths if p]
    if not changed:
        return list(criteria)
    selected: list[Any] = []
    for c in criteria:
        scope = normalize_scope(getattr(c, "scope", None))
        if paths_intersect(scope, changed):
            selected.append(c)
    # Safety: never verify nothing when there were criteria — fall back to full.
    return selected or list(criteria)


def _norm(path: str) -> str:
    text = path.strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    try:
        return str(PurePosixPath(text))
    except Exception:  # noqa: BLE001
        return text


__all__ = [
    "normalize_scope",
    "paths_intersect",
    "select_criteria_for_changes",
]
