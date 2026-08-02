"""Declared write-scope tool guard (principle 8).

When an agent stamps ``ccx_write_scope``, ``file_edit`` / ``file_write``
calls whose ``file_path`` falls outside that scope are rejected before the
underlying tool runs. Watch mode reuses the same guard with the historical
``FIX_SCOPE_VIOLATION`` error code.
"""

from __future__ import annotations

import os
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from .rwset import scopes_from_metadata

WRITE_SCOPE_VIOLATION = "WRITE_SCOPE_VIOLATION"
FIX_SCOPE_VIOLATION = "FIX_SCOPE_VIOLATION"

_GUARDED_TOOLS = ("file_edit", "file_write")


def path_matches_any_glob(path: str, globs: list[str], cwd: str) -> bool:
    """Match ``path`` against scope globs after normalizing under ``cwd``.

    Empty ``globs`` → never matches. Paths that escape ``cwd`` are rejected.
    Matching uses the normalized path so ``..`` cannot bypass the scope.
    """
    if not globs:
        return False
    p = Path(path)
    candidates: set[str] = set()
    if cwd:
        cwd_norm = Path(os.path.normpath(str(Path(cwd))))
        absolute = p if p.is_absolute() else cwd_norm / p
        norm = Path(os.path.normpath(str(absolute)))
        try:
            rel = norm.relative_to(cwd_norm)
        except ValueError:
            return False
        candidates.add(str(rel))
        candidates.add(rel.as_posix())
        candidates.add(str(norm))
        candidates.add(norm.as_posix())
    else:
        norm = Path(os.path.normpath(str(p)))
        if ".." in norm.parts:
            return False
        candidates.add(str(norm))
        candidates.add(norm.as_posix())
    for cand in candidates:
        for glob in globs:
            if fnmatch(cand, glob):
                return True
    return False


class WriteScopeGuard:
    """Composable guard wrapping a cc tool's ``execute`` method."""

    def __init__(
        self,
        wrapped: Any,
        *,
        write_scope: list[str],
        cwd: str,
        error_code: str = WRITE_SCOPE_VIOLATION,
        scope_label: str = "write_scope",
    ) -> None:
        self._wrapped = wrapped
        self._write_scope = list(write_scope)
        self._cwd = cwd
        self._error_code = error_code
        self._scope_label = scope_label

    @property
    def spec(self) -> Any:
        return self._wrapped.spec

    def is_enabled(self, ctx: Any) -> bool:
        return self._wrapped.is_enabled(ctx)

    def is_concurrency_safe(self, arguments: dict[str, Any]) -> bool:
        return False

    def validate_input(self, arguments: dict[str, Any]) -> Any:
        return self._wrapped.validate_input(arguments)

    def check_permissions(self, ctx: Any, arguments: dict[str, Any]) -> Any:
        return self._wrapped.check_permissions(ctx, arguments)

    def to_model_schema(self) -> dict[str, Any]:
        return self._wrapped.to_model_schema()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    async def execute(self, tool_call: Any, ctx: Any) -> Any:
        from core.cc.tools.base import ToolResult

        arguments = getattr(tool_call, "arguments", None) or {}
        path = str(arguments.get("file_path") or "")
        if not path:
            return ToolResult(
                tool_use_id=getattr(tool_call, "tool_use_id", ""),
                tool_name=getattr(tool_call, "tool_name", ""),
                success=False,
                content="file_path is required",
                error_code=self._error_code,
            )
        if not path_matches_any_glob(path, self._write_scope, self._cwd):
            return ToolResult(
                tool_use_id=getattr(tool_call, "tool_use_id", ""),
                tool_name=getattr(tool_call, "tool_name", ""),
                success=False,
                content=(
                    f"path {path!r} is outside {self._scope_label}. "
                    f"You may only edit files matching: {self._write_scope}. "
                    "Pick a different file or report in your final message "
                    "that the fix requires touching out-of-scope code."
                ),
                error_code=self._error_code,
            )
        return await self._wrapped.execute(tool_call, ctx)


def install_write_scope_guard(
    engine: Any,
    *,
    write_scope: list[str],
    cwd: str,
    error_code: str = WRITE_SCOPE_VIOLATION,
    scope_label: str = "write_scope",
    guard_cls: type | None = None,
) -> bool:
    """Wrap ``file_edit`` / ``file_write`` with :class:`WriteScopeGuard`.

    Idempotent: if a tool is already guarded, update scope in place.
    Returns True if at least one tool was wrapped or updated.
    ``guard_cls`` may be a subclass (e.g. watch's ``_FixScopeGuard``) so
    ``isinstance`` checks in callers stay valid.
    """
    if not write_scope:
        return False
    cls = guard_cls or WriteScopeGuard
    orchestrator = getattr(engine, "tool_orchestrator", None)
    registry = getattr(orchestrator, "registry", None)
    if registry is None:
        return False
    tools = getattr(registry, "_tools", None)
    if not isinstance(tools, dict):
        return False
    installed = False
    for name in _GUARDED_TOOLS:
        existing = tools.get(name)
        if existing is None:
            continue
        if isinstance(existing, WriteScopeGuard):
            existing._write_scope = list(write_scope)
            existing._cwd = cwd
            existing._error_code = error_code
            existing._scope_label = scope_label
            installed = True
            continue
        tools[name] = cls(
            existing,
            write_scope=write_scope,
            cwd=cwd,
            error_code=error_code,
            scope_label=scope_label,
        )
        installed = True
    return installed


def write_scope_from_metadata(metadata: Any) -> list[str]:
    """Extract non-empty ``ccx_write_scope`` paths from invocation metadata."""
    return sorted(scopes_from_metadata(metadata).writes)


__all__ = [
    "FIX_SCOPE_VIOLATION",
    "WRITE_SCOPE_VIOLATION",
    "WriteScopeGuard",
    "install_write_scope_guard",
    "path_matches_any_glob",
    "write_scope_from_metadata",
]
