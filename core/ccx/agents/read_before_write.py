"""Refuse ``file_edit`` when the target path was not read this turn.

Enabled only on governance turns (``ccx_patch_first`` or non-empty
``ccx_write_scope``). Tracks paths seen via successful ``file_read`` /
``grep`` match hits; does not treat ``glob`` listings as content-known.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from core.cc.safety.file_rules import resolve_under_cwd

logger = logging.getLogger(__name__)

WRITE_WITHOUT_READ = "WRITE_WITHOUT_READ"


def _norm_resolved(path: str, cwd: str) -> str | None:
    if not path or not str(path).strip():
        return None
    try:
        if cwd:
            resolved = resolve_under_cwd(str(path), cwd)
        else:
            resolved = os.path.normpath(str(path))
        return str(Path(resolved).resolve()) if cwd else str(Path(resolved))
    except Exception:  # noqa: BLE001
        try:
            return str(Path(os.path.normpath(str(path))))
        except Exception:  # noqa: BLE001
            return None


def _extract_grep_paths(result: Any, cwd: str) -> list[str]:
    data = getattr(result, "data", None) or {}
    if not isinstance(data, dict):
        return []
    out: list[str] = []
    matches = data.get("matches") or []
    if isinstance(matches, list):
        for item in matches:
            if isinstance(item, dict):
                fp = item.get("file_path")
                if fp:
                    out.append(str(fp))
    # files_only content is newline-separated paths — also accept when
    # matches carried only file_path keys (already handled above).
    return out


class _SeenPathBook:
    """Shared per-turn path set for the read-before-write wrappers."""

    __slots__ = ("paths", "cwd")

    def __init__(self, cwd: str) -> None:
        self.paths: set[str] = set()
        self.cwd = cwd

    def note(self, path: str) -> None:
        key = _norm_resolved(path, self.cwd)
        if key:
            self.paths.add(key)

    def seen(self, path: str) -> bool:
        key = _norm_resolved(path, self.cwd)
        return bool(key and key in self.paths)


class _ReadTracker:
    """Wrap ``file_read`` / ``grep`` to record successful path observations."""

    def __init__(self, wrapped: Any, book: _SeenPathBook) -> None:
        self._wrapped = wrapped
        self._book = book
        self.spec = getattr(wrapped, "spec", None)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    def is_enabled(self, ctx: Any) -> bool:
        return self._wrapped.is_enabled(ctx)

    def is_concurrency_safe(self, arguments: dict[str, Any]) -> bool:
        return self._wrapped.is_concurrency_safe(arguments)

    def validate_input(self, arguments: dict[str, Any]) -> Any:
        return self._wrapped.validate_input(arguments)

    def check_permissions(self, ctx: Any, arguments: dict[str, Any]) -> Any:
        return self._wrapped.check_permissions(ctx, arguments)

    def to_model_schema(self) -> dict[str, Any]:
        return self._wrapped.to_model_schema()

    async def execute(self, tool_call: Any, ctx: Any) -> Any:
        result = await self._wrapped.execute(tool_call, ctx)
        if not getattr(result, "success", False):
            return result
        name = str(
            getattr(tool_call, "tool_name", None)
            or getattr(result, "tool_name", "")
            or ""
        )
        args = getattr(tool_call, "arguments", None) or {}
        if name == "file_read" or name.endswith("_read"):
            fp = args.get("file_path") or args.get("path")
            if fp:
                self._book.note(str(fp))
        elif name in ("grep", "grep_tool") or name.endswith("grep"):
            for fp in _extract_grep_paths(result, self._book.cwd):
                self._book.note(fp)
        return result


class _EditReadGuard:
    """Refuse ``file_edit`` when the path was not observed this turn."""

    def __init__(self, wrapped: Any, book: _SeenPathBook) -> None:
        self._wrapped = wrapped
        self._book = book
        self.spec = getattr(wrapped, "spec", None)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

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

    async def execute(self, tool_call: Any, ctx: Any) -> Any:
        from core.cc.tools.base import ToolResult

        args = getattr(tool_call, "arguments", None) or {}
        file_path = str(args.get("file_path") or "")
        if not file_path:
            return ToolResult(
                tool_use_id=getattr(tool_call, "tool_use_id", ""),
                tool_name=getattr(tool_call, "tool_name", ""),
                success=False,
                content="file_path is required",
                error_code=WRITE_WITHOUT_READ,
            )
        # Exempt creates: no prior content to read.
        try:
            resolved = resolve_under_cwd(file_path, self._book.cwd or getattr(ctx, "cwd", ""))
            exists = Path(resolved).exists()
        except Exception:  # noqa: BLE001
            exists = False
        if not exists:
            return await self._wrapped.execute(tool_call, ctx)
        if self._book.seen(file_path):
            return await self._wrapped.execute(tool_call, ctx)
        return ToolResult(
            tool_use_id=getattr(tool_call, "tool_use_id", ""),
            tool_name=getattr(tool_call, "tool_name", ""),
            success=False,
            content=(
                f"file_edit refused: {file_path!r} was not file_read / "
                "grep-matched earlier this turn. Read the candidate first "
                "(write-without-read policy on governance turns)."
            ),
            error_code=WRITE_WITHOUT_READ,
        )


def install_read_before_write_guard(engine: Any, *, cwd: str) -> bool:
    """Install read-trackers + edit guard. Returns True if edit guard landed."""
    orchestrator = getattr(engine, "tool_orchestrator", None)
    registry = getattr(orchestrator, "registry", None)
    if registry is None:
        return False
    tools = getattr(registry, "_tools", None)
    if not isinstance(tools, dict):
        return False

    # Reuse book if already installed (idempotent re-install).
    book: _SeenPathBook | None = None
    for name in ("file_read", "grep", "file_edit"):
        existing = tools.get(name)
        if isinstance(existing, _ReadTracker):
            book = existing._book
            break
        if isinstance(existing, _EditReadGuard):
            book = existing._book
            break
    if book is None:
        book = _SeenPathBook(cwd=cwd or "")
    else:
        book.cwd = cwd or book.cwd

    installed_edit = False
    for name in ("file_read", "grep"):
        existing = tools.get(name)
        if existing is None:
            continue
        if isinstance(existing, _ReadTracker):
            existing._book = book
            continue
        # Unwrap one layer of unrelated guards if present by wrapping outside.
        tools[name] = _ReadTracker(existing, book)

    edit = tools.get("file_edit")
    if edit is not None:
        if isinstance(edit, _EditReadGuard):
            edit._book = book
            installed_edit = True
        else:
            tools["file_edit"] = _EditReadGuard(edit, book)
            installed_edit = True
            logger.debug("ccx read-before-write: wrapped file_edit")

    return installed_edit


__all__ = [
    "WRITE_WITHOUT_READ",
    "_EditReadGuard",
    "_ReadTracker",
    "install_read_before_write_guard",
]
