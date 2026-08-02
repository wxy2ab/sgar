"""Patch-first write policy for repair / iteration turns (principle 1).

When enabled, ``file_write`` may only *create* missing files. Overwriting an
existing file requires an explicit ``ccx_rewrite_allowed`` stamp (or the
guard is not installed). Prefer ``file_edit`` for sparse diffs.

``file_edit`` with an empty / missing ``old_string`` on an existing file is
also refused — that path is a full-file rewrite via the edit tool and must
not bypass the sparse-patch discipline.

Opt-in per invocation via metadata ``ccx_patch_first=True``. Governed repair
redrives stamp this automatically. Opt-out of the rewrite ban with
``ccx_rewrite_allowed=True``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from core.cc.safety.file_rules import resolve_under_cwd
from core.cc.tools.base import ToolCall, ToolResult, ValidationResult
from core.cc.tools.context import ToolUseContext

logger = logging.getLogger(__name__)

PATCH_FIRST_METADATA_KEY = "ccx_patch_first"
REWRITE_ALLOWED_METADATA_KEY = "ccx_rewrite_allowed"

PATCH_FIRST_SYSTEM_HINT_EN = (
    "Patch-first discipline (mandatory for this turn):\n"
    "- Prefer ``file_edit`` for a single sparse change to an existing file.\n"
    "- Prefer ``file_edit_batch`` when applying multiple exact-match hunks to "
    "the same file (one checkpoint, atomic commit; avoids half-applied edits).\n"
    "- ``file_write`` may create NEW files only; overwriting an existing "
    "file is refused unless rewrite was explicitly authorized.\n"
    "- ``file_edit`` on an existing file REQUIRES a non-empty ``old_string`` "
    "(sparse patch). Empty ``old_string`` is a full rewrite and is refused.\n"
    "- Keep diffs minimal; do not regenerate unchanged regions.\n"
    "- Full rewrites must be isolated, diffed, and regression-checked. "
    "Do not treat patch as always better than rewrite."
)

PATCH_FIRST_SYSTEM_HINT_ZH = (
    "本轮强制 patch-first：\n"
    "- 单处稀疏修改已有文件请用 ``file_edit``。\n"
    "- 同一文件多处 exact-match 修改请优先 ``file_edit_batch``"
    "（单次 checkpoint、原子提交，避免半改状态）。\n"
    "- ``file_write`` 仅允许创建尚不存在的新文件；覆盖已有文件会被拒绝"
    "（除非已显式授权 rewrite）。\n"
    "- 对已有文件的 ``file_edit`` 必须提供非空 ``old_string``（稀疏补丁）；"
    "空 ``old_string`` 等同整文件重写，会被拒绝。\n"
    "- 保持最小 diff，不要重发未改区域。\n"
    "- 完整重写必须隔离执行，并做 diff 与回归；"
    "不要把 patch 当成永远优于 rewrite 的硬规则。"
)


def patch_first_enabled(metadata: Any) -> bool:
    if not isinstance(metadata, dict):
        return False
    return bool(metadata.get(PATCH_FIRST_METADATA_KEY))


def rewrite_allowed(metadata: Any) -> bool:
    if not isinstance(metadata, dict):
        return False
    return bool(metadata.get(REWRITE_ALLOWED_METADATA_KEY))


def stamp_patch_first(
    metadata: dict[str, Any] | None,
    *,
    enabled: bool = True,
    allow_rewrite: bool = False,
) -> dict[str, Any]:
    """Return a copy of ``metadata`` with patch-first flags set."""
    out = dict(metadata or {})
    if enabled:
        out[PATCH_FIRST_METADATA_KEY] = True
    else:
        out.pop(PATCH_FIRST_METADATA_KEY, None)
    if allow_rewrite:
        out[REWRITE_ALLOWED_METADATA_KEY] = True
    else:
        out.pop(REWRITE_ALLOWED_METADATA_KEY, None)
    return out


def patch_first_hint(*, language: str = "en") -> str:
    lang = (language or "en").lower()
    if lang.startswith("zh"):
        return PATCH_FIRST_SYSTEM_HINT_ZH
    return PATCH_FIRST_SYSTEM_HINT_EN


def append_patch_first_hint(
    system_prompt: str,
    *,
    language: str = "en",
) -> str:
    """Append the patch-first hint unless the prompt already covers it."""
    hint = patch_first_hint(language=language)
    lowered = system_prompt.lower()
    if "patch-first" in lowered or "patch first" in lowered:
        return system_prompt
    if not system_prompt.strip():
        return hint
    return f"{system_prompt.rstrip()}\n\n{hint}"


class _PatchFirstWriteGuard:
    """Refuse ``file_write`` when the target path already exists."""

    def __init__(self, wrapped: Any) -> None:
        self._wrapped = wrapped
        self.spec = getattr(wrapped, "spec", None)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    def validate_input(self, arguments: dict[str, Any]) -> ValidationResult:
        return self._wrapped.validate_input(arguments)

    def check_permissions(self, ctx: ToolUseContext, arguments: dict[str, Any]):
        return self._wrapped.check_permissions(ctx, arguments)

    async def execute(self, tool_call: ToolCall, ctx: ToolUseContext) -> ToolResult:
        file_path = str(tool_call.arguments.get("file_path") or "")
        try:
            resolved = resolve_under_cwd(file_path, ctx.cwd)
        except Exception as exc:  # noqa: BLE001 — surface as tool failure
            return ToolResult(
                tool_use_id=tool_call.tool_use_id,
                tool_name=tool_call.tool_name,
                success=False,
                content=(
                    f"file_write refused under patch-first policy: "
                    f"could not resolve path ({exc})."
                ),
                error_code="PATCH_FIRST_PATH_ERROR",
            )
        if Path(resolved).exists():
            return ToolResult(
                tool_use_id=tool_call.tool_use_id,
                tool_name=tool_call.tool_name,
                success=False,
                content=(
                    "file_write refused under patch-first policy: "
                    f"{file_path!r} already exists. Use file_edit for a "
                    "minimal patch, or request rewrite authorization "
                    f"({REWRITE_ALLOWED_METADATA_KEY}=true)."
                ),
                error_code="PATCH_FIRST_REWRITE_REFUSED",
            )
        return await self._wrapped.execute(tool_call, ctx)


class _PatchFirstEditGuard:
    """Refuse ``file_edit`` full-file rewrites (empty ``old_string`` on existing)."""

    def __init__(self, wrapped: Any) -> None:
        self._wrapped = wrapped
        self.spec = getattr(wrapped, "spec", None)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    def is_concurrency_safe(self, arguments: dict[str, Any]) -> bool:
        return False

    def validate_input(self, arguments: dict[str, Any]) -> ValidationResult:
        return self._wrapped.validate_input(arguments)

    def check_permissions(self, ctx: ToolUseContext, arguments: dict[str, Any]):
        return self._wrapped.check_permissions(ctx, arguments)

    async def execute(self, tool_call: ToolCall, ctx: ToolUseContext) -> ToolResult:
        args = tool_call.arguments or {}
        file_path = str(args.get("file_path") or "")
        old_string = args.get("old_string")
        old_empty = old_string is None or str(old_string) == ""
        if not old_empty:
            return await self._wrapped.execute(tool_call, ctx)
        try:
            resolved = resolve_under_cwd(file_path, ctx.cwd)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                tool_use_id=tool_call.tool_use_id,
                tool_name=tool_call.tool_name,
                success=False,
                content=(
                    f"file_edit refused under patch-first policy: "
                    f"could not resolve path ({exc})."
                ),
                error_code="PATCH_FIRST_PATH_ERROR",
            )
        if not Path(resolved).exists():
            # Creating via empty old_string + create_if_missing is fine.
            return await self._wrapped.execute(tool_call, ctx)
        return ToolResult(
            tool_use_id=tool_call.tool_use_id,
            tool_name=tool_call.tool_name,
            success=False,
            content=(
                "file_edit refused under patch-first policy: "
                f"{file_path!r} already exists and ``old_string`` is empty "
                "(full-file rewrite). Provide a non-empty ``old_string`` for "
                "a sparse patch, or request rewrite authorization "
                f"({REWRITE_ALLOWED_METADATA_KEY}=true)."
            ),
            error_code="PATCH_FIRST_REWRITE_REFUSED",
        )


def install_patch_first_guard(engine: Any, *, allow_rewrite: bool = False) -> bool:
    """Wrap ``file_write`` + ``file_edit`` under patch-first. Returns True if any wrapped."""
    if allow_rewrite:
        return False
    orchestrator = getattr(engine, "tool_orchestrator", None)
    registry = getattr(orchestrator, "registry", None)
    if registry is None:
        return False
    tools = getattr(registry, "_tools", None)
    if not isinstance(tools, dict):
        return False
    installed = False
    write_tool = tools.get("file_write")
    if write_tool is not None:
        if isinstance(write_tool, _PatchFirstWriteGuard):
            installed = True
        else:
            tools["file_write"] = _PatchFirstWriteGuard(write_tool)
            installed = True
            logger.debug("ccx patch-first: wrapped file_write to refuse overwrites")
    edit_tool = tools.get("file_edit")
    if edit_tool is not None:
        if isinstance(edit_tool, _PatchFirstEditGuard):
            installed = True
        else:
            tools["file_edit"] = _PatchFirstEditGuard(edit_tool)
            installed = True
            logger.debug(
                "ccx patch-first: wrapped file_edit to refuse empty old_string"
            )
    return installed


__all__ = [
    "PATCH_FIRST_METADATA_KEY",
    "REWRITE_ALLOWED_METADATA_KEY",
    "_PatchFirstEditGuard",
    "_PatchFirstWriteGuard",
    "append_patch_first_hint",
    "install_patch_first_guard",
    "patch_first_enabled",
    "patch_first_hint",
    "rewrite_allowed",
    "stamp_patch_first",
]
