"""Shared read-only enforcement primitive for ccx subagent runners.

ccx_research was the first runner to need a hard tool whitelist (the LLM
must not be able to call ``file_edit`` / ``shell`` even if the cc
registry exposes them). doc / ask modes have the same need. This module
holds the single source of truth so all three import the same whitelist
and enforcement function instead of copy-pasting.

Two-layer policy:

1. Explicit allowlist by tool name (``DEFAULT_READ_ONLY_WHITELIST``).
2. Fallback to ``tool.spec.is_read_only`` (every cc ``ToolSpec``
   self-reports — see ``core/cc/tools/base.py``). A direct
   ``tool.is_read_only`` attribute is also honored for test fakes that
   don't bother wrapping in a spec object.

A tool survives the filter if either layer accepts it. Any other tool is
deleted from the registry in place.

Tool names must match the cc registry — i.e. the value passed to
``ToolSpec(name=...)`` in each tool module under ``core/cc/tools/``,
NOT the Python class name. cc registers `glob` / `grep` / `file_read`
(lower-case, no ``_tool`` suffix); spelling them ``glob_tool`` /
``grep_tool`` here is a silent miss because the registry keys won't
match. Adding a regression test that builds a real cc engine catches
this.
"""

from __future__ import annotations

from typing import Any, Iterable


# Names match the actual cc ToolSpec.name values (see
# core/cc/tools/{glob_tool,grep_tool,file_read,memory_*}.py). Real
# spec.is_read_only would also keep these, but the explicit list keeps
# behaviour stable if someone flips a flag accidentally.
DEFAULT_READ_ONLY_WHITELIST: frozenset[str] = frozenset({
    "file_read",
    "glob",
    "grep",
    # The unified ``memory`` tool exposes action-based dispatch (status,
    # search, store, fact_query, fact_store). It is kept ONLY behind a
    # write-action guard: ``restrict_tool_registry`` wraps it in
    # ``_ReadOnlyMemoryGuard`` so ``store`` / ``fact_store`` are rejected
    # at the tool layer. Prompt text alone is NOT the enforcement
    # mechanism — this module's contract is a hard whitelist.
    "memory",
    # Legacy aliases — kept whitelisted so callers / tests that look
    # them up by name still resolve. The unified ``memory`` tool is the
    # canonical LLM-facing entry point; these are hidden from the model
    # schema via ``is_enabled=False``. Both are genuinely read-only.
    "memory_search",
    "memory_status",
})

# Read actions of the unified ``memory`` tool (mirrors
# ``core/cc/tools/memory.py::_READ_ACTIONS``).
_MEMORY_READ_ACTIONS: frozenset[str] = frozenset({
    "status", "search", "fact_query",
})


class _ReadOnlyMemoryGuard:
    """Proxy around the unified ``memory`` tool that hard-rejects write
    actions (``store`` / ``fact_store``) at both the validation and the
    execution layer. Everything else delegates to the wrapped tool."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def _rejects(self, arguments: dict[str, Any]) -> bool:
        return str(arguments.get("action") or "") not in _MEMORY_READ_ACTIONS

    def validate_input(self, arguments: dict[str, Any]) -> Any:
        if self._rejects(arguments):
            from core.cc.tools.base import ValidationResult
            return ValidationResult(
                ok=False,
                message=(
                    "memory write actions are disabled in read-only mode; "
                    f"allowed actions: {sorted(_MEMORY_READ_ACTIONS)}"
                ),
            )
        return self._inner.validate_input(arguments)

    async def execute(self, tool_call: Any, ctx: Any) -> Any:
        if self._rejects(dict(getattr(tool_call, "arguments", {}) or {})):
            from core.cc.tools.base import ToolResult
            return ToolResult(
                tool_use_id=getattr(tool_call, "tool_use_id", ""),
                tool_name=getattr(tool_call, "tool_name", "memory"),
                success=False,
                content=(
                    "memory write actions are disabled in read-only mode; "
                    f"allowed actions: {sorted(_MEMORY_READ_ACTIONS)}"
                ),
                error_code="read_only_violation",
            )
        return await self._inner.execute(tool_call, ctx)


def _is_read_only(tool: Any) -> bool:
    """Best-effort check for the read-only marker.

    Real cc tools store the flag at ``tool.spec.is_read_only``; test
    fakes may set it directly on the instance. Either is accepted.
    """
    spec = getattr(tool, "spec", None)
    if spec is not None:
        flag = getattr(spec, "is_read_only", None)
        if flag is not None:
            return bool(flag)
    return bool(getattr(tool, "is_read_only", False))


def restrict_tool_registry(
    engine: Any,
    *,
    extra_whitelist: Iterable[str] = (),
) -> tuple[int, list[str]]:
    """Drop any tool from a cc QueryEngine registry that is neither in
    the union whitelist nor self-reports ``is_read_only=True``.

    Mutates ``engine.tool_orchestrator.registry._tools`` in place.

    FAIL-CLOSED: raises ``RuntimeError`` when the registry shape is not
    recognized. This is a security filter — silently removing nothing
    (the old behaviour) would let a "read-only" research/ask turn run
    with ``file_edit`` / ``shell`` fully available after an innocuous
    cc-side refactor of the registry internals.

    Returns ``(removed_count, kept_names_sorted)`` so callers can log /
    assert on the post-restriction state.
    """
    orchestrator = getattr(engine, "tool_orchestrator", None)
    registry = getattr(orchestrator, "registry", None)
    if registry is None:
        raise RuntimeError(
            "restrict_tool_registry: engine has no tool_orchestrator.registry; "
            "refusing to run with read-only enforcement disabled"
        )
    tools = getattr(registry, "_tools", None)
    if tools is None or not isinstance(tools, dict):
        raise RuntimeError(
            "restrict_tool_registry: registry._tools is not a dict (cc "
            "internals changed?); refusing to run with read-only "
            "enforcement disabled"
        )

    union: set[str] = set(DEFAULT_READ_ONLY_WHITELIST)
    union.update(extra_whitelist)

    to_remove: list[str] = []
    for name, tool in tools.items():
        if name in union:
            continue
        if _is_read_only(tool):
            continue
        to_remove.append(name)
    for name in to_remove:
        del tools[name]
    # The unified memory tool is writable (store / fact_store); keeping
    # it raw would contradict the hard-whitelist contract. Wrap it so
    # write actions are rejected at the tool layer.
    mem = tools.get("memory")
    if mem is not None and not isinstance(mem, _ReadOnlyMemoryGuard):
        tools["memory"] = _ReadOnlyMemoryGuard(mem)
    return len(to_remove), sorted(tools.keys())


__all__ = [
    "DEFAULT_READ_ONLY_WHITELIST",
    "restrict_tool_registry",
]
