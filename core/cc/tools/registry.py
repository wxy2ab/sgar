from __future__ import annotations

from .base import BaseTool
from .context import ToolUseContext


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.spec.name] = tool

    def register_many(self, tools: list[BaseTool]) -> None:
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def names(self) -> tuple[str, ...]:
        """Return the sealed registry surface without requiring a runtime context."""

        return tuple(sorted(self._tools))

    def list_visible(self, ctx: ToolUseContext) -> list[BaseTool]:
        # The LLM-facing schema = enabled AND not hidden. ``is_enabled`` gates
        # both visibility and dispatch (a disabled tool is rejected at the
        # executor, TL1009); ``is_hidden`` gates visibility ONLY, so a hidden
        # legacy wire-name alias stays dispatchable by name while a newer
        # unified tool owns the visible surface.
        return [
            self._tools[name]
            for name in sorted(self._tools)
            if self._tools[name].is_enabled(ctx)
            and not self._tools[name].is_hidden(ctx)
        ]

    def export_model_schemas(self, ctx: ToolUseContext) -> list[dict]:
        return [tool.to_model_schema() for tool in self.list_visible(ctx)]
