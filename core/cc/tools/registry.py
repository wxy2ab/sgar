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

    def list_visible(self, ctx: ToolUseContext) -> list[BaseTool]:
        return [
            self._tools[name]
            for name in sorted(self._tools)
            if self._tools[name].is_enabled(ctx)
        ]

    def export_model_schemas(self, ctx: ToolUseContext) -> list[dict]:
        return [tool.to_model_schema() for tool in self.list_visible(ctx)]
