from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    is_read_only: bool = False
    needs_confirmation: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ValidationResult:
    ok: bool
    message: str | None = None


@dataclass(slots=True)
class ToolCall:
    tool_use_id: str
    tool_name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class ToolResult:
    tool_use_id: str
    tool_name: str
    success: bool
    content: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    error_code: str | None = None
    context_modifiers: list[Any] = field(default_factory=list)
    # Native truncation flag. True when ``content`` is only a prefix/window of
    # the tool's full output (file_read past max_bytes, grep/glob past their
    # caps). A typed mirror of the weakly-typed ``data["truncated"]`` key some
    # tools set, so downstream can rely on it without dict spelunking. Defaults
    # False, so every existing ToolResult is unchanged.
    truncated: bool = False


@dataclass(slots=True)
class ToolExecutionEvent:
    tool_use_id: str
    tool_name: str
    event_type: str
    success: bool | None = None
    error_code: str | None = None
    duration_ms: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)


class BaseTool:
    spec: ToolSpec

    def __init__(self, spec: ToolSpec) -> None:
        self.spec = spec

    def is_enabled(self, ctx: Any) -> bool:
        del ctx
        return True

    def is_concurrency_safe(self, arguments: dict[str, Any]) -> bool:
        del arguments
        return self.spec.is_read_only

    def validate_input(self, arguments: dict[str, Any]) -> ValidationResult:
        del arguments
        return ValidationResult(ok=True)

    def check_permissions(self, ctx: Any, arguments: dict[str, Any]) -> str:
        del ctx, arguments
        return "allow"

    async def execute(self, tool_call: ToolCall, ctx: Any) -> ToolResult:
        del ctx
        return ToolResult(
            tool_use_id=tool_call.tool_use_id,
            tool_name=tool_call.tool_name,
            success=True,
            content="",
        )

    def to_model_schema(self) -> dict[str, Any]:
        return {
            "name": self.spec.name,
            "description": self.spec.description,
            "input_schema": self.spec.input_schema,
        }
