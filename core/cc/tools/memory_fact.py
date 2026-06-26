from __future__ import annotations

from typing import Any

from ..memory import MemoryFact, MemoryRuntime
from .base import BaseTool, ToolCall, ToolResult, ToolSpec, ValidationResult
from .context import ToolUseContext


def _safe_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


class MemoryFactTool(BaseTool):
    def __init__(self, *, memory_runtime: MemoryRuntime) -> None:
        super().__init__(
            ToolSpec(
                name="memory_fact",
                description="Query or store structured facts in the configured memory provider.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "entity": {"type": "string"},
                        "as_of": {"type": "string"},
                        "direction": {"type": "string"},
                        "subject": {"type": "string"},
                        "predicate": {"type": "string"},
                        "object": {"type": "string"},
                        "valid_from": {"type": "string"},
                        "valid_to": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["action"],
                },
                is_read_only=False,
            )
        )
        self.memory_runtime = memory_runtime

    def is_enabled(self, ctx):
        del ctx
        return False

    def validate_input(self, arguments: dict[str, Any]) -> ValidationResult:
        action = str(arguments.get("action") or "")
        if action == "query":
            if not str(arguments.get("entity") or "").strip():
                return ValidationResult(ok=False, message="entity is required for query.")
            return ValidationResult(ok=True)
        if action == "store":
            for field in ("subject", "predicate", "object"):
                if not str(arguments.get(field) or "").strip():
                    return ValidationResult(ok=False, message=f"{field} is required for store.")
            return ValidationResult(ok=True)
        return ValidationResult(ok=False, message="action must be query or store.")

    def is_concurrency_safe(self, arguments: dict[str, Any]) -> bool:
        return str(arguments.get("action") or "") == "query"

    async def execute(self, tool_call: ToolCall, ctx: ToolUseContext) -> ToolResult:
        del ctx
        action = str(tool_call.arguments["action"])
        if action == "query":
            facts = self.memory_runtime.explicit_query_facts(
                entity=str(tool_call.arguments["entity"]),
                as_of=tool_call.arguments.get("as_of"),
                direction=str(tool_call.arguments.get("direction") or "both"),
            )
            content = "\n".join(
                f"{fact.subject} -> {fact.predicate} -> {fact.object}" for fact in facts
            ) or "No facts found."
            return ToolResult(
                tool_use_id=tool_call.tool_use_id,
                tool_name=tool_call.tool_name,
                success=True,
                content=content,
                data={
                    "facts": [
                        {
                            "subject": fact.subject,
                            "predicate": fact.predicate,
                            "object": fact.object,
                            "valid_from": fact.valid_from,
                            "valid_to": fact.valid_to,
                            "confidence": fact.confidence,
                        }
                        for fact in facts
                    ]
                },
            )
        result = self.memory_runtime.explicit_store_fact(
            MemoryFact(
                subject=str(tool_call.arguments["subject"]),
                predicate=str(tool_call.arguments["predicate"]),
                object=str(tool_call.arguments["object"]),
                valid_from=tool_call.arguments.get("valid_from"),
                valid_to=tool_call.arguments.get("valid_to"),
                confidence=_safe_float(tool_call.arguments.get("confidence"), 1.0),
            )
        )
        return ToolResult(
            tool_use_id=tool_call.tool_use_id,
            tool_name=tool_call.tool_name,
            success=result.success,
            content=result.message,
            data={"memory_id": result.memory_id, "stored": result.stored},
            error_code=None if result.success else "SC1003",
        )
