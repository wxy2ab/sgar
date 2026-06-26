from __future__ import annotations

import json
from typing import Any

from ..tools.base import ToolResult
from .prompt_catalog import PromptCatalog


def implementation_followup_instruction(*, prompt_catalog: PromptCatalog, prompt_language: str) -> str:
    try:
        return prompt_catalog.resolve("system.implementation_followup", prompt_language)
    except Exception:
        return "请继续实现剩余的任务。不要输出总结文字，直接调用工具。"


def implementation_task_sync_instruction(*, prompt_catalog: PromptCatalog, prompt_language: str) -> str:
    try:
        return prompt_catalog.resolve("system.implementation_task_sync_followup", prompt_language)
    except Exception:
        return "你已完成了代码修改，请检查任务列表并更新已完成任务的状态。"


def agent_mode_incomplete_instruction(*, prompt_catalog: PromptCatalog, prompt_language: str) -> str:
    try:
        return prompt_catalog.resolve("system.agent_mode_followup", prompt_language)
    except Exception:
        return (
            "当前处于 agent 模式，但你还没有完成任何子代理协作。\n"
            "请先调用 agent 工具，把任务委派给至少一个子代理，再继续综合结论。\n"
            "不要直接输出最终答案。"
        )


def serialize_follow_up_prompt(
    *,
    user_input: str,
    assistant_response: dict[str, Any],
    tool_results: list[ToolResult],
    prompt_catalog: PromptCatalog,
    prompt_language: str,
    instruction_override: str | None = None,
) -> str:
    if instruction_override:
        instruction = instruction_override
    else:
        try:
            instruction = prompt_catalog.resolve("system.tool_followup", prompt_language)
        except Exception:
            instruction = "请根据工具执行结果继续完成任务。"
    payload = {
        "instruction": instruction,
        "user_input": user_input,
        "assistant_response": {
            "content": assistant_response.get("content", ""),
            "tool_call_count": len(assistant_response.get("tool_calls", [])),
        },
        "tool_results": [
            {
                "tool_use_id": result.tool_use_id,
                "tool_name": result.tool_name,
                "success": result.success,
                "content": result.content[:1000],
                "error_code": result.error_code,
                "data_summary": summarize_tool_data(result.data),
            }
            for result in tool_results
        ],
    }
    return json.dumps(payload, ensure_ascii=False)


def summarize_tool_data(data: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            summary[key] = value if not isinstance(value, str) else value[:200]
            continue
        if isinstance(value, list):
            summary[key] = {"type": "list", "count": len(value)}
            continue
        if isinstance(value, dict):
            summary[key] = {"type": "dict", "keys": sorted(value.keys(), key=str)[:10]}
            continue
        summary[key] = {"type": type(value).__name__}
    return summary


def build_continue_prompt(
    *,
    prompt_catalog: PromptCatalog,
    prompt_language: str,
    previous_user_text: str,
    partial_response: str,
) -> str:
    try:
        instruction = prompt_catalog.resolve("system.query_continue", prompt_language)
    except Exception:
        instruction = "Please continue your previous response."
    payload = {
        "instruction": instruction,
        "previous_user_text": previous_user_text,
        "partial_response": partial_response,
    }
    return json.dumps(payload, ensure_ascii=False)
