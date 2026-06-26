from __future__ import annotations

from core.utils.prompt_language import choose_prompt_text


def build_response_protocol(prompt_language: str, enabled_tools: list[str]) -> str:
    tool_list = ", ".join(enabled_tools) if enabled_tools else "(none)"
    return choose_prompt_text(
        prompt_language,
        zh=(
            "# 输出协议\n"
            "如果不需要调用工具，请直接输出自然语言，或输出 JSON：\n"
            '{"content":"给用户的答复","tool_calls":[]}\n'
            "如果需要调用工具，必须输出 JSON，对象格式如下：\n"
            '{"content":"你将做什么","tool_calls":[{"tool_name":"file_edit","arguments":{"file_path":"...","old_string":"...","new_string":"..."}}]}\n'
            f"当前可用工具: {tool_list}\n"
            "不要在 JSON 外层再包额外解释。"
        ),
        en=(
            "# Response Protocol\n"
            "If no tool is needed, respond in natural language or as JSON:\n"
            '{"content":"final answer for the user","tool_calls":[]}\n'
            "If a tool is needed, you must emit JSON in this shape:\n"
            '{"content":"what you are about to do","tool_calls":[{"tool_name":"file_edit","arguments":{"file_path":"...","old_string":"...","new_string":"..."}}]}\n'
            f"Currently available tools: {tool_list}\n"
            "Do not add extra explanation outside the JSON object."
        ),
    )
