"""Inline tool-use ledger for messages-mode tool sessions.

The ``run_single_turn`` loop in :mod:`query_loop` accumulates all
assistant ``tool_calls`` and tool results into ``conversation_messages``,
but a long multi-round session can lead a reasoning model to "forget"
what it has already done — leading to redundant ``file_read`` / ``grep``
/ ``glob`` calls. These helpers extract a deduplicated ledger from the
current message history so the runner can fold a "don't repeat these"
reminder into the next system prompt.

The helpers are deliberately scoped to the three read-only enumeration
tools (``_LEDGER_TOOLS``). Re-issuing the same ``file_write`` /
``edit`` / agent invocation is intentional in most flows; only the
read tools have the "same call, same result" property that makes a
repeat-call ledger meaningful.
"""

from __future__ import annotations

import json
from typing import Any


_LEDGER_TOOLS = ("file_read", "grep", "glob")
_DEFAULT_INJECT_THRESHOLD: int = 5
_DEFAULT_LEDGER_LIMIT: int = 40


def _key_arg(tool: str, arguments: dict[str, Any]) -> str:
    if tool == "file_read":
        return str(arguments.get("file_path") or "")
    if tool in ("grep", "glob"):
        return str(arguments.get("pattern") or "")
    return ""


def _normalize_args(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        return {}
    if tool == "file_read":
        mb = arguments.get("max_bytes")
        if isinstance(mb, int) and mb > 0:
            return {"max_bytes": mb}
        return {}
    if tool == "grep":
        out: dict[str, Any] = {}
        for k in ("cwd", "glob", "file_type", "files_only", "context_lines", "max_results"):
            v = arguments.get(k)
            if v in (None, "", 0, False):
                continue
            out[k] = v
        return out
    if tool == "glob":
        out = {}
        for k in ("cwd", "max_results"):
            v = arguments.get(k)
            if v in (None, "", 0):
                continue
            out[k] = v
        return out
    return {}


def _parse_arguments(raw: Any) -> dict[str, Any]:
    """``tool_calls[i]['function']['arguments']`` is a JSON string in
    OpenAI format. Parse it back to a dict; return ``{}`` on malformed
    input (rather than raising) so a stray malformed call doesn't kill
    the ledger build."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def extract_ledger_from_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Walk OpenAI-format conversation messages and return a deduped
    ledger of ``file_read`` / ``grep`` / ``glob`` calls seen in
    assistant ``tool_calls``.

    Each entry: ``{"tool": str, "key": str, "args": dict, "count": int}``.
    Same ``(tool, key, args)`` collapses to one entry with ``count``
    incremented."""
    ledger: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue
        tcs = msg.get("tool_calls")
        if not isinstance(tcs, list):
            continue
        for tc in tcs:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            name = str(fn.get("name") or "")
            if name not in _LEDGER_TOOLS:
                continue
            args = _parse_arguments(fn.get("arguments"))
            key = _key_arg(name, args)
            norm = _normalize_args(name, args)
            for entry in ledger:
                if entry["tool"] == name and entry["key"] == key and entry["args"] == norm:
                    entry["count"] += 1
                    break
            else:
                ledger.append({"tool": name, "key": key, "args": norm, "count": 1})
    return ledger


def format_inline_reminder(
    ledger: list[dict[str, Any]],
    *,
    language: str = "en",
    threshold: int = _DEFAULT_INJECT_THRESHOLD,
    limit: int = _DEFAULT_LEDGER_LIMIT,
) -> str:
    """Render a "don't repeat these" reminder for inclusion in the next
    system prompt.

    Returns ``""`` when the ledger is too short to be worth reminding
    about (``threshold`` total calls, AND no duplicates yet). Callers
    can cheap-skip injection on an empty string and avoid prompt
    churn during short sessions.
    """
    if not ledger:
        return ""
    total_calls = sum(int(e.get("count", 1)) for e in ledger)
    has_dup = any(int(e.get("count", 1)) > 1 for e in ledger)
    if total_calls < threshold and not has_dup:
        return ""
    is_zh = language.startswith("zh")
    ranked = sorted(ledger, key=lambda e: -int(e.get("count", 1)))
    shown = ranked[:limit]
    lines: list[str] = []
    for e in shown:
        args = e.get("args") or {}
        args_bits: list[str] = []
        for k in sorted(args):
            v = args[k]
            args_bits.append(f"{k}={v!r}" if isinstance(v, str) else f"{k}={v}")
        args_str = (" " + " ".join(args_bits)) if args_bits else ""
        count = int(e.get("count", 1))
        if count > 1:
            tail = (
                f"（已调用 {count} 次，重复 {count - 1} 次）"
                if is_zh
                else f" (already called {count} times — {count - 1} redundant)"
            )
        else:
            tail = ""
        lines.append(f"- `{e['tool']}` {e['key']}{args_str}{tail}")
    body = "\n".join(lines)
    if is_zh:
        return (
            "## 本轮已发起的工具调用台账（不要再发起重复的）\n\n"
            "下面这些 (tool, key, args) 组合**这一轮已经调用过**。\n"
            "**不要再发起完全相同的调用**——结果就在上面的 conversation "
            "messages 里。需要新信息时换 path / 换 pattern / 加大 \n"
            "``max_bytes``；不需要更多信息就直接 emit 最终输出（JSON / "
            "Markdown / 文本）。重复调用算失败。\n\n" + body
        )
    return (
        "## Tool calls already made this turn (do NOT repeat)\n\n"
        "The (tool, key, args) combinations below have ALREADY been "
        "issued this turn. Their results are in the conversation "
        "messages above. **Do NOT issue the same combination again**. "
        "Need new information? Change the path / pattern, or raise "
        "``max_bytes``. Don't need more? Emit the final answer NOW "
        "(JSON / Markdown / text). Repeats are scored as failure.\n\n"
        + body
    )
