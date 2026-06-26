from __future__ import annotations

from collections.abc import AsyncIterator
import json
import logging
from typing import Any
import uuid

logger = logging.getLogger(__name__)

from ..command_runner import CheckVerdict, run_check_command_async
from ..tools.base import ToolCall, ToolResult
from ..tools.context import ToolUseContext
from ..tools.result_mapper import ToolResultMapper
from .context_assembler import ContextAssembler
from .llm_adapter import LLMAdapter
from .models import SessionEvent, SessionMessage, SystemPromptParts
from .prompt_builder import SystemPromptBuilder
from .prompt_catalog import PromptCatalog
from .query_loop_followup import (
    agent_mode_incomplete_instruction as _agent_mode_incomplete_instruction,
    build_continue_prompt as _build_continue_prompt,
    implementation_followup_instruction as _implementation_followup_instruction,
    implementation_task_sync_instruction as _implementation_task_sync_instruction,
    serialize_follow_up_prompt as _serialize_follow_up_prompt,
)
from .query_loop_implementation_sync import (
    MAX_CODE_ONLY_GRACE_ROUNDS as _MAX_CODE_ONLY_GRACE_ROUNDS,
    MAX_IMPLEMENTATION_STALL_ROUNDS as _MAX_IMPLEMENTATION_STALL_ROUNDS,
    auto_complete_tasks as _auto_complete_tasks,
    implementation_requires_task_sync as _implementation_requires_task_sync,
    implementation_round_made_progress as _implementation_round_made_progress,
    implementation_tasks_incomplete as _implementation_tasks_incomplete,
    implementation_tasks_snapshot as _implementation_tasks_snapshot,
)
from .query_loop_mode_transitions import (
    mode_exited as _mode_exited,
    plan_artifacts_incomplete as _plan_artifacts_incomplete,
    plan_incomplete_instruction as _plan_incomplete_instruction,
    should_auto_exit_plan as _should_auto_exit_plan,
    should_auto_exit_spec as _should_auto_exit_spec,
    spec_artifacts_incomplete as _spec_artifacts_incomplete,
    spec_incomplete_instruction as _spec_incomplete_instruction,
    todos_incomplete as _todos_incomplete,
    todos_incomplete_instruction as _todos_incomplete_instruction,
    todos_stall_rescue_instruction as _todos_stall_rescue_instruction,
)
from .query_loop_tool_events import (
    normalize_tool_call as _normalize_tool_call,
    run_additional_tool_calls as _run_additional_tool_calls,
)
from .session import QuerySession
from .tool_ledger import (
    extract_ledger_from_messages as _extract_ledger_from_messages,
    format_inline_reminder as _format_inline_ledger_reminder,
)


def _cc_schemas_to_openai_tools(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert CC tool schemas to OpenAI function-calling format."""
    tools = []
    for schema in schemas:
        tools.append({
            "type": "function",
            "function": {
                "name": schema.get("name", ""),
                "description": schema.get("description", ""),
                "parameters": schema.get("input_schema", {}),
            },
        })
    return tools


_DEFAULT_MAX_TOOL_ROUNDS: int | None = None
_MAX_INCOMPLETE_AGENT_REPROMPTS = 2
_MAX_INCOMPLETE_IMPLEMENTATION_REPROMPTS = 2
_MAX_INCOMPLETE_PLAN_REPROMPTS = 2
_MAX_INCOMPLETE_SPEC_REPROMPTS = 2
_MAX_INCOMPLETE_TODOS_REPROMPTS = 2
_EXIT_REASON_COMPLETED = "completed"
_EXIT_REASON_COMPLETED_AFTER_TOOLS = "completed_after_tools"
_EXIT_REASON_AGENT_REPROMPT_EXHAUSTED = "agent_collaboration_reprompt_exhausted"
_EXIT_REASON_PLAN_REPROMPT_EXHAUSTED = "plan_artifacts_reprompt_exhausted"
_EXIT_REASON_SPEC_REPROMPT_EXHAUSTED = "spec_artifacts_reprompt_exhausted"
_EXIT_REASON_IMPLEMENTATION_REPROMPT_EXHAUSTED = "implementation_reprompt_exhausted"
_EXIT_REASON_TODOS_REPROMPT_EXHAUSTED = "todos_reprompt_exhausted"
_EXIT_REASON_IMPLEMENTATION_STALLED = "implementation_stalled"
_EXIT_REASON_GENERIC_STALLED = "generic_stalled"
_EXIT_REASON_TOOL_ROUND_LIMIT = "tool_round_limit_reached"
_MAX_GENERIC_READ_ONLY_STALL_ROUNDS = 30

# Tools whose successful use means the round mutated source. Shared by the
# implementation grace/stall accounting and the opt-in post-edit verification
# step so both agree on what "this round changed code" means.
_CODE_MUTATION_TOOLS = frozenset({"file_write", "file_edit", "delete_file"})


async def _run_post_edit_verification(
    *, config: Any, cwd: str, collected_results: list[ToolResult]
) -> CheckVerdict | None:
    """Opt-in (default OFF): after a round that mutated code, run the configured
    verification command and return its exit-code-gated verdict.

    Returns ``None`` — no command run, behavior byte-equivalent to before — when
    the feature is off, no command is configured, or no code-mutation tool
    succeeded this round.
    """
    if not getattr(config, "auto_post_edit_verify", False):
        return None
    command = (getattr(config, "post_edit_verify_command", None) or "").strip()
    if not command:
        return None
    if not any(
        r.success and r.tool_name in _CODE_MUTATION_TOOLS for r in collected_results
    ):
        return None
    timeout_ms = int(getattr(config, "post_edit_verify_timeout_ms", 0) or 120_000)
    return await run_check_command_async(command=command, cwd=cwd, timeout_ms=timeout_ms)


def _post_edit_verdict_note(verdict: CheckVerdict) -> str:
    """Conversation-facing summary of a post-edit verification verdict."""
    if verdict.passed:
        return f"[post-edit verification] PASSED — {verdict.evidence_line()}"
    if verdict.unrunnable:
        return (
            "[post-edit verification] the configured verification command could "
            "NOT run (harness/config defect, not a code failure): "
            f"{verdict.evidence_line()}"
        )
    return (
        "[post-edit verification] FAILED — your edits did not pass the project's "
        "verification command. Fix the failures before finishing.\n"
        f"{verdict.evidence_line()}"
    )


def _post_edit_blocks_autocomplete(
    config: Any, latest_verify_passed: bool | None
) -> bool:
    """Whether to block implementation-task auto-completion this round.

    True only when post-edit verification is enabled AND the most recent verdict
    was RED (``latest_verify_passed is False``). When the feature is off,
    ``latest_verify_passed`` stays ``None`` and this is always False, so the
    auto-complete grace logic behaves byte-identically to before.
    """
    if not getattr(config, "auto_post_edit_verify", False):
        return False
    return latest_verify_passed is False


def _effective_tool_limit(base_limit: int, session: QuerySession) -> int:
    """Extend the tool round limit by 50% when there are active (incomplete) todos."""
    from .query_loop_mode_transitions import todos_incomplete as _check_todos
    if _check_todos(session):
        return int(base_limit * 1.5)
    return base_limit


_MAX_CONVERSATION_MESSAGES = 80
_CONVERSATION_KEEP_RECENT = 50
_TOOL_RESULT_CONTENT_LIMIT = 2000
# Content-delivery tools exist to put source/match content in front of the
# model for inspection. Re-truncating their result to 2000 chars (~30 lines)
# here is the root cause of doc/review investigators reading only a file's head
# and then (with no size signal) asserting "function missing / dead code". These
# tools already bound their own output (file_read at max_bytes, default 100KB,
# with its own end-marker; grep at max_results), so capping again to 2000 is a
# double-truncation that throws away ~98% of what the tool was asked to return.
# Honour the tool-level cap instead: read-tool results pass through up to the
# same 100KB file_read bounds them to (side-effect tools keep the small cap so
# their assembly stays byte-identical). The marker reports the real size so the
# model knows when a read is still partial — the old bare "[truncated]" hid
# that, so the model assumed it had seen the whole file.
_READ_TOOL_RESULT_CONTENT_LIMIT = 100_000
_CONTENT_DELIVERY_TOOLS = frozenset({"file_read", "grep", "glob"})
# Token-budget-aware compaction. When the LLM client advertises a context
# window (context_window_tokens), compaction triggers on the estimated prompt
# size approaching that window rather than a fixed message count — so 1M-context
# models (DeepSeek-V4) retain far more history AND rebuild the cache-stable
# prefix far less often. _COMPACT_TRIGGER_RATIO leaves headroom because this
# tool-loop path has no reactive context-length-400 fallback; the message-count
# ceiling is a backstop against pathological floods of tiny messages.
_COMPACT_TRIGGER_RATIO = 0.75
_HARD_MESSAGE_CEILING = 600
# Cap on messages kept after a token-budget compaction. Normally the char
# budget limits the tail first; this also bounds the count so a flood of tiny
# messages (which the char budget alone wouldn't trim) is actually reduced.
_COMPACT_KEEP_RECENT_MAX = 400


def _truncate_tool_result_content(result: ToolResult) -> str:
    """Cap a tool result's content for the conversation the model sees next.

    Content-delivery tools (file_read/grep/glob) keep up to the large read cap
    so investigators actually see the file they asked for; side-effect tools
    keep the small cap (byte-identical to the old behaviour). When truncation
    does happen the marker reports the real size so the model knows the read is
    partial — the previous bare "[truncated]" hid that.
    """
    limit = (
        _READ_TOOL_RESULT_CONTENT_LIMIT
        if result.tool_name in _CONTENT_DELIVERY_TOOLS
        else _TOOL_RESULT_CONTENT_LIMIT
    )
    content = result.content
    if len(content) <= limit:
        return content
    return (
        content[:limit]
        + f"\n[truncated: showing {limit} of {len(content)} chars; "
        "re-read with a larger max_bytes or a narrower scope for more]"
    )


def _estimate_messages_chars(messages: list[dict[str, Any]]) -> int:
    """Total characters across message contents + tool_calls. A character count
    is a safe upper bound on token count (a token spans >= 1 char), so using it
    as the budget estimate never under-counts into a surprise context-length
    400 — it only ever compacts a little early."""
    total = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            total += sum(len(str(part)) for part in content)
        elif content is not None:
            total += len(str(content))
        tool_calls = m.get("tool_calls")
        if tool_calls:
            total += len(str(tool_calls))
    return total


def _cc_tool_calls_to_openai(tool_calls: list[dict[str, Any] | Any]) -> list[dict[str, Any]]:
    """Convert CC-format tool_calls (dicts or ToolCall objects) to OpenAI function-calling format."""
    result = []
    for tc in tool_calls:
        if isinstance(tc, dict):
            tc_id = str(tc.get("tool_use_id") or tc.get("id") or "")
            tc_name = str(tc.get("tool_name") or tc.get("name") or "")
            args = tc.get("arguments", {})
        else:
            tc_id = str(getattr(tc, "tool_use_id", "") or "")
            tc_name = str(getattr(tc, "tool_name", "") or "")
            args = getattr(tc, "arguments", {})
        args_str = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args)
        result.append({
            "id": tc_id,
            "type": "function",
            "function": {
                "name": tc_name,
                "arguments": args_str,
            },
        })
    return result


def _compact_conversation_messages(
    messages: list[dict[str, Any]],
    *,
    token_budget: int | None = None,
) -> list[dict[str, Any]]:
    """Keep conversation messages within budget by trimming middle rounds.

    Token-budget mode (``token_budget`` set, the preferred path when the client
    advertises a context window): compact only as the estimated prompt
    approaches the window, keeping the system + first user turn and as many
    recent messages as fit in the budget. A high message-count ceiling guards
    against a flood of tiny messages. Falls back to the legacy message-count
    trim for clients that don't advertise a window.
    """
    if token_budget is not None and token_budget > 0:
        if (
            _estimate_messages_chars(messages) <= token_budget
            and len(messages) <= _HARD_MESSAGE_CEILING
        ):
            return messages
        head = messages[:2]
        # Keep the most recent messages that fit in ~70% of budget, leaving
        # headroom for the next few rounds before compaction triggers again.
        keep_budget = int(token_budget * 0.7)
        kept_reversed: list[dict[str, Any]] = []
        acc = 0
        for m in reversed(messages[2:]):
            if len(kept_reversed) >= _COMPACT_KEEP_RECENT_MAX:
                break
            cost = _estimate_messages_chars([m])
            if acc + cost > keep_budget and kept_reversed:
                break
            kept_reversed.append(m)
            acc += cost
        tail = list(reversed(kept_reversed))
        # Don't start the kept tail with an orphan tool result whose assistant
        # tool_calls turn was dropped — it would be discarded downstream anyway.
        while tail and tail[0].get("role") == "tool":
            tail.pop(0)
        if len(messages) - len(head) - len(tail) <= 0:
            return messages  # nothing actually dropped
        return head + [{"role": "user", "content": "[Earlier conversation rounds have been compacted.]"}] + tail

    if len(messages) <= _MAX_CONVERSATION_MESSAGES:
        return messages
    head = messages[:2]
    tail = messages[-_CONVERSATION_KEEP_RECENT:]
    return head + [{"role": "user", "content": "[Earlier conversation rounds have been compacted.]"}] + tail


async def run_single_turn(
    *,
    session: QuerySession,
    turn_id: str,
    user_input: str | list[dict[str, Any]],
    llm_adapter: LLMAdapter,
    prompt_parts: SystemPromptParts,
    tool_orchestrator: Any,
    tool_ctx: ToolUseContext,
    prompt_catalog: PromptCatalog,
    context_assembler: ContextAssembler,
    prompt_builder: SystemPromptBuilder | None = None,
    max_tool_rounds: int | None = _DEFAULT_MAX_TOOL_ROUNDS,
) -> AsyncIterator[SessionEvent]:
    content = user_input if isinstance(user_input, str) else str(user_input)
    user_message = SessionMessage(
        message_id=f"msg_{uuid.uuid4().hex[:10]}",
        turn_id=turn_id,
        role="user",
        content=content,
        kind="user_input",
    )
    yield SessionEvent(event_type="message_created", turn_id=turn_id, message=user_message)

    registry = getattr(tool_orchestrator, "registry", None)
    api_tools: list[dict[str, Any]] | None = None
    if registry is not None and hasattr(llm_adapter.llm_client, "tool_invoke"):
        api_tools = _cc_schemas_to_openai_tools(registry.export_model_schemas(tool_ctx))

    use_messages_mode = api_tools is not None
    conversation_messages: list[dict[str, Any]] = []
    if use_messages_mode:
        conversation_messages = [
            {"role": "system", "content": prompt_parts.combined},
            {"role": "user", "content": content},
        ]

    # Token-budget-aware compaction budget, derived from the client's advertised
    # context window (DeepSeek-V4 = 1,048,576) minus reserved output, scaled by a
    # safety ratio. None for clients that don't advertise a window → the legacy
    # message-count trim is used. The 0.5*window floor guards a pathological
    # max_tokens >= window config.
    _ctx_window = getattr(llm_adapter.llm_client, "context_window_tokens", None)
    _reserved_output = getattr(llm_adapter.llm_client, "max_tokens", 0) or 0
    compact_token_budget: int | None = None
    if isinstance(_ctx_window, int) and _ctx_window > 0:
        usable = max(_ctx_window - _reserved_output, int(_ctx_window * 0.5))
        compact_token_budget = int(usable * _COMPACT_TRIGGER_RATIO)

    tool_mapper = ToolResultMapper()
    current_prompt = content
    tool_round = 0
    # Once a mode transition (plan->implementation / spec->render) rebuilds the
    # system prompt, that prompt must persist across every later tool round of
    # this turn — the implementation/render phase spans many rounds (read plan,
    # read tasks, write code, sync tasks...). Without this, only the single
    # round immediately after the exit got the implementation prompt and the
    # next ``exited_mode is None`` rebuild reset it back to the generic default.
    active_prompt_key: str | None = None
    incomplete_agent_reprompts = 0
    incomplete_impl_reprompts = 0
    incomplete_plan_reprompts = 0
    incomplete_spec_reprompts = 0
    incomplete_todos_reprompts = 0
    previous_tasks_snapshot = _implementation_tasks_snapshot(session)
    implementation_stall_rounds = 0
    implementation_code_only_rounds = 0
    implementation_pending_task_sync = False
    generic_read_only_stall_rounds = 0
    generic_stall_reprompts = 0
    # Most recent opt-in post-edit verification verdict. None until a
    # verification actually runs (feature on + code mutated + command set); a
    # RED verdict here blocks implementation-task auto-completion (see
    # ``_post_edit_blocks_autocomplete``). Stays None when the feature is off.
    latest_post_edit_verify_passed: bool | None = None
    while max_tool_rounds is None or tool_round < _effective_tool_limit(max_tool_rounds, session):
        if use_messages_mode:
            conversation_messages = _compact_conversation_messages(
                conversation_messages, token_budget=compact_token_budget,
            )
            # Inline tool-use ledger reminder. Reasoning models in long
            # multi-round tool sessions sometimes re-issue the same
            # file_read / grep / glob calls because the original tool
            # results get visually buried by later content. We surface a
            # fresh summary of "what you've already called this turn" so the
            # LLM sees a compact view of its own history every round.
            # Returns "" for short, no-duplicate sessions so it adds
            # no overhead until the LLM actually starts repeating.
            #
            # Cache stability: the reminder changes every round, so it must
            # NOT touch the system prompt — rewriting conversation_messages[0]
            # would invalidate DeepSeek's automatic prefix cache (cache-hit
            # input tokens bill ~10x cheaper) on exactly the long sessions
            # where caching matters most. Keep the base prompt byte-stable and
            # ride the reminder on an EPHEMERAL turn-tail message that is NOT
            # persisted into conversation_messages, so the next round's real
            # appends still extend the same stable prefix.
            _ledger = _extract_ledger_from_messages(conversation_messages)
            _ledger_reminder = _format_inline_ledger_reminder(
                _ledger, language=session.prompt_language,
            )
            conversation_messages[0] = {
                "role": "system", "content": prompt_parts.combined,
            }
            call_messages = conversation_messages
            if _ledger_reminder:
                call_messages = conversation_messages + [
                    {"role": "user", "content": _ledger_reminder}
                ]
            response = await llm_adapter.complete_with_messages(
                messages=call_messages,
                tools=api_tools,
            )
            continue_count = 0
        else:
            response, continue_count = await llm_adapter.complete_with_continue(
                system_prompt=prompt_parts.combined,
                user_text=current_prompt,
                max_auto_continue=2,
                continue_prompt_builder=lambda previous_user_text, partial_response: _build_continue_prompt(
                    prompt_catalog=prompt_catalog,
                    prompt_language=session.prompt_language,
                    previous_user_text=previous_user_text,
                    partial_response=partial_response,
                ),
                tools=api_tools,
            )
        tool_calls = list(response.tool_calls)
        completion_exit_reason = (
            _EXIT_REASON_COMPLETED if tool_round == 0 else _EXIT_REASON_COMPLETED_AFTER_TOOLS
        )
        if not tool_calls and _agent_collaboration_incomplete(session):
            if incomplete_agent_reprompts < _MAX_INCOMPLETE_AGENT_REPROMPTS:
                incomplete_agent_reprompts += 1
                _reprompt_text = _agent_mode_incomplete_instruction(
                    prompt_catalog=prompt_catalog,
                    prompt_language=session.prompt_language,
                )
                if use_messages_mode:
                    conversation_messages.append({"role": "assistant", "content": response.content or ""})
                    conversation_messages.append({"role": "user", "content": _reprompt_text})
                    conversation_messages[0] = {"role": "system", "content": prompt_parts.combined}
                else:
                    current_prompt = _serialize_follow_up_prompt(
                        user_input=content,
                        assistant_response=response.payload,
                        tool_results=[],
                        prompt_catalog=prompt_catalog,
                        prompt_language=session.prompt_language,
                        instruction_override=_reprompt_text,
                    )
                continue
            completion_exit_reason = _EXIT_REASON_AGENT_REPROMPT_EXHAUSTED
        if not tool_calls and _plan_artifacts_incomplete(session):
            if incomplete_plan_reprompts < _MAX_INCOMPLETE_PLAN_REPROMPTS:
                incomplete_plan_reprompts += 1
                _reprompt_text = _plan_incomplete_instruction(session)
                if use_messages_mode:
                    conversation_messages.append({"role": "assistant", "content": response.content or ""})
                    conversation_messages.append({"role": "user", "content": _reprompt_text})
                    conversation_messages[0] = {"role": "system", "content": prompt_parts.combined}
                else:
                    current_prompt = _serialize_follow_up_prompt(
                        user_input=content,
                        assistant_response=response.payload,
                        tool_results=[],
                        prompt_catalog=prompt_catalog,
                        prompt_language=session.prompt_language,
                        instruction_override=_reprompt_text,
                    )
                continue
            completion_exit_reason = _EXIT_REASON_PLAN_REPROMPT_EXHAUSTED
        if not tool_calls and _spec_artifacts_incomplete(session):
            if incomplete_spec_reprompts < _MAX_INCOMPLETE_SPEC_REPROMPTS:
                incomplete_spec_reprompts += 1
                _reprompt_text = _spec_incomplete_instruction(session)
                if use_messages_mode:
                    conversation_messages.append({"role": "assistant", "content": response.content or ""})
                    conversation_messages.append({"role": "user", "content": _reprompt_text})
                    conversation_messages[0] = {"role": "system", "content": prompt_parts.combined}
                else:
                    current_prompt = _serialize_follow_up_prompt(
                        user_input=content,
                        assistant_response=response.payload,
                        tool_results=[],
                        prompt_catalog=prompt_catalog,
                        prompt_language=session.prompt_language,
                        instruction_override=_reprompt_text,
                    )
                continue
            completion_exit_reason = _EXIT_REASON_SPEC_REPROMPT_EXHAUSTED
        if not tool_calls and _implementation_tasks_incomplete(session):
            if incomplete_impl_reprompts < _MAX_INCOMPLETE_IMPLEMENTATION_REPROMPTS:
                incomplete_impl_reprompts += 1
                _reprompt_text = _implementation_followup_instruction(
                    prompt_catalog=prompt_catalog,
                    prompt_language=session.prompt_language,
                )
                if use_messages_mode:
                    conversation_messages.append({"role": "assistant", "content": response.content or ""})
                    conversation_messages.append({"role": "user", "content": _reprompt_text})
                    conversation_messages[0] = {"role": "system", "content": prompt_parts.combined}
                else:
                    current_prompt = _serialize_follow_up_prompt(
                        user_input=content,
                        assistant_response=response.payload,
                        tool_results=[],
                        prompt_catalog=prompt_catalog,
                        prompt_language=session.prompt_language,
                        instruction_override=_reprompt_text,
                    )
                continue
            completion_exit_reason = _EXIT_REASON_IMPLEMENTATION_REPROMPT_EXHAUSTED
        if not tool_calls and _todos_incomplete(session):
            if incomplete_todos_reprompts < _MAX_INCOMPLETE_TODOS_REPROMPTS:
                incomplete_todos_reprompts += 1
                _reprompt_text = _todos_incomplete_instruction(session)
                if use_messages_mode:
                    conversation_messages.append({"role": "assistant", "content": response.content or ""})
                    conversation_messages.append({"role": "user", "content": _reprompt_text})
                    conversation_messages[0] = {"role": "system", "content": prompt_parts.combined}
                else:
                    current_prompt = _serialize_follow_up_prompt(
                        user_input=content,
                        assistant_response=response.payload,
                        tool_results=[],
                        prompt_catalog=prompt_catalog,
                        prompt_language=session.prompt_language,
                        instruction_override=_reprompt_text,
                    )
                continue
            completion_exit_reason = _EXIT_REASON_TODOS_REPROMPT_EXHAUSTED
        if not tool_calls:
            # Empty-response salvage: when the LLM voluntarily stops
            # with NO tool calls AND essentially empty content after
            # doing substantive work (≥10 tool rounds), do ONE no-tools
            # follow-up requesting a real final answer. Conversation
            # compaction (kicks in at 80 messages, keeps last 50) loses
            # earlier file-read content when investigators chain many
            # tool calls, and the LLM sometimes responds with an empty
            # string instead of synthesising what's left in context.
            # Without this, investigators that did 100+ tool calls end
            # up with ``status=empty`` and zero usable output. Only
            # fires in messages_mode and only once per turn.
            _raw_content = (response.content or "").strip()
            if (
                use_messages_mode
                and not _raw_content
                and tool_round >= 10
                and incomplete_agent_reprompts == 0  # avoid double-reprompt
            ):
                try:
                    salvage_instruction = (
                        "Your previous response was empty. Based on the "
                        "tool results you have already gathered in this "
                        "turn, emit your final answer NOW per the "
                        "original instructions in the system prompt. Do "
                        "NOT call more tools; just produce the final "
                        "text (or JSON, if the system prompt requires "
                        "structured output). Empty replies are not "
                        "acceptable — even a partial summary is better."
                    )
                    salvage_messages = list(conversation_messages) + [
                        {"role": "assistant", "content": ""},
                        {"role": "user", "content": salvage_instruction},
                    ]
                    salvage_response = await llm_adapter.complete_with_messages(
                        messages=salvage_messages,
                        tools=None,
                    )
                    salvage_text = (
                        getattr(salvage_response, "content", "") or ""
                    ).strip()
                    if salvage_text:
                        logger.info(
                            "query_loop: empty-response salvage produced "
                            "%d chars after %d tool rounds",
                            len(salvage_text), tool_round,
                        )
                        final_message = SessionMessage(
                            message_id=f"msg_{uuid.uuid4().hex[:10]}",
                            turn_id=turn_id,
                            role="assistant",
                            content=salvage_text,
                            kind="assistant_text",
                            metadata={
                                **_build_completion_metadata(
                                    tool_round=tool_round,
                                    continue_count=continue_count,
                                    exit_reason=completion_exit_reason,
                                ),
                                "empty_response_salvaged": True,
                            },
                        )
                        yield SessionEvent(
                            event_type="assistant_followup_completed",
                            turn_id=turn_id,
                            message=final_message,
                        )
                        return
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "query_loop: empty-response salvage call failed: "
                        "%s — falling through to original empty reply",
                        exc,
                    )
            final_message = SessionMessage(
                message_id=f"msg_{uuid.uuid4().hex[:10]}",
                turn_id=turn_id,
                role="assistant",
                content=response.content,
                kind="assistant_text",
                metadata=_build_completion_metadata(
                    tool_round=tool_round,
                    continue_count=continue_count,
                    exit_reason=completion_exit_reason,
                ),
            )
            yield SessionEvent(
                event_type="assistant_completed" if tool_round == 0 else "assistant_followup_completed",
                turn_id=turn_id,
                message=final_message,
            )
            return
        incomplete_impl_reprompts = 0
        incomplete_todos_reprompts = 0

        assistant_message = SessionMessage(
            message_id=f"msg_{uuid.uuid4().hex[:10]}",
            turn_id=turn_id,
            role="assistant",
            content=response.content,
            kind="assistant_tool_use",
            metadata={"tool_call_count": len(tool_calls), "continue_count": continue_count},
        )
        yield SessionEvent(
            event_type="assistant_tool_plan",
            turn_id=turn_id,
            message=assistant_message,
        )

        normalized_calls = [_normalize_tool_call(item) for item in tool_calls]
        collected_results: list[ToolResult] = []
        pre_tool_state = dict(session.metadata.state)

        for call in normalized_calls:
            assistant_tool_message = SessionMessage(
                message_id=f"msg_{uuid.uuid4().hex[:10]}",
                turn_id=turn_id,
                role="assistant",
                content=f"tool:{call.tool_name}",
                kind="assistant_tool_use",
                tool_name=call.tool_name,
                tool_use_id=call.tool_use_id,
                metadata={"structured_payload": {"tool_name": call.tool_name, "arguments": call.arguments}},
            )
            yield SessionEvent(
                event_type="assistant_tool_use",
                turn_id=turn_id,
                message=assistant_tool_message,
                payload={"tool_call": {"tool_name": call.tool_name, "arguments": call.arguments}},
            )

        async for tool_event in tool_orchestrator.run_tool_calls(normalized_calls, tool_ctx):
            if tool_event.event_type == "tool_context_updated":
                maybe_context = tool_event.payload.get("tool_context")
                if isinstance(maybe_context, ToolUseContext):
                    tool_ctx = maybe_context
                    context_assembler.apply_tool_context(session=session, tool_ctx=tool_ctx)
                continue

            progress_message = SessionMessage(
                message_id=f"msg_{uuid.uuid4().hex[:10]}",
                turn_id=turn_id,
                role="system",
                content=f"tool_progress:{tool_event.tool_name}:{tool_event.event_type}",
                kind="progress_message",
                tool_name=tool_event.tool_name,
                tool_use_id=tool_event.tool_use_id,
                metadata={"structured_payload": tool_mapper.to_progress_message(tool_event)},
            )
            yield SessionEvent(
                event_type=tool_event.event_type,
                turn_id=turn_id,
                message=progress_message,
                payload={"tool_progress": tool_mapper.to_progress_message(tool_event)},
            )

            result = tool_event.payload.get("result")
            if result is not None:
                collected_results.append(result)
                yield SessionEvent(
                    event_type="tool_result",
                    turn_id=turn_id,
                    message=tool_mapper.to_session_message(turn_id, result),
                )

        auto_exit_calls: list[ToolCall] = []
        if _should_auto_exit_plan(session):
            auto_exit_calls.append(
                ToolCall(
                    tool_use_id=f"toolu_{uuid.uuid4().hex[:8]}",
                    tool_name="exit_plan_mode",
                    arguments={},
                )
            )
        if _should_auto_exit_spec(session):
            auto_exit_calls.append(
                ToolCall(
                    tool_use_id=f"toolu_{uuid.uuid4().hex[:8]}",
                    tool_name="exit_spec_mode",
                    arguments={},
                )
            )
        if auto_exit_calls:
            async for auto_event in _run_additional_tool_calls(
                turn_id=turn_id,
                tool_calls=auto_exit_calls,
                tool_orchestrator=tool_orchestrator,
                tool_ctx=tool_ctx,
                tool_mapper=tool_mapper,
                context_assembler=context_assembler,
                session=session,
            ):
                if auto_event.event_type == "tool_context_updated":
                    maybe_context = auto_event.payload.get("tool_context")
                    if isinstance(maybe_context, ToolUseContext):
                        tool_ctx = maybe_context
                        context_assembler.apply_tool_context(session=session, tool_ctx=tool_ctx)
                    continue
                result = auto_event.payload.get("result")
                if result is not None:
                    collected_results.append(result)
                yield auto_event

        if use_messages_mode:
            openai_tc = _cc_tool_calls_to_openai([*normalized_calls, *auto_exit_calls])
            assistant_payload = {
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": openai_tc,
            }
            reasoning_content = response.payload.get("reasoning_content")
            if reasoning_content:
                assistant_payload["reasoning_content"] = str(reasoning_content)
            conversation_messages.append(assistant_payload)
            for result in collected_results:
                conversation_messages.append({
                    "role": "tool",
                    "tool_call_id": result.tool_use_id,
                    "content": _truncate_tool_result_content(result),
                })

        _update_agent_collaboration_state(session=session, tool_results=collected_results)

        # --- Opt-in post-edit verification (default OFF; byte-equivalent when
        # off). After a round that mutated code, run the configured verification
        # command and feed the exit-code-gated verdict back into the loop so the
        # model sees a red suite and the auto-complete grace can't declare the
        # work done on red. Returns None (no-op) when the feature is disabled. ---
        post_edit_verdict = await _run_post_edit_verification(
            config=tool_ctx.config,
            cwd=tool_ctx.cwd,
            collected_results=collected_results,
        )
        if post_edit_verdict is not None:
            latest_post_edit_verify_passed = post_edit_verdict.passed
            _verdict_note = _post_edit_verdict_note(post_edit_verdict)
            if use_messages_mode:
                conversation_messages.append({"role": "user", "content": _verdict_note})
            yield SessionEvent(
                event_type="post_edit_verification",
                turn_id=turn_id,
                message=SessionMessage(
                    message_id=f"msg_{uuid.uuid4().hex[:10]}",
                    turn_id=turn_id,
                    role="system",
                    content=_verdict_note,
                    kind="progress_message",
                    metadata={
                        "post_edit_verification": True,
                        "passed": post_edit_verdict.passed,
                        "exit_code": post_edit_verdict.exit_code,
                        "unrunnable": post_edit_verdict.unrunnable,
                        "command": post_edit_verdict.command,
                    },
                ),
            )

        exited_mode = _mode_exited(pre_tool_state, dict(session.metadata.state))
        current_tasks_snapshot = _implementation_tasks_snapshot(session)
        followup_instruction: str | None = None
        if prompt_builder is not None and exited_mode is not None:
            rebuild_prompt_key: str | None = None
            if exited_mode == "plan":
                rebuild_prompt_key = "system.plan_implementation"
                try:
                    followup_instruction = prompt_catalog.resolve(
                        "system.implementation_followup", session.prompt_language,
                    )
                except Exception:
                    pass
            elif exited_mode == "spec":
                rebuild_prompt_key = "system.spec_render"
                try:
                    followup_instruction = prompt_catalog.resolve(
                        "system.render_followup", session.prompt_language,
                    )
                except Exception:
                    pass
            new_tool_ctx = context_assembler.build_tool_context(session=session, turn_id=turn_id)
            new_registry = getattr(tool_orchestrator, "registry", None)
            new_enabled = [t.spec.name for t in new_registry.list_visible(new_tool_ctx)] if new_registry else []
            prompt_extra: dict[str, Any] = {}
            if api_tools is not None:
                prompt_extra["native_tool_calling"] = True
                if new_registry is not None:
                    api_tools = _cc_schemas_to_openai_tools(new_registry.export_model_schemas(new_tool_ctx))
            # Remember the transition prompt so subsequent rounds (which see
            # ``exited_mode is None``) keep rendering it instead of resetting
            # to the generic default.
            active_prompt_key = rebuild_prompt_key
            prompt_parts = prompt_builder.build(
                prompt_language=session.prompt_language,
                prompt_key=rebuild_prompt_key,
                context=context_assembler.build_prompt_context(
                    session=session,
                    tool_ctx=new_tool_ctx,
                    enabled_tools=new_enabled,
                    extra=prompt_extra,
                ),
            ).parts
        elif prompt_builder is not None and exited_mode is None:
            refreshed_tool_ctx = context_assembler.build_tool_context(session=session, turn_id=turn_id)
            refreshed_registry = getattr(tool_orchestrator, "registry", None)
            refreshed_enabled = [t.spec.name for t in refreshed_registry.list_visible(refreshed_tool_ctx)] if refreshed_registry else []
            refresh_extra: dict[str, Any] = {}
            if api_tools is not None:
                refresh_extra["native_tool_calling"] = True
                if refreshed_registry is not None:
                    api_tools = _cc_schemas_to_openai_tools(refreshed_registry.export_model_schemas(refreshed_tool_ctx))
            prompt_parts = prompt_builder.build(
                prompt_language=session.prompt_language,
                prompt_key=active_prompt_key,
                context=context_assembler.build_prompt_context(
                    session=session,
                    tool_ctx=refreshed_tool_ctx,
                    enabled_tools=refreshed_enabled,
                    extra=refresh_extra,
                ),
            ).parts
        if followup_instruction is None and _implementation_tasks_incomplete(session):
            try:
                if _implementation_requires_task_sync(
                    previous_tasks_snapshot=previous_tasks_snapshot,
                    current_tasks_snapshot=current_tasks_snapshot,
                    tool_results=collected_results,
                ):
                    followup_instruction = _implementation_task_sync_instruction(
                        prompt_catalog=prompt_catalog,
                        prompt_language=session.prompt_language,
                    )
                else:
                    followup_instruction = _implementation_followup_instruction(
                        prompt_catalog=prompt_catalog,
                        prompt_language=session.prompt_language,
                    )
            except Exception:
                pass
        if followup_instruction is None and _agent_collaboration_incomplete(session):
            try:
                followup_instruction = _agent_mode_incomplete_instruction(
                    prompt_catalog=prompt_catalog,
                    prompt_language=session.prompt_language,
                )
            except Exception:
                pass
        if followup_instruction is None and _todos_incomplete(session):
            followup_instruction = _todos_incomplete_instruction(session)

        if _implementation_tasks_incomplete(session):
            _made_tasks_progress = _implementation_round_made_progress(
                previous_tasks_snapshot=previous_tasks_snapshot,
                current_tasks_snapshot=current_tasks_snapshot,
                tool_results=collected_results,
            )
            _has_code_mutation = any(
                r.success and r.tool_name in _CODE_MUTATION_TOOLS for r in collected_results
            )
            if _made_tasks_progress:
                implementation_stall_rounds = 0
                implementation_code_only_rounds = 0
                implementation_pending_task_sync = False
            elif _has_code_mutation:
                implementation_pending_task_sync = True
                implementation_code_only_rounds += 1
                implementation_stall_rounds = 0
                if implementation_code_only_rounds >= _MAX_CODE_ONLY_GRACE_ROUNDS and not _post_edit_blocks_autocomplete(
                    tool_ctx.config, latest_post_edit_verify_passed
                ):
                    _auto_complete_tasks(session)
                    current_tasks_snapshot = _implementation_tasks_snapshot(session)
                    implementation_stall_rounds = 0
                    implementation_code_only_rounds = 0
                    implementation_pending_task_sync = False
            elif implementation_pending_task_sync:
                implementation_code_only_rounds += 1
                implementation_stall_rounds = 0
                if implementation_code_only_rounds >= _MAX_CODE_ONLY_GRACE_ROUNDS and not _post_edit_blocks_autocomplete(
                    tool_ctx.config, latest_post_edit_verify_passed
                ):
                    _auto_complete_tasks(session)
                    current_tasks_snapshot = _implementation_tasks_snapshot(session)
                    implementation_stall_rounds = 0
                    implementation_code_only_rounds = 0
                    implementation_pending_task_sync = False
            else:
                implementation_stall_rounds += 1
        else:
            implementation_stall_rounds = 0
            implementation_code_only_rounds = 0
            implementation_pending_task_sync = False
        previous_tasks_snapshot = current_tasks_snapshot
        if implementation_stall_rounds >= _MAX_IMPLEMENTATION_STALL_ROUNDS:
            stalled_message = SessionMessage(
                message_id=f"msg_{uuid.uuid4().hex[:10]}",
                turn_id=turn_id,
                role="assistant",
                content=(
                    f"Implementation stalled: tasks.md showed no progress for "
                    f"{_MAX_IMPLEMENTATION_STALL_ROUNDS} consecutive tool rounds."
                ),
                kind="assistant_text",
                metadata={
                    "implementation_stalled": True,
                    "stall_rounds": implementation_stall_rounds,
                    "exit_reason": _EXIT_REASON_IMPLEMENTATION_STALLED,
                    "tool_rounds": tool_round,
                },
            )
            yield SessionEvent(
                event_type="assistant_followup_completed",
                turn_id=turn_id,
                message=stalled_message,
            )
            return

        _progress_tools = {"file_write", "file_edit", "delete_file", "shell", "powershell"}
        # When the session's tool registry has NO write tools registered
        # (read-only investigator sessions for doc / ask / research modes
        # — the registry is restricted by ``restrict_tool_registry``
        # before submit), the "no writes = no progress" heuristic
        # misclassifies normal behaviour as a stall and kills the loop
        # at ``_MAX_GENERIC_READ_ONLY_STALL_ROUNDS``. Detect read-only
        # sessions by inspecting the registry and treat ANY successful
        # tool call as progress; the real budget is still bounded by
        # ``max_tool_rounds``.
        _current_registry = getattr(tool_orchestrator, "registry", None)
        _has_write_tools = False
        if _current_registry is not None:
            try:
                _registered = set(getattr(_current_registry, "_tools", {}).keys())
            except Exception:
                _registered = set()
            _has_write_tools = bool(_registered & _progress_tools)
        if _has_write_tools:
            _progress_hit = any(
                r.success and r.tool_name in _progress_tools
                for r in collected_results
            )
        else:
            _progress_hit = any(r.success for r in collected_results)
        if _progress_hit:
            generic_read_only_stall_rounds = 0
        else:
            generic_read_only_stall_rounds += 1
        effective_stall_limit = _MAX_GENERIC_READ_ONLY_STALL_ROUNDS
        if _todos_incomplete(session):
            effective_stall_limit = _MAX_GENERIC_READ_ONLY_STALL_ROUNDS * 2
        if generic_read_only_stall_rounds >= effective_stall_limit:
            if _todos_incomplete(session) and generic_stall_reprompts < 1:
                generic_stall_reprompts += 1
                generic_read_only_stall_rounds = effective_stall_limit // 2
                _rescue_text = _todos_stall_rescue_instruction(session)
                if use_messages_mode:
                    conversation_messages.append({"role": "user", "content": _rescue_text})
                else:
                    current_prompt = _serialize_follow_up_prompt(
                        user_input=content,
                        assistant_response=response.payload,
                        tool_results=collected_results,
                        prompt_catalog=prompt_catalog,
                        prompt_language=session.prompt_language,
                        instruction_override=_rescue_text,
                    )
                tool_round += 1
                continue
            generic_stall_msg = SessionMessage(
                message_id=f"msg_{uuid.uuid4().hex[:10]}",
                turn_id=turn_id,
                role="assistant",
                content=(
                    f"Generic stall detected: {generic_read_only_stall_rounds} consecutive "
                    f"tool rounds with no progress. Stopping."
                ),
                kind="assistant_text",
                metadata={
                    "generic_stalled": True,
                    "stall_rounds": generic_read_only_stall_rounds,
                    "exit_reason": _EXIT_REASON_GENERIC_STALLED,
                    "tool_rounds": tool_round,
                },
            )
            yield SessionEvent(
                event_type="assistant_followup_completed",
                turn_id=turn_id,
                message=generic_stall_msg,
            )
            return

        if use_messages_mode:
            if followup_instruction:
                conversation_messages.append({"role": "user", "content": followup_instruction})
            conversation_messages[0] = {"role": "system", "content": prompt_parts.combined}
        else:
            current_prompt = _serialize_follow_up_prompt(
                user_input=content,
                assistant_response=response.payload,
                tool_results=collected_results,
                prompt_catalog=prompt_catalog,
                prompt_language=session.prompt_language,
                instruction_override=followup_instruction,
            )
        tool_round += 1

    if max_tool_rounds is not None:
        actual_limit = _effective_tool_limit(max_tool_rounds, session)
        # Before declaring "stopped", give the LLM ONE final no-tools
        # turn to wrap up. The LLM was about to do the next action —
        # without this, work-in-progress (JSON emit, summary, "let me
        # compile the results") is lost and the caller sees only the
        # round-cap notice. This was the root cause of doc-mode
        # investigators returning the generic notice as their final
        # text instead of the actual report. Falls back to the
        # original notice if the wrap call errors (in messages_mode
        # only — non-messages mode keeps the legacy behavior).
        final_wrap_text = ""
        if use_messages_mode:
            try:
                wrap_instruction = (
                    "You have reached the tool-round budget for this "
                    "turn. Do NOT call any more tools — calls will be "
                    "ignored. Based on what you have gathered so far, "
                    "emit your FINAL answer NOW per the original "
                    "instructions in the system prompt. If the format "
                    "is JSON, emit ONLY the JSON; if it's prose, give "
                    "the wrap-up directly. No more questions, no more "
                    "tool plans — just the final answer."
                )
                wrap_messages = list(conversation_messages) + [
                    {"role": "user", "content": wrap_instruction},
                ]
                wrap_response = await llm_adapter.complete_with_messages(
                    messages=wrap_messages,
                    tools=None,
                )
                final_wrap_text = (
                    getattr(wrap_response, "content", "") or ""
                ).strip()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "tool-round-limit final wrap call failed: %s — "
                    "falling back to generic stopped notice", exc,
                )
                final_wrap_text = ""
        if final_wrap_text:
            wrap_msg = SessionMessage(
                message_id=f"msg_{uuid.uuid4().hex[:10]}",
                turn_id=turn_id,
                role="assistant",
                content=final_wrap_text,
                kind="assistant_text",
                metadata={
                    "tool_round_limit_reached": True,
                    "tool_rounds": tool_round,
                    "effective_tool_limit": actual_limit,
                    "exit_reason": _EXIT_REASON_TOOL_ROUND_LIMIT,
                    "final_wrap_after_limit": True,
                },
            )
            yield SessionEvent(
                event_type="assistant_followup_completed",
                turn_id=turn_id,
                message=wrap_msg,
            )
        else:
            limit_message = SessionMessage(
                message_id=f"msg_{uuid.uuid4().hex[:10]}",
                turn_id=turn_id,
                role="assistant",
                content=f"Tool round limit ({actual_limit}) reached. Stopping automatic tool execution.",
                kind="assistant_text",
                metadata={
                    "tool_round_limit_reached": True,
                    "tool_rounds": tool_round,
                    "effective_tool_limit": actual_limit,
                    "exit_reason": _EXIT_REASON_TOOL_ROUND_LIMIT,
                },
            )
            yield SessionEvent(
                event_type="assistant_followup_completed",
                turn_id=turn_id,
                message=limit_message,
            )


def _build_completion_metadata(
    *,
    tool_round: int,
    continue_count: int,
    exit_reason: str,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "continue_count": continue_count,
        "exit_reason": exit_reason,
        "tool_rounds": tool_round,
    }
    if tool_round == 0:
        metadata["tool_call_count"] = 0
    else:
        metadata["final_after_tools"] = True
    return metadata


def _agent_collaboration_incomplete(session: QuerySession) -> bool:
    state = dict(session.metadata.state)
    if session.agent_mode != "agent":
        return False
    if not (state.get("system_prompt_context") or {}).get("agent_collaboration_required"):
        return False
    return not bool(state.get("agent_collaboration_completed"))


def _update_agent_collaboration_state(*, session: QuerySession, tool_results: list[ToolResult]) -> None:
    if session.agent_mode != "agent":
        return
    successful_agent_calls = [
        result for result in tool_results if result.success and result.tool_name == "agent"
    ]
    if not successful_agent_calls:
        return
    state = dict(session.metadata.state)
    state["agent_collaboration_completed"] = True
    state["agent_collaboration_count"] = int(state.get("agent_collaboration_count", 0)) + len(successful_agent_calls)
    prompt_context = dict(state.get("system_prompt_context") or {})
    prompt_context["agent_collaboration_completed"] = True
    prompt_context["agent_collaboration_count"] = int(prompt_context.get("agent_collaboration_count", 0)) + len(
        successful_agent_calls,
    )
    state["system_prompt_context"] = prompt_context
    session.metadata.state = state


