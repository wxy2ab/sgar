"""ask_human cc tool — put a question/decision to a human and resume.

ccx runs autonomously, but a host can register an interaction handler on its
``CodeAgent`` to let the agent ask a human mid-run. When a handler is present,
``CcAgentRunner`` registers this tool into the per-turn cc registry; otherwise
it is hidden entirely (so the default path — system prompt, tool list, prompt
cache — is byte-identical).

Mechanism: the tool reads the host callback off the current v5 dispatch context
(``current_dispatch_context().interaction_fn``, threaded there exactly like
``report_cost_fn``), and blocks on it under a bounded, heartbeated wait
(:func:`core.ccx.services.interaction.run_interaction`). The block lives inside
an already-running tool dispatch on a worker thread — it never creates an
approval-gated v5 node, so the deliberately-forbidden ``requires_approval``
"wait forever" failure mode does not apply here.

Degradation: on timeout / no-answer the tool returns ``success=True`` with a
"proceed on your best judgment" sentinel — it never errors or aborts the run.
An autonomous agent must be able to continue when no human is reachable.
"""

from __future__ import annotations

import time
from typing import Any

from core.cc.tools.base import (
    BaseTool,
    ToolCall,
    ToolResult,
    ToolSpec as CcToolSpec,
    ValidationResult,
)
from core.deepstack_v5.execution.dispatch_context import current_dispatch_context

from ..services.interaction import (
    DEFAULT_INTERACTION_TIMEOUT_S,
    EVENT_ANSWERED,
    EVENT_REQUESTED,
    STATUS_ANSWERED,
    STATUS_NO_HANDLER,
    STATUS_REFUSED,
    STATUS_TIMEOUT,
    InteractionRequest,
    normalize_severity,
    run_interaction,
)

_TOOL_NAME = "ask_human"

_TOOL_DESCRIPTION = (
    "Ask a human a question or put a decision to them, then continue with "
    "their answer. Use this ONLY when you genuinely need a human's input — an "
    "ambiguous requirement, a risky/irreversible action that wants sign-off, "
    "or a choice only the user can make. Provide a clear 'question'; offer "
    "discrete 'options' when the answer should be a pick. This blocks briefly "
    "while the human responds. If no human answers in time, you receive a "
    "notice to proceed on your best judgment — so never depend on a human "
    "being present; ask only when the value is real."
)

_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "question": {
            "type": "string",
            "description": "The question or decision to put to the human.",
        },
        "options": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Optional discrete choices. When given, the human is expected "
                "to pick one; the chosen option is returned in data.selected."
            ),
        },
        "context": {
            "type": "string",
            "description": (
                "Optional background the human needs to decide (what you are "
                "doing, why the question arose, the trade-offs)."
            ),
        },
        "severity": {
            "type": "string",
            "enum": ["info", "decision", "blocking"],
            "description": (
                "info = FYI, you may proceed; decision = pick/answer wanted; "
                "blocking = you should not proceed without an answer. "
                "Defaults to 'decision'."
            ),
        },
    },
    "required": ["question"],
}


class CcxAskHumanTool(BaseTool):
    """Synchronous cc tool that bridges the agent to a host interaction handler.

    The handler is NOT held on the tool — it is read from the per-call dispatch
    context at ``execute`` time, so one ``CcAgentRunner`` instance reused across
    runs always reaches the correct (per-run) callback.
    """

    def __init__(self, timeout_s: float = DEFAULT_INTERACTION_TIMEOUT_S) -> None:
        super().__init__(spec=CcToolSpec(
            name=_TOOL_NAME,
            description=_TOOL_DESCRIPTION,
            input_schema=_INPUT_SCHEMA,
            is_read_only=True,
            needs_confirmation=False,
            metadata={"ccx": True, "ask_human": True},
        ))
        self._timeout_s = timeout_s

    def is_enabled(self, ctx: Any) -> bool:
        del ctx
        return True

    def is_concurrency_safe(self, arguments: dict[str, Any]) -> bool:
        del arguments
        return True

    def validate_input(self, arguments: dict[str, Any]) -> ValidationResult:
        question = arguments.get("question")
        if not isinstance(question, str) or not question.strip():
            return ValidationResult(
                ok=False,
                message="ask_human requires a non-empty 'question'",
            )
        options = arguments.get("options")
        if options is not None and not isinstance(options, list):
            return ValidationResult(
                ok=False,
                message="ask_human 'options' must be a list of strings",
            )
        return ValidationResult(ok=True)

    async def execute(self, tool_call: ToolCall, ctx: Any) -> ToolResult:
        del ctx
        args = dict(tool_call.arguments or {})
        dispatch_ctx = current_dispatch_context()
        interaction_fn = (
            getattr(dispatch_ctx, "interaction_fn", None)
            if dispatch_ctx is not None
            else None
        )
        run_id = dispatch_ctx.run_id if dispatch_ctx is not None else None
        node_id = dispatch_ctx.node_id if dispatch_ctx is not None else None

        if interaction_fn is None:
            # Defensive: normally the tool is hidden when no handler exists, so
            # this only fires if something registered it without plumbing a
            # callback. Degrade to autonomy rather than erroring.
            return ToolResult(
                tool_use_id=tool_call.tool_use_id,
                tool_name=tool_call.tool_name,
                success=True,
                content=(
                    "No human is reachable right now; proceed using your best "
                    "judgment."
                ),
                data={"status": STATUS_NO_HANDLER},
            )

        options = tuple(
            str(o) for o in (args.get("options") or []) if str(o).strip()
        )
        request = InteractionRequest(
            question=str(args.get("question") or "").strip(),
            options=options,
            context=str(args.get("context") or ""),
            severity=normalize_severity(args.get("severity")),
            run_id=run_id,
            node_id=node_id,
        )

        emit = getattr(dispatch_ctx, "emit", None) if dispatch_ctx is not None else None
        self._emit(emit, EVENT_REQUESTED, {
            "run_id": run_id,
            "node_id": node_id,
            "question": request.question,
            "options": list(request.options),
            "severity": request.severity,
        })

        started = time.monotonic()
        resp = run_interaction(
            interaction_fn,
            request,
            timeout_s=self._timeout_s,
            emit=emit,
        )
        latency_ms = int((time.monotonic() - started) * 1000)

        self._emit(emit, EVENT_ANSWERED, {
            "run_id": run_id,
            "node_id": node_id,
            "status": resp.status,
            "selected": resp.selected,
            "latency_ms": latency_ms,
        })

        return self._result_for(tool_call, request, resp)

    # ----- helpers ----------------------------------------------------------

    @staticmethod
    def _emit(emit: Any, kind: str, payload: dict[str, Any]) -> None:
        if emit is None:
            return
        try:
            emit(kind, payload)
        except Exception:  # noqa: BLE001 — observability is best-effort
            pass

    def _result_for(
        self,
        tool_call: ToolCall,
        request: InteractionRequest,
        resp: Any,
    ) -> ToolResult:
        status = getattr(resp, "status", STATUS_TIMEOUT)
        answer = str(getattr(resp, "answer", "") or "")
        selected = getattr(resp, "selected", None)

        if status == STATUS_ANSWERED:
            content = answer or (selected or "")
            if selected and selected not in content:
                content = f"{selected}: {content}" if content else str(selected)
            content = content or "(the human answered with no text)"
        elif status == STATUS_REFUSED:
            content = (
                "The human declined to answer"
                + (f": {answer}" if answer else "")
                + ". Proceed using your best judgment."
            )
        else:  # timeout / no_handler / unknown
            content = (
                "No human answered within the time limit; use your best "
                "judgment and proceed."
            )

        return ToolResult(
            tool_use_id=tool_call.tool_use_id,
            tool_name=tool_call.tool_name,
            success=True,
            content=content,
            data={
                "status": status,
                "selected": selected,
                "answer": answer,
                "question": request.question,
            },
        )


__all__ = ["CcxAskHumanTool"]
