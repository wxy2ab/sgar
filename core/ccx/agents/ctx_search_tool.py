"""ctx_search cc tool — recall large prior tool outputs from ContentStore.

Phase 5 of the context-mode port. Phase 2 wired ContentStore as an *inlet*:
when a cc ``tool_result`` body exceeds 4 KB the event bridge enqueues the
full content into an FTS5 index and stamps ``full_content_ref`` on the v5
event payload. The LLM never sees the full body — it sees a 240-char
preview only.

This tool is the *outlet*. When the LLM, two turns later, realises "I need
the full output of that earlier grep", it calls ``ctx_search`` to either
search the index by keyword or fetch a specific source_id verbatim. Without
this, ContentStore is observable to humans (``ccx watch --stats``) but
contributes nothing to the LLM's working memory.

Two actions, one wire name:

* ``ctx_search(action="search", query="...", top_k=5)`` → ranked chunks
  (relevance = ``-bm25`` so higher = better; original bm25 retained for
  debugging). Returns source_id + ord + label + body preview per hit;
  full body is in ``data["hits"][i]["body"]``.
* ``ctx_search(action="fetch", source_id="cc.N1.tu5", max_bytes=64000)`` →
  full reassembled content, truncated to ``max_bytes`` if longer.

Run scoping: ``run_id`` defaults to the current v5 dispatch's run_id (read
via ``current_dispatch_context`` at execute time, not construction time —
the same CcAgentRunner instance may be reused across runs). An explicit
``run_id`` argument overrides; ``None`` (no dispatch + no override) falls
through to ContentStore's cross-run search.

ContentStore-missing degradation: ``CcAgentRunner`` only registers this
tool when its ``content_store`` is non-None, but the tool also handles
``content_store=None`` at execute() time defensively — returns a clear
``ctx_search.unavailable`` error rather than raising.
"""

from __future__ import annotations

from typing import Any

from core.cc.tools.base import (
    BaseTool,
    ToolCall,
    ToolResult,
    ToolSpec as CcToolSpec,
    ValidationResult,
)
from core.deepstack_v5.execution.dispatch_context import current_dispatch_context
from core.deepstack_v5.memory import ContentStore


_TOOL_NAME = "ctx_search"

_TOOL_DESCRIPTION = (
    "Recall the FULL content of a prior large tool_result that was "
    "truncated in the transcript. Large tool outputs (>4 KB, e.g. a "
    "wide-scope grep, a long file read) are auto-indexed into an FTS5 "
    "store; the transcript shows only a 240-char preview. Use this when "
    "you realise you need the complete body of an earlier result. Two "
    "actions: action='search' (BM25-rank chunks by keyword, returns "
    "source_ids and body previews — use to discover what's in the index) "
    "and action='fetch' (return one source's full body by source_id — use "
    "after search to read the complete content). The tool is read-only "
    "and scoped to the current run by default."
)

_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["search", "fetch"],
            "description": (
                "'search' to BM25-rank chunks for a query; "
                "'fetch' to return a specific source's full body."
            ),
        },
        "query": {
            "type": "string",
            "description": (
                "Keyword query (required when action='search'). Plain "
                "natural language; FTS5 syntax characters are stripped."
            ),
        },
        "top_k": {
            "type": "integer",
            "description": (
                "Max hits to return for action='search'. Default 5; "
                "clamped to 1..20."
            ),
        },
        "source_id": {
            "type": "string",
            "description": (
                "Source identifier to fetch (required when action='fetch'). "
                "Look up via action='search' first; source_id looks like "
                "'cc.<node_id>.<tool_use_id>' for cc-originated content."
            ),
        },
        "max_bytes": {
            "type": "integer",
            "description": (
                "Truncate the fetched body at this many bytes. Default "
                "64000; hard cap 256000. The full byte count is reported "
                "in data.bytes regardless of truncation."
            ),
        },
        "run_id": {
            "type": "string",
            "description": (
                "Override the run_id scope (defaults to the current "
                "dispatch's run; omit unless you intentionally want to "
                "search a different run)."
            ),
        },
    },
    "required": ["action"],
}


_DEFAULT_TOP_K = 5
_MAX_TOP_K = 20
_DEFAULT_FETCH_MAX_BYTES = 64_000
_HARD_CAP_FETCH_MAX_BYTES = 256_000
_BODY_PREVIEW_CHARS = 400
_CONTENT_TEXT_MAX_BYTES = 8_192


def _utf8_len(text: str) -> int:
    return len(text.encode("utf-8"))


class CcxCtxSearchTool(BaseTool):
    """Synchronous read-only cc tool over a ContentStore.

    Construct with a ``ContentStore`` (or ``None`` for degraded mode);
    register into a cc tool registry exactly like any BaseTool. No buffer,
    no post-turn drain — the LLM gets the result inline within the same
    multi-tool round.
    """

    def __init__(self, content_store: ContentStore | None) -> None:
        super().__init__(spec=CcToolSpec(
            name=_TOOL_NAME,
            description=_TOOL_DESCRIPTION,
            input_schema=_INPUT_SCHEMA,
            is_read_only=True,
            needs_confirmation=False,
            metadata={"ccx": True, "ctx_search": True},
        ))
        self.content_store = content_store

    def is_enabled(self, ctx: Any) -> bool:
        del ctx
        return True

    def is_concurrency_safe(self, arguments: dict[str, Any]) -> bool:
        del arguments
        return True

    def validate_input(self, arguments: dict[str, Any]) -> ValidationResult:
        action = arguments.get("action")
        if not action:
            return ValidationResult(
                ok=False,
                message="ctx_search requires 'action' ('search' or 'fetch')",
            )
        if action not in ("search", "fetch"):
            return ValidationResult(
                ok=False,
                message=f"ctx_search: unknown action {action!r}",
            )
        if action == "search":
            query = arguments.get("query")
            if not isinstance(query, str) or not query.strip():
                return ValidationResult(
                    ok=False,
                    message="ctx_search action='search' requires non-empty 'query'",
                )
        else:  # fetch
            source_id = arguments.get("source_id")
            if not isinstance(source_id, str) or not source_id.strip():
                return ValidationResult(
                    ok=False,
                    message="ctx_search action='fetch' requires non-empty 'source_id'",
                )
        return ValidationResult(ok=True)

    async def execute(self, tool_call: ToolCall, ctx: Any) -> ToolResult:
        del ctx
        if self.content_store is None:
            return ToolResult(
                tool_use_id=tool_call.tool_use_id,
                tool_name=tool_call.tool_name,
                success=False,
                content=(
                    "content store not attached to this run; "
                    "ctx_search is unavailable."
                ),
                error_code="ctx_search.unavailable",
            )

        args = dict(tool_call.arguments or {})
        action = args.get("action")

        run_id = args.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            dispatch_ctx = current_dispatch_context()
            run_id = dispatch_ctx.run_id if dispatch_ctx is not None else None

        try:
            self.content_store.flush(timeout_s=0.5)
        except Exception:
            pass

        if action == "search":
            return self._do_search(tool_call, args, run_id)
        return self._do_fetch(tool_call, args)

    # ----- action handlers --------------------------------------------------

    def _do_search(
        self,
        tool_call: ToolCall,
        args: dict[str, Any],
        run_id: str | None,
    ) -> ToolResult:
        query = str(args.get("query") or "").strip()
        top_k_raw = args.get("top_k", _DEFAULT_TOP_K)
        try:
            top_k = int(top_k_raw)
        except (TypeError, ValueError):
            top_k = _DEFAULT_TOP_K
        top_k = max(1, min(top_k, _MAX_TOP_K))

        assert self.content_store is not None  # narrowed by execute()
        hits = self.content_store.search(query, run_id=run_id, top_k=top_k)

        hit_dicts: list[dict[str, Any]] = []
        for h in hits:
            hit_dicts.append({
                "source_id": h.source_id,
                "ord": h.ord,
                "label": h.label,
                "body": h.body,
                "relevance": -float(h.score),
                "bm25": float(h.score),
            })

        if not hit_dicts:
            content_text = (
                f"No matching chunks for query={query!r} "
                f"(run_id={run_id or '<any>'}, top_k={top_k})."
            )
        else:
            header = (
                f"Found {len(hit_dicts)} chunk(s) "
                f"(run_id={run_id or '<any>'}, query={query!r}):"
            )
            lines = [header]
            running_bytes = _utf8_len(header)
            for idx, h in enumerate(hit_dicts, start=1):
                preview = h["body"]
                if len(preview) > _BODY_PREVIEW_CHARS:
                    preview = preview[:_BODY_PREVIEW_CHARS] + "…"
                line = (
                    f"  {idx}. [{h['source_id']}, ord={h['ord']}, "
                    f"rel={h['relevance']:.2f}] "
                    f"{h['label'] or '<no label>'}\n"
                    f"     {preview}"
                )
                join_sep_bytes = 1 if lines else 0
                line_bytes = _utf8_len(line)
                if (
                    running_bytes + join_sep_bytes + line_bytes
                    > _CONTENT_TEXT_MAX_BYTES
                ):
                    elided = (
                        f"  … {len(hit_dicts) - idx + 1} more hit(s) elided "
                        "from this preview; fetch by source_id to read them."
                    )
                    if (
                        running_bytes + join_sep_bytes + _utf8_len(elided)
                        <= _CONTENT_TEXT_MAX_BYTES
                    ):
                        lines.append(elided)
                    break
                lines.append(line)
                running_bytes += join_sep_bytes + line_bytes
            content_text = "\n".join(lines)

        return ToolResult(
            tool_use_id=tool_call.tool_use_id,
            tool_name=tool_call.tool_name,
            success=True,
            content=content_text,
            data={
                "action": "search",
                "run_id": run_id,
                "query": query,
                "top_k": top_k,
                "hits": hit_dicts,
            },
        )

    def _do_fetch(
        self,
        tool_call: ToolCall,
        args: dict[str, Any],
    ) -> ToolResult:
        source_id = str(args.get("source_id") or "").strip()
        max_bytes_raw = args.get("max_bytes", _DEFAULT_FETCH_MAX_BYTES)
        try:
            max_bytes = int(max_bytes_raw)
        except (TypeError, ValueError):
            max_bytes = _DEFAULT_FETCH_MAX_BYTES
        max_bytes = max(1, min(max_bytes, _HARD_CAP_FETCH_MAX_BYTES))

        assert self.content_store is not None
        body = self.content_store.fetch(source_id)
        if not body:
            return ToolResult(
                tool_use_id=tool_call.tool_use_id,
                tool_name=tool_call.tool_name,
                success=False,
                content=(
                    f"No content found for source_id={source_id!r}. "
                    "Use action='search' to discover indexed source_ids."
                ),
                error_code="ctx_search.not_found",
            )

        encoded = body.encode("utf-8")
        full_bytes = len(encoded)
        truncated = full_bytes > max_bytes
        emit_body = (
            encoded[:max_bytes].decode("utf-8", errors="ignore")
            if truncated
            else body
        )
        emitted_bytes = len(emit_body.encode("utf-8"))

        if truncated:
            content_text = (
                f"Fetched {source_id} — truncated to {emitted_bytes} of "
                f"{full_bytes} bytes (max_bytes={max_bytes}). Increase "
                "max_bytes to see more.\n\n"
                f"{emit_body}"
            )
        else:
            content_text = (
                f"Fetched {source_id} — {full_bytes} bytes "
                f"(no truncation).\n\n{emit_body}"
            )

        return ToolResult(
            tool_use_id=tool_call.tool_use_id,
            tool_name=tool_call.tool_name,
            success=True,
            content=content_text,
            data={
                "action": "fetch",
                "source_id": source_id,
                "body": emit_body,
                "bytes": full_bytes,
                "emitted_bytes": emitted_bytes,
                "truncated": truncated,
                "max_bytes": max_bytes,
            },
        )


__all__ = ["CcxCtxSearchTool"]
