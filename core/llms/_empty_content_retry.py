"""Shared guard against silently-empty LLM completions.

Root cause (first observed with VolcCodingClient / doubao-seed-code, but shared
by the whole OpenAI-compatible client family): a reasoning-capable model can
spend an entire turn on hidden ``reasoning_content`` and return a message whose
visible ``content`` is ``None`` / ``""`` while ``finish_reason`` is still
``"stop"``. No exception is raised, so ``one_chat``'s tenacity ``@retry`` never
fires and the empty string propagates verbatim to every consumer (in ccx a
doc-mode investigator / lite agent then "succeeds" with an empty ``final_text``
— the "occasional empty output" symptom). Setting ``max_tokens`` does NOT bound
the reasoning trace on these endpoints, so a token budget cannot prevent it; the
reliable fix is to detect the empty visible content and re-issue.

This module centralises the retry *policy* so every client shares it; each
client supplies only its own ``create`` (issue one request) and ``extract``
(pull visible content + record meta/stats) callables.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from ..utils.log import logger

#: Default number of re-issues after an empty visible content. The LAST attempt
#: forces ``enable_thinking=False`` so a reasoning model must emit visible text.
DEFAULT_EMPTY_CONTENT_RETRIES = 3


def is_blank(content: Any) -> bool:
    """A completion is "empty" when its visible content is None or whitespace."""
    return content is None or not str(content).strip()


def reissue_completion_on_empty(
    *,
    create: Callable[[Dict[str, Any]], Any],
    extract: Callable[[Any], Optional[str]],
    kwargs: Dict[str, Any],
    enable_thinking: Optional[bool],
    retries: int = DEFAULT_EMPTY_CONTENT_RETRIES,
    client_name: str = "LLM",
    last_finish_reason: Any = None,
    last_reasoning: str = "",
) -> Optional[str]:
    """Re-issue a completion whose visible content came back empty.

    ``create(call_kwargs)`` performs one request and returns the raw completion;
    ``extract(completion)`` returns its visible content (may be None/"") and is
    responsible for recording finish_reason / reasoning / usage stats. The first
    non-empty content wins; on the FINAL attempt ``enable_thinking`` is forced
    off via a *copy* of ``kwargs`` (never mutating shared client state, so this
    is safe even if the client instance is shared across threads). Returns the
    last (still-empty) value if every attempt is empty — it never fabricates
    content, leaving a genuine model failure visible to the caller.
    """
    retries = max(1, int(retries))
    result: Optional[str] = None
    for attempt in range(retries):
        force_no_thinking = attempt == retries - 1
        call_kwargs = kwargs
        if force_no_thinking and enable_thinking is not None:
            call_kwargs = dict(kwargs)
            extra_body = dict(call_kwargs.get("extra_body") or {})
            extra_body["enable_thinking"] = False
            call_kwargs["extra_body"] = extra_body
        logger.warning(
            "%s: empty visible content (finish_reason=%s, reasoning=%d chars); "
            "re-issuing (attempt %d/%d%s)",
            client_name, last_finish_reason, len(last_reasoning or ""),
            attempt + 1, retries,
            ", thinking disabled" if force_no_thinking else "",
        )
        completion = create(call_kwargs)
        content = extract(completion)
        if not is_blank(content):
            return content
        result = content
    return result
