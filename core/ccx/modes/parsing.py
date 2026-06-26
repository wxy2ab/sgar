"""Shared LLM-emitted-JSON parser for ccx mode runners.

Borrowed from Reasonix's ``repair/truncation.ts`` approach to handling
LLM output drift: when the model emits JSON, *something* parseable can
almost always be recovered, and when it can't, the right answer is a
caller-defined degraded structure — never an exception that bubbles up
and aborts a long DAG.

Before this util, ccx had six near-identical "strip the fence and
try ``json.loads``" implementations, each with its own failure
posture:

- ``plan.py:_parse_plan_response`` — returned an empty plan dict.
- ``spec.py:_parse_spec_response`` — line-for-line copy of plan.
- ``agent.py:_parse_agent_response`` — returned raw text as
  ``final_text`` (a smart degraded form).
- ``watch.py:_parse_plan_json`` — returned ``None``.
- ``structured_flow.py:_parse_task_list`` — returned ``[]``.
- ``llm_monitor.py:_parse_llm_response`` — returned ``None``.

Three of those swallowed errors and three returned sentinel falsy
values, and a parsing tweak (e.g. "tolerate ``` ``` `` `` fences with
the language tag missing") had to be applied to all six independently.

``parse_llm_json`` collapses the strategy into one place:

1. ``json.loads`` on the raw response — handles the common case where
   the model emitted clean JSON.
2. Strip a markdown ``` fence (with optional ``json`` language tag)
   and retry — handles the next-most-common case.
3. Find the first balanced ``{...}`` or ``[...]`` substring and retry
   — handles preamble like "Sure, here is the plan: { ... }".

Layer 3 is capped at 64 candidate openers per bracket kind. When the
cap is hit, the parser emits a dedicated warning and continues the
fallback cascade. ``expected_type`` only filters object-vs-array shape;
callers that require a specific top-level key should validate that key
after parsing.

If all three fail, or the parsed value is not an instance of
``expected_type`` (when provided), the caller's ``fallback_factory``
is invoked with the original response and its return value is yielded.

The util explicitly does NOT do partial-JSON repair (Reasonix's
truncation pillar): that requires bracket-matching and string-state
tracking that's an order of magnitude more code, and ``doc.py`` already
has its own heavier ``_robust_json_object`` that uses
``core.utils.json_from_text.extract_json_from_text`` for that. v1 of
this util keeps to the 80/20 strategy.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Match a fenced code block, optionally with a language tag (json /
# JSON / etc.). DOTALL so ``.`` spans newlines; IGNORECASE for the
# language tag.
_FENCE_RE = re.compile(
    r"```(?:[a-zA-Z]+)?\s*\n?(.*?)\n?```",
    re.DOTALL | re.IGNORECASE,
)
_MAX_BALANCED_CANDIDATES = 64


def parse_llm_json(
    response: str,
    *,
    schema_name: str,
    fallback_factory: Callable[[str], Any],
    expected_type: type | None = None,
) -> Any:
    """Extract a JSON structure from an LLM response, with a typed
    fallback when parsing fails or returns the wrong shape.

    ``response`` is the raw LLM text. ``schema_name`` is a label used
    only in log messages (``"plan"``, ``"spec"``, ``"task_list"``,
    ...). ``fallback_factory`` is called with the original response
    when no parse succeeds; it must return a valid degraded structure
    of the same shape the caller would otherwise consume — never raise.
    ``expected_type`` (e.g. ``dict`` or ``list``) is a soft schema
    check: when set, parse layers that produce a wrong-typed value are
    treated as failures and fall through to ``fallback_factory``.

    Returns whatever ``json.loads`` produces on success, or whatever
    ``fallback_factory`` produces on failure.
    """
    parsed = _try_parse_layers(response, expected_type=expected_type)
    if parsed is None:
        logger.warning(
            "parse_llm_json[%s]: all parse layers failed; "
            "falling back (raw len=%d)",
            schema_name, len(response or ""),
        )
        return fallback_factory(response)
    if expected_type is not None and not isinstance(parsed, expected_type):
        logger.warning(
            "parse_llm_json[%s]: parsed value is %s, expected %s; "
            "falling back",
            schema_name, type(parsed).__name__, expected_type.__name__,
        )
        return fallback_factory(response)
    return parsed


def _try_parse_layers(response: str, *, expected_type: type | None = None) -> Any:
    """Three-layer recovery cascade. Returns parsed value or ``None``
    if every layer failed. ``None`` here is distinct from the JSON
    value ``null`` only by the absence of any parse success — callers
    should treat both as "couldn't make sense of it".

    ``expected_type`` orders layer 3's bracket search: when the caller
    expects a ``list``, ``[...]`` is tried before ``{...}`` so that an
    array wrapped in prose isn't mis-extracted as its first element.
    """
    if not response or not response.strip():
        return None
    s = response.strip()

    # Layer 1: raw response is already JSON.
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        pass

    # Layer 2: strip markdown fences and parse the inside. Try every
    # fence; a prose fence or wrong-shaped JSON fence must not hide a
    # later valid JSON fence.
    for m in _FENCE_RE.finditer(s):
        candidate = m.group(1).strip()
        if candidate:
            try:
                parsed = json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                pass
            else:
                if expected_type is None or isinstance(parsed, expected_type):
                    return parsed

    # Layer 3: find the first balanced top-level {...} or [...] and
    # try that. Handles preamble like "Sure, here you go: { ... }".
    # Bracket order follows ``expected_type``: a caller expecting a
    # list gets ``[`` tried first — otherwise an array-with-preamble
    # would match the first element's ``{...}`` and be mis-typed.
    pairs = (("{", "}"), ("[", "]"))
    if expected_type is list:
        pairs = (("[", "]"), ("{", "}"))
    for open_ch, close_ch in pairs:
        start = -1
        attempts = 0
        while True:
            start = s.find(open_ch, start + 1)
            if start == -1:
                break
            attempts += 1
            if attempts > _MAX_BALANCED_CANDIDATES:
                logger.warning(
                    "parse_llm_json: balanced candidate scan cap (%d) "
                    "reached for %r opener; continuing fallback cascade",
                    _MAX_BALANCED_CANDIDATES,
                    open_ch,
                )
                break
            candidate = _extract_balanced_at(s, open_ch, close_ch, start=start)
            if candidate is None:
                continue
            try:
                parsed = json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                continue
            if expected_type is None or isinstance(parsed, expected_type):
                return parsed
    return None


def _extract_balanced_at(
    s: str,
    open_ch: str,
    close_ch: str,
    *,
    start: int,
) -> str | None:
    """Return a balanced substring while ignoring brackets in strings."""
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(s)):
        c = s[i]
        if escaped:
            escaped = False
            continue
        if c == "\\" and in_string:
            escaped = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


__all__ = ["parse_llm_json"]
