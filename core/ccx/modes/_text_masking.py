"""Shared text masking helpers for command and marker parsing."""

from __future__ import annotations

import logging


_MASK_CHAR = "x"


def _mask_char(char: str, mask_char: str) -> str:
    if char in "\r\n":
        return char
    return mask_char


def _mask_range(chars: list[str], start: int, stop: int, mask_char: str) -> None:
    for pos in range(start, stop):
        chars[pos] = _mask_char(chars[pos], mask_char)


def mask_fenced_segments(
    text: str,
    *,
    logger: logging.Logger | None = None,
    mask_char: str = _MASK_CHAR,
) -> str:
    """Mask Markdown fenced blocks while preserving length and newlines.

    ``mask_char`` must be non-whitespace so line-anchored regexes using
    ``^[ \t]*`` do not start accepting markers that were merely preceded by a
    masked inline fence.
    """
    chars = list(text)
    idx = 0
    while idx < len(chars):
        if text.startswith("```", idx):
            end = text.find("```", idx + 3)
            unclosed = end == -1
            stop = len(chars) if unclosed else end + 3
            _mask_range(chars, idx, stop, mask_char)
            if unclosed and logger is not None:
                logger.info("masking unterminated fenced block to end of text")
            idx = stop
            continue
        idx += 1
    return "".join(chars)


def mask_quoted_segments(
    text: str,
    *,
    mask_char: str = _MASK_CHAR,
) -> str:
    """Mask shell-style quoted spans while preserving length and newlines."""
    chars = list(text)
    idx = 0
    quote: str | None = None
    quote_start = 0
    while idx < len(chars):
        char = text[idx]
        if quote is None:
            if char in {"'", '"'}:
                quote = char
                quote_start = idx
            idx += 1
            continue
        if quote == '"' and char == "\\" and idx + 1 < len(chars):
            idx += 2
            continue
        if char == quote:
            _mask_range(chars, quote_start, idx + 1, mask_char)
            quote = None
        idx += 1
    if quote is not None:
        _mask_range(chars, quote_start, len(chars), mask_char)
    return "".join(chars)


__all__ = [
    "mask_fenced_segments",
    "mask_quoted_segments",
]
