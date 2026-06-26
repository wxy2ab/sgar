"""Mid-turn steer inbox — thread-safe queue for caller-side guidance.

Borrowed from Reasonix's ``CacheFirstLoop.steer(text)`` + the
``MID_TURN_STEER_WRAPPER`` text. The problem: once
``CodeAgent.run_sync`` is mid-flight inside a long DAG (plan → spec →
agent), the caller has no in-band way to add a constraint short of
cancelling and restarting (which discards every node already finished).

A ``SteerInbox`` is a FIFO list with a lock. The caller pushes
strings from any thread via ``CodeAgent.push_steer(text)``; the
ccx-side ``_make_mode_tool`` wrapper drains the inbox right before
each subagent fn runs and prepends a wrapped block to the goal so the
LLM sees: "this is a constraint, not a new task".

Lifecycle: one ``SteerInbox`` per ``CodeAgent`` instance, reused
across ``run_sync`` calls. ``CodeAgent`` clears the inbox in the
``finally`` block of each run to avoid leaking unconsumed steers
across runs.

Concurrency (v1): drain-and-clear under a single lock. If multiple v5
nodes invoke the wrapper fn in parallel, the first to acquire the
lock takes all pending steers; siblings see an empty drain. This
matches Reasonix's per-turn semantics — a single steer is for the
"next node about to run", not a broadcast. If a broadcast variant is
ever wanted, switch to per-call sequence cursors without changing the
public API.
"""

from __future__ import annotations

import hashlib
import threading
import warnings
from dataclasses import dataclass, field

MAX_STEER_BODY_BYTES = 8192

_WRAPPER_EN_HEADER = "[MID-TURN STEER FROM USER]"
_WRAPPER_EN_INTRO = (
    "The following is supplemental guidance the user added while this run "
    "was in flight. Treat it as an additional constraint on the current "
    "node's goal — not as a replacement, not as a new task."
)
_WRAPPER_EN_FOOTER = "[END MID-TURN STEER]"

_WRAPPER_ZH_HEADER = "[用户中途追加的约束]"
_WRAPPER_ZH_INTRO = (
    "以下是用户在本次 run 进行中追加的补充指引。"
    "请把它视为对当前节点目标的额外约束 —— 不是替换原目标，也不是新任务。"
)
_WRAPPER_ZH_FOOTER = "[END]"

# Public (header, footer) marker pairs for consumers that need to
# recognize and skip an injected steer block inside a goal string
# (e.g. the SGAR command resolver must not treat steer text as a
# command). Kept here so the markers have a single source of truth.
STEER_BLOCK_MARKERS: tuple[tuple[str, str], ...] = (
    (_WRAPPER_EN_HEADER, _WRAPPER_EN_FOOTER),
    (_WRAPPER_ZH_HEADER, _WRAPPER_ZH_FOOTER),
)


@dataclass(slots=True)
class SteerInbox:
    """Thread-safe FIFO queue of pending steer texts."""

    _items: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def push(self, text: str) -> None:
        """Append a steer text. Empty / whitespace-only strings are
        dropped silently. Texts longer than ``MAX_STEER_BODY_BYTES`` are
        truncated (UTF-8 byte length) with a ``UserWarning`` so the
        caller can shorten on the next push.
        """
        if not text or not text.strip():
            return
        encoded = text.encode("utf-8")
        if len(encoded) > MAX_STEER_BODY_BYTES:
            warnings.warn(
                f"steer body truncated from {len(encoded)} to "
                f"{MAX_STEER_BODY_BYTES} bytes",
                UserWarning,
                stacklevel=2,
            )
            text = encoded[:MAX_STEER_BODY_BYTES].decode("utf-8", errors="ignore")
        with self._lock:
            self._items.append(text)

    def drain(self) -> list[str]:
        """Return all pending steers and clear the inbox. Atomic."""
        with self._lock:
            items = self._items
            self._items = []
        return items

    def clear(self) -> None:
        """Discard all pending steers without returning them."""
        with self._lock:
            self._items = []

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


def format_steer_block(items: list[str], language: str = "en") -> str:
    """Wrap a list of steer texts into a single block ready to prepend
    to a goal. Empty input returns the empty string so the caller can
    unconditionally do ``goal = block + goal`` without a branch.
    """
    if not items:
        return ""
    if language == "zh":
        header, intro, footer = _WRAPPER_ZH_HEADER, _WRAPPER_ZH_INTRO, _WRAPPER_ZH_FOOTER
    else:
        header, intro, footer = _WRAPPER_EN_HEADER, _WRAPPER_EN_INTRO, _WRAPPER_EN_FOOTER
    body = "\n\n".join(items)
    return f"{header}\n{intro}\n\n{body}\n\n{footer}\n"


def steer_payload_hash(items: list[str]) -> str:
    """Short stable hash over the steer items for event payloads.
    16 hex chars — enough to distinguish nearby steers in an event log
    without bloating payloads with the full text.
    """
    h = hashlib.sha256()
    for item in items:
        h.update(item.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


__all__ = [
    "MAX_STEER_BODY_BYTES",
    "STEER_BLOCK_MARKERS",
    "SteerInbox",
    "format_steer_block",
    "steer_payload_hash",
]
