from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

TODOS_LEDGER_FILENAME = "todos.jsonl"


def todos_ledger_path(session_dir: str | Path) -> Path:
    """Path of the per-session todo ledger, co-located with the session's other
    on-disk artifacts (``messages.jsonl``, ``turns.jsonl``, ``session.json``)."""
    return Path(session_dir) / TODOS_LEDGER_FILENAME


def _normalize(todos: object) -> list[dict[str, str]]:
    if not isinstance(todos, list):
        return []
    out: list[dict[str, str]] = []
    for item in todos:
        if not isinstance(item, dict) or not item.get("content"):
            continue
        out.append(
            {
                "content": str(item.get("content", "")),
                "status": str(item.get("status", "pending")),
            }
        )
    return out


def append_todos(session_dir: str | Path, todos: list[dict[str, str]]) -> None:
    """Append one full-snapshot record of the todo list to the session ledger.

    The agent's fine-grained progress is flushed to disk the moment ``todo_write``
    runs, so a process killed mid-drive — the documented wedge failure mode —
    still leaves the completed sub-steps on disk for the next run to replay.

    Best-effort by construction (the same discipline the goal ledger applies in
    ``governed_goal._GoalLedger._append``): an IO failure is logged and swallowed
    so a progress-ledger write can never crash a tool call. Each line is a
    COMPLETE snapshot of the merged todo set, so replay is just "take the last
    parseable record" — no delta reconstruction.
    """
    path = todos_ledger_path(session_dir)
    record = {
        "ts": time.time(),
        "op": "todo_write",
        "todos": _normalize(todos),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except Exception:  # noqa: BLE001 — a progress-ledger write must never crash a tool
        logger.warning("cc todo: failed to append todos ledger record", exc_info=True)


def load_todos(session_dir: str | Path) -> list[dict[str, str]]:
    """Replay the session todo ledger to its latest snapshot.

    Returns the ``todos`` of the LAST successfully-parsed record (each record is
    a full snapshot), or ``[]`` if the ledger is absent/unreadable/empty. A torn
    final line from a crash mid-append is tolerated: every line is parsed and the
    newest that decodes wins, so a half-written tail is simply skipped.
    """
    path = todos_ledger_path(session_dir)
    latest: list[dict[str, str]] = []
    try:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except (ValueError, TypeError):
                    continue
                if isinstance(record, dict) and isinstance(record.get("todos"), list):
                    latest = _normalize(record["todos"])
    except Exception:  # noqa: BLE001 — reload is best-effort, never blocks a resume
        logger.warning("cc todo: failed to load todos ledger", exc_info=True)
        return []
    return latest
