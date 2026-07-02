"""Read-only watcher CLI for ccx runs.

Inspects the SQLite runtime DB that EngineV5 persists under
``<cwd>/.ccx/runtime/runtime.db`` and prints either a list of runs, a
node table for a given run, or a single-node detail view. With
``--follow`` it reprints snapshots on an interval.

The watcher opens the DB with ``mode=ro`` via the URI form so it cannot
write, and so the engine's WAL writers can keep going untouched. It
deliberately does NOT import any v5 store classes — it speaks raw SQL
against the schema, so a crashed or partially-initialised engine still
yields useful output. ``NodeState`` is imported from
``core.deepstack_v5.types`` only to validate ``--state`` values.

Two viewing modes:

* **Snapshot mode** (default) — re-renders the runs / nodes / single-node
  view on every tick. ``--follow`` reprints every ``--interval`` seconds.

* **Tail mode** (``--tail``) — cursor-tails the ``events`` table by
  ``sequence``. Each new row becomes one rendered line. The v5 EventBus
  already persists every published event to that table inside the same
  transaction that mutates ``nodes`` / ``leases``, so this is the closest
  thing to push-style observability available out-of-process without
  changing the engine. Without ``--follow`` it drains from ``--since``
  to the current max and exits; with ``--follow`` it keeps polling.

Usage::

    python -m core.ccx.watch [--cwd PATH]
    python -m core.ccx.watch --cwd PATH --run-id RUN_ID
    python -m core.ccx.watch --cwd PATH --run-id RUN_ID --follow
    python -m core.ccx.watch --cwd PATH --run-id RUN_ID --state RUNNING --state FAILED
    python -m core.ccx.watch --cwd PATH --run-id RUN_ID --node-id N
    python -m core.ccx.watch --cwd PATH --format json

    # Tail the event stream for a run from where it stands now, forever:
    python -m core.ccx.watch --cwd PATH --run-id RUN_ID --tail --follow

    # Replay every event for a run from the start, then exit:
    python -m core.ccx.watch --cwd PATH --run-id RUN_ID --tail --since 0

    # Filter by kind prefix (repeatable) and emit JSONL:
    python -m core.ccx.watch --cwd PATH --run-id RUN_ID --tail --follow \\
        --kind node. --format json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

_TAIL_EVENTS_BATCH_LIMIT = 1000
_FALLBACK_NODE_STATES = {
    "pending",
    "ready",
    "running",
    "succeeded",
    "failed",
    "blocked",
    "approval_hang",
    "timer_hang",
    "abandoned",
    "skipped",
    "cancelled",
}


# --------------------------------------------------------------------------- #
# Connection
# --------------------------------------------------------------------------- #


def resolve_db_path(cwd: str | Path) -> Path:
    return Path(cwd) / ".ccx" / "runtime" / "runtime.db"


def connect_ro(db_path: Path) -> sqlite3.Connection:
    """Open the runtime DB read-only.

    Uses the URI form so concurrent writers (the running engine) are
    unaffected. WAL mode (set by the engine on first open) lets us read
    while writes proceed.
    """
    if not db_path.exists():
        raise FileNotFoundError(
            f"runtime DB not found at {db_path} — "
            "no ccx run has been started under this cwd yet."
        )
    uri = f"file:{db_path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row
    # Defence in depth: even if mode=ro got dropped somehow, query_only
    # blocks writes at the statement level.
    conn.execute("PRAGMA query_only = 1")
    return conn


# --------------------------------------------------------------------------- #
# Row helpers (schema-tolerant)
# --------------------------------------------------------------------------- #


def _row_get(row: sqlite3.Row, key: str, default: Any = None) -> Any:
    try:
        return row[key]
    except (IndexError, KeyError):
        return default


def _loads(value: Any) -> Any:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _ms_to_iso(ms: Any) -> str:
    if not isinstance(ms, (int, float)) or ms <= 0:
        return ""
    return time.strftime("%m-%d %H:%M:%S", time.localtime(ms / 1000))


def _truncate(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    if width <= 1:
        return value[:width]
    return value[: width - 1] + "…"


def _now_ms() -> int:
    return int(time.time() * 1000)


# --------------------------------------------------------------------------- #
# Degraded-completion honest signal
# --------------------------------------------------------------------------- #


def degraded_completion(
    status: Any, state_counts: dict[str, int] | None
) -> dict[str, int] | None:
    """Run reports a success-shaped terminal status yet carries failure nodes.

    v5 stamps ``status='completed'`` on a run that finished best-effort —
    ``graph.all_terminal()`` with at least one ``succeeded`` node — even when
    some nodes were ``abandoned`` (engine.py ``_build_verdict``). That is a
    *partial / degraded* outcome the result layer already flags as
    ``session_snapshot['abandoned_warning']`` (api.py). The monitors (watch /
    report) are the operator's primary truth window, and their dominant signal
    — the ``completed`` status cell / green ``s-completed`` badge — reads
    identically to a clean 100%-success run; the degradation is demoted to
    secondary count columns / badges that are easy to skim past.

    This returns ``{'abandoned': n, 'failed': m}`` when a ``completed`` run
    carries abandoned or failed nodes, so callers can render a prominent
    run-level banner; otherwise ``None`` (so clean / non-completed runs render
    byte-for-byte as before).

    Predicate note: ``completed`` is the only success-shaped ``RunStatus``, so
    only it can mask degradation as success. We key on ``abandoned > 0 OR
    failed > 0``. The reachable production case is ``abandoned > 0`` (FAILED is
    not in ``TERMINAL_NODE_STATES``, so a live v5 ``completed`` run cannot carry
    a ``failed`` node); ``failed`` is included as harmless defence-in-depth so
    the monitor stays honest against a partial / crashed DB write.
    """
    if str(status or "").strip().lower() != "completed":
        return None
    counts = state_counts or {}
    abandoned = int(counts.get("abandoned", 0) or 0)
    failed = int(counts.get("failed", 0) or 0)
    if abandoned <= 0 and failed <= 0:
        return None
    return {"abandoned": abandoned, "failed": failed}


def _degraded_phrase(degraded: dict[str, int]) -> str:
    """``{'abandoned': 2, 'failed': 1}`` -> ``"2 abandoned, 1 failed"``."""
    parts: list[str] = []
    if degraded.get("abandoned"):
        parts.append(f"{degraded['abandoned']} abandoned")
    if degraded.get("failed"):
        parts.append(f"{degraded['failed']} failed")
    return ", ".join(parts)


# --------------------------------------------------------------------------- #
# Queries
# --------------------------------------------------------------------------- #


def _query(
    conn: sqlite3.Connection,
    sql: str,
    params: Sequence[Any] = (),
) -> list[sqlite3.Row]:
    """Run a SELECT, swallowing OperationalErrors so the watcher keeps
    rendering even if the schema has drifted.
    """
    try:
        return list(conn.execute(sql, params).fetchall())
    except sqlite3.OperationalError as exc:
        print(f"warning: query failed ({exc}); skipping", file=sys.stderr)
        return []


def list_runs(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = _query(
        conn,
        "SELECT * FROM runs ORDER BY updated_at_ms DESC",
    )
    out: list[dict[str, Any]] = []
    for r in rows:
        run_id = _row_get(r, "run_id", "")
        counts = _state_counts(conn, run_id)
        total = sum(counts.values())
        out.append(
            {
                "run_id": run_id,
                "status": _row_get(r, "status", ""),
                "goal": _row_get(r, "goal", ""),
                "created_at_ms": _row_get(r, "created_at_ms"),
                "updated_at_ms": _row_get(r, "updated_at_ms"),
                "state_counts": counts,
                "node_total": total,
                "node_succeeded": counts.get("succeeded", 0),
                "node_failed": counts.get("failed", 0),
                "node_abandoned": counts.get("abandoned", 0),
            }
        )
    return out


def _state_counts(conn: sqlite3.Connection, run_id: str) -> dict[str, int]:
    rows = _query(
        conn,
        "SELECT state, COUNT(*) AS n FROM nodes WHERE run_id = ? GROUP BY state",
        (run_id,),
    )
    return {_row_get(r, "state", ""): int(_row_get(r, "n", 0) or 0) for r in rows}


def get_run(conn: sqlite3.Connection, run_id: str) -> dict[str, Any] | None:
    rows = _query(conn, "SELECT * FROM runs WHERE run_id = ?", (run_id,))
    if not rows:
        return None
    r = rows[0]
    return {
        "run_id": _row_get(r, "run_id", ""),
        "status": _row_get(r, "status", ""),
        "goal": _row_get(r, "goal", ""),
        "created_at_ms": _row_get(r, "created_at_ms"),
        "updated_at_ms": _row_get(r, "updated_at_ms"),
    }


def list_nodes(
    conn: sqlite3.Connection,
    run_id: str,
    state_filter: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    rows = _query(
        conn,
        "SELECT * FROM nodes WHERE run_id = ? ORDER BY created_at_ms ASC",
        (run_id,),
    )
    leases = {l["node_id"]: l for l in list_leases(conn, run_id)}
    now = _now_ms()
    states_lc = (
        {s.lower() for s in state_filter} if state_filter else None
    )
    out: list[dict[str, Any]] = []
    for r in rows:
        state = _row_get(r, "state", "") or ""
        if states_lc is not None and state.lower() not in states_lc:
            continue
        spec = _loads(_row_get(r, "spec_json")) or {}
        attempts = _loads(_row_get(r, "attempts_json")) or []
        failure = _loads(_row_get(r, "failure_json")) or {}
        node_id = _row_get(r, "node_id", "")
        lease = leases.get(node_id)
        lease_info: dict[str, Any] | None = None
        if lease is not None:
            heartbeat = lease.get("heartbeat_at_ms") or 0
            expires = lease.get("expires_at_ms") or 0
            age_sec = max(0, (now - heartbeat) // 1000) if heartbeat else None
            lease_info = {
                "worker_id": lease.get("worker_id"),
                "heartbeat_at_ms": heartbeat,
                "expires_at_ms": expires,
                "age_sec": age_sec,
                "expired": bool(expires) and expires < now,
            }
        deps_raw = spec.get("depends_on") if isinstance(spec, dict) else None
        deps: list[str] = list(deps_raw) if isinstance(deps_raw, list) else []
        tool = spec.get("tool") if isinstance(spec, dict) else None
        out.append(
            {
                "node_id": node_id,
                "state": state,
                "attempts": len(attempts) if isinstance(attempts, list) else 0,
                "tool": tool or "",
                "deps": deps,
                "last_failure_kind": (
                    failure.get("kind") if isinstance(failure, dict) else ""
                )
                or "",
                "lease": lease_info,
                "created_at_ms": _row_get(r, "created_at_ms"),
                "updated_at_ms": _row_get(r, "updated_at_ms"),
            }
        )
    return out


def list_leases(conn: sqlite3.Connection, run_id: str) -> list[dict[str, Any]]:
    rows = _query(
        conn,
        "SELECT * FROM leases WHERE run_id = ?",
        (run_id,),
    )
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "lease_id": _row_get(r, "lease_id", ""),
                "node_id": _row_get(r, "node_id", ""),
                "worker_id": _row_get(r, "worker_id", ""),
                "granted_at_ms": _row_get(r, "granted_at_ms"),
                "heartbeat_at_ms": _row_get(r, "heartbeat_at_ms"),
                "expires_at_ms": _row_get(r, "expires_at_ms"),
            }
        )
    return out


def node_detail(
    conn: sqlite3.Connection, run_id: str, node_id: str
) -> dict[str, Any] | None:
    rows = _query(
        conn,
        "SELECT * FROM nodes WHERE run_id = ? AND node_id = ?",
        (run_id, node_id),
    )
    if not rows:
        return None
    r = rows[0]
    return {
        "run_id": run_id,
        "node_id": node_id,
        "state": _row_get(r, "state", ""),
        "spec": _loads(_row_get(r, "spec_json")),
        "attempts": _loads(_row_get(r, "attempts_json")),
        "result": _loads(_row_get(r, "result_json")),
        "failure": _loads(_row_get(r, "failure_json")),
        "created_at_ms": _row_get(r, "created_at_ms"),
        "updated_at_ms": _row_get(r, "updated_at_ms"),
        "events": node_events(conn, run_id, node_id),
    }


def current_max_sequence(
    conn: sqlite3.Connection, *, run_id: str | None
) -> int:
    """Return the largest ``events.sequence`` currently stored.

    Used as the default cursor origin for ``--tail`` so a fresh tail
    only sees events that happen *after* the watcher starts. Pass
    ``run_id`` to scope to a single run; ``None`` queries the whole DB.
    """
    if run_id is None:
        rows = _query(conn, "SELECT MAX(sequence) AS m FROM events")
    else:
        rows = _query(
            conn,
            "SELECT MAX(sequence) AS m FROM events WHERE run_id = ?",
            (run_id,),
        )
    if not rows:
        return 0
    val = _row_get(rows[0], "m", 0)
    return int(val or 0)


def tail_events(
    conn: sqlite3.Connection,
    *,
    after_sequence: int,
    run_id: str | None = None,
    kind_prefixes: Sequence[str] | None = None,
    node_id: str | None = None,
    limit: int = _TAIL_EVENTS_BATCH_LIMIT,
    return_scan_sequence: bool = False,
    return_scan_count: bool = False,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], int] | tuple[list[dict[str, Any]], int, int]:
    """Read the ``events`` table by sequence cursor.

    Returns events strictly after ``after_sequence``, ascending. Caller
    advances its own cursor to ``events[-1]['sequence']`` after each
    drain. The query is bounded by ``limit`` so one tick can't pull a
    runaway batch — long backlogs spill across ticks at ``--interval``
    cadence.

    Kind filter is a prefix match (one of ``kind_prefixes`` must be a
    prefix of ``event.kind``); pushed into SQL with a ``LIKE`` per
    prefix joined by ``OR`` to keep it index-friendly. Node filter is
    applied in Python after parsing payload_json — the events table is
    not indexed on payload contents, and checking the decoded
    ``node_id`` field (as ``node_events`` also does) is independent of
    how the writer serialized the payload.

    When requested, ``return_scan_count`` reports SQL rows scanned before
    Python-side node filtering, not the number of returned events.
    """
    clauses: list[str] = ["sequence > ?"]
    params: list[Any] = [int(after_sequence)]
    if run_id is not None:
        clauses.append("run_id = ?")
        params.append(run_id)
    if kind_prefixes:
        ors: list[str] = []
        for prefix in kind_prefixes:
            ors.append("kind LIKE ?")
            params.append(f"{prefix}%")
        clauses.append("(" + " OR ".join(ors) + ")")
    sql = (
        "SELECT sequence, run_id, kind, created_at_ms, payload_json "
        "FROM events WHERE " + " AND ".join(clauses)
        + " ORDER BY sequence ASC LIMIT ?"
    )
    params.append(int(limit))
    rows = _query(conn, sql, params)
    out: list[dict[str, Any]] = []
    scanned_sequence = int(after_sequence)
    for r in rows:
        scanned_sequence = max(scanned_sequence, int(_row_get(r, "sequence") or 0))
        payload = _loads(_row_get(r, "payload_json")) or {}
        if node_id is not None:
            evt_node_id = (
                payload.get("node_id") if isinstance(payload, dict) else None
            )
            if evt_node_id != node_id:
                continue
        out.append(
            {
                "sequence": _row_get(r, "sequence"),
                "run_id": _row_get(r, "run_id", ""),
                "kind": _row_get(r, "kind", ""),
                "created_at_ms": _row_get(r, "created_at_ms"),
                "payload": payload,
            }
        )
    if return_scan_sequence and return_scan_count:
        return out, scanned_sequence, len(rows)
    if return_scan_sequence:
        return out, scanned_sequence
    if return_scan_count:
        return out, len(rows)
    return out


def node_events(
    conn: sqlite3.Connection,
    run_id: str,
    node_id: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Most recent events whose decoded payload ``node_id`` matches.

    Filters on the parsed payload — same as ``tail_events`` — instead of
    a LIKE pattern shaped after one particular serialization: the v5
    EventStore writes payloads with ``json.dumps``' default separators
    (``"node_id": "n1"``, WITH a space after the colon), so a pattern
    baked to the compact variant silently matches nothing against a real
    EngineV5 DB. A loose substring LIKE prefilters rows in SQL so only
    plausible candidates get decoded; it is skipped when JSON string
    escaping would alter the id (the literal substring might then not
    appear in payload_json). The decoded comparison is authoritative.
    """
    sql = (
        "SELECT sequence, kind, created_at_ms, payload_json "
        "FROM events WHERE run_id = ?"
    )
    params: list[Any] = [run_id]
    if json.dumps(node_id, ensure_ascii=False)[1:-1] == node_id:
        sql += " AND payload_json LIKE ?"
        params.append(f"%{node_id}%")
    sql += " ORDER BY sequence DESC"
    out: list[dict[str, Any]] = []
    try:
        for r in conn.execute(sql, params):
            payload = _loads(_row_get(r, "payload_json"))
            if not isinstance(payload, dict) or payload.get("node_id") != node_id:
                continue
            out.append(
                {
                    "sequence": _row_get(r, "sequence"),
                    "kind": _row_get(r, "kind", ""),
                    "created_at_ms": _row_get(r, "created_at_ms"),
                    "payload": payload,
                }
            )
            if len(out) >= limit:
                break
    except sqlite3.OperationalError as exc:
        print(f"warning: query failed ({exc}); skipping", file=sys.stderr)
        return []
    out.reverse()  # ascending for human reading
    return out


# --------------------------------------------------------------------------- #
# Rendering — table mode
# --------------------------------------------------------------------------- #


def _render_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    aligns: Sequence[str] | None = None,
) -> str:
    if not rows:
        widths = [len(h) for h in headers]
    else:
        widths = [
            max(len(headers[i]), *(len(str(r[i])) for r in rows))
            for i in range(len(headers))
        ]
    aligns = aligns or ["l"] * len(headers)

    def fmt_cell(text: str, width: int, align: str) -> str:
        s = str(text)
        if align == "r":
            return s.rjust(width)
        return s.ljust(width)

    lines = [
        "  ".join(fmt_cell(h, widths[i], aligns[i]) for i, h in enumerate(headers)),
        "  ".join("-" * widths[i] for i in range(len(headers))),
    ]
    for row in rows:
        lines.append(
            "  ".join(
                fmt_cell(row[i], widths[i], aligns[i]) for i in range(len(headers))
            )
        )
    return "\n".join(lines)


def _runs_degraded_banner(runs: Sequence[dict[str, Any]]) -> str:
    """Prominent leading banner enumerating COMPLETED-but-degraded runs.

    Empty string when no run is degraded, so a list of clean runs renders
    byte-for-byte as before.
    """
    flagged: list[tuple[str, dict[str, int]]] = []
    for r in runs:
        degraded = degraded_completion(r.get("status"), r.get("state_counts") or {})
        if degraded:
            flagged.append((r.get("run_id") or "", degraded))
    if not flagged:
        return ""
    lines = [
        f"⚠ DEGRADED: {len(flagged)} run(s) reported 'completed' but carry "
        "abandoned/failed nodes (partial/best-effort, NOT a clean success):"
    ]
    for run_id, degraded in flagged:
        lines.append(f"    {_truncate(run_id, 24)}: {_degraded_phrase(degraded)}")
    return "\n".join(lines) + "\n\n"


def render_runs_table(runs: Sequence[dict[str, Any]]) -> str:
    if not runs:
        return "no runs found"
    headers = [
        "run_id",
        "status",
        "created",
        "updated",
        "succ/total",
        "failed",
        "aband",
        "goal",
    ]
    rows: list[list[str]] = []
    for r in runs:
        succ_total = f"{r['node_succeeded']}/{r['node_total']}"
        goal = (r.get("goal") or "").replace("\n", " ")
        rows.append(
            [
                _truncate(r.get("run_id") or "", 24),
                _truncate(r.get("status") or "", 10),
                _ms_to_iso(r.get("created_at_ms")),
                _ms_to_iso(r.get("updated_at_ms")),
                succ_total,
                str(r.get("node_failed", 0)),
                str(r.get("node_abandoned", 0)),
                _truncate(goal, 60),
            ]
        )
    aligns = ["l", "l", "l", "l", "r", "r", "r", "l"]
    return _runs_degraded_banner(runs) + _render_table(headers, rows, aligns=aligns)


def _format_deps(deps: Sequence[str]) -> str:
    if not deps:
        return ""
    if len(deps) <= 3:
        return ",".join(deps)
    return f"{','.join(deps[:3])}+{len(deps) - 3}"


def _format_lease(lease: dict[str, Any] | None) -> str:
    if not lease:
        return ""
    age = lease.get("age_sec")
    age_str = f"{age}s" if isinstance(age, int) else "?"
    if lease.get("expired"):
        return f"EXPIRED({age_str})"
    return age_str


def render_nodes_table(
    run: dict[str, Any] | None,
    nodes: Sequence[dict[str, Any]],
    *,
    state_counts: dict[str, int] | None = None,
    leases: Sequence[dict[str, Any]] | None = None,
) -> str:
    if run is None:
        return "run not found"
    headers = ["node_id", "state", "att", "tool", "deps", "fail", "lease"]
    rows: list[list[str]] = []
    for n in nodes:
        rows.append(
            [
                _truncate(n.get("node_id") or "", 30),
                _truncate(n.get("state") or "", 14),
                str(n.get("attempts", 0)),
                _truncate(n.get("tool") or "", 22),
                _truncate(_format_deps(n.get("deps") or []), 22),
                _truncate(n.get("last_failure_kind") or "", 18),
                _format_lease(n.get("lease")),
            ]
        )
    aligns = ["l", "l", "r", "l", "l", "l", "l"]
    body = _render_table(headers, rows, aligns=aligns)

    # Summary line
    counts = state_counts or {}
    counts_str = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "-"
    leases = list(leases or [])
    now = _now_ms()
    expired = sum(
        1
        for l in leases
        if (l.get("expires_at_ms") or 0) and (l.get("expires_at_ms") or 0) < now
    )
    active = len(leases) - expired
    degraded = degraded_completion(run.get("status"), counts)
    degraded_tag = (
        f" ⚠ DEGRADED ({_degraded_phrase(degraded)})" if degraded else ""
    )
    summary = (
        f"\nRun {run.get('run_id')} [{run.get('status')}]{degraded_tag}: {counts_str} "
        f"| leases: {active} active, {expired} expired"
    )
    return body + summary


_EVENT_SUMMARY_WIDTH = 80


def _event_summary(kind: str, payload: dict[str, Any]) -> str:
    """One-line summary string for the kinds the v5 EventBus publishes.

    Falls back to a compact key=value join when the kind is unknown so
    nothing the engine emits ends up rendered as a blank line.
    """
    if not isinstance(payload, dict):
        return _truncate(str(payload), _EVENT_SUMMARY_WIDTH)
    if kind == "node.created":
        spec = payload.get("spec") or {}
        tool = spec.get("tool") if isinstance(spec, dict) else None
        return _truncate(f"tool={tool or '?'}", _EVENT_SUMMARY_WIDTH)
    if kind == "node.completed":
        summary = payload.get("result_summary") or ""
        body = str(summary).replace("\n", " ")
        return _truncate(f"OK {body}".strip(), _EVENT_SUMMARY_WIDTH)
    if kind == "ccx.governance.verdict":
        # Run-level governance verdict (emitted at the outermost run boundary
        # when CCX_EMIT_GOVERNANCE_EVENTS is on). Render the run-level truth a
        # node-level "all green" view would otherwise hide.
        passed = payload.get("passed")
        if passed is True:
            head = "governance PASS"
        elif passed is False:
            head = "governance NOT-PASSED"
        else:
            head = "governance ungoverned"
        bits = [head]
        for label, key in (
            ("contract", "contract_verdict"),
            ("run_audit", "run_audit_verdict"),
            ("goal", "goal_verdict"),
        ):
            sub = payload.get(key)
            if isinstance(sub, dict):
                mark = "ok" if sub.get("passed") else "NO"
                reason = sub.get("stop_reason") or sub.get("status") or "?"
                bits.append(f"{label}={mark}/{reason}")
        if payload.get("abandoned_warning"):
            bits.append("DEGRADED")
        return _truncate(" ".join(bits), _EVENT_SUMMARY_WIDTH)
    if kind.startswith("replan."):
        added = payload.get("added") or []
        removed = payload.get("removed")
        if removed is None:
            removed = payload.get("skipped_existing") or []
        added_n = len(added) if isinstance(added, list) else added
        removed_n = len(removed) if isinstance(removed, list) else removed
        return f"+{added_n} -{removed_n}"
    if kind == "compaction.completed":
        snapshot_id = payload.get("snapshot_id") or ""
        trigger = payload.get("triggered_by") or ""
        return _truncate(
            f"snapshot={snapshot_id} trigger={trigger}".strip(),
            _EVENT_SUMMARY_WIDTH,
        )
    if kind == "budget.warning":
        msg = payload.get("message") or payload.get("reason") or ""
        return _truncate(str(msg).replace("\n", " "), _EVENT_SUMMARY_WIDTH)
    # cc.* kinds published by the cc → v5 event bridge. Each one
    # surfaces a different facet of in-agent LLM activity, so the
    # summary picks the most useful field per kind.
    if kind == "cc.tool_use":
        tool_name = payload.get("tool_name") or "?"
        # The bridge pulls one distinctive arg ("command", "pattern",
        # "path", "query", "action", "mode"); render it inline so the
        # watcher can see *what* the LLM is asking the tool to do, not
        # just *which* tool.
        for hint in ("command", "pattern", "path", "query", "action", "mode"):
            if hint in payload:
                return _truncate(
                    f"-> {tool_name}({hint}={payload[hint]})",
                    _EVENT_SUMMARY_WIDTH,
                )
        keys = payload.get("arg_keys") or []
        if isinstance(keys, list) and keys:
            return _truncate(
                f"-> {tool_name}({','.join(keys[:3])})",
                _EVENT_SUMMARY_WIDTH,
            )
        return _truncate(f"-> {tool_name}", _EVENT_SUMMARY_WIDTH)
    if kind in {"cc.tool_completed", "cc.tool_failed"}:
        ok = payload.get("success")
        marker = "OK" if ok else "FAIL"
        tool_name = payload.get("tool_name") or "?"
        err = payload.get("error_code")
        dur = payload.get("duration_ms")
        bits = [f"{marker} {tool_name}"]
        if err:
            bits.append(f"err={err}")
        if isinstance(dur, (int, float)) and dur:
            bits.append(f"{int(dur)}ms")
        return _truncate(" ".join(bits), _EVENT_SUMMARY_WIDTH)
    if kind == "cc.tool_result":
        tool_name = payload.get("tool_name") or "?"
        preview = (payload.get("preview") or "").replace("\n", " ")
        return _truncate(f"<- {tool_name} {preview}", _EVENT_SUMMARY_WIDTH)
    if kind == "cc.assistant_text":
        preview = (payload.get("preview") or "").replace("\n", " ")
        chars = payload.get("chars")
        if isinstance(chars, int) and chars > 0:
            return _truncate(f"[{chars}ch] {preview}", _EVENT_SUMMARY_WIDTH)
        return _truncate(preview, _EVENT_SUMMARY_WIDTH)
    if kind == "cc.turn_failed":
        reason = (payload.get("reason") or "").replace("\n", " ")
        err = payload.get("error_code")
        return _truncate(
            f"FAIL {err or ''} {reason}".strip(), _EVENT_SUMMARY_WIDTH
        )
    if kind == "cc.turn_completed":
        return "turn end"
    # Generic fallback: top-level key=value, skipping verbose fields.
    skip = {"spec", "stack", "traceback", "raw"}
    bits = []
    for k, v in payload.items():
        if k in skip:
            continue
        if isinstance(v, (dict, list)):
            v_str = f"{type(v).__name__}({len(v)})"
        else:
            v_str = str(v).replace("\n", " ")
        bits.append(f"{k}={_truncate(v_str, 24)}")
        if len(bits) >= 4:
            break
    return _truncate(" ".join(bits), _EVENT_SUMMARY_WIDTH)


def render_event_line(event: dict[str, Any]) -> str:
    """Render one event from the events table as a single text line.

    Layout (column widths tuned to fit a typical 120-col terminal)::

        seq  | mm-dd HH:MM:SS | KIND                 | run | node | summary

    ``run`` and ``node`` are truncated to 12/16 chars so the summary
    column gets the bulk of the space.
    """
    seq = event.get("sequence", "?")
    kind = event.get("kind", "")
    payload = event.get("payload") or {}
    node_id = (
        payload.get("node_id")
        if isinstance(payload, dict)
        else None
    ) or ""
    return (
        f"{str(seq).rjust(6)}  "
        f"{_ms_to_iso(event.get('created_at_ms'))}  "
        f"{kind.ljust(22)[:22]}  "
        f"{_truncate(str(event.get('run_id') or ''), 12).ljust(12)}  "
        f"{_truncate(str(node_id), 16).ljust(16)}  "
        f"{_event_summary(kind, payload)}"
    )


def render_node_detail(detail: dict[str, Any] | None) -> str:
    if detail is None:
        return "node not found"

    def block(label: str, value: Any) -> str:
        body = json.dumps(value, indent=2, ensure_ascii=False, default=str)
        return f"== {label} ==\n{body}"

    parts = [
        f"run_id   : {detail.get('run_id')}",
        f"node_id  : {detail.get('node_id')}",
        f"state    : {detail.get('state')}",
        f"created  : {_ms_to_iso(detail.get('created_at_ms'))}",
        f"updated  : {_ms_to_iso(detail.get('updated_at_ms'))}",
        block("spec", detail.get("spec")),
        block("attempts", detail.get("attempts")),
        block("result", detail.get("result")),
        block("failure", detail.get("failure")),
        block("recent events (heuristic)", detail.get("events")),
    ]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Snapshot orchestration
# --------------------------------------------------------------------------- #


def build_watch_snapshot(
    conn: sqlite3.Connection,
    *,
    run_id: str | None,
    state_filter: Sequence[str] | None,
    node_id: str | None,
) -> dict[str, Any]:
    if run_id is None:
        return {"view": "runs", "runs": list_runs(conn)}
    if node_id is not None:
        return {
            "view": "node",
            "run_id": run_id,
            "node": node_detail(conn, run_id, node_id),
        }
    run = get_run(conn, run_id)
    nodes = list_nodes(conn, run_id, state_filter=state_filter)
    counts = _state_counts(conn, run_id)
    leases = list_leases(conn, run_id)
    return {
        "view": "nodes",
        "run": run,
        "nodes": nodes,
        "state_counts": counts,
        "leases": leases,
    }

def compute_run_stats(
    conn: sqlite3.Connection, *, run_id: str,
) -> dict[str, Any]:
    """Aggregate context-saving metrics for a run.

    Reads ``cc.tool_use`` rows for call counts and ``cc.tool_result``
    rows for preview / full-content byte totals (both produced by the
    cc → v5 event bridge in Phase 1a + Phase 2). When a tool result
    body exceeded the 4 KB threshold, the bridge stamped
    ``full_content_bytes`` on the payload and offloaded the body to the
    ContentStore; the savings equation is
    ``saved = full_content_bytes - preview_bytes``.

    Returns a dict shaped for :func:`render_stats_text` / JSON output:

    ::

        {
          "run_id": "...",
          "tools": [
            {
              "tool_name": "Read",
              "calls": 12,
              "preview_bytes": 2880,
              "full_content_bytes": 480000,
              "saved_bytes": 477120,
              "saved_ratio": 0.994
            },
            ...
          ],
          "totals": {...}
        }
    """
    rows = conn.execute(
        """
        SELECT kind, payload_json
        FROM events
        WHERE run_id = ?
          AND kind IN ('cc.tool_use', 'cc.tool_result', 'cc.tool_completed',
                       'cc.tool_failed')
        ORDER BY sequence ASC
        """,
        (run_id,),
    ).fetchall()

    per_tool: dict[str, dict[str, int]] = {}
    for r in rows:
        payload = _loads(r["payload_json"]) or {}
        tool_name = str(payload.get("tool_name") or "(unknown)")
        bucket = per_tool.setdefault(tool_name, {
            "calls": 0,
            "preview_bytes": 0,
            "full_content_bytes": 0,
            "failures": 0,
        })
        kind = r["kind"]
        if kind == "cc.tool_use":
            bucket["calls"] += 1
        elif kind == "cc.tool_result":
            preview = payload.get("preview") or ""
            bucket["preview_bytes"] += len(str(preview).encode("utf-8"))
            full = payload.get("full_content_bytes")
            if isinstance(full, int):
                bucket["full_content_bytes"] += full
        elif kind in ("cc.tool_failed", "cc.tool_completed"):
            if payload.get("success") is False:
                bucket["failures"] += 1

    tools: list[dict[str, Any]] = []
    for tool_name, b in sorted(per_tool.items()):
        full = b["full_content_bytes"]
        prev = b["preview_bytes"]
        saved = max(0, full - prev) if full > 0 else 0
        ratio = (saved / full) if full > 0 else 0.0
        tools.append({
            "tool_name": tool_name,
            "calls": b["calls"],
            "preview_bytes": prev,
            "full_content_bytes": full,
            "failures": b["failures"],
            "saved_bytes": saved,
            "saved_ratio": ratio,
        })

    totals = {
        "calls": sum(t["calls"] for t in tools),
        "preview_bytes": sum(t["preview_bytes"] for t in tools),
        "full_content_bytes": sum(t["full_content_bytes"] for t in tools),
        "saved_bytes": sum(t["saved_bytes"] for t in tools),
        "failures": sum(t["failures"] for t in tools),
    }
    totals["saved_ratio"] = (
        totals["saved_bytes"] / totals["full_content_bytes"]
        if totals["full_content_bytes"] > 0 else 0.0
    )
    return {"run_id": run_id, "tools": tools, "totals": totals}


def render_stats_text(stats: dict[str, Any]) -> str:
    """Format :func:`compute_run_stats` output as a fixed-width table."""
    tools = stats.get("tools") or []
    if not tools:
        return f"=== Run {stats.get('run_id', '?')} stats ===\n(no cc tool activity)"

    lines = [f"=== Run {stats['run_id']} stats ==="]
    header = f"{'tool':<14} {'calls':>6} {'preview':>12} {'full':>14} {'saved%':>8} {'fail':>5}"
    lines.append(header)
    lines.append("-" * len(header))
    for t in tools:
        saved_pct = (
            f"{t['saved_ratio'] * 100:>7.1f}%"
            if t["full_content_bytes"] > 0 else "      -"
        )
        full = (
            _format_bytes(t["full_content_bytes"])
            if t["full_content_bytes"] > 0 else "-"
        )
        lines.append(
            f"{_truncate(t['tool_name'], 14):<14} "
            f"{t['calls']:>6} "
            f"{_format_bytes(t['preview_bytes']):>12} "
            f"{full:>14} "
            f"{saved_pct:>8} "
            f"{t['failures']:>5}"
        )
    lines.append("-" * len(header))
    tot = stats["totals"]
    saved_pct = (
        f"{tot['saved_ratio'] * 100:>7.1f}%"
        if tot["full_content_bytes"] > 0 else "      -"
    )
    full = (
        _format_bytes(tot["full_content_bytes"])
        if tot["full_content_bytes"] > 0 else "-"
    )
    lines.append(
        f"{'TOTAL':<14} "
        f"{tot['calls']:>6} "
        f"{_format_bytes(tot['preview_bytes']):>12} "
        f"{full:>14} "
        f"{saved_pct:>8} "
        f"{tot['failures']:>5}"
    )
    return "\n".join(lines)


def _format_bytes(n: int) -> str:
    """Compact human-readable byte size (no decimals for the small case)."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


# --------------------------------------------------------------------------- #
# Cross-run failure patterns
# --------------------------------------------------------------------------- #


#: Node states that count as a failure for cross-run aggregation. ``abandoned``
#: is the reachable terminal failure in v5 (``failed`` is not in
#: ``TERMINAL_NODE_STATES`` — see :func:`degraded_completion`); ``failed`` is
#: included as harmless defence-in-depth against a partial / crashed DB write.
_FAILURE_STATES: tuple[str, ...] = ("failed", "abandoned")


def compute_failure_patterns(
    conn: sqlite3.Connection,
    *,
    run_id: str | None = None,
    limit_runs: int | None = None,
) -> dict[str, Any]:
    """Aggregate failed / abandoned nodes ACROSS runs by failure kind.

    ``compute_run_stats`` and ``report.py`` are per-run; this is the cross-run
    view an operator needs after an unattended scheduler campaign: which
    failure KINDS recur, how often, in which tools, across how many runs, and
    when they first / last appeared. It reads the typed ``nodes.failure_json``
    (the same ``Failure`` dict ``list_nodes`` parses — ``kind`` / ``message``)
    — no log scraping — and is read-only.

    Scope: all runs by default. ``run_id`` restricts to one run; ``limit_runs``
    restricts to the most recent N runs (by ``runs.updated_at_ms``), handy when
    a long campaign has accrued many runs. ``run_id`` takes precedence.

    Returns a dict shaped for :func:`render_failure_patterns_text` / JSON::

        {
          "scope": "all-runs",
          "kinds": [
            {"kind": "tool_error", "count": 7, "runs": 4,
             "run_sample": [...], "tools": [{"tool": "Bash", "count": 5}, ...],
             "states": {"abandoned": 7}, "first_seen_ms": ..., "last_seen_ms": ...,
             "sample_node": "run-3/n2", "sample_message": "..."},
            ...
          ],
          "totals": {"failures": 9, "kinds": 2, "runs_affected": 4}
        }
    """
    allowed_runs: set[str] | None = None
    if run_id is not None:
        allowed_runs = {run_id}
    elif limit_runs is not None and limit_runs > 0:
        recent = _query(
            conn,
            "SELECT run_id FROM runs ORDER BY updated_at_ms DESC LIMIT ?",
            (limit_runs,),
        )
        allowed_runs = {_row_get(r, "run_id", "") for r in recent}

    placeholders = ",".join("?" * len(_FAILURE_STATES))
    rows = _query(
        conn,
        "SELECT run_id, node_id, state, spec_json, failure_json, updated_at_ms "
        f"FROM nodes WHERE state IN ({placeholders}) ORDER BY updated_at_ms ASC",
        _FAILURE_STATES,
    )

    by_kind: dict[str, dict[str, Any]] = {}
    for r in rows:
        rid = _row_get(r, "run_id", "") or ""
        if allowed_runs is not None and rid not in allowed_runs:
            continue
        failure = _loads(_row_get(r, "failure_json")) or {}
        kind = (failure.get("kind") if isinstance(failure, dict) else "") or "unknown"
        message = (
            failure.get("message") if isinstance(failure, dict) else ""
        ) or ""
        spec = _loads(_row_get(r, "spec_json")) or {}
        tool = (spec.get("tool") if isinstance(spec, dict) else "") or "(none)"
        state = _row_get(r, "state", "") or ""
        ts = _row_get(r, "updated_at_ms") or 0

        b = by_kind.get(kind)
        if b is None:
            b = by_kind[kind] = {
                "runs": set(),
                "tools": {},
                "states": {},
                "count": 0,
                "first_seen_ms": ts or None,
                "last_seen_ms": ts or None,
                "sample_message": "",
                "sample_node": "",
            }
        b["count"] += 1
        b["runs"].add(rid)
        b["tools"][tool] = b["tools"].get(tool, 0) + 1
        b["states"][state] = b["states"].get(state, 0) + 1
        if ts:
            b["first_seen_ms"] = min(b["first_seen_ms"] or ts, ts)
            b["last_seen_ms"] = max(b["last_seen_ms"] or ts, ts)
        if not b["sample_message"] and message:
            b["sample_message"] = message
            b["sample_node"] = f"{rid}/{_row_get(r, 'node_id', '')}"

    kinds: list[dict[str, Any]] = []
    for kind, b in by_kind.items():
        top_tools = sorted(b["tools"].items(), key=lambda kv: (-kv[1], kv[0]))
        kinds.append({
            "kind": kind,
            "count": b["count"],
            "runs": len(b["runs"]),
            "run_sample": sorted(b["runs"])[:3],
            "tools": [{"tool": t, "count": c} for t, c in top_tools[:5]],
            "states": b["states"],
            "first_seen_ms": b["first_seen_ms"],
            "last_seen_ms": b["last_seen_ms"],
            "sample_node": b["sample_node"],
            "sample_message": b["sample_message"],
        })
    kinds.sort(key=lambda k: (-k["count"], k["kind"]))

    runs_affected = {rid for b in by_kind.values() for rid in b["runs"]}
    if run_id is not None:
        scope = run_id
    elif limit_runs:
        scope = f"recent-{limit_runs}"
    else:
        scope = "all-runs"
    return {
        "scope": scope,
        "kinds": kinds,
        "totals": {
            "failures": sum(k["count"] for k in kinds),
            "kinds": len(kinds),
            "runs_affected": len(runs_affected),
        },
    }


def render_failure_patterns_text(report: dict[str, Any]) -> str:
    """Format :func:`compute_failure_patterns` output as a fixed-width table."""
    kinds = report.get("kinds") or []
    scope = report.get("scope", "?")
    if not kinds:
        return (
            f"=== Failure patterns ({scope}) ===\n"
            "(no failed / abandoned nodes)"
        )

    lines = [f"=== Failure patterns ({scope}) ==="]
    header = (
        f"{'kind':<18} {'count':>6} {'runs':>5} "
        f"{'top tool':<16} {'last seen':<15}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for k in kinds:
        top_tool = k["tools"][0]["tool"] if k.get("tools") else "-"
        lines.append(
            f"{_truncate(k['kind'], 18):<18} "
            f"{k['count']:>6} "
            f"{k['runs']:>5} "
            f"{_truncate(top_tool, 16):<16} "
            f"{_ms_to_iso(k.get('last_seen_ms')):<15}"
        )
    lines.append("-" * len(header))
    tot = report["totals"]
    lines.append(
        f"{'TOTAL':<18} {tot['failures']:>6} {tot['runs_affected']:>5} "
        f"{_truncate(str(tot['kinds']) + ' kind(s)', 16):<16}"
    )

    # The human-actionable bit: a sample message per top kind for drill-down.
    samples = [k for k in kinds[:5] if k.get("sample_message")]
    if samples:
        lines.append("")
        for k in samples:
            lines.append(
                f"• {k['kind']} ({k['count']}×, {k['runs']} run(s)) "
                f"e.g. {k['sample_node']}: "
                f"{_truncate(k['sample_message'], 100)}"
            )
    return "\n".join(lines)


def compute_governance_patterns(
    conn: sqlite3.Connection,
    *,
    run_id: str | None = None,
    limit_runs: int | None = None,
) -> dict[str, Any]:
    """Aggregate run-level governance NOT-PASSED verdicts ACROSS runs.

    The cross-run twin of :func:`compute_failure_patterns`, but over the
    run-boundary ``ccx.governance.verdict`` event (opt-in via
    ``CCX_EMIT_GOVERNANCE_EVENTS``) instead of node failures. It surfaces the
    self-improvement signal a node-level view misses: runs whose nodes all went
    green but whose RUN-level verdict said NO (performative completion), and
    which governance gate + ``stop_reason`` recurs across a campaign.

    Reads the typed event payload (``passed`` + the three sub-verdicts, each
    carrying ``passed`` / ``stop_reason``); no log scraping; read-only. When the
    emit flag was never on there are no such events and the report is empty
    (``totals.not_passed == 0``, ``runs_evaluated == 0``) — harmless.

    Grouping: one pattern per failing sub-verdict, keyed ``<gate>:<stop_reason>``
    (gate ∈ ``contract`` / ``run_audit`` / ``goal``), plus a
    ``degraded:abandoned_warning`` pattern for partial/degraded completions. A
    run tripping several gates counts once per gate. The LATEST governance event
    per run wins (a re-driven run supersedes its earlier verdict).

    Scope: all runs by default; ``run_id`` one run; ``limit_runs`` the most
    recent N (by ``runs.updated_at_ms``). ``run_id`` takes precedence.
    """
    from core.ccx.services.governance_events import GOVERNANCE_VERDICT_EVENT_KIND

    allowed_runs: set[str] | None = None
    if run_id is not None:
        allowed_runs = {run_id}
    elif limit_runs is not None and limit_runs > 0:
        recent = _query(
            conn,
            "SELECT run_id FROM runs ORDER BY updated_at_ms DESC LIMIT ?",
            (limit_runs,),
        )
        allowed_runs = {_row_get(r, "run_id", "") for r in recent}

    rows = _query(
        conn,
        "SELECT run_id, payload_json, created_at_ms FROM events "
        "WHERE kind = ? ORDER BY sequence ASC",
        (GOVERNANCE_VERDICT_EVENT_KIND,),
    )
    # Latest governance verdict per run (ascending order ⇒ last wins).
    latest: dict[str, dict[str, Any]] = {}
    for r in rows:
        rid = _row_get(r, "run_id", "") or ""
        if allowed_runs is not None and rid not in allowed_runs:
            continue
        payload = _loads(_row_get(r, "payload_json"))
        if not isinstance(payload, dict):
            continue
        latest[rid] = {"payload": payload, "ts": _row_get(r, "created_at_ms") or 0}

    by_key: dict[str, dict[str, Any]] = {}

    def _bump(key: str, rid: str, ts: int, sample: str) -> None:
        b = by_key.get(key)
        if b is None:
            b = by_key[key] = {
                "runs": set(), "count": 0,
                "first_seen_ms": ts or None, "last_seen_ms": ts or None,
                "sample_run": "", "sample_detail": "",
            }
        b["count"] += 1
        b["runs"].add(rid)
        if ts:
            b["first_seen_ms"] = min(b["first_seen_ms"] or ts, ts)
            b["last_seen_ms"] = max(b["last_seen_ms"] or ts, ts)
        if not b["sample_run"]:
            b["sample_run"] = rid
            b["sample_detail"] = sample

    for rid, entry in latest.items():
        payload = entry["payload"]
        ts = entry["ts"]
        matched = False
        for source in ("contract_verdict", "run_audit_verdict", "goal_verdict"):
            sub = payload.get(source)
            if isinstance(sub, dict) and sub.get("passed") is False:
                reason = sub.get("stop_reason") or sub.get("status") or "unknown"
                gate = source.replace("_verdict", "")
                _bump(f"{gate}:{reason}", rid, ts, f"status={payload.get('status')}")
                matched = True
        # Overall not-passed but no sub-verdict explicitly False (edge case).
        if payload.get("passed") is False and not matched:
            _bump("unknown:not_passed", rid, ts, f"status={payload.get('status')}")
        if payload.get("abandoned_warning"):
            _bump(
                "degraded:abandoned_warning", rid, ts,
                f"abandoned={payload.get('abandoned')}",
            )

    kinds: list[dict[str, Any]] = []
    for key, b in by_key.items():
        kinds.append({
            "kind": key,
            "count": b["count"],
            "runs": len(b["runs"]),
            "run_sample": sorted(b["runs"])[:3],
            "first_seen_ms": b["first_seen_ms"],
            "last_seen_ms": b["last_seen_ms"],
            "sample_run": b["sample_run"],
            "sample_detail": b["sample_detail"],
        })
    kinds.sort(key=lambda k: (-k["count"], k["kind"]))

    runs_affected = {rid for b in by_key.values() for rid in b["runs"]}
    if run_id is not None:
        scope = run_id
    elif limit_runs:
        scope = f"recent-{limit_runs}"
    else:
        scope = "all-runs"
    return {
        "scope": scope,
        "kinds": kinds,
        "totals": {
            "not_passed": sum(k["count"] for k in kinds),
            "kinds": len(kinds),
            "runs_affected": len(runs_affected),
            "runs_evaluated": len(latest),
        },
    }


def render_governance_patterns_text(report: dict[str, Any]) -> str:
    """Format :func:`compute_governance_patterns` output as a fixed-width table."""
    kinds = report.get("kinds") or []
    scope = report.get("scope", "?")
    tot = report.get("totals") or {}
    evaluated = int(tot.get("runs_evaluated", 0) or 0)
    if not kinds:
        note = (
            "(no not-passed governance verdicts)"
            if evaluated
            else "(no ccx.governance.verdict events — set "
                 "CCX_EMIT_GOVERNANCE_EVENTS=1 to record them)"
        )
        return f"=== Governance patterns ({scope}) ===\n{note}"

    lines = [f"=== Governance patterns ({scope}) ==="]
    header = f"{'gate:reason':<28} {'count':>6} {'runs':>5} {'last seen':<15}"
    lines.append(header)
    lines.append("-" * len(header))
    for k in kinds:
        lines.append(
            f"{_truncate(k['kind'], 28):<28} "
            f"{k['count']:>6} {k['runs']:>5} "
            f"{_ms_to_iso(k.get('last_seen_ms')):<15}"
        )
    lines.append("-" * len(header))
    lines.append(
        f"{'TOTAL':<28} {tot.get('not_passed', 0):>6} "
        f"{tot.get('runs_affected', 0):>5}  "
        f"({tot.get('kinds', 0)} kind(s) / {evaluated} governed run(s))"
    )

    samples = [k for k in kinds[:5] if k.get("sample_run")]
    if samples:
        lines.append("")
        for k in samples:
            lines.append(
                f"• {k['kind']} ({k['count']}×, {k['runs']} run(s)) "
                f"e.g. {k['sample_run']}: {_truncate(k.get('sample_detail', ''), 80)}"
            )
    return "\n".join(lines)


def render_snapshot_text(snap: dict[str, Any]) -> str:
    view = snap.get("view")
    if view == "runs":
        return render_runs_table(snap.get("runs") or [])
    if view == "node":
        return render_node_detail(snap.get("node"))
    if view == "nodes":
        return render_nodes_table(
            snap.get("run"),
            snap.get("nodes") or [],
            state_counts=snap.get("state_counts") or {},
            leases=snap.get("leases") or [],
        )
    return "(unknown view)"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _valid_node_states() -> set[str]:
    """Pull NodeState values from v5 for --state validation. Falls back
    to a hard-coded set if v5 isn't importable.
    """
    try:
        from core.deepstack_v5.types import NodeState  # type: ignore

        return {s.value for s in NodeState}
    except ImportError:
        return set(_FALLBACK_NODE_STATES)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m core.ccx.watch",
        description="Read-only watcher for ccx runs (snapshots from .ccx/runtime/runtime.db).",
    )
    p.add_argument("--cwd", default=".", help="ccx workspace cwd (default: current dir)")
    p.add_argument("--run-id", default=None, help="show node table for this run")
    p.add_argument(
        "--node-id",
        default=None,
        help="show full detail for a single node (requires --run-id)",
    )
    p.add_argument(
        "--state",
        action="append",
        default=None,
        help="filter nodes by state (repeatable; case-insensitive)",
    )
    p.add_argument(
        "--follow",
        action="store_true",
        help="re-print snapshots on an interval until Ctrl-C",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="seconds between snapshots in --follow mode (default: 2)",
    )
    p.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="output format (default: table). With --follow + json => jsonl",
    )
    p.add_argument(
        "--tail",
        action="store_true",
        help=(
            "Cursor-tail the events table instead of re-snapshotting. "
            "Combine with --follow to stream forever; without --follow "
            "drains from --since to the current max and exits."
        ),
    )
    p.add_argument(
        "--since",
        type=int,
        default=None,
        help=(
            "Tail mode only: starting sequence (exclusive). Default "
            "depends on --follow: with --follow it sits at the current "
            "max so only new events stream (tail -f semantics); without "
            "--follow it starts at 0 to replay everything for the run "
            "(cat semantics). Pass an explicit number to resume from a "
            "known boundary."
        ),
    )
    p.add_argument(
        "--kind",
        action="append",
        default=None,
        help=(
            "Tail mode only: filter by event kind prefix (e.g. 'node.', "
            "'replan.'). Repeatable; events matching ANY prefix are kept."
        ),
    )
    p.add_argument(
        "--stats",
        action="store_true",
        help=(
            "Print per-tool context-saving stats for the run (requires "
            "--run-id). Aggregates cc.tool_use and cc.tool_result events "
            "from the v5 events table; shows preview vs. full-content "
            "bytes and the savings ratio when ContentStore (Phase 2) is "
            "enabled."
        ),
    )
    p.add_argument(
        "--failures",
        action="store_true",
        help=(
            "One-shot CROSS-RUN failure-pattern report: aggregates "
            "failed/abandoned nodes across all runs by failure kind, with "
            "per-kind counts, run coverage, top tools and a sample message. "
            "Unlike --stats (per-run) this is the campaign-triage view. "
            "Defaults to all runs; scope with --run-id (one run) or "
            "--limit-runs N (most recent N runs)."
        ),
    )
    p.add_argument(
        "--governance",
        action="store_true",
        help=(
            "One-shot CROSS-RUN governance-verdict report: aggregates "
            "run-level NOT-PASSED verdicts (the ccx.governance.verdict event, "
            "opt-in via CCX_EMIT_GOVERNANCE_EVENTS) by gate:stop_reason — the "
            "performative-completion / recurring-gate view a node-level report "
            "misses. Defaults to all runs; scope with --run-id or "
            "--limit-runs N."
        ),
    )
    p.add_argument(
        "--limit-runs",
        type=int,
        default=None,
        metavar="N",
        help="With --failures / --governance: restrict to the most recent N runs.",
    )
    return p


def _normalise_states(raw: Sequence[str] | None) -> list[str] | None:
    if not raw:
        return None
    valid = _valid_node_states()
    out: list[str] = []
    for s in raw:
        lc = s.strip().lower()
        if lc not in valid:
            raise SystemExit(
                f"unknown state '{s}'. Valid states: {sorted(valid)}"
            )
        out.append(lc)
    return out


def _emit(snap: dict[str, Any], fmt: str) -> None:
    if fmt == "json":
        print(json.dumps(snap, ensure_ascii=False, default=str))
    else:
        print(render_snapshot_text(snap))


def _emit_event(event: dict[str, Any], fmt: str) -> None:
    if fmt == "json":
        print(json.dumps(event, ensure_ascii=False, default=str))
    else:
        print(render_event_line(event))


def _emit_follow_header(fmt: str) -> None:
    if fmt == "json":
        return  # jsonl line speaks for itself
    print(f"\n--- snapshot @ {time.strftime('%H:%M:%S')} ---")


def _run_tail(
    conn: sqlite3.Connection,
    *,
    run_id: str | None,
    node_id: str | None,
    kind_prefixes: Sequence[str] | None,
    since: int | None,
    follow: bool,
    interval: float,
    fmt: str,
) -> int:
    """Tail-mode driver. Drains events into ``_emit_event`` lines.

    Cursor semantics:

    * ``since=None`` + ``follow=False`` → replay every event for the
      scope (whole run if ``run_id`` set, else everything) and exit.
      Acts like ``cat`` on the events table.
    * ``since=None`` + ``follow=True`` → start at current max (only see
      events emitted after the watcher attaches). Acts like ``tail -f``.
    * ``since=N`` → start strictly after N regardless of follow. Useful
      for resuming a previous tail at a known sequence boundary.
    """
    if since is None:
        cursor = current_max_sequence(conn, run_id=run_id) if follow else 0
    else:
        cursor = int(since)
    try:
        while True:
            events, scanned_sequence, scanned_count = tail_events(
                conn,
                after_sequence=cursor,
                run_id=run_id,
                kind_prefixes=kind_prefixes,
                node_id=node_id,
                limit=_TAIL_EVENTS_BATCH_LIMIT,
                return_scan_sequence=True,
                return_scan_count=True,
            )
            for ev in events:
                _emit_event(ev, fmt)
            cursor = max(cursor, scanned_sequence)
            sys.stdout.flush()
            if not follow:
                if scanned_count < _TAIL_EVENTS_BATCH_LIMIT:
                    return 0
                continue
            time.sleep(max(0.05, float(interval)))
    except KeyboardInterrupt:
        print("", file=sys.stderr)
        return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    states = _normalise_states(args.state)
    if args.node_id and not args.run_id:
        raise SystemExit("--node-id requires --run-id")
    if args.tail and args.state:
        raise SystemExit(
            "--state filters the snapshot view; in --tail mode use "
            "--kind to filter event kinds instead"
        )
    if args.stats:
        if not args.run_id:
            raise SystemExit("--stats requires --run-id")
        if args.tail or args.follow:
            raise SystemExit("--stats is a one-shot report; can't combine "
                             "with --tail or --follow")
    if args.failures:
        if args.stats:
            raise SystemExit("--failures and --stats are separate reports; "
                             "pick one")
        if args.tail or args.follow:
            raise SystemExit("--failures is a one-shot report; can't combine "
                             "with --tail or --follow")
    if args.governance:
        if args.stats or args.failures:
            raise SystemExit("--governance, --failures and --stats are "
                             "separate reports; pick one")
        if args.tail or args.follow:
            raise SystemExit("--governance is a one-shot report; can't combine "
                             "with --tail or --follow")

    db_path = resolve_db_path(args.cwd)
    try:
        conn = connect_ro(db_path)
    except (FileNotFoundError, sqlite3.OperationalError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        if args.stats:
            stats = compute_run_stats(conn, run_id=args.run_id)
            if args.format == "json":
                print(json.dumps(stats, ensure_ascii=False, default=str))
            else:
                print(render_stats_text(stats))
            return 0

        if args.failures:
            report = compute_failure_patterns(
                conn, run_id=args.run_id, limit_runs=args.limit_runs,
            )
            if args.format == "json":
                print(json.dumps(report, ensure_ascii=False, default=str))
            else:
                print(render_failure_patterns_text(report))
            return 0

        if args.governance:
            report = compute_governance_patterns(
                conn, run_id=args.run_id, limit_runs=args.limit_runs,
            )
            if args.format == "json":
                print(json.dumps(report, ensure_ascii=False, default=str))
            else:
                print(render_governance_patterns_text(report))
            return 0

        if args.tail:
            return _run_tail(
                conn,
                run_id=args.run_id,
                node_id=args.node_id,
                kind_prefixes=args.kind,
                since=args.since,
                follow=args.follow,
                interval=float(args.interval),
                fmt=args.format,
            )

        if not args.follow:
            snap = build_watch_snapshot(
                conn,
                run_id=args.run_id,
                state_filter=states,
                node_id=args.node_id,
            )
            _emit(snap, args.format)
            return 0

        interval = max(0.1, float(args.interval))
        try:
            while True:
                _emit_follow_header(args.format)
                snap = build_watch_snapshot(
                    conn,
                    run_id=args.run_id,
                    state_filter=states,
                    node_id=args.node_id,
                )
                _emit(snap, args.format)
                sys.stdout.flush()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("", file=sys.stderr)  # newline after ^C
            return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
