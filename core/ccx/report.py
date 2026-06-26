"""Static HTML report generator for ccx runs.

Reads ``<cwd>/.ccx/runtime/runtime.db`` (read-only) and emits a single
self-contained HTML file describing one run: header, Mermaid DAG,
filterable/sortable node table, per-node details, and an events
timeline. Designed for after-the-fact inspection — for live status,
use ``core.ccx.watch --follow``.

This module deliberately speaks only the runtime DB schema. It does
not import any v5 store classes, and it never starts an HTTP server.
The Python side is stdlib-only; the HTML has one CDN script tag for
Mermaid (everything else is inlined).

Data access reuses the pure functions exported by ``core.ccx.watch``
(``connect_ro``, ``list_nodes``, ``list_leases`` etc). Three queries
that watch did not need are added here as private helpers
(``_load_run_full``, ``_load_edges``, ``_load_events``); none of them
duplicate SQL that watch already runs.

CLI::

    python -m core.ccx.report --cwd PATH --run-id ID [--out PATH]
                              [--include-events N] [--max-dag-nodes N]
                              [--open]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import html
import json
import sqlite3
import sys
import webbrowser
from pathlib import Path
from typing import Any, Sequence

from core.ccx.services.governance_events import GOVERNANCE_VERDICT_EVENT_KIND
from core.ccx.watch import (
    _degraded_phrase,
    _loads,
    _ms_to_iso,
    _query,
    _row_get,
    connect_ro,
    degraded_completion,
    get_run,
    list_nodes,
    resolve_db_path,
)


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #


_TEMPLATE_PATH = Path(__file__).parent / "templates" / "report.html.j2"

DEFAULT_INCLUDE_EVENTS = 200
DEFAULT_MAX_DAG_NODES = 150

# State -> CSS class fragment used in the template (matches `.badge.s-<x>`).
_STATE_CLASSES = {
    "succeeded": "succeeded",
    "completed": "completed",
    "failed": "failed",
    "abandoned": "abandoned",
    "budget_exhausted": "budget_exhausted",
    "aborted": "aborted",
    "running": "running",
    # A run blocked awaiting human approval is an attention state, not a
    # benign "pending"; map it to the orange hang class (mirrors the
    # node-level approval_hang) so it is not rendered as a neutral grey badge.
    "waiting_approval": "approval_hang",
    "pending": "pending",
    "ready": "ready",
    "skipped": "skipped",
    "cancelled": "cancelled",
    "approval_hang": "approval_hang",
    "timer_hang": "timer_hang",
    "blocked": "blocked",
}

# Mermaid classDef -> hex colours, kept in sync with the CSS.
_MERMAID_CLASSDEFS = """
classDef s_succeeded fill:#c8e6c9,stroke:#2e7d32,color:#1b5e20;
classDef s_failed fill:#ffcdd2,stroke:#c62828,color:#b71c1c;
classDef s_abandoned fill:#ffcdd2,stroke:#c62828,color:#b71c1c;
classDef s_running fill:#bbdefb,stroke:#1565c0,color:#0d47a1,stroke-dasharray: 4 2;
classDef s_pending fill:#eeeeee,stroke:#616161,color:#212121;
classDef s_ready fill:#eeeeee,stroke:#616161,color:#212121;
classDef s_skipped fill:#f5f5f5,stroke:#9e9e9e,color:#616161;
classDef s_cancelled fill:#f5f5f5,stroke:#9e9e9e,color:#616161;
classDef s_approval_hang fill:#fff3e0,stroke:#ef6c00,color:#e65100;
classDef s_timer_hang fill:#fff3e0,stroke:#ef6c00,color:#e65100;
classDef s_blocked fill:#fff3e0,stroke:#ef6c00,color:#e65100;
""".strip()


# --------------------------------------------------------------------------- #
# Extra queries (not present in watch.py)
# --------------------------------------------------------------------------- #


def _load_run_full(conn: sqlite3.Connection, run_id: str) -> dict[str, Any] | None:
    """Run row including budget_json / config_json / metadata_json (if
    present) and a derived ``duration_ms``.
    """
    base = get_run(conn, run_id)
    if base is None:
        return None
    rows = _query(
        conn,
        "SELECT budget_json, config_json, metadata_json FROM runs WHERE run_id = ?",
        (run_id,),
    )
    if rows:
        r = rows[0]
        base["budget"] = _loads(_row_get(r, "budget_json"))
        base["config"] = _loads(_row_get(r, "config_json"))
        base["metadata"] = _loads(_row_get(r, "metadata_json"))
    else:
        base.setdefault("budget", None)
        base.setdefault("config", None)
        base.setdefault("metadata", None)
    created = base.get("created_at_ms") or 0
    updated = base.get("updated_at_ms") or 0
    base["duration_ms"] = max(0, int(updated) - int(created)) if created and updated else None
    return base


def _load_edges(conn: sqlite3.Connection, run_id: str) -> list[tuple[str, str]]:
    rows = _query(
        conn,
        "SELECT src_node_id, dst_node_id FROM edges WHERE run_id = ?",
        (run_id,),
    )
    return [
        (_row_get(r, "src_node_id", "") or "", _row_get(r, "dst_node_id", "") or "")
        for r in rows
    ]


def _load_events(
    conn: sqlite3.Connection, run_id: str, limit: int
) -> list[dict[str, Any]]:
    rows = _query(
        conn,
        """
        SELECT sequence, kind, created_at_ms, payload_json
        FROM events
        WHERE run_id = ?
        ORDER BY sequence DESC LIMIT ?
        """,
        (run_id, max(0, int(limit))),
    )
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "sequence": _row_get(r, "sequence"),
                "kind": _row_get(r, "kind", "") or "",
                "created_at_ms": _row_get(r, "created_at_ms"),
                "payload": _loads(_row_get(r, "payload_json")),
            }
        )
    out.reverse()  # ascending for chronological reading
    return out


def _load_node_details(
    conn: sqlite3.Connection, run_id: str
) -> dict[str, dict[str, Any]]:
    rows = _query(
        conn,
        "SELECT * FROM nodes WHERE run_id = ? ORDER BY created_at_ms, node_id",
        (run_id,),
    )
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        node_id = _row_get(r, "node_id", "") or ""
        if not node_id:
            continue
        out[node_id] = {
            "run_id": run_id,
            "node_id": node_id,
            "state": _row_get(r, "state", ""),
            "spec": _loads(_row_get(r, "spec_json")),
            "attempts": _loads(_row_get(r, "attempts_json")),
            "result": _loads(_row_get(r, "result_json")),
            "failure": _loads(_row_get(r, "failure_json")),
            "created_at_ms": _row_get(r, "created_at_ms"),
            "updated_at_ms": _row_get(r, "updated_at_ms"),
        }
    return out


def _load_node_events_by_node(
    conn: sqlite3.Connection,
    run_id: str,
    *,
    per_node_limit: int = 20,
) -> dict[str, list[dict[str, Any]]]:
    rows = _query(
        conn,
        """
        SELECT sequence, kind, created_at_ms, payload_json
        FROM events
        WHERE run_id = ?
        ORDER BY sequence DESC
        """,
        (run_id,),
    )
    out: dict[str, list[dict[str, Any]]] = {}
    limit = max(0, int(per_node_limit))
    if limit <= 0:
        return out
    for r in rows:
        payload = _loads(_row_get(r, "payload_json"))
        if not isinstance(payload, dict):
            continue
        node_id = payload.get("node_id")
        if not isinstance(node_id, str) or not node_id:
            continue
        bucket = out.setdefault(node_id, [])
        if len(bucket) >= limit:
            continue
        bucket.append(
            {
                "sequence": _row_get(r, "sequence"),
                "kind": _row_get(r, "kind", "") or "",
                "created_at_ms": _row_get(r, "created_at_ms"),
                "payload": payload,
            }
        )
    for events in out.values():
        events.reverse()
    return out


# --------------------------------------------------------------------------- #
# Format helpers
# --------------------------------------------------------------------------- #


def _e(value: Any) -> str:
    """HTML-escape any value, coercing to string first. Used everywhere
    DB content meets the rendered HTML."""
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def _format_duration_ms(ms: int | None) -> str:
    if not isinstance(ms, (int, float)) or ms <= 0:
        return ""
    seconds = int(ms // 1000)
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes}m {seconds}s"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h {minutes}m"


def _state_class(state: str | None) -> str:
    s = (state or "").lower()
    return _STATE_CLASSES.get(s, "pending")


def _state_badge(state: str | None) -> str:
    s = (state or "").lower()
    cls = _state_class(s)
    return f'<span class="badge s-{_e(cls)}">{_e(s or "?")}</span>'


def _pretty_json_block(value: Any, *, max_chars: int = 0) -> str:
    """Pretty-print ``value`` as JSON inside an escaped ``<pre><code>``.
    Returns an empty string for None to keep the layout tight.
    """
    if value is None:
        return ""
    try:
        text = json.dumps(value, indent=2, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(value)
    if max_chars and len(text) > max_chars:
        text = text[:max_chars] + "\n…(truncated)"
    return f"<pre><code>{_e(text)}</code></pre>"


def _mermaid_text_escape(s: str) -> str:
    """Escape characters that have meaning inside a Mermaid quoted label.

    Mermaid renders the characters between the double-quotes; we make
    sure no literal ``"`` breaks out, and HTML-encode angle brackets and
    ``&`` so a malicious node_id like ``<script>`` cannot escape Mermaid
    into raw HTML.
    """
    return (
        str(s)
        .replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", " ")
    )


# --------------------------------------------------------------------------- #
# Section renderers
# --------------------------------------------------------------------------- #


def _render_header(
    run: dict[str, Any],
    state_counts: dict[str, int],
    node_count: int,
    *,
    governance: dict[str, Any] | None = None,
) -> str:
    status = run.get("status") or ""
    status_badge = _state_badge(status)
    goal = run.get("goal") or ""
    goal_html = _render_goal(goal)
    created = _ms_to_iso(run.get("created_at_ms"))
    updated = _ms_to_iso(run.get("updated_at_ms"))
    duration = _format_duration_ms(run.get("duration_ms"))

    counts_pieces: list[str] = []
    for state in ("succeeded", "running", "failed", "abandoned", "skipped",
                  "pending", "ready", "blocked", "approval_hang", "timer_hang",
                  "cancelled"):
        n = state_counts.get(state, 0)
        if n:
            counts_pieces.append(
                f'<span class="badge s-{_e(_state_class(state))}">{n} {_e(state)}</span>'
            )
    if not counts_pieces:
        counts_pieces.append(
            f'<span class="muted">{node_count} node(s), no state breakdown</span>'
        )
    counts_html = '<div class="counts">' + " ".join(counts_pieces) + "</div>"

    governance_html = _render_governance_banner(governance)
    degraded_html = _render_degraded_banner(status, state_counts)
    metadata_html = _render_run_metadata(run)
    return (
        f'<h1>ccx run <code>{_e(run.get("run_id"))}</code></h1>'
        f'<div>{status_badge} <span class="muted">·</span> '
        f'<span class="mono">{_e(created)}</span> → '
        f'<span class="mono">{_e(updated)}</span>'
        f'<span class="muted"> ({_e(duration or "in progress")})</span></div>'
        f'{governance_html}'
        f'{degraded_html}'
        f'<div class="field"><h3>Goal</h3>{goal_html}</div>'
        f'{counts_html}'
        f'{metadata_html}'
    )


def _render_degraded_banner(status: str, state_counts: dict[str, int]) -> str:
    """Prominent run-level banner for a COMPLETED-but-degraded run.

    Empty string for clean / non-completed runs, so their header renders
    byte-for-byte as before. Reuses the existing ``.warn`` banner style and
    sits directly under the status line so it qualifies the green
    ``s-completed`` badge at a glance, mirroring
    ``session_snapshot['abandoned_warning']`` (api.py) at the operator layer.
    """
    degraded = degraded_completion(status, state_counts)
    if not degraded:
        return ""
    return (
        '<div class="warn">⚠ Degraded run: reported '
        f'<strong>completed</strong> but carries {_e(_degraded_phrase(degraded))} '
        "node(s) — partial / best-effort completion, NOT a clean success.</div>"
    )


def _latest_governance_verdict(
    events: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    """Payload of the run's most recent ``ccx.governance.verdict`` event.

    Reads the events ALREADY loaded for the report (see :func:`build_html`), so
    it adds no DB round-trip — the task's explicit data-source constraint.
    ``events`` is in ascending ``sequence`` order (``_load_events`` reverses the
    ``DESC`` query), so the LAST match is the most recent. Returns ``None`` when
    there is no such event — the default, since emission is opt-in via
    ``CCX_EMIT_GOVERNANCE_EVENTS`` — or its payload is not a dict. Best-effort:
    never raises. Mirrors ``llm_monitor.latest_governance_verdict`` without
    importing it, keeping ``report`` free of cross-renderer coupling.
    """
    latest: dict[str, Any] | None = None
    for ev in events:
        if not isinstance(ev, dict):
            continue
        if ev.get("kind") != GOVERNANCE_VERDICT_EVENT_KIND:
            continue
        payload = ev.get("payload")
        if isinstance(payload, dict):
            latest = payload
    return latest


def _render_governance_banner(payload: Any) -> str:
    """Prominent run-level banner when the run-boundary governance verdict
    did NOT pass — the report-renderer twin of watch's
    ``ccx.governance.verdict`` line and llm_monitor's
    ``performative_completion`` alert.

    The motivating failure: every node SUCCEEDED — so the node-state view, and
    the green ``s-completed`` badge, read as a clean success — while the
    run-level ``goal_verdict`` / ``run_audit_verdict`` / ``contract_verdict``
    says NO. Without this banner the operator reads performative completion as
    real completion.

    Fires ONLY on an explicit not-passed verdict (``payload['passed'] is
    False``). Returns ``""`` for a passed=True verdict, a missing / None /
    non-dict / malformed payload, or no governance event at all — so a run with
    no ``ccx.governance.verdict`` event (the default-OFF path, and every
    pre-Tier-1 run) renders byte-for-byte as before. Best-effort: never raises.

    Orthogonal to and co-exists with :func:`_render_degraded_banner`: a
    run-level NOT-PASSED and node-level degradation are independent signals, so
    both banners may appear together. Reuses the existing ``.warn`` style and
    adds no template/CSS, so the no-event path stays byte-identical end to end.
    """
    if not isinstance(payload, dict):
        return ""
    if payload.get("passed") is not False:
        return ""
    items: list[str] = []
    harness_defect = False
    for label, key in (
        ("contract", "contract_verdict"),
        ("run audit", "run_audit_verdict"),
        ("goal", "goal_verdict"),
    ):
        sub = payload.get(key)
        if not isinstance(sub, dict):
            continue
        mark = "passed" if sub.get("passed") else "NOT-PASSED"
        reason = sub.get("stop_reason") or sub.get("status") or "?"
        detail = (
            f"<strong>{_e(label)}</strong>: {_e(mark)} "
            f'<span class="mono">({_e(reason)})</span>'
        )
        unrunnable = sub.get("unrunnable_criterion_ids")
        if isinstance(unrunnable, (list, tuple)) and unrunnable:
            harness_defect = True
            detail += (
                " — <em>unrunnable criteria: "
                f"{_e(', '.join(str(c) for c in unrunnable))}</em>"
            )
        if str(reason) == "harness_defect":
            harness_defect = True
        items.append(f"<li>{detail}</li>")
    sub_html = (
        f'<ul class="gov-subverdicts">{"".join(items)}</ul>' if items else ""
    )
    defect_note = (
        " <strong>Note:</strong> a sub-verdict stopped on a "
        "<code>harness_defect</code> / unrunnable criterion — this points at a "
        "check/harness gap, NOT a substantive task failure; distinguish the two "
        "before treating the run as a real No-Go."
        if harness_defect
        else ""
    )
    return (
        '<div class="warn">⚠ Governance verdict: this run reported '
        "<strong>NOT-PASSED</strong> at the run boundary. Node states may read "
        "as a clean success, but the run-level governance gate did not pass — "
        "inspect the sub-verdicts, not just the green badges."
        f"{defect_note}{sub_html}</div>"
    )


def _render_goal(goal: str) -> str:
    if not goal:
        return '<div class="muted">(no goal recorded)</div>'
    if len(goal) <= 300:
        return f'<div class="goal">{_e(goal)}</div>'
    head, tail = goal[:300], goal[300:]
    return (
        '<div class="collapsible-goal">'
        f'<span class="goal">{_e(head)}<span class="hidden-tail">{_e(tail)}</span></span> '
        '<span class="toggle">[show more]</span>'
        '</div>'
    )


def _render_run_metadata(run: dict[str, Any]) -> str:
    pairs = [
        ("budget", run.get("budget")),
        ("config", run.get("config")),
        ("metadata", run.get("metadata")),
    ]
    blocks: list[str] = []
    for label, value in pairs:
        if value in (None, "", {}, []):
            continue
        body = _pretty_json_block(value, max_chars=20_000)
        if body:
            blocks.append(
                f'<details><summary>{_e(label)}</summary>{body}</details>'
            )
    if not blocks:
        return ""
    return '<div class="run-metadata">' + "".join(blocks) + "</div>"


def _render_dag_section(
    nodes: Sequence[dict[str, Any]],
    edges: Sequence[tuple[str, str]],
    *,
    max_dag_nodes: int,
) -> str:
    n = len(nodes)
    if n == 0:
        return (
            '<section id="dag"><h2>DAG</h2>'
            '<div class="muted">No nodes recorded for this run.</div>'
            "</section>"
        )
    if n > max_dag_nodes:
        return (
            '<section id="dag"><h2>DAG</h2>'
            f'<div class="warn">DAG omitted: {n} nodes exceeds threshold '
            f"({max_dag_nodes}). Use <code>--max-dag-nodes</code> to override."
            "</div></section>"
        )

    # Build deterministic alias mapping so node_ids with special chars
    # (or worse, attempted XSS) cannot break Mermaid.
    aliases: dict[str, str] = {}
    for i, node in enumerate(nodes):
        aliases[node["node_id"]] = f"N{i}"

    lines = ["flowchart TD"]
    for node in nodes:
        nid = node["node_id"]
        alias = aliases[nid]
        state = (node.get("state") or "").lower()
        label = f'{_mermaid_text_escape(nid)}\\n[{_mermaid_text_escape(state or "?")}]'
        lines.append(f'    {alias}["{label}"]')
        cls = _state_class(state)
        lines.append(f"    class {alias} s_{cls}")
    for src, dst in edges:
        if src in aliases and dst in aliases:
            lines.append(f"    {aliases[src]} --> {aliases[dst]}")
    lines.append(_MERMAID_CLASSDEFS)
    diagram = "\n".join(lines)
    # Mermaid reads textContent; escape only HTML-meta chars to keep
    # the syntax intact while preventing tag-injection.
    safe_diagram = (
        diagram.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return (
        '<section id="dag"><details open><summary><h2 style="display:inline-block;'
        'border:0;margin:0;padding:0;">DAG</h2></summary>'
        '<div class="dag-host">'
        f'<pre class="mermaid">{safe_diagram}</pre>'
        "</div></details></section>"
    )


def _render_controls() -> str:
    return (
        '<button data-filter="all" class="active">All</button>'
        '<button data-filter="succeeded">Succeeded</button>'
        '<button data-filter="failed">Failed</button>'
        '<button data-filter="running">Running</button>'
        '<button data-filter="other">Other</button>'
        '<input type="text" id="nodes-search" placeholder="search node_id…">'
        '<button id="jump-failures" type="button">Jump to failures</button>'
    )


def _render_nodes_table(nodes: Sequence[dict[str, Any]]) -> str:
    headers = [
        ("node_id", "str"),
        ("state", "str"),
        ("attempts", "num"),
        ("tool", "str"),
        ("deps", "num"),
        ("duration", "num"),
        ("last_failure_kind", "str"),
    ]
    head_html = "".join(
        f'<th data-sort-idx="{i}" data-sort-kind="{kind}">{_e(name)}</th>'
        for i, (name, kind) in enumerate(headers)
    )
    rows: list[str] = []
    for node in nodes:
        state = (node.get("state") or "").lower()
        node_id = node.get("node_id") or ""
        attempts = node.get("attempts", 0) or 0
        tool = node.get("tool") or ""
        deps = node.get("deps") or []
        deps_n = len(deps) if isinstance(deps, list) else 0
        deps_title = ", ".join(deps) if isinstance(deps, list) and deps else ""
        last_fail = node.get("last_failure_kind") or ""

        terminal = state in {"succeeded", "failed", "abandoned",
                              "skipped", "cancelled"}
        dur_ms = (
            int(node.get("updated_at_ms") or 0) - int(node.get("created_at_ms") or 0)
            if terminal
            else 0
        )
        dur_str = _format_duration_ms(dur_ms) if terminal and dur_ms > 0 else ""

        rows.append(
            f'<tr data-node-id="{_e(node_id)}" data-state="{_e(state)}">'
            f'<td class="node-id">{_e(node_id)}</td>'
            f"<td>{_state_badge(state)}</td>"
            f'<td data-sort="{int(attempts)}">{_e(attempts)}</td>'
            f"<td>{_e(tool)}</td>"
            f'<td data-sort="{deps_n}" title="{_e(deps_title)}">{deps_n}</td>'
            f'<td data-sort="{dur_ms}">{_e(dur_str)}</td>'
            f"<td>{_e(last_fail)}</td>"
            "</tr>"
        )
    if not rows:
        rows.append('<tr><td colspan="7" class="muted">No nodes.</td></tr>')
    return (
        '<table class="nodes" id="nodes-table">'
        f"<thead><tr>{head_html}</tr></thead>"
        f'<tbody id="nodes-tbody">{"".join(rows)}</tbody>'
        "</table>"
    )


def _render_node_details(
    nodes: Sequence[dict[str, Any]],
    details_by_node: dict[str, dict[str, Any]],
    events_by_node: dict[str, list[dict[str, Any]]],
) -> str:
    chunks: list[str] = []
    for node in nodes:
        node_id = node.get("node_id") or ""
        state = (node.get("state") or "").lower()
        detail = details_by_node.get(node_id) or {}
        spec = detail.get("spec")
        attempts = detail.get("attempts")
        result = detail.get("result")
        failure = detail.get("failure")
        events = events_by_node.get(node_id) or []

        spec_html = _pretty_json_block(spec) if spec else ""
        attempts_html = _render_attempts(attempts)
        result_html = _pretty_json_block(result) if result else ""
        failure_html = _pretty_json_block(failure) if failure else ""
        events_html = _render_node_events(events)

        body_parts: list[str] = [
            f'<dl class="kv">'
            f'<dt>state</dt><dd>{_state_badge(state)}</dd>'
            f"<dt>created</dt><dd>{_e(_ms_to_iso(detail.get('created_at_ms')))}</dd>"
            f"<dt>updated</dt><dd>{_e(_ms_to_iso(detail.get('updated_at_ms')))}</dd>"
            "</dl>"
        ]
        if spec_html:
            body_parts.append(f'<div class="field"><h4>spec</h4>{spec_html}</div>')
        if attempts_html:
            body_parts.append(
                f'<div class="field"><h4>attempts</h4>{attempts_html}</div>'
            )
        if result_html:
            body_parts.append(
                f'<div class="field"><h4>result</h4>{result_html}</div>'
            )
        if failure_html:
            body_parts.append(
                f'<div class="field failure"><h4>failure</h4>{failure_html}</div>'
            )
        if events_html:
            body_parts.append(
                f'<div class="field"><h4>recent events</h4>'
                f"{events_html}</div>"
            )
        body = "".join(body_parts)
        chunks.append(
            f'<details class="node" data-node-id="{_e(node_id)}" '
            f'data-state="{_e(state)}">'
            f"<summary>{_e(node_id)} {_state_badge(state)}</summary>"
            f'<div class="body">{body}</div>'
            "</details>"
        )
    if not chunks:
        return '<div class="muted">No nodes.</div>'
    return "".join(chunks)


def _render_attempts(attempts: Any) -> str:
    if not isinstance(attempts, list) or not attempts:
        return ""
    items: list[str] = []
    for i, a in enumerate(attempts):
        if not isinstance(a, dict):
            items.append(f"<li>{_pretty_json_block(a)}</li>")
            continue
        worker = a.get("worker_id") or ""
        outcome = a.get("outcome") or ""
        started = _ms_to_iso(a.get("started_at_ms"))
        ended = _ms_to_iso(a.get("ended_at_ms"))
        summary = (
            f"#{i + 1} worker={_e(worker)} outcome={_e(outcome)} "
            f"{_e(started)} → {_e(ended)}"
        )
        items.append(
            f"<li><details><summary>{summary}</summary>"
            f"{_pretty_json_block(a)}</details></li>"
        )
    return f'<ul class="attempts">{"".join(items)}</ul>'


def _render_node_events(events: Sequence[dict[str, Any]]) -> str:
    if not events:
        return ""
    pieces: list[str] = []
    for ev in events:
        kind = ev.get("kind") or ""
        seq = ev.get("sequence")
        ts = _ms_to_iso(ev.get("created_at_ms"))
        payload = ev.get("payload")
        preview = _payload_preview(payload)
        pieces.append(
            f'<div class="event" data-kind="{_e(kind)}">'
            f'<span class="seq">#{_e(seq)}</span>'
            f'<span class="kind">{_e(kind)} <span class="muted">{_e(ts)}</span></span>'
            f'<details><summary><span class="preview">{_e(preview)}</span></summary>'
            f"{_pretty_json_block(payload, max_chars=10_000)}</details>"
            "</div>"
        )
    return f'<div class="events">{"".join(pieces)}</div>'


def _render_events_section(events: Sequence[dict[str, Any]], total_seen: int) -> str:
    if not events:
        return '<div class="muted">No events recorded for this run.</div>'
    body = _render_node_events(events)
    note = (
        f'<div class="muted">Showing {len(events)} most recent events '
        f"(of {total_seen} fetched; cap via <code>--include-events</code>).</div>"
    )
    return note + body


def _payload_preview(payload: Any) -> str:
    if payload is None:
        return ""
    try:
        text = json.dumps(payload, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(payload)
    if len(text) > 200:
        return text[:200] + "…"
    return text


# --------------------------------------------------------------------------- #
# Top-level orchestration
# --------------------------------------------------------------------------- #


def _state_counts(nodes: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for n in nodes:
        s = (n.get("state") or "").lower()
        counts[s] = counts.get(s, 0) + 1
    return counts


def _load_template() -> str:
    return _TEMPLATE_PATH.read_text(encoding="utf-8")


def _render_template(template: str, mapping: dict[str, str]) -> str:
    out = template
    for key, value in mapping.items():
        out = out.replace("{{ " + key + " }}", value)
    return out


def build_html(
    conn: sqlite3.Connection,
    run_id: str,
    *,
    include_events: int = DEFAULT_INCLUDE_EVENTS,
    max_dag_nodes: int = DEFAULT_MAX_DAG_NODES,
) -> str:
    """Render an HTML report string for ``run_id``.

    Returns the full HTML document. Caller is responsible for writing
    it to disk. Raises ``LookupError`` if the run does not exist.
    """
    run = _load_run_full(conn, run_id)
    if run is None:
        raise LookupError(f"run not found: {run_id!r}")
    nodes = list_nodes(conn, run_id)
    edges = _load_edges(conn, run_id)
    events = _load_events(conn, run_id, include_events)
    details_by_node = _load_node_details(conn, run_id)
    events_by_node = _load_node_events_by_node(conn, run_id)
    counts = _state_counts(nodes)
    governance = _latest_governance_verdict(events)

    template = _load_template()
    mapping = {
        "TITLE": _e(f"ccx report — {run_id}"),
        "GENERATED_AT": _e(_dt.datetime.now(_dt.timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )),
        "HEADER": _render_header(run, counts, len(nodes), governance=governance),
        "DAG_SECTION": _render_dag_section(
            nodes, edges, max_dag_nodes=max_dag_nodes
        ),
        "CONTROLS": _render_controls(),
        "NODES_TABLE": _render_nodes_table(nodes),
        "NODE_DETAILS": _render_node_details(
            nodes,
            details_by_node,
            events_by_node,
        ),
        "EVENTS_SECTION": _render_events_section(events, len(events)),
    }
    return _render_template(template, mapping)


def write_report(
    cwd: str | Path,
    run_id: str,
    out_path: str | Path | None = None,
    *,
    include_events: int = DEFAULT_INCLUDE_EVENTS,
    max_dag_nodes: int = DEFAULT_MAX_DAG_NODES,
) -> Path:
    """Connect, render, write. Returns the resolved output path."""
    db_path = resolve_db_path(cwd)
    if out_path is None:
        out_path = Path(cwd) / ".ccx" / "reports" / f"{run_id}.html"
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    conn = connect_ro(db_path)
    try:
        html_str = build_html(
            conn,
            run_id,
            include_events=include_events,
            max_dag_nodes=max_dag_nodes,
        )
    finally:
        conn.close()
    out.write_text(html_str, encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m core.ccx.report",
        description="Generate a self-contained HTML report for one ccx run.",
    )
    p.add_argument("--cwd", default=".", help="ccx workspace cwd (default: current dir)")
    p.add_argument(
        "--run-id",
        default=None,
        help="run id to report on (required; use `python -m core.ccx.watch --cwd <p>` to list)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="output HTML path (default: <cwd>/.ccx/reports/<run_id>.html)",
    )
    p.add_argument(
        "--include-events",
        type=int,
        default=DEFAULT_INCLUDE_EVENTS,
        help=f"max events to embed (default: {DEFAULT_INCLUDE_EVENTS})",
    )
    p.add_argument(
        "--max-dag-nodes",
        type=int,
        default=DEFAULT_MAX_DAG_NODES,
        help=(
            "skip Mermaid DAG when node count exceeds this "
            f"(default: {DEFAULT_MAX_DAG_NODES})"
        ),
    )
    p.add_argument(
        "--open",
        action="store_true",
        help="open the generated HTML in the default browser",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.run_id:
        print(
            "error: --run-id is required. List runs with:\n"
            f"  python -m core.ccx.watch --cwd {args.cwd}",
            file=sys.stderr,
        )
        return 2
    db_path = resolve_db_path(args.cwd)
    if not db_path.exists():
        print(
            f"error: runtime DB not found at {db_path} — "
            "no ccx run has been started under this cwd yet.",
            file=sys.stderr,
        )
        return 2
    try:
        out = write_report(
            args.cwd,
            args.run_id,
            args.out,
            include_events=args.include_events,
            max_dag_nodes=args.max_dag_nodes,
        )
    except LookupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except sqlite3.OperationalError as exc:
        print(f"error: db read failed: {exc}", file=sys.stderr)
        return 2
    print(str(out))
    if args.open:
        webbrowser.open(out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
