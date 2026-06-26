"""LLM-supervised monitor for ccx runs.

Reads the same read-only SQLite runtime DB that ``core.ccx.watch`` reads
(``<cwd>/.ccx/runtime/runtime.db``), polls it on an interval, and runs
two layers of supervision over each snapshot:

1. **Heuristics** (always on, zero LLM cost) — cheap rules over a
   snapshot/diff that fire alerts for obvious problems: failed nodes,
   expired or starving leases, abandoned/approval-hang nodes, runs that
   stop making progress.

2. **LLM assessment** (opt-in via ``--enable-llm``) — at a configurable
   cadence, or on heuristic escalation, send a compact snapshot + delta
   + heuristic hits to an LLM and ask for a structured verdict on
   subtler patterns (retry storms, dependency stalls, worker imbalance).

The LLM path is off by default so the tool has no token cost out of the
box. Heuristics are independent of the LLM, so critical alerts surface
even with the LLM disabled or failing.

Run ID resolution: when ``--run-id`` is omitted, the monitor picks the
most recently updated run with status in {running, queued}; if none are
active it falls back to the most recently updated run overall.

Usage::

    # Heuristic-only, auto-pick the latest active run
    python -m core.ccx.llm_monitor --cwd PATH

    # Enable LLM, write a JSONL alert log
    python -m core.ccx.llm_monitor --cwd PATH --enable-llm \\
        --llm-cadence 60 --log-file /tmp/ccx-monitor.jsonl

    # Single assessment cycle (smoke test / cron)
    python -m core.ccx.llm_monitor --cwd PATH --once

Note: ``ClaudeClient`` does not yet support prompt caching; at the
default 1/min cadence this is acceptable. Caching is a future
optimization.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from core.ccx.modes.parsing import parse_llm_json
from core.ccx.watch import (
    build_watch_snapshot,
    connect_ro,
    list_runs,
    resolve_db_path,
)


_FALLBACK_TERMINAL_RUN_STATUSES = {
    "completed",
    "failed",
    "cancelled",
    "budget_exhausted",
    "aborted",
}
_FALLBACK_ACTIVE_RUN_STATUSES = {"running", "waiting_approval"}


# --------------------------------------------------------------------------- #
# Run-id selection
# --------------------------------------------------------------------------- #


def _load_status_sets() -> tuple[set[str], set[str]]:
    try:
        from core.deepstack_v5 import RunStatus, TERMINAL_RUN_STATUSES
    except ImportError:
        return (
            set(_FALLBACK_TERMINAL_RUN_STATUSES),
            set(_FALLBACK_ACTIVE_RUN_STATUSES),
        )
    terminal = {status.value for status in TERMINAL_RUN_STATUSES}
    active = {
        status.value
        for status in RunStatus
        if status.value not in terminal
    }
    return terminal, active


_TERMINAL_STATUSES, _ACTIVE_STATUSES = _load_status_sets()


def pick_latest_active_run(conn: sqlite3.Connection) -> str | None:
    """Pick the run-id to monitor when --run-id is not given.

    Prefers the most recently updated run whose status is active; falls
    back to the most recently updated run overall. Returns None if the
    DB has no runs at all.
    """
    runs = list_runs(conn)
    if not runs:
        return None
    for r in runs:  # already sorted updated_at_ms DESC
        if (r.get("status") or "").lower() in _ACTIVE_STATUSES:
            return r["run_id"]
    return runs[0]["run_id"]


# --------------------------------------------------------------------------- #
# Snapshot diffing
# --------------------------------------------------------------------------- #


_BLOCKED_STATES = {"blocked", "approval_hang", "timer_hang"}


def _empty_delta(view: str | None, *, first: bool, elapsed_ms: int = 0) -> dict[str, Any]:
    return {
        "first_snapshot": first,
        "view": view,
        "state_counts_delta": {},
        "newly_running": [],
        "newly_failed": [],
        "newly_succeeded": [],
        "newly_blocked": [],
        "newly_abandoned": [],
        "state_transitions": [],
        "lease_age_crossed": [],
        "leases_expired_new": [],
        "new_node_ids": [],
        "removed_node_ids": [],
        "elapsed_ms": elapsed_ms,
    }


def diff_snapshots(
    prev: dict[str, Any] | None,
    curr: dict[str, Any],
    *,
    lease_warn_threshold: int = 60,
) -> dict[str, Any]:
    """Compute a delta between two ``build_snapshot`` outputs.

    Only meaningful for ``view == "nodes"`` snapshots; for ``"runs"`` or
    ``"node"`` views returns a near-empty delta marked first_snapshot.
    """
    view = curr.get("view") if isinstance(curr, dict) else None
    if curr.get("view") != "nodes":
        return _empty_delta(view, first=True)
    if prev is None or prev.get("view") != "nodes":
        return _empty_delta(view, first=True)

    elapsed_ms = 0
    prev_run = prev.get("run") or {}
    curr_run = curr.get("run") or {}
    if prev_run.get("updated_at_ms") and curr_run.get("updated_at_ms"):
        elapsed_ms = max(
            0, int(curr_run["updated_at_ms"]) - int(prev_run["updated_at_ms"])
        )

    prev_nodes = {n["node_id"]: n for n in prev.get("nodes", [])}
    curr_nodes = {n["node_id"]: n for n in curr.get("nodes", [])}

    transitions: list[dict[str, Any]] = []
    newly_running: list[str] = []
    newly_failed: list[str] = []
    newly_succeeded: list[str] = []
    newly_blocked: list[str] = []
    newly_abandoned: list[str] = []
    new_node_ids: list[str] = []

    for nid, n in curr_nodes.items():
        prev_n = prev_nodes.get(nid)
        state = (n.get("state") or "").lower()
        if prev_n is None:
            new_node_ids.append(nid)
            # Treat first appearance as a transition from "<new>" so the
            # LLM can see freshly-spawned nodes; categorize by destination.
            transitions.append(
                {
                    "node_id": nid,
                    "from": "<new>",
                    "to": state,
                    "tool": n.get("tool") or "",
                    "failure_kind": n.get("last_failure_kind") or "",
                }
            )
        else:
            prev_state = (prev_n.get("state") or "").lower()
            if prev_state == state:
                continue
            transitions.append(
                {
                    "node_id": nid,
                    "from": prev_state,
                    "to": state,
                    "tool": n.get("tool") or "",
                    "failure_kind": n.get("last_failure_kind") or "",
                }
            )

        if state == "running":
            newly_running.append(nid)
        elif state == "failed":
            newly_failed.append(nid)
        elif state == "succeeded":
            newly_succeeded.append(nid)
        elif state in _BLOCKED_STATES:
            newly_blocked.append(nid)
        elif state == "abandoned":
            newly_abandoned.append(nid)

    removed_node_ids = [nid for nid in prev_nodes if nid not in curr_nodes]

    prev_counts = prev.get("state_counts") or {}
    curr_counts = curr.get("state_counts") or {}
    keys = set(prev_counts) | set(curr_counts)
    state_counts_delta = {
        k: int(curr_counts.get(k, 0)) - int(prev_counts.get(k, 0))
        for k in keys
        if int(curr_counts.get(k, 0)) != int(prev_counts.get(k, 0))
    }

    # Lease age crossings: the per-node lease dict on each snapshot
    # already carries age_sec / expired (computed against now in
    # list_nodes). We compare prev vs curr on each shared node.
    lease_age_crossed: list[dict[str, Any]] = []
    leases_expired_new: list[str] = []
    for nid, n in curr_nodes.items():
        lease = n.get("lease")
        if not lease:
            continue
        age = lease.get("age_sec")
        if not isinstance(age, int):
            age = 0
        prev_lease = (prev_nodes.get(nid) or {}).get("lease") or {}
        prev_age = prev_lease.get("age_sec") if isinstance(prev_lease.get("age_sec"), int) else 0
        if age >= lease_warn_threshold and prev_age < lease_warn_threshold:
            lease_age_crossed.append(
                {
                    "node_id": nid,
                    "worker_id": lease.get("worker_id"),
                    "age_sec": age,
                    "threshold": lease_warn_threshold,
                    "expired": bool(lease.get("expired")),
                }
            )
        if lease.get("expired") and not prev_lease.get("expired"):
            leases_expired_new.append(nid)

    transitions = transitions[:20]

    return {
        "first_snapshot": False,
        "view": view,
        "state_counts_delta": state_counts_delta,
        "newly_running": newly_running,
        "newly_failed": newly_failed,
        "newly_succeeded": newly_succeeded,
        "newly_blocked": newly_blocked,
        "newly_abandoned": newly_abandoned,
        "state_transitions": transitions,
        "lease_age_crossed": lease_age_crossed,
        "leases_expired_new": leases_expired_new,
        "new_node_ids": new_node_ids,
        "removed_node_ids": removed_node_ids,
        "elapsed_ms": elapsed_ms,
    }


# --------------------------------------------------------------------------- #
# Heuristics
# --------------------------------------------------------------------------- #


def evaluate_heuristics(
    curr: dict[str, Any],
    delta: dict[str, Any],
    *,
    lease_age_warn: int,
    lease_age_crit: int,
    elapsed_wall_ms: int,
    no_progress_window_ms: int,
    prev_running: int | None,
    governance_verdict: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run cheap rules. Returns alert dicts; empty list when clean.

    ``elapsed_wall_ms`` is the wall-clock time since the last observed
    state transition (not since the previous poll) — see run_monitor's
    ``last_progress_wall_ms``. The no_progress rule compares it against
    ``no_progress_window_ms``.

    ``governance_verdict`` is the payload of the run's latest
    ``ccx.governance.verdict`` event (or ``None`` when no such event exists —
    the default, and the only state unless ``CCX_EMIT_GOVERNANCE_EVENTS`` is on).
    When present it powers the performative-completion rule; absent, that rule is
    inert and the heuristic set is byte-identical to before.
    """
    if curr.get("view") != "nodes":
        return []
    counts = curr.get("state_counts") or {}
    delta_counts = delta.get("state_counts_delta") or {}
    hits: list[dict[str, Any]] = []

    newly_failed = list(delta.get("newly_failed") or [])
    if not delta.get("first_snapshot") and newly_failed:
        hits.append(
            {
                "rule": "failed_increased",
                "severity": "error",
                "message": (
                    f"{len(newly_failed)} node(s) "
                    "transitioned to failed"
                ),
                "node_ids": newly_failed,
                "delta_failed": int(delta_counts.get("failed", 0)),
            }
        )

    if delta.get("leases_expired_new"):
        hits.append(
            {
                "rule": "lease_expired",
                "severity": "error",
                "message": "lease expired on running node(s)",
                "node_ids": list(delta.get("leases_expired_new") or []),
            }
        )

    crit_nodes = [
        n["node_id"]
        for n in curr.get("nodes", [])
        if (n.get("lease") or {}).get("age_sec") is not None
        and (n["lease"].get("age_sec") or 0) >= lease_age_crit
    ]
    if crit_nodes:
        hits.append(
            {
                "rule": "lease_critical",
                "severity": "error",
                "message": f"lease age >= {lease_age_crit}s on running node(s)",
                "node_ids": crit_nodes,
            }
        )
    else:
        warn_nodes = [
            n["node_id"]
            for n in curr.get("nodes", [])
            if (n.get("lease") or {}).get("age_sec") is not None
            and (n["lease"].get("age_sec") or 0) >= lease_age_warn
        ]
        if warn_nodes:
            hits.append(
                {
                    "rule": "lease_starving",
                    "severity": "warn",
                    "message": f"lease age >= {lease_age_warn}s on running node(s)",
                    "node_ids": warn_nodes,
                }
            )

    if delta.get("newly_blocked"):
        hits.append(
            {
                "rule": "approval_hang_appeared",
                "severity": "warn",
                "message": "node(s) entered a blocked/approval-hang state",
                "node_ids": list(delta.get("newly_blocked") or []),
            }
        )
    if delta.get("newly_abandoned"):
        hits.append(
            {
                "rule": "abandoned_appeared",
                "severity": "warn",
                "message": "node(s) entered abandoned state",
                "node_ids": list(delta.get("newly_abandoned") or []),
            }
        )

    running_now = int(counts.get("running", 0))
    if (
        not delta.get("first_snapshot")
        and running_now > 0
        and not delta.get("state_transitions")
        and elapsed_wall_ms >= no_progress_window_ms
    ):
        hits.append(
            {
                "rule": "no_progress",
                "severity": "warn",
                "message": (
                    f"{running_now} running node(s) but no state transitions in "
                    f"{elapsed_wall_ms // 1000}s"
                ),
                "node_ids": [],
            }
        )

    run_status = ((curr.get("run") or {}).get("status") or "").lower()
    if (
        prev_running is not None
        and prev_running > 0
        and running_now == 0
        and run_status not in _TERMINAL_STATUSES
    ):
        hits.append(
            {
                "rule": "running_dropped_to_zero",
                "severity": "warn",
                "message": (
                    f"running count fell from {prev_running} to 0 while run "
                    f"status is {run_status or '?'}"
                ),
                "node_ids": [],
            }
        )

    perf = _heuristic_performative_completion(counts, governance_verdict)
    if perf is not None:
        hits.append(perf)

    return hits


def _heuristic_performative_completion(
    counts: dict[str, int],
    governance_verdict: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Warn when nodes are all green but the run-level governance verdict is NO.

    The motivating failure: a goal-mode run whose every node SUCCEEDED — so the
    node-level view reads as clean success — while its ``goal_verdict.passed``
    (or ``run_audit_verdict.passed``) is ``False``. The operator, seeing only the
    node states, is misled into "inspect the artifacts, not the flags". This rule
    promotes the run-level truth into a ``warn`` alert.

    Fires only on an EXPLICIT not-passed run-level verdict
    (``governance_verdict["passed"] is False``) with a clean node view
    (≥1 succeeded, 0 failed, 0 abandoned). The negative control is the Goodhart
    guard: all-green + ``passed=True`` (or no governance verdict at all) must NOT
    fire. A run that already has a failed/abandoned node is covered by the
    ``failed_increased`` / ``abandoned_appeared`` rules, so we stay silent there
    to avoid redundant noise.
    """
    if not isinstance(governance_verdict, dict):
        return None
    if governance_verdict.get("passed") is not False:
        return None
    succeeded = int(counts.get("succeeded", 0) or 0)
    failed = int(counts.get("failed", 0) or 0)
    abandoned = int(counts.get("abandoned", 0) or 0)
    if succeeded <= 0 or failed > 0 or abandoned > 0:
        return None
    reasons: list[str] = []
    for label, key in (
        ("contract", "contract_verdict"),
        ("run_audit", "run_audit_verdict"),
        ("goal", "goal_verdict"),
    ):
        sub = governance_verdict.get(key)
        if isinstance(sub, dict) and sub.get("passed") is False:
            reason = sub.get("stop_reason") or sub.get("status") or "?"
            reasons.append(f"{label}:{reason}")
    suffix = f" ({', '.join(reasons)})" if reasons else ""
    return {
        "rule": "performative_completion",
        "severity": "warn",
        "message": (
            f"all {succeeded} node(s) succeeded but the run-level governance "
            f"verdict is NOT passed{suffix}"
        ),
        "node_ids": [],
    }


def latest_governance_verdict(
    conn: sqlite3.Connection, run_id: str,
) -> dict[str, Any] | None:
    """Payload of the run's most recent ``ccx.governance.verdict`` event, or None.

    Read-only; returns ``None`` on any error or when no such event exists (the
    default, since emission is opt-in via ``CCX_EMIT_GOVERNANCE_EVENTS``).
    """
    try:
        cur = conn.execute(
            "SELECT payload_json FROM events "
            "WHERE run_id = ? AND kind = ? "
            "ORDER BY sequence DESC LIMIT 1",
            (run_id, "ccx.governance.verdict"),
        )
        row = cur.fetchone()
    except sqlite3.Error:
        return None
    if not row:
        return None
    try:
        data = json.loads(row[0])
    except (TypeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _alert_key(hit: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    node_ids = hit.get("node_ids") or []
    if not isinstance(node_ids, list):
        node_ids = []
    return (
        str(hit.get("rule") or ""),
        tuple(sorted(str(node_id) for node_id in node_ids)),
    )


def _filter_alert_cooldown(
    hits: Sequence[dict[str, Any]],
    *,
    last_alert_at: dict[tuple[str, tuple[str, ...]], float],
    now: float,
    cooldown_s: float,
) -> list[dict[str, Any]]:
    if cooldown_s <= 0:
        for hit in hits:
            last_alert_at[_alert_key(hit)] = now
        return list(hits)
    out: list[dict[str, Any]] = []
    for hit in hits:
        key = _alert_key(hit)
        last = last_alert_at.get(key)
        if last is not None and (now - last) < cooldown_s:
            continue
        last_alert_at[key] = now
        out.append(hit)
    return out


# --------------------------------------------------------------------------- #
# Snapshot compaction (for LLM input)
# --------------------------------------------------------------------------- #


_NODE_CAP = 80


def compact_snapshot(snap: dict[str, Any]) -> dict[str, Any]:
    """Strip heavy fields and cap the node list for LLM consumption."""
    if snap.get("view") != "nodes":
        return {"view": snap.get("view"), "run": snap.get("run")}

    nodes = snap.get("nodes") or []
    nodes_sorted = sorted(
        nodes,
        key=lambda n: int(n.get("updated_at_ms") or 0),
        reverse=True,
    )
    truncated = len(nodes_sorted) > _NODE_CAP
    kept = nodes_sorted[:_NODE_CAP]
    compact_nodes: list[dict[str, Any]] = []
    for n in kept:
        lease = n.get("lease") or {}
        compact_nodes.append(
            {
                "node_id": n.get("node_id"),
                "state": n.get("state"),
                "tool": n.get("tool"),
                "attempts": n.get("attempts"),
                "last_failure_kind": n.get("last_failure_kind") or "",
                "lease_age_sec": lease.get("age_sec"),
                "lease_expired": bool(lease.get("expired")) if lease else False,
            }
        )

    out = {
        "view": "nodes",
        "run": {
            "run_id": (snap.get("run") or {}).get("run_id"),
            "status": (snap.get("run") or {}).get("status"),
        },
        "state_counts": snap.get("state_counts") or {},
        "nodes": compact_nodes,
    }
    if truncated:
        out["nodes_truncated"] = {
            "kept": len(kept),
            "total": len(nodes_sorted),
            "note": "showing most-recently-updated nodes only",
        }
    return out


# --------------------------------------------------------------------------- #
# LLM client wrapper
# --------------------------------------------------------------------------- #


_SYSTEM_PROMPT = """\
You are a supervisor for a ccx multi-agent workflow engine. You receive
periodic compact snapshots of a running workflow plus a delta describing
changes since the last check, and a list of cheap heuristic findings.

Your job: decide whether a human operator should be alerted, and why.
Bias toward silence. The heuristics already catch obvious failures —
your value is in subtler patterns: retry storms, dependency stalls,
worker imbalance, repeated failure on the same tool, suspicious lack
of progress that heuristics missed.

Respond with EXACTLY one JSON object, no prose, no markdown fence:
{
  "alert": <bool>,
  "severity": "info" | "warn" | "error",
  "summary": "<one sentence, <120 chars>",
  "suspect_nodes": ["node_id", ...],
  "reasoning": "<2-4 sentences for the log; not shown in headline>"
}
If nothing actionable, set alert=false and severity="info".\
"""


def _parse_llm_response(text: str) -> dict[str, Any] | None:
    """Best-effort JSON extraction. Tolerates fences and surrounding prose."""
    if not isinstance(text, str):
        return None
    return parse_llm_json(
        text,
        schema_name="llm_monitor",
        fallback_factory=lambda _raw: None,
        expected_type=dict,
    )


class MonitorLLM:
    """Wraps an LLMApiClient and produces structured monitor verdicts."""

    def __init__(self, client: Any, *, model: str | None, max_tokens: int):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    def assess(
        self,
        compact: dict[str, Any],
        delta: dict[str, Any],
        hits: list[dict[str, Any]],
        *,
        elapsed_since_last_check_sec: float,
    ) -> dict[str, Any] | None:
        run = compact.get("run") or {}
        user_msg = (
            f"elapsed_since_last_check_sec: {elapsed_since_last_check_sec:.1f}\n"
            f"run_id: {run.get('run_id')}\n"
            f"run_status: {run.get('status')}\n"
            f"state_counts: {json.dumps(compact.get('state_counts') or {}, ensure_ascii=False)}\n"
            f"delta: {json.dumps(delta, ensure_ascii=False, default=str)}\n"
            f"heuristic_hits: {json.dumps(hits, ensure_ascii=False)}\n"
            f"nodes (compact, max {_NODE_CAP}): "
            f"{json.dumps(compact.get('nodes') or [], ensure_ascii=False)}"
        )
        full = _SYSTEM_PROMPT + "\n\n" + user_msg
        try:
            raw = self.client.text_chat(full, is_stream=False)
        except Exception as exc:  # pragma: no cover — surfaced to caller
            raise RuntimeError(f"LLM call failed: {exc}") from exc
        if not isinstance(raw, str):
            try:
                raw = "".join(raw)  # in case a streaming iterator slipped through
            except Exception:
                return None
        parsed = _parse_llm_response(raw)
        if not isinstance(parsed, dict):
            return None
        # Normalize fields with defaults so downstream code doesn't KeyError.
        parsed.setdefault("alert", False)
        parsed.setdefault("severity", "info")
        parsed.setdefault("summary", "")
        parsed.setdefault("suspect_nodes", [])
        parsed.setdefault("reasoning", "")
        return parsed


# --------------------------------------------------------------------------- #
# Alert sink
# --------------------------------------------------------------------------- #


class AlertSink:
    """Emits alerts to stderr (always) and an optional JSONL log file."""

    def __init__(
        self,
        *,
        log_file: Path | None = None,
        quiet: bool = False,
        run_id: str | None = None,
        stderr_stream: Any = None,
    ):
        self.log_file = log_file
        self.quiet = quiet
        self.run_id = run_id
        self.stderr = stderr_stream if stderr_stream is not None else sys.stderr
        if log_file is not None:
            log_file.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, severity: str, source: str, payload: dict[str, Any]) -> None:
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "ts_ms": int(time.time() * 1000),
            "run_id": self.run_id,
            "severity": severity,
            "source": source,
            "payload": payload,
        }
        if self.log_file is not None:
            try:
                with self.log_file.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            except OSError as exc:
                print(
                    f"warning: alert log write failed: {exc}",
                    file=self.stderr,
                )

        if self.quiet and severity == "info":
            return
        self._print_human(record)

    def _print_human(self, record: dict[str, Any]) -> None:
        sev = (record.get("severity") or "info").upper()
        src = record.get("source") or "?"
        run = record.get("run_id") or "-"
        payload = record.get("payload") or {}
        if record.get("source") == "heuristic":
            rule = payload.get("rule") or "?"
            msg = payload.get("message") or ""
            nids = payload.get("node_ids") or []
            nid_str = f" nodes={','.join(nids[:5])}" if nids else ""
            line = f"[{record['ts']}] {sev} {src}/{rule} run={run} {msg}{nid_str}"
        elif record.get("source") == "llm":
            summary = payload.get("summary") or "(no summary)"
            suspects = payload.get("suspect_nodes") or []
            sup = f" suspects={','.join(suspects[:5])}" if suspects else ""
            line = f"[{record['ts']}] {sev} {src} run={run} {summary}{sup}"
        else:
            # llm_error or other
            line = (
                f"[{record['ts']}] {sev} {src} run={run} "
                f"{json.dumps(payload, ensure_ascii=False, default=str)}"
            )
        print(line, file=self.stderr, flush=True)


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #


def run_monitor(
    *,
    conn: sqlite3.Connection,
    run_id: str,
    interval: float,
    llm_cadence: float,
    llm_debounce: float,
    enable_llm: bool,
    llm: MonitorLLM | None,
    sink: AlertSink,
    lease_age_warn: int,
    lease_age_crit: int,
    once: bool = False,
    sleeper: Any = time.sleep,
    monotonic: Any = time.monotonic,
    wall_clock: Any = time.time,
) -> int:
    """Polling loop. Returns 0 on clean exit (Ctrl-C or once=True)."""
    prev_snapshot: dict[str, Any] | None = None
    prev_running: int | None = None
    # Baseline for the no_progress rule: wall-clock time of the last
    # observed state transition (or monitor start / last no_progress
    # alert). Must NOT reset every poll — elapsed_wall_ms has to measure
    # the stall duration, and a per-poll reset caps it at ~interval so
    # the rule could never reach no_progress_window_ms.
    last_progress_wall_ms = int(wall_clock() * 1000)
    last_llm_at = 0.0
    last_escalation_at = 0.0
    last_alert_at: dict[tuple[str, tuple[str, ...]], float] = {}
    no_progress_window_ms = int(max(llm_cadence, 1.0) * 1000)

    try:
        while True:
            snap = build_watch_snapshot(
                conn, run_id=run_id, state_filter=None, node_id=None
            )
            delta = diff_snapshots(
                prev_snapshot, snap, lease_warn_threshold=lease_age_warn
            )
            now_wall_ms = int(wall_clock() * 1000)
            elapsed_wall_ms = max(0, now_wall_ms - last_progress_wall_ms)

            hits = evaluate_heuristics(
                snap,
                delta,
                lease_age_warn=lease_age_warn,
                lease_age_crit=lease_age_crit,
                elapsed_wall_ms=elapsed_wall_ms,
                no_progress_window_ms=no_progress_window_ms,
                prev_running=prev_running,
                governance_verdict=latest_governance_verdict(conn, run_id),
            )
            now = monotonic()
            hits = _filter_alert_cooldown(
                hits,
                last_alert_at=last_alert_at,
                now=now,
                cooldown_s=llm_debounce,
            )
            for h in hits:
                sink.emit(h["severity"], "heuristic", h)

            if enable_llm and llm is not None:
                escalation = bool(hits) and (now - last_escalation_at) >= llm_debounce
                periodic = (
                    (now - last_llm_at) >= llm_cadence
                    and not delta.get("first_snapshot")
                )
                if escalation or periodic:
                    elapsed_since = now - last_llm_at if last_llm_at else float("inf")
                    elapsed_since_disp = (
                        elapsed_since if elapsed_since != float("inf") else 0.0
                    )
                    try:
                        verdict = llm.assess(
                            compact_snapshot(snap),
                            delta,
                            hits,
                            elapsed_since_last_check_sec=elapsed_since_disp,
                        )
                    except Exception as exc:
                        sink.emit(
                            "warn",
                            "llm_error",
                            {"error": str(exc), "phase": "call"},
                        )
                        verdict = None
                    if verdict is None:
                        sink.emit(
                            "warn",
                            "llm_error",
                            {"error": "malformed or empty LLM response"},
                        )
                    elif verdict.get("alert"):
                        sev = verdict.get("severity") or "warn"
                        if sev not in {"info", "warn", "error"}:
                            sev = "warn"
                        sink.emit(sev, "llm", verdict)
                    else:
                        sink.emit("info", "llm", verdict)
                    last_llm_at = now
                    if escalation:
                        last_escalation_at = now

            prev_snapshot = snap
            prev_running = int((snap.get("state_counts") or {}).get("running", 0))
            if (
                delta.get("first_snapshot")
                or delta.get("state_transitions")
                or any(h["rule"] == "no_progress" for h in hits)
            ):
                # Progress observed (or we just alerted): restart the
                # stall clock. Re-arming on fire makes a continued stall
                # re-alert once per window instead of on every poll.
                last_progress_wall_ms = now_wall_ms

            if once:
                return 0
            sleeper(interval)
    except KeyboardInterrupt:
        print("", file=sys.stderr)
        return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m core.ccx.llm_monitor")
    p.add_argument("--cwd", default=".", help="ccx workspace (contains .ccx/runtime/runtime.db)")
    p.add_argument(
        "--run-id",
        default=None,
        help="run to monitor; defaults to the most recently updated active run",
    )
    p.add_argument("--interval", type=float, default=2.0, help="DB poll seconds (default 2.0)")
    p.add_argument(
        "--enable-llm",
        action="store_true",
        help="enable LLM assessment (off by default; LLM never invoked unless set)",
    )
    p.add_argument(
        "--llm-cadence",
        type=float,
        default=60.0,
        help="periodic LLM check interval seconds (default 60)",
    )
    p.add_argument(
        "--llm-debounce",
        type=float,
        default=15.0,
        help="min seconds between heuristic-triggered LLM escalations (default 15)",
    )
    p.add_argument("--model", default=None, help="LLM model name passed to LLMFactory")
    p.add_argument("--max-tokens", type=int, default=1024, help="LLM response cap")
    p.add_argument(
        "--lease-age-warn",
        type=int,
        default=60,
        help="seconds to trigger lease_starving (default 60)",
    )
    p.add_argument(
        "--lease-age-crit",
        type=int,
        default=180,
        help="seconds to trigger lease_critical (default 180)",
    )
    p.add_argument(
        "--log-file",
        default=None,
        help="append JSONL alert records here in addition to stderr",
    )
    p.add_argument("--quiet", action="store_true", help="suppress severity=info on stderr")
    p.add_argument("--once", action="store_true", help="run a single assessment cycle and exit")
    return p


def _build_default_llm(model: str | None, max_tokens: int) -> Any:
    """Return an LLMApiClient via LLMFactory. Raises on failure."""
    from core.llms.llm_factory import LLMFactory  # lazy import; LLM path is opt-in

    kwargs: dict[str, Any] = {"max_tokens": max_tokens}
    if model:
        kwargs["model"] = model
    return LLMFactory().get_instance("claude_client", **kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    db_path = resolve_db_path(args.cwd)
    try:
        conn = connect_ro(db_path)
    except (FileNotFoundError, sqlite3.OperationalError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        run_id = args.run_id
        if not run_id:
            run_id = pick_latest_active_run(conn)
            if not run_id:
                print(
                    "error: no runs found in DB; pass --run-id explicitly",
                    file=sys.stderr,
                )
                return 2
            print(f"monitoring auto-selected run: {run_id}", file=sys.stderr)

        llm: MonitorLLM | None = None
        if args.enable_llm:
            try:
                client = _build_default_llm(args.model, args.max_tokens)
            except Exception as exc:
                print(f"error: failed to build LLM client: {exc}", file=sys.stderr)
                return 2
            llm = MonitorLLM(client, model=args.model, max_tokens=args.max_tokens)

        sink = AlertSink(
            log_file=Path(args.log_file) if args.log_file else None,
            quiet=args.quiet,
            run_id=run_id,
        )

        return run_monitor(
            conn=conn,
            run_id=run_id,
            interval=args.interval,
            llm_cadence=args.llm_cadence,
            llm_debounce=args.llm_debounce,
            enable_llm=args.enable_llm,
            llm=llm,
            sink=sink,
            lease_age_warn=args.lease_age_warn,
            lease_age_crit=args.lease_age_crit,
            once=args.once,
        )
    finally:
        conn.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
