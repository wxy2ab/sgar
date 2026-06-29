"""AuditRunner — a read-only LLM turn that PROPOSES narrative-fidelity mismatches.

Phase 1 of the hard-feedback audit agent
(``docs/audit_agent_design_2026-06-28.md`` §1.5). The auditor reads a finished
run's narrative report (claim side) plus the pre-assembled machine ground-truth
(evidence side) and emits **structured candidate mismatches**. It NEVER decides a
finding — the deterministic teeth in :mod:`core.ccx.audit.consistency` re-derive
the truth and confirm/refuse each candidate. The LLM proposes; Python disposes.

Read-only by construction (mirrors :class:`ResearchRunner`): the cc QueryEngine's
tool registry is filtered through ``restrict_tool_registry`` (FAIL-CLOSED — it
raises if the registry shape is unrecognized), so ``file_edit`` / ``shell`` are
gone for this turn even if cc's default registry exposes them. This is NEVER the
full ``CcAgentRunner``. The ground-truth is pre-assembled into the prompt, so the
auditor needs no DB/exec tools at all; the read-only file tools remain only so a
future kind (cited-span verification) can re-read evidence.

Hard timeout: the bounded wrapper :func:`propose_candidates_bounded` runs the
turn on a daemon thread with ``join(timeout)`` (the ``_call_llm_bounded`` pattern;
never ``ThreadPoolExecutor`` — its atexit join can deadlock on a hung stream). On
timeout it returns ``[]`` — the audit is INFORM-only, so a hung auditor degrades
to "no advisories", never a blocked or flipped gate.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from typing import Any

from core.cc.config import CCConfig
from core.cc.runtime import build_default_query_engine

from .cc_agent import _is_turn_timeout_message, _run_in_fresh_loop
from .read_only_runner import DEFAULT_READ_ONLY_WHITELIST, restrict_tool_registry

logger = logging.getLogger(__name__)

__all__ = [
    "AuditRunner",
    "parse_audit_candidates",
    "propose_candidates_bounded",
    "render_ground_truth_brief",
    "AUDIT_TOOL_WHITELIST",
]

#: The auditor runs with the SAME read-only whitelist as research/doc/ask — no
#: widening. The gating test asserts ``file_edit``/``shell`` are absent.
AUDIT_TOOL_WHITELIST = DEFAULT_READ_ONLY_WHITELIST


_AUDIT_SYSTEM_PROMPT = """\
You are a READ-ONLY narrative-fidelity AUDITOR. You are given (1) a finished \
run's REPORT text (what the agent claimed) and (2) the machine GROUND-TRUTH the \
run actually produced (the goal verdict, each acceptance check's real pass/fail, \
the run status and node counts). Your ONE job: find places where the REPORT \
makes a positive claim (success / completion / a specific check passing) that \
the GROUND-TRUTH contradicts.

You do NOT decide anything. You PROPOSE candidates; a deterministic checker \
re-reads the ground-truth and confirms or rejects each one. So:
  * Only cite claims that are VERBATIM spans of the REPORT (copy the exact words \
into `claim_text`). A span that is not in the report is auto-rejected.
  * Only reference check ids (`criterion_id`) that appear in the GROUND-TRUTH. A \
made-up id is auto-rejected.
  * If the report is consistent with the ground-truth, emit an EMPTY list. Do \
NOT invent mismatches to seem useful.

Mismatch kinds (use exactly these strings):
  * "verdict_contradiction" — the report asserts overall success/completion but \
the ground-truth goal verdict did NOT pass. locator: {}.
  * "check_outcome_contradiction" — the report says a SPECIFIC named check \
passed, but that check's real outcome is a genuine FAIL. \
locator: {"criterion_id": "<the id from ground-truth>"}.
  * "node_state_contradiction" — the report asserts all steps completed, but the \
run finished degraded (abandoned/failed nodes). locator: {}.

Output: a final message with THIS exact JSON shape and no surrounding prose:

{
  "candidates": [
    {"mismatch_kind": "<one of the three>", "locator": {...}, \
"claim_text": "<verbatim report span>", "reasoning": "<one sentence>"}
  ]
}

Emit {"candidates": []} when there is no contradiction.
"""


def render_ground_truth_brief(ground_truth: dict[str, Any]) -> str:
    """Render the machine ground-truth as a compact, prompt-friendly block.

    Gives the LLM the exact ``criterion_id``s and real pass/fail so it can
    reference them precisely (and so it cannot need a DB/exec tool to discover
    them). ``ground_truth`` is the dict from
    :func:`core.ccx.audit.consistency.assemble_ground_truth`.
    """
    gt = ground_truth or {}
    lines: list[str] = []
    lines.append(f"goal_verdict.passed = {gt.get('passed')!r}")
    lines.append(f"run status = {gt.get('status')!r}")
    counts = gt.get("counts") or {}
    lines.append(
        "node counts = succeeded={s}, failed={f}, abandoned={a}".format(
            s=counts.get("succeeded", 0), f=counts.get("failed", 0),
            a=counts.get("abandoned", 0),
        )
    )
    degraded = gt.get("degraded")
    lines.append(f"degraded_completion = {degraded!r}")
    checks = gt.get("check_evidence") or {}
    if checks:
        lines.append("acceptance checks:")
        for cid, e in checks.items():
            lines.append(
                f"  - {cid}: passed={e.get('passed')!r}, "
                f"executable={e.get('executable')!r}, command={e.get('command')!r}"
            )
    else:
        lines.append("acceptance checks: (none recorded)")
    return "\n".join(lines)


def _build_audit_user_prompt(report_text: str, ground_truth: dict[str, Any]) -> str:
    return (
        "## REPORT (the agent's narrative — the CLAIM side)\n"
        f"{(report_text or '').strip() or '(empty report)'}\n\n"
        "## GROUND-TRUTH (machine truth — the EVIDENCE side)\n"
        f"{render_ground_truth_brief(ground_truth)}\n\n"
        "Emit the JSON candidates now (empty list if the report is consistent)."
    )


def parse_audit_candidates(final_text: str) -> list[dict[str, Any]]:
    """Parse the auditor's final text into a list of candidate dicts.

    Robust to fenced markdown / prose preamble (reuses doc-mode's extractor).
    Drops malformed entries and any unknown ``mismatch_kind`` so only
    well-formed candidates reach the deterministic checker. Returns ``[]`` on any
    parse failure (the audit is best-effort / INFORM-only).
    """
    from ..audit.consistency import MISMATCH_KINDS
    from ..modes.doc import _robust_json_object

    parsed = _robust_json_object((final_text or "").strip())
    if not isinstance(parsed, dict):
        return []
    raw = parsed.get("candidates")
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("mismatch_kind") or "")
        if kind not in MISMATCH_KINDS:
            continue
        locator = item.get("locator")
        out.append({
            "mismatch_kind": kind,
            "locator": locator if isinstance(locator, dict) else {},
            "claim_text": str(item.get("claim_text") or ""),
            "reasoning": str(item.get("reasoning") or ""),
        })
    return out


@dataclass(slots=True)
class AuditRunner:
    """Drive one read-only audit turn through cc's QueryEngine (fail-closed)."""

    cc_config: CCConfig
    llm_provider: Any  # LLMClientProvider
    cwd: str
    max_tool_rounds: int | None = None
    mode_name: str = "audit"

    def propose_candidates(
        self, *, report_text: str, ground_truth: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Run the audit turn (sync entry) and return parsed candidate mismatches."""
        return _run_in_fresh_loop(self._run_async(report_text, ground_truth))

    async def _run_async(
        self, report_text: str, ground_truth: dict[str, Any],
    ) -> list[dict[str, Any]]:
        engine = build_default_query_engine(
            cwd=self.cwd, config=self.cc_config, llm_client_provider=self.llm_provider,
        )
        # FAIL-CLOSED read-only enforcement: file_edit / shell are removed.
        removed, kept = restrict_tool_registry(engine)
        logger.debug(
            "audit runner: filtered cc tool registry, removed %d non-read-only "
            "tools (kept: %s)", removed, kept,
        )
        framed = (
            f"<system>\n{_AUDIT_SYSTEM_PROMPT}\n</system>\n\n"
            f"{_build_audit_user_prompt(report_text, ground_truth)}"
        )
        final_text = ""
        turn_timed_out = False
        try:
            async for event in engine.submit_message(
                framed, max_tool_rounds=self.max_tool_rounds, purpose="audit",
            ):
                msg = getattr(event, "message", None)
                if msg is None:
                    continue
                if _is_turn_timeout_message(msg):
                    turn_timed_out = True
                if (
                    getattr(msg, "role", "") == "assistant"
                    and getattr(msg, "kind", "") == "assistant_text"
                ):
                    final_text = str(getattr(msg, "content", ""))
        finally:
            engine.close()
        if turn_timed_out:
            raise TimeoutError(final_text or "audit turn timed out")
        return parse_audit_candidates(final_text)


def propose_candidates_bounded(
    runner: AuditRunner,
    *,
    report_text: str,
    ground_truth: dict[str, Any],
    timeout_s: float,
) -> list[dict[str, Any]]:
    """Hard-timeout wrapper around ``runner.propose_candidates``.

    Runs the turn on a DAEMON thread and ``join(timeout_s)``; on timeout the
    thread is abandoned (daemon ⇒ never blocks process exit) and ``[]`` is
    returned. Never ``ThreadPoolExecutor`` (its atexit join deadlocks on a hung
    DeepSeek stream — the 7h-hang vector). INFORM-only: a hung auditor simply
    yields no advisories.
    """
    box: dict[str, Any] = {"candidates": [], "done": False}

    def _worker() -> None:
        try:
            box["candidates"] = runner.propose_candidates(
                report_text=report_text, ground_truth=ground_truth,
            )
        except Exception:
            logger.warning("ccx audit: auditor turn raised", exc_info=True)
        finally:
            box["done"] = True

    thread = threading.Thread(target=_worker, name="ccx-audit-propose", daemon=True)
    thread.start()
    thread.join(timeout_s)
    if not box["done"]:
        logger.warning(
            "ccx audit: auditor did not return within %.0fs; abandoning the "
            "thread and emitting no advisories", timeout_s,
        )
        return []
    result = box["candidates"]
    return result if isinstance(result, list) else []
