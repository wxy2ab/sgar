"""Run-level externalized hard audit for ccx — the per-spawn contract, hoisted.

``core.ccx.agents.governed_spawn`` attaches a machine-verified, bounded
verify-repair contract to a *single spawned subagent*. This module hoists the
SAME idea up to the **run boundary**: after a whole ``CodeAgent.run`` DAG
reaches quiescence, an external, deterministic judge (``run_criterion_check``,
exit-0-is-truth, sharing no context with the producing DAG) decides "is this
run done?" — not an LLM grading an accumulating context.

Why a separate module from ``governed_spawn`` (rather than reuse
``run_governed_spawn`` directly):

* **Granularity.** ``run_governed_spawn`` gates *one agent turn* and returns a
  ``SubagentResult``; the unit here is a full ``AgentRunRequest`` →
  ``AgentRunResult`` re-drive of the entire v5 DAG.
* **The ``spawned_children → skipped`` branch must be removed.** At run level the
  root *always* decomposes into children, so that branch would skip every run.
  The run-level loop never inspects ``result.subtasks`` and never self-declares
  success — only a green ``run_criterion_check`` passes it.
* **A tighter iteration ceiling.** Each run-level iteration re-drives the ENTIRE
  DAG (plan + spec + parallel agents + recursive spawns) — roughly 10× the cost
  of a per-spawn iteration. ``parse_contract`` is reused verbatim (it still
  clamps ``loop.max_iters`` to ``governed_spawn._HARD_MAX_ITERS`` = 10), and on
  top of that ``run_run_audit_loop`` applies the strictly-tighter
  :data:`_HARD_MAX_ITERS_RUN`. This only ever *lowers* the bound, never raises it.

The contract JSON shape is identical to the per-spawn ``ccx_contract``
(``{acceptance:[{id,text,check}], verify, loop}``) and is parsed by the same
``parse_contract``. It is carried under a DISTINCT metadata key
(:data:`CCX_RUN_CONTRACT_METADATA_KEY`) so the two never collide, and that key is
never lifted into node metadata / ``INHERITABLE_METADATA_KEYS`` — a run-level
contract governs the whole run, it must not auto-propagate to spawned children.

Default-OFF: with no contract present (no ctor knob and no metadata key) this
module is never engaged, and ``CodeAgent.run`` drives the DAG exactly once as
before — the only observable difference is that ``session_snapshot`` carries a
new ``run_audit_verdict`` key whose value is ``None`` (behaviour unchanged).

**Documented limitation (do NOT engineer around it in the MVC):** a run-level
re-drive is for **idempotent / workspace-converging** tasks. Side-effecting
tasks (git commits, external writes) and per-iteration memory finalization may
double-apply across iterations — such flows should use the per-spawn contract
or ``WatchModeRunner`` (which has commit-scoping), not run-level audit.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Awaitable, Callable

from core.cc.api import AgentRunRequest, AgentRunResult

from ..sgar.checks import (
    CheckOutcome,
    check_unrunnable,
    run_criterion_check,
    stop_on_unrunnable_enabled,
)
from .governed_spawn import CONTRACT_METADATA_KEY, SpawnContract, parse_contract
from .incremental_verify import select_criteria_for_changes
from .progress import EverPassedTracker, monotone_progress_enabled, observe_progress
from .syndrome import build_syndrome, changed_paths_from_git, format_syndrome_detail

logger = logging.getLogger(__name__)

#: Metadata key a caller writes the run-level contract under. Deliberately
#: DISTINCT from ``governed_spawn.CONTRACT_METADATA_KEY`` ("ccx_contract") so a
#: per-spawn contract and a run-level contract never collide, and deliberately
#: NOT added to ``INHERITABLE_METADATA_KEYS`` / lifted into node metadata: a
#: run-level contract governs the whole run and must not auto-propagate to the
#: children the run spawns.
CCX_RUN_CONTRACT_METADATA_KEY = "ccx_run_contract"

#: Hard ceiling on run-level iterations, applied ON TOP of ``parse_contract``'s
#: own ``_HARD_MAX_ITERS`` (=10) clamp. Tighter because each run-level iteration
#: re-drives the WHOLE DAG (~10× a per-spawn iteration); this only lowers the
#: effective bound, never raises it. See ``run_run_audit_loop`` (added in the
#: outer-loop commit).
_HARD_MAX_ITERS_RUN = 3

#: Operator-facing surfacing of the documented re-drive limitation (see the
#: module docstring). The double-apply itself is by_design with a stated
#: mitigation (max_iters=1); the gap an adversarial probe confirmed is that the
#: warning lived ONLY in source docstrings — at runtime, a verdict carrying
#: ``iters > 1`` gave the bare count but never the *interpretation* that a
#: non-idempotent step re-applied. These two surfaces (a ``log`` line on the
#: first re-drive + ``run_audit_verdict.non_idempotency_warning`` /
#: ``re_drives``) make it visible to an operator who never read the source.
#: Additive and conditional on ``iters > 1`` — the single-iteration path (the
#: common case and the documented mitigation) is byte-identical to before.
_REDRIVE_LIVE_WARNING = (
    "re-driving the WHOLE DAG (attempt {attempt}); each re-drive re-runs every "
    "step, so any NON-idempotent action (git commit, external write, network "
    "POST, persistent-memory write) re-applies. Pin loop.max_iters=1 (or use "
    "the per-spawn contract / WatchModeRunner) for non-idempotent tasks."
)


def _redrive_warning_text(iters: int) -> str:
    return (
        f"this run re-drove the whole DAG {iters}× (iterate-until-verified); each "
        f"re-drive re-runs every step, so any NON-idempotent action (git commit, "
        f"external write, network POST, persistent-memory write) applied up to "
        f"{iters}×. Pin loop.max_iters=1 (or use the per-spawn contract / "
        f"WatchModeRunner) for non-idempotent tasks."
    )


def parse_run_audit_contract(
    metadata: object, ctor_default: object,
) -> SpawnContract | None:
    """Resolve and parse a run-level audit contract, or ``None`` when absent.

    Source resolution (request metadata wins over the ctor default, mirroring
    how ``AgentRunRequest.max_tool_rounds`` overrides the ctor knob):

    * ``metadata[CCX_RUN_CONTRACT_METADATA_KEY]`` if present and non-``None``;
    * else ``ctor_default`` (the ``CodeAgent(run_audit_contract=...)`` knob);
    * else ``None`` (the default-off path).

    The resolved raw contract is parsed by reusing :func:`parse_contract`
    **unchanged** — we present it under the key that function expects. A
    present-but-malformed contract raises :class:`~.governed_spawn.ContractError`
    (fail-loud: a malformed run-level contract must never silently degrade to an
    un-audited single run). The ``CodeAgent.run`` boundary converts that into a
    distinctly-coded failed result for drop-in parity.
    """
    raw: object | None = None
    if isinstance(metadata, dict):
        candidate = metadata.get(CCX_RUN_CONTRACT_METADATA_KEY)
        if candidate is not None:
            raw = candidate
    if raw is None:
        raw = ctor_default
    if raw is None:
        return None
    # Reuse parse_contract verbatim by presenting the raw contract under the
    # metadata key it reads — same validation, same ContractError surface.
    return parse_contract({CONTRACT_METADATA_KEY: raw})


# --------------------------------------------------------------------------- #
# Outer verify-repair loop
# --------------------------------------------------------------------------- #

# drive_once(request) -> AgentRunResult, awaited. The caller injects its
# whole-DAG drive (build runtime → engine.run → build result → shutdown) here;
# it owns the asyncio.shield + cancellation discipline, so cancelling the run
# reaches whichever iteration's worker is live. Each invocation re-drives the
# ENTIRE DAG from scratch.
DriveOnce = Callable[[AgentRunRequest], Awaitable[AgentRunResult]]


def _noop_log(_message: str) -> None:
    return None


async def run_run_audit_loop(
    drive_once: DriveOnce,
    request: AgentRunRequest,
    contract: SpawnContract,
    *,
    cwd: str,
    check_timeout_s: float,
    log: Callable[[str], None] = _noop_log,
) -> AgentRunResult:
    """Drive the whole DAG inside a bounded, externally-judged verify-repair loop.

    Each iteration re-drives the entire DAG via ``drive_once`` to a terminal
    verdict, then an INDEPENDENT judge (``run_criterion_check``, run against the
    post-run workspace at ``cwd``) decides pass/fail — never the producer's own
    ``final_text``. The loop:

    * passes ONLY when every ``[check:]`` is green (it never self-declares
      success, and — unlike ``run_governed_spawn`` — it does NOT inspect
      ``result.subtasks`` / skip on spawned children: at run level the root
      always decomposes, and the checks are run after the whole DAG quiesces);
    * on failures, feeds the **check delta** (which checks are still red + their
      machine evidence) back into the next iteration's instruction — never a
      free-form "are you done?";
    * is bounded by the same progress rule as ``run_governed_spawn``
      (give up when the failing-check set stops shrinking for
      ``no_progress_stop`` rounds) and by ``max_iters``, clamped to the tighter
      run-level :data:`_HARD_MAX_ITERS_RUN`.

    Stamps the verdict onto ``session_snapshot["run_audit_verdict"]`` (same shape
    as the per-spawn ``contract_verdict``) and never raises for an ordinary
    failed contract — that is a ``passed=False`` verdict, the honest outcome.
    """
    criteria = contract.checkable_criteria()

    # verify='none' (a 'check' contract with zero [check:] is rejected at parse
    # time, so an empty criteria set here means verify='none'): drive once,
    # no gating — mirror run_governed_spawn's ungated path.
    if contract.verify == "none" or not criteria:
        result = await drive_once(request)
        # Ungated is not verified completion (principle 6).
        return _stamp(result, {
            "passed": False,
            "status": "ungated",
            "verify": contract.verify,
            "iters": 1,
            "stop_reason": "no_checks",
            "evidence": [],
        })

    max_iters = min(contract.max_iters, _HARD_MAX_ITERS_RUN)
    if contract.max_iters > max_iters:
        log(
            f"run-level max_iters={contract.max_iters} re-clamped to "
            f"{max_iters} (each iteration re-drives the whole DAG)"
        )

    detail: str | None = None
    prev_failing: int | None = None
    prev_failing_ids: frozenset[str] | None = None
    no_progress = 0
    last_result: AgentRunResult | None = None
    last_evidence: list[dict[str, Any]] = []
    last_syndrome: dict[str, Any] | None = None
    warned_redrive = False
    monotone = monotone_progress_enabled()
    progress_tracker = EverPassedTracker() if monotone else None

    for attempt in range(1, max_iters + 1):
        req = request if detail is None else _augment_request(
            request, attempt=attempt, detail=detail,
        )
        if attempt >= 2 and not warned_redrive:
            warned_redrive = True
            log(_REDRIVE_LIVE_WARNING.format(attempt=attempt))
        result = await drive_once(req)
        last_result = result

        if result.failed and result.error_code == "CCX_RUN_FAILED":
            log(f"attempt {attempt}: run failed to execute — aborting audit")
            return _stamp(result, {
                "passed": False,
                "status": "failed",
                "verify": "check",
                "iters": attempt,
                "stop_reason": "run_failed",
                "evidence": last_evidence,
                "syndrome": last_syndrome,
            })

        # Incremental verify (principle 7): re-run only the checks whose scope
        # intersects this round's working-tree diff; unscoped checks always run.
        # An empty/unknown diff falls back to the full suite.
        changed_paths = changed_paths_from_git(cwd)
        to_run = select_criteria_for_changes(
            criteria, changed_paths if changed_paths else None,
        )
        outcomes = [
            run_criterion_check(c, cwd=cwd, timeout_s=check_timeout_s)
            for c in to_run
        ]
        failing = [o for o in outcomes if not o.passed]
        # A green subset is not a green run: an under-declared scope would
        # otherwise pass the gate. Confirm on the full suite before passing.
        if not failing and len(to_run) < len(criteria):
            log(
                f"attempt {attempt}: incremental subset green "
                f"({len(to_run)}/{len(criteria)}) — confirming on full suite"
            )
            outcomes = [
                run_criterion_check(c, cwd=cwd, timeout_s=check_timeout_s)
                for c in criteria
            ]
            failing = [o for o in outcomes if not o.passed]
        last_evidence = [_outcome_dict(o) for o in outcomes]

        if not failing:
            log(f"attempt {attempt}: all {len(criteria)} check(s) passed")
            return _stamp(result, {
                "passed": True,
                "status": "passed",
                "verify": "check",
                "iters": attempt,
                "stop_reason": "satisfied",
                "evidence": last_evidence,
                "syndrome": last_syndrome,
            })

        log(
            f"attempt {attempt}: {len(failing)}/{len(outcomes)} check(s) failing"
        )
        syndrome = build_syndrome(cwd, failing, changed_paths=changed_paths or None)
        last_syndrome = syndrome.to_dict()
        detail = format_syndrome_detail(syndrome, attempt=attempt)

        unrunnable_now = [o for o in failing if check_unrunnable(o)]
        if len(unrunnable_now) == len(failing):
            log(
                f"attempt {attempt}: ALL {len(failing)} failing check(s) are "
                f"UNRUNNABLE (harness defect — a re-drive cannot repair an "
                f"immutable check). Affected: "
                f"{[o.criterion_id for o in unrunnable_now]}"
            )
            if stop_on_unrunnable_enabled():
                return _stamp(result, {
                    "passed": False,
                    "status": "failed",
                    "verify": "check",
                    "iters": attempt,
                    "stop_reason": "harness_defect",
                    "evidence": last_evidence,
                    "syndrome": last_syndrome,
                })

        made_progress, failing_ids = observe_progress(
            monotone=monotone,
            ever_tracker=progress_tracker,
            prev_failing_ids=prev_failing_ids,
            passed_ids=(o.criterion_id for o in outcomes if o.passed),
            failing_ids=(o.criterion_id for o in failing),
            prev_failing_count=prev_failing,
        )
        if prev_failing is not None:
            no_progress = 0 if made_progress else no_progress + 1
        prev_failing = len(failing)
        prev_failing_ids = failing_ids

        if no_progress >= contract.no_progress_stop:
            log(
                f"stopping: {no_progress} round(s) without progress "
                f"(no_progress_stop={contract.no_progress_stop})"
            )
            return _stamp(result, {
                "passed": False,
                "status": "failed",
                "verify": "check",
                "iters": attempt,
                "stop_reason": "no_progress",
                "evidence": last_evidence,
                "syndrome": last_syndrome,
            })

    log(f"stopping: reached run-level max_iters={max_iters}")
    assert last_result is not None  # loop ran at least once (max_iters >= 1)
    return _stamp(last_result, {
        "passed": False,
        "status": "failed",
        "verify": "check",
        "iters": max_iters,
        "stop_reason": "max_iters",
        "evidence": last_evidence,
        "syndrome": last_syndrome,
    })


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _annotate_redrive(verdict: dict[str, Any]) -> dict[str, Any]:
    """Surface the documented re-drive double-apply risk in the verdict itself.

    Additive and conditional: only when the run actually re-drove (``iters > 1``)
    do we add ``re_drives`` (the count) and ``non_idempotency_warning`` (the
    interpretation). A single-iteration verdict is returned unchanged — the
    common-case / max_iters=1 mitigation path stays byte-identical.
    """
    iters = verdict.get("iters")
    if isinstance(iters, int) and iters > 1 and "re_drives" not in verdict:
        verdict = dict(verdict)
        verdict["re_drives"] = iters
        verdict["non_idempotency_warning"] = _redrive_warning_text(iters)
    return verdict


def _stamp(
    result: AgentRunResult, verdict: dict[str, Any],
) -> AgentRunResult:
    """Set ``session_snapshot["run_audit_verdict"]`` on a copy of the result.

    Built once, from the outer loop's verdict — kept DISTINCT from the per-node
    ``contract_verdict`` (which ``_build_result`` scavenges from node extras).
    """
    verdict = _annotate_redrive(verdict)
    # Additive derived key: which failing checks were harness defects (could not
    # execute). Derived from the verdict's own ``evidence`` so it never re-runs a
    # check; absent only if a future verdict omits ``evidence``.
    if "unrunnable_criterion_ids" not in verdict:
        verdict = dict(verdict)
        verdict["unrunnable_criterion_ids"] = _unrunnable_criterion_ids(
            verdict.get("evidence") or []
        )
    snapshot = dict(result.session_snapshot or {})
    snapshot["run_audit_verdict"] = verdict
    return replace(result, session_snapshot=snapshot)


def _outcome_dict(outcome: CheckOutcome) -> dict[str, Any]:
    # Same evidence shape as governed_spawn._outcome_dict / governed_goal, so a
    # run_audit_verdict and a contract_verdict are interchangeable to a CLI /
    # output_json reader. The ``executable`` flag (additive) uses the shared
    # ``check_unrunnable`` predicate so run-level audit can distinguish a real
    # failure from a harness defect, exactly as goal mode already does.
    return {
        "criterion_id": outcome.criterion_id,
        "command": outcome.command,
        "passed": outcome.passed,
        "returncode": outcome.returncode,
        "timed_out": outcome.timed_out,
        "output_tail": outcome.output_tail,
        "executable": not check_unrunnable(outcome),
        "line": outcome.evidence_line(),
    }


def _unrunnable_criterion_ids(evidence: list[dict[str, Any]]) -> list[str]:
    """Criterion ids in a verdict's evidence whose check could NOT execute.

    Derived from the already-computed ``executable`` flag (``executable is
    False`` ⟺ ``check_unrunnable`` was true). An unrunnable check is always
    failing, so this is "the failing checks that are harness defects". Additive;
    empty when no evidence carries an unrunnable check.
    """
    return [
        str(ev.get("criterion_id"))
        for ev in evidence
        if ev.get("executable") is False
    ]


def _augment_request(
    request: AgentRunRequest, *, attempt: int, detail: str,
) -> AgentRunRequest:
    """Rebuild the instruction from the ORIGINAL instruction + latest syndrome.

    Built from ``request.instruction`` each round (not cumulatively appended) so
    a multi-round repair doesn't pile stale evidence blocks on each other.
    Mirrors ``governed_spawn._augment_goal`` with a distinct ``[RUN AUDIT
    RETRY]`` marker. Stamps patch-first into request metadata for child agents.
    """
    from .patch_first import PATCH_FIRST_METADATA_KEY

    augmented = (
        f"{request.instruction}\n\n"
        f"---\n"
        f"[RUN AUDIT RETRY — attempt {attempt}]\n"
        f"{detail}"
    )
    meta = dict(request.metadata or {})
    meta[PATCH_FIRST_METADATA_KEY] = True
    return replace(request, instruction=augmented, metadata=meta)


__all__ = [
    "CCX_RUN_CONTRACT_METADATA_KEY",
    "parse_run_audit_contract",
    "run_run_audit_loop",
]
