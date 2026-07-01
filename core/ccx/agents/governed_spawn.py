"""Optional, machine-verified, bounded-iteration contract for ccx spawns.

A *spawn contract* lets a spawner (a human, a planner, or any non-reasoning
client) attach machine-checkable acceptance criteria to a single spawned
subagent. When the contract is present AND the runner has the feature enabled,
the subagent's turn runs inside a **bounded verification-driven repair loop**:

1. run the subagent's turn (one full cc query loop);
2. *independently* verify each acceptance criterion that carries a
   ``[check: <cmd>]`` by running the command (exit 0 == pass) — never by
   trusting the subagent's own claim;
3. if every hard check passes → return with
   ``extras["contract_verdict"] = {"passed": True, ...}``;
4. if some fail and there is iteration budget left AND the failing-check count
   went down (progress) → feed the *concrete failing-check evidence* back into
   the goal and run again;
5. on ``max_iters`` / ``no_progress_stop`` / per-check timeout → return
   ``{"passed": False, ...}``. **The loop never declares success on its own.**

This is the SGAR ``autobuild`` repair loop (``core.ccx.sgar.autobuild``)
generalised from "one stage of a governed project" to "any spawned subagent",
reusing the same machinery rather than re-implementing it:

* ``core.ccx.sgar.checks.run_criterion_check`` — the deterministic ``[check:]``
  executor (no shell, bounded output tail, exit-code-is-truth).
* ``core.ccx.sgar.models.ExitCriterion`` — the criterion shape checks consume.

Design constraints this module deliberately enforces (the difference between a
useful contract and just another way to hang a run):

* **Independent gate only.** A contract can only *pass* on a green ``[check:]``.
  A criterion with no check is informational (recorded in evidence, never a hard
  gate) — exactly SGAR's "trust-the-implementer unless a check opts in" rule.
  ``verify="none"`` runs the turn once with no gating at all.
* **Always bounded.** ``max_iters`` (hard count, clamped to ``_HARD_MAX_ITERS``),
  ``no_progress_stop`` (early exit when failures stop shrinking), and the per-turn
  cc wall clock + per-check timeout together make an infinite repair loop
  impossible — the whole point, since iterate-until-pass against a reasoning
  model is otherwise a cost bomb.
* **Author-written, child-consumed.** The contract is data the spawner writes
  into ``metadata["ccx_contract"]``; the subagent only consumes the failing-check
  evidence fed into its goal. We never ask the reasoning child to *emit* the
  contract JSON (a short trailing JSON object reliably triggers the
  "Let me emit the JSON." → EOS stall on reasoning models).

Deferred (explicitly NOT in v1 — keep the validated-first surface small):

* adversarial / test-author verifiers (spawn N skeptics to refute a claim);
* soft / LLM-judged acceptance (every gate here is a shell exit code);
* a new DAG engine — cross-subagent ordering already exists via
  ``metadata.ccx_depends_on`` / ``ccx_depends_on_previous`` / ``sequential``;
* evaluating a contract whose agent itself spawns children — checks run on the
  agent's *own* turn, before any spawned child executes, so a contract attached
  to a spawning agent is reported ``status="skipped"`` (``spawned_children``)
  rather than falsely passed/failed. Attach contracts to terminal (leaf) agents.

The contract may later be promoted from ``metadata["ccx_contract"]`` to a typed
``SubagentInvocation`` field; carrying it in metadata keeps v1 100% additive and
byte-equivalent when absent (no signature change).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from ..sgar.checks import (
    CheckOutcome,
    check_unrunnable,
    run_criterion_check,
    stop_on_unrunnable_enabled,
)
from ..sgar.models import ExitCriterion
from .progress import EverPassedTracker, monotone_progress_enabled
from .subagent import SubagentInvocation, SubagentResult

logger = logging.getLogger(__name__)


#: Metadata key the spawner writes the contract under. Kept out of
#: ``INHERITABLE_METADATA_KEYS`` on purpose: a contract is specific to one
#: agent's task and must NOT auto-propagate to the children that agent spawns.
CONTRACT_METADATA_KEY = "ccx_contract"

#: Hard ceiling on ``loop.max_iters`` regardless of what the contract asks for.
#: A spawner that writes ``max_iters: 1000`` gets clamped (with a log) — the
#: bound is a safety property of the runtime, not a number we trust callers to
#: pick responsibly.
_HARD_MAX_ITERS = 10

_VALID_VERIFY = ("check", "none")


class ContractError(ValueError):
    """A present-but-malformed ``ccx_contract``.

    Raised by :func:`parse_contract` only when a contract IS present and is
    structurally invalid (a spawner-author bug, analogous to SGAR's "malformed
    plan can't even bootstrap"). An *absent* contract is not an error — it
    returns ``None`` so the no-contract path stays byte-equivalent.
    """


@dataclass(slots=True)
class SpawnContract:
    """Parsed, validated spawn contract."""

    acceptance: list[ExitCriterion]
    verify: str  # "check" | "none"
    max_iters: int
    no_progress_stop: int

    def checkable_criteria(self) -> list[ExitCriterion]:
        """Acceptance criteria that carry a hard ``[check:]`` gate."""
        return [c for c in self.acceptance if c.check]


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

def parse_contract(metadata: Any) -> SpawnContract | None:
    """Extract and validate a contract from invocation metadata.

    Returns ``None`` when no contract is present (the default-off,
    byte-equivalent path). Raises :class:`ContractError` when a contract IS
    present but malformed — a malformed contract must fail loudly, never
    silently degrade to "run once, no verification".
    """
    if not isinstance(metadata, dict):
        return None
    raw = metadata.get(CONTRACT_METADATA_KEY)
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ContractError(
            f"{CONTRACT_METADATA_KEY} must be a JSON object, got "
            f"{type(raw).__name__}"
        )

    verify = str(raw.get("verify") or "check").strip().lower()
    if verify not in _VALID_VERIFY:
        raise ContractError(
            f"contract.verify={verify!r} not supported; "
            f"choose one of {list(_VALID_VERIFY)}"
        )

    acceptance = _parse_acceptance(raw.get("acceptance"))

    if verify == "check" and not any(c.check for c in acceptance):
        raise ContractError(
            "contract.verify='check' requires at least one acceptance "
            "criterion with a [check:] command; nothing to verify "
            "independently (use verify='none' for an ungated single run)"
        )

    max_iters, no_progress_stop = _parse_loop(raw.get("loop"))
    return SpawnContract(
        acceptance=acceptance,
        verify=verify,
        max_iters=max_iters,
        no_progress_stop=no_progress_stop,
    )


def _parse_acceptance(raw: Any) -> list[ExitCriterion]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ContractError("contract.acceptance must be a list")
    criteria: list[ExitCriterion] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ContractError(
                f"contract.acceptance[{i}] must be an object with at least "
                f"'text'"
            )
        text = str(item.get("text") or "").strip()
        if not text:
            raise ContractError(
                f"contract.acceptance[{i}] requires non-empty 'text'"
            )
        criterion_id = str(item.get("id") or f"C{i + 1}").strip() or f"C{i + 1}"
        check = item.get("check")
        check_str = str(check).strip() if check else None
        criteria.append(ExitCriterion(
            criterion_id=criterion_id,
            description=text,
            blocking=True,
            check=check_str or None,
        ))
    return criteria


def _parse_loop(raw: Any) -> tuple[int, int]:
    loop = raw if isinstance(raw, dict) else {}
    max_iters = _coerce_positive_int(loop.get("max_iters"), default=3, field="loop.max_iters")
    no_progress_stop = _coerce_positive_int(
        loop.get("no_progress_stop"), default=2, field="loop.no_progress_stop",
    )
    if max_iters > _HARD_MAX_ITERS:
        logger.warning(
            "ccx spawn-contract: loop.max_iters=%d exceeds the hard ceiling "
            "%d; clamping. Bounded iteration is a runtime safety property.",
            max_iters, _HARD_MAX_ITERS,
        )
        max_iters = _HARD_MAX_ITERS
    return max_iters, no_progress_stop


def _coerce_positive_int(value: Any, *, default: int, field: str) -> int:
    if value is None:
        return default
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"contract.{field} must be an integer") from exc
    if out < 1:
        raise ContractError(f"contract.{field} must be >= 1, got {out}")
    return out


# --------------------------------------------------------------------------- #
# Control loop
# --------------------------------------------------------------------------- #

# run_once(invocation) -> SubagentResult. The runner passes its single-turn
# executor here; the loop re-invokes it with an evidence-augmented goal.
RunOnce = Callable[[SubagentInvocation], SubagentResult]


def _noop_log(_message: str) -> None:
    return None


def run_governed_spawn(
    run_once: RunOnce,
    invocation: SubagentInvocation,
    contract: SpawnContract,
    *,
    cwd: str | Path,
    check_timeout_s: float,
    log: Callable[[str], None] = _noop_log,
) -> SubagentResult:
    """Drive ``invocation`` through the bounded verification-repair loop.

    ``run_once`` executes one subagent turn (no contract awareness). The loop
    owns: re-running with evidence, the independent ``[check:]`` gate, the
    progress / iteration bounds, and stamping ``extras["contract_verdict"]``.
    Never raises for an ordinary failed contract — that's a ``passed=False``
    verdict, the honest outcome.
    """
    criteria = contract.checkable_criteria()

    # verify='none', or a 'check' contract with only informational criteria
    # (the latter is rejected at parse time, so this is the verify='none' path):
    # run exactly once, no gating.
    if contract.verify == "none" or not criteria:
        result = run_once(invocation)
        return _attach_verdict(result, {
            "passed": True,
            "status": "ungated",
            "verify": contract.verify,
            "iters": 1,
            "stop_reason": "no_checks",
            "evidence": [],
        })

    detail: str | None = None
    prev_failing: int | None = None
    no_progress = 0
    last_result: SubagentResult | None = None
    last_evidence: list[dict[str, Any]] = []
    # Progress signal (default OFF ⇒ count-delta, byte-identical). Under
    # CCX_MONOTONE_PROGRESS the ever-passed set replaces the count delta so an
    # oscillating repair (a check that re-fails after passing) can no longer
    # keep the loop alive past no_progress_stop. See progress.py.
    monotone = monotone_progress_enabled()
    progress_tracker = EverPassedTracker() if monotone else None

    for attempt in range(1, contract.max_iters + 1):
        inv = invocation if detail is None else _augment_goal(
            invocation, attempt=attempt, detail=detail,
        )
        result = run_once(inv)
        last_result = result

        # A contract attached to an agent that itself spawns children can't be
        # verified here — the checks would run before the children do. Report
        # honestly as skipped rather than falsely passing/failing, and let the
        # children spawn (return the result untouched apart from the verdict).
        if result.subtasks:
            log(f"attempt {attempt}: agent spawned children — contract skipped")
            return _attach_verdict(result, {
                "passed": False,
                "status": "skipped",
                "verify": "check",
                "iters": attempt,
                "stop_reason": "spawned_children",
                "evidence": [],
            })

        outcomes = [
            run_criterion_check(c, cwd=cwd, timeout_s=check_timeout_s)
            for c in criteria
        ]
        last_evidence = [_outcome_dict(o) for o in outcomes]
        failing = [o for o in outcomes if not o.passed]

        if not failing:
            log(f"attempt {attempt}: all {len(criteria)} check(s) passed")
            return _attach_verdict(result, {
                "passed": True,
                "status": "passed",
                "verify": "check",
                "iters": attempt,
                "stop_reason": "satisfied",
                "evidence": last_evidence,
            })

        log(
            f"attempt {attempt}: {len(failing)}/{len(criteria)} check(s) "
            f"failing"
        )
        detail = _format_failure_detail(failing)

        # Harness-defect early stop (opt-in, default OFF). When EVERY failing
        # check this round is UNRUNNABLE (malformed command / missing binary /
        # shell syntax error), a re-drive can never repair it. Always surface
        # that; under ``CCX_STOP_ON_UNRUNNABLE`` also stop NOW with
        # ``stop_reason="harness_defect"`` instead of re-running the agent's turn
        # against an immutable check that cannot execute. Default OFF ⇒ control
        # flow is byte-identical (only an extra log line, and only in the
        # already-abnormal all-unrunnable case).
        unrunnable_now = [o for o in failing if check_unrunnable(o)]
        if len(unrunnable_now) == len(failing):
            log(
                f"attempt {attempt}: ALL {len(failing)} failing check(s) are "
                f"UNRUNNABLE (harness defect — a re-drive cannot repair an "
                f"immutable check). Affected: "
                f"{[o.criterion_id for o in unrunnable_now]}"
            )
            if stop_on_unrunnable_enabled():
                return _attach_verdict(result, {
                    "passed": False,
                    "status": "failed",
                    "verify": "check",
                    "iters": attempt,
                    "stop_reason": "harness_defect",
                    "evidence": last_evidence,
                })

        # Progress = the failing-check count went DOWN versus the previous
        # round. The first failing round just records the baseline. Under
        # CCX_MONOTONE_PROGRESS, "progress" instead means a check passed that
        # had never passed before (a strictly-monotone measure oscillation
        # cannot reset); the OFF branch below is the unchanged count-delta.
        if monotone:
            newly = progress_tracker.observe(
                o.criterion_id for o in outcomes if o.passed
            )
            if prev_failing is not None:
                no_progress = 0 if newly else no_progress + 1
        elif prev_failing is not None:
            if len(failing) >= prev_failing:
                no_progress += 1
            else:
                no_progress = 0
        prev_failing = len(failing)

        if no_progress >= contract.no_progress_stop:
            log(
                f"stopping: {no_progress} round(s) without progress "
                f"(no_progress_stop={contract.no_progress_stop})"
            )
            return _attach_verdict(result, {
                "passed": False,
                "status": "failed",
                "verify": "check",
                "iters": attempt,
                "stop_reason": "no_progress",
                "evidence": last_evidence,
            })

    # max_iters exhausted with failures still present.
    log(f"stopping: reached max_iters={contract.max_iters}")
    assert last_result is not None  # loop ran at least once (max_iters >= 1)
    return _attach_verdict(last_result, {
        "passed": False,
        "status": "failed",
        "verify": "check",
        "iters": contract.max_iters,
        "stop_reason": "max_iters",
        "evidence": last_evidence,
    })


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _attach_verdict(
    result: SubagentResult, verdict: dict[str, Any],
) -> SubagentResult:
    # Additive derived key: which failing checks were harness defects (could not
    # execute). Derived from the verdict's own ``evidence`` so it never re-runs a
    # check; kept on a copy so the input dict is not mutated.
    if "unrunnable_criterion_ids" not in verdict:
        verdict = dict(verdict)
        verdict["unrunnable_criterion_ids"] = _unrunnable_criterion_ids(
            verdict.get("evidence") or []
        )
    extras = dict(result.extras) if result.extras else {}
    extras["contract_verdict"] = verdict
    return replace(result, extras=extras)


def _outcome_dict(outcome: CheckOutcome) -> dict[str, Any]:
    # ``executable`` (additive) uses the shared ``check_unrunnable`` predicate so
    # a spawn contract_verdict can distinguish a real failure from a harness
    # defect, matching governed_run / governed_goal evidence shape.
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


def _format_failure_detail(failing: list[CheckOutcome]) -> str:
    lines = [
        "The following machine-verified acceptance checks are still FAILING. "
        "Fix the underlying problem so each one passes. The checks are re-run "
        "independently after your turn — do NOT claim success yourself.",
        "",
    ]
    for outcome in failing:
        lines.append(f"- [{outcome.criterion_id}] {outcome.evidence_line()}")
    return "\n".join(lines)


def _augment_goal(
    invocation: SubagentInvocation, *, attempt: int, detail: str,
) -> SubagentInvocation:
    """Rebuild the goal from the ORIGINAL goal + the latest failure detail.

    Built from ``invocation.goal`` each round (not cumulatively appended) so a
    multi-round repair doesn't pile stale evidence blocks on top of each other.
    """
    augmented = (
        f"{invocation.goal}\n\n"
        f"---\n"
        f"[CONTRACT RETRY — attempt {attempt}]\n"
        f"{detail}"
    )
    return replace(invocation, goal=augmented)


__all__ = [
    "CONTRACT_METADATA_KEY",
    "ContractError",
    "SpawnContract",
    "parse_contract",
    "run_governed_spawn",
]
