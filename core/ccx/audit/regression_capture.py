"""Regression-capture ratchet: promote ONE confirmed finding to a permanent guard.

The highest-ROI half of the adversarial-debug productization. A monitored probe
finds a bug, you fix it — but the probe is a throwaway script that never becomes
a resident test, so the next refactor can silently re-break the same code. This
module closes that gap with a single, anti-gaming promotion pipeline.

:func:`promote_finding` is ONE function (not two paths that can drift) that:

1. reads recidivism for the ``(file, func)`` location (the cross-run axis
   ``runtime.db`` lacks — see :mod:`core.ccx.audit.finding_ledger`);
2. scaffolds the permanent guard test (operator-supplied assertions) and wires
   its repro as a hermetic-safe ``[check:]``;
3. verifies the guard is GREEN on the fixed tree (``run_criterion_check`` — exit
   code is the authoritative floor; ``check_unrunnable`` ⇒ abort, not a verdict);
4. **MANDATORY** proves the new guard has TEETH by mutating the fix back out in
   an isolated worktree (the shared mutation engine) — a GREEN-under-mutation
   blind spot REFUSES promotion;
5. (optional) runs the code-task "only-tests-changed ⇒ NEEDS_REVIEW" backstop;
6. records the outcome to the finding ledger keyed by ``(file, func)``.

SEPARATION OF POWERS (anti-cheating, structural — not documentation):
the guard's ``test_source`` (assertions) and the ``mutations`` fed to the teeth
proof are **required** operator/independent-judge inputs. This function takes no
``llm=`` and has no generation fallback: omit either and you get a ``TypeError``
(missing required kwarg) / ``ValueError`` (empty). The LLM that wrote the fix
cannot supply its own assertions (it would write always-pass ones) or its own
mutations (it would pick easy-to-catch ones and skip the one that exposes the
blind spot). The teeth proof is mandatory and non-optional.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..sgar.checks import CheckOutcome, check_unrunnable, run_criterion_check
from ..sgar.models import ExitCriterion
from .check_template import validate_check_command
from .finding_ledger import (
    append_finding,
    code_location_key,
    count_prior,
    defect_lifecycle,
)
from .mutation import Mutation, MutationResult, ephemeral_worktree, run_mutation_campaign

__all__ = [
    "PromotionResult",
    "promote_finding",
    "scaffold_test",
    "PROMOTED",
    "REJECTED_BLIND_SPOT",
    "ABORTED_HARNESS_DEFECT",
    "ABORTED_POSTFIX_RED",
    "NEEDS_REVIEW",
]

PROMOTED = "PROMOTED"
REJECTED_BLIND_SPOT = "REJECTED_BLIND_SPOT"
ABORTED_HARNESS_DEFECT = "ABORTED_HARNESS_DEFECT"
ABORTED_POSTFIX_RED = "ABORTED_POSTFIX_RED"
NEEDS_REVIEW = "NEEDS_REVIEW"


@dataclass
class PromotionResult:
    promoted: bool
    status: str
    code_location: dict[str, str]
    recidivism: int = 0
    post_fix: CheckOutcome | None = None
    teeth: list[MutationResult] = field(default_factory=list)
    ledger_record: dict[str, Any] | None = None
    detail: str = ""
    #: Lifecycle state of this ``(file, func)`` *before* this promotion's record
    #: (theory §7.3). ``"regression_passed"`` means a resident guard already
    #: existed for it.
    prior_state: str = "unknown"
    #: True when ``prior_state == "regression_passed"`` — a new confirmed
    #: promotion at a location whose guard had already passed = the guard did
    #: NOT hold (a silent re-break the ratchet exists to surface). Strictly
    #: sharper than the scalar ``recidivism`` count.
    is_reopen: bool = False


def scaffold_test(repo_root: str | Path, test_path: str, test_source: str) -> Path:
    """Write the permanent guard test (operator assertions) into the working tree.

    Returns the absolute path written. Creates parent dirs. Overwrites an
    existing file at ``test_path`` (the caller owns naming / collision policy).
    """
    if not test_source.strip():
        raise ValueError("test_source is empty — operator must supply the guard's assertions")
    dst = Path(repo_root) / test_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(test_source, encoding="utf-8")
    return dst


def _only_tests_changed(cwd: str | Path, base_ref: str) -> bool:
    """Reuse the code-task classifier: test ``.py`` changed but no production ``.py``."""
    from . import gitdiff

    if not gitdiff.is_inside_work_tree(str(cwd)):
        return False
    changes = gitdiff.collect_changes(str(cwd), base_ref)
    return bool(changes.test_py) and not bool(changes.prod_py)


def promote_finding(
    *,
    finding: dict[str, Any],
    code_location: tuple[str, str] | dict[str, str],
    test_path: str,
    test_source: str,
    mutations: Sequence[Mutation],
    repro_check: str,
    repo_root: str | Path,
    ledger_path: str | Path,
    cwd: str | Path | None = None,
    timeout_s: float = 600.0,
    overlay_files: Sequence[str] = (),
    base_ref: str | None = None,
    track: str | None = None,
    pybin: str | None = None,
    mismatch_type: str | None = None,
    severity: str | None = None,
) -> PromotionResult:
    """Promote ``finding`` to a permanent, teeth-proven regression guard.

    ``test_source`` and ``mutations`` are MANDATORY operator inputs (no LLM
    fallback — see module docstring). ``repro_check`` must be a hermetic-safe
    pytest command (validated). Returns a :class:`PromotionResult`; never raises
    for an ordinary refusal — that is ``promoted=False`` with a status, the
    honest result.

    Pipeline (single function, shared key, one ledger):

    * post-fix GREEN gate — ``run_criterion_check`` (exit code = floor);
      unrunnable ⇒ ``ABORTED_HARNESS_DEFECT``; red ⇒ ``ABORTED_POSTFIX_RED``;
    * MANDATORY teeth proof — mutate the fix back out, run the new guard; any
      blind spot (or un-appliable mutation) ⇒ ``REJECTED_BLIND_SPOT``;
    * optional ``base_ref`` backstop — only-tests-changed ⇒ ``NEEDS_REVIEW``;
    * else ``PROMOTED``.
    """
    import sys as _sys

    pybin = pybin or _sys.executable
    cwd = cwd or repo_root
    track = track or str(finding.get("track") or "regression-capture")
    key = code_location_key(code_location)

    # --- structural three-power enforcement (no llm fallback exists) ---
    if not isinstance(test_source, str) or not test_source.strip():
        raise ValueError(
            "test_source (the guard's assertions) is mandatory and "
            "operator-supplied; the fix-writing LLM may not generate it"
        )
    if not mutations:
        raise ValueError(
            "mutations (the teeth proof) are mandatory and operator-supplied; "
            "a promotion with no teeth proof is refused by construction"
        )
    validate_check_command(repro_check)

    repro_text = str(finding.get("repro") or "")
    recidivism = count_prior(ledger_path, key)
    # Lifecycle state BEFORE this promotion's record. A prior ``regression_passed``
    # means a resident guard already existed here — so a new confirmed promotion
    # is a re-break the guard did not hold (sharper than the scalar recidivism).
    prior_state = defect_lifecycle(ledger_path, key)
    is_reopen = prior_state == "regression_passed"

    def _record(verdict: str, observed: str, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        evidence: dict[str, Any] = {
            "code_location": key,
            "recidivism": recidivism,
            "prior_state": prior_state,
            "is_reopen": is_reopen,
            "test_path": test_path,
            "finding_id": finding.get("id"),
        }
        if extra:
            evidence.update(extra)
        return append_finding(
            track=track,
            hypothesis=(
                f"Promote finding to a resident guard at {key['file']}::{key['func']}. "
                "The guard must be GREEN on the fixed tree and go RED when the fix "
                "is mutated back out (proven teeth)."
            ),
            expected="post-fix GREEN + new guard RED under the fix-reverting mutation",
            observed=observed,
            verdict=verdict,
            repro=repro_text or repro_check,
            path=ledger_path,
            evidence=evidence,
            mismatch_type=mismatch_type,
            severity=severity,
        )

    # (1) scaffold the permanent guard into the working tree.
    scaffold_test(repo_root, test_path, test_source)

    # (2) POST-FIX GREEN gate — the guard must pass on the fixed tree.
    criterion = ExitCriterion(
        criterion_id="regression_repro",
        description=f"regression guard for {key['file']}::{key['func']}",
        blocking=True,
        check=repro_check,
    )
    post = run_criterion_check(criterion, cwd=cwd, timeout_s=timeout_s)
    if check_unrunnable(post):
        rec = _record(
            "uncertain",
            f"post-fix check UNRUNNABLE (harness defect): {post.evidence_line()}",
            extra={"phase": "post_fix", "unrunnable": True},
        )
        return PromotionResult(
            promoted=False, status=ABORTED_HARNESS_DEFECT, code_location=key,
            recidivism=recidivism, post_fix=post, ledger_record=rec,
            prior_state=prior_state, is_reopen=is_reopen,
            detail="repro check could not execute; not a verdict",
        )
    if not post.passed:
        rec = _record(
            "uncertain",
            f"post-fix check RED — the guard does not pass on the fixed tree "
            f"(fix missing or guard wrong): {post.evidence_line()}",
            extra={"phase": "post_fix", "passed": False},
        )
        return PromotionResult(
            promoted=False, status=ABORTED_POSTFIX_RED, code_location=key,
            recidivism=recidivism, post_fix=post, ledger_record=rec,
            prior_state=prior_state, is_reopen=is_reopen,
            detail="guard not green on the fixed tree",
        )

    # (3) MANDATORY teeth proof — mutate the fix back out; the guard must go RED.
    overlay = list(dict.fromkeys([test_path, *overlay_files]))
    with ephemeral_worktree(repo_root, suffix="regcap") as wt:
        for rel in overlay:
            src = Path(repo_root) / rel
            if src.exists():
                dst = wt / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
        teeth = run_mutation_campaign(
            wt, mutations, pybin=pybin, test_timeout_s=timeout_s,
        )

    not_applied = [r.name for r in teeth if not r.applied]
    blind = [r.name for r in teeth if r.is_blind_spot]
    if blind or not_applied:
        reasons = []
        if blind:
            reasons.append(f"blind-spot mutations (guard stayed GREEN): {blind}")
        if not_applied:
            reasons.append(f"un-appliable mutations (anchor miss): {not_applied}")
        rec = _record(
            "false_positive",
            "REFUSED: teeth proof failed — " + "; ".join(reasons),
            extra={
                "phase": "teeth", "blind_spots": blind,
                "anchor_misses": not_applied,
                "teeth": [r.name for r in teeth if r.red],
            },
        )
        return PromotionResult(
            promoted=False, status=REJECTED_BLIND_SPOT, code_location=key,
            recidivism=recidivism, post_fix=post, teeth=teeth, ledger_record=rec,
            prior_state=prior_state, is_reopen=is_reopen,
            detail="; ".join(reasons),
        )

    # (4) optional backstop — only-tests-changed ⇒ NEEDS_REVIEW (no fix landed).
    if base_ref is not None and _only_tests_changed(cwd, base_ref):
        rec = _record(
            "uncertain",
            "NEEDS_REVIEW: only test .py changed vs base and no production .py — "
            "a guard with no accompanying fix is possible test-gaming",
            extra={"phase": "backstop", "base_ref": base_ref},
        )
        return PromotionResult(
            promoted=False, status=NEEDS_REVIEW, code_location=key,
            recidivism=recidivism, post_fix=post, teeth=teeth, ledger_record=rec,
            prior_state=prior_state, is_reopen=is_reopen,
            detail="only-tests-changed backstop tripped",
        )

    # (5) PROMOTED — green on the fix, proven teeth.
    rec = _record(
        "confirmed",
        f"PROMOTED: guard GREEN on fixed tree; {len(teeth)} mutation(s) all caught "
        f"(RED) → proven teeth. recidivism={recidivism}"
        + (f" REOPEN (prior_state={prior_state})" if is_reopen else ""),
        extra={"phase": "promoted", "teeth": [r.name for r in teeth]},
    )
    return PromotionResult(
        promoted=True, status=PROMOTED, code_location=key,
        recidivism=recidivism, post_fix=post, teeth=teeth, ledger_record=rec,
        prior_state=prior_state, is_reopen=is_reopen,
        detail="green on fix + proven teeth",
    )
