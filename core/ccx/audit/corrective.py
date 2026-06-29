"""Deterministic hard-feedback teeth for the audit agent (execution-oracle case).

This is the Phase-0 primitive of the hard-feedback audit agent
(``core/ccx/docs/audit_agent_design_2026-06-28.md``). It is **pure and
LLM-free**: it turns an audit *claim* into a finding whose ``verdict='confirmed'``
is earned by a **deterministic re-runnable query against a constructed oracle**
(design §1.5 — generalized teeth), never by the agent's prose.

The oracle this module constructs is the **execution oracle** (the §1.5 special
case, fidelity source #3 "execution vs reference"): a proposed ``evidence_check``
is a hermetic-safe pytest command; ``is_actionable`` re-runs it on the *current,
unfixed* tree and confirms the claim only when it reproduces a genuine RED that is
**tied to the originating failure** (not merely *any* RED). The consistency-oracle
teeth for the narrative-fidelity MVP (design §1.5, fidelity source #1) live in a
sibling Phase-1 module; this one stays the pure execution-oracle primitive that
must land + prove teeth before any LLM surface exists.

Trust-root discipline (design §3, §4): the corrective answer rides the existing
free-form ``evidence`` dict — **no finding-schema change** — and the agent is
never trusted on its say-so. ``validate_check_command`` is the single accept
boundary for the ``evidence_check`` (hermetic-safe, pytest-only), and grounding
is checked against the **original red ``CheckOutcome``** supplied by the trusted
harness, never against anything the agent authored.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..sgar.checks import CheckOutcome, check_unrunnable, run_criterion_check
from ..sgar.models import ExitCriterion
from .check_template import validate_check_command

__all__ = [
    "build_corrective_requirement",
    "is_actionable",
    "default_audit_findings_path",
]

#: Default repo-relative ledger path the Phase-1 writer and Phase-2 reader agree
#: on (design §5 / §8 "Ledger path: [chosen]").
_LEDGER_RELPATH = (".ccx", "audit", "findings.jsonl")

#: Matches a pytest file target (``pkg/test_x.py``); a trailing ``::nodeid`` is
#: ignored for the file-level grounding tie.
_PYTEST_FILE_RE = re.compile(r"[\w./\\-]+\.py")


def default_audit_findings_path(cwd: str | Path) -> Path:
    """Repo-relative audit-agent findings ledger path: ``<cwd>/.ccx/audit/findings.jsonl``.

    Pinned here (design §5 critique-mandate) so the Phase-1 ``append_finding``
    writer and the Phase-2 ``ledger_stats`` reader resolve the *same* file.
    """
    return Path(cwd).joinpath(*_LEDGER_RELPATH)


def build_corrective_requirement(
    claim: str, corrective_answer: str, evidence_check: str
) -> dict[str, str]:
    """Assemble the corrective triple destined for ``evidence['corrective_requirement']``.

    Validates ``evidence_check`` at this single accept boundary
    (``validate_check_command`` — hermetic-safe, pytest-only). Raises
    ``ValueError`` if it is not a hermetic-safe pytest invocation (a cwd-relative
    ``python -c "import …"`` / ``python script.py`` would FAIL hermetically and be
    indistinguishable from a real RED — design §3).

    No teeth are claimed here: the returned dict is *advisory* until
    :func:`is_actionable` re-reproduces the failure. ``claim`` and
    ``corrective_answer`` are natural-language (the proposed *what-is-wrong* and
    *fix-requirement*); only ``evidence_check`` is machine-verified.
    """
    validate_check_command(evidence_check)
    return {
        "claim": str(claim),
        "evidence_check": str(evidence_check).strip(),
        "corrective_answer": str(corrective_answer),
    }


def _pytest_file_targets(text: str) -> set[str]:
    """Set of pytest ``.py`` file targets named in ``text``, keyed by basename.

    Basename is the pragmatic "same test file" identity: it unifies a bare
    ``test_x.py``, a path ``pkg/test_x.py``, and a nodeid ``pkg/test_x.py::t`` so
    a grounded reproduction matches regardless of how the path was spelled, while
    an *unrelated* file (``test_y.py``) never collides.
    """
    targets: set[str] = set()
    for match in _PYTEST_FILE_RE.findall(text or ""):
        targets.add(Path(match).name)
    return targets


def _grounded_in_original(evidence_check: str, original_outcome: CheckOutcome) -> bool:
    """Is the proposed ``evidence_check`` tied to the *originating* failure?

    Enforcement of design §3's grounding mandate (the scs_v6 lesson): a check that
    is RED for a reason *unrelated* to the original criterion must not yield
    ``confirmed``. We require the ``evidence_check`` to name at least one pytest
    file target that also appears in the original criterion's command or failing
    output. Conservative by construction: when no shared target can be
    established (e.g. a non-pytest original criterion), grounding fails and the
    claim is refused rather than confirmed.
    """
    repro_targets = _pytest_file_targets(evidence_check)
    if not repro_targets:
        return False
    origin_text = f"{original_outcome.command}\n{original_outcome.output_tail}"
    return bool(repro_targets & _pytest_file_targets(origin_text))


def is_actionable(
    corrective_requirement: dict[str, str],
    *,
    original_outcome: CheckOutcome,
    cwd: str | Path,
    timeout_s: float = 600.0,
) -> bool:
    """Deterministic teeth: may this corrective be marked ``verdict='confirmed'``?

    Returns ``True`` **only** when every clause holds (design §3); otherwise the
    caller must downgrade the claim to ``uncertain`` and demote the corrective to
    advisory. The agent's prose is never trusted — the check is re-run.

    ``original_outcome`` is the **trusted** originating red ``CheckOutcome``
    supplied by the harness (the goal's own acceptance criterion that actually
    failed), NOT anything the agent authored — grounding must be machine-grounded.

    Clauses:

    * (a) the originating check was a *real* defect: ``not original_outcome.passed``
      and ``check_unrunnable(original_outcome) is False`` (it ran and reported
      false — a real defect exists, not a harness defect);
    * (b) ``evidence_check`` is non-empty and a hermetic-safe pytest command
      (``validate_check_command``);
    * (c) grounding: the ``evidence_check`` targets the **same** defect as the
      originating criterion (``_grounded_in_original``) — not merely *any* RED;
    * (d) running ``evidence_check`` on the current unfixed tree yields a genuine
      non-zero RED (runnable — ``check_unrunnable`` False — and ``not passed``).
    """
    # (a) the originating failure must be a real, runnable defect to ground against.
    if original_outcome.passed or check_unrunnable(original_outcome):
        return False

    evidence_check = str(corrective_requirement.get("evidence_check") or "").strip()
    if not evidence_check:
        return False

    # (b) single accept boundary — hermetic-safe pytest-only.
    try:
        validate_check_command(evidence_check)
    except ValueError:
        return False

    # (c) the reproduction must target the originating failure, not any RED.
    if not _grounded_in_original(evidence_check, original_outcome):
        return False

    # (d) genuine RED on the current (unfixed) tree.
    criterion = ExitCriterion(
        criterion_id=f"audit_repro::{original_outcome.criterion_id}",
        description="audit-agent corrective evidence re-reproduction",
        blocking=True,
        check=evidence_check,
    )
    repro = run_criterion_check(criterion, cwd=cwd, timeout_s=timeout_s)
    if check_unrunnable(repro):
        return False
    return not repro.passed
