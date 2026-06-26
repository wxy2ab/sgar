"""The single source of the code-task ``[check:]`` criterion.

Every governed mode injects the SAME audit command — wrapped either as a
run-audit / spawn *contract dict* (``shape="contract"``) or as a list of
:class:`~core.ccx.sgar.models.ExitCriterion` (``shape="criteria"``) — so there
is one place that decides what "code task done" is verified by.

The criterion's ``[check:]`` is ``<python> -m core.ccx.audit.code_task`` (see
:mod:`core.ccx.audit.code_task`). The working directory and timeout are supplied
by :func:`core.ccx.sgar.checks.run_criterion_check` at execution time, so the
command itself carries no ``--cwd``.
"""

from __future__ import annotations

import os
import shlex
import sys
from typing import Any

from ..sgar.models import ExitCriterion

_DEFAULT_CRITERION_ID = "ccx_code_task"
_DESCRIPTION = "CODE task definition of done (wiring + scoped tests green)"

#: Repo root = the directory that contains the ``core/`` package. Resolved from
#: this file (``<root>/core/ccx/audit/contract.py``) so the audit command can
#: ``sys.path.insert`` it explicitly — ``python -m core.ccx...`` would NOT
#: resolve under ``run_criterion_check``'s hermetic env (``PYTHONSAFEPATH=1``
#: strips the cwd that ``-m`` would otherwise prepend, and ``PYTHONPATH`` is
#: dropped). A ``-c`` bootstrap that pins the known root is cwd- and
#: env-independent, so the audit imports cleanly from any workspace.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


def _audit_command(*, test_cmd: str | None) -> str:
    """One ``shlex.split``-safe command string for ``run_criterion_check``.

    Uses a ``-c`` bootstrap that inserts the repo root on ``sys.path`` before
    importing the audit, so it resolves regardless of cwd / hermetic env.
    """
    bootstrap = (
        "import sys; sys.path.insert(0, %r); "
        "from core.ccx.audit.code_task import main; sys.exit(main())" % _REPO_ROOT
    )
    parts = [sys.executable, "-c", bootstrap]
    if test_cmd:
        # The audit re-``shlex.split``s this single token internally.
        parts += ["--test-cmd", test_cmd]
    return " ".join(shlex.quote(p) for p in parts)


def build_code_task_contract(
    shape: str = "contract",
    *,
    cwd: str = ".",
    test_cmd: str | None = None,
    criterion_id: str = _DEFAULT_CRITERION_ID,
) -> Any:
    """Build the code-task audit criterion in the requested ``shape``.

    * ``shape="contract"`` → a run-audit / spawn contract ``dict`` with
      ``loop.max_iters=1`` (gate-once: verify once after the DAG, never auto
      re-drive — the run-level loop is non-idempotent).
    * ``shape="criteria"`` → ``[ExitCriterion(...)]`` for callers that already
      hold an exit-criteria list (goal verification, sgar close).

    ``cwd`` is currently reserved (the check executor supplies the working
    directory); kept in the signature so callers can pass it uniformly.
    """
    _ = cwd  # reserved; cwd is supplied by run_criterion_check at exec time
    command = _audit_command(test_cmd=test_cmd)
    if shape == "contract":
        return {
            "verify": "check",
            "acceptance": [
                {"id": criterion_id, "text": _DESCRIPTION, "check": command},
            ],
            "loop": {"max_iters": 1, "no_progress_stop": 1},
        }
    if shape == "criteria":
        return [
            ExitCriterion(
                criterion_id=criterion_id,
                description=_DESCRIPTION,
                blocking=True,
                check=command,
            )
        ]
    raise ValueError(f"unknown contract shape {shape!r}; expected 'contract' or 'criteria'")


__all__ = ["build_code_task_contract"]
