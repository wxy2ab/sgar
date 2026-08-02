"""sgarx autobuild entry — frontier mode by default.

Stable callers should keep using ``core.ccx.sgar.autobuild`` (flag default
False). This thin wrapper flips ``use_candidate_frontier=True`` for sgarx
workspaces under ``.sgarx/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..sgar.autobuild import (
    AutobuildReport,
    ImplementResult,
    Implementer,
    ProjectPlan,
    StagePlan,
    StageReport,
    autobuild as _sgar_autobuild,
)


def autobuild(
    plan: ProjectPlan,
    *,
    cwd: str | Path,
    implement: Implementer,
    session: str | None = None,
    max_verify_attempts: int = 4,
    check_timeout_s: float = 120.0,
    log: Callable[[str], None] | None = None,
    use_candidate_frontier: bool = True,
) -> AutobuildReport:
    """Drive ``plan`` with candidate frontier enabled by default."""
    kwargs: dict = {
        "cwd": cwd,
        "implement": implement,
        "session": session,
        "max_verify_attempts": max_verify_attempts,
        "check_timeout_s": check_timeout_s,
        "use_candidate_frontier": use_candidate_frontier,
    }
    if log is not None:
        kwargs["log"] = log
    return _sgar_autobuild(plan, **kwargs)


__all__ = [
    "AutobuildReport",
    "ImplementResult",
    "Implementer",
    "ProjectPlan",
    "StagePlan",
    "StageReport",
    "autobuild",
]
