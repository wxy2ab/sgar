"""Filesystem storage for sgarx workspaces.

Mirrors :mod:`core.ccx.sgar.store` but redirects all paths from
``.sgar/`` to ``.sgarx/``. Achieved by subclassing ``SgarStore`` and
overriding ``__init__`` only — every other path-derivation property
(state, blueprint, roadmap, stages, missions, trace) reuses the parent
implementation because each one is computed from ``self.root``.

Candidate-frontier path helpers delegate to
:class:`core.ccx.sgarx.candidates.store.CandidateStore`.
"""

from __future__ import annotations

from pathlib import Path

from ..sgar.store import SgarStore, _normalize_session_id

SGARX_DIR = ".sgarx"


class SgarxStore(SgarStore):
    def __init__(self, cwd: str | Path = ".", session_id: str | None = None) -> None:
        self.cwd = Path(cwd).resolve()
        self.session_id = _normalize_session_id(session_id)
        if self.session_id:
            self.root = self.cwd / SGARX_DIR / "sessions" / self.session_id
        else:
            self.root = self.cwd / SGARX_DIR

    def frontier_path(self, stage_id: str) -> Path:
        from .candidates.store import CandidateStore

        return CandidateStore(self).frontier_path(stage_id)

    def candidate_dir(self, stage_id: str, candidate_id: str) -> Path:
        from .candidates.store import CandidateStore

        return CandidateStore(self).candidate_dir(stage_id, candidate_id)
