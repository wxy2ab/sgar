"""Filesystem storage for sgarx workspaces.

Mirrors :mod:`core.ccx.sgar.store` but redirects all paths from
``.sgar/`` to ``.sgarx/``. Achieved by subclassing ``SgarStore`` and
overriding ``__init__`` only — every other path-derivation property
(state, blueprint, roadmap, stages, missions, trace) reuses the parent
implementation because each one is computed from ``self.root``.
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
