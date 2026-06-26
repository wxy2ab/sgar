"""Run-scoped repository outline cache.

cc rebuilds the repository outline once per turn (see
``core/cc/conversation/turn_pipeline.py:116-142``). For ccx with N
parallel doc investigators that's N redundant filesystem scans per run.
This cache amortizes the scan to once per (cwd, depth-class) per run.

Two depth classes are supported because cc treats ask and doc
differently:

* ``shallow`` — ``max_depth=3``, ``max_entries_per_dir=6`` (ask default)
* ``deep``    — ``max_depth=4``, ``max_entries_per_dir=8`` (doc default)

The mapping mirrors ``core/cc/conversation/turn_pipeline.py`` so ccx's
prompts see the same outline shape cc would have produced.

For deep paths the user explicitly named that fall outside the
truncation budget of the top-level outline, ``get_focused_text`` returns
a subtree-anchored outline so the prompt can include "look here"
context without expanding the whole tree. This is the workaround for
truncation-induced "directory not in outline" misreads (e.g.
``core/`` has 51 subdirectories, only ~8 fit the doc-style outline).

Thread-safety: lazy build is guarded by a lock so multiple v5 worker
threads asking simultaneously only scan once.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class RepositoryOutlineCache:
    cwd: str
    _shallow: dict[str, Any] | None = None
    _deep: dict[str, Any] | None = None
    _focused: dict[tuple[str, int, int], dict[str, Any]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def get(self, *, deep: bool = False) -> dict[str, Any]:
        """Return the cached outline, building on first access.

        ``deep=True`` returns the doc-style outline (depth=4, entries=8);
        ``deep=False`` returns the ask-style outline (depth=3, entries=6).
        Each is cached independently in the same instance.
        """
        # Fast path — read without lock when populated.
        if deep and self._deep is not None:
            return self._deep
        if not deep and self._shallow is not None:
            return self._shallow
        with self._lock:
            if deep:
                if self._deep is None:
                    self._deep = self._build(deep=True)
                return self._deep
            if self._shallow is None:
                self._shallow = self._build(deep=False)
            return self._shallow

    def get_text(self, *, deep: bool = False) -> str:
        outline = self.get(deep=deep)
        return str(outline.get("text", ""))

    def get_focused(
        self,
        relative_path: str,
        *,
        max_depth: int = 3,
        max_entries_per_dir: int = 12,
    ) -> dict[str, Any]:
        """Build (and cache) an outline anchored at a specific subtree.

        ``relative_path`` is interpreted relative to ``self.cwd`` if it
        isn't absolute. If the path doesn't resolve to an existing
        directory under cwd, returns ``{"text": "", ...}`` so callers
        can detect missing paths without raising.

        Caching key is ``(resolved_path_str, max_depth, max_entries_per_dir)``;
        re-asking for the same key is free.
        """
        target = Path(relative_path)
        if not target.is_absolute():
            target = Path(self.cwd) / target
        try:
            resolved = target.resolve()
        except OSError:
            return self._empty(target, max_depth, max_entries_per_dir)
        if not resolved.exists() or not resolved.is_dir():
            return self._empty(resolved, max_depth, max_entries_per_dir)

        key = (str(resolved), max_depth, max_entries_per_dir)
        cached = self._focused.get(key)
        if cached is not None:
            return cached
        with self._lock:
            cached = self._focused.get(key)
            if cached is not None:
                return cached
            from core.cc.conversation.mode_strategy import build_repository_outline
            built = build_repository_outline(
                resolved,
                max_depth=max_depth,
                max_entries_per_dir=max_entries_per_dir,
            )
            self._focused[key] = built
            return built

    def get_focused_text(
        self,
        relative_path: str,
        *,
        max_depth: int = 3,
        max_entries_per_dir: int = 12,
    ) -> str:
        return str(
            self.get_focused(
                relative_path,
                max_depth=max_depth,
                max_entries_per_dir=max_entries_per_dir,
            ).get("text", "")
        )

    @staticmethod
    def _empty(
        root: Path, max_depth: int, max_entries_per_dir: int,
    ) -> dict[str, Any]:
        return {
            "text": "",
            "root": str(root),
            "max_depth": max_depth,
            "max_entries_per_dir": max_entries_per_dir,
        }

    def _build(self, *, deep: bool) -> dict[str, Any]:
        # Imported lazily so test environments that don't have cc loaded
        # for some reason won't fail at module import time.
        from core.cc.conversation.mode_strategy import build_repository_outline
        if deep:
            return build_repository_outline(
                self.cwd, max_depth=4, max_entries_per_dir=8,
            )
        return build_repository_outline(
            self.cwd, max_depth=3, max_entries_per_dir=6,
        )


__all__ = ["RepositoryOutlineCache"]
