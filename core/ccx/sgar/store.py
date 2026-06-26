"""Filesystem storage for SGAR workspaces."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .models import ProjectMode, ProjectState, SgarError


SGAR_DIR = ".sgar"
STAGE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_stage_id(stage_id: str) -> str:
    stage_id = str(stage_id or "").strip()
    if not STAGE_ID_RE.match(stage_id):
        raise SgarError(
            "stage id must be 1-128 chars of letters, digits, dot, dash, "
            "or underscore, and start with a letter or digit"
        )
    return stage_id


def _ensure_inside(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise SgarError(f"{label} escapes SGAR workspace: {path}") from exc


class SgarStore:
    def __init__(self, cwd: str | Path = ".", session_id: str | None = None) -> None:
        self.cwd = Path(cwd).resolve()
        self.session_id = _normalize_session_id(session_id)
        if self.session_id:
            self.root = self.cwd / SGAR_DIR / "sessions" / self.session_id
        else:
            self.root = self.cwd / SGAR_DIR

    @property
    def config_path(self) -> Path:
        return self.root / "config.json"

    @property
    def state_path(self) -> Path:
        return self.root / "state.json"

    @property
    def blueprint_path(self) -> Path:
        return self.root / "blueprint.md"

    @property
    def roadmap_path(self) -> Path:
        return self.root / "roadmap.md"

    @property
    def stages_root(self) -> Path:
        return self.root / "stages"

    @property
    def missions_root(self) -> Path:
        return self.root / "missions"

    def stage_dir(self, stage_id: str) -> Path:
        stage_id = validate_stage_id(stage_id)
        root = self.stages_root.resolve()
        path = (root / stage_id).resolve()
        _ensure_inside(path, root, "stage directory")
        return path

    def stage_spec_path(self, stage_id: str) -> Path:
        return self.stage_dir(stage_id) / "spec.md"

    def stage_plan_path(self, stage_id: str) -> Path:
        return self.stage_dir(stage_id) / "plan.md"

    def stage_tasks_path(self, stage_id: str) -> Path:
        return self.stage_dir(stage_id) / "tasks.md"

    def stage_context_path(self, stage_id: str) -> Path:
        return self.stage_dir(stage_id) / "context.md"

    def verification_json_path(self, stage_id: str) -> Path:
        return self.stage_dir(stage_id) / "verification.json"

    def verification_md_path(self, stage_id: str) -> Path:
        return self.stage_dir(stage_id) / "verification.md"

    def handoff_path(self, stage_id: str) -> Path:
        return self.stage_dir(stage_id) / "handoff.md"

    def exists(self) -> bool:
        return self.root.exists()

    def ensure_exists(self) -> None:
        if not self.root.exists():
            raise SgarError(f"SGAR workspace not initialized: {self.root}")

    def initialize(
        self, *, project_name: str | None = None, force: bool = False,
    ) -> ProjectState:
        """Create a fresh workspace. Refuses to overwrite an existing
        one unless ``force=True`` — re-running ``init`` used to silently
        reset ``state.json`` (mode, accepted hashes, stage history),
        destroying governance state while reporting success."""
        if not force and (self.state_path.exists() or self.config_path.exists()):
            raise SgarError(
                f"SGAR workspace already initialized at {self.root}; "
                "refusing to reset state. Re-run with force=True "
                "(CLI: --force) to wipe governance state deliberately."
            )
        self.root.mkdir(parents=True, exist_ok=True)
        self.stages_root.mkdir(parents=True, exist_ok=True)
        self.missions_root.mkdir(parents=True, exist_ok=True)
        name = project_name or self.cwd.name or "sgar-project"
        state = ProjectState(project_name=name, mode=ProjectMode.BLUEPRINT.value)
        self.write_json(self.config_path, {
            "project_name": name,
            "storage": "filesystem",
            "version": 1,
        })
        self.write_state(state)
        self.write_text_if_missing(self.blueprint_path, blueprint_template(name))
        self.write_text_if_missing(self.roadmap_path, roadmap_template())
        stage_dir = self.stage_dir("stage-01")
        stage_dir.mkdir(parents=True, exist_ok=True)
        self.write_text_if_missing(
            self.stage_spec_path("stage-01"),
            stage_spec_template("stage-01"),
        )
        self._ensure_gitignore_entries()
        return state

    def read_text(self, path: Path) -> str:
        if not path.exists():
            raise SgarError(f"missing file: {path}")
        return path.read_text(encoding="utf-8")

    def write_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(path, text)

    def write_text_if_missing(self, path: Path, text: str) -> None:
        if not path.exists():
            self.write_text(path, text)

    def _ensure_gitignore_entries(self) -> None:
        gitignore_path = self.cwd / ".gitignore"
        if not gitignore_path.exists() or not gitignore_path.is_file():
            return

        existing_text = gitignore_path.read_text(encoding="utf-8")
        existing_lines = {line.strip() for line in existing_text.splitlines()}
        missing_entries = [
            entry
            for entry in (".sgar/", ".sgarx/")
            if entry not in existing_lines
        ]
        if not missing_entries:
            return

        updated_text = existing_text
        if updated_text and not updated_text.endswith("\n"):
            updated_text += "\n"
        updated_text += "".join(f"{entry}\n" for entry in missing_entries)
        _atomic_write_text(gitignore_path, updated_text)

    def read_json(self, path: Path) -> dict[str, Any]:
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except FileNotFoundError as exc:
            raise SgarError(f"missing file: {path}") from exc
        except json.JSONDecodeError as exc:
            raise SgarError(f"invalid JSON: {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise SgarError(f"expected object JSON: {path}")
        return data

    def write_json(self, path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(
            path,
            json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )

    def stat_token(self, path: Path) -> tuple[int, int, int] | None:
        """A cheap generation token for ``path`` to pair with
        :meth:`write_json_cas` — ``None`` when ``path`` is absent. Capture it
        when you READ a file, hand it back when you WRITE; a changed token means
        a concurrent writer raced in between. See :func:`_stat_token`."""
        return _stat_token(path)

    def write_json_cas(
        self,
        path: Path,
        data: dict[str, Any],
        *,
        expected: Any,
        precondition: Callable[[], bool] | None = None,
    ) -> None:
        """Atomic JSON write guarded by a compare-and-swap on ``path``'s
        generation token (and an optional ``precondition``).

        ``expected`` is the token captured by :meth:`stat_token` when the caller
        last read ``path`` (``None`` if it was absent). Just before the atomic
        rename, ``path`` is re-stat'd; if its token differs from ``expected`` —
        a concurrent writer landed — the write ABORTS with a loud
        :class:`SgarError` instead of silently clobbering that writer's update.
        ``precondition``, if given, is evaluated in the same just-before-rename
        window and must return truthy or the write aborts likewise (used to
        assert e.g. "this stage is still current").

        This is a single-writer-contract guard, NOT a lock (no wedge on a
        crash) and NOT a guaranteed-lossless CAS (``os.replace`` itself is
        unconditional, so a racer landing in the rename window is still a
        narrow residual) — it converts the COMMON silent clobber into a
        recoverable rejection. The serialized bytes are identical to
        :meth:`write_json`; with ``expected`` matching and ``precondition``
        holding (the single-writer path) it is byte-for-byte equivalent."""
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(
            path,
            json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            expected=expected,
            precondition=precondition,
        )

    def load_state(self) -> ProjectState:
        self.ensure_exists()
        return ProjectState.from_dict(self.read_json(self.state_path))

    def write_state(self, state: ProjectState) -> None:
        self.write_json(self.state_path, state.to_dict())

    def file_hash(self, path: Path) -> str:
        return sha256_text(self.read_text(path))


class _Unchecked:
    """Sentinel distinct from ``None``: ``expected=None`` means "the file was
    absent at read time" (a real CAS expectation), whereas the default means
    "no CAS guard at all" (the plain atomic write)."""

    __slots__ = ()


_UNCHECKED = _Unchecked()


def _stat_token(path: Path) -> tuple[int, int, int] | None:
    """Generation token for ``path``: ``(st_ino, st_size, st_mtime_ns)``, or
    ``None`` if it does not exist.

    Every store write goes through ``mkstemp`` + ``os.replace``, so each atomic
    write lands a brand-new inode under ``path`` — the inode alone flips on
    every write; size and ``mtime_ns`` are belt-and-suspenders for the rare
    inode-reuse case. A token that changed between a read and the matching write
    means a concurrent writer replaced the file in the gap."""
    try:
        st = path.stat()
    except (FileNotFoundError, NotADirectoryError):
        return None
    return (st.st_ino, st.st_size, st.st_mtime_ns)


def _atomic_write_text(
    path: Path,
    content: str,
    *,
    expected: Any = _UNCHECKED,
    precondition: Callable[[], bool] | None = None,
) -> None:
    """Write via a unique temp file in the same directory + ``os.replace``.

    A plain truncate-then-write leaves a half-written ``state.json`` if
    the process dies mid-write, after which every store operation raises
    ``invalid JSON`` until a human repairs the file. The rename is atomic
    on POSIX, so readers always see either the old or the new content.

    Optional compare-and-swap guard (OFF by default → byte-identical to the
    historical behaviour): when ``expected`` is supplied, ``path``'s generation
    token is re-read AFTER the temp file is fully written and IMMEDIATELY BEFORE
    ``os.replace``; if it differs from ``expected`` (a concurrent writer landed
    since the caller read the file) the write is aborted with a loud
    :class:`SgarError` rather than silently clobbering. ``precondition``, if
    given, is checked in the same just-before-rename window and must be truthy.
    Both are best-effort single-writer guards, not a lock — they raise (no
    wedge) and the temp file is always cleaned up on abort."""
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        # Check the precondition FIRST (it may read other files, e.g. state),
        # then re-stat the target LAST so the generation CAS — guarding the
        # headline lost-update — has the tightest possible window: nothing but
        # the os.replace itself stands between the re-stat and the rename.
        if precondition is not None and not precondition():
            raise SgarError(
                f"compare-and-swap aborted: the write precondition for "
                f"{path.name} no longer holds (concurrent state change); "
                "refusing to write a now-stale record"
            )
        if expected is not _UNCHECKED and _stat_token(path) != expected:
            raise SgarError(
                f"compare-and-swap aborted: {path.name} changed since it was "
                "read (a concurrent writer raced this update); refusing to "
                "clobber it"
            )
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def blueprint_template(project_name: str) -> str:
    return f"""# Blueprint: {project_name}

## Problem
Describe the problem this long-horizon coding project solves.

## Goals
- Establish the intended outcomes.

## Non-Goals
- Keep unrelated work out of scope.

## Constraints
- Record technical, operational, and project constraints.

## Success Criteria
- Define how the project will be judged successful.
"""


def roadmap_template() -> str:
    return """# Roadmap

## Stages
- stage-01: Establish the first governed stage.
"""


def stage_spec_template(stage_id: str) -> str:
    return f"""# Stage Spec: {stage_id}

## Objective
Describe the stage objective.

## Scope
- Keep this stage narrow enough to verify.

## Exit Criteria
- C1: Stage implementation is complete.
- C2: Relevant tests or checks have been run and recorded.
"""


def _normalize_session_id(session_id: str | None) -> str | None:
    if session_id is None:
        return None
    value = str(session_id).strip()
    if not value:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise SgarError(
            "SGAR session id may only contain letters, digits, dot, underscore, and hyphen"
        )
    if value in {".", ".."}:
        raise SgarError("invalid SGAR session id")
    return value


__all__ = [
    "SGAR_DIR",
    "SgarStore",
    "blueprint_template",
    "_normalize_session_id",
    "roadmap_template",
    "sha256_text",
    "stage_spec_template",
    "utc_now",
]
