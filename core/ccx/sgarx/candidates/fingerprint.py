"""Candidate content fingerprints for propose defaults and close-drift gates."""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

from ...sgar.models import SgarError
from .store import normalize_artifact_paths

#: Governance / VCS roots excluded from empty-path workspace fingerprints.
_STORE_DIR_NAMES = frozenset({".sgarx", ".sgar", ".git"})

_STATUS_LINE_RE = re.compile(r"^[MADRCU?!]{1,2}\s+(.*)$")


def _is_store_rel(rel: str) -> bool:
    parts = Path(rel.replace("\\", "/")).parts
    if not parts:
        return False
    return parts[0] in _STORE_DIR_NAMES or any(
        p.startswith(".git") for p in parts
    )


def _git_head(root: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return (proc.stdout or "").strip() or None


def _git_status_paths_excluding_store(root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain", "-uall"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if proc.returncode != 0:
        return ""
    kept: list[str] = []
    for line in (proc.stdout or "").splitlines():
        match = _STATUS_LINE_RE.match(line.strip())
        if match is None:
            continue
        entry = match.group(1).strip().strip('"')
        if " -> " in entry:
            _old, _, new = entry.partition(" -> ")
            entry = new.strip().strip('"')
        if entry and not _is_store_rel(entry):
            kept.append(line.strip())
    return "\n".join(kept)


def _fallback_dir_fingerprint_excluding_store(root: Path) -> str:
    h = hashlib.sha256()
    try:
        entries = sorted(root.rglob("*"))
    except OSError:
        return hashlib.sha256(str(root).encode()).hexdigest()[:16]
    count = 0
    for path in entries:
        if not path.is_file():
            continue
        try:
            rel = str(path.relative_to(root))
        except ValueError:
            continue
        if _is_store_rel(rel):
            continue
        try:
            st = path.stat()
        except OSError:
            continue
        h.update(rel.encode("utf-8", errors="replace"))
        h.update(str(st.st_mtime_ns).encode())
        h.update(str(st.st_size).encode())
        count += 1
        if count >= 500:
            break
    return h.hexdigest()[:16]


def workspace_fingerprint_excluding_store(cwd: Path) -> tuple[str, str | None]:
    """Like syndrome ``workspace_fingerprint``, but ignores ``.sgarx`` / ``.sgar``.

    Used only for empty-path candidate fingerprints so governance writes under
    the store root cannot invalidate close-stage drift checks.
    """
    root = Path(cwd).resolve()
    git_head = _git_head(root)
    status_blob = _git_status_paths_excluding_store(root)
    if git_head is not None or status_blob:
        material = f"head={git_head or ''}\n{status_blob}"
        digest = hashlib.sha256(material.encode("utf-8", errors="replace")).hexdigest()
        return digest[:16], git_head
    return _fallback_dir_fingerprint_excluding_store(root), None


def compute_candidate_fingerprint(
    cwd: Path,
    artifact_paths: list[str] | tuple[str, ...] | None,
) -> str:
    """Stable fingerprint shared by propose defaults and close-stage drift checks.

    * Non-empty ``artifact_paths`` → full sha256 hex over sorted
      ``path\\0bytes`` chunks (missing file → ``SgarError``).
    * Empty paths → workspace fingerprint excluding ``.sgarx`` / ``.sgar``
      (16-char digest). Prefer non-empty paths at propose/from-checks.
    """
    root = Path(cwd).resolve()
    paths = normalize_artifact_paths(list(artifact_paths or ()))
    if not paths:
        digest, _git = workspace_fingerprint_excluding_store(root)
        return digest

    hasher = hashlib.sha256()
    for rel in sorted(paths):
        full = (root / rel).resolve()
        try:
            full.relative_to(root)
        except ValueError as exc:
            raise SgarError(f"artifact path escapes workspace: {rel!r}") from exc
        if not full.is_file():
            raise SgarError(f"artifact file missing for fingerprint: {rel!r}")
        hasher.update(rel.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(full.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()


def compute_candidate_fingerprint_with_git(
    cwd: Path,
    artifact_paths: list[str] | tuple[str, ...] | None,
) -> tuple[str, str | None]:
    """Like :func:`compute_candidate_fingerprint`, also returning optional git HEAD.

    Git head is only filled on the empty-paths workspace fallback; path-based
    hashes leave ``git_head`` as ``None`` unless the caller supplies one.
    """
    root = Path(cwd).resolve()
    paths = normalize_artifact_paths(list(artifact_paths or ()))
    if not paths:
        return workspace_fingerprint_excluding_store(root)
    return compute_candidate_fingerprint(root, paths), None


__all__ = [
    "compute_candidate_fingerprint",
    "compute_candidate_fingerprint_with_git",
    "workspace_fingerprint_excluding_store",
]
