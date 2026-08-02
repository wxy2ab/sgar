"""Version-bound residual / syndrome for governed repair loops (principle 2).

Given a candidate workspace state, verification is residual computation: run
the independent ``[check:]`` gates and package the failing evidence together
with a fingerprint of the candidate the evidence applies to. Old syndromes
must never be used to patch a newer candidate — each redrive builds a fresh
syndrome from the workspace after the latest turn.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


#: One ``git status --porcelain`` line: 1-2 status chars, then the path.
_STATUS_LINE_RE = re.compile(r"^[MADRCU?!]{1,2}\s+(.*)$")


@dataclass(slots=True)
class Syndrome:
    """Structured residual diagnosis bound to one candidate version."""

    candidate_hash: str
    failing_criterion_ids: list[str]
    evidence: list[dict[str, Any]] = field(default_factory=list)
    changed_paths: list[str] = field(default_factory=list)
    git_head: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def workspace_fingerprint(cwd: str | Path) -> tuple[str, str | None]:
    """Return ``(candidate_hash, git_head_or_None)`` for ``cwd``.

    Prefer ``git rev-parse HEAD`` + a short status digest so uncommitted
    edits also move the fingerprint. Fall back to a directory mtime/size
    hash when git is unavailable.
    """
    root = Path(cwd)
    git_head = _git_head(root)
    status_blob = _git_status_blob(root)
    if git_head is not None or status_blob:
        material = f"head={git_head or ''}\n{status_blob}"
        digest = hashlib.sha256(material.encode("utf-8", errors="replace")).hexdigest()
        return digest[:16], git_head
    return _fallback_dir_fingerprint(root), None


def build_syndrome(
    cwd: str | Path,
    failing: Sequence[Any],
    *,
    changed_paths: Iterable[str] | None = None,
) -> Syndrome:
    """Build a syndrome from failing check outcomes (CheckOutcome-like)."""
    candidate_hash, git_head = workspace_fingerprint(cwd)
    evidence: list[dict[str, Any]] = []
    ids: list[str] = []
    for outcome in failing:
        cid = str(getattr(outcome, "criterion_id", "") or "")
        if cid:
            ids.append(cid)
        line = ""
        if hasattr(outcome, "evidence_line"):
            try:
                line = str(outcome.evidence_line())
            except Exception:  # noqa: BLE001
                line = ""
        evidence.append({
            "criterion_id": cid,
            "command": getattr(outcome, "command", None),
            "passed": bool(getattr(outcome, "passed", False)),
            "returncode": getattr(outcome, "returncode", None),
            "timed_out": bool(getattr(outcome, "timed_out", False)),
            "output_tail": getattr(outcome, "output_tail", None),
            "line": line or str(outcome),
        })
    paths = sorted({str(p) for p in (changed_paths or []) if p})
    return Syndrome(
        candidate_hash=candidate_hash,
        failing_criterion_ids=ids,
        evidence=evidence,
        changed_paths=paths,
        git_head=git_head,
    )


def build_syndrome_from_evidence(
    cwd: str | Path,
    failing_evidence: Sequence[dict[str, Any]],
    *,
    changed_paths: Iterable[str] | None = None,
) -> Syndrome:
    """Build a syndrome from already-serialized evidence dicts (goal path)."""
    candidate_hash, git_head = workspace_fingerprint(cwd)
    ids = [
        str(ev.get("criterion_id"))
        for ev in failing_evidence
        if ev.get("criterion_id")
    ]
    paths = sorted({str(p) for p in (changed_paths or []) if p})
    return Syndrome(
        candidate_hash=candidate_hash,
        failing_criterion_ids=ids,
        evidence=[dict(ev) for ev in failing_evidence],
        changed_paths=paths,
        git_head=git_head,
    )


def changed_paths_from_git(cwd: str | Path) -> list[str]:
    """Working-tree paths that differ from HEAD, for incremental verification.

    Returns ``[]`` when git is unavailable or the tree is clean; callers treat
    an empty set as "unknown" and fall back to the full check suite, so a
    missing git never weakens the gate.
    """
    paths: list[str] = []
    for line in _git_status_blob(Path(cwd)).splitlines():
        # ``XY <path>``. Matched rather than sliced at a fixed offset because
        # the blob is stripped, so the first line has lost its leading space.
        match = _STATUS_LINE_RE.match(line.strip())
        if match is None:
            continue
        entry = match.group(1).strip()
        if not entry:
            continue
        # Renames are reported as "old -> new"; both sides can invalidate a check.
        if " -> " in entry:
            old, _, new = entry.partition(" -> ")
            for part in (old, new):
                cleaned = part.strip().strip('"')
                if cleaned and cleaned not in paths:
                    paths.append(cleaned)
            continue
        cleaned = entry.strip('"')
        if cleaned not in paths:
            paths.append(cleaned)
    return sorted(paths)


def format_syndrome_detail(syndrome: Syndrome, *, attempt: int | None = None) -> str:
    """Human-readable block injected into the next repair turn's goal."""
    header = "VERSION-BOUND SYNDROME (residual of the CURRENT candidate)"
    if attempt is not None:
        header = f"{header} — attempt {attempt}"
    lines = [
        header,
        f"candidate_hash={syndrome.candidate_hash}",
    ]
    if syndrome.git_head:
        lines.append(f"git_head={syndrome.git_head}")
    if syndrome.changed_paths:
        lines.append(
            "changed_paths=" + ", ".join(syndrome.changed_paths[:40])
        )
    lines.append(
        "Failing checks below apply ONLY to this candidate version. "
        "Do not reuse an older syndrome. Apply a minimal patch, then "
        "re-verification will recompute the residual."
    )
    lines.append("")
    for ev in syndrome.evidence:
        cid = ev.get("criterion_id") or "?"
        line = ev.get("line") or ev.get("output_tail") or ""
        lines.append(f"- [{cid}] {line}".rstrip())
    return "\n".join(lines)


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


def _git_status_blob(root: Path) -> str:
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
    return (proc.stdout or "").strip()


def _fallback_dir_fingerprint(root: Path) -> str:
    h = hashlib.sha256()
    try:
        entries = sorted(root.rglob("*"))
    except OSError:
        return hashlib.sha256(str(root).encode()).hexdigest()[:16]
    count = 0
    for path in entries:
        if not path.is_file():
            continue
        if any(part.startswith(".git") for part in path.parts):
            continue
        try:
            st = path.stat()
        except OSError:
            continue
        rel = str(path.relative_to(root))
        h.update(rel.encode("utf-8", errors="replace"))
        h.update(str(st.st_mtime_ns).encode())
        h.update(str(st.st_size).encode())
        count += 1
        if count >= 500:
            break
    return h.hexdigest()[:16]


__all__ = [
    "Syndrome",
    "build_syndrome",
    "changed_paths_from_git",
    "build_syndrome_from_evidence",
    "format_syndrome_detail",
    "workspace_fingerprint",
]
