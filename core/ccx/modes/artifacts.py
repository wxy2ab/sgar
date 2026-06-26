"""Markdown artifact writers for plan / spec modes.

cc's plan and spec modes write `plan.md` / `tasks.md` (and corresponding
spec files) under ``<cwd>/.cc/plans/<id>/`` and ``<cwd>/.cc/specs/<id>/``
so the human user can read what the agent decomposed and tick off tasks
manually if desired. These functions reproduce that behaviour from
ccx's mode runners.

Artifacts are advisory — the v5 DAG is the source of truth for execution
state. The markdown files are for humans / external tools that want to
see what the planner produced.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass(slots=True)
class ArtifactPaths:
    artifact_id: str
    root: Path
    plan_or_spec_md: Path
    tasks_md: Path

    def to_dict(self) -> dict[str, str]:
        return {
            "artifact_id": self.artifact_id,
            "root": str(self.root),
            "plan_or_spec_md": str(self.plan_or_spec_md),
            "tasks_md": str(self.tasks_md),
        }


def _new_artifact_id(prefix: str, goal: str) -> str:
    """Mint a compact artifact id without same-second overwrite risk."""
    digest = hashlib.sha256(goal.encode("utf-8")).hexdigest()[:6]
    return f"{prefix}-{time.time_ns()}-{digest}"


def _resolve_artifact_root(
    *, cwd: str | Path, kind: str, artifact_root: str | Path | None,
) -> Path:
    """If artifact_root is given (relative or absolute), use it; else
    default to ``<cwd>/.cc/<kind>``. Always returns an absolute Path."""
    cwd_path = Path(cwd).resolve() if cwd else Path.cwd().resolve()
    if artifact_root is None:
        return cwd_path / ".cc" / kind
    p = Path(artifact_root)
    return p if p.is_absolute() else cwd_path / p


def _format_plan_md(
    *, goal: str, items: Iterable[dict], rationale: str,
) -> str:
    items = list(items)
    lines = [
        f"# Plan: {goal}",
        "",
        f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        "",
    ]
    if rationale:
        lines += ["## Rationale", "", rationale, ""]
    lines += ["## Plan items", ""]
    for i, item in enumerate(items, 1):
        prefix = "→" if item.get("depends_on_previous") else "•"
        lines.append(f"{i}. {prefix} {item['goal']}")
    return "\n".join(lines) + "\n"


def _format_spec_md(
    *, plan_item_goal: str, items: Iterable[dict], rationale: str,
) -> str:
    items = list(items)
    lines = [
        f"# Spec: {plan_item_goal}",
        "",
        f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        "",
    ]
    if rationale:
        lines += ["## Rationale", "", rationale, ""]
    lines += ["## Spec items", ""]
    for i, item in enumerate(items, 1):
        prefix = "→" if item.get("depends_on_previous") else "•"
        lines.append(f"{i}. {prefix} {item['goal']}")
    return "\n".join(lines) + "\n"


def _format_tasks_md(*, header: str, items: Iterable[dict]) -> str:
    lines = [
        f"# Tasks: {header}",
        "",
        "Tick boxes as you complete each item; the agent runtime updates",
        "its DAG independently — this file is purely for humans.",
        "",
    ]
    for item in items:
        lines.append(f"- [ ] {item['goal']}")
    return "\n".join(lines) + "\n"


def write_plan_artifacts(
    *,
    cwd: str | Path,
    goal: str,
    items: Iterable[dict],
    rationale: str = "",
    artifact_root: str | Path | None = None,
) -> ArtifactPaths:
    items = list(items)
    root = _resolve_artifact_root(cwd=cwd, kind="plans",
                                  artifact_root=artifact_root)
    artifact_id = _new_artifact_id("plan", goal)
    target_dir = root / artifact_id
    target_dir.mkdir(parents=True, exist_ok=True)
    plan_md_path = target_dir / "plan.md"
    tasks_md_path = target_dir / "tasks.md"
    plan_md_path.write_text(
        _format_plan_md(goal=goal, items=items, rationale=rationale),
        encoding="utf-8",
    )
    tasks_md_path.write_text(
        _format_tasks_md(header=goal, items=items),
        encoding="utf-8",
    )
    return ArtifactPaths(
        artifact_id=artifact_id,
        root=target_dir,
        plan_or_spec_md=plan_md_path,
        tasks_md=tasks_md_path,
    )


def write_spec_artifacts(
    *,
    cwd: str | Path,
    plan_item_goal: str,
    items: Iterable[dict],
    rationale: str = "",
    artifact_root: str | Path | None = None,
) -> ArtifactPaths:
    items = list(items)
    root = _resolve_artifact_root(cwd=cwd, kind="specs",
                                  artifact_root=artifact_root)
    artifact_id = _new_artifact_id("spec", plan_item_goal)
    target_dir = root / artifact_id
    target_dir.mkdir(parents=True, exist_ok=True)
    spec_md_path = target_dir / "spec.md"
    tasks_md_path = target_dir / "tasks.md"
    spec_md_path.write_text(
        _format_spec_md(plan_item_goal=plan_item_goal, items=items,
                        rationale=rationale),
        encoding="utf-8",
    )
    tasks_md_path.write_text(
        _format_tasks_md(header=plan_item_goal, items=items),
        encoding="utf-8",
    )
    return ArtifactPaths(
        artifact_id=artifact_id,
        root=target_dir,
        plan_or_spec_md=spec_md_path,
        tasks_md=tasks_md_path,
    )


# ``_new_artifact_id`` / ``_resolve_artifact_root`` are exported with a
# leading underscore to mark them as "intentional sibling-mode helpers":
# they are the same id-minting / root-resolving utilities the public
# ``write_plan_artifacts`` / ``write_spec_artifacts`` build on, and
# ``modes/doc.py`` shares the convention so doc-mode artifacts live
# alongside plan/spec under the same per-run id namespace. Exporting
# them via ``__all__`` keeps the C2 boundary scanner from flagging the
# cross-mode import as a private-name violation while still signalling
# to readers that these are not the recommended public surface.
__all__ = [
    "ArtifactPaths",
    "_new_artifact_id",
    "_resolve_artifact_root",
    "write_plan_artifacts",
    "write_spec_artifacts",
]
