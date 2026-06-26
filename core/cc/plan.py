from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifact_state import (
    ArtifactConfig,
    artifacts_ready,
    build_artifact_map,
    ensure_artifact_state,
    is_artifact_path,
    resolve_artifact_root,
    slugify_artifact_task,
)
from .config import CCConfig

PLAN_ARTIFACT_NAMES = ("plan", "tasks")
PLAN_READY_STATUSES = {"ready", "completed"}
_PLAN_CONFIG = ArtifactConfig(
    mode_key="plan_mode",
    root_key="plan_root",
    root_default_attr="plan_root",
    task_slug_key="plan_task_slug",
    phase_key="plan_phase",
    default_phase="planning",
    artifacts_key="plan_artifacts",
    status_key="plan_artifact_status",
    artifact_names=PLAN_ARTIFACT_NAMES,
    artifact_filenames={"plan": "plan.md", "tasks": "tasks.md"},
    ready_statuses=PLAN_READY_STATUSES,
    ready_flag_key="plan_ready",
    slug_fallback_prefix="plan",
)


def slugify_plan_task(value: str | None, *, fallback: str = "plan-task") -> str:
    return slugify_artifact_task(value, fallback=fallback)


def resolve_plan_root(*, cwd: str | Path, config: CCConfig, state: dict[str, Any] | None = None) -> Path:
    return resolve_artifact_root(cwd=cwd, config=config, state=state, artifact_config=_PLAN_CONFIG)


def build_plan_artifact_map(
    *,
    cwd: str | Path,
    config: CCConfig,
    state: dict[str, Any] | None = None,
    task_slug: str | None = None,
) -> dict[str, str]:
    return build_artifact_map(
        cwd=cwd,
        config=config,
        state=state,
        task_slug=task_slug,
        artifact_config=_PLAN_CONFIG,
    )


def ensure_plan_state(
    *,
    current_state: dict[str, Any] | None,
    cwd: str | Path,
    config: CCConfig,
    task_slug: str | None = None,
    plan_root: str | Path | None = None,
    source_text: str | None = None,
    enabled: bool | None = None,
) -> dict[str, Any]:
    return ensure_artifact_state(
        current_state=current_state,
        cwd=cwd,
        config=config,
        task_slug=task_slug,
        root_override=plan_root,
        source_text=source_text,
        enabled=enabled,
        artifact_config=_PLAN_CONFIG,
    )


def plan_artifact_ready(state: dict[str, Any] | None) -> bool:
    return artifacts_ready(state, artifact_config=_PLAN_CONFIG)


def is_plan_artifact_path(
    file_path: str | Path,
    *,
    cwd: str | Path,
    config: CCConfig,
    state: dict[str, Any] | None = None,
) -> bool:
    return is_artifact_path(
        file_path,
        cwd=cwd,
        config=config,
        state=state,
        artifact_config=_PLAN_CONFIG,
    )
