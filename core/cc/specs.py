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

SPEC_ARTIFACT_NAMES = ("tasks", "checklist", "spec")
SPEC_READY_STATUSES = {"ready", "completed"}
_SPEC_CONFIG = ArtifactConfig(
    mode_key="spec_mode",
    root_key="spec_root",
    root_default_attr="spec_root",
    task_slug_key="spec_task_slug",
    phase_key="spec_phase",
    default_phase="task",
    artifacts_key="spec_artifacts",
    status_key="spec_artifact_status",
    artifact_names=SPEC_ARTIFACT_NAMES,
    artifact_filenames={"tasks": "tasks.md", "checklist": "checklist.md", "spec": "spec.md"},
    ready_statuses=SPEC_READY_STATUSES,
    ready_flag_key="render_ready",
    slug_fallback_prefix="spec",
)


def slugify_spec_task(value: str | None, *, fallback: str = "spec-task") -> str:
    return slugify_artifact_task(value, fallback=fallback)


def resolve_spec_root(*, cwd: str | Path, config: CCConfig, state: dict[str, Any] | None = None) -> Path:
    return resolve_artifact_root(cwd=cwd, config=config, state=state, artifact_config=_SPEC_CONFIG)


def build_spec_artifact_map(
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
        artifact_config=_SPEC_CONFIG,
    )


def ensure_spec_state(
    *,
    current_state: dict[str, Any] | None,
    cwd: str | Path,
    config: CCConfig,
    task_slug: str | None = None,
    spec_root: str | Path | None = None,
    source_text: str | None = None,
    enabled: bool | None = None,
) -> dict[str, Any]:
    return ensure_artifact_state(
        current_state=current_state,
        cwd=cwd,
        config=config,
        task_slug=task_slug,
        root_override=spec_root,
        source_text=source_text,
        enabled=enabled,
        artifact_config=_SPEC_CONFIG,
    )


def spec_artifacts_ready(state: dict[str, Any] | None) -> bool:
    return artifacts_ready(state, artifact_config=_SPEC_CONFIG)


def is_spec_artifact_path(
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
        artifact_config=_SPEC_CONFIG,
    )
