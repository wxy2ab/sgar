from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from .config import CCConfig
from .safety.permission_mode import normalize_execute_policy


@dataclass(frozen=True, slots=True)
class ArtifactConfig:
    mode_key: str
    root_key: str
    root_default_attr: str
    task_slug_key: str
    phase_key: str
    default_phase: str
    artifacts_key: str
    status_key: str
    artifact_names: tuple[str, ...]
    artifact_filenames: dict[str, str]
    ready_statuses: set[str]
    ready_flag_key: str
    slug_fallback_prefix: str


def slugify_artifact_task(value: str | None, *, fallback: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower())
    text = text.strip("-")
    if not text:
        return fallback
    return text[:64]


def resolve_artifact_root(
    *,
    cwd: str | Path,
    config: CCConfig,
    state: dict[str, Any] | None,
    artifact_config: ArtifactConfig,
) -> Path:
    current_state = dict(state or {})
    root_value = current_state.get(artifact_config.root_key) or getattr(config, artifact_config.root_default_attr)
    root = Path(str(root_value))
    if not root.is_absolute():
        root = Path(cwd) / root
    return root.resolve()


def build_artifact_map(
    *,
    cwd: str | Path,
    config: CCConfig,
    state: dict[str, Any] | None,
    task_slug: str | None,
    artifact_config: ArtifactConfig,
) -> dict[str, str]:
    current_state = dict(state or {})
    slug = task_slug or str(current_state.get(artifact_config.task_slug_key) or artifact_config.slug_fallback_prefix)
    target_dir = resolve_artifact_root(
        cwd=cwd,
        config=config,
        state=current_state,
        artifact_config=artifact_config,
    ) / slug
    return {
        artifact_name: str((target_dir / artifact_config.artifact_filenames[artifact_name]).resolve())
        for artifact_name in artifact_config.artifact_names
    }


def artifacts_ready(state: dict[str, Any] | None, *, artifact_config: ArtifactConfig) -> bool:
    current_state = dict(state or {})
    artifacts = dict(current_state.get(artifact_config.artifacts_key) or {})
    statuses = dict(current_state.get(artifact_config.status_key) or {})
    for artifact_name in artifact_config.artifact_names:
        if not artifacts.get(artifact_name):
            return False
        if str(statuses.get(artifact_name, "pending")).lower() not in artifact_config.ready_statuses:
            return False
    return True


def ensure_artifact_state(
    *,
    current_state: dict[str, Any] | None,
    cwd: str | Path,
    config: CCConfig,
    task_slug: str | None,
    root_override: str | Path | None,
    source_text: str | None,
    enabled: bool | None,
    artifact_config: ArtifactConfig,
) -> dict[str, Any]:
    state = dict(current_state or {})
    if root_override:
        state[artifact_config.root_key] = str(root_override)
    slug = slugify_artifact_task(
        task_slug or state.get(artifact_config.task_slug_key) or source_text,
        fallback=f"{artifact_config.slug_fallback_prefix}-{Path(cwd).resolve().name}".strip("-"),
    )
    resolved_root = str(
        resolve_artifact_root(
            cwd=cwd,
            config=config,
            state=state,
            artifact_config=artifact_config,
        )
    )
    artifact_map = build_artifact_map(
        cwd=cwd,
        config=config,
        state=state,
        task_slug=slug,
        artifact_config=artifact_config,
    )
    statuses = dict(state.get(artifact_config.status_key) or {})
    for artifact_name in artifact_config.artifact_names:
        statuses.setdefault(artifact_name, "pending")

    next_state = dict(state)
    next_state[artifact_config.mode_key] = bool(enabled if enabled is not None else state.get(artifact_config.mode_key, True))
    next_state[artifact_config.task_slug_key] = slug
    next_state[artifact_config.root_key] = resolved_root
    next_state[artifact_config.phase_key] = str(state.get(artifact_config.phase_key) or artifact_config.default_phase)
    next_state["execute_policy"] = normalize_execute_policy(
        state.get("execute_policy") or config.execute_policy
    )
    next_state[artifact_config.artifacts_key] = artifact_map
    next_state[artifact_config.status_key] = statuses
    next_state[artifact_config.ready_flag_key] = artifacts_ready(next_state, artifact_config=artifact_config)
    return next_state


def is_artifact_path(
    file_path: str | Path,
    *,
    cwd: str | Path,
    config: CCConfig,
    state: dict[str, Any] | None,
    artifact_config: ArtifactConfig,
) -> bool:
    current_state = dict(state or {})
    normalized = Path(file_path).resolve()
    for path_value in build_artifact_map(
        cwd=cwd,
        config=config,
        state=current_state,
        task_slug=None,
        artifact_config=artifact_config,
    ).values():
        if normalized == Path(path_value).resolve():
            return True
    return False
