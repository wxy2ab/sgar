"""File-backed SGAR mission isolation records."""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path
from typing import Any

from .models import SgarError
from .store import SgarStore, utc_now


MISSION_STATUS_ACTIVE = "active"
MISSION_STATUS_COMPLETED = "completed"
VALID_MISSION_STATUSES = {MISSION_STATUS_ACTIVE, MISSION_STATUS_COMPLETED}
MISSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def validate_mission_id(mission_id: str) -> str:
    mission_id = mission_id.strip()
    if not MISSION_ID_RE.match(mission_id):
        raise SgarError(
            "mission id must be 1-128 chars of letters, digits, dot, dash, "
            "or underscore, and start with a letter or digit"
        )
    return mission_id


def missions_root(store: SgarStore) -> Path:
    return store.missions_root


def mission_dir(store: SgarStore, mission_id: str) -> Path:
    mission_id = validate_mission_id(mission_id)
    root = missions_root(store).resolve()
    path = (root / mission_id).resolve()
    _ensure_inside(path, root, "mission directory")
    return path


def mission_manifest_path(store: SgarStore, mission_id: str) -> Path:
    return mission_dir(store, mission_id) / "manifest.json"


def mission_context_path(store: SgarStore, mission_id: str) -> Path:
    return mission_dir(store, mission_id) / "context.md"


def create_mission(
    store: SgarStore,
    *,
    mission_id: str,
    kind: str,
    objective: str,
    input_paths: list[str | Path],
    expected_outputs: list[str],
    allowed_scope: list[str] | None = None,
) -> dict[str, Any]:
    store.ensure_exists()
    if not kind.strip():
        raise SgarError("mission kind is required")
    if not objective.strip():
        raise SgarError("mission objective is required")
    if not input_paths:
        raise SgarError("mission requires at least one --input")
    if not expected_outputs:
        raise SgarError("mission requires at least one --expected-output")

    path = mission_dir(store, mission_id)
    if path.exists():
        raise SgarError(f"mission already exists: {mission_id}")

    inputs = [_input_record(store, item) for item in input_paths]
    scope = [
        _clean_scope_item(item)
        for item in (allowed_scope or [record["path"] for record in inputs])
    ]
    expected = [_clean_expected_output(item) for item in expected_outputs]
    manifest = {
        "mission_id": validate_mission_id(mission_id),
        "kind": kind.strip(),
        "objective": objective.strip(),
        "allowed_scope": scope,
        "input_files": inputs,
        "expected_outputs": expected,
        "recorded_outputs": [],
        "status": MISSION_STATUS_ACTIVE,
        "created_at": utc_now(),
        "completed_at": None,
    }
    path.mkdir(parents=True, exist_ok=False)
    store.write_json(path / "manifest.json", manifest)
    store.write_text(path / "context.md", _format_context(store, manifest))
    return manifest


def load_mission(store: SgarStore, mission_id: str) -> dict[str, Any]:
    store.ensure_exists()
    return store.read_json(mission_manifest_path(store, mission_id))


def list_missions(store: SgarStore) -> list[dict[str, Any]]:
    store.ensure_exists()
    root = missions_root(store)
    if not root.exists():
        return []
    manifests: list[dict[str, Any]] = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if not path.is_dir():
            continue
        manifest_path = path / "manifest.json"
        if manifest_path.exists():
            manifests.append(store.read_json(manifest_path))
    return manifests


def mission_staleness(store: SgarStore, manifest: dict[str, Any]) -> list[str]:
    stale: list[str] = []
    for record in _manifest_inputs(manifest):
        rel_path = str(record.get("path") or "")
        expected_hash = str(record.get("sha256") or "")
        try:
            current_hash = _hash_workspace_file(store, rel_path)
        except SgarError:
            stale.append(rel_path or "<missing input>")
            continue
        if expected_hash != current_hash:
            stale.append(rel_path)
    return stale


def complete_mission(
    store: SgarStore,
    *,
    mission_id: str,
    result_path: str | Path,
) -> dict[str, Any]:
    store.ensure_exists()
    mission_id = validate_mission_id(mission_id)
    path = mission_dir(store, mission_id)
    manifest = load_mission(store, mission_id)
    if manifest.get("status") == MISSION_STATUS_COMPLETED:
        raise SgarError(f"mission already completed: {mission_id}")
    if manifest.get("status") not in VALID_MISSION_STATUSES:
        raise SgarError(f"invalid mission status: {manifest.get('status')}")

    source = _resolve_workspace_path(store, result_path, must_exist=True)
    if source.is_dir():
        raise SgarError(f"mission result must be a file: {source}")
    root = path.resolve()
    if _is_inside(source, root):
        recorded_path = source
    else:
        recorded_path = _safe_result_destination(root, source.name)
        shutil.copyfile(source, recorded_path)

    output_record = {
        "path": recorded_path.relative_to(root).as_posix(),
        "sha256": _sha256_file(recorded_path),
        "source_path": _workspace_relative(store, source),
        "recorded_at": utc_now(),
    }
    manifest["recorded_outputs"] = [output_record]
    manifest["status"] = MISSION_STATUS_COMPLETED
    manifest["completed_at"] = utc_now()
    store.write_json(path / "manifest.json", manifest)
    return manifest


def format_mission_status(store: SgarStore, mission_id: str) -> str:
    manifest = load_mission(store, mission_id)
    stale = mission_staleness(store, manifest)
    lines = [
        f"Mission: {manifest.get('mission_id')}",
        f"Kind: {manifest.get('kind')}",
        f"Status: {manifest.get('status')}",
        f"Objective: {manifest.get('objective')}",
        f"Context: {mission_context_path(store, mission_id)}",
        f"Stale inputs: {'yes' if stale else 'no'}",
    ]
    for path in stale:
        lines.append(f"- stale: {path}")
    outputs = manifest.get("recorded_outputs") or []
    if outputs:
        lines.append("Recorded outputs:")
        for output in outputs:
            if isinstance(output, dict):
                lines.append(f"- {output.get('path')}")
    return "\n".join(lines)


def format_mission_list(store: SgarStore) -> str:
    manifests = list_missions(store)
    if not manifests:
        return "No missions."
    lines: list[str] = []
    for manifest in manifests:
        stale = mission_staleness(store, manifest)
        marker = " stale" if stale else ""
        lines.append(
            f"{manifest.get('mission_id')} [{manifest.get('status')}] "
            f"{manifest.get('kind')}{marker}"
        )
    return "\n".join(lines)


def validate_mission_records(store: SgarStore) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    warnings: list[str] = []
    root = missions_root(store)
    if not root.exists():
        return issues, warnings

    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if not path.is_dir():
            continue
        try:
            mission_id = validate_mission_id(path.name)
        except SgarError:
            issues.append(f"invalid mission directory name: {path.name}")
            continue
        manifest_path = path / "manifest.json"
        context_path = path / "context.md"
        if not manifest_path.exists():
            issues.append(f"mission missing manifest: {path}")
            continue
        try:
            manifest = store.read_json(manifest_path)
        except SgarError as exc:
            issues.append(str(exc))
            continue
        issues.extend(_structural_issues(store, mission_id, manifest))
        if not context_path.exists():
            issues.append(f"mission missing context packet: {mission_id}")
        stale = mission_staleness(store, manifest)
        status = manifest.get("status")
        if status == MISSION_STATUS_ACTIVE:
            if stale:
                issues.append(
                    f"active mission has stale inputs: {mission_id}: "
                    + ", ".join(stale)
                )
            else:
                warnings.append(f"incomplete mission: {mission_id}")
        elif status == MISSION_STATUS_COMPLETED and stale:
            warnings.append(
                f"completed mission has stale historical inputs: {mission_id}: "
                + ", ".join(stale)
            )
    return issues, warnings


def _structural_issues(
    store: SgarStore, mission_id: str, manifest: dict[str, Any],
) -> list[str]:
    issues: list[str] = []
    if manifest.get("mission_id") != mission_id:
        issues.append(f"mission id mismatch: {mission_id}")
    if manifest.get("status") not in VALID_MISSION_STATUSES:
        issues.append(f"invalid mission state: {mission_id}")
    for field in [
        "kind",
        "objective",
        "allowed_scope",
        "input_files",
        "expected_outputs",
        "recorded_outputs",
        "created_at",
    ]:
        if field not in manifest:
            issues.append(f"mission manifest missing {field}: {mission_id}")
    if manifest.get("status") == MISSION_STATUS_COMPLETED:
        outputs = manifest.get("recorded_outputs")
        if not outputs:
            issues.append(f"completed mission missing recorded output: {mission_id}")
        elif isinstance(outputs, list):
            root = mission_dir(store, mission_id)
            for output in outputs:
                if not isinstance(output, dict):
                    issues.append(f"invalid mission output record: {mission_id}")
                    continue
                rel_path = str(output.get("path") or "")
                try:
                    path = _resolve_mission_relative(root, rel_path)
                except SgarError as exc:
                    issues.append(str(exc))
                    continue
                if not path.exists():
                    issues.append(
                        f"completed mission recorded output missing: "
                        f"{mission_id}: {rel_path}"
                    )
    return issues


def _input_record(store: SgarStore, item: str | Path) -> dict[str, Any]:
    path = _resolve_workspace_path(store, item, must_exist=True)
    if path.is_dir():
        raise SgarError(f"mission input must be a file: {path}")
    return {
        "path": _workspace_relative(store, path),
        "sha256": _sha256_file(path),
    }


def _manifest_inputs(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    inputs = manifest.get("input_files") or []
    if not isinstance(inputs, list):
        return []
    return [item for item in inputs if isinstance(item, dict)]


def _hash_workspace_file(store: SgarStore, rel_path: str) -> str:
    path = _resolve_workspace_path(store, rel_path, must_exist=True)
    if path.is_dir():
        raise SgarError(f"mission input must be a file: {path}")
    return _sha256_file(path)


def _format_context(store: SgarStore, manifest: dict[str, Any]) -> str:
    lines = [
        f"# Mission Context: {manifest['mission_id']}",
        "",
        f"Kind: {manifest['kind']}",
        f"Created: {manifest['created_at']}",
        "",
        "## Objective",
        "",
        str(manifest["objective"]),
        "",
        "## Isolation Rules",
        "",
        "- Use only the input files listed in this packet.",
        "- Treat chat/session history as non-authoritative.",
        "- Write outputs as mission-local result artifacts.",
        "- Review missions must write review artifacts and not mutate reviewed files.",
        "",
        "## Allowed Scope",
        "",
    ]
    lines.extend(f"- {item}" for item in manifest["allowed_scope"])
    lines.extend(["", "## Expected Outputs", ""])
    lines.extend(f"- {item}" for item in manifest["expected_outputs"])
    lines.extend(["", "## Input Files", ""])
    for record in _manifest_inputs(manifest):
        rel_path = str(record["path"])
        path = _resolve_workspace_path(store, rel_path, must_exist=True)
        lines.extend([
            f"### {rel_path}",
            "",
            f"- sha256: `{record['sha256']}`",
            "",
            "```",
            _read_context_input(path),
            "```",
            "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def _clean_scope_item(item: str) -> str:
    item = str(item).strip()
    if not item:
        raise SgarError("mission allowed scope entries cannot be empty")
    if _looks_like_path(item):
        _validate_relative_path_label(item, "allowed scope")
    return item.replace("\\", "/")


def _clean_expected_output(item: str) -> str:
    item = str(item).strip()
    if not item:
        raise SgarError("mission expected outputs cannot be empty")
    if _looks_like_path(item):
        _validate_relative_path_label(item, "expected output")
    return item.replace("\\", "/")


def _looks_like_path(item: str) -> bool:
    return "/" in item or "\\" in item or item.startswith(".")


def _validate_relative_path_label(item: str, label: str) -> None:
    path = Path(item)
    if path.is_absolute() or ".." in path.parts:
        raise SgarError(f"mission {label} must not escape the workspace: {item}")


def _safe_result_destination(root: Path, name: str) -> Path:
    name = Path(name).name
    if not name:
        raise SgarError("mission result file must have a name")
    destination = (root / name).resolve()
    _ensure_inside(destination, root, "mission result")
    if destination.exists():
        raise SgarError(f"mission result already exists: {destination.name}")
    return destination


def _resolve_workspace_path(
    store: SgarStore,
    item: str | Path,
    *,
    must_exist: bool,
) -> Path:
    raw = Path(item)
    path = raw if raw.is_absolute() else store.cwd / raw
    path = path.resolve()
    _ensure_inside(path, store.cwd, "mission path")
    if must_exist and not path.exists():
        raise SgarError(f"missing mission path: {path}")
    return path


def _resolve_mission_relative(root: Path, item: str) -> Path:
    raw = Path(item)
    if raw.is_absolute() or ".." in raw.parts:
        raise SgarError(f"mission output path escapes mission directory: {item}")
    path = (root / raw).resolve()
    _ensure_inside(path, root, "mission output")
    return path


def _workspace_relative(store: SgarStore, path: Path) -> str:
    _ensure_inside(path.resolve(), store.cwd, "mission path")
    return path.resolve().relative_to(store.cwd).as_posix()


def _ensure_inside(path: Path, root: Path, label: str) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise SgarError(f"{label} escapes workspace: {path}") from exc


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_context_input(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return "[binary content omitted]"


__all__ = [
    "complete_mission",
    "create_mission",
    "format_mission_list",
    "format_mission_status",
    "list_missions",
    "load_mission",
    "mission_context_path",
    "mission_dir",
    "mission_manifest_path",
    "mission_staleness",
    "missions_root",
    "validate_mission_records",
]
