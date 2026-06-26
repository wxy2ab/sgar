from __future__ import annotations

from pathlib import Path

from .command_rules import classify_command
from .decision import PermissionDecision
from .file_rules import is_unc_path, normalize_path, path_matches_any, resolve_under_cwd
from .permission_mode import PermissionMode, normalize_permission_mode


def classify_file_permission(
    *,
    file_path: str | Path,
    cwd: str | Path,
    mode: str,
    allowed_paths: list[str | Path],
    denied_paths: list[str | Path] | None = None,
    operation: str = "read",
) -> PermissionDecision:
    cwd_path = normalize_path(cwd)
    path = resolve_under_cwd(file_path, cwd_path)
    denied = denied_paths or []
    normalized_mode = normalize_permission_mode(mode)

    if is_unc_path(path):
        return PermissionDecision(status="deny", reason="UNC paths are blocked.", source="file_rules")
    if path_matches_any(path, denied):
        return PermissionDecision(status="deny", reason="Path matches deny rules.", source="file_rules")
    if path_matches_any(path, [cwd_path, *allowed_paths]):
        if normalized_mode in {PermissionMode.PLAN.value, PermissionMode.SPEC.value} and operation in {"write", "edit"}:
            return PermissionDecision(
                status="deny",
                reason="Planning/spec modes are read-mostly and block file mutations.",
                source="permission_mode",
            )
        return PermissionDecision(status="allow", reason="Path is allowed.", source="file_rules")
    if operation == "read":
        return PermissionDecision(status="ask", reason="Path is outside allowed roots.", source="file_rules")
    return PermissionDecision(status="deny", reason="Writing outside allowed roots is blocked.", source="file_rules")


def classify_command_permission(
    *,
    command: str,
    shell_kind: str,
    cwd: str | Path,
    target_cwd: str | Path,
    mode: str,
    allowed_paths: list[str | Path],
    allow_dangerous_commands: bool,
) -> PermissionDecision:
    normalized_mode = normalize_permission_mode(mode)
    classification = classify_command(command, shell_kind=shell_kind)
    target = normalize_path(target_cwd)
    roots = [cwd, *allowed_paths]
    in_allowed_root = path_matches_any(target, roots)

    if not in_allowed_root:
        return PermissionDecision(status="deny", reason="Command cwd is outside allowed roots.", source="command_rules")
    if classification.is_destructive:
        if normalized_mode in {PermissionMode.PLAN.value, PermissionMode.SPEC.value}:
            return PermissionDecision(
                status="deny",
                reason="Destructive commands are blocked in planning/spec modes.",
                source="permission_mode",
            )
        return PermissionDecision(
            status="allow" if allow_dangerous_commands or normalized_mode == PermissionMode.BYPASS.value else "deny",
            reason="Destructive command detected.",
            source="command_rules",
        )
    if normalized_mode in {PermissionMode.PLAN.value, PermissionMode.SPEC.value} and classification.touches_workspace:
        return PermissionDecision(
            status="deny",
            reason="Planning/spec modes block workspace-modifying commands.",
            source="permission_mode",
        )
    if classification.category == "unknown":
        return PermissionDecision(status="ask", reason="Command could not be safely classified.", source="command_rules")
    if classification.touches_network:
        return PermissionDecision(status="ask", reason="Network command requires approval.", source="command_rules")
    return PermissionDecision(status="allow", reason="Command classified as allowed.", source="command_rules")
