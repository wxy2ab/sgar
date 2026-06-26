from __future__ import annotations

from enum import Enum


class PermissionMode(str, Enum):
    DEFAULT = "default"
    ACCEPT_EDITS = "accept_edits"
    PLAN = "plan"
    SPEC = "spec"
    BYPASS = "bypass"


_SESSION_PERMISSION_MODES = {
    PermissionMode.DEFAULT.value,
    PermissionMode.ACCEPT_EDITS.value,
    PermissionMode.PLAN.value,
    PermissionMode.SPEC.value,
    PermissionMode.BYPASS.value,
}


def normalize_permission_mode(value: str | None) -> str:
    text = str(value or PermissionMode.DEFAULT.value).strip().lower()
    if text in _SESSION_PERMISSION_MODES:
        return text
    return PermissionMode.DEFAULT.value


VALID_EXECUTE_POLICIES = {"auto_execute", "approval_required"}


def normalize_execute_policy(value: str | None, *, default: str = "auto_execute") -> str:
    text = str(value or default).strip().lower()
    return text if text in VALID_EXECUTE_POLICIES else default
