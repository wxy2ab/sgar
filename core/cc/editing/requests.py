from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class FileEditRequest:
    file_path: str
    old_string: str = ""
    new_string: str = ""
    replace_all: bool = False
    create_if_missing: bool = False
    expected_hash: str | None = None
    validate_python_syntax: bool = True
    runtime_command: str | None = None
    runtime_shell: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PatchPreview:
    file_path: str
    line_count_changed: int
    before_hash: str
    after_hash: str
    diff: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EditValidationResult:
    ok: bool
    stage: str
    messages: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EditResult:
    success: bool
    file_path: str
    content: str
    before_hash: str
    after_hash: str
    preview: PatchPreview | None = None
    validation_results: list[EditValidationResult] = field(default_factory=list)
    checkpoint_id: str | None = None
    rollback_performed: bool = False
    error_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.preview is not None:
            payload["preview"] = self.preview.to_dict()
        payload["validation_results"] = [item.to_dict() for item in self.validation_results]
        return payload


@dataclass(slots=True)
class RollbackResult:
    success: bool
    checkpoint_id: str
    file_path: str
    restored_hash: str | None = None
    error_code: str | None = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
