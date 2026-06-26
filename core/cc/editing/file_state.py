from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any

from ..errors import CCError


def compute_file_hash(content: str) -> str:
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


@dataclass(slots=True)
class FileStateSnapshot:
    file_path: str
    exists: bool
    content: str
    file_hash: str
    size_bytes: int


@dataclass(slots=True)
class FileStateCache:
    snapshots: dict[str, FileStateSnapshot] = field(default_factory=dict)

    def read(self, file_path: str) -> FileStateSnapshot:
        path = Path(file_path)
        if path.exists():
            content = path.read_text(encoding="utf-8")
            snapshot = FileStateSnapshot(
                file_path=str(path),
                exists=True,
                content=content,
                file_hash=compute_file_hash(content),
                size_bytes=len(content.encode("utf-8")),
            )
        else:
            snapshot = FileStateSnapshot(
                file_path=str(path),
                exists=False,
                content="",
                file_hash=compute_file_hash(""),
                size_bytes=0,
            )
        self.snapshots[str(path)] = snapshot
        return snapshot


def assert_file_not_modified(
    *,
    current_hash: str,
    expected_hash: str | None,
) -> None:
    if expected_hash is not None and current_hash != expected_hash:
        raise CCError(
            f"File hash mismatch: expected {expected_hash}, got {current_hash}",
            error_code="ED1004",
        )
