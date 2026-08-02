"""Transactional multi-hunk exact-match edits on a single file."""

from __future__ import annotations

import difflib
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .file_state import compute_file_hash
from .rollback import RollbackManager
from .validator import EditValidator


@dataclass(slots=True)
class BatchEditReceipt:
    index: int
    address: str | None
    old_string: str
    new_string: str
    ok: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "address": self.address,
            "old_string": self.old_string,
            "new_string": self.new_string,
            "ok": self.ok,
            "error": self.error,
        }


@dataclass(slots=True)
class BatchEditResult:
    success: bool
    before_hash: str
    after_hash: str
    file_path: str = ""
    checkpoint_id: str | None = None
    diff: str = ""
    receipts: list[BatchEditReceipt] = field(default_factory=list)
    rollback_performed: bool = False
    error_code: str | None = None
    content: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "file_path": self.file_path,
            "before_hash": self.before_hash,
            "after_hash": self.after_hash,
            "checkpoint_id": self.checkpoint_id,
            "diff": self.diff,
            "receipts": [r.to_dict() for r in self.receipts],
            "rollback_performed": self.rollback_performed,
            "error_code": self.error_code,
        }


def _unified_diff(before: str, after: str, path: str) -> str:
    return "\n".join(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=f"{path}:before",
            tofile=f"{path}:after",
            lineterm="",
        )
    )


def apply_batch_edits_in_memory(
    content: str,
    edits: list[dict[str, Any]],
) -> tuple[str | None, list[BatchEditReceipt]]:
    """Apply ordered exact-match edits in memory. Fail closed on any error."""
    current = content
    receipts: list[BatchEditReceipt] = []
    for i, edit in enumerate(edits):
        old = str(edit.get("old_string", ""))
        new = str(edit.get("new_string", ""))
        address = edit.get("address")
        if address is not None:
            address = str(address)
        if not old:
            receipts.append(
                BatchEditReceipt(
                    index=i,
                    address=address,
                    old_string=old,
                    new_string=new,
                    ok=False,
                    error="empty old_string forbidden in batch patch",
                )
            )
            return None, receipts
        count = current.count(old)
        if count == 0:
            receipts.append(
                BatchEditReceipt(
                    index=i,
                    address=address,
                    old_string=old,
                    new_string=new,
                    ok=False,
                    error="old_string not found",
                )
            )
            return None, receipts
        if count > 1:
            receipts.append(
                BatchEditReceipt(
                    index=i,
                    address=address,
                    old_string=old,
                    new_string=new,
                    ok=False,
                    error=f"old_string matched {count} times",
                )
            )
            return None, receipts
        current = current.replace(old, new, 1)
        receipts.append(
            BatchEditReceipt(
                index=i,
                address=address,
                old_string=old,
                new_string=new,
                ok=True,
            )
        )
    return current, receipts


def apply_batch_edit_to_file(
    *,
    file_path: str | Path,
    edits: list[dict[str, Any]],
    expected_hash: str | None = None,
    rollback_manager: RollbackManager | None = None,
    validator: EditValidator | None = None,
    runtime_command: str | None = None,
    runtime_shell: str | None = None,
) -> BatchEditResult:
    """Apply a batch of exact-match edits as one atomic commit.

    All edits are applied in memory first. The file is written only if every
    edit succeeds; optional runtime validation can still roll back via checkpoint.
    """
    path = Path(file_path)
    file_path_str = str(path)
    if not path.exists():
        return BatchEditResult(
            success=False,
            file_path=file_path_str,
            before_hash="",
            after_hash="",
            error_code="missing_file",
        )
    original = path.read_text(encoding="utf-8")
    before_hash = compute_file_hash(original)
    if expected_hash is not None and before_hash != expected_hash:
        return BatchEditResult(
            success=False,
            file_path=file_path_str,
            before_hash=before_hash,
            after_hash=before_hash,
            error_code="stale_hash",
        )

    updated, receipts = apply_batch_edits_in_memory(original, edits)
    if updated is None:
        return BatchEditResult(
            success=False,
            file_path=file_path_str,
            before_hash=before_hash,
            after_hash=before_hash,
            receipts=receipts,
            rollback_performed=True,
            error_code="edit_failed",
            content=original,
        )

    checkpoint_id: str | None = None
    if rollback_manager is not None:
        checkpoint = rollback_manager.create_checkpoint(
            file_path=file_path_str,
            content=original,
            existed_before=True,
        )
        checkpoint_id = checkpoint.checkpoint_id

    tmp_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as tmp:
            tmp.write(updated)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, path)
        tmp_path = None
    except OSError:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
        if checkpoint_id and rollback_manager is not None:
            rollback_manager.restore_checkpoint(checkpoint_id)
        return BatchEditResult(
            success=False,
            file_path=file_path_str,
            before_hash=before_hash,
            after_hash=before_hash,
            checkpoint_id=checkpoint_id,
            receipts=receipts,
            rollback_performed=True,
            error_code="io_error",
            content=original,
        )

    if runtime_command:
        edit_validator = validator or EditValidator()
        runtime_result = edit_validator.validate_runtime(
            file_path=file_path_str,
            runtime_command=runtime_command,
            runtime_shell=runtime_shell,
        )
        if not runtime_result.ok:
            rolled = False
            if checkpoint_id and rollback_manager is not None:
                rolled = rollback_manager.restore_checkpoint(checkpoint_id).success
            return BatchEditResult(
                success=False,
                file_path=file_path_str,
                before_hash=before_hash,
                after_hash=before_hash,
                checkpoint_id=checkpoint_id,
                receipts=receipts,
                rollback_performed=rolled,
                error_code=runtime_result.error_code or "verifier_fail",
                content=original,
            )

    after_hash = compute_file_hash(updated)
    return BatchEditResult(
        success=True,
        file_path=file_path_str,
        before_hash=before_hash,
        after_hash=after_hash,
        checkpoint_id=checkpoint_id,
        receipts=receipts,
        diff=_unified_diff(original, updated, file_path_str),
        content=updated,
    )
