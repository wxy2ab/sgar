from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
import time
import uuid

from .file_state import compute_file_hash
from .requests import RollbackResult

logger = logging.getLogger(__name__)


def _atomic_write_text(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` via temp file + ``os.replace`` (same dir).

    Mirrors ``core.ccx.sgar.store._atomic_write_text`` so a crash mid-write
    cannot leave a truncated ``checkpoints.json`` / ``.bak`` that would wipe
    the rollback index on the next load.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=path.name + ".",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


@dataclass(slots=True)
class RollbackCheckpoint:
    checkpoint_id: str
    file_path: str
    before_hash: str
    backup_path: str
    existed_before: bool = True
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class RollbackManager:
    """Persisted checkpoint store for file-level edits.

    Checkpoints are subject to two GC policies (whichever fires first):
    ``max_checkpoints`` keeps the index bounded, and ``ttl_seconds`` prunes
    entries older than that horizon. Pass ``0`` to disable a given policy.
    """

    def __init__(
        self,
        checkpoint_root: str | Path,
        *,
        max_checkpoints: int = 256,
        ttl_seconds: float = 7 * 24 * 3600,
    ) -> None:
        self.checkpoint_root = Path(checkpoint_root)
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.checkpoint_root / "checkpoints.json"
        self.max_checkpoints = max(0, int(max_checkpoints))
        self.ttl_seconds = max(0.0, float(ttl_seconds))
        self._index: dict[str, RollbackCheckpoint] = {}
        self._load()
        self._gc()

    def create_checkpoint(self, *, file_path: str, content: str, existed_before: bool) -> RollbackCheckpoint:
        checkpoint_id = f"ckpt_{uuid.uuid4().hex[:12]}"
        backup_path = self.checkpoint_root / f"{checkpoint_id}.bak"
        _atomic_write_text(backup_path, content)
        checkpoint = RollbackCheckpoint(
            checkpoint_id=checkpoint_id,
            file_path=file_path,
            before_hash=compute_file_hash(content),
            backup_path=str(backup_path),
            existed_before=existed_before,
        )
        self._index[checkpoint_id] = checkpoint
        self._gc()
        self._persist()
        return checkpoint

    def get(self, checkpoint_id: str) -> RollbackCheckpoint | None:
        return self._index.get(checkpoint_id)

    def restore_checkpoint(self, checkpoint_id: str) -> RollbackResult:
        checkpoint = self.get(checkpoint_id)
        if checkpoint is None:
            return RollbackResult(
                success=False,
                checkpoint_id=checkpoint_id,
                file_path="",
                error_code="ED1007",
                message="Checkpoint not found.",
            )
        backup_path = Path(checkpoint.backup_path)
        file_path = Path(checkpoint.file_path)
        try:
            if not checkpoint.existed_before:
                if file_path.exists():
                    file_path.unlink()
                restored_content = ""
            else:
                file_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(backup_path, file_path)
                restored_content = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            return RollbackResult(
                success=False,
                checkpoint_id=checkpoint_id,
                file_path=checkpoint.file_path,
                error_code="ED1007",
                message=str(exc),
            )
        return RollbackResult(
            success=True,
            checkpoint_id=checkpoint_id,
            file_path=checkpoint.file_path,
            restored_hash=compute_file_hash(restored_content),
            message="Rollback completed.",
        )

    def _gc(self) -> None:
        """Prune checkpoints exceeding TTL or count limits.

        Backup files for evicted checkpoints are removed best-effort.
        """
        if not self._index:
            return
        if self.ttl_seconds > 0:
            cutoff = time.time() - self.ttl_seconds
            for cid in [c.checkpoint_id for c in self._index.values() if c.created_at < cutoff]:
                self._evict(cid)
        if self.max_checkpoints > 0 and len(self._index) > self.max_checkpoints:
            ordered = sorted(self._index.values(), key=lambda c: c.created_at)
            overflow = len(self._index) - self.max_checkpoints
            for ckpt in ordered[:overflow]:
                self._evict(ckpt.checkpoint_id)

    def _evict(self, checkpoint_id: str) -> None:
        ckpt = self._index.pop(checkpoint_id, None)
        if ckpt is None:
            return
        try:
            Path(ckpt.backup_path).unlink(missing_ok=True)
        except OSError:
            pass

    def _persist(self) -> None:
        payload = [checkpoint.to_dict() for checkpoint in self._index.values()]
        _atomic_write_text(
            self.index_path,
            json.dumps(payload, ensure_ascii=False, indent=2),
        )

    def _load(self) -> None:
        if not self.index_path.exists():
            return
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                logger.warning(
                    "Checkpoint index at %s is not a JSON list; resetting index",
                    self.index_path,
                )
                self._index = {}
                return
            self._index = {
                item["checkpoint_id"]: RollbackCheckpoint(**item)
                for item in payload
                if isinstance(item, dict) and "checkpoint_id" in item
            }
        except (OSError, json.JSONDecodeError, TypeError, KeyError):
            logger.warning(
                "Corrupt checkpoint index at %s; resetting index",
                self.index_path,
                exc_info=True,
            )
            self._index = {}
