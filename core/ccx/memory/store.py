"""cross-run persistent memory; for single-chain resume see deepstack_v5.memory."""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .models import MemoryEntry, normalize_entry, parse_datetime


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MemoryAppendResult:
    stored: int = 0
    duplicates: int = 0
    pruned: int = 0


@dataclass(frozen=True, slots=True)
class _LoadResult:
    entries: list[MemoryEntry]
    skipped: int = 0
    corrupt_rows: list[bytes] | None = None


class JsonlMemoryStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.path = self.root / "memories.jsonl"
        self.corrupt_path = self.root / "memories.corrupt.jsonl"
        self.lock_path = self.root / ".memories.lock"
        self._lock = threading.RLock()

    def load(self) -> tuple[list[MemoryEntry], int]:
        try:
            result = self._read_unlocked()
        except OSError:
            logger.warning("ccx memory: failed to read %s", self.path, exc_info=True)
            return [], 0
        if result.skipped:
            logger.warning(
                "ccx memory: skipped %d unparseable line(s) in %s",
                result.skipped,
                self.path,
            )
        return result.entries, result.skipped

    def append(
        self,
        entries: Iterable[MemoryEntry],
        *,
        max_total_entries: int,
        entry_text_max_chars: int,
        now: datetime | None = None,
    ) -> MemoryAppendResult:
        incoming = [
            normalize_entry(entry, entry_text_max_chars=entry_text_max_chars)
            for entry in entries
            if entry.title and entry.text
        ]
        if not incoming:
            return MemoryAppendResult()
        now = now or datetime.now().astimezone()
        with self._lock:
            try:
                self.root.mkdir(parents=True, exist_ok=True)
                with self.lock_path.open("a+b") as lock_handle:
                    _lock_file(lock_handle)
                    try:
                        load_result = self._load_unlocked_result()
                        existing = load_result.entries
                        if load_result.skipped:
                            logger.warning(
                                "ccx memory: skipped %d unparseable line(s) in %s",
                                load_result.skipped,
                                self.path,
                            )
                        fingerprints = {
                            entry.fingerprint for entry in existing
                            if entry.fingerprint
                        }
                        stored: list[MemoryEntry] = []
                        duplicates = 0
                        for entry in incoming:
                            if entry.fingerprint in fingerprints:
                                duplicates += 1
                                continue
                            fingerprints.add(entry.fingerprint)
                            stored.append(entry)
                        if stored:
                            with self.path.open("ab") as data_handle:
                                data_handle.seek(0, os.SEEK_END)
                                for entry in stored:
                                    line = json.dumps(
                                        entry.to_json_dict(), ensure_ascii=False,
                                    )
                                    data_handle.write((line + "\n").encode("utf-8"))
                                data_handle.flush()
                        all_entries = existing + stored
                        pruned = self._prune_unlocked(
                            all_entries,
                            max_total_entries=max_total_entries,
                            now=now,
                            corrupt_rows=load_result.corrupt_rows or [],
                        )
                    finally:
                        _unlock_file(lock_handle)
            except OSError:
                logger.warning(
                    "ccx memory: failed to append to %s", self.path, exc_info=True,
                )
                return MemoryAppendResult()
        return MemoryAppendResult(
            stored=len(stored),
            duplicates=duplicates,
            pruned=pruned,
        )

    def tag_vocabulary(self, *, limit: int = 30) -> list[tuple[str, int]]:
        entries, _ = self.load()
        counts: dict[str, int] = {}
        for entry in entries:
            for tag in entry.tags:
                counts[tag] = counts.get(tag, 0) + 1
        return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]

    def _load_unlocked(self) -> tuple[list[MemoryEntry], int]:
        result = self._load_unlocked_result()
        return result.entries, result.skipped

    def _load_unlocked_result(self) -> _LoadResult:
        return self._read_unlocked()

    def _read_unlocked(self) -> _LoadResult:
        if not self.path.exists():
            return _LoadResult(entries=[], skipped=0, corrupt_rows=[])
        entries: list[MemoryEntry] = []
        skipped = 0
        corrupt_rows: list[bytes] = []
        with self.path.open("rb") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line.decode("utf-8"))
                    if not isinstance(data, dict):
                        raise ValueError("jsonl row is not an object")
                    entries.append(MemoryEntry.from_json_dict(data))
                except Exception:
                    skipped += 1
                    corrupt_rows.append(_ensure_newline(raw))
        return _LoadResult(
            entries=entries,
            skipped=skipped,
            corrupt_rows=corrupt_rows,
        )

    def _prune_unlocked(
        self,
        entries: list[MemoryEntry],
        *,
        max_total_entries: int,
        now: datetime,
        corrupt_rows: list[bytes],
    ) -> int:
        cap = max(0, max_total_entries)
        active = [
            entry for entry in entries
            if not _is_expired(entry, now)
        ]
        if cap == 0:
            kept: list[MemoryEntry] = []
        elif len(active) <= cap:
            kept = active
        else:
            pinned = [entry for entry in active if entry.pinned]
            unpinned = [entry for entry in active if not entry.pinned]
            unpinned.sort(
                key=lambda entry: _created_sort_key(entry),
                reverse=True,
            )
            remaining = max(0, cap - len(pinned))
            kept = pinned + unpinned[:remaining]
        pruned = len(entries) - len(kept)
        if pruned <= 0:
            return 0
        if corrupt_rows and not self._quarantine_corrupt_rows_unlocked(corrupt_rows):
            return 0
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp_path.open("wb") as tmp:
            for entry in kept:
                line = json.dumps(entry.to_json_dict(), ensure_ascii=False)
                tmp.write((line + "\n").encode("utf-8"))
            tmp.flush()
        os.replace(tmp_path, self.path)
        return pruned

    def _quarantine_corrupt_rows_unlocked(self, rows: list[bytes]) -> bool:
        try:
            with self.corrupt_path.open("ab") as handle:
                for row in rows:
                    handle.write(_ensure_newline(row))
                handle.flush()
        except OSError:
            logger.warning(
                "ccx memory: failed to quarantine %d unparseable line(s) "
                "from %s; leaving journal unchanged",
                len(rows),
                self.path,
                exc_info=True,
            )
            return False
        logger.warning(
            "ccx memory: quarantined %d unparseable line(s) from %s to %s",
            len(rows),
            self.path,
            self.corrupt_path,
        )
        return True


def _created_sort_key(entry: MemoryEntry) -> datetime:
    return parse_datetime(entry.created_at, default_now=True) or datetime.now().astimezone()


def _is_expired(entry: MemoryEntry, now: datetime) -> bool:
    expires_at = parse_datetime(entry.expires_at)
    return expires_at is not None and expires_at < now


def _ensure_newline(row: bytes) -> bytes:
    return row if row.endswith(b"\n") else row + b"\n"


def _lock_file(handle) -> None:
    try:
        import msvcrt  # type: ignore

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        return
    except ImportError:
        pass
    import fcntl  # type: ignore

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _unlock_file(handle) -> None:
    try:
        import msvcrt  # type: ignore

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    except ImportError:
        pass
    import fcntl  # type: ignore

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


__all__ = ["JsonlMemoryStore", "MemoryAppendResult"]
