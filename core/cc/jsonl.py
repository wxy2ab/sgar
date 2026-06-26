from __future__ import annotations

import json
import os
from pathlib import Path
import threading
from typing import Any


def append_jsonl_sync(path: str | Path, payload: dict[str, Any]) -> int:
    return append_jsonl_many_sync(path, [payload])


def append_jsonl_many_sync(path: str | Path, payloads: list[dict[str, Any]]) -> int:
    if not payloads:
        return 0
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = b"".join((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8") for payload in payloads)
    with file_path.open("a+b") as handle:
        _lock_file(handle)
        try:
            handle.seek(0, os.SEEK_END)
            handle.write(encoded)
            handle.flush()
        finally:
            _unlock_file(handle)
    return len(encoded)


class JsonlTailReader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._offset = 0
        self._remainder = b""
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self._offset = 0
            self._remainder = b""

    def advance_offset(self, delta: int) -> None:
        if delta <= 0:
            return
        with self._lock:
            self._offset += delta

    def read_new(self) -> list[dict[str, Any]]:
        with self._lock:
            if not self.path.exists():
                return []
            with self.path.open("rb") as handle:
                _lock_file(handle)
                try:
                    handle.seek(self._offset)
                    chunk = handle.read()
                finally:
                    _unlock_file(handle)
            if not chunk:
                return []
            self._offset += len(chunk)
            data = self._remainder + chunk
            lines = data.splitlines(keepends=True)
            if lines and not lines[-1].endswith((b"\n", b"\r")):
                self._remainder = lines.pop()
            else:
                self._remainder = b""
            payloads: list[dict[str, Any]] = []
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped.decode("utf-8"))
                if isinstance(payload, dict):
                    payloads.append(payload)
            return payloads


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
