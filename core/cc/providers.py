"""Environment provider abstractions.

Separates *what a tool does* from *where / how it accesses the environment*.
Inspired by the ``FsProvider`` / ``ShellProvider`` / ``Environment`` pattern
in *open-harness* (``packages/core/src/providers/types.ts``), adapted for the
Python ``core.cc`` tool system.

Design goals:

* **Zero behavioural change** — the default implementations delegate directly
  to ``pathlib`` / ``subprocess`` just like today's tool code.
* **Testability** — tests can inject in-memory providers to avoid real I/O.
* **Future extensibility** — worktree isolation, sandbox, VFS, or remote
  backends can be plugged in via the same interface.

Providers are injected through ``ToolUseContext`` (see the ``environment``
field) so every tool automatically receives them.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# FileSystem provider
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class FileStat:
    is_file: bool
    is_directory: bool
    size: int


@dataclass(slots=True)
class DirEntry:
    name: str
    is_file: bool
    is_directory: bool


@runtime_checkable
class FileSystemProvider(Protocol):
    """Abstraction over filesystem access."""

    async def read_file(self, path: str) -> str: ...

    async def write_file(self, path: str, content: str) -> None: ...

    async def exists(self, path: str) -> bool: ...

    async def stat(self, path: str) -> FileStat: ...

    async def read_dir(self, path: str) -> list[DirEntry]: ...

    async def mkdir(self, path: str, *, parents: bool = False) -> None: ...

    async def remove(self, path: str, *, recursive: bool = False) -> None: ...

    async def rename(self, old_path: str, new_path: str) -> None: ...

    def resolve_path(self, path: str) -> str: ...


# ---------------------------------------------------------------------------
# Shell / command provider
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ShellResult:
    stdout: str
    stderr: str
    exit_code: int
    duration_ms: int = 0
    was_timeout: bool = False

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.was_timeout

    def to_dict(self) -> dict[str, Any]:
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "duration_ms": self.duration_ms,
            "was_timeout": self.was_timeout,
        }


@runtime_checkable
class CommandProvider(Protocol):
    """Abstraction over shell command execution."""

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout_ms: int | None = None,
        shell_kind: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ShellResult: ...


# ---------------------------------------------------------------------------
# Composite environment
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Environment:
    """Bundles filesystem and command providers into a single injectable unit."""

    fs: FileSystemProvider
    shell: CommandProvider


# ---------------------------------------------------------------------------
# Default implementations (delegate to real OS primitives)
# ---------------------------------------------------------------------------

class LocalFileSystemProvider:
    """Real filesystem backed by ``pathlib``."""

    async def read_file(self, path: str) -> str:
        return await asyncio.to_thread(Path(path).read_text, encoding="utf-8")

    async def write_file(self, path: str, content: str) -> None:
        p = Path(path)

        def _write() -> None:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")

        await asyncio.to_thread(_write)

    async def exists(self, path: str) -> bool:
        return await asyncio.to_thread(Path(path).exists)

    async def stat(self, path: str) -> FileStat:
        def _stat() -> FileStat:
            p = Path(path)
            st = p.stat()
            return FileStat(is_file=p.is_file(), is_directory=p.is_dir(), size=st.st_size)

        return await asyncio.to_thread(_stat)

    async def read_dir(self, path: str) -> list[DirEntry]:
        def _read_dir() -> list[DirEntry]:
            return [
                DirEntry(name=entry.name, is_file=entry.is_file(), is_directory=entry.is_dir())
                for entry in Path(path).iterdir()
            ]

        return await asyncio.to_thread(_read_dir)

    async def mkdir(self, path: str, *, parents: bool = False) -> None:
        await asyncio.to_thread(Path(path).mkdir, parents=parents, exist_ok=True)

    async def remove(self, path: str, *, recursive: bool = False) -> None:
        import shutil

        def _remove() -> None:
            p = Path(path)
            if p.is_dir() and recursive:
                shutil.rmtree(p)
            else:
                p.unlink(missing_ok=True)

        await asyncio.to_thread(_remove)

    async def rename(self, old_path: str, new_path: str) -> None:
        await asyncio.to_thread(Path(old_path).rename, new_path)

    def resolve_path(self, path: str) -> str:
        return str(Path(path).resolve())


class LocalCommandProvider:
    """Real shell backed by ``asyncio.create_subprocess_*``."""

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout_ms: int | None = None,
        shell_kind: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ShellResult:
        from .command_runner import execute_command_async

        result = await execute_command_async(
            command=command,
            cwd=cwd or os.getcwd(),
            shell_kind=shell_kind,
            timeout_ms=timeout_ms,
        )
        return ShellResult(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            duration_ms=result.duration_ms,
            was_timeout=result.was_timeout,
        )


def default_environment() -> Environment:
    """Create an ``Environment`` backed by the real local OS."""
    return Environment(fs=LocalFileSystemProvider(), shell=LocalCommandProvider())
