from __future__ import annotations

import asyncio
import logging
from pathlib import Path
import re
import shutil
import time
from typing import Any

from ..safety import classify_file_permission
from ..safety.file_rules import resolve_under_cwd
from .base import BaseTool, ToolCall, ToolResult, ToolSpec, ValidationResult
from .context import ToolUseContext


logger = logging.getLogger(__name__)


def _has_parent_traversal(pattern: str) -> bool:
    """True if a glob pattern contains a ``..`` path segment.

    The ``check_permissions`` gate anchors the *search root* to the workspace,
    but ``root.glob("../*")`` would still climb above that anchored root. Globs
    legitimately never need ``..`` (callers narrow with ``cwd=``), so reject it
    as an information-disclosure vector. NOTE: only the glob argument is checked
    — a grep ``pattern`` is a regex where ``..`` means "any two chars".
    """
    return any(part == ".." for part in str(pattern).replace("\\", "/").split("/"))

_DEFAULT_MAX_RESULTS = 200
_DEFAULT_MAX_FILE_BYTES = 1_000_000
_RG_TIMEOUT_SECONDS = 60
_PYTHON_FALLBACK_TIMEOUT_SECONDS = 60

# Directories the Python fallback skips outright. Without this, a recursive
# ``**/*`` over a workspace that contains a Python virtualenv (`env/` /
# `.venv/`) or `node_modules/` walks 100k+ files and busts the deadline. The
# rg binary already ignores most of these via .gitignore — the fallback has
# no such ergonomics, so we hard-code the worst offenders.
_FALLBACK_SKIP_DIR_NAMES = frozenset({
    ".git", ".hg", ".svn",
    "env", ".venv", "venv",
    "node_modules",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox",
    "dist", "build", ".next", ".nuxt", ".cache",
    ".idea", ".vscode",
})

# When the LLM passes ``file_type="py"`` (or html / md / etc.) we translate
# that to a glob suffix the fallback can use to prune. The mapping mirrors
# rg's well-known type names enough to cover the common cases; unknown types
# fall through to the unfiltered glob (rg semantics: unrecognised type =
# error, but here we degrade gracefully).
_FILE_TYPE_GLOB_SUFFIXES: dict[str, tuple[str, ...]] = {
    "py": ("py",),
    "pyi": ("pyi",),
    "js": ("js", "mjs", "cjs"),
    "ts": ("ts",),
    "tsx": ("tsx",),
    "jsx": ("jsx",),
    "html": ("html", "htm"),
    "css": ("css",),
    "md": ("md", "markdown"),
    "json": ("json",),
    "yaml": ("yaml", "yml"),
    "toml": ("toml",),
    "rust": ("rs",),
    "go": ("go",),
    "java": ("java",),
    "c": ("c", "h"),
    "cpp": ("cpp", "cc", "cxx", "hpp", "hxx", "h"),
    "sh": ("sh", "bash", "zsh"),
    "rb": ("rb",),
    "lua": ("lua",),
}


class GrepTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            ToolSpec(
                name="grep",
                description="Search for a regex pattern in text files under the workspace.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "cwd": {"type": "string"},
                        "glob": {"type": "string"},
                        "max_results": {"type": "integer"},
                        "context_lines": {
                            "type": "integer",
                            "description": "Number of context lines before and after each match (rg -C).",
                        },
                        "file_type": {
                            "type": "string",
                            "description": "Restrict search to file type, e.g. 'py', 'js', 'ts' (rg -t).",
                        },
                        "files_only": {
                            "type": "boolean",
                            "description": "Only list filenames that contain a match (rg --files-with-matches).",
                        },
                    },
                    "required": ["pattern"],
                },
                is_read_only=True,
            )
        )

    def validate_input(self, arguments: dict[str, Any]) -> ValidationResult:
        if not arguments.get("pattern"):
            return ValidationResult(ok=False, message="pattern is required.")
        try:
            re.compile(str(arguments["pattern"]))
        except re.error as exc:
            return ValidationResult(ok=False, message=f"invalid regex: {exc}")
        glob_arg = arguments.get("glob")
        if glob_arg and _has_parent_traversal(glob_arg):
            return ValidationResult(ok=False, message="glob must not contain '..' segments.")
        return ValidationResult(ok=True)

    def check_permissions(self, ctx: ToolUseContext, arguments: dict[str, Any]):
        # Anchor the search root to the workspace. grep is read-only, so an
        # unconstrained ``cwd`` (e.g. ``../..``) is an information-disclosure
        # surface. Reuse the same file-permission classifier file_read uses;
        # operation="read" => a root outside the allowed set returns "ask"
        # (the executor turns that into a blocked result, not a leak).
        return classify_file_permission(
            file_path=arguments.get("cwd") or ctx.cwd,
            cwd=ctx.cwd,
            mode=ctx.permissions.mode,
            allowed_paths=ctx.permissions.allowed_paths,
            denied_paths=ctx.permissions.denied_paths,
            operation="read",
        )

    async def execute(self, tool_call: ToolCall, ctx: ToolUseContext) -> ToolResult:
        root = resolve_under_cwd(tool_call.arguments.get("cwd") or ctx.cwd, ctx.cwd)
        pattern = str(tool_call.arguments["pattern"])
        regex = re.compile(pattern)
        glob_pattern = str(tool_call.arguments.get("glob") or "**/*")
        max_results = max(1, int(tool_call.arguments.get("max_results") or _DEFAULT_MAX_RESULTS))
        raw_context_lines = tool_call.arguments.get("context_lines")
        try:
            context_lines = int(raw_context_lines) if raw_context_lines is not None else None
        except (ValueError, TypeError):
            context_lines = None
        file_type = tool_call.arguments.get("file_type")
        files_only = bool(tool_call.arguments.get("files_only", False))

        rg_result = await self._search_with_rg(
            root=root,
            pattern=pattern,
            glob_pattern=glob_pattern,
            max_results=max_results,
            context_lines=context_lines,
            file_type=str(file_type) if file_type else None,
            files_only=files_only,
        )
        if rg_result is not None:
            return ToolResult(
                tool_use_id=tool_call.tool_use_id,
                tool_name=tool_call.tool_name,
                success=True,
                content=rg_result["content"],
                data=rg_result,
                truncated=bool(rg_result.get("truncated")),
            )

        logger.debug(
            "grep python fallback start cwd=%s pattern=%r glob=%s files_only=%s",
            root, pattern[:80], glob_pattern, files_only,
        )

        # Translate file_type → allowed suffixes for the fallback. Without
        # this the fallback scans every file regardless of the LLM's
        # ``file_type=html`` hint, which is what makes a single grep call
        # walk a 180k-file workspace.
        allowed_suffixes: set[str] | None = None
        if file_type:
            mapped = _FILE_TYPE_GLOB_SUFFIXES.get(str(file_type).lower())
            if mapped is not None:
                allowed_suffixes = {f".{ext}" for ext in mapped}
            # Unknown file_type: fall through (no suffix filter), matches
            # historical behaviour and avoids zero-result runs when callers
            # use exotic type names.

        # Use os.walk + dir-pruning for the default glob ``**/*``; a custom
        # ``glob=`` from the LLM falls back to ``Path.glob`` (original
        # behaviour) since they've expressed an explicit shape. Both paths
        # honour ``allowed_suffixes`` so file_type=html still narrows.
        use_walk = glob_pattern == "**/*"

        def _iter_paths() -> Any:
            """Yield candidate file paths.

            Default path uses ``os.walk`` and prunes well-known noise dirs
            via topdown dirnames mutation — what rg does automatically via
            .gitignore. Custom glob path uses Path.glob unchanged.
            """
            if use_walk:
                import os
                for dirpath, dirnames, filenames in os.walk(
                    str(root), topdown=True, followlinks=False,
                ):
                    dirnames[:] = [d for d in dirnames if d not in _FALLBACK_SKIP_DIR_NAMES]
                    for fname in filenames:
                        if allowed_suffixes is not None:
                            dot = fname.rfind(".")
                            if dot < 0 or fname[dot:].lower() not in allowed_suffixes:
                                continue
                        yield Path(dirpath) / fname
            else:
                for path in root.glob(glob_pattern):
                    if not path.is_file():
                        continue
                    if allowed_suffixes is not None:
                        if path.suffix.lower() not in allowed_suffixes:
                            continue
                    yield path

        def run_search() -> tuple[list[dict[str, object]], int, bool, int]:
            matches: list[dict[str, object]] = []
            seen_files: set[str] = set()
            skipped_large_files = 0
            deadline = time.monotonic() + _PYTHON_FALLBACK_TIMEOUT_SECONDS
            timed_out = False
            entries_visited = 0
            for path in _iter_paths():
                entries_visited += 1
                if time.monotonic() > deadline:
                    timed_out = True
                    break
                try:
                    if path.stat().st_size > _DEFAULT_MAX_FILE_BYTES:
                        skipped_large_files += 1
                        continue
                except OSError:
                    continue
                try:
                    with path.open("r", encoding="utf-8") as handle:
                        for line_no, line in enumerate(handle, start=1):
                            if line_no & 0x3FFF == 0 and time.monotonic() > deadline:
                                timed_out = True
                                break
                            if regex.search(line):
                                if files_only:
                                    fp = str(path)
                                    if fp not in seen_files:
                                        seen_files.add(fp)
                                        matches.append({"file_path": fp})
                                    break
                                matches.append(
                                    {
                                        "file_path": str(path),
                                        "line_number": line_no,
                                        "line": line.rstrip("\n"),
                                    }
                                )
                                if len(matches) >= max_results:
                                    return matches, skipped_large_files, False, entries_visited
                except (UnicodeDecodeError, OSError):
                    continue
                if timed_out:
                    break
            return matches, skipped_large_files, timed_out, entries_visited

        fallback_started_at = time.monotonic()
        matches, skipped_large_files, timed_out, entries_visited = await asyncio.to_thread(run_search)
        fallback_elapsed = time.monotonic() - fallback_started_at
        if timed_out:
            logger.warning(
                "grep python fallback timed out after %.2fs cwd=%s pattern=%r glob=%s entries_visited=%d matches=%d",
                fallback_elapsed, root, pattern[:80], glob_pattern, entries_visited, len(matches),
            )
        else:
            logger.debug(
                "grep python fallback done elapsed=%.2fs entries_visited=%d matches=%d",
                fallback_elapsed, entries_visited, len(matches),
            )
        if files_only:
            content = "\n".join(str(item["file_path"]) for item in matches)
        else:
            content = "\n".join(
                f"{item['file_path']}:{item['line_number']}:{item['line']}"
                for item in matches
            )
        if not files_only and len(matches) >= max_results:
            content = f"{content}\n\n[truncated to {max_results} matches]"
        if timed_out:
            timeout_note = (
                f"\n\n[python fallback hit {_PYTHON_FALLBACK_TIMEOUT_SECONDS}s deadline after "
                f"visiting {entries_visited} entries; partial results — install ripgrep "
                f"or narrow the search with `glob`/`file_type`]"
            )
            content = f"{content}{timeout_note}" if content else timeout_note.lstrip()
        return ToolResult(
            tool_use_id=tool_call.tool_use_id,
            tool_name=tool_call.tool_name,
            success=True,
            content=content,
            data={
                "matches": matches,
                "count": len(matches),
                "cwd": str(root),
                "max_results": max_results,
                "truncated": not files_only and len(matches) >= max_results,
                "engine": "python",
                "skipped_large_files": skipped_large_files,
                "timed_out": timed_out,
                "entries_visited": entries_visited,
            },
            truncated=not files_only and len(matches) >= max_results,
        )

    async def _search_with_rg(
        self,
        *,
        root: Path,
        pattern: str,
        glob_pattern: str,
        max_results: int,
        context_lines: int | None = None,
        file_type: str | None = None,
        files_only: bool = False,
    ) -> dict[str, object] | None:
        rg_path = shutil.which("rg")
        if not rg_path:
            logger.debug(
                "grep rg not on PATH; falling back to Python (cwd=%s pattern=%r)",
                root, pattern[:80],
            )
            return None

        cmd: list[str] = [
            rg_path,
            "--no-heading",
            "--line-number",
            "--color", "never",
        ]
        if files_only:
            cmd.append("--files-with-matches")
        else:
            cmd.extend(["--max-count", str(max_results)])
        if context_lines is not None and context_lines > 0 and not files_only:
            cmd.extend(["-C", str(min(context_lines, 10))])
        if file_type:
            cmd.extend(["-t", file_type])
        else:
            cmd.extend(["--glob", glob_pattern])
        cmd.extend([pattern, "."])

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        rg_started_at = time.monotonic()
        logger.debug(
            "grep rg subprocess spawned pid=%s cwd=%s pattern=%r files_only=%s",
            proc.pid, root, pattern[:80], files_only,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=_RG_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "grep rg subprocess timed out pid=%s after %.2fs cwd=%s pattern=%r",
                proc.pid, time.monotonic() - rg_started_at, root, pattern[:80],
            )
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass
            return None

        rg_elapsed = time.monotonic() - rg_started_at
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        logger.debug(
            "grep rg subprocess exited pid=%s rc=%s elapsed=%.2fs stdout_bytes=%d stderr_bytes=%d",
            proc.pid, proc.returncode, rg_elapsed, len(stdout_bytes), len(stderr_bytes),
        )
        if proc.returncode not in {0, 1}:
            return None

        if files_only:
            file_list = [line.strip() for line in stdout.splitlines() if line.strip()]
            content = "\n".join(str((root / f).resolve()) for f in file_list)
            return {
                "matches": [{"file_path": str((root / f).resolve())} for f in file_list],
                "count": len(file_list),
                "cwd": str(root),
                "max_results": max_results,
                "truncated": False,
                "engine": "rg",
                "stderr": stderr.strip(),
                "content": content,
            }

        if context_lines and context_lines > 0:
            content = stdout.strip()
            match_count = sum(1 for line in stdout.splitlines() if not line.startswith("-") and ":" in line)
            return {
                "matches": [],
                "count": match_count,
                "cwd": str(root),
                "max_results": max_results,
                "truncated": match_count >= max_results and proc.returncode == 0,
                "engine": "rg",
                "stderr": stderr.strip(),
                "content": content if content else "(no matches)",
            }

        matches: list[dict[str, object]] = []
        for line in stdout.splitlines():
            parts = line.split(":", 2)
            if len(parts) != 3:
                continue
            file_part, line_number, matched_line = parts
            try:
                line_no = int(line_number)
            except ValueError:
                continue
            matches.append(
                {
                    "file_path": str((root / file_part).resolve()),
                    "line_number": line_no,
                    "line": matched_line,
                }
            )

        content = "\n".join(
            f"{item['file_path']}:{item['line_number']}:{item['line']}"
            for item in matches
        )
        truncated = len(matches) >= max_results and proc.returncode == 0
        if truncated:
            content = f"{content}\n\n[truncated to {max_results} matches]"
        return {
            "matches": matches,
            "count": len(matches),
            "cwd": str(root),
            "max_results": max_results,
            "truncated": truncated,
            "engine": "rg",
            "stderr": stderr.strip(),
            "content": content,
        }
