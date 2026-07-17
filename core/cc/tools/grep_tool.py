from __future__ import annotations

import asyncio
from collections.abc import Callable
import logging
from pathlib import Path
import re
import shutil
import threading
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
# Ceiling on how long ``_search_with_rg`` waits to reap the rg subprocess after
# it SIGKILLs a timed-out rg. ``proc.wait()`` resolves only when the process
# has exited AND its stdout/stderr pipes reach EOF; a child that spawned a
# descendant which inherited those pipes (or an rg wedged in an uninterruptible
# syscall on a slow filesystem) keeps them open, so an unbounded ``await
# proc.wait()`` blocks the whole event loop. The OS + the asyncio child watcher
# still collect the killed process's zombie independently, so abandoning the
# reap here leaks nothing.
_RG_KILL_WAIT_SECONDS = 5
# Hard ceiling above ``_PYTHON_FALLBACK_TIMEOUT_SECONDS`` for the off-loop
# fallback thread. ``run_search`` self-deadlines, but it only checks the
# deadline BETWEEN filesystem entries, so a single ``stat``/``open`` wedged on a
# slow/networked path could run past it. The slack lets the cooperative internal
# deadline win in the normal case (yielding real partial results) while still
# guaranteeing the grep coroutine returns even if a syscall never does.
_FALLBACK_HARD_DEADLINE_SLACK_SECONDS = 10

# Structured tuple ``run_search`` returns: (matches, skipped_large_files,
# timed_out, entries_visited).
_FallbackResult = tuple[list[dict[str, object]], int, bool, int]

# Directories the Python fallback skips outright. Without this, a recursive
# ``**/*`` over a workspace that contains a virtualenv or ``node_modules/``
# walks 100k+ files and busts the deadline. Restricted to CLEARLY-GENERATED
# dirs only: generic source-like names (``env``/``venv``/``build``/``dist``)
# are NOT pruned, because rg searches a *tracked* ``build/`` (it only skips
# gitignored paths) and a bare-basename prune here would make the fallback
# silently miss matches rg finds — an engine-dependent false negative.
_FALLBACK_SKIP_DIR_NAMES = frozenset({
    ".git", ".hg", ".svn",
    ".venv",
    "node_modules",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox",
    ".next", ".nuxt", ".cache",
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


def _is_rg_match_line(line: str) -> bool:
    """True for rg match lines (``path:line:text``), not context (``path-line-text``)."""
    parts = line.split(":", 2)
    if len(parts) != 3:
        return False
    try:
        int(parts[1])
    except ValueError:
        return False
    return True


def _parse_rg_match_lines(stdout: str, root: Path) -> list[dict[str, object]]:
    """Parse rg ``path:line:text`` output into structured matches.

    Shared by the plain and context-lines paths. rg context lines use ``-`` as
    the field separator (``path-line-text``) and group separators are ``--``, so
    they don't parse as ``path:int:text`` and are skipped — leaving only the
    real match lines.
    """
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
    return matches


def _limit_rg_context_content(stdout: str, max_matches: int) -> str:
    """Keep rg ``-C`` stdout covering only the first ``max_matches`` match lines.

    Context lines belonging to kept matches are preserved; later match groups
    (and any excess matches in a shared group) are dropped so ``content`` obeys
    the same global cap as structured ``matches``.
    """
    text = stdout.strip()
    if not text or max_matches <= 0:
        return ""
    kept: list[str] = []
    seen = 0
    for line in text.splitlines():
        if line == "--":
            if seen >= max_matches:
                break
            kept.append(line)
            continue
        if _is_rg_match_line(line):
            if seen >= max_matches:
                break
            seen += 1
        kept.append(line)
    return "\n".join(kept).strip()


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
        # A caller (commonly an LLM) may pass a *file* as ``cwd`` meaning
        # "search within this file". Spawning rg — or running os.walk /
        # Path.glob — with a file as the working directory raises
        # NotADirectoryError ([Errno 20] Not a directory) and sinks the entire
        # call with no signal the model can recover from (observed live: a doc
        # investigator issued 6 greps with cwd=<a .py file>, every one failed
        # ENOTDIR, and the whole review dimension produced nothing). Reinterpret
        # it as a single-file search: anchor the root at the file's parent and
        # narrow the glob to just that file's name. This only triggers when
        # ``cwd`` is an existing file — where the prior behaviour was an
        # unconditional crash — so it is strictly a recovery, not a change to
        # any working path.
        if root.is_file():
            glob_pattern = root.name
            root = root.parent
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

        # Cooperative stop flag for the off-loop worker. When the grep coroutine
        # is cancelled (e.g. a doc investigator's wall-clock timeout) or the hard
        # deadline fires, the walk is ABANDONED; this lets the abandoned thread
        # exit within one entry / 16k lines instead of scanning a whole 180k-file
        # tree after the caller has already given up.
        cancel_event = threading.Event()

        def run_search() -> _FallbackResult:
            matches: list[dict[str, object]] = []
            seen_files: set[str] = set()
            skipped_large_files = 0
            deadline = time.monotonic() + _PYTHON_FALLBACK_TIMEOUT_SECONDS
            timed_out = False
            entries_visited = 0
            for path in _iter_paths():
                entries_visited += 1
                if cancel_event.is_set() or time.monotonic() > deadline:
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
                            if line_no & 0x3FFF == 0 and (
                                cancel_event.is_set() or time.monotonic() > deadline
                            ):
                                timed_out = True
                                break
                            if regex.search(line):
                                if files_only:
                                    fp = str(path)
                                    if fp not in seen_files:
                                        seen_files.add(fp)
                                        matches.append({"file_path": fp})
                                        if len(matches) >= max_results:
                                            return (
                                                matches,
                                                skipped_large_files,
                                                False,
                                                entries_visited,
                                            )
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
        search_result = await self._run_fallback_off_loop(
            run_search,
            cancel_event=cancel_event,
            deadline_seconds=(
                _PYTHON_FALLBACK_TIMEOUT_SECONDS + _FALLBACK_HARD_DEADLINE_SLACK_SECONDS
            ),
        )
        if search_result is None:
            # Hard deadline abandoned a wedged walk (run_search never returned);
            # synthesise a timed-out result so the caller still gets a clean,
            # bounded answer instead of a hang.
            matches, skipped_large_files, timed_out, entries_visited = [], 0, True, 0
        else:
            matches, skipped_large_files, timed_out, entries_visited = search_result
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
        truncated = len(matches) >= max_results
        if files_only:
            content = "\n".join(str(item["file_path"]) for item in matches)
            if truncated:
                content = f"{content}\n\n[truncated to {max_results} files]"
        else:
            content = "\n".join(
                f"{item['file_path']}:{item['line_number']}:{item['line']}"
                for item in matches
            )
            if truncated:
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
                "truncated": truncated,
                "engine": "python",
                "skipped_large_files": skipped_large_files,
                "timed_out": timed_out,
                "entries_visited": entries_visited,
            },
            truncated=truncated,
        )

    async def _run_fallback_off_loop(
        self,
        run_search: Callable[[], _FallbackResult],
        *,
        cancel_event: threading.Event,
        deadline_seconds: float,
    ) -> _FallbackResult | None:
        """Run the blocking python fallback OFF the event loop's default executor.

        ``asyncio.to_thread`` schedules work on the loop's *default*
        ``ThreadPoolExecutor``, which ``asyncio.run`` teardown JOINS via
        ``loop.shutdown_default_executor()`` (up to ``THREAD_JOIN_TIMEOUT`` =
        300s). If the grep coroutine is cancelled — e.g. a doc investigator's
        wall-clock ``asyncio.wait_for`` deadline — while this fallback is still
        walking a huge tree, the worker is un-cancellable and keeps running, so
        teardown wedges joining it. That is the observed cc_query_loop deadlock:
        two concurrent grep calls both fall through rg's 60s timeout into this
        fallback, one gets cancelled, and the whole investigator loop hangs at
        0% CPU (``sample`` shows Thread.join + lock.acquire) for minutes.

        Running the fallback in a *daemon* thread we can ABANDON — never in the
        default executor — means teardown never waits on it. On cancellation or
        the hard ``deadline_seconds`` we set ``cancel_event`` (so the abandoned
        walk stops promptly) and either re-raise (cancellation must propagate)
        or return ``None`` (deadline; caller synthesises a timed-out result).
        Mirrors ``core.cc.conversation.llm_adapter._execute_sync_with_deadline``.
        """
        loop = asyncio.get_running_loop()
        done = asyncio.Event()
        box: dict[str, Any] = {}

        def _worker() -> None:
            try:
                box["result"] = run_search()
            except BaseException as exc:  # noqa: BLE001 — propagate to caller
                box["error"] = exc
            finally:
                try:
                    loop.call_soon_threadsafe(done.set)
                except RuntimeError:
                    pass  # loop already closed; this thread was abandoned

        threading.Thread(
            target=_worker, name="cc-grep-fallback", daemon=True,
        ).start()
        try:
            await asyncio.wait_for(done.wait(), timeout=deadline_seconds)
        except asyncio.CancelledError:
            # Turn/tool cancellation — signal the walk and let cancellation
            # propagate. The daemon thread is abandoned, not joined.
            cancel_event.set()
            raise
        except asyncio.TimeoutError:
            # run_search is wedged past its own internal deadline (a syscall
            # never returned). Abandon it and report a bounded timeout.
            cancel_event.set()
            logger.warning(
                "grep python fallback exceeded hard deadline %.0fs — abandoning "
                "the search thread and returning a timed-out result",
                deadline_seconds,
            )
            return None
        if "error" in box:
            raise box["error"]
        return box.get("result")

    @staticmethod
    async def _kill_and_reap_rg(proc: Any, *, root: Path, pattern: str) -> None:
        """SIGKILL a timed-out rg and reap it under a bounded, non-cancelling wait.

        ``await proc.wait()`` resolves only once the process has exited AND its
        stdout/stderr pipes reach EOF. If rg spawned a descendant that inherited
        those pipes, or rg is wedged in an uninterruptible syscall on a slow
        filesystem, the pipes stay open and an unbounded wait blocks the whole
        event loop — starving every other coroutine in the investigator's turn.

        Two subtleties make this trickier than a plain ``wait_for``:
        (1) ``asyncio.wait_for(proc.wait(), t)`` *cancels* ``proc.wait()`` on
            timeout, and that cancellation re-enters the subprocess transport
            and can itself block on pipe-EOF — reintroducing the hang. So we
            bound with ``asyncio.wait`` (which never cancels) and leave the reap
            detached if it overruns.
        (2) A detached-but-open transport keeps its pipe fds registered on the
            loop; closing the transport releases them so neither the turn nor
            ``asyncio.run`` teardown ever waits on them.

        SIGKILL is already delivered and the OS + asyncio child watcher collect
        the zombie independently, so abandoning the reap leaks no process.
        """
        try:
            proc.kill()
        except ProcessLookupError:
            return  # already gone
        reap = asyncio.ensure_future(proc.wait())
        _, pending = await asyncio.wait({reap}, timeout=_RG_KILL_WAIT_SECONDS)
        if not pending:
            return
        logger.warning(
            "grep rg reap exceeded %.0fs pid=%s cwd=%s pattern=%r — releasing "
            "pipes and abandoning reap (zombie reaped by OS/child watcher)",
            _RG_KILL_WAIT_SECONDS, proc.pid, root, pattern[:80],
        )
        # Best-effort: swallow the eventual result so an abandoned reap never
        # surfaces as "Future exception was never retrieved", and release the
        # pipe fds so the loop stops waiting on them.
        reap.add_done_callback(
            lambda f: f.cancelled() or f.exception() is not None or None
        )
        transport = getattr(proc, "_transport", None)
        if transport is not None:
            try:
                transport.close()
            except Exception:  # noqa: BLE001 — best-effort fd release
                pass

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
            # Skip files larger than the python fallback's own cap. Two reasons:
            # (1) TRIGGER REDUCTION — on a data-heavy repo, rg from cwd=repo-root
            # otherwise spends its whole 60s deadline scanning multi-GB
            # non-gitignored data/binary files, which is what pushes it into the
            # timeout path in the first place. (2) PARITY — the fallback already
            # skips ``st_size > _DEFAULT_MAX_FILE_BYTES``, so without this rg and
            # the fallback would disagree (rg matches a big file, the fallback
            # silently misses it). ``--max-filesize`` takes a raw byte count.
            "--max-filesize", str(_DEFAULT_MAX_FILE_BYTES),
        ]
        if files_only:
            cmd.append("--files-with-matches")
        # NOTE: rg's --max-count is PER FILE, so it cannot serve as the global
        # result cap (it would return up to num_files * max_results lines while
        # the python fallback stops at max_results total). The total cap is
        # applied by slicing the parsed matches below instead.
        if context_lines is not None and context_lines > 0 and not files_only:
            cmd.extend(["-C", str(min(context_lines, 10))])
        # rg accepts -t and --glob together; honor both when supplied so the
        # glob still narrows within a file_type (previously the glob was
        # silently dropped whenever file_type was set).
        if file_type:
            cmd.extend(["-t", file_type])
            if glob_pattern and glob_pattern != "**/*":
                cmd.extend(["--glob", glob_pattern])
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
            await self._kill_and_reap_rg(proc, root=root, pattern=pattern)
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
            truncated = len(file_list) > max_results
            if truncated:
                file_list = file_list[:max_results]
            content = "\n".join(str((root / f).resolve()) for f in file_list)
            if truncated:
                content = f"{content}\n\n[truncated to {max_results} files]"
            return {
                "matches": [{"file_path": str((root / f).resolve())} for f in file_list],
                "count": len(file_list),
                "cwd": str(root),
                "max_results": max_results,
                "truncated": truncated,
                "engine": "rg",
                "stderr": stderr.strip(),
                "content": content,
            }

        if context_lines and context_lines > 0:
            matches = _parse_rg_match_lines(stdout, root)
            truncated = len(matches) > max_results
            if truncated:
                matches = matches[:max_results]
            content = _limit_rg_context_content(stdout, max_results)
            if truncated:
                content = (
                    f"{content}\n\n[truncated to {max_results} matches]"
                    if content
                    else f"[truncated to {max_results} matches]"
                )
            return {
                "matches": matches,
                "count": len(matches),
                "cwd": str(root),
                "max_results": max_results,
                "truncated": truncated,
                "engine": "rg",
                "stderr": stderr.strip(),
                # Context stdout is capped to the same global match limit as
                # structured ``matches`` so display content cannot bypass it.
                "content": content if content else "(no matches)",
            }

        matches = _parse_rg_match_lines(stdout, root)
        # rg's --max-count is per-file, so apply the global cap here by slicing
        # the total parsed matches; ``truncated`` means more than max_results
        # matches existed across all files (parity with the python fallback).
        truncated = len(matches) > max_results
        if truncated:
            matches = matches[:max_results]
        content = "\n".join(
            f"{item['file_path']}:{item['line_number']}:{item['line']}"
            for item in matches
        )
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
