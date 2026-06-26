"""Closed-set verification kinds for ``WatchModeRunner``.

The verifier never runs LLM-decided code at runtime — every check
kind is a hand-written pure-Python predicate over an
``ObservationDigest`` (command output + side-effect snapshot). The
LLM's job in Phase 0 is to *select* and *parametrise* these kinds;
running them is fully deterministic.

A ``check_kind`` is rejected at plan-load time if the name is not in
``CHECK_KINDS``. The LLM cannot smuggle in a new kind by emitting one
the runtime hasn't seen.

Each check is invoked with the ``ObservationDigest`` from the latest
``subprocess.run`` plus the working directory. Result is a
``CheckResult``:

* ``ok=True`` — the predicate held.
* ``ok=False`` — predicate failed; ``reason`` + ``observed`` +
  ``expected`` are surfaced to the fixer prompt so it knows precisely
  what to repair.

Adding a new kind is a deliberate code change: register it in
``CHECK_KINDS`` and document the schema in the Phase 0 prompt. The
prompt advertises the *exact same* closed list — keeping the two in
sync is the responsibility of whoever edits this module.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class ObservationDigest:
    """Snapshot of one command execution + repo side-effects.

    Built by the watch loop after each subprocess.run and consumed by
    the verifier. Contains everything a closed-set check predicate
    might need; nothing here is LLM-derived.
    """
    command: str
    exit_code: int
    stdout_tail: str
    stderr_tail: str
    duration_s: float
    cwd: str = ""
    files_touched: tuple[str, ...] = field(default_factory=tuple)
    timed_out: bool = False
    stdout_path: str = ""
    stderr_path: str = ""
    commit_note: str = ""
    _stdout_full_cache: str | None = field(
        default=None, init=False, repr=False, compare=False,
    )
    _stderr_full_cache: str | None = field(
        default=None, init=False, repr=False, compare=False,
    )

    @property
    def combined_output(self) -> str:
        stdout = self.stdout_full
        stderr = self.stderr_full
        return stdout + ("\n" if stderr else "") + stderr

    @staticmethod
    def _read_spool(path: str, fallback: str) -> str:
        if not path:
            return fallback
        try:
            return Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return fallback

    @property
    def stdout_full(self) -> str:
        if self._stdout_full_cache is None:
            self._stdout_full_cache = self._read_spool(
                self.stdout_path, self.stdout_tail,
            )
        return self._stdout_full_cache

    @property
    def stderr_full(self) -> str:
        if self._stderr_full_cache is None:
            self._stderr_full_cache = self._read_spool(
                self.stderr_path, self.stderr_tail,
            )
        return self._stderr_full_cache

    def to_dict(self) -> dict[str, Any]:
        """Serializable dict form. Used when stuffing the last digest
        into ``SubagentResult.extras`` so a parent invocation can read
        it back without depending on ccx-internal dataclasses."""
        return {
            "command": self.command,
            "exit_code": self.exit_code,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
            "duration_s": self.duration_s,
            "cwd": self.cwd,
            "files_touched": list(self.files_touched),
            "timed_out": self.timed_out,
            "stdout_path": None,
            "stderr_path": None,
            "commit_note": self.commit_note,
        }


@dataclass(slots=True)
class CheckResult:
    ok: bool
    kind: str
    reason: str = ""
    observed: Any = None
    expected: Any = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "ok": self.ok,
            "reason": self.reason,
            "observed": self.observed,
            "expected": self.expected,
        }


CheckSpec = dict[str, Any]
CheckFn = Callable[[CheckSpec, ObservationDigest], CheckResult]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _resolve_path(cwd: str, p: str) -> Path:
    path = Path(p)
    if not path.is_absolute() and cwd:
        path = Path(cwd) / path
    return path


def _walk_json_dot_path(data: Any, dotted: str) -> tuple[bool, Any]:
    """Walk ``data`` by a simple dot-separated path. Returns
    ``(found, value)``. Supports object keys and integer array indices
    (``items.0.id`` style)."""
    cur: Any = data
    if not dotted:
        return True, cur
    for part in dotted.split("."):
        if part == "":
            continue
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
            continue
        if isinstance(cur, list):
            try:
                idx = int(part)
            except ValueError:
                return False, None
            if 0 <= idx < len(cur):
                cur = cur[idx]
                continue
            return False, None
        return False, None
    return True, cur


def _compile_pattern(pattern: str) -> re.Pattern[str] | None:
    try:
        return re.compile(pattern, re.MULTILINE)
    except re.error:
        return None


# --------------------------------------------------------------------------- #
# Check implementations
# --------------------------------------------------------------------------- #

def _check_exit_code(spec: CheckSpec, obs: ObservationDigest) -> CheckResult:
    expected = spec.get("expected", 0)
    try:
        expected = int(expected)
    except (TypeError, ValueError):
        return CheckResult(
            ok=False, kind="exit_code",
            reason="expected must be an int",
            observed=obs.exit_code, expected=expected,
        )
    ok = obs.exit_code == expected
    return CheckResult(
        ok=ok, kind="exit_code",
        reason="" if ok else f"exit_code={obs.exit_code}, expected {expected}",
        observed=obs.exit_code, expected=expected,
    )


def _regex_check(
    kind: str, haystack: str, spec: CheckSpec,
) -> CheckResult:
    pattern = str(spec.get("pattern") or "")
    must_match = spec.get("must_match", True)
    if not pattern:
        return CheckResult(
            ok=False, kind=kind,
            reason="pattern is required",
            observed=None, expected=None,
        )
    rex = _compile_pattern(pattern)
    if rex is None:
        return CheckResult(
            ok=False, kind=kind,
            reason=f"invalid regex {pattern!r}",
            observed=None, expected=pattern,
        )
    matched = rex.search(haystack) is not None
    ok = matched == bool(must_match)
    if ok:
        return CheckResult(
            ok=True, kind=kind, observed=matched, expected=must_match,
        )
    if must_match:
        reason = f"pattern {pattern!r} not found in output"
    else:
        reason = f"forbidden pattern {pattern!r} found in output"
    return CheckResult(
        ok=False, kind=kind, reason=reason,
        observed=matched, expected=must_match,
    )


def _check_stdout_regex(spec: CheckSpec, obs: ObservationDigest) -> CheckResult:
    return _regex_check("stdout_regex", obs.stdout_full, spec)


def _check_stderr_regex(spec: CheckSpec, obs: ObservationDigest) -> CheckResult:
    return _regex_check("stderr_regex", obs.stderr_full, spec)


def _check_combined_output_regex(spec: CheckSpec, obs: ObservationDigest) -> CheckResult:
    return _regex_check("combined_output_regex", obs.combined_output, spec)


def _check_file_exists(spec: CheckSpec, obs: ObservationDigest) -> CheckResult:
    path = str(spec.get("path") or "")
    if not path:
        return CheckResult(
            ok=False, kind="file_exists",
            reason="path is required", observed=None, expected=None,
        )
    p = _resolve_path(obs.cwd, path)
    exists = p.exists()
    return CheckResult(
        ok=exists, kind="file_exists",
        reason="" if exists else f"file not found: {path}",
        observed=exists, expected=True,
    )


def _check_file_not_exists(spec: CheckSpec, obs: ObservationDigest) -> CheckResult:
    path = str(spec.get("path") or "")
    if not path:
        return CheckResult(
            ok=False, kind="file_not_exists",
            reason="path is required", observed=None, expected=None,
        )
    p = _resolve_path(obs.cwd, path)
    exists = p.exists()
    return CheckResult(
        ok=not exists, kind="file_not_exists",
        reason="" if not exists else f"file unexpectedly exists: {path}",
        observed=exists, expected=False,
    )


def _check_file_non_empty(spec: CheckSpec, obs: ObservationDigest) -> CheckResult:
    path = str(spec.get("path") or "")
    if not path:
        return CheckResult(
            ok=False, kind="file_non_empty",
            reason="path is required", observed=None, expected=None,
        )
    p = _resolve_path(obs.cwd, path)
    if not p.exists():
        return CheckResult(
            ok=False, kind="file_non_empty",
            reason=f"file not found: {path}",
            observed=None, expected="exists and size>0",
        )
    try:
        size = p.stat().st_size
    except OSError as exc:
        return CheckResult(
            ok=False, kind="file_non_empty",
            reason=f"stat failed: {exc}",
            observed=None, expected="size>0",
        )
    ok = size > 0
    return CheckResult(
        ok=ok, kind="file_non_empty",
        reason="" if ok else f"file is empty: {path}",
        observed=size, expected="size>0",
    )


def _check_file_size_min(spec: CheckSpec, obs: ObservationDigest) -> CheckResult:
    path = str(spec.get("path") or "")
    try:
        min_bytes = int(spec.get("bytes", 0))
    except (TypeError, ValueError):
        return CheckResult(
            ok=False, kind="file_size_min",
            reason="bytes must be an int",
            observed=None, expected=None,
        )
    if not path:
        return CheckResult(
            ok=False, kind="file_size_min",
            reason="path is required", observed=None, expected=None,
        )
    p = _resolve_path(obs.cwd, path)
    if not p.exists():
        return CheckResult(
            ok=False, kind="file_size_min",
            reason=f"file not found: {path}",
            observed=None, expected=f"size>={min_bytes}",
        )
    size = p.stat().st_size
    ok = size >= min_bytes
    return CheckResult(
        ok=ok, kind="file_size_min",
        reason="" if ok else f"file size {size} < {min_bytes}",
        observed=size, expected=f"size>={min_bytes}",
    )


def _check_file_size_max(spec: CheckSpec, obs: ObservationDigest) -> CheckResult:
    path = str(spec.get("path") or "")
    try:
        max_bytes = int(spec.get("bytes", 0))
    except (TypeError, ValueError):
        return CheckResult(
            ok=False, kind="file_size_max",
            reason="bytes must be an int",
            observed=None, expected=None,
        )
    if not path:
        return CheckResult(
            ok=False, kind="file_size_max",
            reason="path is required", observed=None, expected=None,
        )
    p = _resolve_path(obs.cwd, path)
    if not p.exists():
        return CheckResult(
            ok=False, kind="file_size_max",
            reason=f"file not found: {path}",
            observed=None, expected=f"size<={max_bytes}",
        )
    size = p.stat().st_size
    ok = size <= max_bytes
    return CheckResult(
        ok=ok, kind="file_size_max",
        reason="" if ok else f"file size {size} > {max_bytes}",
        observed=size, expected=f"size<={max_bytes}",
    )


def _check_file_regex(spec: CheckSpec, obs: ObservationDigest) -> CheckResult:
    path = str(spec.get("path") or "")
    if not path:
        return CheckResult(
            ok=False, kind="file_regex",
            reason="path is required", observed=None, expected=None,
        )
    p = _resolve_path(obs.cwd, path)
    if not p.exists():
        return CheckResult(
            ok=False, kind="file_regex",
            reason=f"file not found: {path}",
            observed=None, expected=spec.get("pattern"),
        )
    try:
        content = p.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return CheckResult(
            ok=False, kind="file_regex",
            reason=f"read failed: {exc}",
            observed=None, expected=spec.get("pattern"),
        )
    res = _regex_check("file_regex", content, spec)
    # Patch the result so the reason mentions the file path.
    if not res.ok and res.reason and "in output" in res.reason:
        res.reason = res.reason.replace("in output", f"in {path}")
    return res


def _check_file_json_valid(spec: CheckSpec, obs: ObservationDigest) -> CheckResult:
    path = str(spec.get("path") or "")
    if not path:
        return CheckResult(
            ok=False, kind="file_json_valid",
            reason="path is required", observed=None, expected=None,
        )
    p = _resolve_path(obs.cwd, path)
    if not p.exists():
        return CheckResult(
            ok=False, kind="file_json_valid",
            reason=f"file not found: {path}",
            observed=None, expected="valid JSON",
        )
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        return CheckResult(
            ok=False, kind="file_json_valid",
            reason=f"read failed: {exc}",
            observed=None, expected="valid JSON",
        )
    try:
        json.loads(text)
    except json.JSONDecodeError as exc:
        return CheckResult(
            ok=False, kind="file_json_valid",
            reason=f"invalid JSON at line {exc.lineno} col {exc.colno}: {exc.msg}",
            observed=text[:120], expected="valid JSON",
        )
    return CheckResult(ok=True, kind="file_json_valid", observed=True, expected=True)


def _load_json_file(path: Path) -> tuple[bool, Any, str]:
    if not path.exists():
        return False, None, "file not found"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return False, None, f"read failed: {exc}"
    try:
        return True, json.loads(text), ""
    except json.JSONDecodeError as exc:
        return False, None, f"invalid JSON: {exc.msg}"


def _check_file_json_has_key(spec: CheckSpec, obs: ObservationDigest) -> CheckResult:
    path = str(spec.get("path") or "")
    key = str(spec.get("key") or "")
    if not path or not key:
        return CheckResult(
            ok=False, kind="file_json_has_key",
            reason="path and key are required",
            observed=None, expected=None,
        )
    p = _resolve_path(obs.cwd, path)
    loaded, data, err = _load_json_file(p)
    if not loaded:
        return CheckResult(
            ok=False, kind="file_json_has_key",
            reason=f"{err}: {path}",
            observed=None, expected=f"key={key}",
        )
    found, value = _walk_json_dot_path(data, key)
    if not found:
        return CheckResult(
            ok=False, kind="file_json_has_key",
            reason=f"key not present: {key}",
            observed=None, expected=f"key={key}",
        )
    if "value_equals" in spec:
        expected_value = spec["value_equals"]
        ok = value == expected_value
        return CheckResult(
            ok=ok, kind="file_json_has_key",
            reason="" if ok else f"{key}={value!r}, expected {expected_value!r}",
            observed=value, expected=expected_value,
        )
    return CheckResult(
        ok=True, kind="file_json_has_key",
        observed=value, expected=f"key={key}",
    )


def _check_file_json_path_equals(spec: CheckSpec, obs: ObservationDigest) -> CheckResult:
    path = str(spec.get("path") or "")
    jp = str(spec.get("jsonpath") or "")
    if not path or not jp:
        return CheckResult(
            ok=False, kind="file_json_path_equals",
            reason="path and jsonpath are required",
            observed=None, expected=None,
        )
    if "expected" not in spec:
        return CheckResult(
            ok=False, kind="file_json_path_equals",
            reason="expected value is required",
            observed=None, expected=None,
        )
    p = _resolve_path(obs.cwd, path)
    loaded, data, err = _load_json_file(p)
    if not loaded:
        return CheckResult(
            ok=False, kind="file_json_path_equals",
            reason=f"{err}: {path}",
            observed=None, expected=spec["expected"],
        )
    found, value = _walk_json_dot_path(data, jp)
    if not found:
        return CheckResult(
            ok=False, kind="file_json_path_equals",
            reason=f"jsonpath not found: {jp}",
            observed=None, expected=spec["expected"],
        )
    ok = value == spec["expected"]
    return CheckResult(
        ok=ok, kind="file_json_path_equals",
        reason="" if ok else f"{jp}={value!r}, expected {spec['expected']!r}",
        observed=value, expected=spec["expected"],
    )


# Hard ceiling so a hostile LLM can't ask for a 1-hour subprocess.
_SUBPROCESS_MAX_TIMEOUT_S: float = 300.0
# Pull off a small tail of stdout/stderr for diagnostics — no need to
# accumulate megabytes when this is only used in a failure reason.
_SUBPROCESS_OUTPUT_TAIL_BYTES: int = 4_096


def _check_subprocess(spec: CheckSpec, obs: ObservationDigest) -> CheckResult:
    command = str(spec.get("command") or "")
    if not command:
        return CheckResult(
            ok=False, kind="subprocess",
            reason="command is required",
            observed=None, expected=None,
        )
    try:
        expected_exit = int(spec.get("expected_exit_code", 0))
    except (TypeError, ValueError):
        return CheckResult(
            ok=False, kind="subprocess",
            reason="expected_exit_code must be an int",
            observed=None, expected=None,
        )
    try:
        timeout_raw = spec.get("timeout_s", 60.0)
        timeout_s = float(timeout_raw)
    except (TypeError, ValueError):
        return CheckResult(
            ok=False, kind="subprocess",
            reason="timeout_s must be a number",
            observed=None, expected=None,
        )
    timeout_s = max(1.0, min(timeout_s, _SUBPROCESS_MAX_TIMEOUT_S))

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=obs.cwd or None,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        return CheckResult(
            ok=False, kind="subprocess",
            reason=f"subprocess {command!r} timed out after {timeout_s}s",
            observed={"timed_out": True, "stdout_tail": (exc.stdout or "")[-_SUBPROCESS_OUTPUT_TAIL_BYTES:] if isinstance(exc.stdout, str) else "", "stderr_tail": (exc.stderr or "")[-_SUBPROCESS_OUTPUT_TAIL_BYTES:] if isinstance(exc.stderr, str) else ""},
            expected={"exit_code": expected_exit},
        )
    except OSError as exc:
        return CheckResult(
            ok=False, kind="subprocess",
            reason=f"subprocess {command!r} failed to launch: {exc}",
            observed=None, expected={"exit_code": expected_exit},
        )
    ok = result.returncode == expected_exit
    return CheckResult(
        ok=ok, kind="subprocess",
        reason="" if ok else f"subprocess {command!r} exit_code={result.returncode}, expected {expected_exit}",
        observed={
            "exit_code": result.returncode,
            "stdout_tail": (result.stdout or "")[-_SUBPROCESS_OUTPUT_TAIL_BYTES:],
            "stderr_tail": (result.stderr or "")[-_SUBPROCESS_OUTPUT_TAIL_BYTES:],
        },
        expected={"exit_code": expected_exit},
    )


def _check_git_clean(spec: CheckSpec, obs: ObservationDigest) -> CheckResult:
    """Verify the working tree is clean. ``allowed_globs`` may
    whitelist paths whose dirty state is acceptable (typically the
    fix_scope, so fixer commits don't trip this)."""
    allowed_globs = spec.get("allowed_globs") or []
    if not isinstance(allowed_globs, list):
        allowed_globs = []
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=obs.cwd or None,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        return CheckResult(
            ok=False, kind="git_clean",
            reason=f"git status failed: {exc}",
            observed=None, expected="clean",
        )
    if result.returncode != 0:
        return CheckResult(
            ok=False, kind="git_clean",
            reason=f"git status exited {result.returncode}: {result.stderr.strip()[:200]}",
            observed=None, expected="clean",
        )
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    if not lines:
        return CheckResult(
            ok=True, kind="git_clean",
            observed=[], expected="clean",
        )
    from fnmatch import fnmatch
    bad: list[str] = []
    for ln in lines:
        # Porcelain format: "XY path" (X/Y are 1-char status codes).
        path = ln[3:].strip()
        if any(fnmatch(path, glob) for glob in allowed_globs):
            continue
        bad.append(path)
    if not bad:
        return CheckResult(
            ok=True, kind="git_clean",
            observed=lines, expected="clean (allowed_globs honored)",
        )
    return CheckResult(
        ok=False, kind="git_clean",
        reason=f"{len(bad)} uncommitted path(s) outside allowed_globs: {bad[:5]}",
        observed=bad, expected="clean (allowed_globs honored)",
    )


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

CHECK_KINDS: dict[str, CheckFn] = {
    "exit_code": _check_exit_code,
    "stdout_regex": _check_stdout_regex,
    "stderr_regex": _check_stderr_regex,
    "combined_output_regex": _check_combined_output_regex,
    "file_exists": _check_file_exists,
    "file_not_exists": _check_file_not_exists,
    "file_non_empty": _check_file_non_empty,
    "file_size_min": _check_file_size_min,
    "file_size_max": _check_file_size_max,
    "file_regex": _check_file_regex,
    "file_json_valid": _check_file_json_valid,
    "file_json_has_key": _check_file_json_has_key,
    "file_json_path_equals": _check_file_json_path_equals,
    "subprocess": _check_subprocess,
    "git_clean": _check_git_clean,
}


def supported_kinds() -> list[str]:
    """The exhaustive list of check kinds an LLM may emit. Phase 0's
    prompt advertises this list; any other ``kind`` value is rejected."""
    return sorted(CHECK_KINDS.keys())


def validate_check(spec: CheckSpec) -> str:
    """Validate one check ``spec`` and return ``""`` on success or a
    human-readable reason on failure. Plan-load uses this to drop
    unknown kinds without invoking the predicate."""
    if not isinstance(spec, dict):
        return "check must be an object"
    kind = spec.get("kind")
    if not isinstance(kind, str) or not kind:
        return "check.kind is required"
    if kind not in CHECK_KINDS:
        return f"unknown check.kind={kind!r} (supported: {supported_kinds()})"
    if kind in {"stdout_regex", "stderr_regex", "combined_output_regex", "file_regex"}:
        pattern = spec.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            return f"{kind}.pattern is required"
        if _compile_pattern(pattern) is None:
            return f"{kind}.pattern is not a valid regex"
    if kind in {
        "file_exists",
        "file_not_exists",
        "file_non_empty",
        "file_size_min",
        "file_size_max",
        "file_regex",
        "file_json_valid",
        "file_json_has_key",
        "file_json_path_equals",
    }:
        path = spec.get("path")
        if not isinstance(path, str) or not path:
            return f"{kind}.path is required"
    if kind in {"file_size_min", "file_size_max"}:
        try:
            int(spec.get("bytes", 0))
        except (TypeError, ValueError):
            return f"{kind}.bytes must be an integer"
    if kind == "file_json_has_key":
        key = spec.get("key")
        if not isinstance(key, str) or not key:
            return "file_json_has_key.key is required"
    if kind == "file_json_path_equals":
        jsonpath = spec.get("jsonpath")
        if not isinstance(jsonpath, str) or not jsonpath:
            return "file_json_path_equals.jsonpath is required"
        if "expected" not in spec:
            return "file_json_path_equals.expected is required"
    if kind == "subprocess":
        command = spec.get("command")
        if not isinstance(command, str) or not command:
            return "subprocess.command is required"
        try:
            int(spec.get("expected_exit_code", 0))
        except (TypeError, ValueError):
            return "subprocess.expected_exit_code must be an integer"
        try:
            float(spec.get("timeout_s", 60.0))
        except (TypeError, ValueError):
            return "subprocess.timeout_s must be a number"
    return ""


def run_check(spec: CheckSpec, obs: ObservationDigest) -> CheckResult:
    """Dispatch a single check. ``spec`` must have already passed
    ``validate_check``."""
    fn = CHECK_KINDS[spec["kind"]]
    try:
        return fn(spec, obs)
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            ok=False, kind=str(spec.get("kind") or "?"),
            reason=f"check raised: {type(exc).__name__}: {exc}",
            observed=None, expected=None,
        )


def run_verification_plan(
    plan: dict[str, Any], obs: ObservationDigest,
) -> list[CheckResult]:
    """Run every check in a plan; return only the failures (in plan
    order). Callers treat an empty list as "all checks passed"."""
    checks = plan.get("checks") or []
    failures: list[CheckResult] = []
    for spec in checks:
        if not isinstance(spec, dict):
            continue
        err = validate_check(spec)
        if err:
            failures.append(CheckResult(
                ok=False, kind=str(spec.get("kind") or "?"),
                reason=err, observed=spec, expected=None,
            ))
            continue
        res = run_check(spec, obs)
        if not res.ok:
            failures.append(res)
    return failures


__all__ = [
    "CHECK_KINDS",
    "CheckResult",
    "CheckSpec",
    "ObservationDigest",
    "run_check",
    "run_verification_plan",
    "supported_kinds",
    "validate_check",
]
