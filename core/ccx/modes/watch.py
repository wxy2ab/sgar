"""Watch mode runner — derive acceptance from code, run, verify, fix, loop.

``WatchModeRunner`` implements a two-phase pipeline that turns
"run this command and make sure it stays correct" into machine-
executable verification.

* **Phase 0 — CodeAnalyzer**. A read-only cc QueryEngine turn reads
  the source / tests / docs / CI under ``scope`` and emits a
  ``VerificationPlan`` JSON: a list of pre-defined ``check`` specs
  (see ``watch_checks.CHECK_KINDS``). The LLM cannot invent new
  ``kind`` values — Phase 0 validates the plan against the closed
  set before the loop ever starts.
* **Phase 1 — Watch loop**. For up to ``max_iterations`` rounds:
  1. ``subprocess.run`` the user's command.
  2. Build an ``ObservationDigest`` (exit code, output tails, files
     touched).
  3. Run the plan checks (pure Python, no LLM).
  4. If checks pass → return ok.
  5. Otherwise, if a ``fix_scope`` was supplied, drive a cc fixer
     sub-agent with file_edit/file_write **path-guarded** to the
     whitelist. Optionally ``git commit`` the result.

Critical invariants:

* The VerificationPlan never mutates after Phase 0. A fixer that
  can't make checks pass means either the fix is hard (caller's
  problem) or the plan was wrong (also caller's problem) — we never
  let the system "rewrite acceptance to make the test pass".
* ``check.kind`` is a closed enumeration. Any unknown kind makes
  Phase 0 fail. No ``eval`` / ``exec`` / arbitrary Python expressions.
* ``fix_scope`` is enforced at the tool layer via guard wrappers
  around ``file_edit`` and ``file_write`` — the LLM cannot bypass
  it via prompt manipulation. The fixer has NO shell tool, so it
  cannot ``cat > path`` either.
* ``commit_each_fix`` never uses ``--no-verify``. A failing pre-commit
  hook surfaces as a digest annotation on the next iteration so the
  LLM can fix the underlying issue.

Routing matches the other mode runners: a single invocation through
``run()`` covers Phase 0 + the whole Phase 1 loop, so v5 sees one
terminal NodeSpec.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from ..agents.read_only_runner import restrict_tool_registry
from ..agents.subagent import (
    ModeRunner,
    SubagentInvocation,
    SubagentResult,
)
from .llm_client import LLMCallable, text_of
from .parsing import parse_llm_json
from .prompts import PromptLoadError, load_mode_prompts
from .watch_checks import (
    CheckResult,
    ObservationDigest,
    run_verification_plan,
    supported_kinds,
    validate_check,
)


logger = logging.getLogger(__name__)


# Per-iteration cap on the subprocess command. The plan can override
# this via the top-level ``per_iteration_timeout_s`` key (which Phase 0
# may emit), bounded by the remaining wall-clock budget.
_DEFAULT_PER_ITERATION_TIMEOUT_S: float = 600.0
# Cap on stdout/stderr captured into the digest, to keep prompt sizes
# bounded when the LLM sees the failure.
_OUTPUT_TAIL_BYTES: int = 4_096
_SPOOL_DIR_PREFIX = "ccx-watch-output-"
_SPOOL_LEGACY_DIR_NAME = "ccx-watch-output"
_SPOOL_PID_FILE = "owner.pid"
_SPOOL_TTL_S = 24 * 60 * 60
_MODULE_SPOOL_FILE_LIMIT = 256
_MODULE_SPOOL_DIR: Path | None = None
_MODULE_SPOOL_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# Phase 0 system prompts
# --------------------------------------------------------------------------- #

# Built lazily at module-import time; the supported-kinds list is the
# single source of truth for what the LLM may emit.
def _kinds_doc_block() -> str:
    """Render the closed-set check-kinds reference. The Phase 0
    prompt embeds this verbatim so the LLM sees exactly which kinds
    are accepted; any drift between this list and ``CHECK_KINDS``
    would cause Phase 0 to consistently emit rejected kinds."""
    return (
        "Each check has a ``kind`` from this CLOSED SET. The runtime "
        "rejects any other kind — do NOT invent new ones.\n\n"
        "  * ``exit_code`` — {kind, expected: int}\n"
        "  * ``stdout_regex`` — {kind, pattern: str, must_match: bool}\n"
        "  * ``stderr_regex`` — {kind, pattern: str, must_match: bool}\n"
        "  * ``combined_output_regex`` — {kind, pattern, must_match}\n"
        "  * ``file_exists`` — {kind, path: str}\n"
        "  * ``file_not_exists`` — {kind, path: str}\n"
        "  * ``file_non_empty`` — {kind, path: str}\n"
        "  * ``file_size_min`` — {kind, path, bytes: int}\n"
        "  * ``file_size_max`` — {kind, path, bytes: int}\n"
        "  * ``file_regex`` — {kind, path, pattern, must_match}\n"
        "  * ``file_json_valid`` — {kind, path}\n"
        "  * ``file_json_has_key`` — {kind, path, key: \"dot.path.OK\", "
        "value_equals?: any}\n"
        "  * ``file_json_path_equals`` — {kind, path, jsonpath, "
        "expected: any}\n"
        "  * ``subprocess`` — {kind, command, expected_exit_code: int, "
        "timeout_s: number}  ← use sparingly (slows verification)\n"
        "  * ``git_clean`` — {kind, allowed_globs?: list[str]}\n\n"
        "If the check you want isn't in this list, DECOMPOSE it into "
        "the closest combination of these kinds, OR set "
        "``confidence: \"low\"`` so the caller can take over. Do NOT "
        "invent kinds."
    )


_ANALYZER_SYSTEM_EN = """\
You are an ACCEPTANCE PLAN ANALYZER (read-only).

Your job: read the code under ``scope``, then emit a machine-\
executable VerificationPlan that defines what "success" means when \
the user runs their command. You are NOT writing code, NOT fixing \
bugs, NOT running the command. You ONLY read code + tests + docs + \
CI config to derive acceptance criteria.

==========================================================================
TOOLS (case-sensitive — only these)
==========================================================================
* ``file_read`` — args: ``file_path`` (required), ``max_bytes`` \
(default 100_000). Reads UTF-8 text.
* ``glob`` — args: ``pattern`` (required), ``cwd`` (search root), \
``max_results``. Returns matching file paths.
* ``grep`` — args: ``pattern`` (required), ``cwd``, ``glob``, \
``files_only`` (bool), ``file_type`` (e.g. ``"py"``), ``context_lines``, \
``max_results``.

You cannot edit files, run shell commands, or execute the command \
under review. (If you need to "see what it does", read its source.)

==========================================================================
WORKFLOW
==========================================================================
1. Enumerate ``scope`` with ``glob`` (e.g. ``"**/*.py"``).
2. Find entry points: ``main``, ``__main__``, handler functions, \
   ``if __name__ == "__main__"`` blocks. Read them.
3. Read tests in scope — tests are the most concrete statement of \
   "what is correct behavior".
4. Read README / docs / module docstrings for documented invariants \
   (return value shapes, expected files produced, etc.).
5. Read CI config / Makefile / ``pyproject.toml`` ``[tool.*]`` \
   sections — declared lint/test commands, required artifacts.
6. Apply ``caller_hints`` (if supplied) — those are guidance, not a \
   substitute for reading code.

==========================================================================
DEDUPLICATION RULES — never repeat the same call (violation → shallow)
==========================================================================
* DO NOT ``file_read`` the same path twice with the same or smaller \
  ``max_bytes``. If a prior read returned ``[truncated to N bytes]`` \
  and you need more content, DOUBLE the ``max_bytes`` \
  (20_000 → 60_000 → 100_000) on the next read, then stop. \
  **A single path must never be ``file_read`` more than 3 times.**
* DO NOT issue the same ``grep`` ``pattern`` + ``cwd`` combination \
  twice. ``file_read`` the file(s) you already saw — do NOT re-grep.
* DO NOT use ``grep`` to enumerate files. Use ``glob`` for listings.
* Before each tool call, mentally check: have I already called this?

==========================================================================
WHEN TO STOP AND EMIT JSON (hard exit conditions)
==========================================================================
Emit JSON THIS ROUND and stop making tool calls if ANY of:

* You have already ``file_read`` 8-12 files representative of the \
  scope. More reads past this point don't make the plan better.
* You have ≥ 30 total file_read + grep + glob calls AND the last 5 \
  produced no new structural information (no new path, no new \
  invariant, no new criterion).
* You catch yourself wanting to repeat a call you already made — \
  exit signal, emit JSON now.

==========================================================================
OUTPUT — STRICT JSON, no preamble, no fences, no commentary
==========================================================================
{
  "summary": "<1-3 sentences: what should this command do; how do \
we know it succeeded; what would fail look like>",
  "checks": [
    {"kind": "exit_code", "expected": 0},
    {"kind": "stdout_regex", "pattern": "^OK$", "must_match": true},
    ...
  ],
  "confidence": "high" | "medium" | "low",
  "limits": "<what you couldn't determine from the code, or empty>"
}

%(KINDS_DOC)s

Rules:

* ``checks`` must contain AT LEAST one ``exit_code`` check (every \
  command has an exit code; saying nothing about it is hiding a check).
* Every check kind MUST be from the closed list above. Unknown kinds \
  → plan rejected, mode aborts.
* Patterns are Python ``re`` (multiline). Test mentally: would this \
  regex actually match the output? If unsure, prefer a looser \
  pattern + ``must_match=true`` over a strict one that misses real \
  passes.
* Prefer 3-7 checks over 1 mega-check or 20 micro-checks. A "correct \
  run" should fail at the FIRST plausible regression, not pass with \
  empty output.
* Confidence calibration: if you couldn't read tests OR couldn't \
  find a clear success signal in the source, set ``confidence: \
  "low"`` and explain in ``limits``. The caller is told to inspect \
  low-confidence plans before trusting them.
""" % {"KINDS_DOC": _kinds_doc_block()}


_ANALYZER_SYSTEM_ZH = """\
你是**验收方案分析员**（只读）。

任务：阅读 ``scope`` 下的代码，给用户的命令导出一个**机器可执行的 \
VerificationPlan**——定义"这条命令跑成功"具体是什么意思。你**不**\
写代码、**不**修 bug、**不**运行命令。只读代码 + 测试 + 文档 + CI \
配置，推导验收标准。

==========================================================================
工具（区分大小写——只有这些）
==========================================================================
* ``file_read`` —— 参数：``file_path``（必填）、``max_bytes``（默认 \
  100_000）。读 UTF-8 文本。
* ``glob`` —— 参数：``pattern``（必填）、``cwd``、``max_results``。\
  返回匹配的文件路径。
* ``grep`` —— 参数：``pattern``（必填）、``cwd``、``glob``、\
  ``files_only``、``file_type``、``context_lines``、``max_results``。

不能修改文件、不能执行 shell、不能跑被审命令。要"看看它做了啥"——\
读它的源码。

==========================================================================
工作流
==========================================================================
1. 用 ``glob`` 枚举 ``scope``（如 ``"**/*.py"``）。
2. 找入口：``main``、``__main__``、handler 函数、\
   ``if __name__ == "__main__"`` 块。读它们。
3. 读 scope 下的测试——测试是"什么算正确"最具体的表达。
4. 读 README / 文档 / 模块 docstring——已写下的不变式（返回值形状、\
   必产文件等）。
5. 读 CI 配置 / Makefile / ``pyproject.toml`` 的 ``[tool.*]`` 段——\
   声明的 lint / test 命令、要求的 artifact。
6. 参考 ``caller_hints``（若有）——是提示，不替代读代码。

==========================================================================
去重规则——同样的调用不要发第二次
==========================================================================
* 同一文件不要用同样或更小的 ``max_bytes`` ``file_read`` 两次以上。\
  上一次返回了 ``[truncated to N bytes]`` 且需要更多——下次 \
  ``max_bytes`` 翻倍（20_000 → 60_000 → 100_000），然后停。**\
  同一路径一个 turn 内最多 3 次。**
* 同一 ``grep`` ``pattern`` + ``cwd`` 不要发第二次。已经有命中——\
  直接 ``file_read`` 看到的文件。
* **不要**用 ``grep`` 列文件，用 ``glob``。

==========================================================================
何时停下出 JSON（硬性退出条件）
==========================================================================
满足任意一条**这一轮就出 JSON**：

* 已经 ``file_read`` 8-12 个代表性文件——再多对 plan 没有边际价值。
* 累计 ≥ 30 次 file_read + grep + glob 且最近 5 次没产生新的结构信息。
* 想再发"已经发过"的调用——退出信号，立刻出 JSON。

==========================================================================
输出——严格 JSON，无前后文、无 fence、无解说
==========================================================================
{
  "summary": "<1-3 句：这条命令该做什么；怎么算成功；什么样就算挂了>",
  "checks": [
    {"kind": "exit_code", "expected": 0},
    {"kind": "stdout_regex", "pattern": "^OK$", "must_match": true},
    ...
  ],
  "confidence": "high" | "medium" | "low",
  "limits": "<从代码里推不出来的部分，或空字符串>"
}

%(KINDS_DOC)s

规则：

* ``checks`` 必须至少包含一个 ``exit_code`` check——所有命令都有 \
  exit code，对它不作判定就是在藏 check。
* 每条 check 的 ``kind`` 必须在上面的闭集里。未知 kind → 整个 plan \
  作废、mode 退出。
* 模式串是 Python ``re``（multiline）。心里过一遍：这个正则真的能\
  匹中预期输出吗？不确定 → 宁可放宽 + ``must_match=true``，也不要\
  写太死把真实通过情况漏掉。
* **3-7 条 check** 比 1 条巨型 check 或 20 条微型 check 更好。一次"正常\
  跑"应该在**最早出现回归的地方失败**，而不是空着也能过。
* Confidence 校准：如果没读到测试、或源码里看不到清晰的成功信号，\
  设 ``confidence: "low"`` 并在 ``limits`` 里说明。调用方被告知 \
  low-confidence plan 要先人审。
""" % {"KINDS_DOC": _kinds_doc_block()}


# --------------------------------------------------------------------------- #
# Fixer system prompt
# --------------------------------------------------------------------------- #

_FIXER_SYSTEM_EN = """\
You are a CODE FIXER subagent. The user ran a command, the command \
output failed verification, and your job is to make the verification \
pass next time.

==========================================================================
WHAT YOU GET
==========================================================================
* The VerificationPlan (read-only — you cannot modify it).
* The ObservationDigest from the last command run.
* The list of check failures with concrete reasons.
* The ``fix_scope`` whitelist of files you may edit.

==========================================================================
TOOLS
==========================================================================
* ``file_read`` / ``glob`` / ``grep`` — read anything in the repo.
* ``file_edit`` / ``file_write`` — write **only** to paths matching \
  ``fix_scope``. Writes outside that whitelist are rejected by the \
  runtime with ``error_code=FIX_SCOPE_VIOLATION`` and your change \
  does NOT apply. Don't argue with the guard — pick a different file.

You do NOT have shell. You cannot run the command yourself; the \
runner will rerun it after you finish. You cannot bypass git hooks; \
if a commit fails the pre-commit hook, the next iteration's digest \
will tell you what the hook said and you should FIX the underlying \
issue rather than asking for ``--no-verify``.

==========================================================================
WORKFLOW
==========================================================================
1. Read the failures carefully. Each failure tells you the ``kind``, \
   the ``reason``, what was ``observed``, and what was ``expected``.
2. Pick the failure most likely to be the root cause. Many of the \
   listed failures may share one underlying bug — fix that, not the \
   symptoms.
3. ``file_read`` the relevant code under ``fix_scope``. Trace the \
   path from the command's entry point to where the wrong behavior \
   is produced.
4. ``file_edit`` (or ``file_write``) the minimum number of files to \
   correct the behavior. Stay inside ``fix_scope``.
5. When done, emit a short final message describing what you \
   changed and why. The runner re-runs the command after this.

DO NOT modify the test files or success-criteria files to "make the \
test pass". The plan is the contract; your job is to make the code \
satisfy the plan, not the other way around. If you genuinely think \
the plan is wrong, say so in your final message — the runner will \
report it, and a human will arbitrate. Don't change the test/CI \
config to dodge a real failure.
"""


_FIXER_SYSTEM_ZH = """\
你是**代码修复子代理**。用户跑了一条命令，命令输出没通过验收，你的\
任务是让下一次运行通过验收。

==========================================================================
你能拿到的输入
==========================================================================
* VerificationPlan（**只读**——不能修改）。
* 上一次命令运行的 ObservationDigest（exit code、stdout/stderr 尾部、\
  耗时、被改动的文件）。
* check failures 列表，每条都说明了 ``kind``、``reason``、``observed``、\
  ``expected``。
* ``fix_scope`` 白名单——你能改的文件。

==========================================================================
工具
==========================================================================
* ``file_read`` / ``glob`` / ``grep``：读仓库里任何文件。
* ``file_edit`` / ``file_write``：**只能**写命中 ``fix_scope`` glob \
  的路径。写白名单外的文件会被运行时拒绝，返回 \
  ``error_code=FIX_SCOPE_VIOLATION``，改动不会落盘。**不要跟保护机制\
  扯**——换一个能改的文件。

**没有 shell**。你不能自己跑命令；runner 会在你结束后重跑。**不能\
绕过 git hook**；如果 pre-commit hook 让 commit 失败了，下一轮的 \
digest 会告诉你 hook 说了什么——你要去**修根因**，不要请求 \
``--no-verify``。

==========================================================================
工作流
==========================================================================
1. 仔细读 failures。每一条都给了 ``kind``/``reason``/``observed``/\
   ``expected``。
2. 挑最可能是根因的那一条。很多 failures 通常共享一个 bug——\
   **修根因**，不是修症状。
3. ``file_read`` ``fix_scope`` 下的相关代码。从命令入口点一路追到产生\
   错误行为的位置。
4. ``file_edit``（或 ``file_write``）改**最少**的文件让行为对上。\
   **不要出 ``fix_scope`` 范围**。
5. 完成后给一个简短的 final message 说清你改了什么、为什么。runner \
   随后会重跑命令。

**不要**改测试文件或验收标准来"让测试过"。Plan 是契约——你要让代码\
满足 plan，不是反过来。如果你真心觉得 plan 错了——在 final message \
里说出来。runner 会把这句话报告出去，人来仲裁。**不要**改 test / CI \
配置去躲避真实失败。
"""


# --------------------------------------------------------------------------- #
# Prompt loader (R/C1) — TOML is authoritative; constants above are
# byte-equivalent fallbacks for when the data file is missing.
# --------------------------------------------------------------------------- #


def _load_analyzer_system(language: str) -> str:
    """Return the analyzer system prompt, from TOML if available, else
    the compiled-in fallback constant. Mirrors plan.py's pattern; logs
    once per process if the TOML is unavailable so the deploy issue is
    visible.

    Note: the constants ``_ANALYZER_SYSTEM_EN`` / ``_ANALYZER_SYSTEM_ZH``
    embed the closed-set ``%(KINDS_DOC)s`` substitution at module
    import time. The TOML stores the already-substituted final text,
    so this loader does no further substitution. A change to
    ``CHECK_KINDS`` requires regenerating both the constants (auto
    via the `%`-format at import) AND the TOML file — the
    byte-equivalence test catches drift.
    """
    try:
        prompts = load_mode_prompts("watch_analyzer")
        return prompts.system_for(language)
    except PromptLoadError as exc:
        logger.warning(
            "watch_analyzer prompts TOML unavailable (%s); "
            "using fallback constants", exc,
        )
        return (
            _ANALYZER_SYSTEM_ZH if language.startswith("zh")
            else _ANALYZER_SYSTEM_EN
        )


def _load_fixer_system(language: str) -> str:
    """Return the fixer system prompt, from TOML if available, else
    the compiled-in fallback constant."""
    try:
        prompts = load_mode_prompts("watch_fixer")
        return prompts.system_for(language)
    except PromptLoadError as exc:
        logger.warning(
            "watch_fixer prompts TOML unavailable (%s); "
            "using fallback constants", exc,
        )
        return (
            _FIXER_SYSTEM_ZH if language.startswith("zh")
            else _FIXER_SYSTEM_EN
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _parse_plan_json(raw: str) -> dict[str, Any] | None:
    """Best-effort parse of the analyzer's JSON. Returns ``None`` when
    no parseable dict can be recovered — the round skips this plan and
    waits for the next analyzer pass.
    """
    if not raw:
        return None
    return parse_llm_json(
        raw,
        schema_name="watch_plan",
        fallback_factory=lambda _raw: None,
        expected_type=dict,
    )


def _validate_plan(plan: Any) -> tuple[bool, str, list[str]]:
    """Validate a parsed plan object. Returns ``(ok, reason, kept_kinds)``.

    Rules:
    * ``plan`` must be a dict with a list ``checks``.
    * Every entry in ``checks`` must pass ``watch_checks.validate_check``.
    * At least one ``exit_code`` check is required (system prompt
      explicitly asks for one; a plan without it is hiding a critical
      check).
    """
    if not isinstance(plan, dict):
        return False, "plan is not a JSON object", []
    checks = plan.get("checks")
    if not isinstance(checks, list) or not checks:
        return False, "plan.checks must be a non-empty list", []
    kinds: list[str] = []
    for i, spec in enumerate(checks):
        err = validate_check(spec)
        if err:
            return False, f"checks[{i}]: {err}", []
        kinds.append(spec["kind"])
    if "exit_code" not in kinds:
        return False, (
            "plan must include at least one exit_code check; "
            "every command has an exit code and omitting it hides a "
            "critical signal"
        ), kinds
    return True, "", kinds


def _coerce_scope(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(x).strip() for x in raw if str(x).strip()]
    text = str(raw).strip()
    if not text:
        return []
    # Allow comma-separated or whitespace-separated single string.
    parts = [p.strip() for p in re.split(r"[,\n]+", text) if p.strip()]
    return parts


def _coerce_fix_scope(raw: Any) -> list[str]:
    """Parse ``fix_scope`` into a list of glob patterns. Same parser
    as scope, kept separate for callsite readability."""
    return _coerce_scope(raw)


def _path_matches_any_glob(path: str, globs: list[str], cwd: str) -> bool:
    """Match ``path`` against the fix_scope globs. The LLM may give
    either repo-relative paths (``src/foo/bar.py``) or absolute
    paths; we normalize before matching. Empty ``globs`` → never
    matches (= no permission to edit anything).

    Matching happens on the NORMALIZED path (``..`` collapsed) — never
    on the raw string. fnmatch's ``**`` happily matches ``../../..``,
    so a raw match would let ``scope/**/*.py`` approve
    ``scope/../../outside.py`` while the downstream file tool resolves
    the traversal and writes outside the scope. Paths that escape
    ``cwd`` are rejected outright."""
    if not globs:
        return False
    p = Path(path)
    candidates: set[str] = set()
    if cwd:
        cwd_norm = Path(os.path.normpath(str(Path(cwd))))
        absolute = p if p.is_absolute() else cwd_norm / p
        norm = Path(os.path.normpath(str(absolute)))
        try:
            rel = norm.relative_to(cwd_norm)
        except ValueError:
            # Normalized path escapes the workspace — never in scope.
            return False
        candidates.add(str(rel))
        candidates.add(rel.as_posix())
        candidates.add(str(norm))
        candidates.add(norm.as_posix())
    else:
        norm = Path(os.path.normpath(str(p)))
        if ".." in norm.parts:
            # Cannot anchor a traversal without a cwd — reject.
            return False
        candidates.add(str(norm))
        candidates.add(norm.as_posix())
    for cand in candidates:
        for glob in globs:
            if fnmatch(cand, glob):
                return True
    return False


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """Kill the spawned shell and every descendant. On POSIX the child
    was started with ``start_new_session=True`` so its process group id
    equals its pid — ``killpg`` takes down the whole tree. Falls back
    to ``proc.kill()`` (shell only) when the group is already gone or
    on non-POSIX platforms."""
    if os.name == "posix":
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass
    elif os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
            )
            return
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
    try:
        proc.kill()
    except (ProcessLookupError, OSError):
        pass


def _cleanup_stale_spool_dirs() -> None:
    root = Path(tempfile.gettempdir())
    cutoff = time.time() - _SPOOL_TTL_S
    try:
        entries = list(root.glob(f"{_SPOOL_DIR_PREFIX}*"))
        legacy = root / _SPOOL_LEGACY_DIR_NAME
        if legacy.exists():
            entries.append(legacy)
    except OSError:
        return
    for entry in entries:
        try:
            if not entry.is_dir():
                continue
            if entry.stat().st_mtime >= cutoff:
                continue
            if _spool_owner_alive(entry):
                continue
            shutil.rmtree(entry, ignore_errors=True)
        except OSError:
            continue


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return True
        return str(pid) in result.stdout
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except (OSError, ValueError):
        return False
    return True


def _spool_owner_alive(path: Path) -> bool:
    def _pid_from_dir_name() -> int | None:
        match = re.match(
            rf"^{re.escape(_SPOOL_DIR_PREFIX)}(\d+)-",
            path.name,
        )
        if match is None:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    try:
        raw = (path / _SPOOL_PID_FILE).read_text(encoding="utf-8").strip()
        pid = int(raw)
    except (OSError, ValueError):
        pid_from_name = _pid_from_dir_name()
        return _pid_alive(pid_from_name) if pid_from_name is not None else False
    return _pid_alive(pid)


def _write_spool_owner(path: Path) -> None:
    try:
        (path / _SPOOL_PID_FILE).write_text(
            str(os.getpid()),
            encoding="utf-8",
        )
    except OSError:
        pass


def _module_spool_dir() -> Path | None:
    global _MODULE_SPOOL_DIR
    with _MODULE_SPOOL_LOCK:
        if _MODULE_SPOOL_DIR is not None and _MODULE_SPOOL_DIR.is_dir():
            return _MODULE_SPOOL_DIR
        _cleanup_stale_spool_dirs()
        try:
            _MODULE_SPOOL_DIR = Path(
                tempfile.mkdtemp(prefix=f"{_SPOOL_DIR_PREFIX}{os.getpid()}-")
            )
            _write_spool_owner(_MODULE_SPOOL_DIR)
        except OSError:
            _MODULE_SPOOL_DIR = None
        return _MODULE_SPOOL_DIR


def _unlink_spool_paths(paths: tuple[str, ...]) -> None:
    for path in paths:
        if not path:
            continue
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass


class _OutputSpoolManager:
    def __init__(self) -> None:
        _cleanup_stale_spool_dirs()
        try:
            self._dir: Path | None = Path(
                tempfile.mkdtemp(prefix=f"{_SPOOL_DIR_PREFIX}{os.getpid()}-")
            )
            _write_spool_owner(self._dir)
        except OSError:
            self._dir = None
        self._current_paths: tuple[str, ...] = ()

    @property
    def directory(self) -> Path | None:
        return self._dir

    def start_iteration(self) -> None:
        _unlink_spool_paths(self._current_paths)
        self._current_paths = ()

    def record(self, stdout_path: str, stderr_path: str) -> None:
        self._current_paths = (stdout_path, stderr_path)

    def close(self) -> None:
        _unlink_spool_paths(self._current_paths)
        self._current_paths = ()
        if self._dir is not None:
            shutil.rmtree(self._dir, ignore_errors=True)


def _run_command(
    command: str,
    cwd: str,
    timeout_s: float,
    *,
    spool_manager: _OutputSpoolManager | None = None,
) -> ObservationDigest:
    """Execute the user's command, capture an ObservationDigest.

    ``timeout_s`` caps wall clock for this single iteration. A timeout
    sets ``timed_out=True`` in the digest so the verification plan can
    still run (e.g. ``exit_code`` check still fails with non-zero
    rendered as -1, and the LLM sees the timeout flag in the failure).

    CRITICAL: stdout / stderr are drained CONCURRENTLY via background
    threads. A naive ``subprocess.run(..., capture_output=True)`` puts
    both streams on PIPEs but only reads them AFTER the child exits —
    so any long-running command that writes more than the OS pipe
    buffer (typically 64 KB on macOS / Linux) ends up BLOCKED on
    ``write()`` mid-execution, and the watch wall-clock fires before
    the command can finish. We saw this in production: nightly_cycle
    runs ~3 h producing megabytes of logger output and was hanging at
    the very first ``logger.info`` past 64 KB until ``timeout_s``
    killed it. Draining concurrently keeps the child's writes
    non-blocking for the full run.
    """
    cwd_str = cwd or os.getcwd()
    started = time.monotonic()
    timed_out = False

    try:
        # start_new_session puts the shell AND its descendants in a new
        # process group, so a timeout can kill the whole tree. Without
        # it, ``proc.kill()`` only kills /bin/sh and the real workload
        # (e.g. a multi-hour nightly_cycle) keeps running orphaned.
        popen_args: str | list[str] = command
        use_shell = True
        start_new_session = os.name == "posix"
        if os.name == "nt" and "\n" in command:
            popen_args = ["powershell", "-NoProfile", "-Command", command]
            use_shell = False
            start_new_session = False
        proc = subprocess.Popen(
            popen_args,
            shell=use_shell,
            cwd=cwd_str,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=start_new_session,
        )
    except OSError as exc:
        duration_s = time.monotonic() - started
        return ObservationDigest(
            command=command,
            exit_code=-1,
            stdout_tail="",
            stderr_tail=f"[watch] failed to launch command: {exc}",
            duration_s=duration_s,
            cwd=cwd_str,
            files_touched=tuple(_detect_files_touched(cwd_str)),
            timed_out=False,
        )

    stdout_buf = bytearray()
    stderr_buf = bytearray()

    def _drain(stream: Any, buf: bytearray) -> None:
        try:
            while True:
                chunk = stream.read(4096)
                if not chunk:
                    break
                buf.extend(chunk)
        except (OSError, ValueError):
            pass

    stdout_thread = threading.Thread(
        target=_drain, args=(proc.stdout, stdout_buf), daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_drain, args=(proc.stderr, stderr_buf), daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    try:
        exit_code = proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_process_tree(proc)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        exit_code = -1

    # Let the drainer threads finish reading whatever's still in the
    # pipes after the child exited (or was killed). Bounded join so a
    # stuck fd doesn't hang us indefinitely.
    stdout_thread.join(timeout=5.0)
    stderr_thread.join(timeout=5.0)

    stdout = stdout_buf.decode("utf-8", "replace")
    stderr = stderr_buf.decode("utf-8", "replace")
    if timed_out:
        stderr = stderr + f"\n[watch] command timed out after {timeout_s}s"
    stdout_path = _write_output_spool("stdout", stdout, spool_manager)
    stderr_path = _write_output_spool("stderr", stderr, spool_manager)
    if spool_manager is not None:
        spool_manager.record(stdout_path, stderr_path)

    duration_s = time.monotonic() - started
    return ObservationDigest(
        command=command,
        exit_code=exit_code,
        stdout_tail=stdout[-_OUTPUT_TAIL_BYTES:],
        stderr_tail=stderr[-_OUTPUT_TAIL_BYTES:],
        duration_s=duration_s,
        cwd=cwd_str,
        files_touched=tuple(_detect_files_touched(cwd_str)),
        timed_out=timed_out,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )


def _write_output_spool(
    stream_name: str,
    text: str,
    spool_manager: _OutputSpoolManager | None = None,
) -> str:
    try:
        if spool_manager is not None:
            spool_dir = spool_manager.directory
        else:
            spool_dir = _module_spool_dir()
        if spool_dir is None:
            return ""
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            errors="replace",
            delete=False,
            prefix=f"{stream_name}-",
            suffix=".log",
            dir=spool_dir,
        ) as fh:
            fh.write(text)
            path = fh.name
        if spool_manager is None:
            _cleanup_module_spool_files(spool_dir)
        return path
    except OSError:
        return ""


def _cleanup_module_spool_files(
    spool_dir: Path,
    *,
    max_files: int = _MODULE_SPOOL_FILE_LIMIT,
) -> None:
    if max_files <= 0:
        return
    try:
        files = [
            path for path in spool_dir.iterdir()
            if path.is_file() and path.name != _SPOOL_PID_FILE
        ]
    except OSError:
        return
    if len(files) <= max_files:
        return

    def _mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    for path in sorted(files, key=_mtime)[: len(files) - max_files]:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _detect_files_touched(cwd: str) -> list[str]:
    """Use ``git status --porcelain`` to detect files modified in the
    working tree. Best-effort: if git isn't available or cwd is not a
    repo, returns ``[]`` rather than blowing up."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd or None,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return []
    if result.returncode != 0:
        return []
    out: list[str] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        out.append(line[3:].strip())
    return out


# --------------------------------------------------------------------------- #
# Fix-scope guard tool wrapper
# --------------------------------------------------------------------------- #

class _FixScopeGuard:
    """Composable guard that wraps a cc tool's ``execute`` method.

    The wrapper looks at ``arguments["file_path"]`` and rejects the
    call with ``success=False, error_code=FIX_SCOPE_VIOLATION`` when
    the path doesn't match any glob in ``fix_scope``. Otherwise it
    forwards to the underlying tool's ``execute`` unchanged.

    Implemented as a wrapper *instance* (not a subclass) so we don't
    have to know the exact concrete tool class shape — we replace
    the cc ``BaseTool`` in the registry's ``_tools`` dict with this
    object, which forwards every other attribute back through
    ``__getattr__``. The orchestrator only depends on duck-typed
    ``spec`` / ``execute`` / ``is_concurrency_safe`` / ``validate_input``
    / ``check_permissions``, all of which keep working.
    """

    def __init__(
        self,
        wrapped: Any,
        *,
        fix_scope: list[str],
        cwd: str,
    ) -> None:
        self._wrapped = wrapped
        self._fix_scope = list(fix_scope)
        self._cwd = cwd

    # cc duck-typed attributes — delegate everything except execute.
    @property
    def spec(self) -> Any:
        return self._wrapped.spec

    def is_enabled(self, ctx: Any) -> bool:
        return self._wrapped.is_enabled(ctx)

    def is_concurrency_safe(self, arguments: dict[str, Any]) -> bool:
        # Mutating tools are not concurrency-safe; the wrapped tool
        # already reports False, but we mirror that explicitly so
        # behavior is obvious from this class.
        return False

    def validate_input(self, arguments: dict[str, Any]) -> Any:
        return self._wrapped.validate_input(arguments)

    def check_permissions(self, ctx: Any, arguments: dict[str, Any]) -> Any:
        return self._wrapped.check_permissions(ctx, arguments)

    def to_model_schema(self) -> dict[str, Any]:
        return self._wrapped.to_model_schema()

    def __getattr__(self, name: str) -> Any:
        # Fallback for anything else (e.g. ``build_patch_preview``).
        return getattr(self._wrapped, name)

    async def execute(self, tool_call: Any, ctx: Any) -> Any:
        from core.cc.tools.base import ToolResult

        arguments = getattr(tool_call, "arguments", None) or {}
        path = str(arguments.get("file_path") or "")
        if not path:
            return ToolResult(
                tool_use_id=getattr(tool_call, "tool_use_id", ""),
                tool_name=getattr(tool_call, "tool_name", ""),
                success=False,
                content="file_path is required",
                error_code="FIX_SCOPE_VIOLATION",
            )
        if not _path_matches_any_glob(path, self._fix_scope, self._cwd):
            return ToolResult(
                tool_use_id=getattr(tool_call, "tool_use_id", ""),
                tool_name=getattr(tool_call, "tool_name", ""),
                success=False,
                content=(
                    f"path {path!r} is outside fix_scope. "
                    f"You may only edit files matching: {self._fix_scope}. "
                    "Pick a different file or report in your final message "
                    "that the fix requires touching out-of-scope code."
                ),
                error_code="FIX_SCOPE_VIOLATION",
            )
        return await self._wrapped.execute(tool_call, ctx)


def _install_fix_scope_guard(
    engine: Any,
    *,
    fix_scope: list[str],
    cwd: str,
) -> None:
    """Wrap ``file_edit`` / ``file_write`` in the engine's registry
    with ``_FixScopeGuard``. After this call, every file_edit /
    file_write the LLM emits is path-checked before the underlying
    tool runs. Idempotent — if the tool is already guarded (instance
    of ``_FixScopeGuard``), we update the scope on it in place
    instead of double-wrapping."""
    orchestrator = getattr(engine, "tool_orchestrator", None)
    registry = getattr(orchestrator, "registry", None)
    if registry is None:
        return
    tools = getattr(registry, "_tools", None)
    if not isinstance(tools, dict):
        return
    for name in ("file_edit", "file_write"):
        existing = tools.get(name)
        if existing is None:
            continue
        if isinstance(existing, _FixScopeGuard):
            existing._fix_scope = list(fix_scope)
            existing._cwd = cwd
            continue
        tools[name] = _FixScopeGuard(existing, fix_scope=fix_scope, cwd=cwd)


def _remove_writer_tools(engine: Any) -> None:
    """Strip every writer / shell tool from the engine. Used by Phase 0
    (analyzer is read-only). The shared ``restrict_tool_registry``
    already handles this — this helper is kept for an alternative
    "strip everything not file_edit / file_write / read-only" path
    used by the fixer (see ``_strip_to_fixer_tools``)."""
    restrict_tool_registry(engine)


def _strip_to_fixer_tools(engine: Any) -> None:
    """Restrict the engine's registry to the fixer's allowed set:
    read-only enumeration tools PLUS ``file_edit`` and ``file_write``
    (which get wrapped with the fix-scope guard separately). Drops
    ``shell``, ``powershell``, ``todo_write``, ``plan_artifact_write``,
    ``spec_artifact_write``, etc. — anything that could escape the
    fix_scope sandbox by running commands."""
    orchestrator = getattr(engine, "tool_orchestrator", None)
    registry = getattr(orchestrator, "registry", None)
    if registry is None:
        return
    tools = getattr(registry, "_tools", None)
    if not isinstance(tools, dict):
        return
    # We keep the read-only whitelist + the two editing tools.
    allowed = {"file_edit", "file_write"}
    # Plus everything restrict_tool_registry would keep (read-only):
    from ..agents.read_only_runner import (
        DEFAULT_READ_ONLY_WHITELIST,
        _is_read_only,
    )
    keep_names = set(DEFAULT_READ_ONLY_WHITELIST) | allowed
    to_remove: list[str] = []
    for name, tool in tools.items():
        if name in keep_names:
            continue
        if _is_read_only(tool):
            continue
        to_remove.append(name)
    for name in to_remove:
        del tools[name]


# --------------------------------------------------------------------------- #
# Plan rendering for fixer prompt + artifact serialization
# --------------------------------------------------------------------------- #

def _format_failures_for_prompt(failures: list[CheckResult]) -> str:
    """Render the failure list as a markdown block for the fixer
    prompt. Each failure includes ``kind``, ``reason``, ``observed``,
    ``expected`` so the LLM has everything it needs to localize the
    bug without re-reading the plan."""
    if not failures:
        return ""
    lines: list[str] = ["## Verification failures"]
    for i, f in enumerate(failures, 1):
        lines.append(f"### Failure {i}: kind={f.kind}")
        lines.append(f"- reason: {f.reason}")
        if f.observed is not None:
            obs_str = json.dumps(f.observed, ensure_ascii=False, default=str)
            if len(obs_str) > 1200:
                obs_str = obs_str[:1200] + "...[truncated]"
            lines.append(f"- observed: {obs_str}")
        if f.expected is not None:
            exp_str = json.dumps(f.expected, ensure_ascii=False, default=str)
            if len(exp_str) > 600:
                exp_str = exp_str[:600] + "...[truncated]"
            lines.append(f"- expected: {exp_str}")
    return "\n".join(lines)


def _format_digest_for_prompt(digest: ObservationDigest) -> str:
    parts: list[str] = ["## Last command run"]
    parts.append(f"- command: `{digest.command}`")
    parts.append(f"- exit_code: {digest.exit_code}")
    parts.append(f"- duration_s: {digest.duration_s:.2f}")
    if digest.timed_out:
        parts.append("- TIMED OUT (process killed)")
    if digest.files_touched:
        parts.append(
            f"- files_touched (uncommitted): {list(digest.files_touched)}"
        )
    if digest.stdout_tail.strip():
        parts.append("\n### stdout tail")
        parts.append("```\n" + digest.stdout_tail + "\n```")
    if digest.stderr_tail.strip():
        parts.append("\n### stderr tail")
        parts.append("```\n" + digest.stderr_tail + "\n```")
    if digest.commit_note.strip():
        parts.append("\n### previous commit failure")
        parts.append("```\n" + digest.commit_note + "\n```")
    return "\n".join(parts)


def _format_plan_for_prompt(plan: dict[str, Any]) -> str:
    return (
        "## VerificationPlan (READ-ONLY — you cannot modify this)\n"
        f"summary: {plan.get('summary', '')!r}\n\n"
        "```json\n"
        + json.dumps({"checks": plan.get("checks") or []},
                     indent=2, ensure_ascii=False)
        + "\n```"
    )


# --------------------------------------------------------------------------- #
# Commit helper
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class _CommitResult:
    ok: bool
    message: str
    stderr: str = ""

    def as_digest_note(self) -> str:
        if self.ok:
            return f"[watch] previous fix committed: {self.message}"
        return (
            "[watch] previous fix could NOT be committed (pre-commit hook "
            "likely failed). The hook output is below — fix the underlying "
            f"issue rather than asking to bypass:\n{self.stderr.strip()}"
        )


def _commit_fix(
    cwd: str,
    message: str,
    fix_scope: list[str] | None = None,
) -> _CommitResult:
    """Commit fixer-touched files. NEVER passes ``--no-verify`` —
    pre-commit hook failures surface as an ``ok=False`` result whose
    stderr is fed into the next digest.

    When ``fix_scope`` is supplied, this scopes the ``git add`` to those
    globs (using git's ``:(glob)`` pathspec magic) AND unstages anything
    pre-staged outside the scope first. Without scoping, ``git add -u``
    would sweep in unrelated tracked-file edits left in the working tree
    by other agents / the user editing in parallel — the symptom that
    produced the bogus ``watch fix iter N`` commits carrying unrelated
    diffs.
    """
    if fix_scope:
        # Step 1: unstage everything currently staged so we don't
        # commit pre-staged out-of-scope changes alongside the fixer's
        # work. ``git reset HEAD -- .`` resets the index of every path
        # back to HEAD; working-tree contents are untouched. Safe to
        # call even on a fresh repo (will no-op).
        try:
            subprocess.run(
                ["git", "reset", "HEAD", "--", "."],
                cwd=cwd or None,
                capture_output=True, text=True, timeout=30,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            # Best-effort: if reset fails (no HEAD on a fresh repo, etc.),
            # the path-scoped add below is still the primary safeguard.
            pass
        # Step 2: stage only files under fix_scope. ``:(glob)`` pathspec
        # magic gives the same ``**`` semantics as Python's fnmatch, so
        # ``core/deepstack-agent/stock_rec_v3/**/*.py`` matches recursively.
        # Use the same globs the fixer guard enforced — anything the
        # fixer was *allowed* to write is fair game to commit.
        pathspecs = [f":(glob){g}" for g in fix_scope]
        add_cmd = ["git", "add", "--"] + pathspecs
    else:
        # No fix_scope: legacy behaviour. Caller has explicitly disabled
        # fixing, so commit_each_fix shouldn't even fire — but if it
        # does, we keep the original semantics.
        add_cmd = ["git", "add", "-u"]
    try:
        add_result = subprocess.run(
            add_cmd,
            cwd=cwd or None,
            capture_output=True, text=True, timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        return _CommitResult(ok=False, message=message,
                              stderr=f"git add failed: {exc}")
    if add_result.returncode != 0:
        return _CommitResult(
            ok=False, message=message,
            stderr=f"git add exited {add_result.returncode}: "
                   f"{add_result.stderr.strip()}",
        )
    try:
        commit_result = subprocess.run(
            ["git", "commit", "-m", message],
            cwd=cwd or None,
            capture_output=True, text=True, timeout=60,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        return _CommitResult(ok=False, message=message,
                              stderr=f"git commit failed: {exc}")
    if commit_result.returncode != 0:
        return _CommitResult(
            ok=False, message=message,
            stderr=(
                (commit_result.stdout or "") + "\n" +
                (commit_result.stderr or "")
            ).strip(),
        )
    return _CommitResult(ok=True, message=message,
                          stderr=commit_result.stdout.strip())


# --------------------------------------------------------------------------- #
# WatchModeRunner
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class WatchModeRunner(ModeRunner):
    """Run-verify-fix loop driven by an LLM-derived VerificationPlan.

    Designed to be plugged into the v5 runtime the same way as
    ``DocModeRunner`` / ``AskModeRunner``: the orchestrator builds
    one with ``llm``, ``cwd``, ``cc_config``, ``llm_provider`` and
    invokes ``run(invocation)`` per node.

    Invocation parameters are read from ``invocation.metadata``:

    * ``command`` (str, required)
    * ``scope`` (str | list[str], required)
    * ``caller_hints`` (str, optional)
    * ``fix_scope`` (str | list[str], optional — empty disables fixing)
    * ``max_iterations`` (int, default 5)
    * ``max_wallclock_s`` (float, default 1800)
    * ``analysis_timeout_s`` (float, default 600)
    * ``per_iteration_timeout_s`` (float, default 600)
    * ``commit_each_fix`` (bool, default True)
    * ``verification_plan_artifact`` (str path, optional)
    * ``wallclock_budget_s`` (float, optional) — parent-allocated total
      wall-clock budget. When set, ``analysis_timeout_s`` and
      ``max_wallclock_s`` default to 30 % / 70 % of this value, and
      the sum is hard-capped so an explicit override that overshoots
      gets scaled back proportionally.
    * ``nested_invocation`` (bool, optional) — set by parent runners
      that spawn watch as a child node. Currently a single safety
      override: when true, ``commit_each_fix`` is forced ``False``
      regardless of caller value, so a nested watch never pollutes
      the parent's working tree. (If the parent really wants commits
      it should NOT mark the call nested — the flag is the parent's
      own "don't commit on my behalf" declaration.)

    The cc QueryEngine watch builds for Phase 0 and for each fixer
    iteration is ALWAYS fresh and ALWAYS post-filtered by either
    ``restrict_tool_registry`` (Phase 0 — read-only) or
    ``_strip_to_fixer_tools`` + ``_install_fix_scope_guard`` (fixer).
    A parent invocation's registry is structurally inaccessible —
    watch does not inherit it.

    SubagentResult.extras shape (callers / parents can introspect):

    * ``status``: ``"ok"`` (every check passed) or ``"failed"``.
    * ``phase``: ``"analyze"`` (Phase 0 aborted) or ``"loop"``.
    * ``via``: short reason code for which terminal branch fired.
    * ``verification_plan``: the full plan dict (so the parent can
      reuse it instead of re-deriving).
    * ``iterations``: per-iteration log (exit code, duration, failures,
      optional commit result).
    * ``result_summary``: ``{plan_summary, last_digest, failed_checks}``
      — a single dict aimed at parents who only want the headline.
    * ``nested``: echoes ``invocation.metadata["nested_invocation"]``.
    * ``budget``: ``{wallclock_budget_s, analysis_timeout_s,
      max_wallclock_s}`` — the values actually used (after split /
      scale-down).
    * ``effective_commit_each_fix``: the value used at runtime, after
      the nested-invocation override applied.
    """
    llm: LLMCallable
    cwd: str
    cc_config: Any | None = None
    llm_provider: Any | None = None
    language: str = "en"
    mode_name: str = "watch"

    # Tunables
    # Raised from 2 → 3 because deepseek-reasoner has a habit of ending
    # its turn with a preamble like "Let me emit the JSON." without
    # actually emitting any JSON. The first 2 attempts get progressively
    # more pointed retry feedback (including the LLM's own previous
    # preamble verbatim); a 3rd attempt is the cushion that lets that
    # feedback converge.
    ANALYZER_MAX_ATTEMPTS: int = 3
    ANALYZER_MAX_ROUNDS: int = 36
    ANALYZER_WALL_CLOCK_TIMEOUT_S: float = 600.0
    FIXER_MAX_ROUNDS: int = 30
    FIXER_WALL_CLOCK_TIMEOUT_S: float = 600.0
    DEFAULT_MAX_ITERATIONS: int = 5
    DEFAULT_MAX_WALLCLOCK_S: float = 1800.0
    DEFAULT_PER_ITERATION_TIMEOUT_S: float = _DEFAULT_PER_ITERATION_TIMEOUT_S

    # ------------------------------------------------------------------ #
    # Entry point
    # ------------------------------------------------------------------ #

    def run(self, invocation: SubagentInvocation) -> SubagentResult:
        spool_manager = _OutputSpoolManager()
        try:
            return self._run_impl(invocation, spool_manager)
        finally:
            spool_manager.close()

    def _run_impl(
        self,
        invocation: SubagentInvocation,
        spool_manager: _OutputSpoolManager,
    ) -> SubagentResult:
        md = invocation.metadata or {}
        command = str(md.get("command") or "").strip()
        scope = _coerce_scope(md.get("scope"))
        caller_hints = str(md.get("caller_hints") or "").strip()
        fix_scope = _coerce_fix_scope(md.get("fix_scope"))
        max_iterations = int(md.get("max_iterations") or self.DEFAULT_MAX_ITERATIONS)
        per_iteration_timeout_s = float(
            md.get("per_iteration_timeout_s")
            or self.DEFAULT_PER_ITERATION_TIMEOUT_S
        )
        plan_artifact = md.get("verification_plan_artifact")
        nested = bool(md.get("nested_invocation"))
        commit_each_fix = self._resolve_commit_each_fix(md)
        analysis_timeout_s, max_wallclock_s, wallclock_budget_s = (
            self._resolve_budgets(md)
        )
        budget_extras = {
            "wallclock_budget_s": wallclock_budget_s,
            "analysis_timeout_s": analysis_timeout_s,
            "max_wallclock_s": max_wallclock_s,
        }

        if not command:
            return SubagentResult(
                final_text="watch mode: missing `command` parameter",
                subtasks=[],
                extras={
                    "status": "failed",
                    "via": "watch_input_error",
                    "nested": nested,
                },
            )
        if not scope:
            return SubagentResult(
                final_text="watch mode: missing `scope` parameter (no code "
                           "to analyze for acceptance criteria)",
                subtasks=[],
                extras={
                    "status": "failed",
                    "via": "watch_input_error",
                    "nested": nested,
                },
            )

        # ── Phase 0 ──
        try:
            plan = self._run_phase_0(
                command=command,
                scope=scope,
                caller_hints=caller_hints,
                analysis_timeout_s=analysis_timeout_s,
            )
        except _PhaseZeroError as exc:
            return SubagentResult(
                final_text=str(exc),
                subtasks=[],
                extras=self._make_extras(
                    status="failed",
                    phase="analyze",
                    via="watch_phase0_failed",
                    plan=None,
                    iterations=[],
                    last_digest=None,
                    failed_checks=[],
                    nested=nested,
                    budget=budget_extras,
                    commit_each_fix=commit_each_fix,
                    extra={"reason": exc.reason_code},
                ),
            )
        except Exception as exc:  # noqa: BLE001
            return SubagentResult(
                final_text=f"watch mode: Phase 0 failed: {type(exc).__name__}: {exc}",
                subtasks=[],
                extras=self._make_extras(
                    status="failed",
                    phase="analyze",
                    via="watch_phase0_failed",
                    plan=None,
                    iterations=[],
                    last_digest=None,
                    failed_checks=[],
                    nested=nested,
                    budget=budget_extras,
                    commit_each_fix=commit_each_fix,
                    extra={"reason": "phase0_exception"},
                ),
            )

        # Persist artifact (audit trail) BEFORE the loop, so users can
        # inspect what plan was derived even if the loop crashes.
        if plan_artifact:
            self._persist_plan_artifact(str(plan_artifact), plan)
        run_started_at = time.monotonic()

        # ── Phase 1 ──
        iteration_log: list[dict[str, Any]] = []
        prior_commit_failure: _CommitResult | None = None
        last_failures: list[CheckResult] = []
        last_digest: ObservationDigest | None = None

        for iter_idx in range(max_iterations):
            spool_manager.start_iteration()
            if (time.monotonic() - run_started_at) >= max_wallclock_s:
                return SubagentResult(
                    final_text=(
                        f"watch mode: wall-clock budget ({max_wallclock_s}s) "
                        f"exhausted before iteration {iter_idx + 1}"
                    ),
                    subtasks=[],
                    extras=self._make_extras(
                        status="failed",
                        phase="loop",
                        via="watch_wallclock_exhausted",
                        plan=plan,
                        iterations=iteration_log,
                        last_digest=last_digest,
                        failed_checks=last_failures,
                        nested=nested,
                        budget=budget_extras,
                        commit_each_fix=commit_each_fix,
                    ),
                )

            iter_timeout = min(
                per_iteration_timeout_s,
                max(1.0, max_wallclock_s - (time.monotonic() - run_started_at)),
            )
            digest = _run_command(
                command,
                self.cwd,
                iter_timeout,
                spool_manager=spool_manager,
            )
            if prior_commit_failure is not None:
                # Surface hook failures to the next fixer prompt without
                # polluting stderr tails or serialized digest data.
                digest = ObservationDigest(
                    command=digest.command,
                    exit_code=digest.exit_code,
                    stdout_tail=digest.stdout_tail,
                    stderr_tail=digest.stderr_tail,
                    duration_s=digest.duration_s,
                    cwd=digest.cwd,
                    files_touched=digest.files_touched,
                    timed_out=digest.timed_out,
                    stdout_path=digest.stdout_path,
                    stderr_path=digest.stderr_path,
                    commit_note=prior_commit_failure.as_digest_note(),
                )
                prior_commit_failure = None
            last_digest = digest

            failures = run_verification_plan(plan, digest)
            last_failures = failures
            iteration_log.append({
                "iteration": iter_idx + 1,
                "exit_code": digest.exit_code,
                "timed_out": digest.timed_out,
                "duration_s": round(digest.duration_s, 3),
                "failures": [f.as_dict() for f in failures],
            })

            if not failures:
                return SubagentResult(
                    final_text=(
                        f"watch mode: all {len(plan.get('checks') or [])} "
                        f"verification checks passed after {iter_idx + 1} "
                        "iteration(s)."
                    ),
                    subtasks=[],
                    extras=self._make_extras(
                        status="ok",
                        phase="loop",
                        via="watch_ok",
                        plan=plan,
                        iterations=iteration_log,
                        last_digest=digest,
                        failed_checks=[],
                        nested=nested,
                        budget=budget_extras,
                        commit_each_fix=commit_each_fix,
                    ),
                )

            # No fix scope OR last iteration → bail out cleanly.
            if not fix_scope or iter_idx == max_iterations - 1:
                return SubagentResult(
                    final_text=self._build_failed_summary(
                        plan=plan,
                        failures=failures,
                        iter_count=iter_idx + 1,
                        fix_scope_empty=not fix_scope,
                        max_iterations=max_iterations,
                    ),
                    subtasks=[],
                    extras=self._make_extras(
                        status="failed",
                        phase="loop",
                        via="watch_verify_failed",
                        plan=plan,
                        iterations=iteration_log,
                        last_digest=digest,
                        failed_checks=failures,
                        nested=nested,
                        budget=budget_extras,
                        commit_each_fix=commit_each_fix,
                    ),
                )

            # Run fixer (cc QueryEngine with fix_scope-guarded tools).
            try:
                self._run_fixer(
                    plan=plan,
                    digest=digest,
                    failures=failures,
                    fix_scope=fix_scope,
                )
            except _FixerError as exc:
                logger.warning(
                    "watch fixer iteration %d failed: %s", iter_idx + 1, exc,
                )
                iteration_log[-1]["fixer_error"] = str(exc)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "watch fixer iteration %d raised: %s",
                    iter_idx + 1,
                    exc,
                    exc_info=True,
                )
                iteration_log[-1]["fixer_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
                # Don't abort the loop on a fixer error — next iteration
                # will rerun the command and verify again. If fixer
                # consistently fails the iteration cap will catch it.

            # Commit (if requested) so each fix becomes one commit.
            # Pass fix_scope so the commit is path-scoped to what the
            # fixer was *allowed* to write — otherwise unrelated dirty
            # files in the working tree (left by other agents / the
            # user editing in parallel) get swept into the commit with
            # a misleading "watch fix iter N" message.
            if commit_each_fix:
                commit = _commit_fix(
                    self.cwd,
                    f"watch fix iter {iter_idx + 1}: "
                    f"{len(failures)} check(s) failing",
                    fix_scope=fix_scope,
                )
                iteration_log[-1]["commit"] = {
                    "ok": commit.ok,
                    "stderr": commit.stderr,
                }
                if not commit.ok:
                    prior_commit_failure = commit

        # Shouldn't normally reach here (loop returns on every path),
        # but defend against an off-by-one in the future.
        return SubagentResult(
            final_text=self._build_failed_summary(
                plan=plan,
                failures=last_failures,
                iter_count=max_iterations,
                fix_scope_empty=not fix_scope,
                max_iterations=max_iterations,
            ),
            subtasks=[],
            extras=self._make_extras(
                status="failed",
                phase="loop",
                via="watch_iter_cap",
                plan=plan,
                iterations=iteration_log,
                last_digest=last_digest,
                failed_checks=last_failures,
                nested=nested,
                budget=budget_extras,
                commit_each_fix=commit_each_fix,
            ),
        )

    # ------------------------------------------------------------------ #
    # Parameter resolution helpers
    # ------------------------------------------------------------------ #

    def _resolve_commit_each_fix(self, md: dict[str, Any]) -> bool:
        """When the parent declares ``nested_invocation=True``, force
        ``commit_each_fix=False`` regardless of caller value. The
        ``nested_invocation`` flag is the parent's explicit "do not
        touch my working tree" declaration; if the parent really wants
        commits it should NOT mark the call nested. Documented in the
        class docstring."""
        if bool(md.get("nested_invocation")):
            return False
        return bool(md.get("commit_each_fix", True))

    def _resolve_budgets(
        self, md: dict[str, Any],
    ) -> tuple[float, float, float | None]:
        """Compute ``(analysis_timeout_s, max_wallclock_s, wallclock_budget_s)``
        from the invocation metadata.

        * If ``wallclock_budget_s`` is missing, fall back to the runner
          defaults (or per-field overrides if present).
        * If ``wallclock_budget_s`` is set, the per-field defaults
          become a 30 % / 70 % split (analysis / loop).
        * Explicit per-field overrides are respected, BUT the sum
          ``analysis_timeout_s + max_wallclock_s`` cannot exceed
          ``wallclock_budget_s``. An over-spec is scaled down
          proportionally so the parent's budget is always honored.
        """
        raw_budget = md.get("wallclock_budget_s")
        wallclock_budget_s: float | None
        if raw_budget is None:
            wallclock_budget_s = None
        else:
            try:
                wallclock_budget_s = float(raw_budget)
                if wallclock_budget_s <= 0:
                    wallclock_budget_s = None
            except (TypeError, ValueError):
                wallclock_budget_s = None

        explicit_analysis = md.get("analysis_timeout_s")
        explicit_max_wall = md.get("max_wallclock_s")

        if wallclock_budget_s is not None:
            analysis_timeout_s = (
                float(explicit_analysis)
                if explicit_analysis is not None
                else wallclock_budget_s * 0.30
            )
            max_wallclock_s = (
                float(explicit_max_wall)
                if explicit_max_wall is not None
                else wallclock_budget_s * 0.70
            )
            total = analysis_timeout_s + max_wallclock_s
            if total > wallclock_budget_s and total > 0:
                scale = wallclock_budget_s / total
                analysis_timeout_s *= scale
                max_wallclock_s *= scale
        else:
            analysis_timeout_s = float(
                explicit_analysis or self.ANALYZER_WALL_CLOCK_TIMEOUT_S
            )
            max_wallclock_s = float(
                explicit_max_wall or self.DEFAULT_MAX_WALLCLOCK_S
            )
        return analysis_timeout_s, max_wallclock_s, wallclock_budget_s

    # ------------------------------------------------------------------ #
    # Extras builder
    # ------------------------------------------------------------------ #

    def _make_extras(
        self,
        *,
        status: str,
        phase: str,
        via: str,
        plan: dict[str, Any] | None,
        iterations: list[dict[str, Any]],
        last_digest: ObservationDigest | None,
        failed_checks: list[CheckResult],
        nested: bool,
        budget: dict[str, Any],
        commit_each_fix: bool,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the SubagentResult.extras dict. Single source of truth
        so every return path emits the same shape — parents that
        introspect ``extras["verification_plan"]`` or
        ``extras["result_summary"]`` see them on success AND failure."""
        plan_summary = ""
        if isinstance(plan, dict):
            plan_summary = str(plan.get("summary") or "")
        digest_dict: dict[str, Any] | None = (
            last_digest.to_dict() if last_digest is not None else None
        )
        failed = [f.as_dict() for f in failed_checks]
        extras: dict[str, Any] = {
            "status": status,
            "phase": phase,
            "via": via,
            "verification_plan": plan,
            "iterations": iterations,
            "result_summary": {
                "plan_summary": plan_summary,
                "last_digest": digest_dict,
                "failed_checks": failed,
            },
            "failed_checks": failed,
            "nested": nested,
            "budget": dict(budget),
            "effective_commit_each_fix": commit_each_fix,
        }
        if extra:
            extras.update(extra)
        return extras

    # ------------------------------------------------------------------ #
    # Phase 0
    # ------------------------------------------------------------------ #

    def _run_phase_0(
        self,
        *,
        command: str,
        scope: list[str],
        caller_hints: str,
        analysis_timeout_s: float,
    ) -> dict[str, Any]:
        """Drive a read-only QueryEngine to emit a VerificationPlan.

        Has its own wall-clock budget independent of the Watch loop's
        ``max_wallclock_s``. A stuck analyzer must not eat the budget
        the user reserved for fixing.

        Retries on unparseable / invalid output up to
        ``ANALYZER_MAX_ATTEMPTS``. After that, raises ``_PhaseZeroError``
        which the caller surfaces as a clean failed result without
        entering the loop.
        """
        if self.llm_provider is None or self.cc_config is None:
            # No real engine wired — fall back to a single-shot direct
            # LLM call. This is the testing path; production always
            # has both fields set.
            return self._run_phase_0_lite(
                command=command, scope=scope,
                caller_hints=caller_hints,
            )

        from ..agents.cc_agent import _run_in_fresh_loop
        return _run_in_fresh_loop(self._run_phase_0_async(
            command=command,
            scope=scope,
            caller_hints=caller_hints,
            analysis_timeout_s=analysis_timeout_s,
        ))

    def _run_phase_0_lite(
        self,
        *,
        command: str,
        scope: list[str],
        caller_hints: str,
    ) -> dict[str, Any]:
        """No-tools fallback for Phase 0: a single LLM call. Used in
        tests and in any environment without a cc QueryEngine. The
        LLM can only base the plan on the user-provided context, not
        on actually reading the code."""
        system = _load_analyzer_system(self.language)
        user = self._build_analyzer_user_prompt(
            command=command, scope=scope, caller_hints=caller_hints,
            retry_feedback="",
        )
        for attempt in range(self.ANALYZER_MAX_ATTEMPTS):
            response = text_of(self.llm(
                system=system, user=user, purpose="watch_analyze_lite",
            ))
            parsed = _parse_plan_json(response or "")
            if parsed is not None:
                ok, reason, _kinds = _validate_plan(parsed)
                if ok:
                    return parsed
                user = self._build_analyzer_user_prompt(
                    command=command, scope=scope, caller_hints=caller_hints,
                    retry_feedback=(
                        f"Previous attempt produced an invalid plan: {reason}. "
                        "Emit a NEW plan that fixes this. "
                        f"Supported check kinds: {supported_kinds()}."
                    ),
                )
                continue
            user = self._build_analyzer_user_prompt(
                command=command, scope=scope, caller_hints=caller_hints,
                retry_feedback=(
                    "Previous attempt did not return parseable JSON. "
                    "Emit ONLY a JSON object — no preamble, no fences, "
                    "no commentary."
                ),
            )
        raise _PhaseZeroError(
            "watch Phase 0: analyzer failed to produce a valid plan after "
            f"{self.ANALYZER_MAX_ATTEMPTS} attempts. Last response was "
            "not parseable as a valid VerificationPlan.",
            reason_code="analyzer_no_valid_plan",
        )

    async def _run_phase_0_async(
        self,
        *,
        command: str,
        scope: list[str],
        caller_hints: str,
        analysis_timeout_s: float,
    ) -> dict[str, Any]:
        """Drive a read-only cc QueryEngine to emit the plan. Mirrors
        ``DocModeRunner._invoke_investigator_once`` but with a
        watch-shaped system prompt and the VerificationPlan JSON
        parsing/validation gates.
        """
        from core.cc.runtime import build_default_query_engine

        last_response = ""
        retry_feedback = ""
        per_attempt_timeout = max(
            30.0, analysis_timeout_s / max(1, self.ANALYZER_MAX_ATTEMPTS),
        )

        for attempt in range(self.ANALYZER_MAX_ATTEMPTS):
            engine = build_default_query_engine(
                cwd=self.cwd,
                config=self.cc_config,
                llm_client_provider=self.llm_provider,
            )
            _remove_writer_tools(engine)

            system = (
                _ANALYZER_SYSTEM_ZH if self.language.startswith("zh")
                else _ANALYZER_SYSTEM_EN
            )
            user = self._build_analyzer_user_prompt(
                command=command, scope=scope, caller_hints=caller_hints,
                retry_feedback=retry_feedback,
            )
            framed = f"<system>\n{system}\n</system>\n\n{user}"

            final_text = ""

            async def _drain() -> None:
                nonlocal final_text
                async for event in engine.submit_message(
                    framed,
                    max_tool_rounds=self.ANALYZER_MAX_ROUNDS,
                    purpose="watch_phase0",
                ):
                    msg = getattr(event, "message", None)
                    if msg is None:
                        continue
                    if (
                        getattr(msg, "role", "") == "assistant"
                        and getattr(msg, "kind", "") == "assistant_text"
                    ):
                        final_text = str(getattr(msg, "content", ""))

            timed_out = False
            try:
                await asyncio.wait_for(
                    _drain(), timeout=per_attempt_timeout,
                )
            except asyncio.TimeoutError:
                timed_out = True
                logger.warning(
                    "watch Phase 0: analyzer attempt %d timed out after "
                    "%.0fs", attempt + 1, per_attempt_timeout,
                )
            finally:
                engine.close()

            last_response = final_text
            parsed = _parse_plan_json(final_text)
            if parsed is None:
                # Differentiate root causes so attempt 2 actually changes
                # behaviour. Without this, "ran out of tool budget" and
                # "wrote prose instead of JSON" got the same generic
                # "emit valid JSON" nudge, which doesn't help the
                # analyzer cut its tool usage.
                hit_round_cap = (
                    "Tool round limit" in (final_text or "")
                    or "tool round limit" in (final_text or "")
                )
                if timed_out or not (final_text or "").strip():
                    retry_feedback = (
                        "Previous attempt did NOT finish — it ran out of "
                        f"wall-clock time ({per_attempt_timeout:.0f}s). "
                        "Be DRASTICALLY faster this round: read at most "
                        "3-5 small files (the entry point + tests + "
                        "README), then immediately emit the JSON plan. "
                        "Do NOT enumerate or grep the whole repo. Emit "
                        "ONLY a JSON object — no preamble, no fences."
                    )
                elif hit_round_cap:
                    retry_feedback = (
                        "Previous attempt hit the tool-call budget cap "
                        "before emitting JSON. STOP making tool calls. "
                        "Compose the VerificationPlan JSON from what you "
                        "already know (the command name, caller_hints, "
                        "and whatever you already read). Emit JSON in "
                        "the very FIRST assistant turn this round — do "
                        "NOT call file_read / glob / grep again. The "
                        "plan can be coarse; coarse is better than "
                        "missing."
                    )
                else:
                    # Quote the LLM's prior response verbatim so it can
                    # see exactly the failure mode (e.g. "Let me emit
                    # the JSON." with no JSON following). The generic
                    # "emit valid JSON" feedback was insufficient when
                    # the LLM had already convinced itself it was done
                    # and stopped mid-promise — it needed to see its
                    # own preamble to recognize the mistake.
                    preview = (final_text or "").strip()
                    if len(preview) > 600:
                        preview = preview[:300] + " […truncated…] " + preview[-300:]
                    looks_premature = bool(
                        re.search(
                            r"\b(?:let me|i['’]ll|i will|now i|i'?ll now|"
                            r"about to|going to|i can now|i am ready to)\b",
                            preview.lower(),
                        )
                    ) and "{" not in preview
                    if looks_premature:
                        retry_feedback = (
                            "Previous attempt ENDED MID-PROMISE: you "
                            "said you were about to emit the JSON, but "
                            "no JSON object actually followed. The "
                            "engine returned just this assistant turn:\n"
                            f"<<<\n{preview}\n>>>\n"
                            "That preamble IS the failure mode. The "
                            "next assistant turn must consist of "
                            "NOTHING BUT the raw JSON object — start "
                            "with `{` and end with `}`, no commentary "
                            "before or after, no ``` fences, no "
                            "\"Let me\" / \"I'll\" / \"Here is\" "
                            "prefix. The JSON itself IS the entire "
                            "message."
                        )
                    else:
                        retry_feedback = (
                            "Previous attempt did not return parseable "
                            "JSON. The engine returned just this "
                            f"assistant text:\n<<<\n{preview}\n>>>\n"
                            "Emit ONLY a JSON object — start with `{`, "
                            "end with `}`, no preamble, no fences, no "
                            "commentary. The JSON itself IS the entire "
                            "message."
                        )
                continue
            ok, reason, _kinds = _validate_plan(parsed)
            if ok:
                return parsed
            retry_feedback = (
                f"Previous plan was invalid: {reason}. Supported check "
                f"kinds: {supported_kinds()}. Emit a NEW valid plan now."
            )

        raise _PhaseZeroError(
            "watch Phase 0: analyzer failed to produce a valid plan after "
            f"{self.ANALYZER_MAX_ATTEMPTS} attempts. "
            f"Last response preview: {(last_response or '').strip()[:240]!r}",
            reason_code="analyzer_no_valid_plan",
        )

    def _build_analyzer_user_prompt(
        self,
        *,
        command: str,
        scope: list[str],
        caller_hints: str,
        retry_feedback: str,
    ) -> str:
        parts: list[str] = []
        if retry_feedback:
            parts.append(f"## Retry feedback\n{retry_feedback}")
        parts.append(f"## Command\n```\n{command}\n```")
        parts.append("## Scope (read code under these paths)")
        for path in scope:
            parts.append(f"- `{path}`")
        if caller_hints:
            parts.append(
                "## Caller hints (advisory, not a substitute for reading "
                f"code)\n{caller_hints}"
            )
        parts.append(
            "Read the code under scope, then emit the VerificationPlan "
            "JSON now."
        )
        return "\n\n".join(parts)

    # ------------------------------------------------------------------ #
    # Fixer
    # ------------------------------------------------------------------ #

    def _run_fixer(
        self,
        *,
        plan: dict[str, Any],
        digest: ObservationDigest,
        failures: list[CheckResult],
        fix_scope: list[str],
    ) -> None:
        if self.llm_provider is None or self.cc_config is None:
            raise _FixerError(
                "fixer requires cc_config + llm_provider (test stubs "
                "should disable fixing by leaving fix_scope empty)"
            )
        from ..agents.cc_agent import _run_in_fresh_loop
        _run_in_fresh_loop(self._run_fixer_async(
            plan=plan, digest=digest,
            failures=failures, fix_scope=fix_scope,
        ))

    async def _run_fixer_async(
        self,
        *,
        plan: dict[str, Any],
        digest: ObservationDigest,
        failures: list[CheckResult],
        fix_scope: list[str],
    ) -> None:
        from core.cc.runtime import build_default_query_engine

        engine = build_default_query_engine(
            cwd=self.cwd,
            config=self.cc_config,
            llm_client_provider=self.llm_provider,
        )
        _strip_to_fixer_tools(engine)
        _install_fix_scope_guard(engine, fix_scope=fix_scope, cwd=self.cwd)

        system = _load_fixer_system(self.language)
        user_parts = [
            _format_plan_for_prompt(plan),
            _format_digest_for_prompt(digest),
            _format_failures_for_prompt(failures),
            "## fix_scope (you may edit ONLY files matching these globs)",
            "\n".join(f"- `{g}`" for g in fix_scope),
            "Fix the underlying cause and emit a short final message.",
        ]
        user = "\n\n".join(user_parts)
        framed = f"<system>\n{system}\n</system>\n\n{user}"

        async def _drain() -> None:
            async for _event in engine.submit_message(
                framed,
                max_tool_rounds=self.FIXER_MAX_ROUNDS,
                purpose="watch_fixer",
            ):
                pass

        try:
            await asyncio.wait_for(
                _drain(), timeout=self.FIXER_WALL_CLOCK_TIMEOUT_S,
            )
        except asyncio.TimeoutError as exc:
            # The outer watch loop records this as a failed fix attempt
            # and reruns verification, so timeout recovery is fail+retry.
            raise _FixerError(
                f"fixer wall-clock timeout after "
                f"{self.FIXER_WALL_CLOCK_TIMEOUT_S}s"
            ) from exc
        finally:
            engine.close()

    # ------------------------------------------------------------------ #
    # Artifact + final summary
    # ------------------------------------------------------------------ #

    def _persist_plan_artifact(self, raw_path: str, plan: dict[str, Any]) -> None:
        path = Path(raw_path)
        if not path.is_absolute():
            path = Path(self.cwd) / path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(plan, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning(
                "watch mode: could not persist plan artifact to %s: %s",
                path, exc,
            )

    def _build_failed_summary(
        self,
        *,
        plan: dict[str, Any],
        failures: list[CheckResult],
        iter_count: int,
        fix_scope_empty: bool,
        max_iterations: int,
    ) -> str:
        head = (
            f"watch mode: FAILED after {iter_count} iteration(s) "
            f"(max={max_iterations}). "
        )
        if fix_scope_empty:
            head += "fix_scope was empty so no auto-fix was attempted. "
        else:
            head += "auto-fix did not converge inside the iteration budget. "
        head += "VerificationPlan summary: "
        head += repr(str(plan.get("summary") or "(no summary)"))
        head += "\n\n"
        head += _format_failures_for_prompt(failures)
        return head


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #

class _PhaseZeroError(Exception):
    def __init__(self, msg: str, *, reason_code: str) -> None:
        super().__init__(msg)
        self.reason_code = reason_code


class _FixerError(Exception):
    """Raised when the fixer sub-agent crashes or times out. Non-fatal
    at the loop level — caller logs it and proceeds to the next
    iteration so the iteration cap eventually terminates the run."""


__all__ = [
    "WatchModeRunner",
    "_FixScopeGuard",
    "_PhaseZeroError",
    "_FixerError",
    "_install_fix_scope_guard",
    "_path_matches_any_glob",
    "_parse_plan_json",
    "_run_command",
    "_validate_plan",
]
