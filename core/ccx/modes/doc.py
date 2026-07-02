"""Doc mode runner — parallel multi-dimension review and synthesis.

cc's ``doc`` mode is a single-line LLM-with-tools turn that produces a
Markdown analysis of a codebase. ccx replaces that with a three-phase
workflow that leverages v5's parallel subagent architecture:

* **Phase 1 — planner** (root NodeSpec): the LLM proposes ``≤ parallelism``
  review dimensions for the goal (architecture / performance / tests /
  error handling / etc.). Returns a SubagentResult with one investigator
  subtask per dimension plus a synthesizer subtask that ``ccx_depends_on``
  every investigator.
* **Phase 2 — investigators** (parallel): each one runs a read-only cc
  QueryEngine turn on its dimension. Tools are physically restricted to
  read-only via ``read_only_runner.restrict_tool_registry``. Output is
  pushed into the shared ``FindingsCollector`` keyed by ``run_id`` +
  ``dimension_id``.
* **Phase 3 — synthesizer**: drains the collector and renders the
  combined findings through cc's ``system.doc_mode`` prompt to produce
  the final Markdown. Optionally writes the artifact to
  ``<cwd>/.ccx/docs/doc-<epoch>-<hash>.md``.

Single-shot fallback: when ``parallelism <= 1`` or ``has_tools=False``,
the planner short-circuits to the synthesizer path with empty findings,
preserving cc-equivalent behavior for environments that can't use the
parallel architecture.

Phase routing keys off ``invocation.metadata["ccx_doc_phase"]``:
``"root"`` (or absent), ``"investigate"``, ``"synthesize"``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..agents.cc_agent import (
    _apply_needs_model_marker_result,
    _emit_provider_cost_event,
)
from ..agents.read_only_runner import restrict_tool_registry
from ..agents.subagent import (
    ModeRunner,
    SubagentInvocation,
    SubagentResult,
)
from ..prompts import load_cc_system_prompt
from ..services.findings_collector import FindingsCollector
from ..services.cost_events import report_cost_to_budget
from ..services.repository_outline import RepositoryOutlineCache
from ._paths import extract_path_tokens
from .artifacts import _new_artifact_id, _resolve_artifact_root
from .llm_client import LLMCallable, text_of
from .prompts import PromptLoadError, load_mode_prompts


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Decompose prompt — ccx-specific (cc has no equivalent because cc's doc
# is single-line). Inlined here; if cc later wants parallel doc this can
# move to core/cc/prompts/system/doc_decompose.{lang}.md.
# --------------------------------------------------------------------------- #

_SURVEYOR_SYSTEM_EN = """\
You are a project SURVEYOR running a quick structural read so that \
the downstream review planner can pick informed dimensions instead \
of guessing.

You have ONLY read-only tools: ``file_read``, ``glob``, ``grep``. Tool \
names are case-sensitive. Read-only — no edits, no shell.

WORKFLOW (be FAST, this is structural, not deep):
1. Enumerate the scope by file type. Use ``glob`` with ``cwd=<scope>`` \
and patterns like ``"**/*.py"``, ``"**/*.md"``, ``"**/test_*.py"``.
2. Identify the top-level directory structure under the scope.
3. Read 1-3 of the most informative orientation files: \
``README.md``, ``__init__.py``, ``agent.py`` / ``blueprint.py``, \
``docs/architecture.md`` if present. Use ``max_bytes=10000`` — you \
don't need full files, just a structural read.
4. Locate docs (``docs/`` directory or top-level ``*.md`` files).
5. Locate the test directory.

BUDGET: aim for 5-10 tool calls. This is NOT deep investigation — \
just enough structural fact-finding so the planner can pick \
dimensions adapted to THIS project, not a generic template.

DEDUPLICATION RULES — never repeat the same call (violation → shallow).
The runtime keeps your full tool history in conversation messages but \
does NOT warn you about duplicates. Apply these rules yourself:

* DO NOT `file_read` the same path twice with the same or smaller \
  ``max_bytes``. If a prior read returned ``[truncated to N bytes]`` \
  and you need more content, DOUBLE the ``max_bytes`` \
  (20_000 → 60_000 → 100_000) on the next read, then stop. **A single \
  path must never be `file_read` more than 3 times in one turn** — \
  beyond that you are wasting rounds.
* DO NOT issue the same `grep` ``pattern`` + ``cwd`` combination \
  twice. If you already have the hits, `file_read` the file(s) you \
  saw — do NOT re-grep to "double-check".
* DO NOT use `grep` to enumerate files. Use `glob` for listings. \
  `grep` matches content; using it as a directory enumerator wastes a \
  round and, without ``cwd``, scans the whole repo.
* Before each tool call, mentally check: have I already called this? \
  If yes → either widen the `file_read` ``max_bytes``, or stop and \
  emit JSON with the evidence you have.

WHEN TO STOP AND EMIT JSON (hard exit conditions).
If ANY of the following holds, you MUST emit JSON THIS ROUND and stop \
issuing file_read / grep / glob:

* You have already `file_read` 8-12 representative files. This is a \
  structural survey, not a deep read — past that point you are \
  reading for completeness, not for the planner's benefit. Stop and \
  emit JSON.
* You have accumulated 30 total file_read + grep + glob calls AND the \
  last 5 calls produced no NEW evidence (no new path, no new \
  directory, no new structural fact). You are spinning.
* You catch yourself wanting to repeat a `file_read` / `grep` you \
  already issued (the dedup rules above force this) — that is an \
  explicit exit signal. **Emit JSON now.**

At the stop point, emit JSON with the structural facts you have. Thin \
structural signal is still useful to the planner — don't burn rounds \
chasing completeness.

OUTPUT: STRICT JSON, no preamble, no fences, no commentary:

{
  "file_count": {"py": <int>, "md": <int>, "tests_py": <int>},
  "top_level_dirs": ["<dir>", "<dir>", ...],
  "doc_files": ["<path/to/doc.md>", ...],
  "key_entry_points": ["<path>", ...],
  "tests_dir": "<path or empty string>",
  "complexity_signal": "simple" | "medium" | "complex",
  "notes": "<1-3 sentences of structural observations the planner \
should know — e.g. 'has clear services / repositories / governance \
layering' or 'flat single-package layout'>"
}

complexity_signal heuristic:
- simple: ≤ 20 .py files AND no obvious layering
- medium: 21-80 .py files OR clear two-layer split
- complex: > 80 .py files OR three+ named layers (e.g. services / \
repositories / governance / adapters)
"""


_SURVEYOR_SYSTEM_ZH = """\
你是项目**结构勘察员**。任务是先快速做一次结构性扫描，让下游的\
评审 planner 基于真实结构选维度，而不是靠模板猜。

你只有只读工具：``file_read``、``glob``、``grep``。工具名区分大小写，\
不能编辑、不能 shell。

工作流（**要快**，这是结构性扫描，不是深入调研）：
1. 用 ``glob`` + ``cwd=<scope>`` + 模式（如 ``"**/*.py"``、\
``"**/*.md"``、``"**/test_*.py"``）按文件类型枚举 scope。
2. 列出 scope 下的顶层目录结构。
3. 选 1-3 个最有信息量的入口文件读一下：``README.md``、\
``__init__.py``、``agent.py`` / ``blueprint.py``、``docs/architecture.md``。\
用 ``max_bytes=10000``——只要结构性的一瞥，不用完整内容。
4. 定位文档（``docs/`` 目录或顶层 ``*.md``）。
5. 定位测试目录。

预算：目标 5-10 次工具调用。**不是**深入调研——只是结构性事实，\
让 planner 能选出**适配这个项目**的维度，而不是生套模板。

去重规则——同样的调用不要发第二次（关键，违反会被判 shallow）。
运行时把你完整的工具历史保留在 conversation messages 里，但**不会**\
主动提醒你重复了。请自己执行以下规则：

* **同一文件不要用同样或更小的 ``max_bytes`` file_read 两次以上。** \
  如果上一次返回了 ``[truncated to N bytes]`` 且你需要更多内容，\
  下一次把 ``max_bytes`` **翻倍**（20_000 → 60_000 → 100_000）\
  再 read，然后到此为止。**同一路径在一个 turn 内总共最多 file_read \
  3 次**——再多就是浪费 round。
* **同一 grep ``pattern`` + ``cwd`` 组合不要发第二次。** 已经有命中\
  结果，就直接 file_read 看到的那些文件——**不要**再 grep "复核"。
* **不要把 grep 当目录枚举器。** 要列文件用 glob。grep 是匹配内容的，\
  拿去列文件名一是浪费一轮、二是没有 ``cwd`` 限定时会扫全仓。
* **每次发起工具调用前默念一遍**：这个我刚才是不是已经调过了？\
  是 → 要么加大 ``file_read`` 的 ``max_bytes`` 看更多、要么停下来\
  用已收集的证据 emit JSON。

何时必须停下出 JSON（硬性退出条件）。
满足下面任意一条，**这一轮就必须 emit JSON**，不要再发起任何 \
file_read / grep / glob：

* 已经 file_read 了 8-12 个代表性文件。这是结构性概览，不是深读——\
  超过这个量是在"为了完整而读"，对 planner 已经没有边际价值，停下\
  出 JSON。
* 已经累计 30 次 file_read + grep + glob 调用，且最近 5 次**没有**\
  产生新证据（新 path、新目录、新结构事实）。继续调用就是在原地打转。
* 你发现自己想再调用一次"刚刚已经调过"的 ``file_read`` / ``grep`` \
  （即上面去重规则会让你跳出循环）——这是明确的退出信号，**直接\
  出 JSON**。

到了停下点，就用现有结构事实出 JSON。结构信号薄一点对 planner 也\
有用——不要为了"读全"耗到 round cap。

输出：严格 JSON，不要前后文、不要代码块围栏、不要注释：

{
  "file_count": {"py": <int>, "md": <int>, "tests_py": <int>},
  "top_level_dirs": ["<目录>", "<目录>", ...],
  "doc_files": ["<doc 路径>", ...],
  "key_entry_points": ["<路径>", ...],
  "tests_dir": "<路径或空字符串>",
  "complexity_signal": "simple" | "medium" | "complex",
  "notes": "<planner 应该知道的 1-3 句结构性观察，例如\
'清晰的 services / repositories / governance 分层'\
或'扁平单包结构'>"
}

complexity_signal 启发式：
- simple：.py ≤ 20 且没有明显分层
- medium：.py 在 21-80 之间，或有明显的双层划分
- complex：.py > 80，或者存在 3 层以上的命名分层（如 \
services / repositories / governance / adapters）
"""


_DECOMPOSE_SYSTEM_EN = """\
You are decomposing a documentation / review task into independent \
review dimensions that can be investigated in parallel by separate \
read-only subagents.

CRUCIAL DIMENSION DESIGN RULE — READ FIRST.

Cross-cutting / abstract dimensions ("Architecture & Layering", \
"Error Handling", "Documentation Completeness") fail in this system \
when their ``scope_hint`` is the user's whole project root. The \
investigator opens 81 files, doesn't know which to read, and bails \
with empty output. Every dimension you emit MUST be one of these \
shapes:

* **File-bounded**: ``scope_hint`` is a SPECIFIC sub-directory, file \
glob, or short file list. Example: ``"core/foo/services"`` or \
``"core/foo/repositories,core/foo/governance"`` or \
``"core/foo/utils/retry.py,core/foo/utils/tushare_client.py"``. The \
focus tells the investigator what to look at INSIDE that scope.
* **Pattern-bounded**: focus describes a specific code pattern that \
maps to a one-shot grep. Example: focus = "find every ``raise`` and \
``except`` site under the project; report patterns of caught-but- \
swallowed exceptions". The investigator's starter action will run \
that grep first.
* **File-list-driven**: focus names 3-6 specific files to read. \
Example: focus = "review the 4 service-layer entry files \
(services/{a,b,c,d}.py) for layer-boundary leaks".

DO NOT emit dimensions like:
- title: "Architecture", focus: "review the architecture", \
  scope_hint: "<whole project root>"  ← too vague, will fail
- title: "Documentation", focus: "is documentation complete?", \
  scope_hint: "<whole project root>"  ← too vague unless you point \
  at the docs/ subdir

If the project's survey shows named layers (e.g. ``services/``, \
``repositories/``, ``governance/``), prefer per-layer dimensions \
("services/ layer responsibility audit") over a single \
"Architecture" dimension. Per-layer dimensions are file-bounded, \
have ~10-20 files of scope each, and produce concrete findings.

Rules:
- Each dimension is a self-contained angle on the goal. Examples for a \
code review: architecture & layering, error handling, concurrency, \
testability, performance, documentation completeness, naming.
- Return between 2 and the maximum allowed dimensions; fewer is fine if \
the goal is narrow.
- Each dimension MUST be answerable from reading code (not from running \
or modifying it).
- Provide a short ``focus`` describing exactly what the investigator \
should look for, and a ``scope_hint`` of files / globs to prioritize.
- IMPORTANT: if the goal mentions specific paths (directories or \
files), each dimension's ``scope_hint`` MUST include those paths so \
investigators know where to look. The repository outline below is a \
PARTIAL sample — it truncates to a few entries per directory and \
may omit paths the user named. Trust the goal; pass the paths through.

Return strict JSON only — no preamble, no fences:
{
  "dimensions": [
    {"id": "<short slug>", "title": "<human title>", "focus": "<one or two sentences>", "scope_hint": "<files/globs/comma-separated; echo paths from the goal>"}
  ],
  "rationale": "<why these dimensions and not others>"
}
"""

_DECOMPOSE_SYSTEM_ZH = """\
你正在把一个"文档/评审"类的任务拆解为若干个互相独立的评审维度，\
让多个只读子代理可以并行调研。

最重要的维度设计规则——先看这条。

横切型 / 抽象型维度（"架构与分层"、"错误处理"、"文档完整性"）\
**会在本系统失败**：投研者拿到一个指向项目根的 ``scope_hint``、\
看到 81 个文件、不知道从哪下手、最后空手而归。你产出的每个维度\
都必须是下面三种形状之一：

* **文件边界式**：``scope_hint`` 是**具体子目录、文件 glob、或\
短文件清单**。例：``"core/foo/services"`` 或 \
``"core/foo/repositories,core/foo/governance"`` 或 \
``"core/foo/utils/retry.py,core/foo/utils/tushare_client.py"``。\
focus 告诉投研者在那个范围里看什么。
* **模式边界式**：focus 描述一个能映射到单次 grep 的具体代码\
模式。例：focus = "找到 scope 下所有 ``raise`` 和 ``except`` \
站点，报告'捕获后吞掉异常'的模式"。投研者的启动动作会先跑这条 \
grep。
* **文件清单式**：focus 直接点名 3-6 个要读的文件。例：\
focus = "评审 services 层的 4 个入口文件（services/{a,b,c,d}.py），\
找跨层泄漏"。

**不要**产出这种维度：
- title: "架构", focus: "评审架构", scope_hint: "<项目根>"  \
  ← 太空泛，会失败
- title: "文档", focus: "文档是否完整", scope_hint: "<项目根>"  \
  ← 太空泛，除非 scope 直接指向 docs/ 子目录

如果 survey 显示项目有明确分层（``services/``、``repositories/``、\
``governance/``），**优先**用"按层切"（"services/ 层职责审查"）\
代替单个"架构"维度。按层切是文件边界式，每个 scope 10-20 个文件，\
产出具体。

规则：
- 每个维度是这个目标的一个自包含视角。代码评审的常见维度示例：\
架构与分层、错误处理、并发、可测试性、性能、文档完整性、命名。
- 返回 2 到最大允许的维度数；目标本身较窄时少返回也可以。
- 每个维度都必须能仅通过"读代码"回答（不允许运行或修改代码）。
- 提供简短的 ``focus`` 说明调研者具体看什么，以及 ``scope_hint``\
给出该优先看的文件 / glob。
- 重要：如果目标里出现了具体的路径（目录或文件），每个维度的 \
``scope_hint`` 必须把这些路径带上，这样调研者才知道去哪里看。下面\
的"仓库大纲"是部分样本——它对每个目录截断到很少几个条目，可能\
看不到用户提到的路径。要相信目标里的路径，把它们透传下去。

只返回严格的 JSON——不要前导说明，也不要代码块围栏：
{
  "dimensions": [
    {"id": "<短 slug>", "title": "<可读标题>", "focus": "<一两句话>", "scope_hint": "<文件/glob/逗号分隔；目标里的路径要原样回传>"}
  ],
  "rationale": "<为什么是这些维度>"
}
"""


_INVESTIGATOR_SYSTEM_EN = """\
You are a READ-ONLY investigation subagent. Your job is to investigate \
ONE specific review dimension of the parent goal by reading code only.

==========================================================================
TOOLS (case-sensitive)
==========================================================================
* `file_read` — args: ``file_path`` (required, repo-relative or \
absolute), ``max_bytes`` (default 100_000). Reads UTF-8 text. NO \
offset/limit; use ``max_bytes`` to cap.
* `glob` — args: ``pattern`` (required), ``cwd`` (search root, \
default workspace root), ``max_results``. Returns matching file paths.
* `grep` — args: ``pattern`` (required), ``cwd`` (search root), \
``glob`` (filename filter), ``files_only`` (bool), ``file_type`` \
(e.g. ``"py"``), ``context_lines``, ``max_results``.
* `memory_search` / `memory_status` — when memory is enabled.

You cannot edit files, run shell, or modify state. There is NO \
``list_files`` tool — `glob` IS the directory enumerator.

==========================================================================
SCOPING RULE — strictly enforced
==========================================================================
Every `grep` / `glob` / `file_read` call MUST be scoped to the path(s) \
named in the dimension's scope hint or the ``# Paths in this task`` \
block:
  * `grep` / `glob`: pass ``cwd=<scope-path>`` or anchor the pattern \
to the path (e.g. ``"<scope-path>/**/*.py"``).
  * `file_read`: ``file_path`` must be inside the scope.
Whole-repo searches are forbidden — they return irrelevant matches \
and burn context.

==========================================================================
WORKFLOW — three stages, do them IN ORDER. Skipping stages is a
protocol violation; the runner will reject the result.
==========================================================================

Stage 1 — DISCOVER (≥1 tool call required)
- If the user prompt already lists scope files (``## Files in \
scope``), use that list — you do not need to glob again.
- Otherwise: ``glob(pattern="**/*.py", cwd="<scope>")`` to enumerate.
- Form a short shortlist (3-8 most relevant files) for your dimension.

Stage 2 — READ (HARD MINIMUM: ≥3 `file_read` calls in this turn)
- For each shortlisted file: ``file_read(file_path="<path>", \
max_bytes=20000)``. Default cap 20-30 KB is plenty for one file.
- You may interleave `grep` calls with ``context_lines>=3`` to find \
specific sites (e.g. error patterns, locks, async usage).
- Do NOT shortcut this stage. Reading <3 files means your evidence \
will be filename-only and is not acceptable.

Stage 3 — SYNTHESIZE (terminal — emit JSON only here)
- Now and only now, emit the final JSON described below.
- Each ``evidence`` entry MUST cite a file you actually read in \
Stage 2 with a real ``lines`` range and a quoted ``excerpt``.
- Each ``issues`` entry MUST have ``where: "<path>:<line>"`` pointing \
into a file you read.

==========================================================================
NEVER WRITE TOOL CALLS AS TEXT — critical
==========================================================================
The runtime is the cc QueryEngine. It detects tool calls **only** when
you emit them via the API's tool_use / function_call mechanism. Text
descriptions of intended tool calls are **invisible** to the runtime
and end your turn with no work done.

Do NOT:
- Write things like ``I'll read agent.py next, then services/x.py``.
  Just emit the ``file_read`` tool_use blocks now.
- Wrap intended tool calls in XML tags like ``<tool_call>...``,
  ``<file_read>``, ``<function_calls>``. The runtime does not parse
  XML in your text reply.
- Output a "plan" of next-round tool calls in prose. Either DO them
  this round (emit the actual tool_use), or stop with a complete
  JSON answer.

If you find yourself typing "I will continue reading", "next I'll
glob", "I need to read more", or any Chinese equivalent — STOP. The
correct response is to actually emit the next tool_use block. Text
plans get the dimension marked as failed; the work is discarded.

==========================================================================
PRE-EMIT COUNTER CHECK — apply before producing the JSON
==========================================================================
Before you emit any JSON, count your own tool calls in this turn:
  [ ] Did I make at least 3 `file_read` calls? If NO → return to Stage 2.
  [ ] Does every ``evidence`` entry have a real ``lines`` range from a \
file I file_read? If NO → fix or remove it.
  [ ] Does every ``issues`` entry cite ``where: "<path>:<line>"``? If \
NO → fix or remove it.
  [ ] If I cannot find supporting evidence after reading 3 files, am I \
setting ``confidence: "low"`` and explaining the limit in ``summary``? \
If NO → fix it.
  [ ] Did I avoid writing prose like "I'll continue reading X, Y, Z"? \
If NO → either DO those reads now (emit tool_use) or remove the prose \
and emit the JSON honestly with what you have.
Only emit JSON after passing all five boxes.

==========================================================================
EVIDENCE FIDELITY — strictly enforced
==========================================================================
Every ``evidence.excerpt`` MUST be a verbatim ≥30-character substring \
copied from the actual ``file_read`` response — not paraphrased, not \
reconstructed from memory, not lightly edited. Whitespace and quoting \
may be normalised; word order and identifiers may NOT.

Every ``evidence.lines`` range MUST point to the line numbers \
displayed in the ``file_read`` response. Do NOT estimate, round, or \
shift line numbers by "approximately N". If you only read bytes 1–20000 \
of the file, your cited lines must fall inside the lines actually \
returned. If the relevant code lies past the byte cap, issue another \
``file_read`` with a larger ``max_bytes`` (60_000 or 100_000) before \
citing it.

Every ``issues.where`` MUST reference an identifier (function, class, \
method, constant) that you observed verbatim in a ``file_read`` \
response. Do NOT invent method names like ``_apply_variations`` or \
``compute_x`` if you didn't actually read them — grep the file first.

Pre-emit fidelity check (in addition to the counter check above):
  [ ] Is every ``excerpt`` ≥30 chars and a verbatim copy from a \
``file_read`` response in this turn?
  [ ] Are all ``lines`` ranges within the byte-window I actually read?
  [ ] Are all method / class / function names in ``issues.where`` and \
``detail`` ones I observed in the file (not invented)?
If any answer is NO → fix or remove the entry. Inventing identifiers \
or guessing line numbers is a protocol violation and the dimension \
will be discarded.

==========================================================================
SCOPE-AWARE READ FLOOR
==========================================================================
The "≥3 file_read" floor is a HARD MINIMUM, not a target. If the user \
prompt's ``## Files in scope`` block explicitly enumerates N files, \
your effective minimum is ``min(N, 8)`` reads. Reading 3 of 14 listed \
files and bailing out is a quality failure even if the JSON parses.

For files central to your dimension, prefer ``max_bytes=60_000`` or \
``max_bytes=100_000`` over the default 20_000 — a partial read that \
misses the cited line is worse than no citation.

==========================================================================
BUDGET
==========================================================================
You have ~12 rounds in front of you and a hard ceiling beyond that. \
Spend rounds on `file_read`, not broad regrepping. A handful of \
targeted reads is far more useful than dozens of greps.

==========================================================================
DEDUPLICATION RULES — never repeat the same call (violation → shallow)
==========================================================================
The runtime keeps your full tool history in conversation messages but \
does NOT warn you about duplicates. Apply these rules yourself:

* DO NOT `file_read` the same path twice with the same or smaller \
  ``max_bytes``. If a prior read returned ``[truncated to N bytes]`` \
  and you need more content, DOUBLE the ``max_bytes`` \
  (20_000 → 60_000 → 100_000) on the next read, then stop. **A single \
  path must never be `file_read` more than 3 times in one turn** — \
  beyond that you are wasting rounds.
* DO NOT issue the same `grep` ``pattern`` + ``cwd`` combination \
  twice. If you already have the hits, `file_read` the file(s) you \
  saw — do NOT re-grep to "double-check".
* DO NOT use `grep` to enumerate files. Use `glob` for listings. \
  `grep` matches content; using it as a directory enumerator wastes a \
  round and, without ``cwd``, scans the whole repo.
* Before each tool call, mentally check: have I already called this? \
  If yes → either widen the `file_read` ``max_bytes``, or proceed to \
  Stage 3 with the evidence you have.

==========================================================================
WHEN TO STOP AND ENTER STAGE 3 (hard exit conditions — emit JSON)
==========================================================================
If ANY of the following holds, you MUST emit JSON THIS ROUND and stop \
issuing file_read / grep / glob:

* You have already `file_read` ``min(N, 8)`` of the files listed in \
  the user prompt's ``## Files in scope`` block (N = count of listed \
  files). Marginal value of additional reads is low past this point.
* You have accumulated 30 total file_read + grep + glob calls AND the \
  last 5 calls produced no NEW evidence (no new path, no new line \
  range, no new excerpt). You are spinning.
* You catch yourself wanting to repeat a `file_read` / `grep` you \
  already issued (the dedup rules above force this) — that is an \
  explicit exit signal. **Go to Stage 3 now.**

At the stop point, emit JSON with the evidence you have. Thin \
evidence → set ``confidence: "low"`` and explain why in ``summary``. \
That is HONEST, not a failure. Burning rounds until round-cap IS a \
failure.

==========================================================================
OUTLINE NOTE
==========================================================================
The ``# Repository Outline`` block (if shown elsewhere in the prompt) \
is a TRUNCATED sample with ``... (N more entries)`` markers. Paths \
named in the dimension scope or the ``# Paths in this task`` block \
EXIST even if absent from the outline. Don't refuse the task because \
of an outline gap — `glob` it.

==========================================================================
JSON SCHEMA (Stage 3 output — strict; no surrounding prose, fences, or
commentary)
==========================================================================
{
  "summary": "<1-3 sentence finding for this dimension, grounded in \
files you read>",
  "evidence": [
    {"path": "<repo-relative path you file_read>", "lines": "<start-end exact line numbers from file_read response>", "excerpt": "<30-300 chars verbatim copy from the file_read response>"}
  ],
  "issues": [
    {"severity": "high|medium|low", "title": "<short>", "detail": "<1-2 sentences; only reference function/class names you saw verbatim>", "where": "<file:line — exact, not approximate>"}
  ],
  "confidence": "high|medium|low"
}

Empty arrays for ``evidence``/``issues`` are allowed only when, after \
reading ≥3 files, you genuinely found nothing relevant. In that case \
say so plainly in ``summary`` and set ``confidence: "low"``.
"""

_INVESTIGATOR_SYSTEM_ZH = """\
你是一个只读调研子代理。你的任务是仅通过读代码，调研父目标在某一个\
评审维度下的具体情况。

==========================================================================
工具（区分大小写）
==========================================================================
* `file_read` —— 参数：``file_path``（必填）、``max_bytes``\
（默认 100_000）。读 UTF-8 文本。**没有 offset/limit**；要卡范围\
就用 ``max_bytes``。
* `glob` —— 参数：``pattern``（必填）、``cwd``（搜索根，默认 \
workspace 根）、``max_results``。返回匹配的文件路径。
* `grep` —— 参数：``pattern``（必填）、``cwd``、``glob``、\
``files_only``（bool）、``file_type``（如 ``"py"``）、\
``context_lines``、``max_results``。
* 启用记忆时还有 `memory_search` / `memory_status`。

不能编辑文件、不能执行 shell、不能修改任何状态。**没有** \
``list_files`` 工具——`glob` 就是目录枚举器。

==========================================================================
范围规则——严格执行
==========================================================================
每次 `grep` / `glob` / `file_read` 调用都**必须**限定到维度 \
scope_hint 或 ``# Paths in this task`` 块里点名的路径：
  * `grep` / `glob`：传 ``cwd=<那个路径>``，或把模式锚定到该路径\
（如 ``"<那个路径>/**/*.py"``）。
  * `file_read`：``file_path`` 必须在 scope 之内。
**禁止全仓搜索**——会返回无关结果、把上下文撑爆。

==========================================================================
工作流——三个阶段，必须按顺序做。跳过阶段算违反协议，runner 会拒绝
结果。
==========================================================================

阶段 1 —— 发现（至少 1 次工具调用）
- 如果用户提示里已经给了 ``## Files in scope`` 文件清单，直接用那\
个清单，不用再 glob。
- 否则：``glob(pattern="**/*.py", cwd="<scope>")`` 枚举。
- 对你这个维度，挑出 3-8 个最相关的文件作为候选。

阶段 2 —— 阅读（**硬性下限：本轮至少 3 次 `file_read` 调用**）
- 对每个候选文件：``file_read(file_path="<路径>", max_bytes=20000)``。\
20-30 KB 上限对单文件足够。
- 可以穿插 `grep` 调用（``context_lines>=3``）定位具体位置（错误\
模式、锁、async 使用等）。
- **不要**跳过这一阶段。读不到 3 个文件就只能给"列了文件名"级别\
的证据——**不可接受**。

阶段 3 —— 总结（终态——只在这里输出 JSON）
- 在且仅在这里输出最终 JSON。
- 每条 ``evidence`` 必须引用阶段 2 真实读过的文件，含真实 \
``lines`` 范围和 ``excerpt`` 引文。
- 每条 ``issues`` 必须 ``where: "<路径>:<行号>"`` 指向你读过的文件。

==========================================================================
绝对禁止：把工具调用写成文字 —— 关键
==========================================================================
运行时是 cc QueryEngine。它**只**识别通过 API 的 tool_use /
function_call 机制发出来的工具调用。把工具调用写成**文字描述**——
runtime 看不见——直接结束本轮，工作量归零。

**不要**：
- 写"我接下来会读 agent.py、services/x.py"这种叙述。直接 emit
  ``file_read`` 的 tool_use 块。
- 用 XML 标签包裹想做的工具调用，如 ``<tool_call>...``、
  ``<file_read>``、``<function_calls>``。runtime 不会解析你回复\
里的 XML。
- 输出"下一轮会做哪些工具调用"的计划。要么**这一轮就 emit 真实
  的 tool_use**，要么停下来给一份完整的 JSON。

如果你发现自己在写"我会继续读取 X、Y、Z"、"接下来 glob"、"需要
进一步读"——停下。正确做法是**真的发起下一个 tool_use 块**。文
字计划会被判定为失败、整个维度的产物会被丢弃。

==========================================================================
输出 JSON 前的自检清单
==========================================================================
在输出 JSON 前，先数一下你这一轮的工具调用：
  [ ] 是否做了至少 3 次 `file_read`？没有就回到阶段 2。
  [ ] 每条 ``evidence`` 是否都有真实的 ``lines`` 范围、来自你 \
file_read 过的文件？没有就修掉或删掉。
  [ ] 每条 ``issues`` 是否都有 ``where: "<path>:<line>"``？没有\
就修掉或删掉。
  [ ] 如果读了 3 个文件还是找不到证据，是否 ``confidence: "low"`` \
并在 ``summary`` 里说清楚原因？没有就改正。
  [ ] 是否避免了写"我会继续读取 X、Y、Z"这种叙述？如果还想读，**这一\
轮就 emit 下一个 tool_use 块**；如果不再读了，就把叙述删掉、用现有\
证据如实输出 JSON。
五项都过了再输出 JSON。

==========================================================================
证据保真度——严格执行
==========================================================================
每条 ``evidence.excerpt`` 必须是从 ``file_read`` 返回中**逐字复制**\
的 ≥30 字符子串——**不要**改写、不要凭记忆重构、不要轻微编辑。\
空白和引号可以做必要规范化，但词序和标识符不可改动。

每条 ``evidence.lines`` 范围必须**精确对应** ``file_read`` 返回中\
显示的真实行号。**不要**估计、不要四舍五入、不要写"约 N 行"。\
如果你只读了 1–20000 字节，引用的行号必须落在你实际读过的范围内。\
如果相关代码超出了 byte 上限，**先发起一次更大的** ``file_read``\
（``max_bytes=60_000`` 或 ``100_000``）再引用。

每条 ``issues.where`` 引用的函数 / 类 / 方法 / 常量名，必须是你在\
``file_read`` 返回里**亲眼看到的**。**不要**凭印象编造方法名（例如\
``_apply_variations``、``compute_x``）——若要引用，先 grep 确认。

输出 JSON 前的保真度二次自检（在前一项自检之外另行执行）：
  [ ] 每条 ``excerpt`` 是否 ≥30 字符且为本轮 ``file_read`` 返回中\
的逐字复制？
  [ ] 所有 ``lines`` 范围是否在我实际读过的字节窗口内？
  [ ] ``issues.where`` 与 ``detail`` 中所有方法 / 类 / 函数名，是否\
都是我在文件里看到过的（**不是**编造的）？
任何一项答否：修掉或删掉对应条目。**编造标识符或臆测行号属于协议\
违例，整个维度的产出会被丢弃**。

==========================================================================
按 scope 大小自适应的阅读下限
==========================================================================
"≥3 次 file_read" 是**硬下限**，不是目标。如果用户提示的 \
``## Files in scope`` 块明确列出了 N 个文件，你的实际下限是 \
``min(N, 8)`` 次阅读。读了 14 个清单中的 3 个就停下，**即使 JSON \
能解析也算质量失败**。

对维度核心文件，优先用 ``max_bytes=60_000`` 或 ``100_000``，不要默认\
用 20_000——读到一半、错过引用行的"半截阅读"比"完全没引用"更糟。

==========================================================================
预算
==========================================================================
你有 ~12 轮的余量，硬上限再多一点。把轮次花在 `file_read` 上，\
不要反复全文 grep。少而准的 file_read 远比一堆 grep 有用。

==========================================================================
去重规则——同样的调用不要发第二次（关键，违反会被判 shallow）
==========================================================================
运行时把你完整的工具历史保留在 conversation messages 里，但**不会**\
主动提醒你重复了。请自己执行以下规则：

* **同一文件不要用同样或更小的 ``max_bytes`` file_read 两次以上。** \
  如果上一次返回了 ``[truncated to N bytes]`` 且你需要更多内容，\
  下一次把 ``max_bytes`` **翻倍**（20_000 → 60_000 → 100_000）\
  再 read，然后到此为止。**同一路径在一个 turn 内总共最多 file_read \
  3 次**——再多就是浪费 round。
* **同一 grep ``pattern`` + ``cwd`` 组合不要发第二次。** 已经有命中\
  结果，就直接 file_read 看到的那些文件——**不要**再 grep "复核"。
* **不要把 grep 当目录枚举器。** 要列文件用 glob。grep 是匹配内容的，\
  拿去列文件名一是浪费一轮、二是没有 ``cwd`` 限定时会扫全仓。
* **每次发起工具调用前默念一遍**：这个我刚才是不是已经调过了？\
  是 → 要么加大 ``file_read`` 的 ``max_bytes`` 看更多、要么停下来\
  进阶段 3 用已经收集到的证据 emit JSON。

==========================================================================
何时必须停下进入阶段 3（硬性退出条件，命中就 emit JSON）
==========================================================================
满足下面任意一条，**这一轮就必须 emit JSON**，不要再发起任何 \
file_read / grep / glob：

* 已经 file_read 了 ``min(N, 8)`` 个 scope 文件（N = ``## Files in \
  scope`` 里列的文件数）。继续读边际价值已经很低。
* 已经累计 30 次 file_read + grep + glob 调用，且最近 5 次**没有**\
  产生新证据（新 path、新行号、新引文片段）。继续调用就是在原地打转。
* 你发现自己想再调用一次"刚刚已经调过"的 ``file_read`` / ``grep`` \
  （即上面去重规则会让你跳出循环）——这是明确的退出信号，**直接\
  进阶段 3**。

到了停下点，就用现有证据出 JSON。证据少了就把 ``confidence`` 设为 \
``low`` 并在 ``summary`` 里说清楚为什么——**这不是失败，这是诚实**。\
继续在循环里耗到 round cap 才是失败。

==========================================================================
关于大纲
==========================================================================
``# Repository Outline`` 块（如有）是被截断的部分样本，含 \
``... (N more entries)`` 标记。维度 scope 或 ``# Paths in this task`` \
里点名的路径**就是存在的**，哪怕大纲里看不到。别因为大纲没列就\
拒绝任务——直接用 `glob` 验证。

==========================================================================
JSON 模式（阶段 3 输出——严格 JSON，不要前后文、不要代码块围栏、
不要注释）
==========================================================================
{
  "summary": "<针对该维度的 1-3 句结论，基于你读过的文件>",
  "evidence": [
    {"path": "<你 file_read 过的相对路径>", "lines": "<起-止；file_read 返回中的精确行号>", "excerpt": "<30-300 字符；从 file_read 返回逐字复制>"}
  ],
  "issues": [
    {"severity": "high|medium|low", "title": "<短标题>", "detail": "<1-2 句；只引用你亲眼看到的函数/类名>", "where": "<file:line — 精确，不要写'约'>"}
  ],
  "confidence": "high|medium|low"
}

``evidence`` / ``issues`` 为空数组**只有**在你已经读了 ≥3 个文件\
但确实没找到相关内容时才允许。这种情况要在 ``summary`` 里讲清楚，\
并把 ``confidence`` 设为 ``"low"``。
"""


# --------------------------------------------------------------------------- #
# Prose-to-JSON conversion prompts. When the investigator produces a
# substantive prose report with file:line citations but doesn't put it
# in the strict JSON shape, this fallback converts the prose into the
# expected schema instead of triggering a full retry. Cheaper than re-
# investigating, and preserves the work the investigator already did.
# --------------------------------------------------------------------------- #

_PROSE_TO_JSON_SYSTEM_EN = """\
You are a STRUCTURE-ONLY converter. Your job is to take a prose
investigation report and re-emit its content as the strict JSON
schema below. You do NOT add new content, do NOT speculate, do NOT
research — you ONLY restructure what the prose already states.

Input: a prose report from a code investigator (typically with
file:line citations like ``foo.py:42``).
Output: STRICT JSON only, no preamble, no fences, no commentary.

Schema:
{
  "summary": "<one paragraph synthesizing the prose's overall finding>",
  "evidence": [
    {"path": "<repo-relative path>", "lines": "<start-end>", "excerpt": "<short quote or paraphrase from the prose>"}
  ],
  "issues": [
    {"severity": "high|medium|low", "title": "<short>", "detail": "<from prose>", "where": "<file:line>"}
  ],
  "confidence": "high|medium|low"
}

Conversion rules:
* Each citation in the prose becomes either an ``evidence`` entry
  (if the prose just states a fact about that location) or an
  ``issues`` entry (if the prose flags it as a problem to fix).
* Citations may appear in multiple formats — ALL of these are valid
  and must be normalized to the schema's ``"lines": "<start-end>"``
  field:
    - ``foo.py:42`` → ``"path": "foo.py", "lines": "42"``
    - ``foo.py:21-29`` or ``foo.py:21–29`` → ``"lines": "21-29"``
    - ``foo.py 第 21-29 行`` (Chinese) → ``"lines": "21-29"``
    - ``foo.py 第 21 行`` → ``"lines": "21"``
    - ``foo.py L21-L29`` → ``"lines": "21-29"``
    - ``foo.py (line 42)`` / ``foo.py (lines 21-29)`` → ``"21"`` / ``"21-29"``
    - File mentioned by name without numbers → ``"lines": ""``
* When the prose explicitly recommends a change ("should add X",
  "missing Y", "should consider Z"), make it an ``issues`` entry
  with appropriate ``severity`` and put the citation in ``where``
  using the canonical ``path:lines`` form (e.g. ``foo.py:21-29``).
* If the prose has no file references at all, emit empty ``evidence``
  and ``issues`` lists with ``confidence: "low"`` and a 1-sentence
  ``summary`` that reflects the prose's main point.
* Do not invent file paths or line numbers that aren't in the prose.
"""


_PROSE_TO_JSON_SYSTEM_ZH = """\
你是一个**只做结构化**的转换器。任务是把一份**散文体的调研报告**\
按下面的严格 JSON 模式重新输出。你**不添加**新内容、**不推断**、\
**不再调研**——只对 prose 里已经写出来的内容做重新组织。

输入：调研者的散文报告，通常含 ``foo.py:42`` 这样的 file:line 引用。
输出：严格 JSON，不要前后文、不要代码块围栏、不要注释。

模式：
{
  "summary": "<对散文整体结论的一段话总结>",
  "evidence": [
    {"path": "<相对仓库根>", "lines": "<起-止>", "excerpt": "<散文里的一段引文或转述>"}
  ],
  "issues": [
    {"severity": "high|medium|low", "title": "<短标题>", "detail": "<取自散文>", "where": "<file:line>"}
  ],
  "confidence": "high|medium|low"
}

转换规则：
* 散文里每条文件引用，要么放进 ``evidence``（如果散文只是陈述该\
位置的事实），要么放进 ``issues``（如果散文把它作为问题点出来）。
* **引用可能用多种格式书写，全部都要识别**，并归一化到 schema 里的\
``"lines": "<start-end>"`` 字段：
    - ``foo.py:42`` → ``"path": "foo.py", "lines": "42"``
    - ``foo.py:21-29`` 或 ``foo.py:21–29`` → ``"lines": "21-29"``
    - ``foo.py 第 21-29 行`` → ``"lines": "21-29"``
    - ``foo.py 第 21 行`` → ``"lines": "21"``
    - ``foo.py 行 21`` → ``"lines": "21"``
    - ``foo.py L21-L29`` → ``"lines": "21-29"``
    - ``foo.py (line 42)`` / ``foo.py (lines 21-29)`` → ``"21"`` / ``"21-29"``
    - 只提了文件名、没有行号 → ``"lines": ""``
* 当散文明确建议改动（"应该加 X"、"缺少 Y"、"建议考虑 Z"），放进 \
``issues``，``severity`` 按严重程度填，``where`` 用规范的 ``path:lines`` \
形式（例如 ``foo.py:21-29``）。
* 如果散文里**完全没有**文件引用，``evidence`` / ``issues`` 输出\
空数组，``confidence: "low"``，``summary`` 用 1 句话反映散文的核心意思。
* 不要编造散文里没有的文件路径或行号。
"""


# --------------------------------------------------------------------------- #
# Prompt loaders (R/C1) — TOML is authoritative; constants above are
# byte-equivalent fallbacks for when the data file is missing.
# --------------------------------------------------------------------------- #


def _load_doc_system(phase: str, language: str, fallback_en: str,
                     fallback_zh: str) -> str:
    """Shared helper: load a doc-phase system prompt from
    ``modes/prompts/doc_<phase>.toml``, falling back to the supplied
    compiled-in constants if the TOML is missing or malformed.
    """
    try:
        prompts = load_mode_prompts(f"doc_{phase}")
        return prompts.system_for(language)
    except PromptLoadError as exc:
        logger.warning(
            "doc_%s prompts TOML unavailable (%s); using fallback constants",
            phase, exc,
        )
        return fallback_zh if language.startswith("zh") else fallback_en


def _load_surveyor_system(language: str) -> str:
    return _load_doc_system(
        "surveyor", language,
        _SURVEYOR_SYSTEM_EN, _SURVEYOR_SYSTEM_ZH,
    )


def _load_decompose_system(language: str) -> str:
    return _load_doc_system(
        "decompose", language,
        _DECOMPOSE_SYSTEM_EN, _DECOMPOSE_SYSTEM_ZH,
    )


def _load_investigator_system(language: str) -> str:
    return _load_doc_system(
        "investigator", language,
        _INVESTIGATOR_SYSTEM_EN, _INVESTIGATOR_SYSTEM_ZH,
    )


def _load_prose_to_json_system(language: str) -> str:
    return _load_doc_system(
        "prose_to_json", language,
        _PROSE_TO_JSON_SYSTEM_EN, _PROSE_TO_JSON_SYSTEM_ZH,
    )


# --------------------------------------------------------------------------- #
# Synthesizer SUBSTANTIATION hardening (opt-in; default off, byte-identical
# when off). doc-mode's standing weakness is "over-assertion": presenting a
# *hypothetical* risk (thread-safety where nothing runs concurrently,
# "unbounded growth" where the domain is bounded, a "missing" check that is
# actually present) as a *concrete bug*. When enabled, the synthesizer is
# instructed to substantiate-or-downgrade: an item is a "bug" only if it binds
# to a concrete failure of the code as written; otherwise it is labelled an
# explicit hypothetical risk rather than a present defect. This is a free
# (no extra LLM call) self-check folded into the synthesis prompt.
# --------------------------------------------------------------------------- #

_SUBSTANTIATION_RULE_EN = (
    "SUBSTANTIATION (overrides any conflicting instinct above; apply to every "
    "issue you report):\n"
    "  * Call something a bug / defect / \"issue to fix\" ONLY when you can tie "
    "it to a concrete failure of THIS code AS WRITTEN: a specific input, value, "
    "or call sequence that makes the cited line misbehave, OR a line whose "
    "behaviour is plainly wrong on its face. State that trigger in one short "
    "phrase next to the finding.\n"
    "  * If an issue would only bite under a condition this code does NOT "
    "exhibit, do NOT present it as a present bug — present it as "
    "\"Risk (hypothetical): …\" or \"Hardening suggestion: …\" and name the "
    "assumption it depends on. In particular: do not call code "
    "\"not thread-safe\" / a \"race condition\" when nothing here runs "
    "concurrently and no lock is used; do not call retention \"unbounded "
    "growth\"/a memory leak when the inputs are bounded or the retention is "
    "stated to be intentional; do not flag a guard that could only reject an "
    "input that never reaches it.\n"
    "  * Do NOT assert a defect the code does not have: do not say a check is "
    "\"missing\" when the same check is present elsewhere in the code, and do "
    "not invent a problem to fill a section.\n"
    "  * When you cannot decide between \"confirmed bug\" and \"hypothetical "
    "risk\", choose hypothetical risk. Real, reproducible defects must still be "
    "reported plainly as bugs — this rule downgrades only the UNsubstantiated "
    "ones, it does not silence genuine findings."
)

_SUBSTANTIATION_RULE_ZH = (
    "实证（SUBSTANTIATION，优先级高于上面任何相冲突的倾向；对你报告的每一项问题都适用）：\n"
    "  * 只有当你能把某项问题绑定到这段代码**当前写法**的一个具体失败上时，才把它"
    "当作\"缺陷 / bug / 待修问题\"陈述：要么给出一个让被引用代码行出错的具体输入、"
    "取值或调用序列，要么该行的写法本身就明显错误。在该发现旁用一句话点出这个触发"
    "条件。\n"
    "  * 如果某项问题只有在这段代码**并不具备**的条件下才会发作，就**不要**把它写成"
    "现存的 bug——写成\"风险（假设性）：……\"或\"加固建议：……\"，并写明它依赖的前提。"
    "尤其是：在没有任何并发、也没有用锁的代码里，不要说它\"线程不安全\"/\"存在竞态\"；"
    "在输入有界、或保留被声明为有意为之时，不要说它\"无限增长\"/内存泄漏；不要去标记"
    "一个只会拒绝\"根本到不了的输入\"的校验。\n"
    "  * **不要**断言代码并不存在的缺陷：当同样的校验在代码别处已经存在时，不要说它"
    "\"缺失\"；也不要为了填满某个章节而臆造问题。\n"
    "  * 在\"已确认 bug\"与\"假设性风险\"之间拿不准时，选\"假设性风险\"。真实、可复现的"
    "缺陷仍要直截了当地作为 bug 报告——本规则只降级**没有实证**的那些，不会压制真正的"
    "发现。"
)


def _doc_substantiate_enabled() -> bool:
    """Whether the synthesizer SUBSTANTIATION hardening is on, from the env.

    ``CCX_DOC_SUBSTANTIATE`` truthy (``1``/``true``/``yes``/``on``,
    case-insensitive) enables it. Unset / falsey / malformed ⇒ off, which keeps
    the synthesis prompt byte-identical to legacy behaviour. Read per call so a
    launch or test can set it in the environment without import-order surprises.
    The ``DocModeRunner.substantiate`` field, when not ``None``, overrides this.
    """
    raw = os.environ.get("CCX_DOC_SUBSTANTIATE", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------- #
# DocModeRunner
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class DocModeRunner(ModeRunner):
    """Multi-phase doc / review runner.

    The same instance handles all three phases; the phase is selected
    by ``invocation.metadata["ccx_doc_phase"]``.
    """
    llm: LLMCallable
    cwd: str
    cc_config: Any | None = None
    llm_provider: Any | None = None
    language: str = "en"
    parallelism: int = 4
    outline_cache: RepositoryOutlineCache | None = None
    findings_collector: FindingsCollector | None = None
    docs_artifact_root: str | None = None  # Default: ``<cwd>/.ccx/docs``
    # Optional explicit destination for the synthesized Markdown. When
    # set, the synthesizer writes here verbatim (relative paths are
    # resolved under ``cwd``) instead of the auto-generated
    # ``<docs_artifact_root>/<auto-id>.md`` path. Useful when the user
    # asks for the report to land in a known location like
    # ``core/foo/docs/upgrade_advice.md`` so they can commit it.
    output_path: str | None = None
    write_artifact: bool = True
    has_tools: bool = True
    max_tool_rounds: int | None = None
    mode_name: str = "doc"

    # Synthesizer SUBSTANTIATION hardening (see ``_doc_substantiate_enabled``).
    # ``None`` (default) ⇒ defer to the ``CCX_DOC_SUBSTANTIATE`` env var (off by
    # default → byte-identical synthesis prompt). ``True``/``False`` force it on
    # or off regardless of env (used by tests and the FP-study harness).
    substantiate: bool | None = None

    # Lower bound: even at parallelism=2 the planner still spawns 1 synth
    # node, so total spawned = N investigators + 1 synth.
    MIN_PARALLELISM_FOR_FANOUT: int = 2

    # Hard ceiling on tool rounds for an investigator turn. Set when the
    # caller doesn't specify ``max_tool_rounds``. Prevents runaway loops
    # while leaving enough headroom for Stage 1 enumeration + ≥3 reads
    # in Stage 2 + targeted greps + Stage 3 emit.
    # 60 = ~20 rounds budget × 3-stage workflow safety factor.
    # Empirically, even a small project review (read 5–10 files +
    # targeted greps + emit) consumes 50+ rounds with a reasoning
    # model; 36 was too tight and routinely tripped the cap.
    # Override via ``max_tool_rounds`` on the runner or via
    # ``cc_max_tool_rounds`` in build_runtime.
    INVESTIGATOR_DEFAULT_MAX_ROUNDS: int = 60

    # When ``parallelism=1`` the single investigator has no
    # competition for the tool budget. Multiply the round cap so
    # depth-mode runs can actually go deeper. ``2x`` of 60 = 120
    # rounds, plenty for ≥10 file_reads + targeted greps + emit
    # in a non-trivial codebase analysis.
    SINGLE_INVESTIGATOR_ROUND_MULTIPLIER: int = 2

    # Per-investigator wall-clock cap (seconds). When a single LLM
    # HTTP call hangs (e.g. the OpenAI/DeepSeek client retries with
    # exponential backoff on a borderline-too-large prompt), the
    # in-loop ``max_turn_timeout_seconds`` check in query_engine
    # never fires because no event is yielded — the iteration is
    # blocked inside ``llm_adapter.complete_with_messages``. A
    # parallel doc-mode run with 4 investigators can then sit
    # 3+ hours waiting on a single stuck one, blocking synth.
    # Wrap each investigator's ``engine.submit_message`` iteration
    # in ``asyncio.wait_for`` with this timeout so a hung investigator
    # gets cancelled and the dim is marked non-ok (synth still runs).
    # 2700 = 45 min. Deepseek-reasoner generates slowly; 30 min was hit
    # in practice by legitimate investigations that needed retry, forcing
    # the dim to lose its first attempt to wall-clock cancel and only
    # produce findings on the second attempt. 45 min gives reasoning
    # models enough headroom while still cancelling truly hung sessions.
    INVESTIGATOR_WALL_CLOCK_TIMEOUT_S: float = 2700.0

    # When an investigator returns ``shallow`` / ``empty`` /
    # ``unparseable`` (see ``_classify_investigator_outcome``),
    # re-invoke the turn this many times before giving up. Each retry
    # uses progressively more directive feedback (see
    # ``_build_retry_feedback``); the last attempt receives a concrete
    # list of files to read so the LLM cannot dodge.
    # ``0`` disables retries (then the hard-minimum below applies).
    # 1 retry = 1 initial + 1 retry = 2 total attempts. With
    # parallelism=4 and per-attempt cap=60 rounds, 2 attempts per
    # investigator is already 8 LLM turns; a 3rd attempt rarely
    # changes the verdict once status is shallow/empty after the 2nd.
    INVESTIGATOR_SHALLOW_RETRY_LIMIT: int = 1

    # Hard minimum number of LLM-level attempts the investigator loop
    # will make when the previous attempt didn't return ``ok``. This
    # provides a floor independent of ``INVESTIGATOR_SHALLOW_RETRY_LIMIT``
    # so retry behavior is robust to stale config / dataclass slots
    # quirks where the limit is mistakenly set to 0 or 1. The loop
    # always attempts at least this many times before giving up; the
    # only earlier exits are status=ok and explicit no-improvement
    # early-exit (which itself requires attempt > 0).
    INVESTIGATOR_HARD_MIN_ATTEMPTS: int = 3

    # Reasoning-model JSON-emission salvage (forced continuation).
    # Thinking models (deepseek-reasoner / ``SimpleDeepSeekClientReasoning``)
    # routinely finish the investigator tool loop, announce "let me now compile
    # the final JSON", and then EOS WITHOUT emitting it — the terminal turn is a
    # content-free promise, so the response classifies ``unparseable`` and a full
    # retry just reproduces the same EOS (observed: 15 unparseable dims across a
    # reasoning-client doc run on futures_rec_v1; only the one dim that happened
    # to emit JSON survived). When the terminal turn carries NO parseable JSON
    # but the investigation really happened (>=1 ``file_read``), fire ONE forced
    # continuation on the SAME engine: all the gathered evidence is still in the
    # conversation, so the model only has to emit the JSON it already promised.
    # This is strictly better than the post-hoc, evidence-free prose-to-JSON
    # salvage (``_try_convert_prose_to_findings``), which can't recover a
    # content-free preamble. For non-reasoning models the terminal turn already
    # contains JSON, so this never fires (zero cost on the healthy path). Any
    # error in the continuation is swallowed and the original terminal text is
    # kept, so the salvage can never make a dimension worse. Set False to disable.
    INVESTIGATOR_FORCE_JSON_CONTINUATION: bool = True
    # Tool-round budget for the forced JSON-emission continuation. It should need
    # NO tools (just emit the JSON it gathered); a tiny budget bounds the cost.
    INVESTIGATOR_FORCE_JSON_MAX_ROUNDS: int = 2
    INVESTIGATOR_FORCE_JSON_NUDGE: str = (
        "STOP. Do not investigate further and do not narrate. You have already "
        "gathered the evidence above. Output your findings for this dimension NOW "
        "as a SINGLE JSON object matching the required schema exactly. Your entire "
        "reply must begin with '{' and end with '}' — no reasoning, no preamble, "
        "no markdown code fences, and no text before or after the JSON object."
    )

    # Synth-side gate: when too many investigators end up shallow /
    # empty / unparseable, falling through to a full synthesizer call
    # produces a low-quality "padded" report. If the share of dimensions
    # with status != ``ok`` exceeds this threshold, synth writes a
    # short honest stub instead of attempting a full report. Set to
    # ``1.0`` to disable the gate (always run full synth).
    # 0.8 (vs the previous 0.6) lets a 1-good-of-4 run still produce a
    # full synthesised report — the LLM's cross-dimensional weaving is
    # more valuable than a defensive stub when there is at least some
    # substantive evidence to work with. The stub format already
    # preserves successful-dim findings, so worst case we fall back to
    # the same content; best case the LLM produces a coherent report.
    SYNTH_DEGENERATE_FAILURE_RATIO: float = 0.8

    # Maximum number of consecutive ``self.llm(...)`` calls allowed when
    # synthesizing the final Markdown report. Most LLM backends cap a
    # single completion at 4-8K output tokens; a 5-dimension review
    # easily exceeds that. When the synthesizer's output looks
    # truncated (see ``_looks_truncated``) we ask the LLM to continue
    # and concatenate. ``1`` disables continuation.
    SYNTH_MAX_CONTINUATIONS: int = 5

    # Heuristic minimum bytes for a "complete" synth output. Below this
    # we treat the output as suspicious regardless of how it ends.
    SYNTH_MIN_BYTES_PER_FINDING: int = 350
    SYNTH_MIN_BYTES_BASE: int = 800

    # Tool-round cap for the structural survey that runs at the start
    # of the planner phase. Survey is intentionally fast — it just
    # needs to enumerate files and read 1-3 orientation files. A
    # higher cap would just let the LLM start doing real
    # investigation, which is the investigators' job.
    SURVEYOR_MAX_ROUNDS: int = 12
    SURVEYOR_WALL_CLOCK_TIMEOUT_S: float = 180.0

    # When the surveyor reports ``complexity_signal``, planner uses it
    # to scale ``max_dimensions``. The mapping is bounded by
    # ``parallelism`` so the user's budget always wins.
    SURVEY_COMPLEXITY_TO_DIMS: dict[str, int] = field(
        default_factory=lambda: {"simple": 3, "medium": 4, "complex": 6}
    )

    # Per-investigator tool-round cap scaled by the surveyor's
    # ``complexity_signal``. ``INVESTIGATOR_DEFAULT_MAX_ROUNDS`` (60) was
    # tuned for a *reasoning* model reviewing a non-trivial project; on a
    # genuinely simple target (a couple of files) investigators otherwise
    # expand to fill the whole 60-round budget (Parkinson's law — observed
    # 62 rounds / 54 greps to review one 460-line file), wasting tokens and
    # wall-clock. Scaling the cap down for simple/medium targets forces
    # earlier convergence with no quality loss; ``complex`` and unknown /
    # missing signals keep the full default so large reviews never get
    # starved (zero regression for the futures_rec_v1-scale runs). The
    # ``parallelism=1`` depth multiplier still applies on top.
    SURVEY_COMPLEXITY_TO_ROUNDS: dict[str, int] = field(
        default_factory=lambda: {"simple": 30, "medium": 45, "complex": 60}
    )

    def run(self, invocation: SubagentInvocation) -> SubagentResult:
        phase = str(invocation.metadata.get("ccx_doc_phase") or "root")
        if phase == "root":
            return self._run_planner(invocation)
        if phase == "investigate":
            if not self.has_tools:
                return self._run_investigator(invocation)
            context = None
            token = None
            if self.llm_provider is not None and hasattr(self.llm_provider, "begin_invocation"):
                context, token = self.llm_provider.begin_invocation(
                    mode=self.mode_name,
                    metadata=invocation.metadata,
                )
            try:
                result = self._run_investigator(invocation)
            finally:
                if context is not None and context.cost_accumulator:
                    cost_usd = sum(context.cost_accumulator)
                    _emit_provider_cost_event(
                        mode=self.mode_name,
                        cost_usd=cost_usd,
                        call_count=len(context.cost_accumulator),
                        tokens=sum(context.token_accumulator),
                    )
                    report_cost_to_budget(
                        cost_usd=cost_usd,
                        tokens=sum(context.token_accumulator),
                    )
                if token is not None and hasattr(self.llm_provider, "end_invocation"):
                    self.llm_provider.end_invocation(token)
            if context is not None and context.needs_accumulator:
                result = _apply_needs_model_marker_result(
                    result, context.needs_accumulator[-1],
                )
            return result
        if phase == "synthesize":
            return self._run_synthesizer(invocation)
        raise ValueError(
            f"DocModeRunner: unknown ccx_doc_phase={phase!r}; "
            "expected 'root', 'investigate', or 'synthesize'"
        )

    # ------------------------------------------------------------------ #
    # Phase 1 — planner
    # ------------------------------------------------------------------ #

    def _run_planner(self, invocation: SubagentInvocation) -> SubagentResult:
        # No tools at all → there's nothing to fan out (the
        # investigator's three-stage workflow needs Read/Grep/Glob).
        # Drop to a single-LLM single-shot.
        if not self.has_tools:
            return self._run_single_shot(
                invocation, fallback_reason="lite_no_tools",
            )

        # parallelism < 2 with tools → the user explicitly asked for
        # depth-over-breadth. Spawn ONE comprehensive investigator
        # (full goal as focus) plus a synthesizer; skip decomposition.
        # The single investigator gets the entire tool budget — no
        # cross-investigator competition. This is the right shape for:
        #   * tight tool budgets / rate limits
        #   * focused single-subdir reviews ("review services/")
        #   * follow-up runs after a degenerate fan-out failed
        if self.parallelism < self.MIN_PARALLELISM_FOR_FANOUT:
            return self._run_single_investigator_path(invocation)

        # ── Phase 0: structural survey ──
        # Run a fast, tool-enabled survey FIRST so the decomposer can
        # pick dimensions adapted to the actual project shape instead
        # of falling back to generic templates ("architecture",
        # "performance", ...). The survey output is also propagated
        # into every investigator's metadata so all of them share one
        # mental model of the project.
        survey = self._run_surveyor_with_tracking(invocation)
        survey_block = _format_survey_for_prompt(survey)

        outline_text = self._maybe_outline_text(deep=True)
        path_block = self._paths_context_block(invocation.goal)
        system = _load_decompose_system(self.language)
        # Adjust max_dims using the survey's complexity signal (still
        # bounded by ``self.parallelism`` so user budget wins).
        max_dims = self._resolve_max_dimensions(survey)
        user_parts = [
            f"## Goal\n{invocation.goal}",
            f"## Maximum dimensions\n{max_dims}",
        ]
        if survey_block:
            user_parts.append(survey_block)
            user_parts.append(
                "**Use the survey above to pick dimensions adapted to "
                "THIS project**. If the project has named layers (e.g. "
                "`services/` / `repositories/` / `governance/`), prefer "
                "dimensions that name them concretely over generic "
                "templates. If the project has dedicated docs, "
                "consider a 'docs vs code consistency' dimension. "
                "Each dimension's `scope_hint` should point at the "
                "concrete sub-directory (or file glob) the dimension "
                "covers — not the whole repo."
            )
        if path_block:
            user_parts.append(path_block)
        if outline_text:
            user_parts.append(
                "## Repository Outline (PARTIAL — truncated; trust paths from the goal even if missing here)\n"
                f"```\n{outline_text}\n```"
            )
        user_parts.append(
            'Respond with: {"dimensions": [...], "rationale": "..."}. '
            "Echo any paths from the goal into each dimension's scope_hint."
        )
        user = "\n\n".join(user_parts)

        response = text_of(
            self.llm(system=system, user=user, purpose="doc_decompose")
        )
        dimensions = _parse_decompose_response(response)
        if not dimensions:
            logger.info(
                "doc planner: LLM returned no dimensions, falling back to "
                "single-shot synthesis (raw=%r)",
                (response or "")[:200],
            )
            return self._run_single_shot(
                invocation, fallback_reason="empty_decomposition",
                survey=survey,
            )

        # Truncate to parallelism cap.
        dimensions = dimensions[:max_dims]
        run_id = uuid.uuid4().hex
        root_goal = invocation.goal

        # Survey is propagated to investigators via metadata so they
        # don't re-run it. Strip the noisy ``__meta__`` field before
        # passing.
        survey_for_meta = {k: v for k, v in survey.items() if k != "__meta__"}

        investigator_subtasks: list[SubagentInvocation] = []
        for dim in dimensions:
            investigator_subtasks.append(SubagentInvocation(
                goal=dim["focus"] or dim["title"],
                mode="doc",
                metadata={
                    "ccx_doc_phase": "investigate",
                    "ccx_doc_run_id": run_id,
                    "ccx_doc_dimension": {
                        "id": dim["id"],
                        "title": dim["title"],
                        "focus": dim["focus"],
                        "scope_hint": dim.get("scope_hint", ""),
                    },
                    "ccx_doc_root_goal": root_goal,
                    "ccx_doc_survey": survey_for_meta,
                    "ccx_parent_mode": "doc",
                },
            ))
        n_inv = len(investigator_subtasks)

        synthesizer_subtask = SubagentInvocation(
            goal=root_goal,
            mode="doc",
            metadata={
                "ccx_doc_phase": "synthesize",
                "ccx_doc_run_id": run_id,
                "ccx_doc_root_goal": root_goal,
                "ccx_doc_survey": survey_for_meta,
                "ccx_parent_mode": "doc",
                "ccx_depends_on": list(range(n_inv)),
                "ccx_doc_dimension_count": n_inv,
            },
        )

        return SubagentResult(
            final_text="",
            subtasks=investigator_subtasks + [synthesizer_subtask],
            sequential=False,
            extras={
                "ccx_doc_phase": "root",
                "ccx_doc_run_id": run_id,
                "dimensions": dimensions,
                "goal": root_goal,
                "survey": survey,
            },
        )

    # ------------------------------------------------------------------ #
    # Phase 1b — single-shot fallback (parallelism=1 or lite)
    # ------------------------------------------------------------------ #

    def _run_single_investigator_path(
        self, invocation: SubagentInvocation,
    ) -> SubagentResult:
        """Spawn one comprehensive investigator + synth when
        ``parallelism=1``.

        This is NOT the single_shot fallback (which uses a synth-style
        prompt and skips the three-stage workflow). The single-
        investigator path keeps the depth rule, counter check, and
        starter actions engaged — just with one investigator instead
        of N. The user wins back the whole tool budget per turn.

        Survey still runs (it's the cheapest way to make the
        investigator's first move informed). Decomposition is skipped
        since there's only one dimension: the user's goal.
        """
        survey = self._run_surveyor_with_tracking(invocation)
        survey_for_meta = {k: v for k, v in survey.items() if k != "__meta__"}

        run_id = uuid.uuid4().hex
        root_goal = invocation.goal

        # Pick the dimension's scope_hint from the user's named path
        # (if any) — this is what the file enumeration / starter
        # actions will use.
        named_paths = extract_path_tokens(root_goal)
        scope_hint = ",".join(named_paths) if named_paths else ""

        # Use a structurally-neutral title so the comprehensive
        # investigator isn't pigeon-holed into one starter recipe.
        # The focus carries the user's full intent.
        dimension = {
            "id": "comprehensive",
            "title": "Comprehensive review",
            "focus": root_goal,
            "scope_hint": scope_hint,
        }

        investigator_subtask = SubagentInvocation(
            goal=root_goal,
            mode="doc",
            metadata={
                "ccx_doc_phase": "investigate",
                "ccx_doc_run_id": run_id,
                "ccx_doc_dimension": dimension,
                "ccx_doc_root_goal": root_goal,
                "ccx_doc_survey": survey_for_meta,
                "ccx_doc_single_investigator": True,
                "ccx_parent_mode": "doc",
            },
        )
        synthesizer_subtask = SubagentInvocation(
            goal=root_goal,
            mode="doc",
            metadata={
                "ccx_doc_phase": "synthesize",
                "ccx_doc_run_id": run_id,
                "ccx_doc_root_goal": root_goal,
                "ccx_doc_survey": survey_for_meta,
                "ccx_doc_single_investigator": True,
                "ccx_parent_mode": "doc",
                "ccx_depends_on": [0],
                "ccx_doc_dimension_count": 1,
            },
        )

        return SubagentResult(
            final_text="",
            subtasks=[investigator_subtask, synthesizer_subtask],
            sequential=False,
            extras={
                "ccx_doc_phase": "root",
                "ccx_doc_run_id": run_id,
                "single_investigator": True,
                "dimensions": [dimension],
                "goal": root_goal,
                "survey": survey,
            },
        )

    def _run_single_shot(
        self,
        invocation: SubagentInvocation,
        *,
        fallback_reason: str,
        survey: dict[str, Any] | None = None,
    ) -> SubagentResult:
        # When tools are available, do NOT degrade to a permissive
        # ``system.doc_mode`` single-call: that prompt invites the LLM
        # to "produce a Markdown document" without enforcing the
        # ≥3 file_read depth rule, so the LLM stops after one tool
        # call and emits half-baked text. Redirect to the single-
        # investigator path which uses the investigator three-stage
        # workflow + counter check + retry. The only legitimate
        # single-shot caller now is the lite (no-tools) path, which
        # genuinely cannot do better than one LLM call.
        if self.has_tools and fallback_reason != "lite_no_tools":
            logger.info(
                "doc single_shot redirected to single-investigator path "
                "(has_tools=True, fallback_reason=%s) so depth rules "
                "are enforced",
                fallback_reason,
            )
            return self._run_single_investigator_path(invocation)

        outline_text = self._maybe_outline_text(deep=True)
        path_block = self._paths_context_block(invocation.goal)
        survey_block = _format_survey_for_prompt(survey or {})
        system = load_cc_system_prompt("doc_mode", self.language)
        user_parts = [f"## Goal\n{invocation.goal}"]
        if survey_block:
            user_parts.append(survey_block)
        if path_block:
            user_parts.append(path_block)
        if outline_text:
            user_parts.append(
                "## Repository Outline (PARTIAL — truncated; trust paths from the goal)\n"
                f"```\n{outline_text}\n```"
            )
        user_parts.append(
            "Produce a structured Markdown document directly answering the "
            "goal. Use file:line citations for any concrete claim."
        )
        user = "\n\n".join(user_parts)

        # has_tools=True is now redirected at the top of this method,
        # so reaching here means lite (no tools) — single LLM call.
        markdown = text_of(self.llm(
            system=system, user=user, purpose="doc_singleshot",
        ))

        artifact_path = self._maybe_write_artifact(
            goal=invocation.goal, markdown=markdown,
        )
        return SubagentResult(
            final_text=markdown.strip(),
            subtasks=[],
            sequential=False,
            extras={
                "ccx_doc_phase": "single_shot",
                "ccx_doc_fallback_reason": fallback_reason,
                "goal": invocation.goal,
                "artifact_path": artifact_path,
            },
        )

    # ------------------------------------------------------------------ #
    # Phase 2 — investigator
    # ------------------------------------------------------------------ #

    def _run_investigator(self, invocation: SubagentInvocation) -> SubagentResult:
        dim = dict(invocation.metadata.get("ccx_doc_dimension") or {})
        run_id = str(invocation.metadata.get("ccx_doc_run_id") or "")
        root_goal = str(invocation.metadata.get("ccx_doc_root_goal") or "")
        survey = dict(invocation.metadata.get("ccx_doc_survey") or {})

        # Lite fallback: even though planner refuses to fan out under
        # has_tools=False, an external caller could still spawn an
        # investigate node directly. Be defensive.
        if not self.has_tools:
            response = text_of(self.llm(
                system=_load_investigator_system(self.language),
                user=self._build_investigator_user_prompt(
                    invocation.goal, root_goal, dim, survey=survey,
                ),
                purpose="doc_investigate_lite",
            ))
            findings = _parse_investigator_response(response, dim)
            findings["status"] = _classify_investigator_outcome(
                final_text=response,
                findings=findings,
                tool_call_count=0,
                file_read_count=0,
            )
            findings["tool_call_count"] = 0
            findings["file_read_count"] = 0
            self._push_findings(run_id, dim.get("id") or "", findings)
            return SubagentResult(
                final_text=findings.get("summary", ""),
                subtasks=[],
                extras={
                    "ccx_doc_phase": "investigate",
                    "ccx_doc_run_id": run_id,
                    "dimension": dim,
                    "findings": findings,
                    "via": "ccx_doc_lite",
                },
            )

        from ..agents.cc_agent import _run_in_fresh_loop
        return _run_in_fresh_loop(self._run_investigator_async(invocation))

    async def _run_investigator_async(
        self, invocation: SubagentInvocation,
    ) -> SubagentResult:
        if self.llm_provider is None or self.cc_config is None:
            raise RuntimeError(
                "DocModeRunner investigator with has_tools=True requires "
                "`cc_config` and `llm_provider`."
            )

        dim = dict(invocation.metadata.get("ccx_doc_dimension") or {})
        run_id = str(invocation.metadata.get("ccx_doc_run_id") or "")
        root_goal = str(invocation.metadata.get("ccx_doc_root_goal") or "")
        survey = dict(invocation.metadata.get("ccx_doc_survey") or {})

        # Outer loop: re-invoke up to ``INVESTIGATOR_SHALLOW_RETRY_LIMIT``
        # times when the LLM returned shallow/empty output. The retry
        # carries explicit feedback so the model knows what to do
        # differently — most importantly "you must read at least 3 files
        # before emitting JSON".
        # Pre-compute scope files + dimension-relevant suggestions so
        # the orchestrator can prepend a concrete file list to retry
        # feedback (and so the suggester only runs once).
        scope_files = self._resolve_scope_files(dim)
        suggested_files = self._suggest_files_for_dimension(dim, scope_files)

        attempts: list[dict[str, Any]] = []
        retry_feedback = ""
        last_status: str | None = None
        last_file_read_count: int = -1
        # Hard minimum applies ONLY when ``INVESTIGATOR_SHALLOW_RETRY_LIMIT``
        # is degenerate (<=0, e.g. stale config / older dataclass
        # default). When the limit is explicitly set to a positive
        # value, trust it — the user/runner knowingly chose this retry
        # budget and forcing extra attempts wastes 1 full-budget LLM
        # call per dimension (60+ tool rounds × ``parallelism`` dims).
        # Previous behaviour used ``max(HARD_MIN, LIMIT+1)`` which
        # always made HARD_MIN win when LIMIT was set to 1 — defeating
        # the point of the config.
        if self.INVESTIGATOR_SHALLOW_RETRY_LIMIT <= 0:
            max_attempts = self.INVESTIGATOR_HARD_MIN_ATTEMPTS
        else:
            max_attempts = self.INVESTIGATOR_SHALLOW_RETRY_LIMIT + 1
        for attempt in range(max_attempts):
            invoke_error: Exception | None = None
            try:
                (
                    final_text,
                    tool_call_count,
                    file_read_count,
                    tool_ledger,
                ) = await self._invoke_investigator_once(
                    dim=dim, root_goal=root_goal, focus=invocation.goal,
                    retry_feedback=retry_feedback, survey=survey,
                )
            except Exception as exc:  # noqa: BLE001 — deliberate broad catch
                # The wall-clock timeout is handled INSIDE
                # ``_invoke_investigator_once``; everything else —
                # provider auth/network errors, engine build failures
                # (cc's query_engine.submit_message emits turn_failed
                # and then RE-RAISES) — used to escape this runner,
                # fail the v5 node, and after the dispatcher's retries
                # the ABANDONED node cascade-SKIPped the dependent
                # synth node. That threw away every OTHER dimension's
                # findings already sitting in the FindingsCollector and
                # the run produced no report at all. Instead, treat the
                # error like the timeout path treats a hung turn: mark
                # THIS dimension non-ok and keep going so the synth
                # still runs on whatever the surviving dimensions
                # collected. (CancelledError is BaseException and still
                # propagates.)
                invoke_error = exc
                final_text = ""
                tool_call_count = 0
                file_read_count = 0
                tool_ledger = []
                logger.warning(
                    "doc investigator: dim=%r attempt %d/%d raised "
                    "%s: %s — marking dimension non-ok and continuing "
                    "(synth will still run with the other dimensions).",
                    dim.get("id") or dim.get("title") or "?",
                    attempt + 1, max_attempts,
                    type(exc).__name__, exc,
                    exc_info=True,
                )
            if invoke_error is not None:
                error_summary = (
                    f"(investigator error) "
                    f"{type(invoke_error).__name__}: {invoke_error}"
                )
                status = "error"
                findings = {
                    "dimension_id": str(dim.get("id") or ""),
                    "dimension_title": str(dim.get("title") or ""),
                    "summary": error_summary,
                    "evidence": [],
                    "issues": [],
                    "confidence": "low",
                    "error": error_summary,
                }
                xml_protocol_violation = False
                text_plan_stalled = False
                prose_converted = False
            else:
                findings = _parse_investigator_response(final_text, dim)
                status = _classify_investigator_outcome(
                    final_text=final_text,
                    findings=findings,
                    tool_call_count=tool_call_count,
                    file_read_count=file_read_count,
                )
                xml_protocol_violation = _detect_xml_tool_markers(final_text)
                text_plan_stalled = _detect_text_tool_plans(final_text)
                prose_converted = False
                # Prose-to-JSON salvage. When the LLM produced a
                # substantive prose report instead of the strict JSON
                # shape (status=unparseable), don't waste 1-2 more retry
                # attempts — the work is already done in the prose, we
                # just need to restructure it. A single LLM call (no
                # tools) converts the prose into the schema. We only fire
                # this when the prose contains real file:line citations,
                # otherwise it's not worth the call.
                if status == "unparseable" and not xml_protocol_violation:
                    converted = self._try_convert_prose_to_findings(
                        prose=final_text, dim=dim,
                    )
                    if converted is not None:
                        logger.info(
                            "doc investigator: prose-to-JSON conversion "
                            "succeeded for dim=%r — %d evidence + %d issues "
                            "extracted from prose",
                            dim.get("id") or dim.get("title") or "?",
                            len(converted.get("evidence") or []),
                            len(converted.get("issues") or []),
                        )
                        findings = converted
                        prose_converted = True
                        # Re-classify with the converted findings.
                        # ``final_text`` is irrelevant for the re-classify
                        # — the parsed structure is what matters now.
                        status = _classify_investigator_outcome(
                            final_text="(converted from prose)",
                            findings=findings,
                            tool_call_count=tool_call_count,
                            file_read_count=file_read_count,
                        )
                        # Track that this finding was salvaged so the
                        # synth can flag it (mild caveat: converted
                        # findings are slightly less reliable than native
                        # JSON because the LLM might have over-interpreted
                        # the prose).
                        findings["confidence"] = "medium" if findings.get("confidence", "low") == "high" else findings.get("confidence", "low")
            findings["status"] = status
            findings["tool_call_count"] = tool_call_count
            findings["file_read_count"] = file_read_count
            findings["text_plan_stalled"] = text_plan_stalled
            findings["xml_protocol_violation"] = xml_protocol_violation
            findings["prose_converted"] = prose_converted
            findings["attempts"] = attempt + 1
            attempts.append({
                "status": status,
                "tool_call_count": tool_call_count,
                "file_read_count": file_read_count,
                "text_plan_stalled": text_plan_stalled,
                "xml_protocol_violation": xml_protocol_violation,
                "prose_converted": prose_converted,
                "summary_preview": (findings.get("summary") or "")[:160],
            })
            # Visible diagnostic: surface what the LLM actually
            # emitted on this attempt so we can see why retry did /
            # didn't fire. Especially important for the
            # "<tool_call>...</tool_call>" XML failure mode where the
            # raw text reveals the protocol violation directly.
            preview = (final_text or "").strip().replace("\n", " ")[:240]
            logger.info(
                "doc investigator attempt %d/%d: dim=%r status=%s "
                "tool_calls=%d file_reads=%d xml_violation=%s "
                "text_plan=%s | final_text_preview=%r",
                attempt + 1, max_attempts,
                dim.get("id") or dim.get("title") or "?",
                status, tool_call_count, file_read_count,
                xml_protocol_violation, text_plan_stalled, preview,
            )
            if status == "ok":
                logger.info(
                    "doc investigator: attempt %d returned ok, exiting "
                    "retry loop after %d attempt(s)",
                    attempt + 1, attempt + 1,
                )
                break
            # Retry budget reached. ``max_attempts`` already includes
            # the hard minimum, so this is the legitimate stop.
            if attempt + 1 >= max_attempts:
                logger.info(
                    "doc investigator: retry budget exhausted (%d "
                    "attempts) for dim=%r — accepting last status=%s",
                    max_attempts,
                    dim.get("id") or dim.get("title") or "?", status,
                )
                break
            # Early-exit when retries clearly aren't helping. If the
            # previous attempt was also non-ok with the SAME status and
            # the file_read_count didn't increase, the LLM is stuck —
            # don't burn another LLM turn for the same result.
            if (
                attempt > 0
                and last_status == status
                and file_read_count <= last_file_read_count
            ):
                logger.info(
                    "doc investigator early-exit: retry attempt %d "
                    "produced no improvement (status=%s, file_reads "
                    "%d → %d). Accepting result.",
                    attempt + 1, status, last_file_read_count,
                    file_read_count,
                )
                break
            last_status = status
            last_file_read_count = file_read_count
            retry_feedback = self._build_retry_feedback(
                status=status, findings=findings,
                tool_call_count=tool_call_count,
                file_read_count=file_read_count,
                attempt_number=attempt + 1,
                suggested_files=suggested_files,
                tool_ledger=tool_ledger,
            )
            logger.info(
                "doc investigator queuing retry %d/%d: status=%s "
                "tool_calls=%d file_reads=%d (suggested_files=%d)",
                attempt + 2, max_attempts,
                status, tool_call_count, file_read_count,
                len(suggested_files or []),
            )

        # Post-loop diagnostic — surface why the loop terminated. If
        # the final status is non-ok and we ran fewer than the hard
        # minimum, that's a real bug (something broke the loop early
        # which our break conditions shouldn't have).
        final_status = (
            attempts[-1]["status"] if attempts else "(no attempts)"
        )
        actually_ran = len(attempts)
        if final_status != "ok":
            logger.warning(
                "doc investigator dim=%r exited with status=%s after "
                "%d attempt(s) (max=%d, hard_min=%d, retry_limit=%d). "
                "Findings will be marked non-ok in the synth.",
                dim.get("id") or dim.get("title") or "?",
                final_status, actually_ran, max_attempts,
                self.INVESTIGATOR_HARD_MIN_ATTEMPTS,
                self.INVESTIGATOR_SHALLOW_RETRY_LIMIT,
            )
            # Only warn about violating the hard minimum when the
            # retry limit is degenerate. When the limit is explicitly
            # set to a positive value, ``actually_ran`` may legitimately
            # be ``LIMIT + 1`` which is less than HARD_MIN — that is
            # the intended behaviour, not a config bug.
            if (
                self.INVESTIGATOR_SHALLOW_RETRY_LIMIT <= 0
                and actually_ran < self.INVESTIGATOR_HARD_MIN_ATTEMPTS
            ):
                logger.warning(
                    "doc investigator: WARNING — only %d attempt(s) ran "
                    "but hard minimum is %d. Check INVESTIGATOR_*_LIMIT "
                    "config and any early-exit logging above.",
                    actually_ran, self.INVESTIGATOR_HARD_MIN_ATTEMPTS,
                )

        findings["attempt_log"] = attempts
        self._push_findings(run_id, dim.get("id") or "", findings)
        return SubagentResult(
            final_text=findings.get("summary", ""),
            subtasks=[],
            sequential=False,
            extras={
                "ccx_doc_phase": "investigate",
                "ccx_doc_run_id": run_id,
                "dimension": dim,
                "findings": findings,
                "tool_call_count": tool_call_count,
                "attempts": attempts,
                "via": "ccx_doc_with_tools",
            },
        )

    def _try_convert_prose_to_findings(
        self,
        *,
        prose: str,
        dim: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Last-ditch fallback when the investigator returned a
        substantive prose report instead of JSON.

        Runs a single LLM call (no tools — just structural rewrite)
        that takes the prose and re-emits it in the strict JSON
        schema. This preserves the work the investigator already did
        (file:line citations, identified issues) instead of
        discarding it on retry. Cheaper than another full
        investigation turn.

        Returns the parsed findings dict on success, or ``None`` when
        the prose is too short / lacks citations / conversion failed.
        Caller can re-classify with the converted findings to see if
        they now pass the depth rule.
        """
        if not prose:
            return None
        text = prose.strip()
        # Require enough content + at least one file path AND at least
        # one line-number indicator (anywhere in the prose) to justify
        # the conversion call. Avoids spending tokens on pure
        # placeholder / "I gave up" prose. 150 char floor catches the
        # "stub LLM gave up after 1 sentence" case while letting real
        # multi-finding reports (≥3 citations + intro) qualify.
        #
        # We split path detection from line-number detection because
        # real reports use varied citation styles depending on
        # language: ``foo.py:42`` (canonical), ``foo.py 第 21-29 行``
        # (Chinese), ``foo.py L21-L29`` (GitHub-style), ``(line 42)``
        # (English prose). Requiring colon-form misses most Chinese
        # output and was the primary reason prose_converted didn't
        # fire on the user's stock_rec_v3 run.
        if len(text) < 150:
            return None
        if not _has_file_with_line_evidence(text):
            return None
        system = _load_prose_to_json_system(self.language)
        user = (
            f"## Prose investigation report (input)\n\n{text[:6000]}"
            + ("\n\n[truncated]" if len(text) > 6000 else "")
            + f"\n\n## Dimension being structured\n"
            f"- title: {dim.get('title') or '(untitled)'}\n"
            f"- focus: {dim.get('focus') or '(unspecified)'}\n\n"
            "Convert the prose above into the strict JSON schema. "
            "Do NOT add new content. Output JSON only."
        )
        try:
            response = text_of(self.llm(
                system=system, user=user, purpose="doc_prose_to_json",
            ))
        except Exception as exc:  # noqa: BLE001
            logger.warning("prose-to-JSON conversion failed: %s", exc)
            return None
        converted = _parse_investigator_response(response, dim)
        # The conversion is only useful if it produced concrete
        # evidence or issues. If the LLM also returned an empty
        # shell, treat as no-op.
        if not (converted.get("evidence") or converted.get("issues")):
            return None
        return converted

    async def _invoke_investigator_once(
        self,
        *,
        dim: dict[str, Any],
        root_goal: str,
        focus: str,
        retry_feedback: str,
        survey: dict[str, Any] | None = None,
    ) -> tuple[str, int, int, list[dict[str, Any]]]:
        """Single investigator turn: build a fresh cc QueryEngine with
        the read-only registry, frame the prompt (with optional retry
        feedback), and drive the engine. Returns ``(final_text,
        tool_call_count, file_read_count, tool_ledger)``.

        ``tool_ledger`` is a deduplicated list of ``file_read`` /
        ``grep`` / ``glob`` calls made in this attempt, used by the
        retry path to tell the next attempt "don't repeat these".

        Each attempt builds a fresh engine so the retry doesn't inherit
        a poisoned tool-orchestrator state.
        """
        from core.cc.runtime import build_default_query_engine

        engine = build_default_query_engine(
            cwd=self.cwd,
            config=self.cc_config,
            llm_client_provider=self.llm_provider,
        )
        removed, kept = restrict_tool_registry(engine)
        logger.debug(
            "doc investigator: filtered cc registry, removed %d "
            "non-read-only tools (kept %s)",
            removed, kept,
        )

        system = _load_investigator_system(self.language)
        user = self._build_investigator_user_prompt(
            focus, root_goal, dim,
            retry_feedback=retry_feedback,
            survey=survey,
        )
        framed = f"<system>\n{system}\n</system>\n\n{user}"

        final_text = ""
        tool_call_count = 0
        file_read_count = 0
        tool_ledger: list[dict[str, Any]] = []
        # tool_use_id -> identity string ("tool\x00key\x00repr(args)"),
        # used to match the deferred tool_result back to the ledger
        # entry recorded when the tool_use fired.
        pending_ledger: dict[str, str] = {}
        rounds_cap = self._effective_investigator_max_rounds(survey)
        # Observability: surface the per-investigator round cap so
        # users can confirm ``max_tool_rounds`` propagated end-to-end.
        # Banner at task/deep/ccx.py prints the predicted effective
        # value at run start; this log emits the value the runner
        # actually enforces per investigator (post-resolution).
        logger.info(
            "doc investigator: dim=%r rounds_cap=%d "
            "wall_clock_timeout_s=%.0f "
            "(self.max_tool_rounds=%r, default=%d, parallelism=%d)",
            dim.get("id") or dim.get("title") or "?",
            rounds_cap, self.INVESTIGATOR_WALL_CLOCK_TIMEOUT_S,
            self.max_tool_rounds,
            self.INVESTIGATOR_DEFAULT_MAX_ROUNDS, self.parallelism,
        )

        # Wall-clock cap on the whole engine.submit_message iteration.
        # The in-loop ``max_turn_timeout_seconds`` check in
        # query_engine.py only fires after each yielded event, so a
        # hung llm_adapter call (e.g. openai client exponential
        # backoff retry storm) never trips it. Wrapping the iteration
        # in ``asyncio.wait_for`` cancels the underlying coroutine
        # after ``INVESTIGATOR_WALL_CLOCK_TIMEOUT_S`` and surfaces a
        # clean timeout that the retry loop / classifier handles as
        # ``empty``/``unparseable``. Without this, a single stuck
        # investigator can block the entire parallel doc run for
        # hours, starving synth.
        _investigator_started_at = time.monotonic()

        async def _drain() -> None:
            nonlocal final_text, tool_call_count, file_read_count
            async for event in engine.submit_message(
                framed, max_tool_rounds=rounds_cap,
            ):
                if event.event_type == "assistant_tool_use":
                    tool_call_count += 1
                    # Track file_read calls separately. The depth rule
                    # in the investigator system prompt is "≥3
                    # file_read calls"; total tool count alone would
                    # let an agent satisfy the budget by spamming
                    # grep without ever reading code.
                    payload = getattr(event, "payload", None) or {}
                    tool_name = ""
                    arguments: dict[str, Any] = {}
                    tc = payload.get("tool_call") if isinstance(payload, dict) else None
                    if isinstance(tc, dict):
                        tool_name = str(tc.get("tool_name") or "")
                        raw_args = tc.get("arguments")
                        if isinstance(raw_args, dict):
                            arguments = raw_args
                    msg_obj = getattr(event, "message", None)
                    if not tool_name:
                        tool_name = str(getattr(msg_obj, "tool_name", "") or "")
                    if not arguments and msg_obj is not None:
                        meta = getattr(msg_obj, "metadata", None) or {}
                        sp = meta.get("structured_payload") if isinstance(meta, dict) else None
                        if isinstance(sp, dict):
                            raw_args = sp.get("arguments")
                            if isinstance(raw_args, dict):
                                arguments = raw_args
                    if tool_name == "file_read":
                        file_read_count += 1
                    if tool_name in _LEDGER_TOOLS:
                        key = _ledger_key_arg(tool_name, arguments)
                        norm = _ledger_normalize_args(tool_name, arguments)
                        _ledger_record_call(
                            tool_ledger,
                            tool=tool_name, key=key, args=norm,
                        )
                        tool_use_id = (
                            str(getattr(msg_obj, "tool_use_id", "") or "")
                            if msg_obj is not None else ""
                        )
                        if tool_use_id:
                            pending_ledger[tool_use_id] = _ledger_identity(
                                tool_name, key, norm,
                            )
                elif event.event_type == "tool_result":
                    result_msg = getattr(event, "message", None)
                    if result_msg is not None:
                        tool_use_id = str(getattr(result_msg, "tool_use_id", "") or "")
                        meta = getattr(result_msg, "metadata", None) or {}
                        if tool_use_id and isinstance(meta, dict):
                            _ledger_attach_result(
                                tool_ledger,
                                tool_use_id=tool_use_id,
                                pending=pending_ledger,
                                metadata=meta,
                            )
                msg = getattr(event, "message", None)
                if msg is None:
                    continue
                if (
                    getattr(msg, "role", "") == "assistant"
                    and getattr(msg, "kind", "") == "assistant_text"
                ):
                    final_text = str(getattr(msg, "content", ""))

        try:
            await asyncio.wait_for(
                _drain(),
                timeout=self.INVESTIGATOR_WALL_CLOCK_TIMEOUT_S,
            )
            # Reasoning-model JSON-emission salvage. The model may have
            # finished investigating, announced it would emit the JSON, then
            # EOS'd without emitting it (content-free promise → unparseable).
            # The evidence is still in this engine's conversation, so one forced
            # continuation recovers the JSON it owes us. Guarded so it only
            # fires on the otherwise-failing path and never makes things worse.
            if (
                self.INVESTIGATOR_FORCE_JSON_CONTINUATION
                and (final_text or "").strip()
                and file_read_count >= 1
                and _robust_json_object(final_text) is None
                and not _detect_xml_tool_markers(final_text)
            ):
                try:
                    logger.info(
                        "doc investigator: dim=%r terminal turn had no "
                        "parseable JSON (len=%d, file_reads=%d) — forcing one "
                        "JSON-emission continuation on the same engine.",
                        dim.get("id") or dim.get("title") or "?",
                        len((final_text or "").strip()), file_read_count,
                    )
                    forced_text = ""

                    async def _drain_forced() -> None:
                        nonlocal forced_text, tool_call_count
                        async for event in engine.submit_message(
                            self.INVESTIGATOR_FORCE_JSON_NUDGE,
                            max_tool_rounds=self.INVESTIGATOR_FORCE_JSON_MAX_ROUNDS,
                        ):
                            if event.event_type == "assistant_tool_use":
                                tool_call_count += 1
                            fmsg = getattr(event, "message", None)
                            if fmsg is None:
                                continue
                            if (
                                getattr(fmsg, "role", "") == "assistant"
                                and getattr(fmsg, "kind", "") == "assistant_text"
                            ):
                                forced_text = str(getattr(fmsg, "content", ""))

                    try:
                        await asyncio.wait_for(
                            _drain_forced(),
                            timeout=self.INVESTIGATOR_WALL_CLOCK_TIMEOUT_S,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            "doc investigator: forced JSON continuation timed "
                            "out for dim=%r — keeping original terminal text.",
                            dim.get("id") or dim.get("title") or "?",
                        )
                    if (forced_text or "").strip() and _robust_json_object(forced_text) is not None:
                        logger.info(
                            "doc investigator: forced JSON continuation "
                            "recovered parseable JSON for dim=%r (len=%d).",
                            dim.get("id") or dim.get("title") or "?",
                            len(forced_text.strip()),
                        )
                        final_text = forced_text
                    else:
                        logger.info(
                            "doc investigator: forced JSON continuation did not "
                            "yield parseable JSON for dim=%r — keeping original.",
                            dim.get("id") or dim.get("title") or "?",
                        )
                except Exception as _force_exc:  # noqa: BLE001 — salvage must never worsen the result
                    logger.warning(
                        "doc investigator: forced JSON continuation errored for "
                        "dim=%r: %s — keeping original terminal text.",
                        dim.get("id") or dim.get("title") or "?", _force_exc,
                    )
        except asyncio.TimeoutError:
            _elapsed = time.monotonic() - _investigator_started_at
            logger.warning(
                "doc investigator: dim=%r WALL-CLOCK TIMEOUT after "
                "%.0fs (cap=%.0fs). Cancelling stuck LLM iteration "
                "and returning what was captured (final_text len=%d, "
                "tool_calls=%d). Status will likely be empty/unparseable.",
                dim.get("id") or dim.get("title") or "?",
                _elapsed, self.INVESTIGATOR_WALL_CLOCK_TIMEOUT_S,
                len(final_text or ""), tool_call_count,
            )
        finally:
            engine.close()
        return final_text, tool_call_count, file_read_count, tool_ledger

    def _build_retry_feedback(
        self,
        *,
        status: str,
        findings: dict[str, Any],
        tool_call_count: int,
        file_read_count: int = 0,
        attempt_number: int = 1,
        suggested_files: list[str] | None = None,
        tool_ledger: list[dict[str, Any]] | None = None,
    ) -> str:
        """Build a retry-feedback paragraph that goes at the top of
        the next user prompt under the ``## Retry feedback`` header.

        Escalates by attempt:
          * attempt 1 — explain what went wrong, restate the depth rule
          * attempt 2+ — "LAST CHANCE" framing + concrete file targets
            from ``suggested_files`` so the LLM can't dodge by claiming
            it doesn't know what to read

        When ``tool_ledger`` is supplied, the formatted ledger of the
        previous attempt's tool calls is appended to the feedback so
        the next attempt knows which ``(tool, key, args)`` combinations
        already produced results and must not be repeated.

        When ``status == "ok"`` no retry should fire, so return an
        empty string. Without this guard the function would otherwise
        synthesise a "LAST attempt" warning prefix whenever the
        retry-limit setting caused ``is_last`` to evaluate true on the
        first call, which is misleading.
        """
        if status == "ok":
            return ""
        is_zh = self.language.startswith("zh")
        is_last = attempt_number >= self.INVESTIGATOR_SHALLOW_RETRY_LIMIT
        ledger_block = ""
        if tool_ledger:
            ledger_text = _format_ledger_for_prompt(
                tool_ledger, language=self.language,
            )
            if ledger_text:
                ledger_block = "\n\n" + ledger_text
        files_block = ""
        if suggested_files:
            shown = suggested_files[:5]
            file_lines = "\n".join(f"  - {p}" for p in shown)
            if is_zh:
                files_block = (
                    "\n\n**必读文件清单**（这些是按你这个维度的关键词从 "
                    "scope 里挑出的最可能相关的文件，从中至少选 3 个 "
                    "file_read）：\n" + file_lines
                )
            else:
                files_block = (
                    "\n\n**Required reading list** (these are the files "
                    "in scope most likely relevant to your dimension — "
                    "pick AT LEAST 3 of them and `file_read` each):\n"
                    + file_lines
                )

        text_plan_stalled = bool(findings.get("text_plan_stalled"))
        xml_violation = bool(findings.get("xml_protocol_violation"))

        if is_zh:
            prefix = ""
            if is_last:
                prefix = (
                    f"⚠ 这是第 {attempt_number + 1} 次（也是最后一次）尝试。"
                    "如果这次还是 shallow / empty，整个维度会被标记为"
                    "**未调研**，不会再有下一次。\n\n"
                )
            if status == "shallow" and xml_violation:
                core = (
                    f"⛔ 协议违规：上一次回复里出现了 ``<tool_call>``、"
                    "``<file_read>`` 或类似的 XML 工具标签——但 cc "
                    "QueryEngine **不解析回复里的 XML**，它只识别 API "
                    "层的 tool_use 块。所以你写的那些 XML 工具调用一个\n"
                    "都没真的执行，本次产物全部作废。\n\n"
                    "**正确做法**：直接发起 tool_use 块（API 层）。比如\n"
                    "你想读 ``services/x.py``，就 emit 一个真实的 \n"
                    "``file_read`` tool_use，不要在文本里写 \n"
                    "``<file_read path=\"services/x.py\">``。\n\n"
                    "如果你的 LLM 客户端不支持原生 tool_use，那这次就**\n"
                    "不要用工具**了——基于已经读到的内容（如有）直接给\n"
                    "完整 JSON 输出，confidence 标 low、summary 里说明\n"
                    "限制。**不要再写 XML 工具标签**。"
                )
                return prefix + core + files_block + ledger_block
            if status == "shallow" and text_plan_stalled:
                core = (
                    f"上一次你做了 {file_read_count} 次 file_read 之后，"
                    "在回复里**用文字写**了下一步的工具调用计划（"
                    "类似 \"我会继续读 X、Y\"），但**没有真的发起新的"
                    "工具调用**。cc QueryEngine 只识别 API 层的 \n"
                    "tool_use 块，文字描述对它来说不存在，于是循环退出、"
                    "本次产物作废。\n\n"
                    "本次的关键不是\"读更多文件\"——是**真的把工具调用\n"
                    "emit 出来**。如果你还想读 N 个文件，**这一轮就发\n"
                    "起 N 个 file_read tool_use 块**，不要写任何\"接下来\n"
                    "我会读\"的叙述。要么继续做工具调用，要么直接给"
                    "完整 JSON——二选一，不要混着来。"
                )
                return prefix + core + files_block + ledger_block
            if status == "empty":
                core = (
                    "上一次调研没有产生任何可用输出（final_text 为空）。"
                    "本次必须按系统提示里的三阶段工作流执行：先用 glob 或"
                    "上面给的 ``## Files in scope`` 列表挑出候选；"
                    "然后**至少 file_read 3 个文件**；最后才输出 JSON。"
                )
                return prefix + core + files_block + ledger_block
            if status == "shallow":
                core = (
                    f"上一次做了 {tool_call_count} 次工具调用，但其中 "
                    f"file_read 只有 {file_read_count} 次（depth 规则要求 ≥3）"
                    "且 evidence/issues 都很薄。光 grep 不读文件不算调研。"
                    "本次必须 **file_read 至少 3 个文件**，每条 evidence "
                    "给出真实 ``lines`` 范围，每条 issue 给出真实 "
                    "``where: file:line``。否则结果会被丢弃。"
                )
                return prefix + core + files_block + ledger_block
            if status == "unparseable":
                core = (
                    "上一次返回的不是合法 JSON。本次必须严格按系统提示的 "
                    "JSON 模式输出，不要前后文、不要代码块围栏、不要注释。"
                    "也请补足深度——至少读 3 个文件再总结。"
                )
                return prefix + core + files_block + ledger_block
            return prefix + files_block + ledger_block

        prefix = ""
        if is_last:
            prefix = (
                f"⚠ This is attempt {attempt_number + 1} (the LAST "
                "one). If you produce shallow / empty output again, "
                "the entire dimension will be marked **not "
                "investigated** in the final report — no further "
                "retries.\n\n"
            )
        if status == "shallow" and xml_violation:
            core = (
                "⛔ Protocol violation: your previous reply contained "
                "``<tool_call>``, ``<file_read>``, or similar XML "
                "tool tags — but cc's QueryEngine **does not parse "
                "XML in the reply text**. It only sees real tool_use "
                "blocks emitted via the API. None of the XML tool "
                "calls you wrote actually executed; the work is "
                "discarded.\n\n"
                "**Correct behavior**: emit tool_use blocks via the "
                "API. If you want to read ``services/x.py``, emit a "
                "real ``file_read`` tool_use. Do NOT write "
                "``<file_read path=\"services/x.py\">`` in the text.\n\n"
                "If your LLM client doesn't support native tool_use, "
                "stop trying to use tools this turn. Emit a complete "
                "JSON answer based on whatever you've already read, "
                "with confidence=low and an honest summary noting the "
                "limitation. **Do NOT write more XML tool tags.**"
            )
            return prefix + core + files_block + ledger_block
        if status == "shallow" and text_plan_stalled:
            core = (
                f"Last time, after {file_read_count} `file_read` "
                "calls, you wrote your next tool calls as **text** in "
                "the reply (things like \"I'll continue reading X, "
                "Y\") instead of emitting them via the API. cc's "
                "QueryEngine only sees real tool_use blocks; text "
                "plans are invisible to it, so the loop exited and "
                "the work was discarded.\n\n"
                "The fix isn't \"read more files\" — it's **emit the "
                "tool calls for real**. If you still want to read N "
                "more files, emit N more `file_read` tool_use blocks "
                "THIS round. Do NOT write any \"I'll continue\" prose. "
                "Either keep making tool calls, or stop and emit the "
                "complete JSON now — pick one, don't mix."
            )
            return prefix + core + files_block + ledger_block
        if status == "empty":
            core = (
                "Your previous attempt produced no usable output "
                "(final_text was empty). This time you MUST follow the "
                "three-stage workflow in the system prompt: shortlist "
                "candidates via glob or the ``## Files in scope`` "
                "block, then file_read AT LEAST 3 of them, only then "
                "emit JSON."
            )
            return prefix + core + files_block + ledger_block
        if status == "shallow":
            core = (
                f"Your previous attempt made {tool_call_count} tool "
                f"calls but only {file_read_count} of them were "
                "`file_read` (the depth rule requires ≥ 3) and the "
                "evidence / issues lists were empty or trivial. "
                "Grepping without reading any files does NOT count as "
                "investigation. This time: `file_read` AT LEAST 3 "
                "files, give every evidence entry a real ``lines`` "
                "range, and every issue a real ``where: file:line``. "
                "A shallow retry will be discarded."
            )
            return prefix + core + files_block + ledger_block
        if status == "unparseable":
            core = (
                "Your previous response was not valid JSON. Emit "
                "strict JSON exactly matching the system prompt's "
                "schema — no preamble, no code fences, no commentary. "
                "Also do at least 3 file_read calls before emitting."
            )
            return prefix + core + files_block + ledger_block
        return prefix + files_block + ledger_block

    def _build_investigator_user_prompt(
        self, focus: str, root_goal: str, dim: dict[str, Any],
        *,
        retry_feedback: str = "",
        survey: dict[str, Any] | None = None,
    ) -> str:
        outline_text = self._maybe_outline_text(deep=True)
        scope = str(dim.get("scope_hint") or "")
        merged_text = " ".join(filter(None, [root_goal, dim.get("focus") or focus, scope]))
        path_block = self._paths_context_block(merged_text)
        # ``scope`` may be a bare directory name without slashes (e.g.
        # ``"src"``) or a comma-separated list — those wouldn't be
        # picked up by the regex-based path extractor. Pass it through
        # explicitly so the file list is built either way.
        files_block = self._scope_files_block(
            merged_text, extra_paths=[scope] if scope else None,
        )
        survey_block = _format_survey_for_prompt(survey or {})
        starter_actions = self._suggest_starter_actions(dim, scope)
        parts: list[str] = []
        if retry_feedback:
            parts.append(f"## Retry feedback (READ THIS FIRST)\n{retry_feedback}")
        if root_goal:
            parts.append(f"## Parent goal\n{root_goal}")
        parts.append(
            f"## Your dimension: {dim.get('title') or '(untitled)'}\n"
            f"{dim.get('focus') or focus}"
        )
        if scope:
            parts.append(f"## Suggested scope\n{scope}")
        if starter_actions:
            # Concrete first-move recipe for the dimension type. Not a
            # full plan — just enough to remove "where do I even start"
            # ambiguity that causes cross-cutting investigators to
            # bail out.
            steps_md = "\n".join(
                f"{i + 1}. {step}" for i, step in enumerate(starter_actions)
            )
            parts.append(
                "## Starter actions for this dimension (DO THESE FIRST)\n"
                f"{steps_md}\n\n"
                "These are concrete first moves tailored to your "
                "dimension type. Adapt the patterns / paths to your "
                "actual scope, run them, then proceed with deeper "
                "reads as the system prompt's three-stage workflow "
                "describes."
            )
        if survey_block:
            # Survey is shared context across all investigators. It
            # tells you WHAT THIS PROJECT IS — use it to skip the
            # "what does this codebase look like" exploration that
            # would otherwise eat tool rounds.
            parts.append(survey_block)
        if path_block:
            parts.append(path_block)
        if files_block:
            parts.append(files_block)
        if outline_text:
            parts.append(
                "## Repository Outline (PARTIAL — truncated; the dirs in 'Paths in this task' exist even if missing here)\n"
                f"```\n{outline_text}\n```"
            )
        # Comprehensive single-investigator dimensions cover the full
        # user goal in one pass. Without focus-narrowing, the LLM
        # tries to address every angle (architecture + perf + tests
        # + ...) and runs out of room before emitting JSON. Tell it
        # to prioritize depth over breadth: top 3-5 issues, well-
        # cited, beats 10 vague ones or a prose explanation of
        # everything it couldn't fit.
        if str(dim.get("id") or "") == "comprehensive":
            if self.language.startswith("zh"):
                parts.append(
                    "## 收窄要求（重要）\n"
                    "这是一次**综合性单投研者**评审。**不要**试图穷举\n"
                    "所有维度。请挑出 **top 3-5 个最有价值的问题**——\n"
                    "深度优先，质量优先。每条都要有真实的 file:line\n"
                    "引用和具体的修改建议。\n\n"
                    "**3 条精确、有引用的发现 > 10 条模糊的笼统建议\n"
                    "> 一篇'我没全部看完'的散文解释。**\n\n"
                    "如果发现想覆盖的角度太多，**只挑你最有把握、读\n"
                    "过最相关代码的那几个**写进 JSON。其它没读够的角\n"
                    "度直接丢掉，不要在 summary 里解释\"还有哪些没看\"\n"
                    "——那种 meta 解释不是评审产物。"
                )
            else:
                parts.append(
                    "## Focus narrowing (important)\n"
                    "This is a **comprehensive single-investigator**\n"
                    "review. Do NOT try to cover every dimension\n"
                    "exhaustively. Pick the **top 3-5 highest-value\n"
                    "issues** — depth and quality over coverage.\n"
                    "Each one needs a real file:line citation and a\n"
                    "concrete fix recommendation.\n\n"
                    "**3 well-cited findings > 10 vague ones > a prose\n"
                    "explanation of what you couldn't fit.**\n\n"
                    "If you find more angles than you can fit,\n"
                    "**pick the ones backed by the files you actually\n"
                    "read** and put those in JSON. Drop the rest;\n"
                    "don't write a meta-summary about what you didn't\n"
                    "cover — that's not a review deliverable."
                )
        parts.append(
            "Investigate ONLY this dimension following the three-stage "
            "workflow in the system prompt. The pre-emit counter check "
            "(≥3 file_read calls, every evidence/issue cited with "
            "file:line) is mandatory before you emit JSON."
        )
        return "\n\n".join(parts)

    def _scope_files_block(
        self, text: str, *, extra_paths: list[str] | None = None,
    ) -> str:
        """Emit a flat list of code AND doc files under any verified
        scope directory. This saves the investigator a tool round on
        Stage 1 enumeration and gives it concrete file_paths to read
        in Stage 2.

        Both Python source AND Markdown docs are listed (in separate
        sub-sections) so doc-flavored dimensions can see ``*.md``
        files — the previous Python-only enumeration silently hid
        documentation from the Documentation Completeness investigator.

        Path candidates come from two sources:
          1. ``extract_path_tokens(text)`` — paths the regex finds in
             free-text (works for tokens like ``core/foo/bar``).
          2. ``extra_paths`` — explicit list passed by the caller.
             Useful for short scope hints that don't trigger the regex
             (e.g. ``"src"`` or ``"core/foo,core/bar"``).

        The lists are capped per category (default 80 files / scope /
        category) to keep the prompt bounded; a tail message points
        the LLM at `glob` if more are needed.
        """
        candidates: list[str] = list(extract_path_tokens(text))
        if extra_paths:
            for p in extra_paths:
                # Allow comma-separated lists like "core/a, core/b".
                for item in str(p).split(","):
                    cleaned = item.strip().strip("`'\"").rstrip("/")
                    if cleaned and cleaned not in candidates:
                        candidates.append(cleaned)
        if not candidates:
            return ""
        cwd_path = Path(self.cwd) if self.cwd else None
        if cwd_path is None:
            return ""
        try:
            cwd_resolved = cwd_path.resolve()
        except OSError:
            return ""
        chunks: list[str] = []
        cap = 80
        seen_targets: set[str] = set()
        for tok in candidates:
            target = (cwd_resolved / tok) if not Path(tok).is_absolute() else Path(tok)
            try:
                target = target.resolve()
            except OSError:
                continue
            if not target.exists() or not target.is_dir():
                continue
            target_key = str(target)
            if target_key in seen_targets:
                continue
            seen_targets.add(target_key)
            py_files = self._enumerate_files_by_ext(
                target, cwd_resolved, exts=(".py",), max_files=cap,
            )
            md_files = self._enumerate_files_by_ext(
                target, cwd_resolved, exts=(".md",), max_files=cap,
            )
            if not py_files and not md_files:
                continue
            chunks.append(f"### Scope: `{tok}`")
            if py_files:
                chunks.append(
                    f"  Python source ({len(py_files)} files):"
                )
                chunks.extend(f"  - {p}" for p in py_files[:cap])
                if len(py_files) >= cap:
                    chunks.append(
                        f"  - ... ({len(py_files)} or more — capped at "
                        f"{cap}; use `glob` if you need the rest)"
                    )
            if md_files:
                chunks.append(
                    f"  Markdown docs ({len(md_files)} files):"
                )
                chunks.extend(f"  - {p}" for p in md_files[:cap])
                if len(md_files) >= cap:
                    chunks.append(
                        f"  - ... ({len(md_files)} or more — capped at "
                        f"{cap}; use `glob` if you need the rest)"
                    )
            chunks.append("")
        if not chunks:
            return ""
        return (
            "## Files in scope (pre-enumerated — start Stage 2 reads from this list)\n"
            + "\n".join(chunks).rstrip()
        )

    @staticmethod
    def _enumerate_files_by_ext(
        target: Path,
        cwd_resolved: Path,
        *,
        exts: tuple[str, ...],
        max_files: int,
    ) -> list[str]:
        """Walk ``target`` for files matching any extension in
        ``exts`` (case-insensitive) and return repo-relative
        forward-slash paths. Skips ``__pycache__`` and friends.

        ``exts`` should include the leading dot, e.g. ``(".py",)``.
        """
        out: list[str] = []
        ignore_segments = {
            "__pycache__", ".git", ".pytest_cache",
            ".venv", "node_modules", ".mypy_cache",
        }
        ext_set = {e.lower() for e in exts}
        for path in sorted(target.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in ext_set:
                continue
            if any(seg in ignore_segments for seg in path.parts):
                continue
            try:
                rel = path.relative_to(cwd_resolved)
            except ValueError:
                rel = path
            out.append(str(rel).replace("\\", "/"))
            if len(out) > max_files:
                break
        return out

    @staticmethod
    def _enumerate_python_files(
        target: Path, cwd_resolved: Path, *, max_files: int,
    ) -> list[str]:
        """Back-compat wrapper around ``_enumerate_files_by_ext``.

        The retry-feedback file suggester still uses this for now —
        it scores only on Python file names because the keyword tables
        target source code patterns.
        """
        return DocModeRunner._enumerate_files_by_ext(
            target, cwd_resolved, exts=(".py",), max_files=max_files,
        )

    def _resolve_scope_files(
        self, dim: dict[str, Any], *, max_files: int = 80,
    ) -> list[str]:
        """Return the flat list of repo-relative ``.py`` paths under
        the dimension's scope_hint. Same enumeration logic as the
        ``## Files in scope`` prompt block, exposed as a list so retry
        feedback can pick concrete file targets to suggest.
        """
        scope = str(dim.get("scope_hint") or "").strip()
        if not scope or not self.cwd:
            return []
        cwd_resolved = Path(self.cwd).resolve()
        out: list[str] = []
        seen: set[str] = set()
        # Honor comma-separated scope hints.
        for token in scope.split(","):
            tok = token.strip().strip("`'\"").rstrip("/")
            if not tok:
                continue
            target = (cwd_resolved / tok) if not Path(tok).is_absolute() else Path(tok)
            try:
                target = target.resolve()
            except OSError:
                continue
            if not target.exists() or not target.is_dir():
                continue
            for p in self._enumerate_python_files(target, cwd_resolved, max_files=max_files):
                if p in seen:
                    continue
                seen.add(p)
                out.append(p)
                if len(out) >= max_files:
                    return out
        return out

    @staticmethod
    def _suggest_starter_actions(
        dim: dict[str, Any], scope: str,
    ) -> list[str]:
        """Concrete first-move recipes for common cross-cutting
        dimensions.

        Cross-cutting concerns (``architecture``, ``error handling``,
        ``documentation``, ``contracts``, ``concurrency``) don't have
        natural file-bounded scope. Without a starter recipe, the LLM
        often produces shallow output: it greps once, sees too many
        hits, gives up, emits empty JSON.

        Each recipe is a 2-3 line "do this first" plan in cc tool
        syntax, so the investigator can copy-paste-adapt. Returns
        ``[]`` when the dimension's focus doesn't match any known
        keyword group — investigator falls back to its general
        workflow.

        ``scope`` is the dimension's scope_hint (a path string); when
        empty, the recipes use ``cwd=<scope>`` placeholder text and the
        LLM has to substitute its actual scope.
        """
        focus = (
            (str(dim.get("focus") or "") + " " + str(dim.get("title") or ""))
            .lower()
        )
        if not focus.strip():
            return []
        cwd_arg = f'cwd="{scope}"' if scope else 'cwd="<scope>"'
        recipes: list[tuple[tuple[str, ...], list[str]]] = [
            (
                ("error", "exception", "fail", "resilien", "retry"),
                [
                    f'grep(pattern="raise |except |try:|@retry", {cwd_arg}, '
                    'context_lines=2, max_results=40) '
                    '— this finds error-handling sites in one shot.',
                    'For each of the top 3-5 files in those hits, '
                    'file_read it (max_bytes=20000) to see how errors '
                    'are caught / converted / propagated.',
                    'Cite each issue with the exact file:line of the '
                    'raise / except site.',
                ],
            ),
            (
                ("architect", "layer", "boundary", "depend", "module"),
                [
                    f'glob(pattern="**/__init__.py", {cwd_arg}) '
                    '— finds package boundaries.',
                    'file_read each top-level package\'s __init__.py and '
                    'one representative file from each layer (e.g. '
                    'services/<one>.py, repositories/<one>.py).',
                    f'grep(pattern="^from \\.|^from {scope or "<scope>"}", '
                    f'{cwd_arg}, files_only=true) '
                    '— shows the actual import graph (which layer '
                    'depends on which).',
                ],
            ),
            (
                ("doc", "documentation", "readme"),
                [
                    f'glob(pattern="**/*.md", {cwd_arg}) '
                    '— enumerate every doc file in scope.',
                    'file_read EVERY .md you find (they\'re usually '
                    'small; use max_bytes=20000 if any are large). '
                    'Note what each documents.',
                    'Then sample 2-3 source files that the docs '
                    'describe. Cite specific places where docs and '
                    'code diverge or where docs are missing.',
                ],
            ),
            (
                ("contract", "protocol", "interface", "schema"),
                [
                    f'grep(pattern="class .*\\\\(Protocol\\\\)|@runtime_checkable|'
                    f'class .*\\\\(ABC\\\\)|@dataclass", {cwd_arg}, '
                    'files_only=true) — identifies declared contracts.',
                    'file_read 2-3 of those files to see the contract '
                    'shapes.',
                    f'grep(pattern="-> Optional|-> None", {cwd_arg}, '
                    'context_lines=1) — finds ambiguous None-returning '
                    'signatures often associated with contract '
                    'looseness.',
                ],
            ),
            (
                ("perform", "concur", "async", "thread", "lock"),
                [
                    f'grep(pattern="async def|await |asyncio|Lock\\\\(|'
                    f'threading\\\\.|ThreadPool", {cwd_arg}, '
                    'context_lines=2, max_results=30).',
                    'file_read the 2-3 files with the most async/lock '
                    'sites — these are your concurrency hotspots.',
                    'Cite specific lock acquisition / await patterns '
                    'that look risky.',
                ],
            ),
            (
                ("test", "coverage", "ci", "fixture"),
                [
                    f'glob(pattern="**/test_*.py", {cwd_arg}) and '
                    f'glob(pattern="**/tests/**/*.py", {cwd_arg}).',
                    'For each non-test source file in scope, check if '
                    'a corresponding test exists; cite uncovered files.',
                    'Sample 2-3 actual tests via file_read to see if '
                    'they assert behavior or just construction.',
                ],
            ),
            (
                ("naming", "readab", "style", "convention"),
                [
                    f'glob(pattern="**/*.py", {cwd_arg}) and pick the '
                    '5 longest filenames + 3 shortest; file_read each.',
                    f'grep(pattern="# TODO|# FIXME|# HACK|# XXX", '
                    f'{cwd_arg}, context_lines=1) — surfaces stale '
                    'comments and known shortcuts.',
                    'Cite specific naming inconsistencies with file:line.',
                ],
            ),
        ]
        out: list[str] = []
        for triggers, steps in recipes:
            if any(t in focus for t in triggers):
                out.extend(steps)
                # Only one recipe — first match wins. Combining recipes
                # would dilute the prompt.
                break
        return out

    @staticmethod
    def _suggest_files_for_dimension(
        dim: dict[str, Any], scope_files: list[str], *, top_n: int = 5,
    ) -> list[str]:
        """Pick up to ``top_n`` files most likely relevant to the
        dimension, by simple keyword scoring against file paths.

        The score sources, in priority order:
          1. Tokens in ``dim["focus"]`` (3+ chars, alphabetic) appearing
             in the path
          2. Hardcoded keyword expansions for common review dimensions
             ("error handling" → ``error|exception|retry|raise|fail``).
             Catches dimensions whose focus text doesn't directly name
             the relevant filenames.

        Returns paths with a positive score, sorted by descending
        score then by path length (shorter first — usually closer to
        module roots). Empty list when nothing scores.
        """
        if not scope_files:
            return []
        focus = (str(dim.get("focus") or "") + " " + str(dim.get("title") or "")).lower()
        if not focus.strip():
            return []
        # Direct tokens (≥3 alphabetic chars) from focus text.
        import re
        direct_tokens = {
            tok for tok in re.findall(r"[a-z]{3,}", focus)
        }
        # Stopwords we don't want to score on.
        direct_tokens -= {
            "and", "the", "for", "with", "into", "from", "this",
            "that", "are", "any", "all", "code", "file", "files",
            "module", "modules", "review", "should", "would", "must",
            "such", "very", "when", "what", "where", "which", "make",
            "your", "just", "then", "than", "have", "been",
        }
        keyword_groups: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
            # (focus-text triggers, filename-substrings to score on)
            (("error", "exception", "fail"),
             ("error", "exception", "retry", "raise", "fail")),
            (("test",),
             ("test_", "_test", "tests/")),
            (("perform", "speed", "throughput", "latency"),
             ("cache", "pool", "batch", "async", "concurrent", "perf")),
            (("concur", "parallel", "thread", "async", "lock"),
             ("lock", "thread", "async", "concurrent", "queue", "worker")),
            (("doc", "documentation", "comment"),
             ("readme", "doc", "docstring")),
            (("contract", "protocol", "schema", "type"),
             ("contract", "protocol", "schema", "model", "types")),
            (("architect", "layer", "boundary", "depend"),
             ("__init__", "agent", "blueprint", "service", "registry", "runtime")),
            (("naming", "readab", "style", "convention"),
             ("util", "helper", "common")),
        ]
        derived_tokens: set[str] = set()
        for triggers, derived in keyword_groups:
            if any(t in focus for t in triggers):
                derived_tokens.update(derived)
        all_tokens = direct_tokens | derived_tokens
        if not all_tokens:
            return []
        scored: list[tuple[int, int, str]] = []
        for path in scope_files:
            lower = path.lower()
            score = sum(1 for tok in all_tokens if tok in lower)
            if score > 0:
                scored.append((score, len(path), path))
        # Sort: highest score first, then shorter path first.
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [p for _, _, p in scored[:top_n]]

    def _push_findings(
        self, run_id: str, dim_id: str, findings: dict[str, Any],
    ) -> None:
        if self.findings_collector is None:
            logger.debug("doc investigator: no findings_collector wired; skipping push")
            return
        if not run_id or not dim_id:
            logger.debug("doc investigator: missing run_id/dim_id, skipping push")
            return
        self.findings_collector.push(run_id, dim_id, findings)

    # ------------------------------------------------------------------ #
    # Phase 3 — synthesizer
    # ------------------------------------------------------------------ #

    def _run_synthesizer(self, invocation: SubagentInvocation) -> SubagentResult:
        run_id = str(invocation.metadata.get("ccx_doc_run_id") or "")
        root_goal = str(
            invocation.metadata.get("ccx_doc_root_goal") or invocation.goal
        )
        # PEEK here, pop only after the report is rendered. ``pop_all``
        # up front is destructive: if the synth LLM call below raises (a
        # transient provider error), the v5 dispatcher retries this node
        # and the retry would find an EMPTY collector — silently turning
        # a full multi-investigator run into an outline-only report.
        findings: list[dict[str, Any]] = []
        if self.findings_collector is not None and run_id:
            findings = self.findings_collector.peek(run_id)
        expected_count = _coerce_positive_int(
            invocation.metadata.get("ccx_doc_dimension_count")
        )
        if expected_count and len(findings) < expected_count:
            missing = expected_count - len(findings)
            for idx in range(missing):
                findings.append({
                    "dimension_id": f"missing-{len(findings) + 1}",
                    "dimension_title": f"Missing dimension {len(findings) + 1}",
                    "summary": (
                        "No findings were collected for this planned "
                        "dimension before synthesis."
                    ),
                    "evidence": [],
                    "issues": [],
                    "confidence": "low",
                    "status": "error",
                    "tool_call_count": 0,
                    "file_read_count": 0,
                })

        # Degenerate-run gate. If too many investigators failed
        # (status != ``ok``), running a full synthesizer LLM call just
        # produces a padded report. Write a short honest stub instead
        # and skip the synth call entirely. The threshold is
        # ``SYNTH_DEGENERATE_FAILURE_RATIO``.
        if findings:
            non_ok = sum(1 for f in findings if str(f.get("status") or "ok") != "ok")
            failure_ratio = non_ok / len(findings)
            if failure_ratio >= self.SYNTH_DEGENERATE_FAILURE_RATIO:
                stub_md = self._render_degenerate_stub(
                    root_goal=root_goal,
                    findings=findings,
                    failure_ratio=failure_ratio,
                )
                artifact_path = self._maybe_write_artifact(
                    goal=root_goal, markdown=stub_md,
                )
                logger.info(
                    "doc synthesizer: degenerate-run gate triggered "
                    "(%d/%d non-ok dimensions, ratio=%.2f). Wrote stub "
                    "instead of full synth.",
                    non_ok, len(findings), failure_ratio,
                )
                if self.findings_collector is not None and run_id:
                    self.findings_collector.pop_all(run_id)
                return SubagentResult(
                    final_text=stub_md,
                    subtasks=[],
                    sequential=False,
                    extras={
                        "ccx_doc_phase": "synthesize",
                        "ccx_doc_run_id": run_id,
                        "child_count": len(findings),
                        "artifact_path": artifact_path,
                        "goal": root_goal,
                        "synth_llm_calls": 0,
                        "synth_truncation_log": [],
                        "synth_degenerate": True,
                        "synth_failure_ratio": failure_ratio,
                    },
                )

        outline_text = self._maybe_outline_text(deep=True)
        path_block = self._paths_context_block(root_goal)
        survey = dict(invocation.metadata.get("ccx_doc_survey") or {})
        survey_block = _format_survey_for_prompt(survey)
        system = load_cc_system_prompt("doc_mode", self.language)
        user_parts = [f"## Goal\n{root_goal}"]
        if survey_block:
            user_parts.append(survey_block)
        if path_block:
            user_parts.append(path_block)
        if outline_text:
            user_parts.append(
                "## Repository Outline (PARTIAL — truncated)\n"
                f"```\n{outline_text}\n```"
            )
        if findings:
            user_parts.append(
                "## Aggregated Observations\n"
                + _render_findings_for_synth(findings)
            )
        else:
            user_parts.append(
                "## Aggregated Observations\n"
                "(no concrete observations were collected — base the "
                "document on the goal and repository outline alone, "
                "and be explicit about the limitation.)"
            )
        user_parts.append(
            "Produce the final Markdown document. The reader is a "
            "developer working in this codebase, NOT someone debugging "
            "the review tool. Apply these rules:\n"
            "  * VOICE — write as if you reviewed the code yourself. "
            "Never use process / agent-internal terms in the output: "
            "no 'investigator', 'subagent', 'tool calls', 'JSON', "
            "'shallow', 'unparseable', 'empty', '调查器', 'Stage 2', "
            "etc. Those words describe how this report was produced; "
            "the reader does not need to know.\n"
            "  * EVIDENCE — group findings by review dimension; "
            "preserve every file:line citation **verbatim** (do not "
            "round, summarise, or change line numbers). For each issue, "
            "write what is wrong and what to do about it — concrete and "
            "actionable.\n"
            "  * FIDELITY — do NOT invent function / class / method / "
            "constant names. Only reference identifiers that appear "
            "verbatim in an evidence excerpt below. If a finding "
            "describes behaviour without naming a specific identifier, "
            "describe it generically (\"the variation step in this "
            "generator\") rather than fabricating a method name.\n"
            "  * LINE NUMBERS — every cited ``file:line`` MUST come "
            "from an evidence entry. Do not write \"约 N 行\" / "
            "\"approximately line N\" / \"~N\" — if the underlying "
            "evidence is fuzzy, drop the line number rather than "
            "guessing one.\n"
            "  * GAPS — for dimensions marked ``Coverage: limited`` "
            "or ``Coverage: none``, write ONE short paragraph: name "
            "the area, say it needs deeper review (or equivalent), "
            "and stop. Do NOT pad. Do NOT speculate. Do NOT invent "
            "issues. Do NOT promote a one-line summary into multiple "
            "imagined bullets.\n"
            "  * NO FILE-PATH CLAIMS — do not include any "
            "'**输出文件**:', '**Output file**:', 'Saved to:', "
            "'Generated at:' or similar header/footer claiming where "
            "this report is saved. The runner controls where the file "
            "lands; mentioning a path you didn't write is misleading.\n"
            "  * HONESTY — if the union of evidence is too thin to "
            "support real recommendations, say so plainly in 1-2 "
            "sentences at the top and produce a short honest report "
            "rather than a padded one."
        )
        # Opt-in substantiate-or-downgrade hardening. Appended as a separate,
        # self-contained final directive so the synthesis prompt is byte-
        # identical to legacy behaviour when disabled (default).
        substantiate = (
            self.substantiate if self.substantiate is not None
            else _doc_substantiate_enabled()
        )
        if substantiate:
            user_parts.append(
                _SUBSTANTIATION_RULE_ZH if self.language.startswith("zh")
                else _SUBSTANTIATION_RULE_EN
            )
        user = "\n\n".join(user_parts)

        markdown, synth_calls, truncation_log = self._synthesize_with_continuations(
            system=system, user=user, findings_count=len(findings),
        )
        artifact_path = self._maybe_write_artifact(
            goal=root_goal, markdown=markdown,
        )
        # Render succeeded — NOW it is safe to drain the collector.
        if self.findings_collector is not None and run_id:
            self.findings_collector.pop_all(run_id)

        return SubagentResult(
            final_text=str(markdown).strip(),
            subtasks=[],
            sequential=False,
            extras={
                "ccx_doc_phase": "synthesize",
                "ccx_doc_run_id": run_id,
                "child_count": len(findings),
                "artifact_path": artifact_path,
                "goal": root_goal,
                "synth_llm_calls": synth_calls,
                "synth_truncation_log": truncation_log,
                "synth_degenerate": False,
            },
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _maybe_outline_text(self, *, deep: bool) -> str:
        if self.outline_cache is None:
            return ""
        try:
            return self.outline_cache.get_text(deep=deep)
        except Exception as exc:  # noqa: BLE001
            logger.debug("doc runner: outline build failed: %s", exc)
            return ""

    def _render_degenerate_stub(
        self,
        *,
        root_goal: str,
        findings: list[dict[str, Any]],
        failure_ratio: float,
    ) -> str:
        """Render a short, honest Markdown stub when the run produced
        too few good observations to support a real report.

        The stub:
          * names the goal,
          * lists every dimension and whether it was actually covered
            (``ok`` / not investigated),
          * preserves any concrete evidence/issues from the few good
            dimensions,
          * tells the reader to rerun (with hints).

        No fabrication, no padding, no LLM call — produced purely from
        collected data.
        """
        is_zh = self.language.startswith("zh")
        non_ok = [f for f in findings if str(f.get("status") or "ok") != "ok"]
        ok_findings = [f for f in findings if str(f.get("status") or "ok") == "ok"]
        lines: list[str] = []
        if is_zh:
            lines.append(f"# 评审报告（数据不足）")
            lines.append("")
            lines.append(f"**目标**：{root_goal}")
            lines.append("")
            lines.append(
                f"本次评审的 {len(findings)} 个维度中，只有 "
                f"{len(ok_findings)} 个产出了有效证据（"
                f"{int((1 - failure_ratio) * 100)}% 覆盖）。"
                "证据量不足以支撑完整结论，因此本报告只列出**已确认**的"
                "发现，并标出未覆盖的维度。请扩大工具预算或缩小维度后重跑。"
            )
            lines.append("")
            if ok_findings:
                lines.append("## 已覆盖维度")
                lines.append("")
                for f in ok_findings:
                    title = f.get("dimension_title") or f.get("dimension_id") or "(unnamed)"
                    summary = (f.get("summary") or "").strip()
                    lines.append(f"### {title}")
                    if summary:
                        lines.append("")
                        lines.append(summary)
                    issues = f.get("issues") or []
                    if issues:
                        lines.append("")
                        lines.append("**发现的问题**：")
                        for it in issues:
                            sev = it.get("severity", "medium")
                            where = it.get("where") or ""
                            wp = f"（{where}）" if where else ""
                            lines.append(
                                f"- [{sev}] {it.get('title', '')}：{it.get('detail', '')}{wp}"
                            )
                    evidence = f.get("evidence") or []
                    if evidence:
                        lines.append("")
                        lines.append("**证据**：")
                        for ev in evidence:
                            ln = ev.get("lines") or ""
                            ex = (ev.get("excerpt") or "").replace("\n", " ")[:200]
                            lines.append(
                                f"- `{ev.get('path', '')}`:{ln} — {ex}"
                            )
                    lines.append("")
            if non_ok:
                lines.append("## 未覆盖维度")
                lines.append("")
                for f in non_ok:
                    title = f.get("dimension_title") or f.get("dimension_id") or "(unnamed)"
                    f_status = str(f.get("status") or "ok")
                    summary = (f.get("summary") or "").strip()
                    # Filter out placeholder summaries from the
                    # parser's empty-output path.
                    is_placeholder = summary in {
                        "(no investigator output)",
                        "(empty)",
                        "",
                    }
                    lines.append(f"### {title}")
                    lines.append("")
                    lines.append(
                        f"**调研状态**：{f_status}（未达 ok）"
                        f"——这一轮没有产出结构化结论。"
                    )
                    # When the LLM gave an honest prose explanation
                    # (e.g. "I read 3 files but couldn't finish"),
                    # preserve it so the user has SOMETHING actionable
                    # from the run. Cap at 1500 chars to keep the
                    # report bounded.
                    if not is_placeholder and len(summary) > 80:
                        lines.append("")
                        lines.append("**调研者原始笔记**：")
                        lines.append("")
                        excerpt = summary[:1500]
                        if len(summary) > 1500:
                            excerpt += "..."
                        lines.append("> " + excerpt.replace("\n", "\n> "))
                    # Surface any partial evidence/issues even if the
                    # JSON shape was incomplete.
                    issues = f.get("issues") or []
                    if issues:
                        lines.append("")
                        lines.append("**部分发现的问题**：")
                        for it in issues:
                            sev = it.get("severity", "medium")
                            where = it.get("where") or ""
                            wp = f"（{where}）" if where else ""
                            lines.append(
                                f"- [{sev}] {it.get('title', '')}：{it.get('detail', '')}{wp}"
                            )
                    evidence = f.get("evidence") or []
                    if evidence:
                        lines.append("")
                        lines.append("**部分证据**：")
                        for ev in evidence:
                            ln = ev.get("lines") or ""
                            ex = (ev.get("excerpt") or "").replace("\n", " ")[:200]
                            lines.append(
                                f"- `{ev.get('path', '')}`:{ln} — {ex}"
                            )
                    lines.append("")
                    lines.append(
                        "_建议后续专项审查这个维度，或参考下面的"
                        "重跑建议。_"
                    )
                    lines.append("")
            lines.append("## 重跑建议")
            lines.append("")
            lines.append(
                "- 减少 `parallelism`，让单个维度获得更多工具预算\n"
                "- 把范围（`scope_hint`）收窄到具体子目录\n"
                "- 或者把宽维度拆成更具体的子维度后再跑"
            )
        else:
            lines.append("# Review report (insufficient data)")
            lines.append("")
            lines.append(f"**Goal**: {root_goal}")
            lines.append("")
            lines.append(
                f"Of the {len(findings)} dimensions attempted in this "
                f"run, only {len(ok_findings)} produced verified "
                f"evidence ({int((1 - failure_ratio) * 100)}% coverage). "
                "Evidence is too thin to support a full report, so this "
                "document only lists **confirmed** findings and marks "
                "the uncovered dimensions. Rerun with a larger tool "
                "budget or narrower dimensions."
            )
            lines.append("")
            if ok_findings:
                lines.append("## Covered dimensions")
                lines.append("")
                for f in ok_findings:
                    title = f.get("dimension_title") or f.get("dimension_id") or "(unnamed)"
                    summary = (f.get("summary") or "").strip()
                    lines.append(f"### {title}")
                    if summary:
                        lines.append("")
                        lines.append(summary)
                    issues = f.get("issues") or []
                    if issues:
                        lines.append("")
                        lines.append("**Issues**:")
                        for it in issues:
                            sev = it.get("severity", "medium")
                            where = it.get("where") or ""
                            wp = f" ({where})" if where else ""
                            lines.append(
                                f"- [{sev}] {it.get('title', '')}: {it.get('detail', '')}{wp}"
                            )
                    evidence = f.get("evidence") or []
                    if evidence:
                        lines.append("")
                        lines.append("**Evidence**:")
                        for ev in evidence:
                            ln = ev.get("lines") or ""
                            ex = (ev.get("excerpt") or "").replace("\n", " ")[:200]
                            lines.append(
                                f"- `{ev.get('path', '')}`:{ln} — {ex}"
                            )
                    lines.append("")
            if non_ok:
                lines.append("## Uncovered dimensions")
                lines.append("")
                for f in non_ok:
                    title = f.get("dimension_title") or f.get("dimension_id") or "(unnamed)"
                    f_status = str(f.get("status") or "ok")
                    summary = (f.get("summary") or "").strip()
                    is_placeholder = summary in {
                        "(no investigator output)",
                        "(empty)",
                        "",
                    }
                    lines.append(f"### {title}")
                    lines.append("")
                    lines.append(
                        f"**Investigation status**: {f_status} "
                        "(did not reach ok) — no structured conclusions."
                    )
                    # Preserve the LLM's prose explanation when
                    # substantive — it usually contains useful context
                    # like "I read X, Y, Z but ran out of budget for
                    # full analysis".
                    if not is_placeholder and len(summary) > 80:
                        lines.append("")
                        lines.append("**Investigator's raw notes**:")
                        lines.append("")
                        excerpt = summary[:1500]
                        if len(summary) > 1500:
                            excerpt += "..."
                        lines.append("> " + excerpt.replace("\n", "\n> "))
                    issues = f.get("issues") or []
                    if issues:
                        lines.append("")
                        lines.append("**Partial issues found**:")
                        for it in issues:
                            sev = it.get("severity", "medium")
                            where = it.get("where") or ""
                            wp = f" ({where})" if where else ""
                            lines.append(
                                f"- [{sev}] {it.get('title', '')}: {it.get('detail', '')}{wp}"
                            )
                    evidence = f.get("evidence") or []
                    if evidence:
                        lines.append("")
                        lines.append("**Partial evidence**:")
                        for ev in evidence:
                            ln = ev.get("lines") or ""
                            ex = (ev.get("excerpt") or "").replace("\n", " ")[:200]
                            lines.append(
                                f"- `{ev.get('path', '')}`:{ln} — {ex}"
                            )
                    lines.append("")
                    lines.append(
                        "_Recommend a focused follow-up review or see "
                        "the rerun suggestions below._"
                    )
                    lines.append("")
            lines.append("## Suggestions for the next run")
            lines.append("")
            lines.append(
                "- Lower `parallelism` so each dimension gets more "
                "tool budget\n"
                "- Narrow the `scope_hint` to a specific sub-directory\n"
                "- Or split broad dimensions into more focused sub-dimensions"
            )
        return "\n".join(lines).rstrip() + "\n"

    def _synthesize_with_continuations(
        self,
        *,
        system: str,
        user: str,
        findings_count: int,
    ) -> tuple[str, int, list[dict[str, Any]]]:
        """Drive the synthesizer LLM with continuation retries when the
        output looks truncated.

        Most LLM completions cap output around 4-8K tokens. A
        multi-dimension review easily exceeds that, and the LLM
        silently stops mid-sentence — leaving the user with a doc
        that "only has the beginning written". This helper detects
        that with ``_looks_truncated`` and asks the LLM to continue.

        Returns ``(markdown, n_calls, log)`` where ``log`` records what
        each call produced (length, why-continued).
        """
        log: list[dict[str, Any]] = []
        markdown = text_of(
            self.llm(system=system, user=user, purpose="doc_synthesize")
        ).rstrip()
        log.append({
            "call": 1,
            "len": len(markdown),
            "tail": markdown[-120:] if markdown else "",
        })
        calls = 1
        for cont in range(1, self.SYNTH_MAX_CONTINUATIONS):
            reason = self._looks_truncated(markdown, findings_count)
            if not reason:
                break
            cont_user = self._build_synth_continuation_prompt(
                user=user, partial=markdown, reason=reason,
            )
            piece = text_of(self.llm(
                system=system, user=cont_user, purpose="doc_synthesize_continue",
            )).strip()
            calls += 1
            log[-1]["continued_because"] = reason
            log.append({"call": calls, "len": len(piece), "tail": piece[-120:]})
            if not piece or re.fullmatch(r"[\W_]*done[\W_]*", piece, re.I):
                # LLM signalled "done" — either with empty output or
                # the explicit sentinel from the continuation prompt.
                # Accept whatever ``markdown`` has so far without
                # appending the sentinel itself.
                break
            # Stitch: if the previous markdown ended without a newline,
            # add one before the continuation.
            if markdown and not markdown.endswith("\n"):
                markdown += "\n"
            markdown += piece
            markdown = markdown.rstrip()
        return markdown, calls, log

    def _build_synth_continuation_prompt(
        self, *, user: str, partial: str, reason: str,
    ) -> str:
        """Build a continuation user prompt. Carries enough of the
        original to keep the LLM grounded, plus the truncated tail so
        the LLM can stitch seamlessly, plus an explicit "continue, do
        not repeat" rule.
        """
        if self.language.startswith("zh"):
            instruction = (
                "你之前生成的 Markdown 文档**没写完**就停了"
                f"（{reason}）。请**直接续写**：从被截断的位置继续，"
                "不要重复任何已经写过的内容、不要复述前面说过的话、"
                "也不要重写章节标题。如果上一句话写到一半就停了，"
                "直接续上那句话；保持同样的语言、同样的章节结构、"
                "同样的引用格式（file:line）。"
                "全部写完后正常结束（最终一段后接换行即可），"
                "不要再加 \"--- end ---\" 这种标记。"
                "如果你判断报告其实已经完整，回复一行 ``DONE`` "
                "（仅这五个字符，不要其他内容）。"
            )
        else:
            instruction = (
                "The Markdown document you produced earlier was "
                f"**cut off before completion** ({reason}). Continue "
                "from where it stopped — do NOT repeat anything you "
                "already wrote, do NOT restate earlier sections, do "
                "NOT rewrite section headings. If the last sentence "
                "is incomplete, finish that sentence and continue. "
                "Keep the same language, structure, and citation "
                "format (file:line). End naturally when finished — "
                "do not add a trailing \"--- end ---\" marker. "
                "If you judge the document is actually already "
                "complete, reply with the single word ``DONE`` (no "
                "other content)."
            )
        # Show the LLM the last ~1500 chars of what it wrote so it
        # has the immediate context for continuation. Full earlier
        # context is implicit in the system prompt's evidence block,
        # which the original `user` already included.
        tail = partial[-1500:] if len(partial) > 1500 else partial
        parts = [
            "## Continuation request",
            instruction,
            "",
            "## Original task brief (unchanged)",
            user,
            "",
            "## Tail of what you wrote so far (continue AFTER this)",
            "```markdown",
            tail,
            "```",
        ]
        return "\n\n".join(parts)

    def _looks_truncated(
        self, markdown: str, findings_count: int,
    ) -> str:
        """Decide whether ``markdown`` looks cut-off mid-stream.

        Returns the reason as a short string when truncated, or ``""``
        when it looks complete. The reasons are user-facing only via
        ``extras["synth_truncation_log"]`` for telemetry; the LLM-side
        continuation prompt rephrases them.

        Heuristics, in order:
          * ``DONE`` sentinel from a previous continuation → not
            truncated.
          * Output length below the rough expected minimum → likely
            truncated. The expected minimum scales with the number of
            findings.
          * The last non-blank line ends mid-sentence (no terminator
            and not a Markdown-structural ending like ``)`` ``\\``) →
            truncated.
          * Output ends inside a fenced code block (odd number of
            ``\\`\\`\\``` delimiters) → truncated.
        """
        text = (markdown or "").strip()
        if not text:
            return "empty output"
        if text.endswith("DONE") and len(text) <= 8:
            return ""
        # Length check
        expected = (
            self.SYNTH_MIN_BYTES_BASE
            + max(0, findings_count) * self.SYNTH_MIN_BYTES_PER_FINDING
        )
        if len(text) < expected and findings_count > 0:
            return f"output is {len(text)} bytes, expected at least {expected}"
        # Unbalanced fenced code block
        if text.count("```") % 2 == 1:
            return "ends inside an unclosed ``` code fence"
        # Last non-blank line ends mid-sentence
        last_line = ""
        for line in reversed(text.splitlines()):
            stripped = line.rstrip()
            if stripped:
                last_line = stripped
                break
        if last_line:
            terminators = (
                ".", "。", "!", "！", "?", "？", ":", "：",
                ";", "；", ")", "）", "]", "】", "”", '"',
                "`", ">", "*", "-",
            )
            if not last_line.endswith(terminators) and not last_line.startswith(("|", "#", "-", "*")):
                # Not a sentence terminator, not a list item, not a
                # heading — looks like a sentence cut mid-word.
                return "last line ends mid-sentence without a terminator"
        return ""

    def _resolve_max_dimensions(self, survey: dict[str, Any]) -> int:
        """Decide how many dimensions the planner is allowed to emit.

        Logic:
          * Hard floor = ``MIN_PARALLELISM_FOR_FANOUT`` (= 2 today).
          * Hard ceiling = ``self.parallelism`` (the user budget).
          * Within those bounds, scale by survey's complexity_signal:
              - simple  → 3
              - medium  → 4
              - complex → 6 (or whatever the user budget allows)
          * Empty / missing survey → default to ``self.parallelism``
            (current behavior).
        """
        floor = self.MIN_PARALLELISM_FOR_FANOUT
        ceiling = max(floor, self.parallelism)
        signal = str((survey or {}).get("complexity_signal") or "").lower()
        target = self.SURVEY_COMPLEXITY_TO_DIMS.get(signal)
        if target is None:
            return ceiling
        return max(floor, min(target, ceiling))

    def _effective_investigator_max_rounds(
        self, survey: dict[str, Any] | None = None,
    ) -> int:
        """Return the per-investigator tool-round cap.

        Honors an explicit ``self.max_tool_rounds`` if set. Otherwise the
        base cap is scaled by the surveyor's ``complexity_signal`` via
        ``SURVEY_COMPLEXITY_TO_ROUNDS`` (simple=30 / medium=45 /
        complex=60); an unknown or missing signal falls back to the full
        ``INVESTIGATOR_DEFAULT_MAX_ROUNDS`` so a review we couldn't size is
        never starved. A depth boost (``SINGLE_INVESTIGATOR_ROUND_MULTIPLIER``,
        2x) is applied on top when ``parallelism < MIN_PARALLELISM_FOR_FANOUT``,
        since a single investigator has no cross-investigator competition for
        the tool budget and can safely go deeper.
        """
        if self.max_tool_rounds is not None and self.max_tool_rounds > 0:
            return int(self.max_tool_rounds)
        signal = str((survey or {}).get("complexity_signal") or "").lower()
        base = self.SURVEY_COMPLEXITY_TO_ROUNDS.get(
            signal, self.INVESTIGATOR_DEFAULT_MAX_ROUNDS
        )
        if self.parallelism < self.MIN_PARALLELISM_FOR_FANOUT:
            return base * self.SINGLE_INVESTIGATOR_ROUND_MULTIPLIER
        return base

    # Directory names never worth walking for a source-structure scan.
    _SCAN_PRUNE_DIRS: frozenset[str] = frozenset({
        "__pycache__", ".git", "node_modules", ".venv", "venv",
        ".pytest_cache", ".mypy_cache", "build", "dist", ".idea", ".vscode",
    })

    def _scan_scope_structure(
        self, invocation: "SubagentInvocation",
    ) -> tuple[str, int | None, list[str]]:
        """Deterministic structural scan of the review's target scope.

        Counts Python files and lists immediate subdirectories under the
        path(s) named in the goal (or the whole ``cwd`` when none is
        named), returning ``(complexity_signal, py_count, top_level_dirs)``
        using the same ``≤20 / ≤80 / else`` heuristic as the LLM surveyor.
        Used to backfill a flaky or empty survey so round-cap scaling and
        the planner's structural context stay reliable. Best-effort:
        returns ``("", None, [])`` on any error and caps the walk so a
        whole-repo scope resolves quickly to ``complex``.
        """
        import os
        try:
            goal = str(getattr(invocation, "goal", "") or "")
            roots: list[Path] = []
            for tok in extract_path_tokens(goal):
                p = Path(tok)
                if not p.is_absolute():
                    p = Path(self.cwd) / tok
                if p.exists():
                    roots.append(p)
            if not roots:
                # No explicit path → whole-repo review → treat as complex
                # (full round budget) rather than risk under-budgeting.
                roots = [Path(self.cwd)]
            py = 0
            top_dirs: list[str] = []
            capped = False
            for root in roots:
                if root.is_file():
                    if root.suffix == ".py":
                        py += 1
                    continue
                try:
                    top_dirs.extend(sorted(
                        d.name for d in root.iterdir()
                        if d.is_dir() and d.name not in self._SCAN_PRUNE_DIRS
                    )[:12])
                except OSError:
                    pass
                for dirpath, dirnames, filenames in os.walk(root):
                    dirnames[:] = [
                        d for d in dirnames if d not in self._SCAN_PRUNE_DIRS
                    ]
                    py += sum(1 for f in filenames if f.endswith(".py"))
                    if py > 200:
                        capped = True
                        break
                if capped:
                    break
            if py <= 0:
                return ("", None, top_dirs[:12])
            if capped or py > 80:
                complexity = "complex"
            elif py <= 20:
                complexity = "simple"
            else:
                complexity = "medium"
            return (complexity, py, top_dirs[:12])
        except Exception:  # noqa: BLE001 - best-effort; never break the survey
            return ("", None, [])

    def _paths_context_block(self, *texts: str) -> str:
        """Return a "Paths in this task" Markdown block for any path
        tokens found in the supplied texts.

        For each token we resolve under ``self.cwd`` and label it
        ``[verified]`` if it exists, ``[missing]`` if it doesn't, and
        emit a focused subtree outline anchored at any verified
        directory. The block is empty (returns "") when no path tokens
        are found, so callers can append unconditionally.

        Existence verification is the cheap fix for outline truncation:
        the LLM is told "this path exists, regardless of whether it
        appears in the outline above". Focused subtrees give it the
        structural context it would otherwise have to glob for.
        """
        merged = " ".join(t for t in texts if t)
        tokens = extract_path_tokens(merged)
        if not tokens:
            return ""
        cwd_path = Path(self.cwd) if self.cwd else None
        lines: list[str] = ["## Paths in this task"]
        focused_chunks: list[str] = []
        for tok in tokens:
            status = "[?]"
            target: Path | None = None
            if cwd_path is not None:
                cand = (cwd_path / tok) if not Path(tok).is_absolute() else Path(tok)
                try:
                    resolved = cand.resolve()
                except OSError:
                    resolved = cand
                if resolved.exists():
                    status = "[verified]"
                    target = resolved
                else:
                    status = "[missing in cwd — verify with `glob`]"
            lines.append(f"- {status} `{tok}`")
            if (
                target is not None
                and target.is_dir()
                and self.outline_cache is not None
            ):
                try:
                    sub = self.outline_cache.get_focused_text(
                        tok, max_depth=3, max_entries_per_dir=12,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug("doc runner: focused outline failed for %s: %s", tok, exc)
                    sub = ""
                if sub:
                    focused_chunks.append(
                        f"### Focused subtree: `{tok}`\n```\n{sub}\n```"
                    )
        if focused_chunks:
            lines.append("")
            lines.extend(focused_chunks)
        return "\n".join(lines)

    def _maybe_write_artifact(
        self, *, goal: str, markdown: str,
    ) -> str | None:
        if not self.write_artifact:
            return None
        text = str(markdown or "").strip()
        if not text:
            return None
        try:
            if self.output_path:
                # Honor the user-specified destination. Relative paths
                # resolve under ``cwd``; parent dirs are created.
                target = Path(self.output_path)
                if not target.is_absolute():
                    target = Path(self.cwd) / target
                target = target.resolve()
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text + "\n", encoding="utf-8")
                return str(target)
            root = _resolve_artifact_root(
                cwd=self.cwd, kind="docs",
                artifact_root=self.docs_artifact_root,
            )
            root.mkdir(parents=True, exist_ok=True)
            artifact_id = _new_artifact_id("doc", goal)
            target = root / f"{artifact_id}.md"
            target.write_text(text + "\n", encoding="utf-8")
            return str(target)
        except Exception as exc:  # noqa: BLE001
            logger.warning("doc runner: failed to write artifact: %s", exc)
            return None

    def _run_surveyor(
        self, invocation: SubagentInvocation,
    ) -> dict[str, Any]:
        """Run the structural survey at the start of the planner phase.

        Returns a structured dict matching the schema in
        ``_SURVEYOR_SYSTEM_*``. Returns ``{}`` on any failure (no
        tools, no provider, parse error). Callers must tolerate an
        empty dict.

        The returned dict is plumbed into both the decompose user
        prompt (so dimensions adapt to actual structure) and into
        each investigator's metadata (so they share the same mental
        model of the project).
        """
        if not self.has_tools:
            return {}
        if self.llm_provider is None or self.cc_config is None:
            return {}
        from ..agents.cc_agent import _run_in_fresh_loop
        try:
            return _run_in_fresh_loop(self._run_surveyor_async(invocation))
        except Exception as exc:  # noqa: BLE001
            logger.warning("doc surveyor failed: %s — proceeding without survey", exc)
            return {}

    def _run_surveyor_with_tracking(
        self, invocation: SubagentInvocation,
    ) -> dict[str, Any]:
        context = None
        token = None
        if self.llm_provider is not None and hasattr(self.llm_provider, "begin_invocation"):
            context, token = self.llm_provider.begin_invocation(
                mode=self.mode_name,
                metadata=invocation.metadata,
            )
        try:
            return self._run_surveyor(invocation)
        finally:
            if context is not None and context.cost_accumulator:
                cost_usd = sum(context.cost_accumulator)
                _emit_provider_cost_event(
                    mode=self.mode_name,
                    cost_usd=cost_usd,
                    call_count=len(context.cost_accumulator),
                    tokens=sum(context.token_accumulator),
                )
                report_cost_to_budget(
                    cost_usd=cost_usd,
                    tokens=sum(context.token_accumulator),
                )
            if token is not None and hasattr(self.llm_provider, "end_invocation"):
                self.llm_provider.end_invocation(token)

    async def _run_surveyor_async(
        self, invocation: SubagentInvocation,
    ) -> dict[str, Any]:
        from core.cc.runtime import build_default_query_engine

        engine = build_default_query_engine(
            cwd=self.cwd,
            config=self.cc_config,
            llm_client_provider=self.llm_provider,
        )
        restrict_tool_registry(engine)

        system = _load_surveyor_system(self.language)
        user = self._build_surveyor_user_prompt(invocation)
        framed = f"<system>\n{system}\n</system>\n\n{user}"

        final_text = ""
        tool_call_count = 0

        async def _drain() -> None:
            nonlocal final_text, tool_call_count
            async for event in engine.submit_message(
                framed, max_tool_rounds=self.SURVEYOR_MAX_ROUNDS,
            ):
                if event.event_type == "assistant_tool_use":
                    tool_call_count += 1
                msg = getattr(event, "message", None)
                if msg is None:
                    continue
                if (
                    getattr(msg, "role", "") == "assistant"
                    and getattr(msg, "kind", "") == "assistant_text"
                ):
                    final_text = str(getattr(msg, "content", ""))

        try:
            await asyncio.wait_for(
                _drain(),
                timeout=self.SURVEYOR_WALL_CLOCK_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "doc surveyor: wall-clock timeout after %.0fs",
                self.SURVEYOR_WALL_CLOCK_TIMEOUT_S,
            )
        finally:
            engine.close()

        survey = _parse_surveyor_response(final_text)
        # The LLM surveyor uses read-only tools, so it can't opt into API
        # JSON mode, and its final structured emission is stochastically
        # unparseable (observed in practice: survey == {} →
        # ``complexity=? top_dirs=0``). When the structural facts are
        # missing, backfill them from a cheap, deterministic filesystem
        # scan of the target scope so (a) the planner still receives
        # structural context and (b) the investigator round-cap scaling
        # (``SURVEY_COMPLEXITY_TO_ROUNDS``) stays reliable instead of
        # always degrading to the full default on a flaky survey.
        if not isinstance(survey, dict):
            survey = {}
        if not survey.get("complexity_signal") or not survey.get("top_level_dirs"):
            fs_complexity, fs_py, fs_dirs = self._scan_scope_structure(invocation)
            if fs_complexity and not survey.get("complexity_signal"):
                survey["complexity_signal"] = fs_complexity
                survey["__complexity_source__"] = "filesystem"
            if fs_py is not None and not (survey.get("file_count") or {}).get("py"):
                fc = dict(survey.get("file_count") or {})
                fc["py"] = fs_py
                survey["file_count"] = fc
            if fs_dirs and not survey.get("top_level_dirs"):
                survey["top_level_dirs"] = fs_dirs
        if survey:
            survey["__meta__"] = {
                "tool_call_count": tool_call_count,
                "raw_len": len(final_text),
            }
        logger.info(
            "doc surveyor: tool_calls=%d complexity=%s top_dirs=%d",
            tool_call_count,
            survey.get("complexity_signal", "?"),
            len(survey.get("top_level_dirs") or []),
        )
        return survey

    def _build_surveyor_user_prompt(
        self, invocation: SubagentInvocation,
    ) -> str:
        """Surveyor user prompt — short, structural, scope-aware."""
        scope_paths = extract_path_tokens(invocation.goal)
        parts: list[str] = [f"## Goal context\n{invocation.goal}"]
        if scope_paths:
            paths_md = "\n".join(f"- `{p}`" for p in scope_paths[:5])
            parts.append(
                "## Scope to survey (verified paths)\n"
                + paths_md
                + "\n\nAll `glob` / `grep` / `file_read` calls should "
                "scope to one of these paths."
            )
        else:
            parts.append(
                "## Scope to survey\n"
                "Workspace root (no specific path was named in the "
                "goal). Survey the project as a whole."
            )
        parts.append(
            "Produce the structured JSON survey now. Aim for 5-10 "
            "tool calls."
        )
        return "\n\n".join(parts)

    def _call_llm_with_tools(self, *, system: str, user: str) -> str:
        """Single-shot path that still uses cc tools (parallelism=1).

        Builds a cc QueryEngine with read-only restriction and runs one
        framed turn. Returns the final assistant text.
        """
        from ..agents.cc_agent import _run_in_fresh_loop
        return _run_in_fresh_loop(
            self._call_llm_with_tools_async(system=system, user=user),
        )

    async def _call_llm_with_tools_async(
        self, *, system: str, user: str,
    ) -> str:
        if self.llm_provider is None or self.cc_config is None:
            # Misconfiguration — fall back to a tool-less single call.
            return text_of(self.llm(system=system, user=user, purpose="doc_fallback"))
        from core.cc.runtime import build_default_query_engine

        engine = build_default_query_engine(
            cwd=self.cwd,
            config=self.cc_config,
            llm_client_provider=self.llm_provider,
        )
        restrict_tool_registry(engine)
        framed = f"<system>\n{system}\n</system>\n\n{user}"
        final_text = ""
        # Use the effective investigator cap rather than raw
        # ``self.max_tool_rounds``: when nothing is set, ``None``
        # would mean "unlimited" to cc but in practice the LLM stops
        # itself on first emitted text. The effective value gives a
        # real number (default 60, doubled to 120 for parallelism=1)
        # so the budget is observable and enforced.
        rounds_cap = self._effective_investigator_max_rounds()
        try:
            async for event in engine.submit_message(
                framed, max_tool_rounds=rounds_cap,
            ):
                msg = getattr(event, "message", None)
                if msg is None:
                    continue
                if (
                    getattr(msg, "role", "") == "assistant"
                    and getattr(msg, "kind", "") == "assistant_text"
                ):
                    final_text = str(getattr(msg, "content", ""))
        finally:
            engine.close()
        return final_text


# --------------------------------------------------------------------------- #
# Parsing helpers (module-level for testability)
# --------------------------------------------------------------------------- #

class _ThreadLocalSuppressFilter(logging.Filter):
    """Suppress sub-CRITICAL records for threads inside a ``with`` block.

    ``extract_json_from_text`` logs its failure diagnostics through the
    repo-wide shared logger (``core.utils.log.logger``), which trading
    code also uses. The obvious "save level / set CRITICAL / restore"
    dance is NOT thread-safe: doc-mode investigators run concurrently on
    worker threads, and two interleaved save/restore pairs can leave the
    global logger pinned at CRITICAL for the rest of the process (A
    saves INFO, sets CRITICAL; B saves CRITICAL; A restores INFO; B
    "restores" CRITICAL). This filter keeps the suppression strictly
    per-thread instead: it is installed once on the shared logger and
    drops records below CRITICAL only when emitted by a thread currently
    inside the context manager. Other threads — and this thread outside
    the block — log normally. Reentrant via a depth counter.
    """

    def __init__(self) -> None:
        super().__init__()
        self._local = threading.local()

    def __enter__(self) -> "_ThreadLocalSuppressFilter":
        self._local.depth = getattr(self._local, "depth", 0) + 1
        return self

    def __exit__(self, *_exc: object) -> None:
        self._local.depth = getattr(self._local, "depth", 0) - 1

    def filter(self, record: logging.LogRecord) -> bool:
        if (
            getattr(self._local, "depth", 0) > 0
            and record.levelno < logging.CRITICAL
        ):
            return False
        return True


_JFT_SUPPRESS_FILTER = _ThreadLocalSuppressFilter()
_JFT_FILTER_LOCK = threading.Lock()
_jft_filter_installed = False


def _ensure_jft_suppress_filter(jft_logger: logging.Logger) -> None:
    """Install ``_JFT_SUPPRESS_FILTER`` on *jft_logger* exactly once.

    The filter stays attached for the process lifetime — it is a no-op
    for every thread not currently inside its context manager, so
    leaving it installed is harmless and avoids racy add/remove pairs.
    """
    global _jft_filter_installed
    if _jft_filter_installed:
        return
    with _JFT_FILTER_LOCK:
        if not _jft_filter_installed:
            jft_logger.addFilter(_JFT_SUPPRESS_FILTER)
            _jft_filter_installed = True


def _robust_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort JSON-object extraction with progressive fallback.

    Order of strategies (each fails silently and falls through):

    1. **Strip ``` fence + ``json.loads``** — fast path; matches the
       legacy weak parser. Handles the clean "model put one JSON block
       in a fence" case at zero cost.
    2. **``{...}`` substring + ``json.loads``** — fallback for "fence
       was forgotten / prose-wrapped JSON".
    3. **``core.utils.json_from_text.extract_json_from_text``** — the
       heavy-duty extractor maintained for LLM-output cleanup elsewhere
       in the repo. Handles multiple fenced candidates, single-quoted
       strings, unquoted keys, trailing commas, truncated structures,
       over-escaped backslashes, etc. Invoked with ``use_llm=False`` so
       this helper never makes an LLM call (caller is itself inside an
       LLM round and any extra cost / latency would compound).

    Returns the first parseable ``dict``. Returns ``None`` if every
    strategy fails or only non-dict (list/primitive) results emerge.

    Why this exists: doc mode previously had three near-identical
    weak parsers (surveyor / planner / investigator) that only knew
    strategies 1 + 2. When DeepSeek-Reasoning leaked prose around the
    JSON or used markdown bullets in string values, all three would
    fall through to "no parse" and the run would be flagged
    ``unparseable``, eventually tripping the synthesizer's
    degenerate-run gate. Strategy 3 recovers most of those cases.
    """
    text = (text or "").strip()
    if not text:
        return None

    # Strategy 1: strip first ``` fence and try parse.
    candidate = text
    if candidate.startswith("```"):
        first_nl = candidate.find("\n")
        last_fence = candidate.rfind("```")
        if first_nl != -1 and last_fence > first_nl:
            candidate = candidate[first_nl + 1:last_fence].strip()
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: greedy ``{...}`` substring.
    start = candidate.find("{")
    end = candidate.rfind("}")
    if 0 <= start < end:
        try:
            parsed = json.loads(candidate[start:end + 1])
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 3: heavy-duty extractor. Wrapped in a broad except so
    # any import failure / unexpected runtime error in the extractor
    # cannot break ccx (this helper must NEVER raise into the caller).
    #
    # ``extract_json_from_text`` logs a diagnostic block (``ERROR``
    # level with the full failed text) when every strategy misses,
    # which is appropriate when it's the user-facing extractor but
    # spams the ccx investigator log when we're using it as a quiet
    # fallback. Suppress sub-CRITICAL records for THIS THREAD ONLY via
    # ``_ThreadLocalSuppressFilter`` — never by mutating the shared
    # logger's level, which races under concurrent investigators (see
    # the filter's docstring).
    try:
        from core.utils.json_from_text import extract_json_from_text
        from core.utils.log import logger as _jft_logger

        _ensure_jft_suppress_filter(_jft_logger)
        with _JFT_SUPPRESS_FILTER:
            result = extract_json_from_text(
                text, use_llm=False, max_attempts=0,
            )
        if isinstance(result, dict):
            return result
        if isinstance(result, list):
            # If the extractor produced a list, prefer the first dict
            # element. Doc-mode callers all expect an object.
            for item in result:
                if isinstance(item, dict):
                    return item
    except Exception:  # noqa: BLE001 — extractor is best-effort
        # Suppress: the extractor logs its own diagnostics, and the
        # caller will treat ``None`` as "unparseable" same as before.
        pass

    return None


def _parse_surveyor_response(response: str) -> dict[str, Any]:
    """Robust parser for the surveyor LLM output.

    Accepts bare JSON or fenced blocks. Returns ``{}`` on any parse
    failure; planner just proceeds without survey context (degrades
    to today's behavior).

    Normalizes the schema:
      * ``file_count`` is a dict of int counts
      * ``top_level_dirs`` / ``doc_files`` / ``key_entry_points`` are
        unique str lists
      * ``complexity_signal`` is one of ``simple|medium|complex`` (or
        ``"medium"`` if missing/invalid)
      * ``notes`` is a stripped str
    """
    # Robust JSON-object extraction: handles ``` fences, ``{...}``
    # substring fallback, and (via ``_robust_json_object``) the heavy
    # ``extract_json_from_text`` extractor that repairs unquoted keys /
    # trailing commas / truncated JSON / over-escaped backslashes.
    parsed = _robust_json_object(response)
    if parsed is None:
        return {}

    def _str_list(raw: Any) -> list[str]:
        if not isinstance(raw, list):
            return []
        out: list[str] = []
        seen: set[str] = set()
        for item in raw:
            s = str(item).strip()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out

    fc_raw = parsed.get("file_count")
    file_count: dict[str, int] = {}
    if isinstance(fc_raw, dict):
        for k, v in fc_raw.items():
            try:
                file_count[str(k)] = int(v)
            except (TypeError, ValueError):
                continue

    complexity = str(parsed.get("complexity_signal") or "").strip().lower()
    if complexity not in {"simple", "medium", "complex"}:
        # Heuristic fallback from py count if signal is missing.
        py_count = file_count.get("py", 0)
        if py_count <= 20:
            complexity = "simple"
        elif py_count <= 80:
            complexity = "medium"
        else:
            complexity = "complex"

    return {
        "file_count": file_count,
        "top_level_dirs": _str_list(parsed.get("top_level_dirs")),
        "doc_files": _str_list(parsed.get("doc_files")),
        "key_entry_points": _str_list(parsed.get("key_entry_points")),
        "tests_dir": str(parsed.get("tests_dir") or "").strip(),
        "complexity_signal": complexity,
        "notes": str(parsed.get("notes") or "").strip(),
    }


def _format_survey_for_prompt(survey: dict[str, Any]) -> str:
    """Render a survey dict as a concise, planner-friendly Markdown
    block. Skips ``__meta__`` and empty fields so the prompt stays
    short."""
    if not survey:
        return ""
    lines: list[str] = []
    fc = survey.get("file_count") or {}
    if fc:
        pretty = ", ".join(f"{k}={v}" for k, v in sorted(fc.items()))
        lines.append(f"- File counts: {pretty}")
    tld = survey.get("top_level_dirs") or []
    if tld:
        lines.append(f"- Top-level dirs ({len(tld)}): {', '.join(tld)}")
    docs = survey.get("doc_files") or []
    if docs:
        shown = docs[:8]
        suffix = f" ... (+{len(docs) - 8} more)" if len(docs) > 8 else ""
        lines.append(f"- Doc files: {', '.join(shown)}{suffix}")
    entries = survey.get("key_entry_points") or []
    if entries:
        lines.append(f"- Entry points: {', '.join(entries)}")
    tests = survey.get("tests_dir") or ""
    if tests:
        lines.append(f"- Tests dir: {tests}")
    sig = survey.get("complexity_signal") or ""
    if sig:
        lines.append(f"- Complexity: **{sig}**")
    notes = survey.get("notes") or ""
    if notes:
        lines.append(f"- Notes: {notes}")
    if not lines:
        return ""
    return "## Project Survey (from a quick structural read)\n" + "\n".join(lines)


def _parse_decompose_response(response: str) -> list[dict[str, Any]]:
    """Robust parser for the planner LLM output.

    Accepts bare JSON or fenced blocks. Returns an empty list on any
    parse failure; planner falls back to single-shot in that case.
    """
    # Robust JSON-object extraction — see ``_robust_json_object`` for
    # the strategy ladder.
    parsed = _robust_json_object(response)
    if parsed is None:
        return []
    raw = parsed.get("dimensions") or []
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    emitted_ids: set[str] = set()
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or item.get("name") or "").strip()
        focus = str(item.get("focus") or item.get("description") or "").strip()
        if not title and not focus:
            continue
        dim_id = str(item.get("id") or "").strip() or _slug(title or focus, idx=i)
        base_id = dim_id
        count = 1
        while dim_id in emitted_ids:
            count += 1
            dim_id = f"{base_id}-{count}"
        emitted_ids.add(dim_id)
        out.append({
            "id": dim_id,
            "title": title or focus,
            "focus": focus or title,
            "scope_hint": str(item.get("scope_hint") or item.get("scope") or "").strip(),
        })
    return out


def _coerce_positive_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


def _parse_investigator_response(
    response: str, dim: dict[str, Any],
) -> dict[str, Any]:
    """Best-effort parse of an investigator's JSON output.

    Falls back to a low-confidence summary == raw text when JSON cannot
    be recovered, so the synthesizer always sees something usable.
    """
    # Robust JSON-object extraction — see ``_robust_json_object`` for
    # the strategy ladder.
    text = (response or "").strip()
    parsed = _robust_json_object(text)
    base = {
        "dimension_id": str(dim.get("id") or ""),
        "dimension_title": str(dim.get("title") or ""),
    }
    if parsed is None:
        return {
            **base,
            "summary": text or "(no investigator output)",
            "evidence": [],
            "issues": [],
            "confidence": "low",
        }
    summary = str(parsed.get("summary") or "").strip() or text
    evidence_raw = parsed.get("evidence") or []
    evidence: list[dict[str, Any]] = []
    if isinstance(evidence_raw, list):
        for item in evidence_raw:
            if isinstance(item, dict):
                evidence.append({
                    "path": str(item.get("path") or ""),
                    "lines": str(item.get("lines") or ""),
                    "excerpt": str(item.get("excerpt") or "")[:300],
                })
    issues_raw = parsed.get("issues") or []
    issues: list[dict[str, Any]] = []
    if isinstance(issues_raw, list):
        for item in issues_raw:
            if isinstance(item, dict):
                issues.append({
                    "severity": str(item.get("severity") or "medium").lower(),
                    "title": str(item.get("title") or "").strip(),
                    "detail": str(item.get("detail") or "").strip(),
                    "where": str(item.get("where") or "").strip(),
                })
    confidence = str(parsed.get("confidence") or "medium").lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"
    return {
        **base,
        "summary": summary,
        "evidence": evidence,
        "issues": issues,
        "confidence": confidence,
    }


_EMPTY_OUTPUT_MARKERS = (
    "(no investigator output)",
    "(empty)",
    "",
)


_DEPTH_RULE_MIN_FILE_READS: int = 3
_DEPTH_RULE_MIN_EVIDENCE: int = 2


# XML-style tool-call markers. These have **no legitimate use** in a
# final_text — they only appear when the LLM tried to write tool
# calls as text instead of via the API's tool_use mechanism. cc's
# QueryEngine doesn't parse XML in replies; it sees no tool_calls
# and exits. Any of these patterns means the LLM violated the
# tool-use protocol and the run is unrecoverable without a retry.
_XML_TOOL_MARKERS = (
    "<tool_call",       # Anthropic-style attempt
    "<tool_use",
    "<tool_calls",      # plural envelope
    "<file_read",       # bare tool tag
    "<glob",            # bare tool tag
    "<grep",            # bare tool tag
    "<function_call",   # OpenAI-style attempt
    "<function_calls",  # plural envelope
    "<invoke",          # Anthropic claude-code wire shape
    # DeepSeek's chat-template uses ``<｜｜DSML｜｜tool_calls>`` and
    # ``<｜｜DSML｜｜invoke name=...>`` (full-width ｜ U+FF5C, lowercased
    # to "<｜｜dsml｜｜"). When the model leaks these tokens into the
    # assistant TEXT instead of an API-level tool_use, cc's engine
    # cannot parse them — same failure mode as the XML tags above.
    # Observed during stock_rec_v3 doc run attempt 5: investigators
    # did 300+ tool rounds then started narrating DSML tool calls
    # in text and got classified as unparseable.
    "<｜｜dsml｜｜",
)

# Prose narration of intended-but-not-executed tool calls. Soft
# signal: when paired with thin evidence/issues this means the LLM
# stalled mid-investigation. With substantial evidence the prose may
# just be a recommendation in the summary, so we don't force shallow
# on prose alone.
_TEXT_TOOL_PLAN_PROSE = (
    # English
    "i'll read",
    "i will read",
    "i'll continue",
    "i will continue",
    "next i'll",
    "next i will",
    "i need to read",
    "i need to continue",
    "let me read",
    "let me continue",
    "continue reading",
    "continuing to read",
    # Chinese
    "继续阅读",
    "接下来读",
    "接下来读取",
    "下一步读取",
    "我会读取",
    "我将读取",
    "我会继续读",
    "我将继续读",
    "需要进一步",
    "需要继续",
    "需要再读",
    "我会继续",
    "接下来我会",
    "接下来我将",
)


_LEDGER_TOOLS = ("file_read", "grep", "glob")
_LEDGER_SOFT_LIMIT: int = 40


def _ledger_key_arg(tool: str, arguments: dict[str, Any]) -> str:
    if tool == "file_read":
        return str(arguments.get("file_path") or "")
    if tool in ("grep", "glob"):
        return str(arguments.get("pattern") or "")
    return ""


def _ledger_normalize_args(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Keep only the args that change the call's observable effect.

    Used as the dedup key alongside ``(tool, key)``. Default-equivalent
    values (None / 0 / empty / False) are dropped so callers that omit
    a kwarg and callers that pass the default collapse to the same
    entry.
    """
    if not isinstance(arguments, dict):
        return {}
    if tool == "file_read":
        mb = arguments.get("max_bytes")
        if isinstance(mb, int) and mb > 0:
            return {"max_bytes": mb}
        return {}
    if tool == "grep":
        out: dict[str, Any] = {}
        for k in ("cwd", "glob", "file_type", "files_only", "context_lines", "max_results"):
            v = arguments.get(k)
            if v in (None, "", 0, False):
                continue
            out[k] = v
        return out
    if tool == "glob":
        out = {}
        for k in ("cwd", "max_results"):
            v = arguments.get(k)
            if v in (None, "", 0):
                continue
            out[k] = v
        return out
    return {}


def _ledger_result_summary(tool: str, metadata: dict[str, Any]) -> dict[str, Any]:
    """Extract the meaningful result fields from a tool_result message's
    ``metadata`` dict (which mirrors ``ToolResult.data``)."""
    if not isinstance(metadata, dict):
        return {}
    if tool == "file_read":
        size = metadata.get("size")
        return {
            "size": int(size) if isinstance(size, int) else None,
            "truncated": bool(metadata.get("truncated", False)),
        }
    if tool in ("grep", "glob"):
        count = metadata.get("count")
        return {
            "count": int(count) if isinstance(count, int) else None,
            "truncated": bool(metadata.get("truncated", False)),
        }
    return {}


def _ledger_record_call(
    ledger: list[dict[str, Any]],
    *,
    tool: str,
    key: str,
    args: dict[str, Any],
    result: dict[str, Any] | None = None,
) -> None:
    """Dedupe-add a call into ``ledger``. If an entry with the same
    ``(tool, key, args)`` already exists, bump its ``count`` and merge
    in the result if the existing entry didn't capture one yet."""
    for entry in ledger:
        if (
            entry.get("tool") == tool
            and entry.get("key") == key
            and entry.get("args") == args
        ):
            entry["count"] = int(entry.get("count", 1)) + 1
            if result and not entry.get("result"):
                entry["result"] = result
            return
    ledger.append({
        "tool": tool,
        "key": key,
        "args": dict(args),
        "result": dict(result) if result else {},
        "count": 1,
    })


def _ledger_attach_result(
    ledger: list[dict[str, Any]],
    *,
    tool_use_id: str,
    pending: dict[str, str],
    metadata: dict[str, Any],
) -> None:
    """Look up the ledger entry created for ``tool_use_id`` and merge
    the result summary into it. ``pending`` maps tool_use_id → an
    identity string ("tool|key|args_repr") so we can find the right
    entry even when multiple entries with the same (tool, key) but
    different args exist."""
    identity = pending.pop(tool_use_id, None)
    if identity is None:
        return
    tool, key, args_repr = identity.split("\x00", 2)
    summary = _ledger_result_summary(tool, metadata)
    if not summary:
        return
    for entry in ledger:
        if (
            entry.get("tool") == tool
            and entry.get("key") == key
            and repr(sorted((entry.get("args") or {}).items())) == args_repr
        ):
            if not entry.get("result"):
                entry["result"] = summary
            return


def _ledger_identity(tool: str, key: str, args: dict[str, Any]) -> str:
    return f"{tool}\x00{key}\x00{repr(sorted(args.items()))}"


def _format_ledger_for_prompt(
    ledger: list[dict[str, Any]],
    *,
    language: str,
    limit: int = _LEDGER_SOFT_LIMIT,
) -> str:
    """Render the ledger as a markdown block for retry_feedback.

    Sorts by repeat ``count`` descending so the highest-redundancy
    calls appear first — those are the ones the next attempt must
    avoid. Truncates to ``limit`` entries (top by count)."""
    if not ledger:
        return ""
    is_zh = language.startswith("zh")
    ranked = sorted(ledger, key=lambda e: -int(e.get("count", 1)))
    shown = ranked[:limit]
    lines: list[str] = []
    for entry in shown:
        tool = str(entry.get("tool") or "")
        key = str(entry.get("key") or "")
        args = entry.get("args") or {}
        result = entry.get("result") or {}
        count = int(entry.get("count", 1))
        args_bits: list[str] = []
        for k in sorted(args):
            v = args[k]
            args_bits.append(f"{k}={v!r}" if isinstance(v, str) else f"{k}={v}")
        args_str = (" " + " ".join(args_bits)) if args_bits else ""
        if tool == "file_read":
            size = result.get("size")
            if size is None:
                result_str = "(no result captured)"
            else:
                result_str = f"{size} bytes" + (" (truncated)" if result.get("truncated") else "")
        elif tool in ("grep", "glob"):
            count_v = result.get("count")
            if count_v is None:
                result_str = "(no result captured)"
            else:
                result_str = f"{count_v} hits" + (" (truncated)" if result.get("truncated") else "")
        else:
            result_str = ""
        if count > 1:
            if is_zh:
                count_str = f"（上次累计调用 {count} 次，重复 {count - 1} 次）"
            else:
                count_str = f" (called {count} times last attempt — {count - 1} redundant)"
        else:
            count_str = ""
        lines.append(f"- `{tool}` {key}{args_str} → {result_str}{count_str}")
    body = "\n".join(lines)
    n_total = len(ledger)
    n_shown = len(shown)
    if is_zh:
        header = (
            "## 上一次尝试已发起过这些工具调用——本次禁止重复\n\n"
            "下面这些 (tool, key, args) 组合上次已经产生过结果。**本次 "
            "attempt 不要再发起同样的组合**，否则会被判 shallow。需要更多\n"
            "内容时，根据 `result` 摘要决定下一步（read 加大 ``max_bytes`` "
            "/ grep 改 pattern / 直接进 stage 3 出 JSON）。"
        )
        if n_shown < n_total:
            header += f"\n（共 {n_total} 条，按重复次数排序，仅展示前 {n_shown} 条）"
    else:
        header = (
            "## Tool calls already made last attempt — do NOT repeat\n\n"
            "Each (tool, key, args) below already produced a result last "
            "attempt. This attempt MUST NOT issue the same combination "
            "again — repeats are scored shallow. Use the `result` summary "
            "to decide the next move (widen `file_read` ``max_bytes`` / "
            "change grep pattern / go to Stage 3 and emit JSON now)."
        )
        if n_shown < n_total:
            header += (
                f"\n({n_total} total, sorted by repeat count, "
                f"top {n_shown} shown)"
            )
    return f"{header}\n\n{body}"


# Citation-format patterns for prose with file references but no
# strict ``foo.py:42`` colon syntax. Used by
# ``_has_file_with_line_evidence`` to decide whether prose-to-JSON
# conversion is worth firing. Designed to catch the formats real
# investigators actually emit:
#   * ``foo.py:42``                   canonical
#   * ``foo.py 第 21-29 行``          Chinese
#   * ``foo.py 第21行``               Chinese (no spaces)
#   * ``foo.py 行 21-29``             Chinese (alt)
#   * ``foo.py L21`` / ``L21-L29``    GitHub permalink style
#   * ``foo.py (line 42)`` / ``(lines 42-45)``  English prose
#   * ``foo.py @ 42``                 occasional shorthand
import re as _re_for_citations
_FILE_PATH_RE = _re_for_citations.compile(
    r"[A-Za-z_][\w/.\\-]*\.(?:py|md|ts|tsx|js|jsx|json|toml|yaml|yml|sql|sh|cfg|ini)"
)
_LINE_INDICATOR_PATTERNS = (
    _re_for_citations.compile(r":\d+"),                              # :42
    _re_for_citations.compile(r"第\s*\d+\s*[-–至到~]?\s*\d*\s*行"),    # 第 21-29 行
    _re_for_citations.compile(r"行\s*\d+"),                           # 行 21
    _re_for_citations.compile(r"\bL\d+(?:\s*[-–]\s*L?\d+)?"),        # L21 / L21-L29
    _re_for_citations.compile(r"\b[Ll]ines?\s+\d+"),                  # line 21 / lines 21-29
    _re_for_citations.compile(r"\(\s*[Ll]ines?\s+\d+"),               # (line 21)
    _re_for_citations.compile(r"@\s*\d+"),                            # @ 42
)


def _has_file_with_line_evidence(text: str) -> bool:
    """Return True when ``text`` cites at least one source file AND
    at least one line-number indicator (anywhere in the prose).

    Permissive on purpose — see ``_LINE_INDICATOR_PATTERNS`` for the
    formats accepted. The two checks are independent: as long as the
    prose mentions any ``.py``/``.md``/etc path AND has any
    line-number pattern, we treat the prose as worth restructuring.
    Conservative variants (require co-location within ~80 chars) were
    rejected because real LLM output often introduces files by name
    in one paragraph and cites their lines in the next.
    """
    if not text:
        return False
    if not _FILE_PATH_RE.search(text):
        return False
    return any(p.search(text) for p in _LINE_INDICATOR_PATTERNS)


def _detect_xml_tool_markers(text: str) -> bool:
    """Return True when ``text`` contains XML-style tool-call tags.

    This is the **strict** half of text-plan detection — any of these
    markers means the LLM tried to use XML tool syntax in its reply
    instead of the API's tool_use mechanism. cc cannot parse those,
    so the work is incomplete by definition. ALWAYS triggers shallow
    regardless of evidence count.
    """
    if not text:
        return False
    haystack = text.lower()
    return any(m in haystack for m in _XML_TOOL_MARKERS)


def _detect_text_tool_plans(text: str) -> bool:
    """Return True when ``text`` contains XML markers OR prose hints
    that the LLM described future tool calls as text rather than
    emitting them.

    Combined detector kept for back-compat with earlier code paths.
    Prefer ``_detect_xml_tool_markers`` for strict protocol checks.
    """
    if not text:
        return False
    if _detect_xml_tool_markers(text):
        return True
    haystack = text.lower()
    return any(p in haystack for p in _TEXT_TOOL_PLAN_PROSE)


def _classify_investigator_outcome(
    *,
    final_text: str,
    findings: dict[str, Any],
    tool_call_count: int,
    file_read_count: int = 0,
) -> str:
    """Classify how an investigator turn went.

    Returns one of:
      * ``"ok"``         — produced parseable JSON AND satisfied the
                           depth rule (≥ ``_DEPTH_RULE_MIN_FILE_READS``
                           file_read calls OR ≥
                           ``_DEPTH_RULE_MIN_EVIDENCE`` evidence
                           entries OR at least one populated ``issues``
                           entry) AND did NOT stall on a text-form
                           tool plan.
      * ``"shallow"``    — parsed JSON but did not satisfy the depth
                           rule, OR stopped with text describing
                           future tool calls instead of executing
                           them ("I'll continue reading X, Y, Z" or
                           ``<tool_call>...`` XML in the reply).
      * ``"unparseable"``— had text but JSON didn't parse; the summary
                           is the raw response.
      * ``"empty"``      — no usable output at all (final_text empty
                           or only the placeholder).

    The depth rule mirrors the system prompt: "Stage 2 — READ (HARD
    MINIMUM: ≥3 file_read calls)". Counting ``file_read`` specifically
    catches "spammed grep then bailed" patterns that a generic
    ``tool_call_count`` threshold misses. The text-form-tool-plan
    detector catches the trickier case where the LLM uses tools
    correctly for a few rounds, then narrates the rest as prose
    instead of continuing.
    """
    summary = str(findings.get("summary") or "").strip()
    has_evidence = bool(findings.get("evidence"))
    has_issues = bool(findings.get("issues"))
    raw = (final_text or "").strip()
    if not raw or summary in _EMPTY_OUTPUT_MARKERS and not has_evidence and not has_issues:
        return "empty"

    # XML-style tool markers (``<tool_call>``, ``<file_read>``, etc.)
    # are an UNAMBIGUOUS protocol violation — the LLM tried to write
    # tool calls as text but cc's QueryEngine only sees real
    # tool_use blocks. Force shallow regardless of any evidence/issues
    # that might be present (those entries are likely incomplete or
    # speculative since the LLM thought it was still mid-investigation).
    if _detect_xml_tool_markers(raw):
        return "shallow"

    # Heuristic: parse succeeded if either evidence/issues were
    # populated (those only come from a parsed JSON object) OR the
    # summary differs from the raw text (parser overrides).
    parsed_ok = has_evidence or has_issues or (summary and summary != raw)
    if not parsed_ok:
        return "unparseable"

    # Prose-form text tool plan check (soft signal): even when
    # file_read_count looks adequate, the LLM might have narrated
    # future reads as English/Chinese prose rather than executing
    # them. With thin evidence/issues this means the work was cut
    # short. With substantial evidence the prose may just be a
    # summary recommendation, so don't force shallow on prose alone.
    evidence_count = len(findings.get("evidence") or [])
    issue_count = len(findings.get("issues") or [])
    text_plan_stall = _detect_text_tool_plans(raw)
    if text_plan_stall and evidence_count < _DEPTH_RULE_MIN_EVIDENCE and issue_count < 1:
        # The LLM stalled with a text plan AND didn't produce real
        # output. Even if file_read_count >= 3, the work was
        # incomplete — the LLM clearly thought there was more to do.
        return "shallow"

    # Depth rule. The investigator passes if any of:
    #   * It read enough files (Stage 2 hard minimum)
    #   * It cited at least N evidence entries (good output)
    #   * It found at least one concrete issue (work was done)
    if (
        file_read_count >= _DEPTH_RULE_MIN_FILE_READS
        or evidence_count >= _DEPTH_RULE_MIN_EVIDENCE
        or issue_count >= 1
    ):
        return "ok"
    # Total tool calls ≤ 2 with empty findings is the classic
    # "globbed and gave up" pattern.
    if tool_call_count <= 2:
        return "shallow"
    # Otherwise: had several tool calls (e.g. 6 greps) but never read
    # files and didn't produce concrete output. Same shallow bucket.
    return "shallow"


def _render_findings_for_synth(findings: list[dict[str, Any]]) -> str:
    """Render the collected investigator findings as a deterministic
    Markdown block fed to the synthesizer prompt.

    Language is intentionally **end-user neutral**. The synthesizer's
    output is read by a developer reviewing the codebase, not by
    someone debugging the agent. Words like ``investigator``, ``tool
    calls``, ``shallow``, ``unparseable`` therefore do NOT appear in
    this rendering — the synthesizer should never see them and so
    cannot echo them back into the report. The only signal we carry
    forward is ``Coverage:`` (full / partial / limited / none) plus
    the actual evidence/issues when present.
    """
    chunks: list[str] = []
    for f in findings:
        title = f.get("dimension_title") or f.get("dimension_id") or "(unnamed)"
        status = str(f.get("status") or "ok")
        chunks.append(f"### {title}")
        if status == "empty":
            chunks.append(
                "Coverage: none — no observations were collected for "
                "this dimension. Acknowledge briefly in the final "
                "report (e.g. 'this area requires a focused follow-up'); "
                "do NOT invent findings or speculate."
            )
            chunks.append("")
            continue
        if status == "error":
            # The investigation never ran to completion (runtime
            # failure). Same instruction as ``empty`` — and do NOT
            # surface the raw exception text (process-internal) to
            # the synthesizer.
            chunks.append(
                "Coverage: none — this dimension could not be "
                "investigated in this run. Acknowledge briefly in the "
                "final report (e.g. 'this area requires a focused "
                "follow-up'); do NOT invent findings or speculate."
            )
            chunks.append("")
            continue
        if status == "unparseable":
            chunks.append(
                "Coverage: limited — observations are not in a "
                "confidently structured form. Use only what is "
                "concretely supported by the summary or cited "
                "evidence below; treat the rest as inconclusive."
            )
        elif status == "shallow":
            chunks.append(
                "Coverage: limited — only a high-level read; concrete "
                "evidence is sparse. Treat the summary as a hint; in "
                "the final report flag this dimension as needing a "
                "deeper review rather than presenting it as verified."
            )
        else:
            chunks.append(f"Coverage: full | Confidence: {f.get('confidence', 'medium')}")
        chunks.append(f"Summary: {f.get('summary', '')}")
        issues = f.get("issues") or []
        if issues:
            chunks.append("Issues:")
            for issue in issues:
                where = issue.get("where") or ""
                where_part = f" ({where})" if where else ""
                sev = issue.get("severity", "medium")
                chunks.append(
                    f"  - [{sev}] {issue.get('title', '')}: "
                    f"{issue.get('detail', '')}{where_part}"
                )
        evidence = f.get("evidence") or []
        if evidence:
            chunks.append("Evidence:")
            for ev in evidence:
                lines = ev.get("lines") or ""
                excerpt = (ev.get("excerpt") or "").replace("\n", " ")[:200]
                chunks.append(
                    f"  - {ev.get('path', '')}:{lines} — {excerpt}"
                )
        chunks.append("")
    return "\n".join(chunks)


def _slug(text: str, *, idx: int) -> str:
    """Lightweight fallback slug when the LLM omits a dimension id."""
    cleaned = "".join(c if c.isalnum() else "-" for c in text.lower()).strip("-")
    cleaned = "-".join(filter(None, cleaned.split("-")))
    if not cleaned:
        return f"dim-{idx}"
    return cleaned[:32]


__all__ = [
    "DocModeRunner",
    "_parse_decompose_response",
    "_parse_investigator_response",
    "_parse_surveyor_response",
    "_format_survey_for_prompt",
    "_render_findings_for_synth",
]
