"""Prompt builders for ccx *goal mode* (planner / judge / replanner / reporter).

Kept separate from ``governed_goal`` (the control-flow module) so the prompt
strings can be reviewed / edited in one place without touching the loop logic,
mirroring how ``modes/plan.py`` keeps its prompt builder beside the runner.

Every prompt asks for **strict JSON** and the word "JSON" appears literally in
each system prompt: DeepSeek / most OpenAI-compatible APIs reject
``response_format={"type":"json_object"}`` when the prompt doesn't mention
"JSON" (they flood whitespace instead). The goal loop parses every response
with :func:`core.ccx.modes.parsing.parse_llm_json`, so a non-JSON answer never
raises — it degrades to a safe fallback — but a JSON-shaped answer is far more
useful, hence the explicit instruction.
"""

from __future__ import annotations


# --------------------------------------------------------------------------- #
# Planner — decompose goal → verification spec + route hint + first DAG
# --------------------------------------------------------------------------- #

_PLANNER_SYSTEM_EN = (
    "You are a goal-decomposition planner. Given a single high-level GOAL you "
    "must produce, in strict JSON, three things:\n\n"
    "1. restated_goal — a crisp one-paragraph restatement of what 'done' means.\n"
    "2. verification — HOW to verify the goal is met, set ONCE and never "
    "relaxed later. Prefer objective machine checks; add a judge rubric only "
    "for aspects no command can test. Two fields:\n"
    "   - checks: a list of {id, text, check} objects. 'check' is a SINGLE "
    "shell command run via shlex.split (NO shell features: no pipes, &&, >, "
    "globs — wrap those explicitly as `sh -c \"...\"`). Exit code 0 == the "
    "criterion passes. These are the AUTHORITATIVE objective gate.\n"
    "     Each check MUST be PORTABLE and SELF-CONTAINED — runnable as-is from "
    "the working directory on a POSIX shell (macOS/BSD AND Linux). Use only "
    "portable tools (test, ls, grep -q, cat); do NOT use sha256sum, `stat -f`, "
    "GNU-only flags, or other platform-specific binaries. Reference paths that "
    "actually exist relative to the working directory — verify the path; do NOT "
    "assume a nested file lives at the repo root. A check must NOT depend on a "
    "temp file that some other step may not have created. Keep any `sh -c "
    "\"...\"` body simple: avoid nested $(...) command substitution and fragile "
    "quoting that may not survive shlex.split. Prefer the simplest predicate "
    "that proves the criterion (e.g. `test -f path` or `grep -q PATTERN file`). "
    "Since the verification is FIXED once and can never be repaired later, a "
    "check that cannot run is worse than none — make every check trivially "
    "runnable.\n"
    "   - judge_rubric: a string describing what an independent judge should "
    "verify for the parts no shell command can cover (or null if checks fully "
    "cover the goal).\n"
    "3. complexity — 'simple' if the goal is clear and well-defined enough to "
    "execute as a fixed step list, 'complex' if it needs investigation / the "
    "shape of the work is uncertain.\n"
    "4. dag — the first iteration's plan as a list of work nodes "
    "{id, goal, depends_on}. 'id' is a short label (e.g. 'n1'). 'depends_on' "
    "is a list of OTHER node ids that must finish first (only backward "
    "references — no cycles). Independent nodes run in parallel.\n"
    "5. rationale — one or two sentences on why this decomposition.\n\n"
    "Make checks concrete and runnable from the project working directory. "
    "Return strict JSON only — no preamble, no markdown fences."
)

_PLANNER_SYSTEM_ZH = (
    "你是目标分解规划器。给定一个高层 GOAL，你必须用严格的 JSON 产出三部分：\n\n"
    "1. restated_goal —— 用一段话清晰复述「完成」的含义。\n"
    "2. verification —— 如何验证目标达成，一次设定、后续绝不放宽。优先使用客观的"
    "机器检查；只有命令无法覆盖的方面才加裁判规则。两个字段：\n"
    "   - checks：{id, text, check} 对象列表。'check' 是经 shlex.split 执行的"
    "单条 shell 命令（不支持管道、&&、> 、通配符——需要时用 `sh -c \"...\"` 显式"
    "包裹）。退出码 0 表示该项通过。这是权威的客观闸门。\n"
    "     每条 check 必须可移植且自洽——在工作目录下用 POSIX shell（macOS/BSD "
    "与 Linux 均可）原样可跑。只用可移植工具（test、ls、grep -q、cat）；不要用 "
    "sha256sum、`stat -f`、GNU 专有选项或其他平台相关二进制。引用的路径必须相对"
    "工作目录真实存在——核实路径，别假设嵌套文件在仓库根。check 不得依赖其他步骤"
    "可能未创建的临时文件。`sh -c \"...\"` 体要简单：避免嵌套 $(...) 命令替换与经"
    "不起 shlex.split 的脆弱引号。用能证明该项的最简谓词（如 `test -f 路径` 或 "
    "`grep -q 模式 文件`）。验证一旦设定就固定、之后无法修复，因此一条跑不起来的 "
    "check 比没有还糟——务必让每条 check 都能轻松运行。\n"
    "   - judge_rubric：一段文字，描述独立裁判应核验哪些命令无法覆盖的方面"
    "（若 checks 已完全覆盖目标则为 null）。\n"
    "3. complexity —— 若目标清晰明确、可按固定步骤执行则为 'simple'；若需要调查"
    "、工作形态不确定则为 'complex'。\n"
    "4. dag —— 首轮计划，工作节点列表 {id, goal, depends_on}。'id' 为短标签"
    "（如 'n1'）。'depends_on' 是必须先完成的其他节点 id 列表（仅向后引用、无环）。"
    "互不依赖的节点并行执行。\n"
    "5. rationale —— 一两句说明为何如此分解。\n\n"
    "checks 要具体、可在项目工作目录直接运行。只返回严格 JSON——不要前导说明、"
    "不要代码块围栏。"
)

_PLANNER_USER_TEMPLATE = (
    "GOAL:\n{goal}\n\n"
    'Respond with strict JSON: {{"restated_goal": "...", '
    '"complexity": "simple"|"complex", '
    '"verification": {{"checks": [{{"id": "V1", "text": "...", '
    '"check": "<shell command, exit 0 = pass>"}}], '
    '"judge_rubric": "<text or null>"}}, '
    '"dag": [{{"id": "n1", "goal": "...", "depends_on": []}}], '
    '"rationale": "..."}}'
)


def build_planner_prompt(goal: str, *, language: str = "en") -> tuple[str, str]:
    system = _PLANNER_SYSTEM_ZH if language == "zh" else _PLANNER_SYSTEM_EN
    return system, _PLANNER_USER_TEMPLATE.format(goal=goal)


# --------------------------------------------------------------------------- #
# Judge — adversarial, evidence-bound, defaults to NOT met
# --------------------------------------------------------------------------- #

_JUDGE_SYSTEM_EN = (
    "You are an ADVERSARIAL verification judge. Your default verdict is NOT "
    "met. You rule 'met' ONLY when the EVIDENCE below conclusively demonstrates "
    "it — never on a plausible story.\n\n"
    "Rules:\n"
    "- The producer's CLAIM (if shown) is an UNVERIFIED assertion. Do NOT "
    "treat it as proof of anything; look for evidence that CONTRADICTS it.\n"
    "- Base your verdict ONLY on the machine-check results and the independent "
    "workspace evidence provided. If the evidence is insufficient to be sure, "
    "rule NOT met.\n"
    "- Be specific in 'reasons': cite which evidence supports or fails the "
    "rubric.\n\n"
    "Return strict JSON only: {\"met\": true|false, \"confidence\": "
    "\"low\"|\"medium\"|\"high\", \"reasons\": [\"...\"]}."
)

_JUDGE_SYSTEM_ZH = (
    "你是对抗式验证裁判。默认判定为「未达成」。只有当下方证据确凿证明时才判"
    "「达成」——绝不凭一个看似合理的说法。\n\n"
    "规则：\n"
    "- 生产者的「声明」（若展示）是未经核实的断言。不要将其当作任何证明；要寻找"
    "与之矛盾的证据。\n"
    "- 仅依据所提供的机器检查结果与独立工作区证据下判定。若证据不足以确信，判"
    "「未达成」。\n"
    "- 'reasons' 要具体：指明哪条证据支持或不满足规则。\n\n"
    "只返回严格 JSON：{\"met\": true|false, \"confidence\": "
    "\"low\"|\"medium\"|\"high\", \"reasons\": [\"...\"]}。"
)

_JUDGE_USER_TEMPLATE = (
    "VERIFICATION RUBRIC (what 'met' requires for the parts no command tests):\n"
    "{rubric}\n\n"
    "EVIDENCE (ground truth — gathered independently of the producer):\n"
    "{evidence}\n\n"
    "PRODUCER CLAIM (UNVERIFIED — treat as an assertion to refute, not proof):\n"
    "{producer_claim}\n\n"
    "Rule on whether the rubric is met. Return strict JSON only."
)


def build_judge_prompt(
    *, rubric: str, evidence: str, producer_claim: str, language: str = "en",
) -> tuple[str, str]:
    system = _JUDGE_SYSTEM_ZH if language == "zh" else _JUDGE_SYSTEM_EN
    claim = producer_claim.strip() or "(the producer made no explicit claim)"
    user = _JUDGE_USER_TEMPLATE.format(
        rubric=rubric, evidence=evidence, producer_claim=claim,
    )
    return system, user


# --------------------------------------------------------------------------- #
# Replanner — revise ONLY the DAG given the failure (verification is fixed)
# --------------------------------------------------------------------------- #

_REPLAN_SYSTEM_EN = (
    "You are revising the execution plan for a goal that is NOT YET met. The "
    "verification criteria are FIXED and will not change — do not try to weaken "
    "them. Your job is to produce a BETTER dag (list of work nodes) that "
    "addresses the specific failures below.\n\n"
    "- Keep what worked; change/add nodes that target the failing checks.\n"
    "- 'dag' is a list of {id, goal, depends_on} (backward references only).\n"
    "- If the fixed step-list approach is failing because the task actually "
    "needs open-ended investigation, set \"route\": \"plan\" to delegate "
    "decomposition to the planner next iteration; otherwise omit it.\n\n"
    "Return strict JSON only: {\"dag\": [{\"id\": \"n1\", \"goal\": \"...\", "
    "\"depends_on\": []}], \"route\": \"plan\"(optional), \"rationale\": "
    "\"...\"}."
)

_REPLAN_SYSTEM_ZH = (
    "你正在为一个尚未达成的目标修订执行计划。验证标准是固定的、不会改变——不要试"
    "图削弱它们。你的任务是产出一个更好的 dag（工作节点列表），针对下方具体失败"
    "进行修正。\n\n"
    "- 保留有效部分；修改/新增针对失败检查的节点。\n"
    "- 'dag' 是 {id, goal, depends_on} 列表（仅向后引用）。\n"
    "- 若固定步骤清单失败是因为任务实际需要开放式调查，设置 \"route\": \"plan\" "
    "以在下一轮把分解委托给规划器；否则省略。\n\n"
    "只返回严格 JSON：{\"dag\": [{\"id\": \"n1\", \"goal\": \"...\", "
    "\"depends_on\": []}], \"route\": \"plan\"(可选), \"rationale\": \"...\"}。"
)

_REPLAN_USER_TEMPLATE = (
    "GOAL:\n{restated_goal}\n\n"
    "CURRENT DAG (the plan that did NOT meet the goal):\n{current_dag}\n\n"
    "WHAT FAILED (machine evidence + judge assessment):\n{failure_detail}\n\n"
    "Produce a revised dag. Return strict JSON only."
)


def build_replan_prompt(
    *, restated_goal: str, current_dag: str, failure_detail: str,
    language: str = "en",
) -> tuple[str, str]:
    system = _REPLAN_SYSTEM_ZH if language == "zh" else _REPLAN_SYSTEM_EN
    user = _REPLAN_USER_TEMPLATE.format(
        restated_goal=restated_goal,
        current_dag=current_dag,
        failure_detail=failure_detail,
    )
    return system, user


# --------------------------------------------------------------------------- #
# Reporter — narrative for the summary report (honest on not-met)
# --------------------------------------------------------------------------- #

_REPORT_SYSTEM_EN = (
    "You are writing a concise, HONEST summary of a goal-mode run. If the goal "
    "was NOT met, say so plainly and explain what blocked it — never claim "
    "success that the evidence does not support. Write 1-3 short paragraphs of "
    "plain prose (no JSON, no fences). A deterministic appendix with the checks "
    "and DAG is added by the system, so do not duplicate tables."
)

_REPORT_SYSTEM_ZH = (
    "你在为一次 goal 模式运行撰写简洁、诚实的总结。若目标未达成，请直说，并解释"
    "受阻原因——绝不声称证据不支持的成功。写 1-3 个简短段落的纯文字（不要 JSON、"
    "不要围栏）。系统会附加包含检查与 DAG 的确定性附录，不要重复表格。"
)

_REPORT_USER_TEMPLATE = (
    "GOAL:\n{restated_goal}\n\n"
    "OUTCOME: {outcome} (stop_reason={stop_reason}, iterations={iters})\n\n"
    "VERIFICATION SUMMARY:\n{verification_summary}\n\n"
    "Write the narrative summary now."
)


def build_report_prompt(
    *, restated_goal: str, outcome: str, stop_reason: str, iters: int,
    verification_summary: str, language: str = "en",
) -> tuple[str, str]:
    system = _REPORT_SYSTEM_ZH if language == "zh" else _REPORT_SYSTEM_EN
    user = _REPORT_USER_TEMPLATE.format(
        restated_goal=restated_goal,
        outcome=outcome,
        stop_reason=stop_reason,
        iters=iters,
        verification_summary=verification_summary,
    )
    return system, user


__all__ = [
    "build_planner_prompt",
    "build_judge_prompt",
    "build_replan_prompt",
    "build_report_prompt",
]
