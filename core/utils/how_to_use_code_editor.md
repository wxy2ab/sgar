# 代码编辑器选用与使用指南

> 配套文档：[code_editor.md](code_editor.md)（编辑器全景与重构记录）。
> 本文档只回答两个问题：**什么场景用哪个编辑器**，**这个编辑器怎么用**。

---

## 0. 5 秒决策表

| 你想做什么 | 用哪个 |
|---|---|
| **不确定从哪开始 / 想要最简对外接口** | [`CodeAgent`（cc 顶层 facade）](#0-codeagentrun_code_agentbuild_code_with_agentcc-顶层-facade) |
| 通过自然语言指令驱动 agent 改代码（带工具/权限/会话） | [`CodeAgent.run` / `run_code_agent`](#01-codeagent--run_code_agent单轮指令) |
| 给定 goal + constraints 做结构化代码生成 | [`CodeAgent.build_code` / `build_code_with_agent`](#02-codeagentbuild_code--build_code_with_agentgoal--约束) |
| 流式拿事件（增量打字/工具调用） | [`CodeAgent.stream`](#03-codeagentstream流式事件) |
| 上层 agent 内存里改代码（一次或多次小修改） | [`InlineCodeEditor`](#1-inlinecodeeditor) |
| agent runtime 改磁盘上的文件（hash 保护 + 权限 + 持久 rollback） | [`CodeEditFacade`](#2-codeeditfacade) |
| 单文件 LLM 自由编辑（"把这个函数改成 X"） | [`RobustLLMEditor`](#3-robustllmeditor) |
| 多文件 / 仓库级 LLM 编辑 | [`SmartLLMEditorV2`](#4-smartllmeditorv2) |
| 让 agent 反复 edit→run→fix 直到通过验证 | [`AutonomousCodeAgent`](#5-autonomouscodeagent) |
| 已经拿到结构化的块编辑指令字符串想直接 apply | [`LineNumberFreeLLMBlockEditor`](#6-linenumberfreeellmblockeditor) |
| 想要"一个不行换另一个"的双引擎兜底 | [`FallbackLLMEditor`](#7-fallbackllmeditor) |
| 单纯做行级精确替换（legacy LLM 模式） | [`LLMCodeEditor`](#8-llmcodeeditor) |

### 编辑器层次（从高到低）

```
            CodeAgent (顶层 facade — 自然语言指令、工具调度、会话/权限)
                │  内部走 file_edit / file_write 工具
                ▼
            CodeEditFacade  (磁盘级精确编辑 — hash + 验证 + rollback)
                │
   ┌────────────┼─────────────┐
   ▼            ▼             ▼
InlineCodeEditor  RobustLLMEditor   SmartLLMEditorV2
(内存版精确)      (单文件 LLM)       (多文件 LLM)
                    ▲
                    │ 用作底盘
              AutonomousCodeAgent (闭环 edit-run-fix)
```

> **选用建议**：
> - 业务方（外部接入 / 任务驱动）→ 直接用 `CodeAgent`，最稳的对外接口；
> - 框架内部（agent / planner 实现者）→ 视场景选 `InlineCodeEditor` / `RobustLLMEditor` / `SmartLLMEditorV2` / `AutonomousCodeAgent`；
> - 工具层（要写自己的 file_edit 工具）→ 直接用 `CodeEditFacade`。

---

## 0. CodeAgent / `run_code_agent` / `build_code_with_agent`（cc 顶层 facade）

> 位置：`core/cc/api.py`，从 `core.cc` 直接导出。

### 适用场景
- **外部项目 / 任务方**接入 cc 编辑能力的**首选 / 推荐入口**；
- 想用**自然语言指令**驱动一整个 agent 跑通"读文件→编辑→运行→输出"全流程；
- 需要会话管理、工具调度、权限分级、流式事件、审计追踪等"全套" agent 能力；
- 不想自己拼装 `QueryEngine` / `Session` / `LLMClientProvider` 等底层组件。

### 关键 API

| 名字 | 作用 |
|---|---|
| `CodeAgent` | 主类，封装 config + LLM provider，提供 run / stream / build_code 入口 |
| `AgentRunRequest` | 单轮指令请求 |
| `AgentRunResult` | 运行结果（含事件、消息、final_text） |
| `CodeBuildRequest` | 结构化代码生成请求（goal + 约束 + 验收标准） |
| `run_code_agent(...)` | 同步函数式包装，等价 `CodeAgent().run_sync(...)` |
| `build_code_with_agent(...)` | 同步函数式包装，等价 `CodeAgent().build_code_sync(...)` |

### 关键概念
- 内部通过 **`file_edit` / `file_write` 工具**让 LLM 触发 `CodeEditFacade.apply_precise_edit`；调用方不需要直接接触 facade。
- `permission_mode` 决定权限边界：`"default"` / `"plan"` / `"accept_edits"` / `"bypass_permissions"`。
- `agent_mode` 选择运行模式：默认事件流模式；`"structured"` 模式不支持 stream，只能 run。
- `prompt_language` 指定 system prompt 语言（`"zh"` / `"en"`）。

### 0.1 `CodeAgent.run` / `run_code_agent`（单轮指令）

#### 异步 stream 写法（推荐生产）
```python
import asyncio
from core.cc import AgentRunRequest, CCConfig, CodeAgent

async def main():
    config = CCConfig(
        prompt_language="zh",
        permission_mode="default",
        default_llm_client="SimpleDeepSeekClientReasoning",
    )
    agent = CodeAgent(config=config)

    result = await agent.run(AgentRunRequest(
        instruction="把 src/foo.py 里所有 print(...) 换成 logger.info(...)",
        cwd=".",
    ))

    print(result.final_text)
    print(f"used {result.tool_call_count} tool calls, failed={result.failed}")

asyncio.run(main())
```

#### 同步函数式（脚本最简形态）
```python
from core.cc import run_code_agent, CCConfig

result = run_code_agent(
    instruction="给 calculate_alpha 加 logging",
    cwd=".",
    config=CCConfig(prompt_language="zh", permission_mode="accept_edits"),
)
print(result.final_text)
```

### 0.2 `CodeAgent.build_code` / `build_code_with_agent`（goal + 约束）

> 适合**自动生成新代码**（不仅是修改），可以传入约束和验收标准。

```python
import asyncio
from core.cc import CodeAgent, CodeBuildRequest, CCConfig

async def main():
    agent = CodeAgent(config=CCConfig(prompt_language="zh"))
    result = await agent.build_code(CodeBuildRequest(
        goal="生成一个轻量级 CLI，从 stdin 读 JSON 并打印 schema",
        cwd="./build_out",
        constraints=[
            "只用标准库",
            "脚本控制在 80 行内",
        ],
        acceptance_criteria=[
            "Python 3.10+ 可运行",
            "支持嵌套对象的 schema 推断",
        ],
    ))
    print(result.final_text)

asyncio.run(main())
```

同步形式：
```python
from core.cc import build_code_with_agent

result = build_code_with_agent(
    goal="...",
    cwd="./out",
    constraints=["..."],
    acceptance_criteria=["..."],
)
```

### 0.3 `CodeAgent.stream`（流式事件）

```python
import asyncio
from core.cc import AgentRunRequest, CCConfig, CodeAgent

async def main():
    agent = CodeAgent(config=CCConfig(prompt_language="zh"))
    request = AgentRunRequest(
        instruction="重构 utils/parser.py 让它支持流式输入",
        cwd=".",
    )
    async for event in agent.stream(request):
        # event 是 SessionEvent，包含 turn_id / message 等
        if event.message and event.message.role == "assistant":
            print(event.message.content, end="", flush=True)
        elif event.message and event.message.kind == "tool_call":
            print(f"\n[tool] {event.message.content}")

asyncio.run(main())
```

### 0.4 自定义 LLM Provider

```python
from core.cc import CodeAgent, LLMClientProvider

class MyProvider(LLMClientProvider):
    def get_llm_client(self, *, prompt_language: str, model_hint: str | None = None):
        # 返回任意符合 LLMApiClient 协议的对象
        return my_custom_llm

agent = CodeAgent(llm_client_provider=MyProvider())
```

### 0.5 何时**不**用 CodeAgent

| 场景 | 替代 |
|---|---|
| 已经知道精确 `old_string`，不想让 LLM 决策 | `InlineCodeEditor.edit` 或 `CodeEditFacade.apply_precise_edit` |
| 没有 LLM，只想做 deterministic 文本替换 | 直接用 `str.replace` 或 `InlineCodeEditor.edit` |
| 需要细粒度控制 prompt / tools / middleware | 走 advanced API：`build_default_query_engine` + `QuerySession` |
| 需要 edit-run-fix 闭环但不要 cc 的工具/权限层 | `AutonomousCodeAgent` |

### 0.6 返回结果（`AgentRunResult`）

| 字段 | 说明 |
|---|---|
| `final_text` | LLM 最终回答文本 |
| `session_id` / `turn_id` | 会话/轮次标识，可用于审计或续作 |
| `tool_call_count` | 本次调用了多少次工具（含 file_edit/file_write） |
| `failed` / `error_code` / `error_message` | 失败原因（成功时三者都是 falsy） |
| `events` | 完整 `SessionEvent` 列表 |
| `messages` | `SessionMessage` 序列 |
| `session_snapshot` | 序列化后的 session 状态字典 |

---

## 1. InlineCodeEditor

> 位置：`core/utils/inline_code_editor.py`

### 适用场景
- **上层 agent 已经把代码读进内存**（LLM 生成、DB 行、消息体），不一定有文件路径；
- 一轮内做几十次小修改，不希望每次都打到磁盘；
- 需要 rollback，但只想要会话内的内存栈，不想污染磁盘；
- 需要自定义"运行时校验器"（lint、单测函数），不想依赖 shell 命令；
- 想要 **Claude-Code 协议的可靠性**（exact string match + hash 保护 + 多级验证）但不想引入 `core/cc` 的权限/审计依赖。

### 不适合
- 真的要改磁盘文件并需要权限/审计 → 用 `CodeEditFacade`。
- 大批量 LLM 自由编辑 → 用 `RobustLLMEditor` / `SmartLLMEditorV2`（或经由 `edit_with_llm` 调用）。

### 用法

#### 基础精确替换
```python
from core.utils.inline_code_editor import InlineCodeEditor

editor = InlineCodeEditor()

src = "def foo():\n    return 1\n"
result = editor.edit(
    src,
    old_string="return 1",
    new_string="return 2",
)

if result.success:
    new_code = result.code        # 'def foo():\n    return 2\n'
    diff = result.diff
    ckpt = result.checkpoint_id   # 'ckpt_000001'，可用于回滚
else:
    print(result.error_code)      # ED1002 / ED1003 / ED1004 / ED1005...
```

#### Hash 保护（防止"读出代码 → LLM 改 → 写回"被旁路）
```python
result = editor.edit(
    code,
    old_string=old,
    new_string=new,
    expected_hash=earlier_known_hash,  # 不一致直接拒绝
)
if result.error_code == "ED1004":
    # 代码已被其他人修改，重读
    ...
```

#### `replace_all` 多处替换
```python
editor.edit(code, "DEPRECATED_FN", "new_fn", replace_all=True)
```

#### 整文件覆盖
```python
editor.write(current_code, generated_full_code)
```

#### 自定义 runtime validator（替代 shell 命令）
```python
from core.cc.editing.requests import EditValidationResult

def lint(code: str) -> EditValidationResult:
    if "import *" in code:
        return EditValidationResult(
            ok=False, stage="runtime",
            messages=["wildcard import disallowed"],
            error_code="ED1006",
        )
    return EditValidationResult(ok=True, stage="runtime")

result = editor.edit(code, "...", "...", runtime_validator=lint)
```

#### 事务批改（一处失败全员回滚）
```python
from core.cc.editing.requests import FileEditRequest

edits = [
    FileEditRequest(file_path="m.py", old_string="A", new_string="A1"),
    FileEditRequest(file_path="m.py", old_string="B", new_string="B1"),
    FileEditRequest(file_path="m.py", old_string="C", new_string="C1"),
]
batch = editor.apply_many(code, edits)
if not batch.success:
    # 任一编辑失败时 batch.code 是原始代码，rollback_performed=True
    print("rolled back to:", batch.checkpoint_id)
```

#### LLM 兜底（无法手写 old_string 时）
```python
editor = InlineCodeEditor(llm_client=my_llm, default_llm_backend="robust")
result = editor.edit_with_llm(code, instruction="给 calculate 加日志")
# backend 可选 "robust" / "smart_v2" / "lnfree" / "line"
```

#### Checkpoint 手动控制
```python
ckpt = editor.checkpoint(code)
# ... 一系列编辑 ...
recovered = editor.rollback(ckpt)   # 取回 ckpt 时的代码
```

#### 可选：写回磁盘
```python
saved_hash = editor.save_file("path/to/file.py", code, expected_hash=prior_hash)
loaded = editor.load_file("path/to/file.py")
```

### 错误码
| code | 含义 |
|---|---|
| `ED1002` | `old_string` 在代码中没找到 / Edit is empty |
| `ED1003` | `old_string` 多次匹配但未指定 `replace_all` |
| `ED1004` | hash 不匹配（代码已被其他人修改） |
| `ED1005` | 修改后的代码语法错误 |
| `ED1006` | runtime validator 失败 / LLM 后端失败 |
| `ED1007` | 找不到 checkpoint |

---

## 2. CodeEditFacade

> 位置：`core/cc/editing/facade.py`

### 适用场景
- 你在 `core/cc` 的 agent runtime 中，要**真的改磁盘上的文件**；
- 需要 file-hash 并发保护、权限分类（`core/cc/safety`）、持久化 rollback、可选 shell 运行时验证；
- 编辑工具会被 LLM agent 直接调用（封装为 `file_edit` / `file_write` MCP 工具）。

### 不适合
- 代码只在内存里 → 用 `InlineCodeEditor`。
- 需要让 LLM 自由理解并改代码 → 用 `RobustLLMEditor` / `SmartLLMEditorV2`，或 `apply_llm_edit(backend=...)`。

### 用法

#### 精确编辑（推荐路径）
```python
from core.cc.editing import CodeEditFacade, FileEditRequest

facade = CodeEditFacade(default_llm_backend="robust")

req = FileEditRequest(
    file_path="src/foo.py",
    old_string="return 1",
    new_string="return 2",
    replace_all=False,
    expected_hash=None,                    # 或上一次拿到的 hash
    validate_python_syntax=True,
    runtime_command="python -m pytest tests/foo_test.py",  # 可选
)
result = facade.apply_precise_edit(req)
if not result.success:
    # 已自动 rollback 到 checkpoint
    print(result.error_code, [v.messages for v in result.validation_results])
```

#### 预览（不实际写盘）
```python
preview = facade.preview_edit(req)
print(preview.diff)
```

#### LLM 兜底（精确编辑无法表达时）
```python
result = facade.apply_llm_edit(
    instruction="把 calculate 函数改成支持 batch 输入",
    current_code=code_text,
    llm_client=llm,
    prompt_language="zh",
    backend="robust",         # 可选 "line" / "robust" / "smart_v2" / "lnfree"
)
```

#### 显式 rollback
```python
facade.rollback(checkpoint_id="ckpt_xxxxxxxxxxxx")
```

#### 配置 GC（推荐用于长期运行的 agent）
```python
from core.cc.editing.rollback import RollbackManager

mgr = RollbackManager(
    "/var/lib/myagent/checkpoints",
    max_checkpoints=200,
    ttl_seconds=7 * 24 * 3600,
)
facade = CodeEditFacade(rollback_manager=mgr)
```

---

## 3. RobustLLMEditor

> 位置：`core/utils/robust_llm_editor.py`

### 适用场景
- 单文件 / 单段代码做 LLM 驱动的修改（"加日志、改算法、修 bug"）；
- 需要在 LLM 输出不稳定时仍有较高成功率（**3 策略级联回退**：SEARCH/REPLACE → 函数整替 → 全量重写）；
- 是 `AutonomousCodeAgent` 的默认底盘。

### 不适合
- 跨文件 / 仓库级编辑 → 用 `SmartLLMEditorV2`。
- 你已经拿到精确 `old_string` → 直接用 `InlineCodeEditor` / `CodeEditFacade`。

### 用法
```python
from core.utils.robust_llm_editor import RobustLLMEditor

editor = RobustLLMEditor(llm_client=llm)

result = editor.modify(
    code=src,
    instruction="为 calculate_alpha 加上 logging.debug 输出关键中间变量",
    context="这是因子计算流水线",        # 可选
    file_path="alpha.py",                # 可选，仅日志
    max_retries=3,
)

if result.success:
    print("strategy used:", result.strategy_used)  # search_replace / function_replace / full_rewrite
    print(result.diff)
    new_code = result.new_code
else:
    print(result.errors, result.failed_edits)
```

返回字段（`EditResult`）：`success / new_code / diff / applied_edits / failed_edits / strategy_used / errors / warnings`

---

## 4. SmartLLMEditorV2

> 位置：`core/utils/smart_llm_editor_v2.py`

### 适用场景
- **多文件**项目级编辑（"在所有 strategy 文件里把阈值 0.5 改成参数"）；
- 大文件需要 tree-sitter / BM25 / repo map 加持；
- 想要 **CorrectionLoop 自动多轮纠错**。

### 不适合
- 简单单文件编辑 → 用 `RobustLLMEditor`，更轻。
- 没装 `tree_sitter_python` / `rank_bm25` 时仍可用，但日志会打 fallback 警告。

### 用法

#### 单文件
```python
from core.utils.smart_llm_editor_v2 import SmartLLMEditorV2

editor = SmartLLMEditorV2(llm_client=llm, max_context_tokens=8000)
print("backends:", editor.backends)   # {'indexer': 'tree-sitter', 'bm25': 'ok', ...}

result = editor.edit(
    code=src,
    instruction="给所有 process_* 函数加类型注解",
    file_path="data.py",
    context="",
    max_retries=3,
)
print(result.success, result.rounds_used, result.applied_edits)
```

#### 多文件（事务模式）
```python
def read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()

results = editor.edit_project(
    instruction="把所有 LOG_LEVEL = 'DEBUG' 改成从环境变量读取",
    file_paths=["a.py", "b.py", "c.py"],
    read_file_fn=read,
    transactional=True,    # 任一文件失败时所有结果降级为 success=False
)
for r in results:
    if r.success:
        # 调用方负责持久化 r.new_code
        ...
```

#### 查找 / 分析（不改代码）
```python
syms = editor.find(code, query="哪个函数处理 timeout？")
analysis = editor.analyze(code, query="这段代码的瓶颈在哪？")
```

---

## 5. AutonomousCodeAgent

> 位置：`core/utils/autonomous_code_agent.py`
> 完整文档：[`_legacy/llm_ast_editor.md`](_legacy/llm_ast_editor.md) ❌（已废弃）；正确文档在 agent 自己的 docstring 与 `code_editor.md`。

### 适用场景
- 需要 **edit→run→observe→re-edit 闭环**直到代码通过验证（如 review、lint、单测）；
- 错误不止一处 / 不能一次性 LLM 修完；
- 需要错误根因分析、主动插桩、跨轮记忆、智能回滚。

### 不适合
- 一次就能改完的简单编辑 → 用 `RobustLLMEditor.modify()` 或 agent 的 `edit()` 单次入口。
- 修改无副作用的纯文本（无法运行验证）→ 闭环没意义。

### 用法

```python
from core.utils.autonomous_code_agent import AutonomousCodeAgent, RunResult
import traceback as _tb

agent = AutonomousCodeAgent(
    llm_client=llm,
    editor_type="robust",        # 或 "v2" 用 SmartLLMEditorV2
    max_diff_lines=200,
)

def runner_fn(code: str) -> RunResult:
    """调用方决定怎么验证：可以是 lint、单测、re-review、运行脚本……"""
    try:
        compile(code, "<gen>", "exec")
        # 你的领域校验
        result = my_review(code)
        return RunResult(
            success=len(result.errors) == 0,
            errors=result.errors,
            output=result.stdout,
        )
    except Exception as e:
        return RunResult(
            success=False,
            errors=[f"{type(e).__name__}: {e}"],
            traceback=_tb.format_exc(),
        )

fix = agent.fix_until_pass(
    code=src,
    runner_fn=runner_fn,
    max_rounds=10,
    context="高频因子调度框架",
    protected_names=["AlphaConfig", "HighFreqFramework"],
    enable_probes=True,
)

if fix.success:
    final_code = fix.final_code
    print(f"fixed in {fix.rounds_used} rounds")
else:
    print(f"残余错误 {fix.total_errors_final}/{fix.total_errors_initial}")
    print("\n".join(fix.debug_log))
```

#### 不要闭环、只做单次编辑
```python
single = agent.edit(code=src, instruction="...")
```

#### 关键经验
> **runner_fn 决定一切**。`RunResult.errors` 中要尽量带 traceback；没有 traceback 时 agent 会先插桩诊断（多花 1-2 轮）。

---

## 6. LineNumberFreeLLMBlockEditor

> 位置：`core/utils/llm_block_editor_lnfree.py`

### 适用场景
- 已经从 LLM / 其他来源拿到**结构化的块编辑指令字符串**（`REPLACE old → new` / `INSERT AFTER locator`）；
- 需要无行号的内容定位（exact / ast / fingerprint / normalized / similarity / fuzzy 七级匹配）；
- 作为 `FallbackLLMEditor` / `InlineCodeEditor.edit_with_llm(backend="lnfree")` 的底层引擎使用。

### 用法

#### 配合 LLM 一步到位
```python
from core.utils.llm_block_editor_lnfree import (
    LineNumberFreeLLMBlockEditor,
    EditorConfig,
)

editor = LineNumberFreeLLMBlockEditor(
    llm_client=llm,
    config=EditorConfig(
        similarity_threshold=0.85,
        enable_ast_validation=True,
        max_modification_ratio=0.5,    # 拒绝修改超过 50% 行数的指令
        allow_partial_success=True,
    ),
)
result = editor.edit_with_llm(src, instruction="...", context="", file_path="x.py")
```

#### 直接 apply 已有指令字符串
```python
instructions = """
REPLACE
def old_fn():
    return 1
WITH
def old_fn():
    return 2

INSERT AFTER
import os
WITH
import logging
"""
result = editor.apply_instruction_string(src, instructions)
```

---

## 7. FallbackLLMEditor

> 位置：`core/utils/editor_fallback.py`

### 适用场景
- 想在两个块编辑器之间做"主-备"双引擎，提高鲁棒性；
- 主用 lnfree（默认），失败时再调 deprecated 的 `LLMBlockEditor`（或反过来）。

### 用法
```python
from core.utils.editor_fallback import FallbackLLMEditor

editor = FallbackLLMEditor(llm_client=llm, prefer="lnfree")  # 或 "block"

result = editor.edit_with_llm(
    original_code=src,
    instruction="...",
    context="",
    file_path="x.py",
)
# 等价：先试 lnfree；不成功再试 block；最终返回更优的那个
```

> **注意**：`block` 后端（`LLMBlockEditor`）已被 deprecate。一般保持 `prefer="lnfree"` 即可。

---

## 8. LLMCodeEditor

> 位置：`core/utils/llm_code_editor.py`

### 适用场景
- 行级精确编辑（`LineNumberHandler` 动态宽度行号）；
- `core/cc/editing/CodeEditFacade.apply_llm_edit(backend="line")` 的底层后端；
- 想要"LLM 直接输出带行号编辑指令"的传统模式。

### 不适合
- 大文件 / 长上下文（行号容易飘）→ 用 `RobustLLMEditor` / `SmartLLMEditorV2`。
- 多文件 → 用 `SmartLLMEditorV2`。

### 用法
```python
from core.utils.llm_code_editor import LLMCodeEditor

editor = LLMCodeEditor(llm_client=llm)
result = editor.edit_with_llm(
    original_code=src,
    instruction="为 main 函数加错误处理",
    context="",
)
```

---

## 9. ⚠️ 已 Deprecated 的编辑器

下面这些虽然还在仓库里，但**新代码不要用**。已加 `DeprecationWarning`，详见 [code_editor.md](code_editor.md)。

| 模块 | 替代方案 |
|---|---|
| `CodeEditor` | 单次编辑 → `InlineCodeEditor`；闭环修复 → `AutonomousCodeAgent`；多文件 → `SmartLLMEditorV2` |
| `LLMBlockEditor`（带行号） | `LineNumberFreeLLMBlockEditor` 或 `SmartLLMEditorV2` |

下面这些已**移到 `core/utils/_legacy/`**，仅作存档：

- `_legacy/llm_edit_pipeline.py` — fragment 流水线，无业务调用方
- `_legacy/llm_ast_editor.py` — 实验性 AST 语义编辑器，最终路径仍 fallback 到 lnfree
- `_legacy/verify_block_editor_integration.py` — 一次性集成验证脚本

---

## 10. 组合使用：典型工作流

### 工作流 0：外部项目接入 cc（最常用）
```python
from core.cc import run_code_agent, CCConfig

result = run_code_agent(
    instruction="为 src/auth.py 的 verify_token 加上速率限制",
    cwd="/path/to/project",
    config=CCConfig(prompt_language="zh", permission_mode="default"),
)
if not result.failed:
    print(result.final_text)
```
> 内部自动选编辑器、调度工具、做权限校验、写盘 + rollback，调用方无需关心细节。

### 工作流 A：上层 agent 在内存里多步精修
```python
editor = InlineCodeEditor(llm_client=llm)
ckpt = editor.checkpoint(code)

# 一连串精确小修改
r1 = editor.edit(code, "old_a", "new_a")
r2 = editor.edit(r1.code, "old_b", "new_b")
r3 = editor.edit(r2.code, "old_c", "new_c")

# 结果不满意 → 回滚到批前
if not satisfied(r3.code):
    code = editor.rollback(ckpt)
else:
    code = r3.code
```

### 工作流 B：精确编辑失败时优雅降级到 LLM
```python
editor = InlineCodeEditor(llm_client=llm)

r = editor.edit(code, expected_old, new)
if not r.success and r.error_code in ("ED1002", "ED1003"):
    # old_string 没找到或多个匹配 → 让 LLM 直接重写这部分
    r = editor.edit_with_llm(code, instruction=high_level_intent, backend="robust")
```

### 工作流 C：闭环修复（agent + runner）
```python
agent = AutonomousCodeAgent(llm_client=llm, editor_type="robust")
fix = agent.fix_until_pass(code, runner_fn=my_review, max_rounds=10)
```

### 工作流 D：精确编辑写盘 + agent runtime
```python
# 在 core/cc agent 里
facade = CodeEditFacade()
req = FileEditRequest(file_path="a.py", old_string=old, new_string=new)
result = facade.apply_precise_edit(req)
# 已 atomic 写盘、hash 校验、AST 校验、自动 rollback
```

### 工作流 E：仓库级重构
```python
editor = SmartLLMEditorV2(llm_client=llm)
results = editor.edit_project(
    instruction="把 print(...) 全部换成 logger.info(...)",
    file_paths=glob_all_py(),
    read_file_fn=read,
    transactional=True,
)
if all(r.success for r in results):
    for r, fp in zip(results, file_paths):
        save(fp, r.new_code)
```

---

## 11. 错误码速查（统一来自 `core/cc/editing`）

| code | 含义 |
|---|---|
| `ED1001` | 文件不存在且 `create_if_missing=False` |
| `ED1002` | `old_string` 没匹配到 / Edit is empty |
| `ED1003` | `old_string` 多次匹配但未指定 `replace_all` |
| `ED1004` | hash 不匹配（并发修改保护） |
| `ED1005` | 修改后语法错误（`ast.parse` 失败） |
| `ED1006` | runtime command / runtime_validator / LLM 后端失败 |
| `ED1007` | 找不到 checkpoint |
| `ED1008` | 文件超过 `max_file_size_bytes` |
| `ED2001` | 写盘失败（OSError） |

---

## 12. 选型口诀

> **外部接入先 CodeAgent，内部实现再下钻；
> 能精确就精确，不能精确再 LLM；
> 内存里就 Inline，磁盘上就 Facade；
> 单文件用 Robust，多文件用 SmartV2；
> 要闭环就 Autonomous。**
