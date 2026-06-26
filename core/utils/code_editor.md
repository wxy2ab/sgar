# core/utils 下 LLM 代码编辑器全景分析

> 撰写时间：2026-05-08（**2026-05-08 完成首轮重构**）
> 范围：`core/utils/` 中所有以 *editor* / *agent* 命名的代码编辑相关模块，
> 以及对照参考的 `core/cc/editing/` 完整编辑器栈。

> ## 本轮重构已落地的改动
>
> 1. **3 个 dead 编辑器移入 `core/utils/_legacy/`**：
>    `llm_edit_pipeline.py`、`llm_ast_editor.py`、`verify_block_editor_integration.py`（外加 `llm_ast_editor.md`）。
>    无业务调用方需要修改。
> 2. **新增** [`InlineCodeEditor`](inline_code_editor.py)：复用 `core/cc/editing.EditValidator`，
>    支持精确 edit / 整文件 write / LLM 兜底（robust/smart_v2/lnfree/line）/ 内存 checkpoint / 事务批改 (`apply_many`) /
>    自定义 `runtime_validator`。
> 3. **`CodeEditor` / `LLMBlockEditor`** 加 `DeprecationWarning`（仍保留可用，避免破坏现有 8 处 planner 调用）。
> 4. **`LineNumberFreeLLMBlockEditor`** 优化：唯一精确匹配早停、SHA-256 稳定 hash 缓存、删除 4 个空壳 healer。
> 5. **`FallbackLLMEditor`** 优化：`apply_instruction_string` 真正实现双向 fallback。
> 6. **`CodeEditFacade`** 优化：`apply_llm_edit(backend=...)` 可选 line/robust/smart_v2/lnfree；构造参数 `default_llm_backend`。
> 7. **`RollbackManager`** 优化：增加 `max_checkpoints` / `ttl_seconds` GC，自动清理 `.bak` 文件。
> 8. **`SmartLLMEditorV2`** 优化：构造时 `self.backends` 暴露后端选择并打日志；`edit_project(transactional=True)` 一处失败全员降级。
> 9. **`core/utils/__init__.py`** 更新：移除 dead 编辑器的 lazy export，新增
>    `SmartLLMEditorV2` / `AutonomousCodeAgent` / `InlineCodeEditor` / `InlineEditResult`。
>
> Tier-3 的全量迁移（`CodeEditor` / `LLMBlockEditor` 8+ 处调用方改为 `InlineCodeEditor`）保留为下一阶段任务，本轮不动。

---

## 0. TL;DR

| 名称 | 推荐 | 一句话定位 |
|---|---|---|
| `core/cc/editing/CodeEditFacade` | ⭐ **保留 / 主推** | Claude-Code 风格精确编辑器，hash 校验 + 多级验证 + 持久化 checkpoint |
| `RobustLLMEditor` | ⭐ **保留** | SEARCH/REPLACE 三策略级联，`AutonomousCodeAgent` 默认底盘 |
| `SmartLLMEditorV2` | ⭐ **保留** | 4 层流水线 + tree-sitter，唯一支持多文件 + 仓库级索引 |
| `AutonomousCodeAgent` | ⭐ **保留** | edit→run→observe→re-edit 闭环，生产级，eo_updater 在用 |
| `LLMCodeEditor` | 🔶 保留（次级） | 行级编辑，被 `core/cc.facade.apply_llm_edit` 当作兜底 |
| `LineNumberFreeLLMBlockEditor` | 🔶 保留（次级） | 无行号块编辑，定位策略最丰富，作为兜底 |
| `FallbackLLMEditor` | 🔶 保留（适配器） | 41 行薄壳，串接 lnfree↔block，无成本 |
| `CodeEditor` | ⚠️ **逐步淘汰** | 1441 行老式集成器，被新栈覆盖；旧 planner 仍在用，需迁移 |
| `LLMBlockEditor` | ⚠️ **逐步淘汰** | 3193 行带行号的块编辑，复杂度爆炸，被 lnfree + SmartV2 取代 |
| ~~`llm_edit_pipeline.py`~~ | ✅ **已移入 `_legacy/`** | fragment 流水线，全仓库无生产调用 |
| ~~`llm_ast_editor.py`~~ | ✅ **已移入 `_legacy/`** | 实验性 AST 语义编辑器，最终路径仍 fallback 到 lnfree |
| ~~`verify_block_editor_integration.py`~~ | ✅ **已移入 `_legacy/`** | 一次性集成验证脚本 |
| `inline_code_editor.py` (新增) | ⭐ **保留** | 上层 agent 用的内存版精确编辑器，复用 cc.editing 的 validator/protocol |

---

## 1. 模块清单与归类

| 文件 | 行数 | 入口 | 协议 / 策略 |
|---|---:|---|---|
| `code_editor.py` | 1441 | `CodeEditor` | 顶层集成器：index + summarizer + classifier + 多轮取全量代码 + 块编辑 |
| `llm_block_editor.py` | 3193 | `LLMBlockEditor` | `>> REPLACE start:end ... << END`（**带行号**） |
| `llm_block_editor_lnfree.py` | 1094 | `LineNumberFreeLLMBlockEditor` | 块编辑 + 7 级内容定位（exact / ast / fingerprint / normalized / similarity / fuzzy / similarity_relaxed） |
| `llm_code_editor.py` | 1447 | `LLMCodeEditor` | 行级编辑（`LineNumberHandler` 动态宽度行号） |
| `smart_llm_editor_v2.py` | 1532 | `SmartLLMEditorV2` | 4 层流水线：TreeSitterIndexer / RepoMapper / EditPlanner / SearchReplaceEngine + 5 级匹配 + CorrectionLoop |
| `robust_llm_editor.py` | 1009 | `RobustLLMEditor` | 3 策略级联：SEARCH/REPLACE → 函数/类整替 → 全量重写 |
| `llm_ast_editor.py` | 481 | `run_semantic_ast_edit` | 语义 AST 操作（rename/insert/replace_block/modify_imports），失败 fallback 到 lnfree |
| `autonomous_code_agent.py` | 1126 | `AutonomousCodeAgent` | edit-run-observe 闭环：ErrorAnalyzer + ProbeManager + DebugScratchpad + RollbackManager + RobustLLMEditor/SmartV2 |
| `editor_fallback.py` | 41 | `FallbackLLMEditor` | lnfree ↔ block 双向 fallback 适配器 |
| `llm_edit_pipeline.py` | 135 | `run_llm_edit_pipeline` | 旧 fragment 抽取-提示-替换流水线 |
| `verify_block_editor_integration.py` | 167 | `verify_integration` | 一次性集成验证脚本 |

参考：`core/cc/editing/` 包（facade / validator / rollback / file_state / requests）— Claude-Code 风格的精确编辑器栈。

---

## 2. 实际使用情况（生产端）

| 调用方 | 所用编辑器 | 备注 |
|---|---|---|
| `core/cc/tools/file_edit.py`、`file_write.py` | `CodeEditFacade` | 新一代 agent 工具栈（MCP 风格） |
| `core/cc/editing/facade.apply_llm_edit` | `LLMCodeEditor` | LLM 模式的兜底 |
| `task/agents/eo_updater.py` | `AutonomousCodeAgent` → `RobustLLMEditor` | 已生产化的闭环修复 |
| `core/deepstack/skills/code_edit.py` | smart_v2 / robust / block / line 任选 | 默认 `smart_v2` |
| `core/task/factor_factory/utils/{factor,signal}_code_editor.py` | `CodeEditor` | 旧版 |
| `core/task/opt_strategy/*`、`opt_factor/*`、`opt_time_factor/*` | `CodeEditor` | 旧版 planner 直接用 |
| `core/task/opt_strategy_factor_agent/sub_agents/base_factor_agent.py` | `CodeEditor` | 旧版 |
| `llm_edit_pipeline.run_llm_edit_pipeline` | — | **全仓库无调用** |
| `llm_ast_editor.run_semantic_ast_edit` | — | **仅在 `__init__.py` 注册，无业务调用** |

---

## 3. 编辑器优先级

### 3.1 Tier 1（保留，主力）

#### A. `core/cc/editing/CodeEditFacade` — Claude-Code 风格精确编辑器
- 协议：`{file_path, old_string, new_string, replace_all, expected_hash, runtime_command}` —— 与 Claude Code Edit 工具同构。
- 流程：file-hash 并发保护 → 4 段验证（text → structure(AST) → semantic → runtime）→ 临时文件 + `os.replace` 原子替换 → 失败自动 rollback。
- Rollback：`RollbackManager` 持久化到 `.cc/runtime/checkpoints/`。
- 优点：协议最简单、可靠性最高、与权限/审计体系打通。
- 缺点：要求调用方提供精确 `old_string`，对 LLM 输出质量有要求；目前与 `core/cc` 的 agent runtime / 权限子系统强耦合。

#### B. `RobustLLMEditor`
- 协议：SEARCH/REPLACE 块。
- 三策略级联：SR 块 → 函数/类整体替换 → 全量重写。
- 5 级匹配 + 缩进探测修复。
- 是 `AutonomousCodeAgent` 的默认底盘，`eo_updater` 已生产化。

#### C. `SmartLLMEditorV2`
- 4 层流水线（索引→规划→执行→纠错）。
- 唯一同时具备：tree-sitter 索引、BM25 仓库检索、RepoMap、多文件项目编辑、CorrectionLoop。
- 适合大文件 / 跨文件重构；deepstack 默认走它。

#### D. `AutonomousCodeAgent`
- 不是编辑器，是**编辑器的协调层**：edit→run→observe→re-edit 循环 + 错误根因分析 + 主动插桩 + 跨轮记忆 + 智能回滚。
- 唯一在 `task/agents/eo_updater.py` 生产闭环中使用，必须保留。

### 3.2 Tier 2（保留，作为支撑/兜底）

| 模块 | 作用 |
|---|---|
| `LLMCodeEditor` | `core/cc/editing/facade.apply_llm_edit` 兜底；`deepstack` 行级模式 |
| `LineNumberFreeLLMBlockEditor` | 7 级内容定位最丰富，被 `editor_fallback` / `llm_ast_editor` 当兜底 |
| `FallbackLLMEditor` | 41 行薄壳，把 lnfree↔block 串起来，零维护成本 |

### 3.3 Tier 3（保留但应停止扩展，逐步迁移调用方）

#### `CodeEditor`（[code_editor.py](core/utils/code_editor.py)）
- 1441 行，承担"索引 + 摘要 + 分类器 + 多轮取全量代码 + 块编辑"。
- 依赖 `complete_code_fetcher` 多轮对话获取完整代码、`code_classifier` 分类 LLM 输出。
- 仍被 `factor_factory`、`opt_strategy*`、`opt_factor*`、`opt_time_factor` 等 6 处直接 import，**短期不能删**。
- 中期目标：把这些调用迁移到 `AutonomousCodeAgent.edit()`（单次）或 `SmartLLMEditorV2.edit()`，然后退役 `CodeEditor`。

#### `LLMBlockEditor`（[llm_block_editor.py](core/utils/llm_block_editor.py)）
- 3193 行，是 `core/utils` 中最复杂的模块。
- 设计基于"行号 + 块编辑指令"，事实证明 LLM 对精确行号支持差，目前仍在内部反复用 `_force_correct_line_numbers`、`_fix_block_instructions_with_llm`、`_validate_instructions_detailed` 打补丁。
- 已被 `LineNumberFreeLLMBlockEditor`（无行号）+ `SmartLLMEditorV2`（SR 块）两路覆盖。
- 中期目标：当 `CodeEditor` 退役后随之退役；同时把 `editor_fallback` 默认改为 lnfree。

### 3.4 可立即废弃

| 模块 | 理由 |
|---|---|
| `llm_edit_pipeline.py` | 全仓库无生产调用，仅在 `__init__.py` 注册 |
| `llm_ast_editor.py` | 主路径走"语义 plan + AST 变换"，但失败时 fallback 到 lnfree；没有任何业务进入主路径，价值已被吸收到 `CodeEditFacade` 的 AST 验证里 |
| `verify_block_editor_integration.py` | 一次性集成验证脚本，移到 `core/utils/tests/` 即可 |

---

## 4. 可保留的编辑器中需要进一步优化的点

### 4.1 `core/cc/editing/CodeEditFacade`
1. **runtime 验证粒度太粗**：`validate_runtime` 只接受单条 shell 命令（`runtime_command`）。建议引入"validator pipeline"接口，允许传入函数式 validator（如 lint / 单测函数）。
2. **`apply_llm_edit` 退化为只调 `LLMCodeEditor`**：把 `editor_type` 暴露成参数，按需切换到 `RobustLLMEditor` / `SmartLLMEditorV2`，统一作为"非精确路径"出口。
3. **多次小修改的 IO 浪费**：每次 `apply_precise_edit` 都重新 `read_text` 整个文件；同一文件在一轮内被改多次时应该用 `FileStateCache` 的内存版本。
4. **rollback checkpoint 没有 GC**：`.cc/runtime/checkpoints/checkpoints.json` 会无限增长，需要按 TTL / 数量上限做清理。

### 4.2 `RobustLLMEditor`
1. **prompt 中仍带行号**：虽然定位不依赖行号，但 prompt 仍把"左侧行号"塞给 LLM，对长文件浪费 token。可改成"按需才注入行号摘要"。
2. **`_Strategy3_FullRewrite` 缺少"完整性回退"**：当全量重写返回的代码丢失关键函数时，目前只能依赖 `validate_integrity` 失败而回退；应该在策略层补一次"按 protected_names 反向修复"。
3. **`_extract_keywords` 中文停用词写死在代码里**：抽到配置或 `core/utils/string_matcher` 复用。

### 4.3 `SmartLLMEditorV2`
1. **多文件 `edit_project` 没有事务**：每个文件独立提交，部分失败后不会回滚已成功的文件。
2. **tree-sitter / grep_ast / rank_bm25 都是软依赖**，缺失时静默降级到 ast/regex。需要把"当前后端"打到日志里，否则线上很难排查为什么搜索质量下降。
3. **CorrectionLoop 的 prompt 增强是 _augment_instruction 字符串拼接**，已经接近 prompt 上限；建议改成结构化 messages（system/user/tool），与 `core/cc/conversation` 的 middleware 对齐。

### 4.4 `AutonomousCodeAgent`
1. **runner_fn 必须传 traceback**：当 traceback 为空走"插桩诊断"路径，平均多花 1-2 轮。可在 `RunResult` 上加 `traceback_required: bool` 提示调用方。
2. **`RollbackManager` 与 `core/cc/editing/RollbackManager` 同名但不互通**：建议统一到 `core/cc` 那一份，让 agent 的回滚也可以被 audit。
3. **`_simplified_modification_flow` 等"简化路径"散落在 `code_editor.py`**，但 agent 又自己实现了一份相同语义的 fallback；建议提取到 `core/utils/edit_flows.py` 共享。

### 4.5 `LineNumberFreeLLMBlockEditor`
1. **7 级匹配每次都全跑一遍**（exact→ast→fingerprint→normalized→similarity→fuzzy→similarity_relaxed），大文件会慢。可以加"早停"：精确匹配命中且唯一，直接返回。
2. **`_heal_*` 系列方法大多是 `return self.code` 的占位**（healed import order / brace balance / function signature 都没真正实现）；要么补上要么删掉，避免误导。
3. **`ContentLocator._ast_cache` 用 `hash(source)` 当 key**：源码极大时 hash 冲突概率不为零，应当用 sha256。

### 4.6 `LLMCodeEditor`
1. 与 `core/cc/editing/CodeEditFacade.apply_llm_edit` 的关系不清晰。建议：(a) facade 显式声明"LLM 模式"接口契约；(b) `LLMCodeEditor` 只暴露符合该契约的入口，其他细节（动态行号宽度等）封装。

### 4.7 `FallbackLLMEditor`
- 已经够薄；唯一建议：默认 `prefer="lnfree"`（已是默认），并在 `apply_instruction_string` 里也实现真正的双向 fallback（目前只调 lnfree）。

---

## 5. 把 core/cc 完整编辑器做成 inline 版本（用于上层 agent）

### 5.1 为什么需要

`core/cc/editing/CodeEditFacade` 是目前最可靠的编辑器，但它面向"agent runtime / file system"场景：
- 每次调用都 **read 文件 → 校验 hash → write 文件**；
- 强依赖 `core/cc.config` / `core/cc.safety` / `core/cc.command_runner`；
- Rollback checkpoint 持久化到磁盘 `.cc/runtime/checkpoints/`。

而**上层 agent**（factor_factory、opt_strategy、deepstack）调用编辑器的真实场景常常是：
- 代码已经在内存里（刚刚由 LLM 生成 / 从 DB 读出），不一定有文件路径；
- 一轮内会做几十次小改，不希望每次都打到磁盘；
- 需要 rollback，但只想要会话内的内存栈，不想污染磁盘；
- 不需要 runtime command 校验（agent 自己有 runner）；
- 不需要 file permission 校验（agent 已经在自己的沙箱内）。

所以**完全可以**抽取一个 inline 版本，复用 facade 的"协议 + 校验 + rollback 语义"，去掉文件系统/权限/runtime 这部分负担。

### 5.2 设计建议（不写代码，仅设计草案）

新模块：`core/utils/inline_code_editor.py`，类名 `InlineCodeEditor`。

```
class InlineCodeEditor:
    def __init__(
        self,
        *,
        validator: EditValidator | None = None,   # 复用 core/cc.editing.EditValidator
        max_rollback_depth: int = 16,             # 内存栈深度
        llm_client: Any = None,                   # 可选：用于 LLM 兜底
    ): ...

    # 核心：精确编辑（与 CodeEditFacade.apply_precise_edit 同构）
    def edit(
        self,
        code: str,
        old_string: str,
        new_string: str,
        *,
        replace_all: bool = False,
        validate_python_syntax: bool = True,
    ) -> InlineEditResult: ...

    # 整文件覆盖（与 CodeEditFacade.apply_write 同构）
    def write(self, code: str, new_content: str) -> InlineEditResult: ...

    # LLM 兜底：调用 RobustLLMEditor 或 SmartLLMEditorV2
    def edit_with_llm(self, code: str, instruction: str, *, backend="robust") -> InlineEditResult: ...

    # 内存 checkpoint
    def checkpoint(self, code: str) -> str: ...     # 返回 ckpt_id
    def rollback(self, ckpt_id: str) -> str: ...    # 返回恢复后的 code

    # 可选：从文件加载 / 持久化
    def load_file(self, file_path: str) -> str: ...
    def save_file(self, file_path: str, code: str, *, expected_hash: str | None = None) -> None: ...
```

返回类型 `InlineEditResult` 对应 facade 的 `EditResult`，但去掉 `file_path` 必填项：

```
@dataclass
class InlineEditResult:
    success: bool
    code: str                    # 编辑后的代码（失败时是原 code）
    before_hash: str
    after_hash: str
    diff: str
    checkpoint_id: str           # 内存 ckpt id
    validation_results: list[EditValidationResult]
    error_code: str | None
```

### 5.3 实现要点

1. **协议复用**：直接 import `core.cc.editing.requests.{EditValidationResult, PatchPreview}` 和 `core.cc.editing.validator.EditValidator`，不要重新写一份。
2. **`_build_updated_content` 抽公共**：facade 里这个方法纯字符串操作，可下沉到 `core/cc/editing/text_ops.py`，inline 版本和 facade 共用。
3. **Rollback 改成内存栈**：`collections.OrderedDict[str, str]`，超过 `max_rollback_depth` 时 LRU 淘汰，与磁盘版的 `RollbackManager` 接口对齐。
4. **去掉 runtime/permission**：`InlineCodeEditor.edit()` 不接受 `runtime_command`、不调用 `safety/file_rules`，由调用方自己负责。
5. **LLM 兜底统一入口**：`edit_with_llm(backend=...)` 后端选 `"robust"` / `"smart_v2"` / `"lnfree"`，避免每个上层 agent 重复"先尝试精确，再调 LLM"的胶水逻辑。
6. **保留 hash 校验语义**：上层 agent 多步操作时仍可用 `expected_hash` 防止"读出代码 → LLM 改 → 写回"间被旁路修改。

### 5.4 迁移路径

短期：
- 实现 `InlineCodeEditor`；
- 在 `core/utils/__init__.py` 注册（lazy import）。

中期：
- `factor_factory/utils/{factor,signal}_code_editor.py`、`opt_strategy/sub_agents.py`、`opt_factor/opt_factor_planner.py` 等当前直接 `from core.utils.code_editor import CodeEditor` 的位置，按需要选择：
  - 只想做精确字符串替换：用 `InlineCodeEditor.edit`；
  - 想让 LLM 自由编辑：用 `InlineCodeEditor.edit_with_llm(backend="robust")`；
  - 想要闭环修复：用 `AutonomousCodeAgent`（它本身就基于 `RobustLLMEditor`，可以再包一层 inline 适配）。

长期：
- 把 `CodeEditor` / `LLMBlockEditor` / `llm_edit_pipeline` / `llm_ast_editor` / `verify_block_editor_integration` 移到 `core/utils/_legacy/`，并加 `DeprecationWarning`；
- 迁移完成后整组删除，预计可减少 ~6500 行代码。

---

## 6. 一张图

```
                        ┌────────────────────────────────────────┐
                        │  上层 agent (factor_factory / opt_*)    │
                        └──────────────┬─────────────────────────┘
                                       │
                  ┌────────────────────┼─────────────────────────┐
                  │                    │                         │
                  ▼                    ▼                         ▼
       InlineCodeEditor (新)    AutonomousCodeAgent    SmartLLMEditorV2
        ├─ edit()                ├─ fix_until_pass     └─ edit_project()
        ├─ write()               │     │                     (多文件)
        ├─ edit_with_llm()       │     ▼
        └─ checkpoint/rollback   │   RobustLLMEditor
                  │              │     │
                  │              │     ▼
                  │              │   SEARCH/REPLACE
                  │              │   + Function Replace
                  │              │   + Full Rewrite
                  │              │
                  └──────────────┴─── (复用) ──┐
                                               │
                          core/cc/editing/EditValidator
                                  ├─ validate_text
                                  ├─ validate_structure (AST)
                                  ├─ validate_semantics
                                  └─ validate_runtime

   旧栈（待退役）:
   CodeEditor → LLMBlockEditor (3193 行行号块)
              ↘ LineNumberFreeLLMBlockEditor (无行号块, 保留作 fallback)
              ↘ llm_edit_pipeline / llm_ast_editor (无业务调用)
```

---

## 7. 行动清单

### 已完成（2026-05-08）
- [x] 把 `llm_edit_pipeline.py` / `llm_ast_editor.py` / `verify_block_editor_integration.py` 移到 `core/utils/_legacy/`
- [x] 从 `core/utils/__init__.py` 的 `_EXPORTS` 中移除 dead 编辑器
- [x] 给 `LLMBlockEditor` / `CodeEditor` 加 `DeprecationWarning`
- [x] 实现 `InlineCodeEditor`（约 460 行，含 `apply_many` 事务批改和 LLM 兜底）
- [x] `LineNumberFreeLLMBlockEditor` 早停 + sha256 + 删空壳 healer
- [x] `FallbackLLMEditor` 双向 fallback
- [x] `CodeEditFacade.apply_llm_edit(backend=...)` 可配置后端
- [x] `RollbackManager` GC（max_checkpoints + ttl_seconds）
- [x] `SmartLLMEditorV2` 后端日志 + `edit_project(transactional=True)`

### 1-2 周（剩余优化）
- [ ] `RobustLLMEditor` 的 prompt 行号按需注入（§4.2-1）
- [ ] `RobustLLMEditor._extract_keywords` 中文停用词外置到配置（§4.2-3）
- [ ] `RobustLLMEditor._Strategy3_FullRewrite` 加"按 protected_names 反向修复"（§4.2-2）
- [ ] `SmartLLMEditorV2.CorrectionLoop` 改为结构化 messages（§4.3-3）
- [ ] `AutonomousCodeAgent` 的 `RollbackManager` 与 `core/cc.editing.RollbackManager` 统一（§4.4-2）

### 1-2 个月（callers 迁移）
- [ ] 把 `factor_factory/utils/{factor,signal}_code_editor.py`、`factor_factory/signal_agent.py`、
      `opt_strategy/sub_agents.py`、`opt_strategy/opt_strategy_planner.py`、`opt_factor/opt_factor_planner.py`、
      `opt_strategy_factor_agent/sub_agents/base_factor_agent.py`、`opt_time_factor/utils/planner_tools.py` 共 8 处
      从 `CodeEditor` 迁移到 `InlineCodeEditor` + `AutonomousCodeAgent`
- [ ] 迁移完成后删除 `code_editor.py` / `llm_block_editor.py`，预计再减少 ~4600 行
