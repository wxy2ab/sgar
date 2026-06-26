# AutonomousCodeAgent 技术文档

> `core/utils/autonomous_code_agent.py`

## 1. 概述

`AutonomousCodeAgent` 是一个自治代码调试智能体，构建在现有的 `RobustLLMEditor` / `SmartLLMEditorV2` 之上。它将原先散落在调用方（如 `eo_updater._fix_review_errors`）中的 **edit → run → observe → re-edit** 循环内置化，并增加了以下能力：

| 能力 | 组件 | 解决的问题 |
|---|---|---|
| 运行时反馈闭环 | `AutonomousCodeAgent.fix_until_pass()` | 编辑器不知道自己的修改有没有效 |
| 错误根因定位 | `ErrorAnalyzer` | 没有 Traceback 时 LLM 盲修 |
| 主动插桩诊断 | `ProbeManager` | 错误被内部 try/except 吞掉时无法获取运行时信息 |
| 跨轮次记忆 | `DebugScratchpad` | 每轮无状态导致重复走死路 |
| 智能回滚 | `RollbackManager` | 越改越差时没有恢复机制 |
| 最小修改约束 | prompt 强化 + diff 检查 | LLM 一轮改 1000+ 行引入新问题 |

### 架构层次

```
调用方 (eo_updater / 其他 agent)
    │
    │  传入 code + runner_fn
    ▼
AutonomousCodeAgent
    ├── ErrorAnalyzer      ← 分析错误，决定修复策略
    ├── ProbeManager       ← 无 Traceback 时插桩收集运行时信息
    ├── DebugScratchpad    ← 记录每轮假设/结论/已排除方向
    ├── RollbackManager    ← 追踪最优版本，错误增加时回滚
    │
    ├── RobustLLMEditor    ← 底层代码编辑器（默认）
    │   或 SmartLLMEditorV2
    │
    └── runner_fn          ← 调用方提供的运行验证函数
```

### 与现有组件的关系

- **不替代** `RobustLLMEditor` / `SmartLLMEditorV2`：它们仍然是底层的代码变换器，负责将指令翻译为 Search/Replace 编辑操作。
- **不替代** `smart_code_replacer.py`：那是一个独立的智能函数替换工具，用途不同。
- **替代** `eo_updater._fix_review_errors` 中原先的手工修复循环：约 200 行 for-loop + 错误分析 + 指令拼接逻辑被替换为 ~40 行的 `agent.fix_until_pass()` 调用。

---

## 2. 数据类

### `RunResult`

调用方的 `runner_fn` 必须返回此类型。

```python
@dataclass
class RunResult:
    success: bool                              # 代码是否通过验证
    output: str = ""                           # stdout 输出
    errors: List[str] = field(...)             # 错误列表
    traceback: str = ""                        # 完整 traceback 文本
    warnings: List[str] = field(...)           # 警告列表
```

| 字段 | 说明 |
|---|---|
| `success` | `True` 表示代码完全通过验证，Agent 将停止修复循环 |
| `errors` | 错误字符串列表。Agent 依赖此列表判断错误数量和内容变化 |
| `traceback` | 可选。如果 errors 中已包含 Traceback 信息，此字段可为空字符串 |
| `output` | 可选。用于探针诊断时收集 print 输出 |

### `FixResult`

`fix_until_pass()` 的返回值。

```python
@dataclass
class FixResult:
    success: bool                              # 是否修复成功
    final_code: str                            # 最终代码（成功则为通过验证的版本，失败则为最优版本）
    original_code: str = ""                    # 原始代码
    rounds_used: int = 0                       # 使用的修复轮数
    total_errors_initial: int = 0              # 初始错误数
    total_errors_final: int = 0                # 最终错误数
    debug_log: List[str] = field(...)          # 每轮的简要日志
    rollback_count: int = 0                    # 回滚次数
```

### `ErrorAnalysis`

`ErrorAnalyzer.analyze()` 的返回值，内部使用。

```python
@dataclass
class ErrorAnalysis:
    category: str = "unknown"                  # 错误类别（见下方分类表）
    root_cause_hypothesis: str = ""            # LLM 或模式匹配生成的根因假设
    suspect_locations: List[str] = field(...)  # 可疑代码位置（如 "L42 in calc: x = np.sum(values)"）
    fix_strategy: str = "targeted_fix"         # "targeted_fix" 或 "probe_first"
    probe_suggestions: List[Dict] = field(...) # 探针建议（line + reason）
    fix_hints: List[str] = field(...)          # 模式匹配产生的修复提示
    has_traceback: bool = False                # 是否有可用的 Traceback
    traceback_lines: List[Dict] = field(...)   # 结构化的 Traceback 条目
```

### `DebugRound`

单轮调试记录，存储在 `DebugScratchpad` 中。

```python
@dataclass
class DebugRound:
    round_idx: int                             # 轮次编号
    hypothesis: str = ""                       # 本轮的根因假设
    changes_made: str = ""                     # 修改摘要（如 "15 行变更, strategy=search_replace"）
    diff_line_count: int = 0                   # diff 变更行数
    run_result: Optional[RunResult] = None     # 本轮的运行结果
    error_count: int = 0                       # 错误数
    conclusion: str = ""                       # 本轮结论（如 "有改善: 3 -> 1 个错误"）
```

---

## 3. 核心组件详解

### 3.1 AutonomousCodeAgent

主入口类。通过 `fix_until_pass()` 方法驱动整个修复循环。

#### 构造参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `llm_client` | `LLMApiClient` | `None` | LLM 客户端，需要 `one_chat()` 方法 |
| `editor_type` | `str` | `"robust"` | 底层编辑器：`"robust"` = RobustLLMEditor，`"v2"` = SmartLLMEditorV2 |
| `max_diff_lines` | `int` | `200` | 单轮最大允许 diff 行数，超过则记录警告 |

#### `fix_until_pass()` 参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `code` | `str` | (必填) | 初始代码文本 |
| `runner_fn` | `Callable[[str], RunResult]` | (必填) | 运行验证函数 |
| `max_rounds` | `int` | `10` | 最大修复轮数 |
| `context` | `str` | `""` | 额外上下文（如文件用途说明） |
| `protected_names` | `List[str]` | `None` | 不可删除的函数/类名 |
| `enable_probes` | `bool` | `True` | 是否启用主动插桩 |
| `max_probe_rounds` | `int` | `2` | 最多执行几轮插桩诊断 |

#### `edit()` 方法

单次编辑入口（不含运行循环），兼容 `RobustLLMEditor.modify()` 的调用方式。当不需要 edit-run 闭环时可以直接调用。

#### 主循环流程

```
                    ┌──────────────────┐
                    │  runner_fn(code)  │  ← 首次运行
                    └────────┬─────────┘
                             │
                    success?─┤
                     yes ──► │ 返回 FixResult(success=True)
                             │ no
                    ┌────────▼─────────┐
         ┌────────►│  ErrorAnalyzer    │  ← 分析错误根因
         │         │  .analyze()       │
         │         └────────┬─────────┘
         │                  │
         │         strategy == "probe_first"?
         │          yes ──► │
         │         ┌────────▼─────────┐
         │         │  ProbeManager    │  ← 插桩 → 运行 → 收集诊断
         │         │  insert → run →  │
         │         │  parse → remove  │
         │         └────────┬─────────┘
         │                  │
         │         ┌────────▼─────────┐
         │         │  构建修复指令     │  ← 含错误信息 + 根因 + 探针结果 + 历史
         │         └────────┬─────────┘
         │                  │
         │         ┌────────▼─────────┐
         │         │  Editor.modify() │  ← 调用底层编辑器
         │         └────────┬─────────┘
         │                  │
         │         ┌────────▼─────────┐
         │         │  diff 大小检查   │  ← 最小修改约束
         │         └────────┬─────────┘
         │                  │
         │         ┌────────▼─────────┐
         │         │  runner_fn(new)  │  ← 运行验证
         │         └────────┬─────────┘
         │                  │
         │         success? │
         │          yes ──► │ 返回 FixResult(success=True)
         │                  │ no
         │         ┌────────▼─────────┐
         │         │  Scratchpad      │  ← 记录本轮结果
         │         │  .record()       │
         │         └────────┬─────────┘
         │                  │
         │         ┌────────▼─────────┐
         │         │  Rollback 判断   │  ← 错误增加？回滚到最优版本
         │         └────────┬─────────┘
         │                  │
         │         停止条件？│
         │          no ─────┘
         │          yes ──► 返回 FixResult(success=False)
         └─── 下一轮
```

#### 停止条件

1. **成功**：`runner_fn` 返回 `success=True`
2. **连续停滞**：连续 2 轮错误集合完全相同（`stale_count >= 2`）
3. **连续未变化**：连续 2 轮代码未产生任何修改
4. **达到上限**：轮次达到 `max_rounds`

---

### 3.2 ErrorAnalyzer

错误根因定位引擎。结合模式匹配和 LLM 深度分析。

#### 分析流程

1. **Traceback 结构化解析**：从错误文本中提取 `File "xxx", line N, in func` 格式的条目
2. **模式匹配分类**：根据预定义正则匹配错误类别
3. **可疑位置提取**：将 Traceback 行号映射到实际代码行
4. **修复策略决定**：
   - 有 Traceback → `targeted_fix`（直接修复）
   - 无 Traceback → `probe_first`（先插桩诊断）
5. **LLM 深度分析**：对复杂类型错误调用 LLM 做数据流追踪

#### 错误分类表

| 分类 | 匹配模式 | 修复策略 |
|---|---|---|
| `type_error` / `dtype_string` | `dtype('<U`, `ufunc.*add.reduce`, `unsupported operand.*str` | LLM 数据流追踪 |
| `key_error` | `KeyError` | 精准定位 |
| `index_error` | `IndexError` | 精准定位 |
| `attribute_error` | `AttributeError` | 精准定位 |
| `name_error` | `NameError`, `UnboundLocalError` | 精准定位 |
| `value_error` | `ValueError` | 精准定位 |
| `linalg_error` | `singular matrix`, `LinAlgError` | 精准定位 |
| `import_error` | `ImportError`, `ModuleNotFoundError` | 精准定位 |
| `syntax_error` | `SyntaxError` | 静态修复 |
| `unknown` | 以上均不匹配 | LLM 深度分析 |

#### LLM 深度分析触发条件

当错误类别为 `type_error`、`dtype_string` 或 `unknown` 时自动触发。发送包含带行号的完整代码和错误信息给 LLM，要求追踪数据流并给出根因。

---

### 3.3 ProbeManager

主动插桩诊断系统。当错误缺少 Traceback 时，自动在可疑位置插入诊断代码。

#### 工作流程

```
原始代码 ──► insert_probes() ──► 插桩代码 ──► runner_fn() ──► 输出
                                                                │
    原始代码 ◄── remove_probes() ◄─────────────────────────────┘
                                                                │
                         parse_probe_output() ◄─────────────────┘
                                │
                         format_probe_findings() ──► 诊断报告 ──► LLM
```

#### 探针格式

插入的探针行带有 `# __PROBE__` 标记，输出格式为：

```
@@PROBE@@L42@@factor_values=list/[0.5, 0.3, ...], weights=ndarray/[1.0, 1.0]@@
```

每个探针打印目标行中关键变量的 **类型** 和 **值**（截断到 80 字符）。

#### 探针位置选择

根据错误类别自动选择：

- **类型错误** (`type_error`, `dtype_string`)：匹配 `np.sum()`、`np.mean()`、`np.array()`、`.values`、`float()` 等数值操作
- **键/索引错误** (`key_error`, `index_error`)：匹配 `[key]`、`.loc[]`、`.iloc[]` 等索引操作

#### 安全保证

- 每个探针包裹在 `try: ... except: pass` 中，不会影响原始代码的执行逻辑
- `remove_probes()` 通过行标记 `# __PROBE__` 精确移除，不影响原始代码
- 最多插入 8 个探针（`max_probes` 参数）

---

### 3.4 DebugScratchpad

跨轮次调试记忆系统。

#### 核心功能

1. **记录每轮结果**：假设、修改、验证结果、结论
2. **维护排除列表**：当某个修复方向连续2轮无效时，加入排除列表
3. **生成 LLM 摘要**：`summary_for_llm()` 输出结构化的调试历史，注入到修复指令中

#### 摘要输出示例

```markdown
## 调试历史（最近几轮）

### 第1轮 [3个错误]
- 假设: 检测到 numpy 字符串类型混入数值计算...
- 修改: 15 行变更, strategy=search_replace, applied=3
- 结论: 有改善: 5 -> 3 个错误

### 第2轮 [3个错误]
- 假设: 检测到 numpy 字符串类型混入数值计算...
- 修改: 8 行变更, strategy=search_replace, applied=2
- 结论: 错误集合相同 (3 个)

## 已排除的方向（请勿重复尝试）
- 第2轮: 检测到 numpy 字符串类型混入数值计算...
```

#### `stale_count` 属性

统计从最近一轮往回看，连续多少轮的错误集合完全相同。用于判断停止条件。

---

### 3.5 RollbackManager

代码版本栈管理器。

#### 工作逻辑

- **push(code, error_count)**：每轮修改后推入版本栈，自动追踪最优版本
- **should_rollback(current_error_count)**：如果当前错误数超过最优版本的 1.5 倍，建议回滚
- **rollback()**：回滚到最优版本并返回其代码

#### 回滚时机

```
error_count 变化    决策
─────────────────────────────────
减少               接受新代码，更新最优版本
相同但内容不同      接受新代码
相同且内容相同      加入排除列表
增加 ≤ 1.5x        接受新代码（可能是中间状态）
增加 > 1.5x        回滚到最优版本
```

---

## 4. 使用示例

### 4.1 基本用法

```python
from core.utils.autonomous_code_agent import AutonomousCodeAgent, RunResult

# 创建 Agent
agent = AutonomousCodeAgent(llm_client=llm, editor_type="robust")

# 定义领域特定的运行验证函数
def my_runner(code: str) -> RunResult:
    """写入临时文件 → 动态加载 → 执行 → 捕获错误"""
    import tempfile, importlib.util, traceback
    with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as f:
        f.write(code)
        tmp_path = f.name

    try:
        spec = importlib.util.spec_from_file_location("test_mod", tmp_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        result = mod.main()  # 假设有 main() 函数
        return RunResult(success=True, output=str(result))
    except Exception as e:
        return RunResult(
            success=False,
            errors=[f"{type(e).__name__}: {e}"],
            traceback=traceback.format_exc(),
        )

# 修复
result = agent.fix_until_pass(
    code=original_code,
    runner_fn=my_runner,
    max_rounds=10,
    context="这是一个数据处理脚本",
    protected_names=["process_data", "DataConfig"],
)

if result.success:
    print(f"修复成功! 用了 {result.rounds_used} 轮")
    final_code = result.final_code
else:
    print(f"修复失败: {result.total_errors_initial} -> {result.total_errors_final} 个错误")
    print("调试日志:", "\n".join(result.debug_log))
```

### 4.2 在 eo_updater 中的实际用法

`eo_updater._fix_review_errors` 已重构为使用 `AutonomousCodeAgent`。核心变化：

```python
def _fix_review_errors(filepath, review_result, llm_client_name, ...):
    from core.utils.autonomous_code_agent import AutonomousCodeAgent, RunResult

    llm = _create_llm_client(llm_client_name)

    # 将 _review_single_file 封装为 runner_fn
    def runner_fn(code: str) -> RunResult:
        with open(temp_path, "w") as f:
            f.write(code)
        re_review = _review_single_file(temp_path, ...)
        # ... 转换 ReviewResult → RunResult
        return RunResult(success=..., errors=..., ...)

    # 一行调用代替 200 行循环
    agent = AutonomousCodeAgent(llm_client=llm, editor_type=editor_type)
    fix_result = agent.fix_until_pass(
        code=original_code,
        runner_fn=runner_fn,
        max_rounds=max_fix_rounds,
        context="这是高频因子调度框架文件...",
        protected_names=["AlphaConfig", "HighFreqFramework", ...],
    )
```

### 4.3 只做单次编辑（不含运行循环）

```python
agent = AutonomousCodeAgent(llm_client=llm)
edit_result = agent.edit(
    code=source_code,
    instruction="将 calculate_score 函数中的硬编码阈值 0.5 改为参数",
)
if edit_result.success:
    new_code = edit_result.new_code
```

### 4.4 禁用探针

对于不适合插桩的场景（如代码有副作用、写数据库等）：

```python
result = agent.fix_until_pass(
    code=code,
    runner_fn=runner_fn,
    enable_probes=False,  # 禁用插桩
)
```

---

## 5. runner_fn 编写指南

`runner_fn` 是 Agent 与具体业务领域的唯一接口。编写一个好的 `runner_fn` 是使用 Agent 的关键。

### 必须满足的契约

1. **签名**：`def runner_fn(code: str) -> RunResult`
2. **输入**：接收代码字符串（不是文件路径）
3. **输出**：返回 `RunResult` 实例
4. **幂等性**：对相同代码多次调用应返回相同结果
5. **安全性**：不应崩溃主进程（捕获所有异常）

### 最佳实践

```python
def runner_fn(code: str) -> RunResult:
    # 1. 写入临时文件（如果需要动态加载）
    with open(temp_path, "w", encoding="utf-8") as f:
        f.write(code)

    # 2. 执行验证（语法检查 + 运行时测试）
    try:
        result = your_validation_function(temp_path)
    except Exception as e:
        return RunResult(
            success=False,
            errors=[f"{type(e).__name__}: {e}"],
            traceback=traceback.format_exc(),
        )

    # 3. 尽量在 errors 中包含 Traceback 信息
    #    这是 Agent 能否精准修复的关键因素
    errors_with_tb = []
    for err in result.errors:
        errors_with_tb.append(err)  # 确保错误文本中包含 Traceback

    # 4. 区分"真正的错误"和"可接受的警告"
    #    只在 errors 中放需要修复的问题
    return RunResult(
        success=len(errors_with_tb) == 0,
        errors=errors_with_tb,
        output=captured_stdout,      # 探针诊断需要
        warnings=acceptable_warnings,
    )
```

### 关键点：Traceback 决定修复成功率

根据实际调试经验：

- **有 Traceback（含行号）**：Agent 通常 1-2 轮即可修复
- **无 Traceback**：Agent 需要先插桩诊断，增加 1-2 轮开销，且成功率下降

因此 `runner_fn` 应尽量在 errors 中保留完整的 Traceback 信息。

---

## 6. 配置与调优

### 推荐参数

| 场景 | `max_rounds` | `enable_probes` | `max_diff_lines` | `editor_type` |
|---|---|---|---|---|
| 简单 bug 修复 | 5 | True | 100 | robust |
| 复杂框架代码修复 | 10 | True | 200 | robust |
| 大文件 (>500行) | 10 | True | 300 | v2 |
| 快速迭代（省 token） | 3 | False | 100 | robust |

### LLM Token 消耗估算

每轮修复大约消耗：
- ErrorAnalyzer 深度分析：~2000 input + ~500 output tokens
- 底层编辑器调用：~3000-8000 input + ~1000-3000 output tokens
- 探针轮（如果触发）：额外 ~1500 input tokens

10 轮修复的总消耗约 50k-100k tokens。

---

## 7. 公开 API 一览

### 导出符号

```python
from core.utils.autonomous_code_agent import (
    # 主类
    AutonomousCodeAgent,

    # 数据类（调用方需要用到）
    RunResult,
    FixResult,

    # 组件（高级用法可单独使用）
    ErrorAnalyzer,
    ProbeManager,
    DebugScratchpad,
    RollbackManager,

    # 内部数据类（一般不需要直接使用）
    ErrorAnalysis,
    DebugRound,
    ProbeSpec,
    ProbeResult,
)
```
