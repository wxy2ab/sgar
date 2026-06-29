# sgar 怎么接进你的系统

这份文档不讨论“命令怎么敲”，它讨论的是另一件更关键的事：应该把 `sgar` 放到你的系统哪一层，以及用哪种方式接，代价最低、边界最清楚。

因为从产品视角看，目标并不只是“接入一个代码 agent API”，而是让你的每个项目都有机会长出一个部署在业务环境里的内嵌 agent。它应该能贴着真实业务数据持续做 `auto research`、`self-improving`、`self-operation`，而不是只在离线开发阶段被调用一次。

如果你只想快速上手，请先看 [usage.md](./usage.md)。如果你想看完整接口清单，请看 [api.md](./api.md)。

## 先把角色看对

`sgar` 不是只有一种接法。

你可以把它看成三种东西的统一产品入口：

- 一个独立 CLI
- 一个状态治理 runtime
- 一个多 mode code agent 的入口壳

所以在动手接之前，先回答一个问题：

你要接入的是：

- 命令行能力
- 状态治理能力
- 代码编辑/自动修复能力

不同答案，会对应不同集成方式。

但如果你已经很明确地想做“项目内嵌 agent”，那可以把集成目标进一步具体化为：

- 让 agent 能接触业务上下文，而不只是仓库源码
- 让 agent 能在部署环境里持续运行，而不只是临时执行一次
- 让 agent 的研究、改进、操作过程都带有治理状态与审计痕迹

## 三种最常见的接入场景

## 1. 把它当成独立 CLI 接入

适合：

- CI/CD 流水线
- 内部运维脚本
- 定时巡检任务
- 开发者本地工具链

典型形式：

```bash
sgar init --project my-repo
sgar status
sgar run --mode sgar "repair flaky tests and summarize the result"
```

优点：

- 接入最轻
- 无需编写额外 Python 封装
- 很适合先验证产品价值

边界：

- 外部系统只能通过命令调用和结果文本/退出码交互
- 更复杂的运行编排需要你在外层自己做

## 2. 把它当成内部状态治理 runtime 接入

适合：

- 你已经有自己的任务编排系统
- 你想显式管理工作区、阶段、验证和 mission
- 你要把 agent 的推进过程纳入工程状态机

典型形式：

```python
from sgar import SgarRuntime

runtime = SgarRuntime("/path/to/repo")
runtime.init(project_name="demo-project")
runtime.set_stage_spec("stage-01", "...")
runtime.start_stage("stage-01")
```

优点：

- 外部系统可以完全掌控任务生命周期
- `.sgar/` 工作区就是治理记录
- 更适合“有外层编排器，`sgar` 做治理内核”的架构

边界：

- 你仍然需要自己决定何时调 LLM、何时执行 mode、何时推进 stage

## 3. 把它当成代码编辑或自动修复组件接入

适合：

- 研发平台
- 内部机器人
- 自动修复服务
- 带代码编辑能力的产品模块

典型形式：

```python
from core.cc.api import build_code_with_agent

result = build_code_with_agent(
    goal="Fix the failing tests and update docs if needed",
    cwd="/path/to/repo",
    context_paths=["README.md", "tests/test_app.py"],
    constraints=["Preserve the public API"],
    acceptance_criteria=["pytest tests/test_app.py -q passes"],
    agent_mode="agent",
)
```

优点：

- 最容易把自动改码能力嵌入自己的系统
- 接口更贴近“给目标 -> 返回结果”

边界：

- 如果你需要显式 stage 和治理文档，这一层本身不替代 `SgarRuntime`

## 更推荐的接法

## 模式 A：外部系统 + `sgar` CLI

推荐给：

- 想快速接进现有系统
- 想用 shell、任务队列或 CI 做编排

建议架构：

```text
your scheduler / CI / ops workflow
-> sgar CLI
-> .sgar workspace
-> trace / verification / artifacts
```

这是最适合先做 MVP 的接法。

## 模式 B：外部系统 + `SgarRuntime`

推荐给：

- 你已经有自己的编排器
- 你要显式控制阶段、验证和工作区

建议架构：

```text
your orchestrator
-> SgarRuntime
-> .sgar workspace
-> verification / missions / trace
```

这时 `sgar` 更像治理内核，而不是最终调度器。

## 模式 C：外部系统 + `CodeAgent`

推荐给：

- 你主要想获得多 mode 代码 agent 能力
- 你想把 `plan/spec/agent/doc/sgar/sgarx/...` 直接接进产品

建议架构：

```text
your product/service
-> core.ccx.CodeAgent
-> ccx mode execution
-> AgentRunResult / artifacts
```

## 到底该接哪一层

## 用 `sgar` CLI 的时候

- 你要最小接入成本
- 你用 shell/CI/任务平台就能解决编排
- 你先验证流程再决定是否进一步嵌入

## 用 `SgarRuntime` 的时候

- 你要显式治理工作区
- 你要阶段推进、mission、verification 和 trace
- 你要让系统持有硬状态，而不是只持有一次调用结果

## 用 `core.ccx.CodeAgent` 的时候

- 你要通用多 mode agent 能力
- 你要在代码里直接选择 `plan/spec/agent/doc/sgar/sgarx/goal/debug`
- 你不希望所有能力都先经过 SGAR runtime 命令面

## 用 `run_code_agent()` / `build_code_with_agent()` 的时候

- 你想快速集成
- 你更偏函数式调用，不想自己实例化 agent 和 request 对象

## `sgarx` 该怎么放

`sgarx` 更适合作为扩展 mode，而不是独立产品入口。

推荐写法：

```bash
sgar run --mode sgarx "..."
```

或：

```bash
sgar sgarx "..."
```

而不是把它当作一个单独的顶层二进制去设计。

这样做的好处是：

- 产品入口仍然统一叫 `sgar`
- `sgarx` 作为 mode 存在，语义更清楚
- 后续新增 mode 时不会导致产品名碎片化

## 接入时的几个工程建议

## 1. 先把配置边界管住

建议：

- 在开发者机器上使用 `sgar config set`
- 在服务或流水线环境中优先使用环境变量
- 不要把明文凭据直接写进仓库

## 2. 把工作区隔离想清楚

建议：

- 每个仓库独立维护自己的 `.sgar/`
- 如果同一仓库有多条并行任务线，用 `--session <id>` 做隔离

## 3. 别把治理目录提交进仓库

建议：

- 确保 `.sgar/` 与 `.sgarx/` 不进入版本控制
- 现有仓库如果已经有 `.gitignore`，`sgar init` 会自动补入

## 4. 给自己留出观察面

建议把下面几项纳入集成后的观察面：

- `sgar status`
- `sgar doctor`
- `sgar trace`
- verification evidence
- mission 状态

这样一旦运行失败，你不是只看到“失败”，而是能看到：

- 当前状态
- 失败发生在哪个阶段
- 最近 trace 做了什么
- 已记录的验证证据是什么

## 5. 把职责边界拆干净

推荐把职责拆清楚：

- 你的系统负责：
  - 何时发起任务
  - 何时停止任务
  - 何时把结果写回上游系统

- `sgar` 负责：
  - 工作区与硬状态
  - stage/verification/mission/trace
  - 统一的 code-agent 入口

- `ccx` 负责：
  - 多 mode 长程执行

## 一条更稳的接入路线

如果你想低风险推进，建议按这个顺序：

1. 先用 `sgar` CLI 做人工或半自动接入
2. 再把状态治理升级为 `SgarRuntime`
3. 最后把通用 code-agent 能力升级为 `CodeAgent` 或包装函数调用

这条路线的好处是：

- 每一步都可以独立验证
- 不需要一开始就把整个系统绑死在最底层接口上

## 继续看什么

- 想理解产品和代码分层，请看 [architecture.md](./architecture.md)
- 想快速操作命令，请看 [usage.md](./usage.md)
- 想看接口 reference，请看 [api.md](./api.md)
