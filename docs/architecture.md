# sgar 是怎么构成的

如果把 `sgar` 只看成一个 CLI，你会低估它；如果把它只看成一个代码 agent，你又会忽略它真正的独特之处。

`sgar` 更准确的形态是：一个把产品入口、状态治理、长程 mode 执行和底层运行引擎包在一起的工程系统。它既能独立使用，也能嵌入其他平台，让自动化修复、自动化运维和长程代码编辑真正变成可运行、可追踪、可治理的能力。

## 一眼看懂

从外到内，`sgar` 可以分成 4 层：

1. 产品入口层  
   用户直接接触的是 `sgar` CLI 与 `python -m sgar`。这一层负责把不同命令路由到 SGAR runtime 或 `ccx` mode。

2. 状态治理层  
   SGAR runtime 管理 `.sgar/` 工作区、阶段、验证记录、mission 和 trace，让 agent 不是“跑一次就结束”，而是沿着硬状态推进。

3. mode 执行层  
   `ccx` 提供多种 agent mode，例如 `plan`、`spec`、`agent`、`doc`、`sgar`、`sgarx`、`goal`、`debug`。这是多模式长程执行的核心能力面。

4. 底层执行引擎层  
   `deepstack_v5` 负责图式调度、节点执行、事件、持久化和并发，是 `ccx` 的底层运行时。

## 产品是怎么拼起来的

### 1. 最外层：产品入口

- `sgar/cli.py`  
  顶层统一 CLI。它同时处理：
  - `sgar config`
  - SGAR runtime 命令透传，如 `init`、`status`
  - `sgar run --mode ...`
  - `sgar plan`、`sgar agent`、`sgar sgarx` 等快捷子命令

- `sgar/__main__.py`  
  支持 `python -m sgar`

- `sgar/config_cli.py`  
  用户级配置入口，负责写入 `~/.sgar/setting.ini`

### 2. 中间层：SGAR runtime

- `core/ccx/sgar/cli.py`  
  SGAR runtime 的命令行 surface，负责 `init`、`status`、`validate`、`verify`、`mission` 等治理型命令

- `core/ccx/sgar/runtime.py`  
  `SgarRuntime` 的主要实现。对外最重要的方法包括：
  - `init()`
  - `status()`
  - `set_blueprint()`
  - `set_roadmap()`
  - `set_stage_spec()`
  - `validate_*()`
  - `start_stage()`
  - `record_verification()`
  - `close_stage()`
  - `doctor()`
  - `create_mission()`

- `core/ccx/sgar/store.py`  
  负责 `.sgar/` 工作区文件与状态持久化

- `core/ccx/sgar/tracing.py`  
  负责 trace 记录与读取

### 3. 能力层：ccx mode

- `core/ccx/api.py`  
  `ccx` 的统一 API 入口，提供 `CodeAgent` 与 mode 调度

- `core/ccx/__init__.py`  
  对外暴露：
  - `CodeAgent`
  - `AgentRunRequest`
  - `AgentRunResult`
  - `CodeBuildRequest`

- `core/ccx/modes/`  
  各种 mode 的实现，包括：
  - `plan`
  - `spec`
  - `agent`
  - `doc`
  - `ask`
  - `blueprint`
  - `sgarx`
  - `watch`

当前 `ccx` 直接支持的 mode 集包括：

- `plan`
- `spec`
- `agent`
- `doc`
- `ask`
- `blueprint`
- `sgar`
- `sgarx`
- `goal`
- `debug`

## `sgar`、`sgarx` 和 `ccx` 到底是什么关系

### `sgar`

`sgar` 是整个产品的入口名，同时也是默认的状态治理型 coding-agent 工作流名。

它强调：

- 明确的工作区
- 阶段推进
- 验证与审计
- 可回溯的执行痕迹

### `sgarx`

`sgarx` 是扩展 mode，不是另一个独立产品。

它更适合放在统一入口之下，通过：

```bash
sgar run --mode sgarx "..."
```

或：

```bash
sgar sgarx "..."
```

来调用。

它的状态空间与 `.sgar/` 区分开，通常落在 `.sgarx/`。

### `ccx`

`ccx` 是更底层的多 mode agent/runtime 能力层。`sgar` 做的事情，是把它产品化、路由化，再叠加上状态治理和统一 CLI。

一句话说：

- `ccx` 是能力层
- `sgar` 是产品入口和治理外壳
- `sgarx` 是 `ccx` 中被 `sgar` 暴露出来的扩展 mode

## 为什么 `sgar` 强调工作区和硬状态

### `.sgar/`

`sgar init` 后会在仓库内建立 `.sgar/`，用于保存：

- `config.json`
- `state.json`
- `blueprint.md`
- `roadmap.md`
- `stages/<stage>/spec.md`
- `missions/`
- trace 与验证工件

这意味着 `sgar` 的推进不是只靠 prompt 和记忆，而是靠外部化、可检查、可审计的硬状态。

### `.sgarx/`

`.sgarx/` 是扩展 mode 的独立状态空间。它不应该和 `.sgar/` 混写，也不应该被误解成普通缓存目录。

### session 隔离

如果传入 `--session <id>`，SGAR 会把状态隔离到：

```text
.sgar/sessions/<id>/
```

这允许一个仓库里同时维护多条隔离的治理运行线。

## 它为什么能长程运行

`sgar` 的运行模型建立在两个核心思想上：

- Audit Engineering  
  不把“生成了什么”当作终点，而把“如何验证、如何发现缺陷、如何据此继续修正”视为核心循环。

- State-Governed Agent Regime  
  不让 agent 只靠提示词漂移，而是靠外部状态、阶段转换、action 和 delta 持续推进。

所以 `sgar` 真正追求的不是“这一轮答得多漂亮”，而是：

- 长程可持续推进
- 可验证
- 可恢复
- 可回溯
- 可集成

## 实际运行时会发生什么

### 路径一：治理型命令会怎么走

```text
sgar init/status/verify
-> sgar/cli.py
-> core/ccx/sgar/cli.py
-> SgarRuntime
-> .sgar/ workspace
-> state / trace / verification artifacts
```

### 路径二：统一 mode 命令会怎么走

```text
sgar run --mode plan|agent|sgar|sgarx|...
-> sgar/cli.py
-> core.ccx.CodeAgent
-> core/ccx/api.py
-> deepstack_v5 engine
-> AgentRunResult / artifacts / trace
```

## 接下来读什么

如果你刚接触这个项目，推荐按这个顺序继续：

1. [README.md](../README.md)
2. [usage.md](./usage.md)
3. [api.md](./api.md)
4. [integration.md](./integration.md)

如果你更关心它背后的方法论，再回到：

- Audit Engineering
- State-Governed Agent Regime
