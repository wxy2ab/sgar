# sgar 怎么用

这份文档不打算把你淹没在所有命令里，它只想先帮你把两件事讲清楚：

- 怎么把 `sgar` 跑起来
- 什么时候该用 SGAR runtime workflow，什么时候该用统一 mode workflow

如果你想看完整接口列表，请继续阅读 [api.md](./api.md)。如果你想把 `sgar` 嵌入自己的系统，请继续阅读 [integration.md](./integration.md)。

## 先跑起来

### 1. 安装

要求：

- Python `>= 3.12`

安装：

```bash
pip install sgar
```

### 2. 配好模型

如果你只想先把它用起来，最短路径就是：

```bash
sgar config set --client SimpleDeepSeekClient --api-key YOUR_KEY --model deepseek-v4-pro
```

常用辅助命令：

```bash
sgar config where
sgar config list
```

### 3. 初始化仓库工作区

```bash
sgar init --project my-repo
```

初始化后会在当前仓库生成 `.sgar/` 工作区。

如果仓库根目录已经存在 `.gitignore`，`sgar init` 还会自动补入：

```text
.sgar/
.sgarx/
```

### 4. 看看它现在在哪一步

```bash
sgar status
```

最常见的日常命令还有：

```bash
sgar doctor
sgar trace
```

## 你会用到的两条主路径

`sgar` 不是只有一种用法。大多数时候，你会在下面两条路径里选一条。

### 路径一：把它当成治理工作流来用

这条路径更适合你在意这些事情的时候：

- 你要显式维护阶段和治理文档
- 你要跟踪 blueprint、roadmap、stage spec
- 你要积累验证记录、mission 和 trace

最常见的命令包括：

```bash
sgar init --project my-repo
sgar status
sgar set-blueprint --text "..."
sgar set-roadmap --text "..."
sgar set-stage-spec --stage stage-01 --text "..."
sgar validate blueprint --accept
sgar start-stage stage-01
sgar verify --stage stage-01 --criterion c1 --pass --evidence "pytest -q"
sgar close-stage stage-01
```

### 路径二：把它当成统一 mode 入口来用

这条路径更适合你在意这些事情的时候：

- 你想直接运行某种 `ccx` mode
- 你更关心一次长程执行，而不是完整治理文档流
- 你想统一用 `sgar` 作为多 mode 的外层入口

统一入口：

```bash
sgar run --mode sgar "repair flaky tests and close the stage"
sgar run --mode plan "design the migration plan"
sgar run --mode agent "fix the import error and summarize the change"
```

快捷写法：

```bash
sgar plan "design the migration plan"
sgar spec "write a stage-ready spec"
sgar agent "fix the import error"
sgar doc "generate repository documentation"
sgar ask "explain how the runtime is wired"
sgar blueprint "draft a governed project blueprint"
sgar sgarx "continue a harder governed coding run"
sgar goal "reach a verifiable goal state"
sgar debug "investigate and iteratively verify a stubborn bug"
```

## 首页级最常见命令

### 看帮助、配模型

```bash
sgar --help
sgar config --help
sgar config where
sgar config list
```

### 跑治理工作流

```bash
sgar init --project my-repo
sgar status
sgar doctor
sgar trace
```

### 跑 mode

```bash
sgar run --mode sgar "..."
sgar run --mode agent "..."
sgar plan "..."
sgar agent "..."
sgar sgarx "..."
```

### 常用补充参数

统一 mode 命令常用的附加参数包括：

```bash
--cwd /path/to/repo
--prompt-language zh
--permission-mode default
--max-tool-rounds 12
--metadata-json '{"ccx_contract": {"kind": "demo"}}'
--json
```

## 仓库里会多出什么

典型 `.sgar/` 结构大概长这样：

```text
.sgar/
  config.json
  state.json
  blueprint.md
  roadmap.md
  stages/
    stage-01/
      spec.md
  missions/
```

如果你使用 session 隔离：

```bash
sgar --session demo-01 status
```

状态会落在：

```text
.sgar/sessions/demo-01/
```

## 你大概率会这样用

### 示例 1：先把治理工作区建起来

```bash
sgar config set --client SimpleDeepSeekClient --api-key YOUR_KEY --model deepseek-v4-pro
sgar init --project my-repo
sgar status
```

### 示例 2：直接跑一次治理型代码任务

```bash
sgar run --mode sgar "repair the flaky tests, record the evidence, and summarize the result"
```

这适合你已经接受 `sgar` 不是一次性改代码工具，而是一个围绕状态、验证和痕迹工作的长程 mode。

### 示例 3：先规划，再落地

```bash
sgar plan "design a migration plan for the auth module"
sgar agent "implement the approved migration with minimal changes"
```

这条路径更接近通用 `ccx` mode 的使用方式。

### 示例 4：把阶段文档和验证都补完整

```bash
sgar set-stage-spec --stage stage-01 --text "..."
sgar validate stage --stage stage-01
sgar start-stage stage-01
sgar verify --stage stage-01 --criterion c1 --pass --evidence "pytest tests/test_api.py -q"
sgar close-stage stage-01
```

## 到底该选哪条路径

如果你更像在搭一个“长期运行、可治理”的工程流，优先用 SGAR runtime workflow：

- 你要显式管理 stage、verification、mission 和 trace
- 你希望工作区本身成为治理记录
- 你要把 agent 运行纳入“硬状态推进”

如果你更像在调用一组统一的 agent 能力，优先用统一 mode workflow：

- 你只想调用某个 mode
- 你需要 `plan/spec/agent/doc/ask` 这类通用能力
- 你想把 `sgar` 作为 `ccx` 的统一产品入口

## 遇到问题先看哪里

### 配置没生效

先检查：

```bash
sgar config where
sgar config list
```

并确认 `~/.sgar/setting.ini` 或环境变量已包含所需 key。

### 工作区还没建好

如果 `status`、`trace`、`doctor`、`validate`、`verify` 等命令报工作区相关错误，先运行：

```bash
sgar init --project my-repo
```

### 跑失败了先看什么

优先查看：

```bash
sgar doctor
sgar trace
```

如果是 mode 执行失败，再结合 `--json` 查看完整结果：

```bash
sgar run --mode agent --json "fix the failing import path"
```

## 继续往下看

- 想看完整 CLI 与 Python 接口，请阅读 [api.md](./api.md)
- 想理解系统分层，请阅读 [architecture.md](./architecture.md)
- 想把它接进自己的平台或服务，请阅读 [integration.md](./integration.md)
