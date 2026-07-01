---
name: sgar
description: Run a governed, state-machine-driven build or maintenance task where completion is decided by machine-checkable exit criteria, not the agent's say-so. Use when you need long-horizon, auditable, won't-lie-about-done execution; each stage advances only when its [check: <shell cmd>] gates pass under hermetic verification. Invoke via `python cli.py <command>`.
---

# SGAR skill

SGAR(State-Governed Agent Regime)把一个自治的构建/维护循环治理起来，使 agent
**无法自证一个虚假的"已完成"**。状态存在 LLM 上下文之外；一个 stage 只有在它的
可机器校验退出准则通过时才被承认推进。

## 何时使用
- 需要无人值守、长程、可审计地推进一个有明确验收标准的任务
- 你希望"完成"由 `[check: <cmd>]` 的退出码裁定，而不是由模型自己宣布
- 需要分阶段推进、每阶段留可核验证据并支持回滚

## 如何调用
```bash
python cli.py <command> [options]
```
首次使用先安装依赖：`pip install -r requirements.txt`。

用户级 LLM 配置可直接写入 `~/.sgar/setting.ini`：

```bash
python cli.py config where
python cli.py config list
python cli.py config set --client SimpleDeepSeekClient --api-key YOUR_KEY --model deepseek-v4-pro
```

如果某个 client 需要多个凭证键，请使用可重复参数 `--key KEY=VALUE`。

## 两类入口

### 1. 治理型 CLI
适合直接操纵 `.sgar/` 状态机、治理文档和阶段流转。

```bash
python cli.py [global options] <command> [command options]
```

### 2. 统一 agent 入口
适合把 `sgar` 当统一产品 CLI 来跑 `plan/spec/agent/doc/sgarx/goal/...` 等模式。

```bash
python cli.py run --mode <mode> [run options] "<instruction>"
python cli.py plan "<instruction>"
python cli.py spec "<instruction>"
python cli.py agent "<instruction>"
python cli.py doc "<instruction>"
python cli.py ask "<instruction>"
python cli.py blueprint "<instruction>"
python cli.py sgarx "<instruction>"
python cli.py goal "<instruction>"
python cli.py debug "<instruction>"
```

## 治理型 CLI: 全局参数
这些参数位于子命令之前，对所有治理命令都有效。

| 参数 | 说明 |
|------|------|
| `--cwd <dir>` | 项目根目录，默认当前目录。`.sgar/` 工作区在这里解析。 |
| `--session <id>` | 使用隔离的 session，状态存放在 `.sgar/sessions/<id>/` 下。适合并行治理多个任务。 |
| `--run-checks` | 在 `verify` / `close-stage` 时真实执行 stage spec 里的 `[check: <cmd>]`。不加时默认依赖人工填报结果。 |
| `--check-timeout <seconds>` | `--run-checks` 开启后每个 check 的超时时间，默认 `120.0` 秒。 |

## 治理型 CLI: 常用命令

### `init`
初始化 `.sgar/` 工作区。

```bash
python cli.py init [--project <name>] [--force] [--blueprint-text <text>] [--roadmap-text <text>] [--stage-spec-text <text>] [--stage <stage-id>]
```

| 参数 | 说明 |
|------|------|
| `--project <name>` | 显式指定项目名。 |
| `--force` | 强制重建已有工作区，会清空现有治理状态。 |
| `--blueprint-text <text>` | 初始化后立即写入 `blueprint.md`。 |
| `--roadmap-text <text>` | 初始化后立即写入 `roadmap.md`。 |
| `--stage-spec-text <text>` | 初始化后立即写入某个 stage 的 `spec.md`。 |
| `--stage <stage-id>` | 与 `--stage-spec-text` 配合使用，默认 `stage-01`。 |

### `status`
查看当前治理状态。

```bash
python cli.py status
```

### `set-blueprint`
写入蓝图文档。

```bash
python cli.py set-blueprint --text "<markdown>"
```

| 参数 | 说明 |
|------|------|
| `--text <markdown>` | 完整蓝图正文，必填。 |

### `set-roadmap`
写入路线图文档。

```bash
python cli.py set-roadmap --text "<markdown>"
```

| 参数 | 说明 |
|------|------|
| `--text <markdown>` | 完整路线图正文，必填。 |

### `set-stage-spec`
写入阶段规格文档。

```bash
python cli.py set-stage-spec --stage <stage-id> --text "<markdown>"
```

| 参数 | 说明 |
|------|------|
| `--stage <stage-id>` | 目标阶段 ID，必填。 |
| `--text <markdown>` | 阶段规格正文，必填。 |

### `draft-blueprint` / `draft-roadmap` / `draft-stage-spec`
调用 LLM 起草治理文档。

```bash
python cli.py draft-blueprint [--prompt "<extra prompt>"] [--llm-client <ClientName>]
python cli.py draft-roadmap [--prompt "<extra prompt>"] [--llm-client <ClientName>]
python cli.py draft-stage-spec --stage <stage-id> [--prompt "<extra prompt>"] [--llm-client <ClientName>]
```

| 参数 | 说明 |
|------|------|
| `--stage <stage-id>` | 仅 `draft-stage-spec` 需要，指定阶段。 |
| `--prompt <text>` | 给 LLM 的补充提示，默认空字符串。 |
| `--llm-client <ClientName>` | 起草所用模型客户端，默认 `SimpleDeepSeekClient`。 |

### `validate`
校验治理文档是否满足 SGAR 约束。

```bash
python cli.py validate blueprint [--accept]
python cli.py validate roadmap [--accept]
python cli.py validate stage --stage <stage-id> [--accept]
```

| 参数 | 说明 |
|------|------|
| `blueprint \| roadmap \| stage` | 目标文档类型，位置参数，必填。 |
| `--stage <stage-id>` | 当目标为 `stage` 时必填。 |
| `--accept` | 对 blueprint / roadmap 校验通过后顺带接受该版本。 |

### `start-stage`
把某个阶段切换为进行中。

```bash
python cli.py start-stage <stage-id>
```

### `verify`
记录阶段验证证据；这是治理闭环里的关键命令。

```bash
python cli.py verify --stage <stage-id> --criterion <criterion-id> (--pass | --fail) [--evidence <text>] [--notes <text>] [--artifact <path> ...]
python cli.py verify --stage <stage-id> --all-pass --evidence <text> [--notes <text>] [--artifact <path> ...]
```

| 参数 | 说明 |
|------|------|
| `--stage <stage-id>` | 目标阶段，必填。 |
| `--criterion <criterion-id>` | 目标验收项 ID。与 `--all-pass` 二选一。 |
| `--pass` | 将该 criterion 记为通过。 |
| `--fail` | 将该 criterion 记为失败。 |
| `--all-pass` | 读取该 stage spec 中全部 exit criteria，并一次性全部记为通过。要求同时提供 `--evidence`。 |
| `--evidence <text>` | 验证证据文本。单条验证时可为空；`--all-pass` 时必填。 |
| `--notes <text>` | 附加说明，写入验证记录。 |
| `--artifact <path>` | 额外关联的证据文件路径，可重复传入。 |

### `close-stage`
关闭一个已经验证完成的阶段。

```bash
python cli.py close-stage <stage-id>
```

### `mission create`
创建文件系统隔离的 mission。

```bash
python cli.py mission create --kind <kind> --id <mission-id> --input <path> ... --objective "<text>" --expected-output <path-or-desc> ... [--scope <path> ...]
```

| 参数 | 说明 |
|------|------|
| `--kind <kind>` | mission 类型，必填。 |
| `--id <mission-id>` | mission ID，必填。 |
| `--input <path>` | 输入文件/目录，可重复传入，必填。 |
| `--objective <text>` | 任务目标，必填。 |
| `--expected-output <path-or-desc>` | 预期产物，可重复传入，必填。 |
| `--scope <path>` | 允许操作的作用域路径，可重复传入；不传时由调用方自己约束。 |

### `mission status`
查看单个 mission 状态。

```bash
python cli.py mission status <mission-id>
```

### `mission complete`
将 mission 标记完成，并登记结果产物。

```bash
python cli.py mission complete <mission-id> --result <path>
```

| 参数 | 说明 |
|------|------|
| `--result <path>` | 结果文件或结果目录，必填。 |

### `mission list`
列出全部 mission。

```bash
python cli.py mission list
```

### `doctor`
检测缺失文件、配置错误或治理状态不一致。

```bash
python cli.py doctor
```

### `trace`
查看 SGAR 操作轨迹摘要。

```bash
python cli.py trace
```

## 统一 agent 入口: `run` 与快捷命令

### `run`
```bash
python cli.py run --mode <mode> [run options] "<instruction>"
```

| 参数 | 说明 |
|------|------|
| `--mode <mode>` | 必填。可选值包括 `plan`、`spec`、`agent`、`doc`、`ask`、`blueprint`、`sgar`、`sgarx`、`goal`、`debug`。 |
| `"<instruction>"` | 位置参数形式的任务指令。 |
| `--instruction "<text>"` | 当指令以 `-` 开头或不适合做位置参数时使用。 |
| `--cwd <dir>` | 运行目录，默认当前目录。 |
| `--prompt-language <lang>` | 本次运行覆盖提示词语言。 |
| `--permission-mode <mode>` | 本次运行覆盖权限模式。 |
| `--max-tool-rounds <n>` | 限制工具调用轮数。 |
| `--docs-output-path <path>` | 为 `doc` / `goal` 等产物型模式指定输出路径。 |
| `--metadata-json <json-object>` | 注入额外请求元数据，必须解码为 JSON object。 |
| `--json` | 输出完整 `AgentRunResult` JSON，而不是纯文本摘要。 |

### 快捷命令
这些命令等价于 `python cli.py run --mode <mode> ...`：

| 快捷命令 | 等价形式 |
|------|------|
| `python cli.py plan "<instruction>"` | `python cli.py run --mode plan "<instruction>"` |
| `python cli.py spec "<instruction>"` | `python cli.py run --mode spec "<instruction>"` |
| `python cli.py agent "<instruction>"` | `python cli.py run --mode agent "<instruction>"` |
| `python cli.py doc "<instruction>"` | `python cli.py run --mode doc "<instruction>"` |
| `python cli.py ask "<instruction>"` | `python cli.py run --mode ask "<instruction>"` |
| `python cli.py blueprint "<instruction>"` | `python cli.py run --mode blueprint "<instruction>"` |
| `python cli.py sgarx "<instruction>"` | `python cli.py run --mode sgarx "<instruction>"` |
| `python cli.py goal "<instruction>"` | `python cli.py run --mode goal "<instruction>"` |
| `python cli.py debug "<instruction>"` | `python cli.py run --mode debug "<instruction>"` |

## 用户配置命令: `config`

```bash
python cli.py config where
python cli.py config list
python cli.py config set --client <ClientName> [--api-key <key>] [--model <model>] [--key KEY=VALUE ...]
```

| 子命令 / 参数 | 说明 |
|------|------|
| `where` | 输出用户配置路径，即 `~/.sgar/setting.ini`。 |
| `list` | 列出支持的 `ClientName`、凭证键、模型键和一行示例。 |
| `set --client <ClientName>` | 必填；选择要写入的客户端配置。 |
| `set --api-key <key>` | 仅当该 client 恰好只有一个凭证键时可用。 |
| `set --model <model>` | 当该 client 有模型键时写入首选模型。 |
| `set --key KEY=VALUE` | 显式写任意配置键，可重复传入；多凭证 client 应优先用它。 |

完整选项：`python cli.py <command> --help`。

## 退出准则契约
在阶段规格里用 `[check: <shell cmd>]` 声明可机器校验的验收标准，其**退出码即权威**
(0 = 通过)。加 `--run-checks` 后 SGAR 会真正执行这些 check 并拒绝与之矛盾的"通过"；
默认启用 hermetic 执行(剥离 user-site / cwd 注入)，降低 check 被绕过作弊的风险。

## 给调用方 agent 的提示
- 永远给任务一个可自检的目标("让 X check 通过")，而非模糊目标。
- 把 `python cli.py status` / `trace` 的输出当权威进度，而非自己的叙述。
- 某 stage 的 check 过不了就如实报告失败，**不要删改 check 或测试来骗过它**。
