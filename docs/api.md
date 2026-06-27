# sgar 对外接口

如果把 [usage.md](./usage.md) 看成“怎么用”，那这份文档就是“到底有哪些面可以接”。

它覆盖两类接口：

- CLI 接口
- Python 接口

如果你只是想快速上手，先看 [usage.md](./usage.md)。如果你想先建立整体认识，再看 [architecture.md](./architecture.md)。

## CLI 接口

## 从哪里进入

### `sgar`

这是标准命令行入口，也是大多数用户真正接触到的 `sgar`。

适合：

- 日常运行 `sgar`
- 使用 `config`
- 使用 SGAR runtime 命令
- 使用统一 mode 入口和 sugar 命令

### `python -m sgar`

这是与 `sgar` 等价的模块入口。

适合：

- 在没有 shell script 包装时直接运行
- 调试当前源码目录中的分发

## 配置接口

### `sgar config where`

作用：

- 显示用户级配置文件路径

输出：

- `~/.sgar/setting.ini` 的实际位置

### `sgar config list`

作用：

- 列出支持的 `ClientName`
- 列出可写入的 credential key 与 model key
- 显示示例命令

### `sgar config set`

作用：

- 写入用户级 LLM 配置

关键参数：

- `--client`
- `--api-key`
- `--model`
- `--key KEY=VALUE`

示例：

```bash
sgar config set --client SimpleDeepSeekClient --api-key YOUR_KEY --model deepseek-v4-pro
```

当某个 client 需要多个凭证键时：

```bash
sgar config set --client SparkClient \
  --key xunfei_spark_api_key=YOUR_KEY \
  --key xunfei_spark_secret_key=YOUR_SECRET \
  --model 4.0Ultra
```

## SGAR runtime 命令面

这组命令围绕 `.sgar/` 工作区和状态治理展开。你可以把它理解成 `sgar` 更偏“治理系统”的那一面。

### `sgar init`

作用：

- 初始化 `.sgar/` 工作区
- 可选写入初始 blueprint、roadmap、stage spec
- 若仓库已有 `.gitignore`，自动补入 `.sgar/` 与 `.sgarx/`

关键参数：

- `--project`
- `--force`
- `--blueprint-text`
- `--roadmap-text`
- `--stage-spec-text`
- `--stage`

退出码：

- `0` 成功
- `2` 参数或运行时错误

### `sgar status`

作用：

- 显示当前项目状态、阶段信息与治理摘要

### `sgar trace`

作用：

- 显示 SGAR 最近的操作轨迹摘要

### `sgar doctor`

作用：

- 检查缺失文件、状态不一致或工作区异常

退出码：

- `0` 状态健康
- `1` 检测到问题
- `2` 运行时错误

### `sgar set-blueprint`

作用：

- 写入 `blueprint.md`

关键参数：

- `--text`

### `sgar set-roadmap`

作用：

- 写入 `roadmap.md`

关键参数：

- `--text`

### `sgar set-stage-spec`

作用：

- 写入 `stages/<stage>/spec.md`

关键参数：

- `--stage`
- `--text`

### `sgar draft-blueprint`

作用：

- 用 LLM 草拟 blueprint

关键参数：

- `--prompt`
- `--llm-client`

### `sgar draft-roadmap`

作用：

- 用 LLM 草拟 roadmap

关键参数：

- `--prompt`
- `--llm-client`

### `sgar draft-stage-spec`

作用：

- 用 LLM 草拟指定 stage 的 spec

关键参数：

- `--stage`
- `--prompt`
- `--llm-client`

### `sgar validate`

作用：

- 验证治理文档

目标：

- `blueprint`
- `roadmap`
- `stage`

关键参数：

- `target`
- `--stage`
- `--accept`

退出码：

- `0` 验证通过
- `1` 验证未通过
- `2` 参数或运行时错误

### `sgar start-stage`

作用：

- 启动一个 stage

关键参数：

- `stage_id`

### `sgar verify`

作用：

- 记录验证结果与证据

关键参数：

- `--stage`
- `--criterion`
- `--pass`
- `--fail`
- `--all-pass`
- `--evidence`
- `--notes`
- `--artifact`

补充参数：

- 顶层还支持 `--run-checks` 与 `--check-timeout`，用于执行机器可检查退出准则

### `sgar close-stage`

作用：

- 关闭已验证完成的 stage

关键参数：

- `stage_id`

### `sgar mission create`

作用：

- 创建隔离 mission

关键参数：

- `--kind`
- `--id`
- `--input`
- `--objective`
- `--expected-output`
- `--scope`

### `sgar mission status`

作用：

- 查看 mission 状态

关键参数：

- `mission_id`

### `sgar mission complete`

作用：

- 提交 mission 结果工件

关键参数：

- `mission_id`
- `--result`

### `sgar mission list`

作用：

- 列出当前 missions

## 统一 mode 命令面

这组命令由顶层 `sgar/cli.py` 提供，背后调用的是 `core.ccx.CodeAgent`。你可以把它理解成 `sgar` 更偏“统一 agent 入口”的那一面。

### `sgar run --mode <mode> "<instruction>"`

作用：

- 统一运行任意支持的 `ccx` mode

当前支持的 mode：

- `sgar`
- `plan`
- `spec`
- `agent`
- `doc`
- `ask`
- `blueprint`
- `sgarx`
- `goal`
- `debug`

关键参数：

- `--mode`
- `--instruction`
- `--cwd`
- `--prompt-language`
- `--permission-mode`
- `--max-tool-rounds`
- `--docs-output-path`
- `--metadata-json`
- `--json`

退出码：

- `0` 运行成功
- `1` 结果返回 `failed=True`
- `2` 参数解析失败

示例：

```bash
sgar run --mode sgar "repair flaky tests and close the stage"
sgar run --mode agent --cwd /path/to/repo "fix the import error"
```

### sugar 子命令

这些命令等价于 `sgar run --mode ...`：

- `sgar plan`
- `sgar spec`
- `sgar agent`
- `sgar doc`
- `sgar ask`
- `sgar blueprint`
- `sgar sgarx`
- `sgar goal`
- `sgar debug`

示例：

```bash
sgar plan "design a migration plan"
sgar agent "implement the approved fix"
sgar sgarx "continue the governed coding run"
```

## Python 接口

## `from sgar import SgarRuntime`

导入路径：

```python
from sgar import SgarRuntime
```

定位：

- SGAR 状态治理运行时，也是把 `.sgar/` 带进你系统里的最直接入口

适用场景：

- 你想把 `.sgar/` 工作区、stage、verification、mission、trace 嵌入自己的系统

最常用方法：

- `init(project_name=None, force=False)`
- `status()`
- `set_blueprint(text)`
- `set_roadmap(text)`
- `set_stage_spec(stage_id, text)`
- `validate_blueprint()`
- `validate_roadmap()`
- `validate_stage_spec(stage_id)`
- `start_stage(stage_id)`
- `record_verification(...)`
- `close_stage(stage_id)`
- `doctor()`
- `create_mission(...)`

最小示例：

```python
from sgar import SgarRuntime

runtime = SgarRuntime("/path/to/repo")
runtime.init(project_name="demo-project")
print(runtime.status())
```

## `SgarStore`

导入路径：

```python
from sgar import SgarStore
```

定位：

- SGAR 工作区存储层

适用场景：

- 你要直接访问 `.sgar/` 内部文件布局
- 你在 runtime 之下做更细粒度的工程接入

## `from core.ccx import CodeAgent`

导入路径：

```python
from core.ccx import CodeAgent
```

定位：

- 多 mode 长程 agent 的统一 Python 入口

适用场景：

- 你想直接从代码里调用 `plan/spec/agent/doc/sgar/sgarx/goal/debug` 等 mode

最小示例：

```python
from core.ccx import AgentRunRequest, CodeAgent

agent = CodeAgent()
result = agent.run_sync(
    AgentRunRequest(
        instruction="fix the failing tests",
        cwd="/path/to/repo",
        agent_mode="agent",
        prompt_language="zh",
    )
)
print(result.final_text)
```

## `AgentRunRequest`

导入路径：

```python
from core.ccx import AgentRunRequest
```

关键字段：

- `instruction`
- `cwd`
- `config`
- `session`
- `max_tool_rounds`
- `prompt_language`
- `permission_mode`
- `agent_mode`
- `metadata`

用途：

- 描述一次 agent 运行请求

## `AgentRunResult`

导入路径：

```python
from core.ccx import AgentRunResult
```

关键字段：

- `final_text`
- `session_id`
- `turn_id`
- `cwd`
- `session_snapshot`
- `events`
- `messages`
- `tool_call_count`
- `failed`
- `error_code`
- `error_message`

用途：

- 表示一次运行的最终结果，也是你判断成功、失败、工件和 trace 摘要的主返回对象

## `CodeBuildRequest`

导入路径：

```python
from core.ccx import CodeBuildRequest
```

关键字段：

- `goal`
- `cwd`
- `context_paths`
- `constraints`
- `acceptance_criteria`
- `prompt_language`
- `permission_mode`
- `agent_mode`
- `metadata`

用途：

- 面向“给定目标、上下文和验收条件”的代码构建任务

## `run_code_agent()`

导入路径：

```python
from core.cc.api import run_code_agent
```

定位：

- 更轻量的同步包装

适用场景：

- 你只想发送一条自然语言指令，不想自己构建 `CodeAgent` 和 `AgentRunRequest`

最小示例：

```python
from core.cc.api import run_code_agent

result = run_code_agent(
    "inspect the repository, fix regressions, and summarize the outcome",
    cwd="/path/to/repo",
    prompt_language="zh",
    agent_mode="agent",
)
```

## `build_code_with_agent()`

导入路径：

```python
from core.cc.api import build_code_with_agent
```

定位：

- 面向更结构化的“代码构建/修复任务”

适用场景：

- 你想显式提供目标、上下文路径、约束与验收标准

最小示例：

```python
from core.cc.api import build_code_with_agent

result = build_code_with_agent(
    goal="Fix the failing tests and update docs if needed",
    cwd="/path/to/repo",
    context_paths=["README.md", "tests/test_app.py"],
    constraints=["Preserve the public API"],
    acceptance_criteria=["pytest tests/test_app.py -q passes"],
    prompt_language="zh",
    agent_mode="agent",
)
```

## 该选哪一层

如果你只是想把能力跑起来：

- 用 CLI API

如果你要把治理工作区嵌进系统：

- 用 `SgarRuntime`

如果你要直接调用多 mode agent：

- 用 `core.ccx.CodeAgent`

如果你只想要更轻的函数式调用：

- 用 `run_code_agent()` 或 `build_code_with_agent()`
