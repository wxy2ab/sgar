# sgar API Reference

This document covers two interface surfaces:

- the CLI API
- the Python API

If you only want to get started quickly, read [usage.en.md](./usage.en.md). If you want the system structure, read [architecture.en.md](./architecture.en.md).

## CLI API

## Top-level entrypoints

### `sgar`

The standard command-line entrypoint.

Use it for:

- day-to-day `sgar` usage
- `config`
- SGAR runtime commands
- the unified mode entrypoint and shortcut commands

### `python -m sgar`

The module entrypoint equivalent to `sgar`.

Use it when:

- you want to run through Python directly
- you are working from a source checkout

## `config` subcommands

### `sgar config where`

Purpose:

- show the user-level config file path

Output:

- the resolved location of `~/.sgar/setting.ini`

### `sgar config list`

Purpose:

- list supported `ClientName` values
- show credential keys and model keys
- print example commands

### `sgar config set`

Purpose:

- write user-level LLM configuration

Key flags:

- `--client`
- `--api-key`
- `--model`
- `--key KEY=VALUE`

Example:

```bash
sgar config set --client SimpleDeepSeekClient --api-key YOUR_KEY --model deepseek-v4-pro
```

For clients that need multiple credentials:

```bash
sgar config set --client SparkClient \
  --key xunfei_spark_api_key=YOUR_KEY \
  --key xunfei_spark_secret_key=YOUR_SECRET \
  --model 4.0Ultra
```

## SGAR runtime subcommands

These commands revolve around the `.sgar/` workspace and the governance workflow.

### `sgar init`

Purpose:

- initialize the `.sgar/` workspace
- optionally write the initial blueprint, roadmap, and stage spec
- auto-add `.sgar/` and `.sgarx/` to `.gitignore` if the repository already has one

Key flags:

- `--project`
- `--force`
- `--blueprint-text`
- `--roadmap-text`
- `--stage-spec-text`
- `--stage`

Exit codes:

- `0` success
- `2` argument or runtime error

### `sgar status`

Purpose:

- show the current project state, stage, and governance summary

### `sgar trace`

Purpose:

- show a summary of recent SGAR operation traces

### `sgar doctor`

Purpose:

- detect missing files, inconsistent state, or workspace problems

Exit codes:

- `0` healthy
- `1` issues detected
- `2` argument or runtime error

### `sgar set-blueprint`

Purpose:

- write `blueprint.md`

Key flags:

- `--text`

### `sgar set-roadmap`

Purpose:

- write `roadmap.md`

Key flags:

- `--text`

### `sgar set-stage-spec`

Purpose:

- write `stages/<stage>/spec.md`

Key flags:

- `--stage`
- `--text`

### `sgar draft-blueprint`

Purpose:

- draft a blueprint with an LLM

Key flags:

- `--prompt`
- `--llm-client`

### `sgar draft-roadmap`

Purpose:

- draft a roadmap with an LLM

Key flags:

- `--prompt`
- `--llm-client`

### `sgar draft-stage-spec`

Purpose:

- draft a stage spec with an LLM

Key flags:

- `--stage`
- `--prompt`
- `--llm-client`

### `sgar validate`

Purpose:

- validate governance documents

Targets:

- `blueprint`
- `roadmap`
- `stage`

Key flags:

- `target`
- `--stage`
- `--accept`

Exit codes:

- `0` validation passed
- `1` validation failed
- `2` argument or runtime error

### `sgar start-stage`

Purpose:

- start a stage

Key args:

- `stage_id`

### `sgar verify`

Purpose:

- record verification results and evidence

Key flags:

- `--stage`
- `--criterion`
- `--pass`
- `--fail`
- `--all-pass`
- `--evidence`
- `--notes`
- `--artifact`

Additional top-level flags:

- `--run-checks`
- `--check-timeout`

These enable machine-checkable exit criteria.

### `sgar close-stage`

Purpose:

- close a verified stage

Key args:

- `stage_id`

### `sgar mission create`

Purpose:

- create an isolated mission

Key flags:

- `--kind`
- `--id`
- `--input`
- `--objective`
- `--expected-output`
- `--scope`

### `sgar mission status`

Purpose:

- show mission status

Key args:

- `mission_id`

### `sgar mission complete`

Purpose:

- complete a mission with a result artifact

Key args:

- `mission_id`
- `--result`

### `sgar mission list`

Purpose:

- list current missions

## Unified mode subcommands

These commands are provided by the top-level `sgar/cli.py` wrapper and call `core.ccx.CodeAgent` underneath.

### `sgar run --mode <mode> "<instruction>"`

Purpose:

- run any supported `ccx` mode through one unified entrypoint

Currently supported modes:

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

Key flags:

- `--mode`
- `--instruction`
- `--cwd`
- `--prompt-language`
- `--permission-mode`
- `--max-tool-rounds`
- `--docs-output-path`
- `--metadata-json`
- `--json`

Exit codes:

- `0` success
- `1` the returned result has `failed=True`
- `2` argument parsing failure

Examples:

```bash
sgar run --mode sgar "repair flaky tests and close the stage"
sgar run --mode agent --cwd /path/to/repo "fix the import error"
```

### Shortcut subcommands

These are equivalent to `sgar run --mode ...`:

- `sgar plan`
- `sgar spec`
- `sgar agent`
- `sgar doc`
- `sgar ask`
- `sgar blueprint`
- `sgar sgarx`
- `sgar goal`
- `sgar debug`

Examples:

```bash
sgar plan "design a migration plan"
sgar agent "implement the approved fix"
sgar sgarx "continue the governed coding run"
```

## Python API

## `from sgar import SgarRuntime`

Import path:

```python
from sgar import SgarRuntime
```

Role:

- the SGAR state-governed runtime

Best for:

- embedding `.sgar/` workspace management, stages, verification, missions, and traces into your own system

Most common methods:

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

Minimal example:

```python
from sgar import SgarRuntime

runtime = SgarRuntime("/path/to/repo")
runtime.init(project_name="demo-project")
print(runtime.status())
```

## `SgarStore`

Import path:

```python
from sgar import SgarStore
```

Role:

- the SGAR workspace storage layer

Best for:

- direct access to `.sgar/` internals
- lower-level integration underneath the runtime

## `from core.ccx import CodeAgent`

Import path:

```python
from core.ccx import CodeAgent
```

Role:

- the unified Python entrypoint for long-running multi-mode agent execution

Best for:

- directly calling modes such as `plan`, `spec`, `agent`, `doc`, `sgar`, `sgarx`, `goal`, and `debug`

Minimal example:

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

Import path:

```python
from core.ccx import AgentRunRequest
```

Key fields:

- `instruction`
- `cwd`
- `config`
- `session`
- `max_tool_rounds`
- `prompt_language`
- `permission_mode`
- `agent_mode`
- `metadata`

Purpose:

- describe one agent run request

## `AgentRunResult`

Import path:

```python
from core.ccx import AgentRunResult
```

Key fields:

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

Purpose:

- represent the terminal result of one run

## `CodeBuildRequest`

Import path:

```python
from core.ccx import CodeBuildRequest
```

Key fields:

- `goal`
- `cwd`
- `context_paths`
- `constraints`
- `acceptance_criteria`
- `prompt_language`
- `permission_mode`
- `agent_mode`
- `metadata`

Purpose:

- represent a more structured code build or repair task

## `run_code_agent()`

Import path:

```python
from core.cc.api import run_code_agent
```

Role:

- a lighter synchronous wrapper

Best for:

- sending one natural-language instruction without manually constructing `CodeAgent` and `AgentRunRequest`

Minimal example:

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

Import path:

```python
from core.cc.api import build_code_with_agent
```

Role:

- a more structured wrapper for code build or repair tasks

Best for:

- explicitly providing a goal, context paths, constraints, and acceptance criteria

Minimal example:

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

## Selection guide

If you only need a command-line surface:

- use the CLI API

If you need a governed workspace:

- use `SgarRuntime`

If you need direct multi-mode agent execution:

- use `core.ccx.CodeAgent`

If you want simple wrapper functions:

- use `run_code_agent()` or `build_code_with_agent()`
