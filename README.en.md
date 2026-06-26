# sgar: State-Governed Agent Regime / 状态治理代理体制

[中文版本](./README.md)

`sgar` is an embedded coding agent for automated repair, automated operations,
and long-range code editing inside your own systems.

It is simultaneously:

- a skill that gives OpenClaw a stable long-range coding agent
- a standalone CLI for long-running code editing workflows
- an embeddable agent runtime that gives your system self-repair and
  self-maintenance capabilities

`sgar` is not just "ask an LLM to write code once". It combines code editing,
state governance, staged execution, audit, verification, and persistent traces
into a long-running agent model.

## Core Ideas

The design of `sgar` is grounded in two core documents:

- [Audit Engineering](https://github.com/wxy2ab/against-llm-mediocrity/blob/main/docs/audit-engineering.md)
- [State-Governed Agent Regime](https://github.com/wxy2ab/against-llm-mediocrity/blob/main/docs/state-governed-agent-regime.md)

- `Audit Engineering`: turn "can generate" into "can keep correcting". It exploits the generation-verification asymmetry of LLMs and the fact that, in coding, a defect diagnosis is often already the prescription. That is how long-running agents get sharper instead of drifting.
- `State-Governed Agent Regime`: move the agent from prompt drift to hard-state progression. Externalized state constrains the trajectory, while `action` and `delta` drive each improvement step, making long-range execution controllable, traceable, and iterative.

If you want to understand why `sgar` emphasizes auditability, explicit state,
stage transitions, and long-horizon execution, start with those two documents.

## Positioning

You can think of `sgar` as one package with three layers:

- `Embedded Coding Agent`: embed modern code editing capability into your own
  systems, products, platforms, or automation pipelines
- `OpenClaw Long-Range Skill`: give OpenClaw a stable long-range coding agent
- `Standalone CLI`: run governed planning, execution, verification, and
  convergence directly from the command line

This means `sgar` works both as a standalone tool and as an internal component
inside your own system.

## Advantages

The main advantages of `sgar` are:

1. A long-running agent  
   `sgar` is built for staged execution, persistent state, multi-step
   convergence, and long-horizon operation.

2. Modern code editing embedded into your system  
   You can integrate `sgar` into your own stack so your system gains low-cost
   self-update, self-repair, and self-maintenance capabilities.

3. Audit-based reliable iteration  
   `sgar` treats verification, evidence, traceability, and reviewability as
   first-class capabilities.

4. A hard state model  
   `sgar` constrains agent behavior through explicit state and stage
   transitions, reducing drift and loss of control in long-running workflows.

## Installation

Requirements:

- Python `>= 3.12`

Install:

```bash
pip install sgar
```

After installation:

```bash
sgar --help
python -m sgar --help
```

## Configuration

### Option 1: Use `sgar config`

The recommended way is to write user-level configuration through the CLI:

```bash
sgar config where
sgar config list
sgar config set --client SimpleDeepSeekClient --api-key YOUR_KEY --model deepseek-v4-pro
```

This writes to:

```text
~/.sgar/setting.ini
```

A typical config looks like:

```ini
[Default]
llm_api = SimpleDeepSeekClient
cc_default_llm_client = SimpleDeepSeekClient
deepseek_api_key = YOUR_KEY
deepseek_model = deepseek-v4-pro
```

`sgar config list` shows the supported `ClientName` values, credential keys,
model keys, and example commands.

If a client has exactly one credential key, `--api-key` is enough. If it needs
multiple credentials or connection parameters, use repeated `--key` flags:

```bash
sgar config set --client SparkClient \
  --key xunfei_spark_api_key=YOUR_KEY \
  --key xunfei_spark_secret_key=YOUR_SECRET \
  --model 4.0Ultra
```

### Option 2: Edit `setting.ini` manually

You can also create the config file yourself:

```ini
[Default]
llm_api = SimpleDeepSeekClient
cc_default_llm_client = SimpleDeepSeekClient
deepseek_api_key = YOUR_KEY
deepseek_model = deepseek-v4-pro
```

The current lookup precedence in code is:

1. environment variables
2. `setting.ini` in the current working directory
3. `~/.sgar/setting.ini`

So if an environment variable such as `DEEPSEEK_API_KEY` is present, it
overrides the file value.

## Usage

### Simplest Path

- `sgar`: the standard long-range stable coding-agent workflow for init, status, diagnostics, and trace
- `sgarx`: the extended mode for stronger stage-recovery workflows; its state lives under `.sgarx/` and it is typically used as an integrated mode rather than a standalone top-level CLI command

If you just want to get `sgar` running, the shortest path is:

```bash
sgar config set --client SimpleDeepSeekClient --api-key YOUR_KEY --model deepseek-v4-pro
sgar init --project my-repo
sgar status
```

The most common day-to-day commands are:

```bash
sgar status   # show current stage and project state
sgar doctor   # detect missing files or inconsistent state
sgar trace    # show recent operation trace
```

After `sgar init`, a `.sgar/` workspace is created in your repository to store
hard state and governance artifacts:

If a `.gitignore` already exists at the repository root, `sgar init` also adds:

```text
.sgar/
.sgarx/
```

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

If you pass `--session <id>`, the state is isolated under:

```text
.sgar/sessions/<id>/
```

### Advanced Usage

Help:

```bash
sgar --help
sgar config --help
python -m sgar --help
```

Initialization and status:

```bash
sgar init --project my-repo
sgar status
sgar trace
sgar doctor
```

Write or draft governance documents:

```bash
sgar set-blueprint --text "..."
sgar set-roadmap --text "..."
sgar set-stage-spec --stage stage-01 --text "..."

sgar draft-blueprint --llm-client SimpleDeepSeekClient --prompt "Draft a blueprint for this repository"
sgar draft-roadmap --llm-client SimpleDeepSeekClient --prompt "Break the work into stages"
sgar draft-stage-spec --stage stage-01 --llm-client SimpleDeepSeekClient --prompt "Detail stage-01"
```

Validation and stage transitions:

```bash
sgar validate blueprint --accept
sgar validate roadmap --accept
sgar validate stage --stage stage-01
sgar start-stage stage-01
sgar verify --stage stage-01 --criterion c1 --pass --evidence "pytest tests/test_api.py -q"
sgar close-stage stage-01
```

Enable machine-checkable exit criteria:

```bash
sgar --run-checks --check-timeout 120 verify --stage stage-01 --all-pass --evidence "all checks passed"
```

Isolated missions:

```bash
sgar mission create \
  --kind patch \
  --id fix-login \
  --input src/auth.py \
  --objective "Fix the login timeout bug" \
  --expected-output patch.diff

sgar mission status fix-login
sgar mission list
```

## Embedding In Code

`sgar` supports two integration styles:

- SGAR state runtime: embed stage governance, validation, and trace management
- Embedded coding agent API: embed code editing and auto-repair into your own
  application

Note the current import surface:

- the `sgar` CLI and `python -m sgar` are the standalone command-line entrypoints
- `from sgar import SgarRuntime` is the SGAR state-governed runtime surface
- `from core.cc.api import ...` is the embedded coding-agent API surface

### Option 1: Embed the SGAR state runtime

```python
from sgar import SgarRuntime

runtime = SgarRuntime("/path/to/repo")
runtime.init(project_name="demo-project")

runtime.set_blueprint(
    """
    # Problem
    Need a governed repair workflow.
    """
)
runtime.set_roadmap(
    """
    - stage-01: stabilize tests
    """
)
runtime.set_stage_spec(
    "stage-01",
    """
    # Objective
    Make the failing tests pass.
    """,
)

print(runtime.status())
```

This is a good fit when your outer system owns orchestration and uses `sgar` as
its internal state machine and governance kernel.

### Option 2: Embed the coding agent

If you want to plug automated repair or code editing directly into your system,
use `core.cc.api`:

```python
from core.cc.api import build_code_with_agent

result = build_code_with_agent(
    goal="Fix the failing tests in the repository and update the docs if needed",
    cwd="/path/to/repo",
    context_paths=[
        "README.md",
        "src/app.py",
        "tests/test_app.py",
    ],
    constraints=[
        "Preserve the public API",
        "Prefer the smallest necessary change",
    ],
    acceptance_criteria=[
        "pytest tests/test_app.py -q passes",
        "README is updated if behavior changes",
    ],
    prompt_language="en",
)

print(result.final_text)
print(result.tool_call_count)
print(result.failed, result.error_message)
```

If you just want to send a single instruction, use the lighter sync wrapper:

```python
from core.cc.api import run_code_agent

result = run_code_agent(
    "Inspect the repository for regressions, fix them, run focused tests, and summarize the outcome",
    cwd="/path/to/repo",
    prompt_language="en",
)

print(result.final_text)
```

## Best-Fit Scenarios

`sgar` is a strong fit when you want to:

- integrate a coding agent into an engineering platform, internal tool, bot, or
  ops workflow
- run a long-range code editing workflow with auditability inside a repository
- add state governance, staged progress, evidence, and traceability to
  automated repair
- build systems that can keep maintaining their own code instead of only
  generating code once

If you only need a short-lived Q&A assistant, `sgar` is not the smallest tool.
If you need a long-running, governed, embeddable, auditable code agent, that is
exactly what `sgar` is designed for.
