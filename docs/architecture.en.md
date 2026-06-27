# sgar Architecture

`sgar` is not a single CLI script. It is a layered long-range coding-agent system that combines a product entrypoint, state governance, multi-mode execution, and a lower-level runtime engine. That is what makes automated repair, automated operations, and long-range code editing usable both as a standalone tool and as an embeddable subsystem.

## Architecture Overview

You can think of `sgar` as four layers:

1. Product entrypoint  
   This is the surface users interact with directly: `sgar` and `python -m sgar`. It routes commands either to the SGAR runtime or to `ccx` modes.

2. State governance layer  
   The SGAR runtime manages the `.sgar/` workspace, stages, verification records, missions, and traces, so an agent run is not just a one-shot call but a hard-state progression.

3. Mode execution layer  
   `ccx` provides multiple agent modes such as `plan`, `spec`, `agent`, `doc`, `sgar`, `sgarx`, `goal`, and `debug`. This is the core capability surface for long-running execution.

4. Runtime engine layer  
   `deepstack_v5` handles graph scheduling, node execution, events, persistence, and concurrency. It is the runtime substrate behind `ccx`.

## Key Directories

### 1. Product entrypoint

- `sgar/cli.py`  
  The top-level unified CLI. It handles:
  - `sgar config`
  - SGAR runtime command forwarding such as `init` and `status`
  - `sgar run --mode ...`
  - shortcut commands such as `sgar plan`, `sgar agent`, and `sgar sgarx`

- `sgar/__main__.py`  
  Enables `python -m sgar`

- `sgar/config_cli.py`  
  The user-level configuration entrypoint that writes `~/.sgar/setting.ini`

### 2. SGAR runtime

- `core/ccx/sgar/cli.py`  
  The SGAR runtime command surface for governance-oriented commands such as `init`, `status`, `validate`, `verify`, and `mission`

- `core/ccx/sgar/runtime.py`  
  The main implementation of `SgarRuntime`. The most important public methods include:
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
  Persists `.sgar/` workspace files and state

- `core/ccx/sgar/tracing.py`  
  Reads and writes SGAR trace data

### 3. ccx mode layer

- `core/ccx/api.py`  
  The unified API entrypoint for `ccx`, exposing `CodeAgent` and mode dispatch

- `core/ccx/__init__.py`  
  Re-exports:
  - `CodeAgent`
  - `AgentRunRequest`
  - `AgentRunResult`
  - `CodeBuildRequest`

- `core/ccx/modes/`  
  The implementation of multiple modes, including:
  - `plan`
  - `spec`
  - `agent`
  - `doc`
  - `ask`
  - `blueprint`
  - `sgarx`
  - `watch`

The current directly supported `ccx` modes are:

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

## The relationship between `sgar`, `sgarx`, and `ccx`

### `sgar`

`sgar` is both the product entrypoint name and the standard state-governed coding-agent workflow.

It emphasizes:

- an explicit workspace
- staged progression
- verification and audit
- traceable execution history

### `sgarx`

`sgarx` is an extension mode, not a separate product identity.

It is best used through the unified entrypoint:

```bash
sgar run --mode sgarx "..."
```

or:

```bash
sgar sgarx "..."
```

Its state space is distinct from `.sgar/` and usually lives under `.sgarx/`.

### `ccx`

`ccx` is the lower-level multi-mode agent/runtime capability layer. `sgar` productizes it, routes it, and adds state governance and a user-facing CLI on top.

In one sentence:

- `ccx` is the capability layer
- `sgar` is the product entrypoint and governance shell
- `sgarx` is an extension mode exposed by `sgar`

## Hard state and workspace

### `.sgar/`

After `sgar init`, a `.sgar/` workspace is created in the repository to store:

- `config.json`
- `state.json`
- `blueprint.md`
- `roadmap.md`
- `stages/<stage>/spec.md`
- `missions/`
- trace and verification artifacts

This means SGAR progression does not rely only on prompts or ephemeral memory. It relies on externalized, inspectable, auditable hard state.

### `.sgarx/`

`.sgarx/` is the separate state space for the extension mode. It should not be mixed with `.sgar/`, and it should not be treated as a disposable cache directory.

### Session isolation

If `--session <id>` is provided, SGAR isolates state under:

```text
.sgar/sessions/<id>/
```

This allows multiple isolated governance lines to coexist inside the same repository.

## Execution model

The execution model of `sgar` rests on two core ideas:

- Audit Engineering  
  The system does not treat “what was generated” as the end state. It treats verification, defect discovery, and the next corrective step as the core loop.

- State-Governed Agent Regime  
  The agent does not progress by prompt drift alone. It progresses through externalized state, stage transitions, actions, and deltas.

That is why `sgar` is optimized less for one-shot answer quality and more for:

- durable long-range progression
- verifiability
- recoverability
- traceability
- embeddability

## Typical data flows

### Path 1: governance commands

```text
sgar init/status/verify
-> sgar/cli.py
-> core/ccx/sgar/cli.py
-> SgarRuntime
-> .sgar/ workspace
-> state / trace / verification artifacts
```

### Path 2: unified mode commands

```text
sgar run --mode plan|agent|sgar|sgarx|...
-> sgar/cli.py
-> core.ccx.CodeAgent
-> core/ccx/api.py
-> deepstack_v5 engine
-> AgentRunResult / artifacts / trace
```

## Recommended reading order

If you are new to the project, read in this order:

1. [README.en.md](../README.en.md)
2. [usage.en.md](./usage.en.md)
3. [api.en.md](./api.en.md)
4. [integration.en.md](./integration.en.md)

If you want the underlying engineering ideas, then continue with:

- Audit Engineering
- State-Governed Agent Regime
