# sgar Usage Guide

This document focuses on two questions:

- how to get `sgar` running
- when to use the SGAR runtime workflow versus the unified mode workflow

If you want the full interface reference, continue with [api.en.md](./api.en.md). If you want to embed `sgar` into your own system, continue with [integration.en.md](./integration.en.md).

## Quick Start

### 1. Install

Requirements:

- Python `>= 3.12`

Install:

```bash
pip install sgar
```

### 2. Configure an LLM

Shortest path:

```bash
sgar config set --client SimpleDeepSeekClient --api-key YOUR_KEY --model deepseek-v4-pro
```

Helpful companion commands:

```bash
sgar config where
sgar config list
```

### 3. Initialize a workspace

```bash
sgar init --project my-repo
```

This creates a `.sgar/` workspace in the current repository.

If the repository root already contains a `.gitignore`, `sgar init` also adds:

```text
.sgar/
.sgarx/
```

### 4. Check status

```bash
sgar status
```

The most common day-to-day companion commands are:

```bash
sgar doctor
sgar trace
```

## Two main usage paths

`sgar` now supports two main usage paths.

### Path 1: SGAR runtime workflow

Use this path when:

- you want explicit stages and governance documents
- you want to track blueprint, roadmap, and stage specs
- you want persistent verification records, missions, and traces

The most common commands are:

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

### Path 2: unified mode workflow

Use this path when:

- you want to run a specific `ccx` mode directly
- you care more about one long-running execution than a full governance document flow
- you want `sgar` to be the product entrypoint for multiple modes

Unified entrypoint:

```bash
sgar run --mode sgar "repair flaky tests and close the stage"
sgar run --mode plan "design the migration plan"
sgar run --mode agent "fix the import error and summarize the change"
```

Shortcut form:

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

## Most common commands

### Help and configuration

```bash
sgar --help
sgar config --help
sgar config where
sgar config list
```

### Governance workflow

```bash
sgar init --project my-repo
sgar status
sgar doctor
sgar trace
```

### Mode workflow

```bash
sgar run --mode sgar "..."
sgar run --mode agent "..."
sgar plan "..."
sgar agent "..."
sgar sgarx "..."
```

### Common optional flags

The unified mode commands frequently use:

```bash
--cwd /path/to/repo
--prompt-language zh
--permission-mode default
--max-tool-rounds 12
--metadata-json '{"ccx_contract": {"kind": "demo"}}'
--json
```

## Workspace and files

Typical `.sgar/` layout:

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

If you use session isolation:

```bash
sgar --session demo-01 status
```

the state is stored under:

```text
.sgar/sessions/demo-01/
```

## Typical task flows

### Example 1: initialize a governed workspace

```bash
sgar config set --client SimpleDeepSeekClient --api-key YOUR_KEY --model deepseek-v4-pro
sgar init --project my-repo
sgar status
```

### Example 2: execute one governed coding task

```bash
sgar run --mode sgar "repair the flaky tests, record the evidence, and summarize the result"
```

This is a good fit when you already accept `sgar` as a governance-oriented mode and want it to work around state, verification, and traceability.

### Example 3: plan first, execute next

```bash
sgar plan "design a migration plan for the auth module"
sgar agent "implement the approved migration with minimal changes"
```

This path is closer to general-purpose `ccx` mode usage.

### Example 4: maintain stage documents and verification

```bash
sgar set-stage-spec --stage stage-01 --text "..."
sgar validate stage --stage stage-01
sgar start-stage stage-01
sgar verify --stage stage-01 --criterion c1 --pass --evidence "pytest tests/test_api.py -q"
sgar close-stage stage-01
```

## When to use which path

Prefer the SGAR runtime workflow when:

- you want explicit stage, verification, mission, and trace management
- you want the workspace itself to be the governance record
- you want the agent to progress through hard-state transitions

Prefer the unified mode workflow when:

- you only want to call a specific mode
- you need general-purpose capabilities like `plan/spec/agent/doc/ask`
- you want `sgar` to be the unified product entrypoint for `ccx`

## Common troubleshooting

### Configuration not found

Check:

```bash
sgar config where
sgar config list
```

and confirm that `~/.sgar/setting.ini` or the required environment variables contain the necessary keys.

### Workspace not initialized

If commands like `status`, `trace`, `doctor`, `validate`, or `verify` fail because the workspace is missing, run:

```bash
sgar init --project my-repo
```

### Where to look when a run fails

Start with:

```bash
sgar doctor
sgar trace
```

If a mode execution fails, inspect the full result with `--json`:

```bash
sgar run --mode agent --json "fix the failing import path"
```

## Next steps

- For the full CLI and Python reference, read [api.en.md](./api.en.md)
- For the system structure, read [architecture.en.md](./architecture.en.md)
- For engineering integration patterns, read [integration.en.md](./integration.en.md)
