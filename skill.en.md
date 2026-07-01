---
name: sgar
description: Run a governed, state-machine-driven build or maintenance task where completion is decided by machine-checkable exit criteria, not the agent's say-so. Use when you need long-horizon, auditable, won't-lie-about-done execution; each stage advances only when its [check: <shell cmd>] gates pass under hermetic verification. Invoke via `python cli.py <command>`.
---

# SGAR skill

SGAR (State-Governed Agent Regime) governs an autonomous build or maintenance
loop so the agent **cannot self-certify a fake "done"**. State lives outside the
LLM context, and a stage only advances when its machine-checkable exit criteria
actually pass.

## When to use
- You need an unattended, long-horizon, auditable workflow with explicit acceptance criteria.
- You want "done" to be decided by `[check: <cmd>]` exit codes, not by the model's narration.
- You need staged progress, verifiable evidence, and rollback-friendly governance.

## Invocation
```bash
python cli.py <command> [options]
```
Install dependencies first if needed: `pip install -r requirements.txt`.

User-level LLM configuration can be written directly to `~/.sgar/setting.ini`:

```bash
python cli.py config where
python cli.py config list
python cli.py config set --client SimpleDeepSeekClient --api-key YOUR_KEY --model deepseek-v4-pro
```

If a client requires multiple credential keys, use the repeatable `--key KEY=VALUE`
argument.

## Two entry surfaces

### 1. Governance CLI
Use this when you want to manipulate `.sgar/` state, governance docs, and stage
transitions directly.

```bash
python cli.py [global options] <command> [command options]
```

### 2. Unified agent entrypoint
Use this when you want `sgar` as the unified product CLI for `plan`, `spec`,
`agent`, `doc`, `sgarx`, `goal`, and similar modes.

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

## Governance CLI: global options
These options appear before the subcommand and apply to all governance commands.

| Argument | Meaning |
|------|------|
| `--cwd <dir>` | Project root directory. Defaults to the current directory. `.sgar/` is resolved from here. |
| `--session <id>` | Use an isolated session under `.sgar/sessions/<id>/`. Useful for parallel governed tasks. |
| `--run-checks` | Actually execute `[check: <cmd>]` criteria during `verify` / `close-stage`. Without it, SGAR relies on recorded verdicts. |
| `--check-timeout <seconds>` | Per-check timeout when `--run-checks` is enabled. Default: `120.0`. |

## Governance CLI: commands

### `init`
Initialize the `.sgar/` workspace.

```bash
python cli.py init [--project <name>] [--force] [--blueprint-text <text>] [--roadmap-text <text>] [--stage-spec-text <text>] [--stage <stage-id>]
```

| Argument | Meaning |
|------|------|
| `--project <name>` | Explicit project name. |
| `--force` | Reinitialize an existing workspace and wipe current governance state. |
| `--blueprint-text <text>` | Write `blueprint.md` immediately after init. |
| `--roadmap-text <text>` | Write `roadmap.md` immediately after init. |
| `--stage-spec-text <text>` | Write a stage `spec.md` immediately after init. |
| `--stage <stage-id>` | Stage used with `--stage-spec-text`. Default: `stage-01`. |

### `status`
Show current governance status.

```bash
python cli.py status
```

### `set-blueprint`
Write the blueprint document.

```bash
python cli.py set-blueprint --text "<markdown>"
```

| Argument | Meaning |
|------|------|
| `--text <markdown>` | Full blueprint body. Required. |

### `set-roadmap`
Write the roadmap document.

```bash
python cli.py set-roadmap --text "<markdown>"
```

| Argument | Meaning |
|------|------|
| `--text <markdown>` | Full roadmap body. Required. |

### `set-stage-spec`
Write the stage specification document.

```bash
python cli.py set-stage-spec --stage <stage-id> --text "<markdown>"
```

| Argument | Meaning |
|------|------|
| `--stage <stage-id>` | Target stage ID. Required. |
| `--text <markdown>` | Full stage spec body. Required. |

### `draft-blueprint` / `draft-roadmap` / `draft-stage-spec`
Use an LLM to draft governance documents.

```bash
python cli.py draft-blueprint [--prompt "<extra prompt>"] [--llm-client <ClientName>]
python cli.py draft-roadmap [--prompt "<extra prompt>"] [--llm-client <ClientName>]
python cli.py draft-stage-spec --stage <stage-id> [--prompt "<extra prompt>"] [--llm-client <ClientName>]
```

| Argument | Meaning |
|------|------|
| `--stage <stage-id>` | Required only for `draft-stage-spec`. |
| `--prompt <text>` | Extra prompt passed to the LLM. Defaults to an empty string. |
| `--llm-client <ClientName>` | Client used for drafting. Default: `SimpleDeepSeekClient`. |

### `validate`
Validate governance documents against SGAR rules.

```bash
python cli.py validate blueprint [--accept]
python cli.py validate roadmap [--accept]
python cli.py validate stage --stage <stage-id> [--accept]
```

| Argument | Meaning |
|------|------|
| `blueprint \| roadmap \| stage` | Required positional target type. |
| `--stage <stage-id>` | Required when the target is `stage`. |
| `--accept` | When validating `blueprint` or `roadmap`, also accept the current version if validation passes. |

### `start-stage`
Move a stage into the active state.

```bash
python cli.py start-stage <stage-id>
```

### `verify`
Record verification evidence. This is the core command in the governed loop.

```bash
python cli.py verify --stage <stage-id> --criterion <criterion-id> (--pass | --fail) [--evidence <text>] [--notes <text>] [--artifact <path> ...]
python cli.py verify --stage <stage-id> --all-pass --evidence <text> [--notes <text>] [--artifact <path> ...]
```

| Argument | Meaning |
|------|------|
| `--stage <stage-id>` | Target stage. Required. |
| `--criterion <criterion-id>` | Target exit criterion. Mutually exclusive with `--all-pass`. |
| `--pass` | Record the criterion as passed. |
| `--fail` | Record the criterion as failed. |
| `--all-pass` | Parse all exit criteria from the stage spec and mark them all passed in one shot. Requires `--evidence`. |
| `--evidence <text>` | Verification evidence text. Optional for single-criterion verification; required with `--all-pass`. |
| `--notes <text>` | Extra notes attached to the verification record. |
| `--artifact <path>` | Additional artifact path to include in the verification trace. Repeatable. |

### `close-stage`
Close a stage that has already been verified.

```bash
python cli.py close-stage <stage-id>
```

### `mission create`
Create a filesystem-backed isolated mission.

```bash
python cli.py mission create --kind <kind> --id <mission-id> --input <path> ... --objective "<text>" --expected-output <path-or-desc> ... [--scope <path> ...]
```

| Argument | Meaning |
|------|------|
| `--kind <kind>` | Mission type. Required. |
| `--id <mission-id>` | Mission ID. Required. |
| `--input <path>` | Input file or directory. Repeatable and required. |
| `--objective <text>` | Mission objective. Required. |
| `--expected-output <path-or-desc>` | Expected outputs. Repeatable and required. |
| `--scope <path>` | Allowed scope paths. Repeatable. If omitted, the caller must enforce scope externally. |

### `mission status`
Show the status of one mission.

```bash
python cli.py mission status <mission-id>
```

### `mission complete`
Mark a mission complete and register its result artifact.

```bash
python cli.py mission complete <mission-id> --result <path>
```

| Argument | Meaning |
|------|------|
| `--result <path>` | Result file or directory. Required. |

### `mission list`
List all missions.

```bash
python cli.py mission list
```

### `doctor`
Detect missing files, invalid configuration, or inconsistent governance state.

```bash
python cli.py doctor
```

### `trace`
Show a summary of SGAR operation records.

```bash
python cli.py trace
```

## Unified agent entrypoint: `run` and shortcuts

### `run`
```bash
python cli.py run --mode <mode> [run options] "<instruction>"
```

| Argument | Meaning |
|------|------|
| `--mode <mode>` | Required. Available values include `plan`, `spec`, `agent`, `doc`, `ask`, `blueprint`, `sgar`, `sgarx`, `goal`, and `debug`. |
| `"<instruction>"` | Positional instruction text. |
| `--instruction "<text>"` | Use this when the instruction starts with `-` or is awkward as a positional arg. |
| `--cwd <dir>` | Working directory. Defaults to the current directory. |
| `--prompt-language <lang>` | Override prompt language for this run. |
| `--permission-mode <mode>` | Override permission mode for this run. |
| `--max-tool-rounds <n>` | Cap tool-call rounds. |
| `--docs-output-path <path>` | Output path for artifact-producing modes such as `doc` or `goal`. |
| `--metadata-json <json-object>` | Extra request metadata. Must decode to a JSON object. |
| `--json` | Print the full `AgentRunResult` JSON instead of a plain-text summary. |

### Shortcut commands
These are equivalent to `python cli.py run --mode <mode> ...`:

| Shortcut | Equivalent |
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

## User config command: `config`

```bash
python cli.py config where
python cli.py config list
python cli.py config set --client <ClientName> [--api-key <key>] [--model <model>] [--key KEY=VALUE ...]
```

| Subcommand / Argument | Meaning |
|------|------|
| `where` | Print the user config path, which is `~/.sgar/setting.ini`. |
| `list` | Show supported `ClientName` values, credential keys, model keys, and one-line examples. |
| `set --client <ClientName>` | Required. Select the client to configure. |
| `set --api-key <key>` | Only valid when the client has exactly one credential key. |
| `set --model <model>` | Write the preferred model when the client exposes a model key. |
| `set --key KEY=VALUE` | Explicitly write any config key. Repeatable. Preferred for multi-credential clients. |

Full option details: `python cli.py <command> --help`.

## Exit-criteria contract
Declare machine-checkable acceptance criteria in stage specs with
`[check: <shell cmd>]`. The **exit code is the authority** (`0` = pass). With
`--run-checks`, SGAR actually executes those checks and rejects any claimed pass
that contradicts them. Checks run under hermetic execution by default to reduce
the chance of cheating through user-site or cwd injection.

## Guidance for calling agents
- Always phrase the objective as something self-checkable, such as "make check X pass", not as a vague goal.
- Treat `python cli.py status` and `python cli.py trace` as the source of truth, not your own narration.
- If a stage check fails, report failure honestly. **Do not weaken or edit checks/tests just to force a pass.**
