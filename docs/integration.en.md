# sgar Integration Guide

This document is not about how to type each command. It is about where `sgar` fits in your system and how to wire it in cleanly.

From the product point of view, the goal is not merely to expose a coding-agent API. The goal is to let each project grow an embedded agent inside the deployment environment, close to real business data, so it can keep doing `auto research`, `self-improving`, and `self-operation` instead of being invoked only once during offline development.

If you only want quick hands-on usage, start with [usage.en.md](./usage.en.md). If you want the full interface inventory, read [api.en.md](./api.en.md).

## Build the right mental model first

`sgar` is not just one thing.

It is a unified product entrypoint for three different integration surfaces:

- a standalone CLI
- a state-governed runtime
- a shell around a multi-mode coding agent

So before integrating it, ask one question:

What exactly are you trying to embed?

- a command-line capability
- a governance capability
- a code-editing or auto-repair capability

Different answers lead to different integration choices.

If you already know you want a true project-embedded agent, make the target more concrete:

- let the agent see business context, not only repository files
- let the agent keep running in the deployment environment, not only as a one-shot task
- let research, improvement, and operation all leave governance state and audit traces

## Integration scenarios

## 1. As a standalone CLI tool

Best for:

- CI/CD pipelines
- internal ops scripts
- scheduled maintenance tasks
- local developer tooling

Typical form:

```bash
sgar init --project my-repo
sgar status
sgar run --mode sgar "repair flaky tests and summarize the result"
```

Advantages:

- lowest integration cost
- no extra Python wrapper needed
- great for validating product value early

Boundary:

- your outer system mostly interacts through commands, text output, and exit codes
- more advanced orchestration still has to live outside

## 2. As an internal state-governed runtime

Best for:

- systems that already have their own orchestrator
- flows that need explicit workspace, stages, verification, and mission tracking
- systems that want agent progression to become part of an engineering state machine

Typical form:

```python
from sgar import SgarRuntime

runtime = SgarRuntime("/path/to/repo")
runtime.init(project_name="demo-project")
runtime.set_stage_spec("stage-01", "...")
runtime.start_stage("stage-01")
```

Advantages:

- the outer system keeps full lifecycle control
- the `.sgar/` workspace becomes the governance record
- ideal when your system owns orchestration and `sgar` acts as the governance kernel

Boundary:

- you still decide when to invoke LLM-backed work, when to run modes, and when to advance stages

## 3. As a code-editing or auto-repair component

Best for:

- engineering platforms
- internal bots
- automated repair services
- products that need embedded code-editing capabilities

Typical form:

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

Advantages:

- the easiest way to embed automated code modification capability
- closer to a “send goal -> get result” programming model

Boundary:

- this layer does not replace `SgarRuntime` if you need explicit stages and governance records

## Recommended integration patterns

## Pattern A: outer system + `sgar` CLI

Recommended when:

- you want the fastest path into an existing system
- shell, a job runner, or CI is enough for orchestration

Suggested shape:

```text
your scheduler / CI / ops workflow
-> sgar CLI
-> .sgar workspace
-> trace / verification / artifacts
```

This is a strong MVP path.

## Pattern B: outer system + `SgarRuntime`

Recommended when:

- you already have your own orchestrator
- you want explicit stage, verification, and workspace control

Suggested shape:

```text
your orchestrator
-> SgarRuntime
-> .sgar workspace
-> verification / missions / trace
```

In this setup, `sgar` behaves more like a governance kernel than the final scheduler.

## Pattern C: outer system + `CodeAgent`

Recommended when:

- you mainly want the multi-mode coding-agent capability
- you want to call `plan/spec/agent/doc/sgar/sgarx/...` directly from code

Suggested shape:

```text
your product/service
-> core.ccx.CodeAgent
-> ccx mode execution
-> AgentRunResult / artifacts
```

## Which layer to use when

## Use the `sgar` CLI when

- you want the lowest integration cost
- shell, CI, or a task runner is enough
- you want to validate the workflow before embedding lower-level APIs

## Use `SgarRuntime` when

- you want explicit governed workspace management
- you need stages, verification, missions, and traces
- you want your system to hold hard state, not just one run result

## Use `core.ccx.CodeAgent` when

- you want direct multi-mode agent execution
- you want to choose among `plan/spec/agent/doc/sgar/sgarx/goal/debug` in code
- you do not want every capability to pass through the SGAR runtime command surface

## Use `run_code_agent()` or `build_code_with_agent()` when

- you want a fast integration path
- you prefer wrapper functions over constructing agent/request objects yourself

## About `sgarx`

`sgarx` works best as an extension mode, not as a separate product entrypoint.

Recommended form:

```bash
sgar run --mode sgarx "..."
```

or:

```bash
sgar sgarx "..."
```

not as a standalone top-level binary identity.

Why this is the better shape:

- `sgar` remains the single product entrypoint
- `sgarx` stays clearly modeled as a mode
- future modes do not fragment the product naming scheme

## Engineering recommendations

## 1. Configuration management

Recommended practice:

- use `sgar config set` on developer machines
- prefer environment variables in services and pipelines
- do not commit plaintext credentials into the repository

## 2. Workspace isolation

Recommended practice:

- keep one `.sgar/` workspace per repository
- use `--session <id>` when one repository needs multiple parallel work lines

## 3. `.gitignore`

Recommended practice:

- keep `.sgar/` and `.sgarx/` out of version control
- if a repository already has `.gitignore`, `sgar init` will add them automatically

## 4. Observability and troubleshooting

Make these part of your post-integration observability surface:

- `sgar status`
- `sgar doctor`
- `sgar trace`
- verification evidence
- mission state

That way, when a run fails, you do not only see “failed”. You can also inspect:

- the current state
- the stage where it failed
- the recent trace
- the verification evidence already recorded

## 5. Interface boundaries

A clean responsibility split looks like this:

- your system is responsible for:
  - when to start a task
  - when to stop a task
  - when to write results back to upstream systems

- `sgar` is responsible for:
  - workspace and hard state
  - stage, verification, mission, and trace management
  - a unified coding-agent entrypoint

- `ccx` is responsible for:
  - multi-mode long-range execution

## A practical adoption path

If you want the lowest-risk rollout, adopt in this order:

1. start with the `sgar` CLI in manual or semi-automated flows
2. upgrade governance to `SgarRuntime`
3. upgrade general code-agent execution to `CodeAgent` or the wrapper functions

Why this path works well:

- each step is independently testable
- you do not have to bind your system to the lowest-level API on day one

## Next steps

- For the product and code structure, read [architecture.en.md](./architecture.en.md)
- For operational usage, read [usage.en.md](./usage.en.md)
- For interface reference, read [api.en.md](./api.en.md)
