# `core.cc` Usage

## Overview

`core.cc` is a Python code-editing agent runtime.

It can be used in 3 ways:

1. Import the public agent API from other Python projects.
2. Start a one-off code task through `task.py`.
3. Use the lower-level `build_default_query_engine(...)` interface when you need direct control over sessions and events.

The recommended boundary is:

- Public default surface: `CodeAgent`, `AgentRunRequest`, `CodeBuildRequest`, `run_code_agent(...)`, `build_code_with_agent(...)`
- Advanced but lower-level surface: `build_default_query_engine(...)`, `QueryEngine`, `QuerySession`
- Internal composition details: `EngineFactory`, `conversation/turn_pipeline.py`, concrete prompt/context assembly internals

All built-in prompts support `zh` and `en`. Prompt language is controlled by `CCConfig.prompt_language`, and defaults to `zh`.

All LLM access is routed through `LLMFactory().get_instance(...)` via `DefaultLLMClientProvider`.

## Quick Start

### 1. Integrate from another project

`CodeAgent` is now async-first. `run_sync(...)` and `build_code_sync(...)` remain available as compatibility wrappers.

```python
import asyncio

from core.cc import CodeAgent, AgentRunRequest, CCConfig

async def main():
    config = CCConfig(
        prompt_language="zh",
        permission_mode="default",
        default_llm_client="SimpleDeepSeekClientReasoning",
    )
    agent = CodeAgent(config=config)
    result = await agent.run(
        AgentRunRequest(
            instruction="检查当前仓库中的 Python 构建错误并直接修复。",
            cwd="D:/work/my_project",
        )
    )

    print(result.final_text)
    print(result.session_id)

asyncio.run(main())
```

### 2. Use the build-oriented API

```python
import asyncio

from core.cc import CodeAgent, CodeBuildRequest, CCConfig

async def main():
    agent = CodeAgent(config=CCConfig(prompt_language="en"))
    result = await agent.build_code(
        CodeBuildRequest(
            goal="Implement a minimal FastAPI CRUD service for todos.",
            cwd="D:/work/my_project",
            context_paths=["D:/work/my_project/app", "D:/work/my_project/tests"],
            constraints=[
                "Reuse the existing package structure.",
                "Do not introduce new external dependencies unless necessary.",
            ],
            acceptance_criteria=[
                "API routes compile successfully.",
                "Tests for create/list flows are present.",
            ],
        )
    )

    print(result.final_text)

asyncio.run(main())
```

### 3. Convenience helpers

```python
from core.cc import build_code_with_agent, run_code_agent

result = run_code_agent(
    "Read the repository and fix failing imports.",
    cwd="D:/work/my_project",
)

build_result = build_code_with_agent(
    "Add a CLI entrypoint for the current package.",
    cwd="D:/work/my_project",
    constraints=["Keep the existing logging style."],
)
```

## Public API

### `CodeAgent`

Primary integration class.

- `await CodeAgent.run(request)`
- `await CodeAgent.build_code(request)`
- `async for event in CodeAgent.stream(request)`
- `async for event in CodeAgent.stream_build_code(request)`
- `CodeAgent.run_sync(request)`
- `CodeAgent.build_code_sync(request)`

### `AgentRunRequest`

General-purpose request for direct agent execution.

- `instruction`: user task or coding goal
- `cwd`: target repository root
- `config`: optional `CCConfig`
- `session`: optional existing `QuerySession`
- `prompt_language`: override `zh` / `en`
- `permission_mode`: override permission mode (`default` / `accept_edits` / `bypass`)
- `agent_mode`: override workflow mode (`plan` / `spec` / `agent` / `ask` / `doc` / `""`)
  - `agent`: dynamic multi-agent mode; the lead agent plans collaboration first and must delegate to at least one child agent
  - `ask`: repository Q&A mode; complex structure questions may inject a lightweight repository outline first
  - `doc`: analysis + draft mode; prefers producing Markdown-ready documentation grounded in repository facts
- `metadata`: extra state injected into the session
- `system_prompt_key`: optional prompt asset key override
- `system_prompt_context`: extra prompt context merged into runtime context
- `event_sink`: optional callback / async callback for each `SessionEvent`

### `CodeBuildRequest`

Higher-level request for automatic code construction.

- `goal`: build objective
- `cwd`: target repository root
- `context_paths`: files or directories that should receive priority
- `constraints`: hard engineering constraints
- `acceptance_criteria`: desired completion signals
- `config`, `session`, `prompt_language`, `permission_mode`, `agent_mode`, `metadata`
- `build_code(...)` remains build-oriented; `ask` / `doc` are primarily intended for `run(...)`, while `agent` now also applies on the build path

### `AgentRunResult`

- `final_text`: final assistant summary
- `session_id`
- `turn_id`
- `cwd`
- `tool_call_count`
- `session_snapshot`
- `events`
- `messages`
- `failed`
- `error_code`
- `error_message`

## Event Contract

For host/runtime integrations, prefer the implemented event contract described in [15_runtime_event_contract.md](file:///d:/documents/projects/llm_dealer/core/cc/docs/15_runtime_event_contract.md).

The most important stable signals today are:

- `turn_failed` for turn-level runtime exceptions
- assistant message `metadata.exit_reason` for structured stop reasons
- agent/task `waiting_reason` for finer-grained wait states
- swarm `assignment_completed` as the completion signal for worker assignments

For offline troubleshooting, you can also read the unified audit sinks directly:

```python
from core.cc import query_runtime_audit, summarize_runtime_audit

events = query_runtime_audit(
    "D:/work/my_project/.cc/runtime",
    session_id="sess_123",
    event_types=["turn_failed", "tool_failed"],
)
summary = summarize_runtime_audit("D:/work/my_project/.cc/runtime", session_id="sess_123")

print(len(events.all_events))
print(summary.to_dict())
```

Supported audit filters:

- `session_id`
- `turn_id`
- `task_id`
- `tool_name`
- `event_types`
- `error_code`
- `limit`

Notes:

- `query_runtime_audit(...)` returns a filtered snapshot across `session_events.jsonl`, `task_events.jsonl`, and `tool_events.jsonl`
- `summarize_runtime_audit(...)` applies the same filters before aggregation
- `limit` means the global latest `N` matching events, not `N` events per audit file

Command-line example:

```bash
python -m core.cc.examples.audit_summary --runtime-root .cc/runtime --session-id sess_123 --event-type turn_failed --event-type tool_failed --limit 20
```

Human-friendly failure view:

```bash
python -m core.cc.examples.audit_summary --runtime-root .cc/runtime --session-id sess_123 --failures-only --pretty
```

Human-friendly event detail view:

```bash
python -m core.cc.examples.audit_summary --runtime-root .cc/runtime --session-id sess_123 --show-events
```

Machine-readable JSON view:

```bash
python -m core.cc.examples.audit_summary --runtime-root .cc/runtime --session-id sess_123 --json --show-events
```

## Public vs Internal

Use these boundaries when integrating:

- Prefer `CodeAgent` for application integration, service orchestration, and CLI wrappers.
- Use `build_default_query_engine(...)` only when you explicitly need direct event/session control.
- Avoid coupling host code to `EngineFactory` or `conversation/*` assembly details unless you are extending `core.cc` itself.
- Treat sync wrappers as compatibility APIs; async methods are the default integration path.

## `task.py` Entry

The task entrypoint lives at [cc.py](D:/work/llmdealer/task/deep/cc.py) and follows the repository `runner(...)` convention.

### Direct run mode

```bash
python task.py deep.cc prompt="修复当前仓库中的单元测试失败" cwd="D:/work/my_project"
```

### Build mode

```bash
python task.py deep.cc mode=build goal="补齐一个最小可运行的 REST API" cwd="D:/work/my_project"
```

### Plan mode (先分析再落代码)

```bash
python task.py deep.cc agent_mode=plan prompt="重构数据库层" cwd="D:/work/my_project"
```

### Spec mode (先写 spec/tasks/checklist 再落代码)

```bash
python task.py deep.cc agent_mode=spec prompt_file="specs/feature.md" cwd="D:/work/my_project"
```

### Agent mode (动态多 Agent 协作)

```bash
python task.py deep.cc agent_mode=agent prompt="请通过多 agent 协作完成这次复杂重构" cwd="D:/work/my_project"
```

### Ask mode (仓库问答 / 结构分析)

```bash
python task.py deep.cc agent_mode=ask prompt="这个仓库的模块结构和关键入口是什么？" cwd="D:/work/my_project"
```

### Doc mode (分析后产出文档草稿)

```bash
python task.py deep.cc agent_mode=doc prompt="为当前仓库生成一份架构说明文档" cwd="D:/work/my_project"
```

### Useful parameters

- `prompt` or `goal`: task content, at least one is required
- `cwd`: target workspace
- `mode`: `run` or `build`
- `prompt_language`: `zh` or `en`
- `permission_mode`: file/command access permissions (`default` / `accept_edits` / `bypass`)
- `agent_mode`: workflow mode (`plan` / `spec` / `agent` / `ask` / `doc`)
- `llm`: override `default_llm_client`
- `context_paths`: comma-separated priority paths
- `constraints`: comma-separated constraints
- `acceptance`: comma-separated acceptance criteria
- `output_json`: save structured result JSON
- `print_events`: print event stream

## Event Model

The public API returns raw `SessionEvent` objects from the internal query loop.

Common event types:

- `message_created`
- `assistant_completed`
- `assistant_tool_use`
- `tool_result`
- `assistant_followup_completed`
- `compact_applied`

This makes it possible for host applications to:

- stream progress to a UI
- capture tool usage
- persist transcripts
- build custom observability or audit layers

## Safety and Runtime Limits

### Tool round limit

The query loop no longer enforces a fixed default cap on consecutive tool-call rounds. A turn continues until the model produces a final text response or another runtime error interrupts the loop.

When you need an explicit guardrail, set `CCConfig.max_tool_rounds` or pass `max_tool_rounds` through `AgentRunRequest`, `CodeBuildRequest`, or `task.py deep.cc`.

### Path permission checks

Command-cwd containment checks use `Path.is_relative_to` (via `path_matches_any`) instead of string prefix matching, preventing path-traversal false positives such as `/project123` matching `/project1`.

### Permission state preservation

`permission_mode` controls file/command access permissions (`default` / `accept_edits` / `bypass`) and is separate from `agent_mode` which controls the workflow (`plan` / `spec` / `agent` / `ask` / `doc`).

For the newer read-heavy modes:

- `agent` focuses on dynamic multi-agent collaboration and requires at least one child-agent delegation
- `ask` focuses on repository Q&A and explanation
- `doc` focuses on analysis plus document-ready drafting
- `ask` / `doc` may inject a lightweight repository outline when the request depends on structure, module layout, or file discovery

Entering and exiting plan/spec mode or worktree sessions correctly preserves and restores:

- `denied_paths`
- the original permission mode
- the original `allowed_paths` (worktree exit no longer drops pre-existing paths)

The `permissions.mode` used by the classifier is a transient value derived from `app_state` (plan_mode/spec_mode flags), not from `session.permission_mode`.

### Prompt language fallback

`PromptAsset.resolve` now falls back symmetrically: `en -> zh` and `zh -> en`. An error is raised only if both language variants are empty.

### Session message deserialization

`SessionMessage.from_dict` now uses field-name whitelisting, so persisted messages with extra or removed fields do not crash on load.

## Session and Prompt Notes

- `CCConfig.prompt_language` defaults to `zh`.
- Built-in prompt assets are resolved from the package prompt directory automatically, so external projects do not need to run from the repository root.
- Relative `session_root`, `runtime_root`, and `prompt_root` are all resolved against the task `cwd`.

## Recommended Integration Pattern

For other projects, prefer this shape:

1. Build one shared `CCConfig`.
2. Construct one `CodeAgent`.
3. Prefer `await run(...)` / `await build_code(...)` for host integration.
4. Persist `session_id`, `final_text`, and `events` in your own orchestration layer.

If you need lower-level control, start with [runtime.py](file:///d:/documents/projects/llm_dealer/core/cc/runtime.py) and only drop to [query_engine.py](file:///d:/documents/projects/llm_dealer/core/cc/conversation/query_engine.py) when you are intentionally working with advanced engine internals.
