"""Structured execution flow: analyze -> plan tasks -> execute each -> summarize.

Replaces the purely prompt-driven "analyze -> create todo -> execute -> summarize"
pattern with code-controlled phases, each with independent tool-round limits and
clear boundaries.

Phase 3 (task execution) supports per-task ``depends_on`` and runs tasks in
topologically-ordered *waves*: every task whose dependencies are already
satisfied runs concurrently via ``asyncio.gather`` (bounded by ``parallelism``).
A task list with no edges therefore runs fully in parallel, while a chain
falls back to sequential execution. This is a strict generalisation of the
old "always serial" behaviour: a task list without ``depends_on`` fields and
``parallelism=1`` reproduces the legacy order.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from .config import CCConfig, load_cc_config
from .llm import DefaultLLMClientProvider, LLMClientProvider

logger = logging.getLogger(__name__)

ANALYSIS_MAX_TOOL_ROUNDS = 40
TASK_MAX_TOOL_ROUNDS = 60
DEFAULT_PARALLELISM = 4

_TASK_PLAN_SYSTEM_PROMPT = """\
You are a task planner. Based on the analysis report and the user's original instruction,
create a structured list of actionable tasks.

Output a JSON array of task objects, each with:
- "id": a short identifier (e.g., "task_1")
- "description": clear description of what to do (must NOT be empty)
- "type": one of "code_change", "investigation", "documentation", "test"
- "depends_on": (optional) array of task ids this task waits for. Default []

Dependency rules — they affect parallelism:
- Default to PARALLEL: leave depends_on empty whenever a task can run
  independently of every other task. The runner will execute independent
  tasks concurrently.
- Only add an id to depends_on when the later task truly needs an earlier
  task's outputs (e.g. a test task that needs the code-change to exist).
- Mixing parallel and chained tasks in the same plan is encouraged.

Output ONLY the JSON array, no other text. Example:
[
  {"id": "task_1", "description": "Fix the null check in auth.py line 42", "type": "code_change"},
  {"id": "task_2", "description": "Refactor the session helper", "type": "code_change"},
  {"id": "task_3", "description": "Add unit test for the login function",
   "type": "test", "depends_on": ["task_1"]}
]
"""

_ANALYSIS_INSTRUCTION_TEMPLATE = """\
## Analysis Phase

You are in the **analysis-only** phase. Your goal is to thoroughly understand the problem
before any implementation begins.

**Original instruction**: {instruction}

**Rules for this phase**:
1. Use ONLY read-only tools (file_read, grep, glob, list_directory)
2. Do NOT make any file changes
3. Analyze the codebase to understand the relevant code, dependencies, and potential issues
4. At the end, provide a detailed analysis report summarizing:
   - What files/code are relevant
   - What the current behavior is
   - What needs to change and why
   - Any risks or dependencies to consider
"""

_SUMMARY_SYSTEM_PROMPT = """\
You are a task summarizer. Based on the original instruction and all task execution results,
write a clear, concise summary for the user explaining:
1. What was accomplished
2. What files were changed (if any)
3. Any issues encountered
4. Any remaining work or recommendations

Be specific and actionable. Reference file paths and line numbers where relevant.
"""


@dataclass(slots=True)
class TaskDefinition:
    id: str
    description: str
    task_type: str = "code_change"
    depends_on: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PhaseResult:
    phase: str
    success: bool
    output: str
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class StructuredFlowRunner:
    """Orchestrates the structured analysis -> plan -> execute -> summarize flow.

    Phase 3 runs tasks in DAG order with bounded parallelism. Set
    ``parallelism=1`` to reproduce the legacy strictly-serial behaviour
    (still in DAG order — but only one task at a time).
    """

    def __init__(
        self,
        *,
        config: CCConfig | None = None,
        llm_client_provider: LLMClientProvider | None = None,
        parallelism: int | None = None,
    ) -> None:
        self.config = config or load_cc_config()
        self.llm_client_provider = llm_client_provider or DefaultLLMClientProvider()
        self.parallelism = max(
            1,
            int(parallelism if parallelism is not None
                else getattr(self.config, "structured_flow_parallelism",
                             DEFAULT_PARALLELISM)),
        )

    async def run(
        self,
        instruction: str,
        *,
        cwd: str = ".",
        prompt_language: str | None = None,
        permission_mode: str | None = None,
        event_sink: Any | None = None,
    ) -> PhaseResult:
        from .api import AgentRunRequest, AgentRunResult, CodeAgent

        agent = CodeAgent(config=self.config, llm_client_provider=self.llm_client_provider)
        resolved_language = prompt_language or self.config.prompt_language

        # --- Phase 1: Analysis ---
        logger.info("Structured flow: starting analysis phase")
        analysis_instruction = _ANALYSIS_INSTRUCTION_TEMPLATE.format(instruction=instruction)
        analysis_result = await agent.run(AgentRunRequest(
            instruction=analysis_instruction,
            cwd=cwd,
            config=self.config,
            max_tool_rounds=ANALYSIS_MAX_TOOL_ROUNDS,
            prompt_language=resolved_language,
            permission_mode=permission_mode,
            agent_mode="",
            event_sink=event_sink,
        ))
        if analysis_result.failed:
            return PhaseResult(
                phase="analysis",
                success=False,
                output=analysis_result.final_text,
                error=analysis_result.error_message,
            )
        analysis_text = analysis_result.final_text
        logger.info("Structured flow: analysis complete (%d chars)", len(analysis_text))

        # --- Phase 2: Task Planning ---
        logger.info("Structured flow: starting task planning phase")
        tasks, parse_meta = await self._plan_tasks(
            instruction, analysis_text, resolved_language,
        )
        if not tasks:
            return PhaseResult(
                phase="planning",
                success=True,
                output=analysis_text,
                metadata={
                    "note": "No tasks generated, returning analysis only",
                    **parse_meta,
                },
            )
        logger.info(
            "Structured flow: planned %d tasks (dropped %d invalid; %d edges)",
            len(tasks), parse_meta.get("dropped", 0),
            sum(len(t.depends_on) for t in tasks),
        )
        for t in tasks:
            logger.debug(
                "Structured flow: task %s deps=%s type=%s",
                t.id, list(t.depends_on), t.task_type,
            )

        # --- Phase 3: Task Execution (DAG-ordered, parallel) ---
        all_task_outputs, any_failed, exec_meta = await self._execute_tasks(
            tasks=tasks,
            analysis_text=analysis_text,
            instruction=instruction,
            agent=agent,
            cwd=cwd,
            resolved_language=resolved_language,
            permission_mode=permission_mode,
            event_sink=event_sink,
        )

        # --- Phase 4: Summary ---
        logger.info("Structured flow: starting summary phase")
        summary = await self._generate_summary(
            instruction, all_task_outputs, resolved_language,
        )

        return PhaseResult(
            phase="completed",
            success=not any_failed,
            output=summary,
            metadata={
                "task_count": len(tasks),
                "analysis_length": len(analysis_text),
                **parse_meta,
                **exec_meta,
            },
        )

    async def _plan_tasks(
        self,
        instruction: str,
        analysis: str,
        prompt_language: str,
    ) -> tuple[list[TaskDefinition], dict[str, Any]]:
        llm_client = self.llm_client_provider.get_client(
            config=self.config, purpose="structured_flow_planning",
        )
        messages = [
            {"role": "system", "content": _TASK_PLAN_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Original instruction: {instruction}\n\n"
                    f"Analysis report:\n{analysis[:6000]}\n\n"
                    f"Create the task list as a JSON array."
                ),
            },
        ]
        if hasattr(llm_client, "one_chat"):
            raw = llm_client.one_chat(messages)
            if inspect.isawaitable(raw):
                raw = await raw
        else:
            return [], {"dropped": 0, "raw_length": 0}

        return self._parse_task_list(raw)

    async def _execute_tasks(
        self,
        *,
        tasks: list[TaskDefinition],
        analysis_text: str,
        instruction: str,
        agent: Any,
        cwd: str,
        resolved_language: str,
        permission_mode: str | None,
        event_sink: Any | None,
    ) -> tuple[list[str], bool, dict[str, Any]]:
        """Run tasks in DAG order with bounded parallelism.

        Tasks whose dependencies have all completed (succeeded or failed)
        become eligible. Eligible tasks dispatch under a semaphore capped
        at ``self.parallelism``. A task whose dependencies failed is
        marked SKIPPED with a clear status — we do *not* abort the whole
        flow, mirroring the legacy behaviour where the loop continued on
        failure.
        """
        from .api import AgentRunRequest

        by_id: dict[str, TaskDefinition] = {t.id: t for t in tasks}
        # Validate edges: drop edges to unknown ids (avoid deadlock).
        edges_to_unknown = 0
        for t in tasks:
            cleaned = []
            for dep in t.depends_on:
                if dep == t.id:
                    edges_to_unknown += 1
                    continue
                if dep not in by_id:
                    edges_to_unknown += 1
                    continue
                cleaned.append(dep)
            t.depends_on = cleaned
        if edges_to_unknown:
            logger.warning(
                "Structured flow: dropped %d invalid depends_on edges",
                edges_to_unknown,
            )

        # Detect cycles via Kahn-style topo check up front; if a cycle is
        # found we break it by clearing the remaining edges (so the flow
        # still runs rather than deadlocking).
        in_degree = {t.id: len(t.depends_on) for t in tasks}
        ready: list[str] = [tid for tid, d in in_degree.items() if d == 0]
        topo_seen = 0
        # Reverse adjacency for the topo walk.
        reverse: dict[str, list[str]] = {t.id: [] for t in tasks}
        for t in tasks:
            for dep in t.depends_on:
                reverse[dep].append(t.id)
        rem_in = dict(in_degree)
        queue = list(ready)
        while queue:
            tid = queue.pop()
            topo_seen += 1
            for child in reverse[tid]:
                rem_in[child] -= 1
                if rem_in[child] == 0:
                    queue.append(child)
        cycle_detected = topo_seen != len(tasks)
        if cycle_detected:
            logger.warning(
                "Structured flow: dependency cycle detected; breaking edges "
                "for the %d unreachable tasks",
                len(tasks) - topo_seen,
            )
            for t in tasks:
                if rem_in.get(t.id, 0) > 0:
                    t.depends_on = []

        # Build the actual execution.
        completed: dict[str, str] = {}     # task_id -> "OK" / "FAILED" / "SKIPPED"
        results: dict[str, str] = {}       # task_id -> rendered block
        any_failed = False
        max_concurrent = 0
        in_flight = 0
        in_flight_lock = asyncio.Lock()
        sem = asyncio.Semaphore(self.parallelism)

        analysis_blurb = analysis_text[:3000]

        async def run_one(task: TaskDefinition) -> None:
            nonlocal any_failed, in_flight, max_concurrent

            # If any dep failed, skip this task without wasting the LLM call.
            for dep in task.depends_on:
                if completed.get(dep) in {"FAILED", "SKIPPED"}:
                    completed[task.id] = "SKIPPED"
                    results[task.id] = (
                        f"### Task {task.id}: {task.description}\n"
                        f"Status: SKIPPED\n"
                        f"Result: skipped because dependency {dep!r} did not "
                        f"succeed\n"
                    )
                    logger.info(
                        "Structured flow: task %s SKIPPED (dep %s failed)",
                        task.id, dep,
                    )
                    return

            async with sem:
                async with in_flight_lock:
                    in_flight += 1
                    if in_flight > max_concurrent:
                        max_concurrent = in_flight
                try:
                    logger.info(
                        "Structured flow: executing task %s — %s",
                        task.id, task.description,
                    )
                    task_instruction = (
                        # Lineage / context the legacy path lacked.
                        f"## Original instruction\n{instruction}\n\n"
                        f"## Task: {task.description}\n\n"
                        f"## Context from analysis\n{analysis_blurb}\n\n"
                        + (f"## Depends on\n{', '.join(task.depends_on)}\n\n"
                           if task.depends_on else "")
                        + "Execute this specific task. Focus only on this "
                        + "task; the listed dependencies are already done."
                    )
                    task_result = await agent.run(AgentRunRequest(
                        instruction=task_instruction,
                        cwd=cwd,
                        config=self.config,
                        max_tool_rounds=TASK_MAX_TOOL_ROUNDS,
                        prompt_language=resolved_language,
                        permission_mode=permission_mode,
                        agent_mode="",
                        event_sink=event_sink,
                    ))
                    if task_result.failed:
                        completed[task.id] = "FAILED"
                        any_failed = True
                        status = "FAILED"
                    else:
                        completed[task.id] = "OK"
                        status = "OK"
                    results[task.id] = (
                        f"### Task {task.id}: {task.description}\n"
                        f"Status: {status}\n"
                        f"Result: {task_result.final_text}\n"
                    )
                finally:
                    async with in_flight_lock:
                        in_flight -= 1

        # Wave-based scheduling. Each wave gathers all tasks whose deps
        # are already in `completed`. Independent tasks within a wave run
        # under the semaphore.
        remaining = {t.id for t in tasks}
        waves = 0
        while remaining:
            wave = [
                by_id[tid] for tid in list(remaining)
                if all(dep in completed for dep in by_id[tid].depends_on)
            ]
            if not wave:
                # Defensive: shouldn't happen after cycle-breaking above.
                logger.error(
                    "Structured flow: no eligible tasks but %d remaining "
                    "— aborting phase 3", len(remaining),
                )
                for tid in remaining:
                    completed[tid] = "SKIPPED"
                    results[tid] = (
                        f"### Task {tid}: {by_id[tid].description}\n"
                        f"Status: SKIPPED\n"
                        f"Result: scheduler stalled\n"
                    )
                break
            waves += 1
            logger.info(
                "Structured flow: wave %d dispatching %d task(s)",
                waves, len(wave),
            )
            await asyncio.gather(*(run_one(t) for t in wave))
            for t in wave:
                remaining.discard(t.id)

        # Preserve the order of the planner's output for the summary.
        ordered_outputs = [
            results[t.id] for t in tasks if t.id in results
        ]
        meta = {
            "exec_waves": waves,
            "exec_max_concurrent": max_concurrent,
            "exec_succeeded": sum(1 for s in completed.values() if s == "OK"),
            "exec_failed": sum(1 for s in completed.values() if s == "FAILED"),
            "exec_skipped": sum(1 for s in completed.values() if s == "SKIPPED"),
            "edges_to_unknown_dropped": edges_to_unknown,
            "cycle_detected": cycle_detected,
        }
        return ordered_outputs, any_failed, meta

    async def _generate_summary(
        self,
        instruction: str,
        task_outputs: list[str],
        prompt_language: str,
    ) -> str:
        llm_client = self.llm_client_provider.get_client(
            config=self.config, purpose="structured_flow_summary",
        )
        combined_results = "\n\n".join(task_outputs)
        messages = [
            {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Original instruction: {instruction}\n\n"
                    f"Task results:\n{combined_results[:8000]}\n\n"
                    f"Write a summary."
                ),
            },
        ]
        if hasattr(llm_client, "one_chat"):
            raw = llm_client.one_chat(messages)
            if inspect.isawaitable(raw):
                raw = await raw
            if isinstance(raw, dict):
                return str(raw.get("content", raw))
            return str(raw)
        return combined_results

    @staticmethod
    def _parse_task_list(raw: Any) -> tuple[list[TaskDefinition], dict[str, Any]]:
        """Parse the planner's JSON array into TaskDefinitions.

        Returns ``(tasks, meta)`` where ``meta`` carries diagnostic
        counters: ``dropped`` (items skipped because they were not a
        dict or had no description) and ``raw_length`` (length of the
        raw text for trace forensics).
        """
        text = str(raw.get("content", raw)) if isinstance(raw, dict) else str(raw)
        text = text.strip()
        raw_len = len(text)
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3:
                text = "\n".join(lines[1:-1]).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            import re
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                except json.JSONDecodeError:
                    logger.warning("Failed to parse task list from LLM output")
                    return [], {"dropped": 0, "raw_length": raw_len,
                                "parse_error": True}
            else:
                logger.warning("No JSON array found in LLM output for task planning")
                return [], {"dropped": 0, "raw_length": raw_len,
                            "parse_error": True}

        if not isinstance(parsed, list):
            return [], {"dropped": 0, "raw_length": raw_len,
                        "parse_error": True}

        tasks: list[TaskDefinition] = []
        dropped = 0
        for i, item in enumerate(parsed):
            if not isinstance(item, dict):
                dropped += 1
                continue
            description = str(item.get("description", "") or "").strip()
            if not description:
                dropped += 1
                continue
            depends_on_raw = item.get("depends_on") or []
            depends_on: list[str] = []
            if isinstance(depends_on_raw, list):
                for dep in depends_on_raw:
                    s = str(dep or "").strip()
                    if s:
                        depends_on.append(s)
            tasks.append(TaskDefinition(
                id=str(item.get("id", f"task_{i + 1}")),
                description=description,
                task_type=str(item.get("type", "code_change")),
                depends_on=depends_on,
            ))
        if dropped:
            logger.warning(
                "Structured flow: dropped %d invalid task entries during parsing",
                dropped,
            )
        return tasks, {"dropped": dropped, "raw_length": raw_len,
                       "parse_error": False}
