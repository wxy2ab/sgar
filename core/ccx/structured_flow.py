"""ccx StructuredFlowRunner — v5-backed analyze → plan → execute → summarize.

Same 4-PHASE SHAPE as ``core.cc.structured_flow.StructuredFlowRunner``,
but — unlike cc's version — this is a TEXT-ONLY pipeline: every phase
is a bare LLM call with NO tool access. Phase 1 cannot read files,
phase 3 "execution" runs each task through the lite agent runner (a
single tool-less LLM call), so no file is ever read or modified by
this flow. Phase 3 runs as a v5 DAG: each task becomes a sibling
``ccx.agent`` NodeSpec dispatched in parallel.

The output is therefore an ANALYSIS / PROPOSAL document, not a record
of applied changes, and the prompts + result metadata say so
explicitly. Callers that need real tool-backed execution should use
``agent_mode='plan'``/``'spec'``/``'agent'`` with
``agent_runner_kind='cc_query_loop'``, or cc's own structured flow.

Public surface kept compatible with cc:

* ``StructuredFlowRunner(config, llm_client_provider)`` — same constructor
* ``async run(instruction, *, cwd, prompt_language, permission_mode, event_sink)`` →
  ``PhaseResult``
* ``PhaseResult`` & ``TaskDefinition`` dataclasses re-exported

ccx CodeAgent routes ``agent_mode="structured"`` to this runner instead
of raising NotImplementedError.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.cc.config import CCConfig, load_cc_config
from core.cc.llm import DefaultLLMClientProvider, LLMClientProvider

from .agents.cc_agent import LLMCallableProvider
from .modes.llm_client import LLMCallable, from_provider, text_of
from .modes.parsing import parse_llm_json


logger = logging.getLogger(__name__)


_TASK_PLAN_SYSTEM_PROMPT = """\
You are a task planner. Based on the analysis report and the user's original instruction,
create a structured list of actionable tasks. Each task should be independent and specific.

Output a JSON array of task objects, each with:
- "id": a short identifier (e.g., "task_1")
- "description": clear description of what to do
- "type": one of "code_change", "investigation", "documentation", "test"

Output ONLY the JSON array, no other text. Example:
[
  {"id": "task_1", "description": "Fix the null check in auth.py line 42", "type": "code_change"},
  {"id": "task_2", "description": "Add unit test for the login function", "type": "test"}
]
"""


_ANALYSIS_INSTRUCTION_TEMPLATE = """\
## Analysis Phase

You are in the **analysis-only** phase. Your goal is to thoroughly understand the problem
before any implementation begins.

**Original instruction**: {instruction}

**Rules for this phase**:
1. You have NO tools in this phase — you cannot read files or run commands.
   Reason from the instruction and any context it contains; do NOT pretend
   to have inspected files you cannot see.
2. Provide a detailed analysis report summarizing:
   - What code/areas are likely relevant and why
   - What probably needs to change and why
   - Any risks or dependencies to consider
   - What you could NOT determine without reading the code — list these
     explicitly as open questions instead of guessing
"""


_SUMMARY_SYSTEM_PROMPT = """\
You are a task summarizer. Based on the original instruction and all task results,
write a clear, concise summary for the user.

IMPORTANT: the tasks were executed by text-only assistants with NO file or
shell access. Nothing on disk was changed. Frame every result as analysis or
a PROPOSED change — never claim that a file was edited, created, or that
tests were run. Explain:
1. What was analyzed / proposed per task
2. Any issues or open questions encountered
3. Recommended next steps for a human (or a tool-equipped agent) to apply

Be specific and actionable. Reference file paths and line numbers where relevant.
"""


# --------------------------------------------------------------------------- #
# Public dataclasses (mirror cc shape so callers swap imports cleanly)
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class TaskDefinition:
    id: str
    description: str
    task_type: str = "code_change"


@dataclass(slots=True)
class PhaseResult:
    phase: str
    success: bool
    output: str
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

class StructuredFlowRunner:
    """v5-backed structured flow.

    Phase 3 (task execution) is dispatched through a v5 SwarmCoordinator
    so independent tasks run in parallel under v5's lease/heartbeat
    discipline.
    """

    def __init__(
        self,
        *,
        config: CCConfig | None = None,
        llm_client_provider: LLMClientProvider | None = None,
        llm: LLMCallable | None = None,
    ) -> None:
        self.config = config or load_cc_config()
        if llm is not None:
            self._llm: LLMCallable = llm
            self.llm_client_provider = LLMCallableProvider(llm)
        else:
            self.llm_client_provider = (
                llm_client_provider or DefaultLLMClientProvider()
            )
            self._llm = from_provider(self.llm_client_provider, self.config)

    async def run(
        self,
        instruction: str,
        *,
        cwd: str = ".",
        prompt_language: str | None = None,
        permission_mode: str | None = None,
        event_sink: Any | None = None,
        workspace: Path | str | None = None,
    ) -> PhaseResult:
        del prompt_language, permission_mode  # not used in this minimal port
        cwd_path = Path(cwd).resolve() if cwd else Path.cwd().resolve()
        ws = Path(workspace) if workspace else cwd_path / ".ccx" / "structured"
        ws.mkdir(parents=True, exist_ok=True)

        # --- Phase 1: Analysis ---
        analysis_text = text_of(await asyncio.to_thread(
            self._llm,
            system="You are an analysis assistant. Use read-only reasoning.",
            user=_ANALYSIS_INSTRUCTION_TEMPLATE.format(instruction=instruction),
            purpose="structured_flow.analysis",
        ))
        if not analysis_text or not isinstance(analysis_text, str):
            return PhaseResult(
                phase="analysis",
                success=False,
                output="",
                error="analysis returned empty or non-text",
            )

        # --- Phase 2: Task Planning ---
        tasks = await self._plan_tasks(instruction, analysis_text)
        if not tasks:
            return PhaseResult(
                phase="planning",
                success=True,
                output=analysis_text,
                metadata={"note": "No tasks generated, returning analysis only"},
            )

        # --- Phase 3: Parallel Task Execution via v5 ---
        task_outputs, any_failed = await self._execute_tasks_parallel(
            tasks, analysis_text, ws, event_sink,
        )

        # --- Phase 4: Summary ---
        summary = await self._generate_summary(instruction, task_outputs)
        return PhaseResult(
            phase="completed",
            success=not any_failed,
            output=summary,
            metadata={
                "task_count": len(tasks),
                "analysis_length": len(analysis_text),
                "via": "ccx.v5",
                # No phase has tool access; nothing on disk was touched.
                # Consumers must treat the output as analysis/proposals.
                "execution": "text_only_no_tools",
            },
        )

    # -- Phase 2 helper ------------------------------------------------------

    async def _plan_tasks(
        self, instruction: str, analysis: str,
    ) -> list[TaskDefinition]:
        raw = text_of(await asyncio.to_thread(
            self._llm,
            system=_TASK_PLAN_SYSTEM_PROMPT,
            user=(
                f"Original instruction: {instruction}\n\n"
                f"Analysis report:\n{analysis[:6000]}\n\n"
                f"Create the task list as a JSON array."
            ),
            purpose="structured_flow.planning",
        ))
        return self._parse_task_list(raw)

    # -- Phase 3 helper ------------------------------------------------------

    async def _execute_tasks_parallel(
        self,
        tasks: list[TaskDefinition],
        analysis_text: str,
        workspace: Path,
        event_sink: Any | None,
    ) -> tuple[list[str], bool]:
        from .agents.swarm import SwarmCoordinator, WorkerAssignment

        coord = SwarmCoordinator(
            workspace=workspace / "swarm",
            llm=self._llm,
            team_id="structured-flow",
            language=self.config.prompt_language,
            parallelism=max(2, getattr(self.config,
                                       "spec_max_parallel_agents", 4)),
        )
        analysis_blurb = analysis_text[:3000]
        assignments = [
            WorkerAssignment(
                description=t.description,
                prompt=(
                    f"## Task: {t.description}\n\n"
                    f"Context from analysis:\n{analysis_blurb}\n\n"
                    "Work on this specific task only. You have NO file or "
                    "shell access: produce the analysis / concrete proposed "
                    "change (diffs, steps) as text — do not claim to have "
                    "applied anything."
                ),
                runtime_id=t.id,
                timeout_seconds=None,
                max_retries=1,
            )
            for t in tasks
        ]
        summary = await coord.coordinate(
            assignments=assignments,
            event_sink=event_sink,
            stop_on_failure=False,
        )
        outputs: list[str] = []
        any_failed = False
        for run, task in zip(summary.runs, tasks):
            status = "OK" if run.success else "FAILED"
            text = run.result.get("final_text", "") if run.success else (run.error or "")
            outputs.append(
                f"### Task {task.id}: {task.description}\n"
                f"Status: {status}\n"
                f"Result: {text}\n"
            )
            if not run.success:
                any_failed = True
        return outputs, any_failed

    # -- Phase 4 helper ------------------------------------------------------

    async def _generate_summary(
        self, instruction: str, task_outputs: list[str],
    ) -> str:
        combined = "\n\n".join(task_outputs)
        raw = text_of(await asyncio.to_thread(
            self._llm,
            system=_SUMMARY_SYSTEM_PROMPT,
            user=(
                f"Original instruction: {instruction}\n\n"
                f"Task results:\n{combined[:8000]}\n\n"
                f"Write a summary."
            ),
            purpose="structured_flow.summary",
        ))
        if isinstance(raw, dict):
            return str(raw.get("content", raw))
        return str(raw or combined)

    # -- parsing -------------------------------------------------------------

    @staticmethod
    def _parse_task_list(raw: Any) -> list[TaskDefinition]:
        text = str(raw.get("content", raw)) if isinstance(raw, dict) else str(raw)
        parsed = parse_llm_json(
            text,
            schema_name="task_list",
            fallback_factory=lambda _raw: [],
            expected_type=list,
        )
        tasks: list[TaskDefinition] = []
        seen_ids: set[str] = set()
        for i, item in enumerate(parsed):
            if not isinstance(item, dict):
                continue
            description = str(item.get("description", ""))
            if not description:
                continue
            # LLM-supplied ids become v5 node_ids; a duplicate (two
            # tasks both labelled "task_1") would raise an unhandled
            # DuplicateNodeError and abort the whole run. De-dupe here.
            task_id = str(item.get("id") or f"task_{i + 1}") or f"task_{i + 1}"
            if task_id in seen_ids:
                base = task_id
                n = 2
                while f"{base}_{n}" in seen_ids:
                    n += 1
                task_id = f"{base}_{n}"
            seen_ids.add(task_id)
            tasks.append(TaskDefinition(
                id=task_id,
                description=description,
                task_type=str(item.get("type", "code_change")),
            ))
        return tasks


__all__ = [
    "PhaseResult",
    "StructuredFlowRunner",
    "TaskDefinition",
]
