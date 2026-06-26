"""Shared SGAR command parsing and dispatch helpers."""

from __future__ import annotations

import logging
import re
import shlex
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

from ..services.steer_inbox import STEER_BLOCK_MARKERS
from ..sgar import CriterionResult, SgarError, read_trace, result_to_text
from ._text_masking import mask_fenced_segments, mask_quoted_segments


logger = logging.getLogger(__name__)


_COMMAND_ALIASES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("draft-blueprint",), "draft-blueprint"),
    (("draft", "blueprint"), "draft-blueprint"),
    (("draft-roadmap",), "draft-roadmap"),
    (("draft", "roadmap"), "draft-roadmap"),
    (("draft-stage-spec",), "draft-stage-spec"),
    (("draft", "stage", "spec"), "draft-stage-spec"),
    (("set-blueprint",), "set-blueprint"),
    (("write", "blueprint"), "set-blueprint"),
    (("set-roadmap",), "set-roadmap"),
    (("write", "roadmap"), "set-roadmap"),
    (("set-stage-spec",), "set-stage-spec"),
    (("write", "stage", "spec"), "set-stage-spec"),
    (("validate", "blueprint"), "validate-blueprint"),
    (("validate", "roadmap"), "validate-roadmap"),
    (("validate", "stage"), "validate-stage"),
    (("start-stage",), "start-stage"),
    (("verify",), "verify"),
    (("close-stage",), "close-stage"),
    (("reopen-stage",), "reopen-stage"),
    (("abandon-stage",), "abandon-stage"),
    # Mission isolation — previously CLI-only. Wired into the agent-driven
    # surface so a supervisor can drive isolated sub-tasks (input-hash
    # staleness, scoped outputs) that DO NOT advance the hard DAG. Both the
    # hyphenated form and the CLI-style ``mission <verb>`` form resolve.
    (("create-mission",), "create-mission"),
    (("create", "mission"), "create-mission"),
    (("mission", "create"), "create-mission"),
    (("complete-mission",), "complete-mission"),
    (("complete", "mission"), "complete-mission"),
    (("mission", "complete"), "complete-mission"),
    (("mission-status",), "mission-status"),
    (("mission", "status"), "mission-status"),
    (("mission-list",), "mission-list"),
    (("list-missions",), "mission-list"),
    (("mission", "list"), "mission-list"),
    (("doctor",), "doctor"),
    (("trace",), "trace"),
    (("status",), "status"),
    (("init",), "init"),
)

_RESUME_MARKER = "## Prior session context"
_MEMORY_MARKER = "## Persistent memory"
_CURRENT_GOAL_MARKER = "## Current goal"
CCX_GOAL_OFFSET_METADATA_KEY = "ccx_goal_offset"

# Structured error codes stamped onto a mode runner's ``extras`` when a
# governance op is rejected. The boundary keeps the legacy ``ERROR:`` text
# contract (tests + drivers grep it); these codes are the *machine-readable*
# supplement so the orchestration layer can DETECT a rejection instead of
# inferring it from a state machine that silently failed to advance.
#
# The code is decided by whether the instruction even RESOLVES to a known
# command — never by string-matching the SgarError message (which is free
# text and changes). A resolvable command that the runtime still refused is
# a genuine governance rejection; an unresolvable one is a malformed/unknown
# instruction.
SGAR_GOVERNANCE_REJECTED = "SGAR_GOVERNANCE_REJECTED"
SGAR_INSTRUCTION_UNRECOGNIZED = "SGAR_INSTRUCTION_UNRECOGNIZED"


def _strip_steer_blocks(text: str) -> str:
    out = text
    for header, footer in STEER_BLOCK_MARKERS:
        while True:
            start = out.find(header)
            if start == -1:
                break
            end = out.find(footer, start + len(header))
            if end == -1:
                out = out[:start].lstrip()
                break
            out = (out[:start] + out[end + len(footer):]).lstrip()
    return out


def _metadata_goal_offset(
    text: str,
    metadata: Mapping[str, Any] | None,
) -> int | None:
    if not metadata:
        return None
    raw = metadata.get(CCX_GOAL_OFFSET_METADATA_KEY)
    if type(raw) is not int:
        return None
    if raw <= 0 or raw > len(text):
        return None
    prefix = text[:raw].lower()
    if _CURRENT_GOAL_MARKER.lower() in prefix:
        if (
            _RESUME_MARKER.lower() in prefix
            or _MEMORY_MARKER.lower() in prefix
        ):
            return raw
    if _MEMORY_MARKER.lower() in prefix:
        return raw
    for header, _footer in STEER_BLOCK_MARKERS:
        if header.lower() in prefix:
            return raw
    return None


def _command_line(
    goal: str,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    raw_text = str(goal or "")
    offset = _metadata_goal_offset(raw_text, metadata)
    if offset is not None:
        return _strip_steer_blocks(raw_text[offset:]).strip()

    text = _strip_steer_blocks(raw_text).strip()
    if not text:
        return ""
    stripped = text.lstrip()
    lower = stripped.lower()
    if lower.startswith(_RESUME_MARKER.lower()):
        masked = mask_fenced_segments(stripped, logger=logger)
        matches = list(re.finditer(
            r"(?im)^[ \t]*" + re.escape(_CURRENT_GOAL_MARKER) + r"[ \t]*\r?$",
            masked[len(_RESUME_MARKER):],
        ))
        if matches:
            match = matches[-1]
            idx = len(_RESUME_MARKER) + match.end()
            text = stripped[idx:].strip()
    return text


def _strip_fenced_blocks_outside_quotes(command_text: str) -> str:
    out: list[str] = []
    idx = 0
    quote: str | None = None
    while idx < len(command_text):
        char = command_text[idx]
        if quote is not None:
            out.append(char)
            if quote == '"' and char == "\\" and idx + 1 < len(command_text):
                idx += 1
                out.append(command_text[idx])
            elif char == quote:
                quote = None
            idx += 1
            continue
        if char in {"'", '"'}:
            quote = char
            out.append(char)
            idx += 1
            continue
        if char == "\\" and idx + 1 < len(command_text):
            out.append(char)
            idx += 1
            out.append(command_text[idx])
            idx += 1
            continue
        if command_text.startswith("```", idx):
            end = command_text.find("```", idx + 3)
            if end == -1:
                raise SgarError("unterminated fenced block in instruction")
            out.append(" ")
            idx = end + 3
            continue
        out.append(char)
        idx += 1
    return "".join(out)


def _tokens(command_text: str) -> list[str]:
    try:
        return shlex.split(_strip_fenced_blocks_outside_quotes(command_text))
    except ValueError as exc:
        raise SgarError(f"invalid SGAR instruction: {exc}") from exc


def resolve_sgar_command(
    goal: str,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    """Resolve an anchored SGAR command from the current instruction text."""
    command_text = _command_line(goal, metadata=metadata)
    toks = [token.lower() for token in _tokens(command_text)]
    for prefix, command in _COMMAND_ALIASES:
        if tuple(toks[: len(prefix)]) == prefix:
            return command, command_text
    raise SgarError(f"unrecognized SGAR instruction: {command_text}")


def governance_error_extras(
    exc: SgarError,
    instruction: str,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the ``extras`` payload for a governance op that was rejected.

    Called from the mode-runner boundary (``BlueprintModeRunner`` /
    ``BlueprintxModeRunner``) when ``run_sgar_instruction`` raises
    :class:`SgarError`. The runner keeps ``final_text = "ERROR: <exc>"``
    (the existing contract); this only enriches ``extras`` so callers can
    detect the failure structurally:

    * ``error`` — preserved legacy key (was the entire prior extras dict).
    * ``sgar_failed`` — the boolean the orchestration layer keys on.
    * ``sgar_error`` — the message (same as ``error``, named for clarity).
    * ``sgar_error_code`` — ``SGAR_INSTRUCTION_UNRECOGNIZED`` when the text
      doesn't resolve to a known command, else ``SGAR_GOVERNANCE_REJECTED``.
    * ``sgar_command`` — the resolved command name (``None`` when unknown).
    """
    command: str | None = None
    code = SGAR_GOVERNANCE_REJECTED
    try:
        command, _ = resolve_sgar_command(instruction, metadata=metadata)
    except SgarError:
        code = SGAR_INSTRUCTION_UNRECOGNIZED
    message = str(exc)
    return {
        "error": message,
        "sgar_failed": True,
        "sgar_error": message,
        "sgar_error_code": code,
        "sgar_command": command,
    }


def _option(tokens: list[str], name: str, default: str = "") -> str:
    if name not in tokens:
        return default
    idx = tokens.index(name)
    if idx + 1 >= len(tokens):
        raise SgarError(f"{name} requires a value")
    return tokens[idx + 1]


def _option_or_none(tokens: list[str], name: str) -> str | None:
    if name not in tokens:
        return None
    return _option(tokens, name)


def _flag(tokens: list[str], name: str) -> bool:
    return name in tokens


def _session_id(command_text: str) -> str:
    return _option(_tokens(command_text), "--session", "")


def _stage_id(tokens: list[str], *, start: int = 1, default: str = "stage-01") -> str:
    if "--stage" in tokens:
        return _option(tokens, "--stage")
    skip_next = False
    value_options = {"--evidence", "--reason", "--text", "--session", "--prompt", "--artifact"}
    for token in tokens[start:]:
        if skip_next:
            skip_next = False
            continue
        if token in value_options:
            skip_next = True
            continue
        if not token.startswith("-"):
            return token
    return default


def _criterion_ids(tokens: list[str]) -> list[str]:
    prelude_value_options = {"--stage", "--session"}
    closing_value_options = {
        "--evidence",
        "--reason",
        "--text",
        "--prompt",
        "--artifact",
    }
    closing_flags = {"--pass", "--fail", "--all-pass"}
    skip_next = False
    closed = False
    ids: list[str] = []
    for token in tokens[1:]:
        if skip_next:
            skip_next = False
            continue
        if not closed and token in prelude_value_options:
            skip_next = True
            continue
        if token in closing_value_options:
            closed = True
            skip_next = True
            continue
        if token in closing_flags:
            closed = True
            continue
        if token.startswith("-"):
            closed = True
            continue
        if token.startswith("C"):
            if closed:
                raise SgarError(
                    "criterion ids must appear before verify options; "
                    f"unexpected {token!r}"
                )
            ids.append(token)
    return ids


def _block(command_text: str, label: str) -> str | None:
    masked = mask_quoted_segments(command_text)
    match = re.search(r"```" + re.escape(label) + r"(?=$|\s)", masked, re.I)
    if match is None:
        return None
    content_start = command_text.find("\n", match.end())
    if content_start == -1:
        return None
    content_start += 1
    end = command_text.find("```", content_start)
    if end == -1:
        return None
    return command_text[content_start:end].strip()


def _text_payload(tokens: list[str], command_text: str, *labels: str) -> str | None:
    explicit_text = _option_or_none(tokens, "--text")
    if explicit_text not in (None, ""):
        return explicit_text
    for label in labels:
        content = _block(command_text, label)
        if content is not None:
            return content
    return None


def _option_multi(tokens: list[str], name: str) -> list[str]:
    """Collect every value of a repeatable option (e.g. --input a --input b)."""
    values: list[str] = []
    idx = 0
    while idx < len(tokens):
        if tokens[idx] == name:
            if idx + 1 >= len(tokens):
                raise SgarError(f"{name} requires a value")
            values.append(tokens[idx + 1])
            idx += 2
            continue
        idx += 1
    return values


def _artifact_paths(tokens: list[str]) -> list[str]:
    return _option_multi(tokens, "--artifact")


def _format_trace(records: list[dict]) -> str:
    if not records:
        return "No SGAR trace records."
    lines = [f"SGAR trace records: {len(records)}"]
    for record in records[-20:]:
        lines.append(
            f"- {record.get('timestamp')} "
            f"{record.get('operation')} {record.get('status')}"
        )
    return "\n".join(lines)


def _apply_inline_writes(runtime: Any, command_text: str) -> list[str]:
    writes: list[str] = []
    blueprint = _block(command_text, "blueprint")
    if blueprint is not None:
        path = runtime.set_blueprint(blueprint)
        writes.append(f"Blueprint written: {path}")
    roadmap = _block(command_text, "roadmap")
    if roadmap is not None:
        path = runtime.set_roadmap(roadmap)
        writes.append(f"Roadmap written: {path}")
    stage_spec = (
        _block(command_text, "stage-spec")
        or _block(command_text, "stage_spec")
        or _block(command_text, "stage")
    )
    if stage_spec is not None:
        path = runtime.set_stage_spec("stage-01", stage_spec)
        writes.append(f"Stage spec written: {path}")
    return writes


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _serialize(v) for k, v in value.items()}
    return value


def _validation_text(name: str, result: Any) -> str:
    ok = bool(getattr(result, "ok", False))
    issues = list(getattr(result, "issues", []) or [])
    warnings = list(getattr(result, "warnings", []) or [])
    parts = [f"{name}: {'ok' if ok else 'failed'}"]
    if issues:
        parts.append("issues: " + "; ".join(str(item) for item in issues))
    if warnings:
        parts.append("warnings: " + "; ".join(str(item) for item in warnings))
    return "\n".join(parts)


def run_sgar_instruction(
    runtime: Any,
    instruction: str,
    *,
    llm: Any = None,
    supports_reopen_abandon: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    command, command_text = resolve_sgar_command(instruction, metadata=metadata)
    tokens = _tokens(command_text)

    if command == "init":
        state = runtime.init()
        writes = _apply_inline_writes(runtime, command_text)
        return "Initialized SGAR workspace.", {
            "state": _serialize(state),
            "writes": writes,
        }
    if command == "status":
        return runtime.status(), {}
    if command == "doctor":
        result = runtime.doctor()
        return result_to_text(result), {"result": _serialize(result)}
    if command == "set-blueprint":
        content = _text_payload(tokens, command_text, "blueprint")
        if content is None:
            raise SgarError("set-blueprint requires --text or a ```blueprint block")
        path = runtime.set_blueprint(content)
        return f"Blueprint written: {path}", {"path": str(path)}
    if command == "set-roadmap":
        content = _text_payload(tokens, command_text, "roadmap")
        if content is None:
            raise SgarError("set-roadmap requires --text or a ```roadmap block")
        path = runtime.set_roadmap(content)
        return f"Roadmap written: {path}", {"path": str(path)}
    if command == "set-stage-spec":
        stage_id = _stage_id(tokens, start=1)
        content = _text_payload(tokens, command_text, "stage-spec", "stage_spec", "stage")
        if content is None:
            raise SgarError("set-stage-spec requires --text or a ```stage-spec block")
        path = runtime.set_stage_spec(stage_id, content)
        return f"Stage spec written: {path}", {"path": str(path), "stage_id": stage_id}
    if command == "draft-blueprint":
        if llm is None:
            raise SgarError("draft-blueprint requires an LLM callable")
        path = runtime.draft_blueprint(llm=llm, prompt=_option(tokens, "--prompt"))
        return f"Drafted blueprint: {path}", {"path": str(path)}
    if command == "draft-roadmap":
        if llm is None:
            raise SgarError("draft-roadmap requires an LLM callable")
        path = runtime.draft_roadmap(llm=llm, prompt=_option(tokens, "--prompt"))
        return f"Drafted roadmap: {path}", {"path": str(path)}
    if command == "draft-stage-spec":
        if llm is None:
            raise SgarError("draft-stage-spec requires an LLM callable")
        stage_id = _stage_id(tokens, start=1)
        path = runtime.draft_stage_spec(
            stage_id,
            llm=llm,
            prompt=_option(tokens, "--prompt"),
        )
        return f"Drafted stage spec: {path}", {"path": str(path), "stage_id": stage_id}
    if command == "validate-blueprint":
        result = runtime.validate_blueprint(accept=_flag(tokens, "--accept"))
        return _validation_text("blueprint", result), {"result": _serialize(result)}
    if command == "validate-roadmap":
        result = runtime.validate_roadmap(accept=_flag(tokens, "--accept"))
        return _validation_text("roadmap", result), {"result": _serialize(result)}
    if command == "validate-stage":
        stage_id = _stage_id(tokens, start=2)
        result = runtime.validate_stage_spec(stage_id)
        return _validation_text(stage_id, result), {"result": _serialize(result), "stage_id": stage_id}
    if command == "start-stage":
        stage_id = _stage_id(tokens, start=1)
        state = runtime.start_stage(stage_id)
        return f"Started stage {stage_id}.", {"state": _serialize(state), "stage_id": stage_id}
    if command == "verify":
        stage_id = _option(tokens, "--stage", "stage-01")
        evidence = _option(tokens, "--evidence", "")
        if _flag(tokens, "--pass") and _flag(tokens, "--fail"):
            raise SgarError("verify cannot combine --pass and --fail")
        if _flag(tokens, "--all-pass"):
            raise SgarError("verify --all-pass is only available in the SGAR CLI")
        if not _flag(tokens, "--pass") and not _flag(tokens, "--fail"):
            raise SgarError("verify requires --pass or --fail")
        passed = _flag(tokens, "--pass")
        criterion_ids = _criterion_ids(tokens)
        if not criterion_ids:
            raise SgarError("verify requires a criterion id")
        report = runtime.record_verification(
            stage_id,
            results=[
                CriterionResult(criterion_id=criterion_id, passed=passed, evidence=evidence)
                for criterion_id in criterion_ids
            ],
            artifact_paths=_artifact_paths(tokens),
        )
        extras = _serialize(report)
        return f"Recorded verification for {stage_id}.", extras
    if command == "close-stage":
        stage_id = _stage_id(tokens, start=1)
        state = runtime.close_stage(stage_id)
        return f"Closed stage {stage_id}.", {"state": _serialize(state), "stage_id": stage_id}
    if command == "reopen-stage":
        if not supports_reopen_abandon:
            raise SgarError("reopen-stage requires sgarx")
        stage_id = _stage_id(tokens, start=1)
        state = runtime.reopen_stage(stage_id, reason=_option(tokens, "--reason"))
        return f"Reopened stage {stage_id}.", {"state": _serialize(state), "stage_id": stage_id}
    if command == "abandon-stage":
        if not supports_reopen_abandon:
            raise SgarError("abandon-stage requires sgarx")
        stage_id = _stage_id(tokens, start=1)
        state = runtime.abandon_stage(stage_id, reason=_option(tokens, "--reason"))
        return f"Abandoned stage {stage_id}.", {"state": _serialize(state), "stage_id": stage_id}
    if command == "trace":
        records = read_trace(runtime.store)
        return _format_trace(records), {"trace_count": len(records)}
    if command == "create-mission":
        mission_id = _option(tokens, "--id")
        if not mission_id:
            raise SgarError("create-mission requires --id")
        kind = _option(tokens, "--kind")
        if not kind:
            raise SgarError("create-mission requires --kind")
        objective = _option_or_none(tokens, "--objective")
        if objective is None:
            objective = _block(command_text, "objective")
        if not objective:
            raise SgarError(
                "create-mission requires --objective or an ```objective block"
            )
        input_paths = _option_multi(tokens, "--input")
        if not input_paths:
            raise SgarError("create-mission requires at least one --input")
        expected_outputs = _option_multi(tokens, "--expected-output")
        if not expected_outputs:
            raise SgarError(
                "create-mission requires at least one --expected-output"
            )
        scope = _option_multi(tokens, "--scope") or None
        manifest = runtime.create_mission(
            mission_id=mission_id,
            kind=kind,
            objective=objective,
            input_paths=input_paths,
            expected_outputs=expected_outputs,
            allowed_scope=scope,
        )
        return f"Created mission {mission_id}.", {
            "manifest": _serialize(manifest),
            "mission_id": mission_id,
        }
    if command == "complete-mission":
        mission_id = _option(tokens, "--id")
        if not mission_id:
            raise SgarError("complete-mission requires --id")
        result_path = _option(tokens, "--result")
        if not result_path:
            raise SgarError("complete-mission requires --result")
        manifest = runtime.complete_mission(mission_id, result_path=result_path)
        return f"Completed mission {mission_id}.", {
            "manifest": _serialize(manifest),
            "mission_id": mission_id,
        }
    if command == "mission-status":
        mission_id = _option(tokens, "--id")
        if not mission_id:
            raise SgarError("mission-status requires --id")
        return runtime.mission_status(mission_id), {"mission_id": mission_id}
    if command == "mission-list":
        return runtime.mission_list_text(), {}

    raise SgarError(f"unrecognized SGAR instruction: {command_text}")


__all__ = [
    "SGAR_GOVERNANCE_REJECTED",
    "SGAR_INSTRUCTION_UNRECOGNIZED",
    "_session_id",
    "governance_error_extras",
    "resolve_sgar_command",
    "run_sgar_instruction",
]
