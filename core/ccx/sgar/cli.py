"""Command line interface for SGAR."""

from __future__ import annotations

import argparse
import sys

from .models import CriterionResult, SgarError
from .runtime import SgarRuntime, result_to_text
from .tracing import read_trace
from .validation import parse_exit_criteria


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sgar")
    parser.add_argument("--cwd", default=".", help="project directory")
    parser.add_argument(
        "--session",
        default=None,
        help="optional isolated SGAR session id under .sgar/sessions/",
    )
    parser.add_argument(
        "--run-checks",
        action="store_true",
        help=(
            "run machine-checkable exit criteria ([check: <cmd>]) during "
            "verify/close and refuse a pass the command contradicts "
            "(opt-in; default off — criteria self-report)"
        ),
    )
    parser.add_argument(
        "--check-timeout",
        type=float,
        default=120.0,
        help="per-check timeout in seconds when --run-checks is set",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="initialize .sgar workspace")
    p_init.add_argument("--project", default=None, help="project name")
    p_init.add_argument(
        "--force",
        action="store_true",
        help="reinitialize an existing workspace (WIPES governance state)",
    )
    p_init.add_argument("--blueprint-text", default=None)
    p_init.add_argument("--roadmap-text", default=None)
    p_init.add_argument("--stage-spec-text", default=None)
    p_init.add_argument("--stage", default="stage-01")

    sub.add_parser("status", help="show project status")

    p_set_blueprint = sub.add_parser("set-blueprint", help="write blueprint.md")
    p_set_blueprint.add_argument("--text", required=True)

    p_set_roadmap = sub.add_parser("set-roadmap", help="write roadmap.md")
    p_set_roadmap.add_argument("--text", required=True)

    p_set_stage_spec = sub.add_parser(
        "set-stage-spec",
        help="write stages/<stage>/spec.md",
    )
    p_set_stage_spec.add_argument("--stage", required=True)
    p_set_stage_spec.add_argument("--text", required=True)

    for name in ("draft-blueprint", "draft-roadmap"):
        p_draft = sub.add_parser(name, help=f"LLM draft {name.removeprefix('draft-')}")
        p_draft.add_argument("--prompt", default="")
        p_draft.add_argument("--llm-client", default="SimpleDeepSeekClient")
    p_draft_stage = sub.add_parser("draft-stage-spec", help="LLM draft stage spec")
    p_draft_stage.add_argument("--stage", required=True)
    p_draft_stage.add_argument("--prompt", default="")
    p_draft_stage.add_argument("--llm-client", default="SimpleDeepSeekClient")

    p_validate = sub.add_parser("validate", help="validate governance docs")
    p_validate.add_argument("target", choices=["blueprint", "roadmap", "stage"])
    p_validate.add_argument("--stage", default=None)
    p_validate.add_argument("--accept", action="store_true")

    p_start = sub.add_parser("start-stage", help="start a stage")
    p_start.add_argument("stage_id")

    p_verify = sub.add_parser("verify", help="record verification evidence")
    p_verify.add_argument("--stage", required=True)
    p_verify.add_argument("--criterion", default=None)
    verdict = p_verify.add_mutually_exclusive_group()
    verdict.add_argument("--pass", dest="passed", action="store_true")
    verdict.add_argument("--fail", dest="failed", action="store_true")
    p_verify.add_argument("--all-pass", action="store_true")
    p_verify.add_argument("--evidence", default="")
    p_verify.add_argument("--notes", default="")
    p_verify.add_argument(
        "--artifact",
        action="append",
        default=None,
        help="additional artifact path to include in verification trace",
    )

    p_close = sub.add_parser("close-stage", help="close a verified stage")
    p_close.add_argument("stage_id")

    p_mission = sub.add_parser("mission", help="manage isolated missions")
    mission_sub = p_mission.add_subparsers(
        dest="mission_command",
        required=True,
    )
    p_mission_create = mission_sub.add_parser(
        "create",
        help="create a filesystem-backed mission",
    )
    p_mission_create.add_argument("--kind", required=True)
    p_mission_create.add_argument("--id", dest="mission_id", required=True)
    p_mission_create.add_argument("--input", action="append", required=True)
    p_mission_create.add_argument("--objective", required=True)
    p_mission_create.add_argument(
        "--expected-output",
        action="append",
        required=True,
    )
    p_mission_create.add_argument("--scope", action="append", default=None)

    p_mission_status = mission_sub.add_parser(
        "status",
        help="show mission status",
    )
    p_mission_status.add_argument("mission_id")

    p_mission_complete = mission_sub.add_parser(
        "complete",
        help="complete a mission with a result artifact",
    )
    p_mission_complete.add_argument("mission_id")
    p_mission_complete.add_argument("--result", required=True)

    mission_sub.add_parser("list", help="list missions")

    sub.add_parser("doctor", help="detect missing files or inconsistent state")
    sub.add_parser("trace", help="show SGAR operation trace summary")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    runtime = SgarRuntime(
        args.cwd,
        session_id=args.session,
        run_criterion_checks=args.run_checks,
        criterion_check_timeout_s=args.check_timeout,
    )
    try:
        if args.command == "init":
            state = runtime.init(project_name=args.project, force=args.force)
            if args.blueprint_text is not None:
                runtime.set_blueprint(args.blueprint_text)
            if args.roadmap_text is not None:
                runtime.set_roadmap(args.roadmap_text)
            if args.stage_spec_text is not None:
                runtime.set_stage_spec(args.stage, args.stage_spec_text)
            print(f"Initialized SGAR workspace for {state.project_name}")
            return 0
        if args.command == "status":
            print(runtime.status())
            return 0
        if args.command == "set-blueprint":
            runtime.set_blueprint(args.text)
            print(f"Wrote blueprint: {runtime.store.blueprint_path}")
            return 0
        if args.command == "set-roadmap":
            runtime.set_roadmap(args.text)
            print(f"Wrote roadmap: {runtime.store.roadmap_path}")
            return 0
        if args.command == "set-stage-spec":
            runtime.set_stage_spec(args.stage, args.text)
            print(f"Wrote stage spec: {runtime.store.stage_spec_path(args.stage)}")
            return 0
        if args.command == "draft-blueprint":
            runtime.draft_blueprint(
                llm=_default_llm_callable(args.llm_client),
                prompt=args.prompt,
            )
            print(f"Drafted blueprint: {runtime.store.blueprint_path}")
            return 0
        if args.command == "draft-roadmap":
            runtime.draft_roadmap(
                llm=_default_llm_callable(args.llm_client),
                prompt=args.prompt,
            )
            print(f"Drafted roadmap: {runtime.store.roadmap_path}")
            return 0
        if args.command == "draft-stage-spec":
            runtime.draft_stage_spec(
                args.stage,
                llm=_default_llm_callable(args.llm_client),
                prompt=args.prompt,
            )
            print(f"Drafted stage spec: {runtime.store.stage_spec_path(args.stage)}")
            return 0
        if args.command == "validate":
            if args.target == "blueprint":
                result = runtime.validate_blueprint(accept=args.accept)
            elif args.target == "roadmap":
                result = runtime.validate_roadmap(accept=args.accept)
            else:
                if not args.stage:
                    raise SgarError("validate stage requires --stage")
                result = runtime.validate_stage_spec(args.stage)
            print(result_to_text(result))
            return 0 if result.ok else 1
        if args.command == "start-stage":
            runtime.start_stage(args.stage_id)
            print(f"Started stage: {args.stage_id}")
            return 0
        if args.command == "verify":
            results = _verification_results(runtime, args)
            runtime.record_verification(
                args.stage,
                results=results,
                notes=args.notes,
                artifact_paths=args.artifact,
            )
            print(f"Recorded verification for {args.stage}")
            return 0
        if args.command == "close-stage":
            runtime.close_stage(args.stage_id)
            print(f"Closed stage: {args.stage_id}")
            return 0
        if args.command == "mission":
            if args.mission_command == "create":
                manifest = runtime.create_mission(
                    mission_id=args.mission_id,
                    kind=args.kind,
                    objective=args.objective,
                    input_paths=args.input,
                    expected_outputs=args.expected_output,
                    allowed_scope=args.scope,
                )
                print(f"Created mission: {manifest['mission_id']}")
                return 0
            if args.mission_command == "status":
                print(runtime.mission_status(args.mission_id))
                return 0
            if args.mission_command == "complete":
                manifest = runtime.complete_mission(
                    args.mission_id,
                    result_path=args.result,
                )
                print(f"Completed mission: {manifest['mission_id']}")
                return 0
            if args.mission_command == "list":
                print(runtime.mission_list_text())
                return 0
        if args.command == "doctor":
            result = runtime.doctor()
            print(result_to_text(result))
            return 0 if result.ok else 1
        if args.command == "trace":
            records = read_trace(runtime.store)
            print(_format_trace(records))
            return 0
    except SgarError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    parser.error(f"unhandled command: {args.command}")
    return 2


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


def _verification_results(runtime: SgarRuntime, args) -> list[CriterionResult]:
    if args.all_pass:
        if not str(args.evidence or "").strip():
            raise SgarError("verify --all-pass requires --evidence")
        text = runtime.store.read_text(runtime.store.stage_spec_path(args.stage))
        return [
            CriterionResult(
                criterion_id=criterion.criterion_id,
                passed=True,
                evidence=args.evidence,
            )
            for criterion in parse_exit_criteria(text)
        ]
    if not args.criterion:
        raise SgarError("verify requires --criterion or --all-pass")
    if not args.passed and not args.failed:
        raise SgarError("verify requires --pass, --fail, or --all-pass")
    return [
        CriterionResult(
            criterion_id=args.criterion,
            passed=bool(args.passed and not args.failed),
            evidence=args.evidence,
        )
    ]


def _default_llm_callable(name: str):
    from core.llms.llm_factory import LLMFactory

    client = LLMFactory().get_instance(name)

    def _call(*, system: str, user: str, purpose: str) -> str:
        del purpose
        return str(client.one_chat([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]))

    return _call


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
