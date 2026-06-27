from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
from typing import Any

from . import config_cli


MODE_ALIASES: dict[str, str] = {
    "plan": "plan",
    "spec": "spec",
    "agent": "agent",
    "doc": "doc",
    "ask": "ask",
    "blueprint": "blueprint",
    "sgarx": "sgarx",
    "goal": "goal",
    "debug": "debug",
}


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(_top_level_help())
        return 0
    if args[0] == "config":
        return config_cli.main(args[1:])
    if args[0] == "run":
        return _run_mode_cli(args[1:])
    if args[0] in MODE_ALIASES:
        return _run_mode_cli(args[1:], mode=MODE_ALIASES[args[0]])
    return _forward_to_core(args)


def _forward_to_core(argv: list[str]) -> int:
    try:
        from core.ccx.sgar.cli import main as core_main
    except Exception as exc:  # pragma: no cover - friendly failure for thin wrapper
        print(f"[sgar] 无法加载 core CLI：{exc}", file=sys.stderr)
        print("请先安装依赖：pip install -r requirements.txt", file=sys.stderr)
        return 2
    return core_main(argv)


def _run_mode_cli(argv: list[str], *, mode: str | None = None) -> int:
    parser = build_mode_parser(mode=mode)
    args = parser.parse_args(argv)
    effective_mode = mode or args.mode
    instruction = _resolve_instruction(parser, args)
    metadata = _parse_metadata_json(parser, args.metadata_json)
    if args.docs_output_path:
        metadata["docs_output_path"] = args.docs_output_path
    result = _run_code_agent(
        instruction=instruction,
        mode=effective_mode,
        cwd=args.cwd,
        prompt_language=args.prompt_language,
        permission_mode=args.permission_mode,
        max_tool_rounds=args.max_tool_rounds,
        metadata=metadata,
    )
    return _render_agent_result(result, json_output=args.json)


def build_mode_parser(mode: str | None = None) -> argparse.ArgumentParser:
    if mode is None:
        prog = "sgar run"
        description = "Run any ccx mode through the unified sgar entrypoint."
    else:
        prog = f"sgar {mode}"
        description = f"Shortcut for `sgar run --mode {mode}`."
    parser = argparse.ArgumentParser(prog=prog, description=description)
    if mode is None:
        parser.add_argument(
            "--mode",
            required=True,
            choices=sorted({"sgar", *MODE_ALIASES.values()}),
            help="ccx agent mode to run",
        )
    parser.add_argument("instruction", nargs="?", help="task instruction")
    parser.add_argument(
        "--instruction",
        dest="instruction_flag",
        default=None,
        help="task instruction; useful when the text starts with `-`",
    )
    parser.add_argument("--cwd", default=".", help="working directory")
    parser.add_argument(
        "--prompt-language",
        default=None,
        help="override prompt language for this run",
    )
    parser.add_argument(
        "--permission-mode",
        default=None,
        help="override permission mode for this run",
    )
    parser.add_argument(
        "--max-tool-rounds",
        type=int,
        default=None,
        help="limit tool rounds for this run",
    )
    parser.add_argument(
        "--docs-output-path",
        default=None,
        help="optional docs output path for doc/goal-style runs",
    )
    parser.add_argument(
        "--metadata-json",
        default=None,
        help="raw request metadata as a JSON object",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the full AgentRunResult as JSON",
    )
    return parser


def _resolve_instruction(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> str:
    instruction = args.instruction_flag or args.instruction
    if not instruction or not str(instruction).strip():
        parser.error("missing instruction; pass it positionally or via --instruction")
    return str(instruction)


def _parse_metadata_json(
    parser: argparse.ArgumentParser,
    raw: str | None,
) -> dict[str, Any]:
    if raw is None:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        parser.error(f"invalid --metadata-json: {exc}")
    if not isinstance(parsed, dict):
        parser.error("--metadata-json must decode to a JSON object")
    return parsed


def _run_code_agent(
    *,
    instruction: str,
    mode: str,
    cwd: str,
    prompt_language: str | None,
    permission_mode: str | None,
    max_tool_rounds: int | None,
    metadata: dict[str, Any],
):
    from core.cc.config import load_cc_config
    from core.ccx import AgentRunRequest, CodeAgent

    config = load_cc_config()
    agent = CodeAgent(config=config)
    return agent.run_sync(
        AgentRunRequest(
            instruction=instruction,
            cwd=cwd,
            config=config,
            max_tool_rounds=max_tool_rounds,
            prompt_language=prompt_language,
            permission_mode=permission_mode,
            agent_mode=mode,
            metadata=dict(metadata),
        )
    )


def _render_agent_result(result: Any, *, json_output: bool) -> int:
    if json_output:
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2, default=str))
    else:
        _print_text_result(result)
    return 1 if getattr(result, "failed", False) else 0


def _print_text_result(result: Any) -> None:
    final_text = str(getattr(result, "final_text", "") or "").strip()
    snapshot = dict(getattr(result, "session_snapshot", {}) or {})
    if final_text:
        print(final_text)
    artifact_path = snapshot.get("artifact_path")
    if artifact_path:
        print(f"\nartifact: {artifact_path}")
    child_artifacts = snapshot.get("child_artifacts") or []
    if child_artifacts:
        print("\nchild artifacts:")
        for item in child_artifacts:
            print(f"- {item.get('artifact_path')}")
    if getattr(result, "failed", False):
        error_code = getattr(result, "error_code", None) or "CCX_RUN_FAILED"
        error_message = getattr(result, "error_message", None) or "run failed"
        print(f"ERROR: {error_code}: {error_message}", file=sys.stderr)


def _top_level_help() -> str:
    try:
        from core.ccx.sgar.cli import build_parser
    except Exception:
        core_help = "无法加载 core CLI 帮助，请先安装依赖。"
    else:
        core_help = build_parser().format_help().rstrip()
    return (
        f"{core_help}\n\n"
        "Additional wrapper commands:\n"
        "  config    管理 ~/.sgar/setting.ini 中的用户级 LLM 配置\n\n"
        "Unified ccx mode entrypoints:\n"
        "  run       通过 `sgar run --mode <mode>` 运行任意 ccx mode\n"
        "  plan      `sgar run --mode plan` 的快捷方式\n"
        "  spec      `sgar run --mode spec` 的快捷方式\n"
        "  agent     `sgar run --mode agent` 的快捷方式\n"
        "  doc       `sgar run --mode doc` 的快捷方式\n"
        "  ask       `sgar run --mode ask` 的快捷方式\n"
        "  blueprint `sgar run --mode blueprint` 的快捷方式\n"
        "  sgarx     `sgar run --mode sgarx` 的快捷方式\n"
        "  goal      `sgar run --mode goal` 的快捷方式\n"
        "  debug     `sgar run --mode debug` 的快捷方式\n\n"
        "Examples:\n"
        "  sgar config where\n"
        "  sgar config list\n"
        "  sgar config set --client SimpleDeepSeekClient --api-key <KEY> --model <MODEL>\n"
        "  sgar run --mode sgar \"repair flaky tests and close the stage\"\n"
        "  sgar plan \"design a migration plan for the auth module\"\n"
        "  sgar sgarx \"continue the governed coding run with a harder task\"\n"
    )
