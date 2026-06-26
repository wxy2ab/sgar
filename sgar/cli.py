from __future__ import annotations

import sys

from . import config_cli


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(_top_level_help())
        return 0
    if args[0] == "config":
        return config_cli.main(args[1:])
    return _forward_to_core(args)


def _forward_to_core(argv: list[str]) -> int:
    try:
        from core.ccx.sgar.cli import main as core_main
    except Exception as exc:  # pragma: no cover - friendly failure for thin wrapper
        print(f"[sgar] 无法加载 core CLI：{exc}", file=sys.stderr)
        print("请先安装依赖：pip install -r requirements.txt", file=sys.stderr)
        return 2
    return core_main(argv)


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
        "Examples:\n"
        "  sgar config where\n"
        "  sgar config list\n"
        "  sgar config set --client SimpleDeepSeekClient --api-key <KEY> --model <MODEL>\n"
    )
