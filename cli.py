#!/usr/bin/env python3
"""sgar 根目录命令行入口 —— ``python cli.py <command> ...``

薄封装：把根目录入口转发到库内 SGAR CLI（``core.ccx.sgar.cli:main``）。脚本就地运行时
把自身所在目录加入 ``sys.path``，使打包进来的 ``core`` 包可被 import，因此无需先
``pip install`` 也能 ``python cli.py``。等价入口：``python -m core.ccx.sgar``。

作为 openclaw / Claude Code 技能调用时，技能说明见同目录 ``skill.md``。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(argv: list[str] | None = None) -> int:
    try:
        from core.ccx.sgar.cli import main as sgar_main
    except Exception as exc:  # pragma: no cover - 友好失败而非堆栈
        print(f"[cli] 无法加载 SGAR CLI：{exc}", file=sys.stderr)
        print("请先安装依赖：pip install -r requirements.txt", file=sys.stderr)
        return 2
    return sgar_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
