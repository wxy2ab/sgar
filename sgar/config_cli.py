from __future__ import annotations

import argparse
import configparser
from pathlib import Path

from .client_registry import ClientMetadata, get_client_metadata_map, list_client_metadata


USER_CONFIG_PATH = Path.home() / ".sgar" / "setting.ini"
CONFIG_SECTION = "Default"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sgar config",
        description="管理 sgar 的用户级 LLM 配置",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("where", help="显示用户配置文件路径")
    sub.add_parser("list", help="列出支持的 ClientName、配置键和示例")

    p_set = sub.add_parser("set", help="写入用户配置")
    p_set.add_argument("--client", required=True, help="要设置的 ClientName")
    p_set.add_argument("--api-key", default=None, help="当 client 只有一个凭证键时可直接设置")
    p_set.add_argument("--model", default=None, help="当 client 有模型键时写入首选模型键")
    p_set.add_argument(
        "--key",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="显式写入任意配置键，可重复传入",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "where":
        print(USER_CONFIG_PATH)
        return 0
    if args.command == "list":
        print(_format_client_table(list_client_metadata()))
        return 0
    if args.command == "set":
        return _handle_set(args)
    parser.error(f"unknown command: {args.command}")
    return 2


def _handle_set(args: argparse.Namespace) -> int:
    metadata_map = get_client_metadata_map()
    item = metadata_map.get(args.client)
    if item is None:
        valid_names = ", ".join(sorted(metadata_map))
        raise SystemExit(f"未知 ClientName: {args.client}\n可选值: {valid_names}")

    updates = {
        "llm_api": item.client_name,
        "cc_default_llm_client": item.client_name,
    }

    if args.api_key:
        if len(item.credential_keys) != 1:
            keys_text = ", ".join(item.credential_keys) if item.credential_keys else "无可自动判定键"
            raise SystemExit(
                "--api-key 只适用于恰好有一个凭证键的 client。\n"
                f"{item.client_name} 的候选凭证键: {keys_text}\n"
                "请改用重复参数: --key KEY=VALUE"
            )
        updates[item.credential_keys[0]] = args.api_key

    if args.model:
        if not item.model_keys:
            raise SystemExit(
                f"{item.client_name} 没有可自动判定的模型键，请不要传 --model。"
            )
        updates[item.model_keys[0]] = args.model

    for raw_item in args.key or []:
        key, value = _parse_key_value(raw_item)
        updates[key] = value

    config = _load_config(USER_CONFIG_PATH)
    if not config.has_section(CONFIG_SECTION):
        config.add_section(CONFIG_SECTION)
    for key, value in updates.items():
        config[CONFIG_SECTION][key] = value
    _save_config(USER_CONFIG_PATH, config)

    print(f"已写入用户配置: {USER_CONFIG_PATH}")
    for key, value in updates.items():
        print(f"{key} = {value}")
    return 0


def _format_client_table(items: list[ClientMetadata]) -> str:
    headers = ("ClientName", "credential_keys", "model_keys", "example")
    rows = [
        (
            item.client_name,
            ", ".join(item.credential_keys) or "-",
            ", ".join(item.model_keys) or "-",
            item.one_liner,
        )
        for item in items
    ]
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def _line(values: tuple[str, str, str, str]) -> str:
        return " | ".join(value.ljust(widths[index]) for index, value in enumerate(values))

    divider = "-+-".join("-" * width for width in widths)
    lines = [_line(headers), divider]
    lines.extend(_line(row) for row in rows)
    return "\n".join(lines)


def _parse_key_value(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise SystemExit(f"无效的 --key 参数: {text!r}，应为 KEY=VALUE")
    key, value = text.split("=", 1)
    key = key.strip()
    if not key:
        raise SystemExit(f"无效的 --key 参数: {text!r}，KEY 不能为空")
    return key, value


def _load_config(path: Path) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    if path.exists():
        config.read(path, encoding="utf-8")
    return config


def _save_config(path: Path, config: configparser.ConfigParser) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        config.write(handle)
