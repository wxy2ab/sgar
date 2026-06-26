from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ..safety import classify_file_permission
from ..safety.file_rules import resolve_under_cwd
from .base import BaseTool, ToolCall, ToolResult, ToolSpec, ValidationResult
from .context import ToolUseContext


_DEFAULT_MAX_BYTES = 100_000


class FileReadTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            ToolSpec(
                name="file_read",
                description=(
                    "Read a UTF-8 text file from the current workspace. Large "
                    "files can be paged with max_bytes + offset."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "max_bytes": {"type": "integer"},
                        "offset": {
                            "type": "integer",
                            "description": (
                                "Byte offset to start reading from (default 0). "
                                "Use the offset reported in a truncation marker "
                                "to page through a file larger than max_bytes."
                            ),
                        },
                    },
                    "required": ["file_path"],
                },
                is_read_only=True,
            )
        )

    def validate_input(self, arguments: dict[str, Any]) -> ValidationResult:
        if not arguments.get("file_path"):
            return ValidationResult(ok=False, message="file_path is required.")
        return ValidationResult(ok=True)

    def check_permissions(self, ctx: ToolUseContext, arguments: dict[str, Any]):
        return classify_file_permission(
            file_path=arguments["file_path"],
            cwd=ctx.cwd,
            mode=ctx.permissions.mode,
            allowed_paths=ctx.permissions.allowed_paths,
            denied_paths=ctx.permissions.denied_paths,
            operation="read",
        )

    async def execute(self, tool_call: ToolCall, ctx: ToolUseContext) -> ToolResult:
        file_path = resolve_under_cwd(tool_call.arguments["file_path"], ctx.cwd)
        max_bytes = max(1, int(tool_call.arguments.get("max_bytes") or _DEFAULT_MAX_BYTES))
        offset = max(0, int(tool_call.arguments.get("offset") or 0))
        try:
            fs = ctx.get_fs()
            if fs is not None:
                stat = await fs.stat(str(file_path))
                content = await fs.read_file(str(file_path))
                raw_bytes = content.encode("utf-8")
                size_bytes = stat.size
            else:
                raw_bytes = await asyncio.to_thread(file_path.read_bytes)
                content = raw_bytes.decode("utf-8")
                size_bytes = len(raw_bytes)
        except FileNotFoundError:
            return ToolResult(
                tool_use_id=tool_call.tool_use_id,
                tool_name=tool_call.tool_name,
                success=False,
                content=f"File not found: {file_path}",
                error_code="ED1001",
            )
        except UnicodeDecodeError:
            return ToolResult(
                tool_use_id=tool_call.tool_use_id,
                tool_name=tool_call.tool_name,
                success=False,
                content=f"Cannot read file as UTF-8 text (binary or non-UTF-8 encoding): {file_path}",
                error_code="ED1002",
            )
        except (PermissionError, OSError) as exc:
            return ToolResult(
                tool_use_id=tool_call.tool_use_id,
                tool_name=tool_call.tool_name,
                success=False,
                content=f"Cannot read file: {exc}",
                error_code="ED1003",
            )
        total_bytes = len(raw_bytes)
        if offset == 0:
            # Default path — byte-identical to the pre-offset behaviour.
            truncated = total_bytes > max_bytes
            if truncated:
                content = raw_bytes[:max_bytes].decode("utf-8", errors="replace")
                content = f"{content}\n\n[truncated to {max_bytes} bytes out of {size_bytes} bytes]"
        else:
            # Windowed read for continuation. The window may split a multibyte
            # char at either edge, so decode with replacement.
            window = raw_bytes[offset:offset + max_bytes]
            end = offset + len(window)
            truncated = end < total_bytes
            if offset >= total_bytes:
                content = (
                    f"[offset {offset} is at or past end of file "
                    f"({total_bytes} bytes); nothing to read]"
                )
            else:
                tail = (
                    f"; re-read with offset={end} for more]" if truncated else "]"
                )
                content = (
                    f"{window.decode('utf-8', errors='replace')}"
                    f"\n\n[bytes {offset}-{end} of {total_bytes}{tail}"
                )
        return ToolResult(
            tool_use_id=tool_call.tool_use_id,
            tool_name=tool_call.tool_name,
            success=True,
            content=content,
            data={
                "file_path": str(file_path),
                "size": len(content),
                "size_bytes": size_bytes,
                "max_bytes": max_bytes,
                "offset": offset,
                "truncated": truncated,
            },
            truncated=truncated,
        )
