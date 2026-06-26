from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

from ..command_runner import default_shell_kind, execute_command
from .requests import EditValidationResult, FileEditRequest


class EditValidator:
    def __init__(self, *, max_file_size_bytes: int = 1_000_000) -> None:
        self.max_file_size_bytes = max_file_size_bytes

    def validate_text(self, request: FileEditRequest, *, original_exists: bool, original_content: str) -> EditValidationResult:
        path = Path(request.file_path)
        if not original_exists and not request.create_if_missing:
            return EditValidationResult(
                ok=False,
                stage="text",
                messages=[f"File does not exist: {path}"],
                error_code="ED1001",
            )
        if original_exists and len(original_content.encode("utf-8")) > self.max_file_size_bytes:
            return EditValidationResult(
                ok=False,
                stage="text",
                messages=[f"File too large: {path}"],
                error_code="ED1008",
            )
        if original_exists and not request.old_string and request.new_string == original_content:
            return EditValidationResult(
                ok=False,
                stage="text",
                messages=["Edit is empty."],
                error_code="ED1002",
            )
        return EditValidationResult(ok=True, stage="text")

    def validate_structure(self, *, code: str, file_path: str, validate_python_syntax: bool) -> EditValidationResult:
        if not validate_python_syntax or not file_path.endswith(".py"):
            return EditValidationResult(ok=True, stage="structure")
        try:
            ast.parse(code, filename=file_path)
        except SyntaxError as exc:
            return EditValidationResult(
                ok=False,
                stage="structure",
                messages=[f"Syntax error at line {exc.lineno}: {exc.msg}"],
                error_code="ED1005",
            )
        return EditValidationResult(ok=True, stage="structure")

    def validate_semantics(self, *, code: str, file_path: str) -> EditValidationResult:
        if not file_path.endswith(".py"):
            return EditValidationResult(ok=True, stage="semantic")
        try:
            tree = ast.parse(code, filename=file_path)
        except SyntaxError:
            return EditValidationResult(
                ok=False,
                stage="semantic",
                messages=["Semantic validation skipped because syntax parsing failed."],
                error_code="ED1005",
            )
        imports = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        warnings: list[str] = []
        if not imports:
            warnings.append("No imports detected after edit.")
        return EditValidationResult(ok=True, stage="semantic", warnings=warnings)

    def validate_runtime(
        self,
        *,
        file_path: str,
        runtime_command: str | None,
        runtime_shell: str | None = None,
    ) -> EditValidationResult:
        if not runtime_command:
            return EditValidationResult(ok=True, stage="runtime")
        completed = execute_command(
            command=runtime_command,
            cwd=str(Path(file_path).resolve().parent),
            shell_kind=runtime_shell or default_shell_kind(),
        )
        if not completed.success:
            messages = [msg for msg in [completed.stdout.strip(), completed.stderr.strip()] if msg]
            return EditValidationResult(
                ok=False,
                stage="runtime",
                messages=messages or [f"Runtime command failed: {runtime_command}"],
                error_code="ED1006",
            )
        return EditValidationResult(
            ok=True,
            stage="runtime",
            messages=[completed.stdout.strip()] if completed.stdout.strip() else [],
        )

    def ensure_all_ok(self, results: Iterable[EditValidationResult]) -> None:
        for result in results:
            if not result.ok:
                raise RuntimeError(result.error_code or "ED1006")
