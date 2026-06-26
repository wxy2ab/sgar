from __future__ import annotations

import difflib
import os
import tempfile
from pathlib import Path
from typing import Any

from ..errors import CCError
from .file_state import FileStateCache, assert_file_not_modified, compute_file_hash
from .requests import EditResult, EditValidationResult, FileEditRequest, PatchPreview, RollbackResult
from .rollback import RollbackManager
from .validator import EditValidator


_LLM_BACKENDS = {"line", "robust", "smart_v2", "lnfree"}


def _instantiate_llm_backend(name: str, llm_client: Any) -> Any:
    """Construct one of the supported LLM editor backends by short name."""
    if name == "line":
        from core.utils.llm_code_editor import LLMCodeEditor
        return LLMCodeEditor(llm_client)
    if name == "robust":
        from core.utils.robust_llm_editor import RobustLLMEditor
        return RobustLLMEditor(llm_client=llm_client)
    if name == "smart_v2":
        from core.utils.smart_llm_editor_v2 import SmartLLMEditorV2
        return SmartLLMEditorV2(llm_client=llm_client)
    if name == "lnfree":
        from core.utils.llm_block_editor_lnfree import (
            EditorConfig,
            LineNumberFreeLLMBlockEditor,
        )
        return LineNumberFreeLLMBlockEditor(
            llm_client=llm_client, config=EditorConfig()
        )
    raise CCError(f"Unknown LLM editor backend: {name!r}", error_code="ED1006")


class CodeEditFacade:
    def __init__(
        self,
        *,
        file_state_cache: FileStateCache | None = None,
        validator: EditValidator | None = None,
        rollback_manager: RollbackManager | None = None,
        default_llm_backend: str = "line",
    ) -> None:
        if default_llm_backend not in _LLM_BACKENDS:
            raise ValueError(
                f"default_llm_backend must be one of {_LLM_BACKENDS}, got {default_llm_backend!r}"
            )
        self.file_state_cache = file_state_cache or FileStateCache()
        self.validator = validator or EditValidator()
        self.rollback_manager = rollback_manager or RollbackManager(Path(".cc/runtime/checkpoints"))
        self.default_llm_backend = default_llm_backend

    def preview_edit(self, request: FileEditRequest) -> PatchPreview:
        original_snapshot = self.file_state_cache.read(request.file_path)
        updated_content = self._build_updated_content(request, original_snapshot.content, original_snapshot.exists)
        return self._build_patch_preview(
            file_path=request.file_path,
            before_content=original_snapshot.content,
            after_content=updated_content,
        )

    def apply_precise_edit(self, request: FileEditRequest) -> EditResult:
        original_snapshot = self.file_state_cache.read(request.file_path)
        assert_file_not_modified(
            current_hash=original_snapshot.file_hash,
            expected_hash=request.expected_hash,
        )
        text_result = self.validator.validate_text(
            request,
            original_exists=original_snapshot.exists,
            original_content=original_snapshot.content,
        )
        if not text_result.ok:
            return self._failure_result(request, original_snapshot.file_hash, text_result)

        checkpoint = self.rollback_manager.create_checkpoint(
            file_path=request.file_path,
            content=original_snapshot.content,
            existed_before=original_snapshot.exists,
        )
        try:
            updated_content = self._build_updated_content(
                request,
                original_snapshot.content,
                original_snapshot.exists,
            )
        except CCError as exc:
            return EditResult(
                success=False,
                file_path=request.file_path,
                content=original_snapshot.content,
                before_hash=original_snapshot.file_hash,
                after_hash=original_snapshot.file_hash,
                checkpoint_id=checkpoint.checkpoint_id,
                error_code=exc.error_code,
            )
        preview = self._build_patch_preview(
            file_path=request.file_path,
            before_content=original_snapshot.content,
            after_content=updated_content,
        )

        validation_results = [
            text_result,
            self.validator.validate_structure(
                code=updated_content,
                file_path=request.file_path,
                validate_python_syntax=request.validate_python_syntax,
            ),
        ]
        if all(result.ok for result in validation_results):
            validation_results.append(
                self.validator.validate_semantics(code=updated_content, file_path=request.file_path)
            )
        if all(result.ok for result in validation_results):
            validation_results.append(
                self.validator.validate_runtime(
                    file_path=request.file_path,
                    runtime_command=request.runtime_command,
                    runtime_shell=request.runtime_shell,
                )
            )

        first_failure = next((item for item in validation_results if not item.ok), None)
        if first_failure is not None:
            rollback_result = self.rollback_manager.restore_checkpoint(checkpoint.checkpoint_id)
            return EditResult(
                success=False,
                file_path=request.file_path,
                content=original_snapshot.content,
                before_hash=original_snapshot.file_hash,
                after_hash=original_snapshot.file_hash,
                preview=preview,
                validation_results=validation_results,
                checkpoint_id=checkpoint.checkpoint_id,
                rollback_performed=rollback_result.success,
                error_code=first_failure.error_code,
            )

        file_path = Path(request.file_path)
        tmp_path: Path | None = None
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=file_path.parent,
                prefix=f".{file_path.name}.",
                delete=False,
            ) as tmp:
                tmp.write(updated_content)
                tmp_path = Path(tmp.name)
            os.replace(tmp_path, file_path)
        except OSError as exc:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
            self.rollback_manager.restore_checkpoint(checkpoint.checkpoint_id)
            return EditResult(
                success=False,
                file_path=request.file_path,
                content=original_snapshot.content,
                before_hash=original_snapshot.file_hash,
                after_hash=original_snapshot.file_hash,
                checkpoint_id=checkpoint.checkpoint_id,
                rollback_performed=True,
                error_code="ED2001",
            )
        after_hash = compute_file_hash(updated_content)
        self.file_state_cache.read(request.file_path)
        return EditResult(
            success=True,
            file_path=request.file_path,
            content=updated_content,
            before_hash=original_snapshot.file_hash,
            after_hash=after_hash,
            preview=preview,
            validation_results=validation_results,
            checkpoint_id=checkpoint.checkpoint_id,
        )

    def apply_llm_edit(
        self,
        *,
        instruction: str,
        current_code: str,
        llm_client: Any,
        prompt_language: str,
        facade_context: dict[str, Any] | None = None,
        backend: str | None = None,
    ) -> EditResult:
        """Apply an instruction-driven LLM edit to ``current_code``.

        ``backend`` selects the underlying engine (``"line"``, ``"robust"``,
        ``"smart_v2"``, ``"lnfree"``). Defaults to ``self.default_llm_backend``.
        """
        del prompt_language, facade_context
        backend_name = backend or self.default_llm_backend
        editor = _instantiate_llm_backend(backend_name, llm_client)
        llm_result = editor.edit_with_llm(current_code, instruction)
        if not getattr(llm_result, "success", False):
            return EditResult(
                success=False,
                file_path="",
                content=current_code,
                before_hash=compute_file_hash(current_code),
                after_hash=compute_file_hash(current_code),
                error_code="ED1006",
            )
        new_code = getattr(llm_result, "new_code", current_code)
        preview = self._build_patch_preview(
            file_path="",
            before_content=current_code,
            after_content=new_code,
        )
        return EditResult(
            success=True,
            file_path="",
            content=new_code,
            before_hash=compute_file_hash(current_code),
            after_hash=compute_file_hash(new_code),
            preview=preview,
        )

    def rollback(self, checkpoint_id: str) -> RollbackResult:
        return self.rollback_manager.restore_checkpoint(checkpoint_id)

    def _build_updated_content(self, request: FileEditRequest, original_content: str, original_exists: bool) -> str:
        if not original_exists and request.create_if_missing:
            return request.new_string
        if not request.old_string:
            return request.new_string
        match_count = original_content.count(request.old_string)
        if match_count == 0:
            raise CCError("old_string did not match the file content.", error_code="ED1002")
        if match_count > 1 and not request.replace_all:
            raise CCError(
                f"old_string matched {match_count} times; set replace_all to proceed.",
                error_code="ED1003",
            )
        if request.replace_all:
            return original_content.replace(request.old_string, request.new_string)
        return original_content.replace(request.old_string, request.new_string, 1)

    def _build_patch_preview(self, *, file_path: str, before_content: str, after_content: str) -> PatchPreview:
        diff = "\n".join(
            difflib.unified_diff(
                before_content.splitlines(),
                after_content.splitlines(),
                fromfile=f"{file_path}:before",
                tofile=f"{file_path}:after",
                lineterm="",
            )
        )
        before_lines = before_content.splitlines()
        after_lines = after_content.splitlines()
        line_count_changed = abs(len(after_lines) - len(before_lines))
        return PatchPreview(
            file_path=file_path,
            line_count_changed=line_count_changed,
            before_hash=compute_file_hash(before_content),
            after_hash=compute_file_hash(after_content),
            diff=diff,
        )

    def _failure_result(
        self,
        request: FileEditRequest,
        before_hash: str,
        failure: EditValidationResult,
    ) -> EditResult:
        return EditResult(
            success=False,
            file_path=request.file_path,
            content="",
            before_hash=before_hash,
            after_hash=before_hash,
            validation_results=[failure],
            error_code=failure.error_code,
        )
