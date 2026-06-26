"""Bidirectional fallback adapter between block-style LLM editors.

The adapter chains two backends so that a transient failure in one engine
(usually parsing/locator drift) is masked by the other. Both ``edit_with_llm``
and ``apply_instruction_string`` apply the same fallback strategy.

The block-number-based :class:`LLMBlockEditor` is deprecated, so ``"lnfree"``
is the recommended primary backend.
"""

from __future__ import annotations

from typing import Any, Optional


class FallbackLLMEditor:
    def __init__(self, llm_client: Any, prefer: str = "lnfree", config: Optional[object] = None):
        if prefer not in ("lnfree", "block"):
            raise ValueError(f"prefer must be 'lnfree' or 'block', got {prefer!r}")
        self.llm_client = llm_client
        self.prefer = prefer
        self._config = config
        self._lnfree = None
        self._block = None

    # -- lazy backend factories ---------------------------------------------

    def _get_lnfree(self):
        if self._lnfree is None:
            from core.utils.llm_block_editor_lnfree import (
                EditorConfig,
                LineNumberFreeLLMBlockEditor,
            )
            self._lnfree = LineNumberFreeLLMBlockEditor(
                llm_client=self.llm_client,
                config=self._config or EditorConfig(),
            )
        return self._lnfree

    def _get_block(self):
        if self._block is None:
            from core.utils.llm_block_editor import LLMBlockEditor
            self._block = LLMBlockEditor(llm_client=self.llm_client)
        return self._block

    def _ordered_backends(self):
        if self.prefer == "lnfree":
            return self._get_lnfree, self._get_block
        return self._get_block, self._get_lnfree

    # -- public API ----------------------------------------------------------

    def edit_with_llm(
        self,
        original_code: str,
        instruction: str,
        context: str = "",
        file_path: str = "",
    ):
        primary_factory, secondary_factory = self._ordered_backends()
        primary = primary_factory().edit_with_llm(
            original_code, instruction, context, file_path
        )
        if getattr(primary, "success", False):
            return primary
        secondary = secondary_factory().edit_with_llm(
            original_code, instruction, context, file_path
        )
        return secondary if getattr(secondary, "success", False) else primary

    def apply_instruction_string(self, original_code: str, instruction_string: str):
        primary_factory, secondary_factory = self._ordered_backends()
        primary = primary_factory().apply_instruction_string(
            original_code, instruction_string
        )
        if getattr(primary, "success", False):
            return primary
        try:
            secondary = secondary_factory().apply_instruction_string(
                original_code, instruction_string
            )
        except (AttributeError, NotImplementedError):
            return primary
        return secondary if getattr(secondary, "success", False) else primary
