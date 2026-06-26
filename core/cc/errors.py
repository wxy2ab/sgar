from __future__ import annotations


class CCError(Exception):
    """Base exception for the cc package."""

    error_code = "CC1000"

    def __init__(self, message: str, *, error_code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if error_code is not None:
            self.error_code = error_code

    @property
    def code(self) -> str:
        """Backward-compatible alias for older call sites."""
        return str(self.error_code)


class ConfigError(CCError):
    error_code = "CF1001"


class PromptNotFoundError(CCError):
    error_code = "CF1006"


class PromptLanguageError(CCError):
    error_code = "CF1002"


class SessionPersistenceError(CCError):
    error_code = "QE1006"


class ToolValidationError(CCError):
    error_code = "TL1002"


class ToolExecutionError(CCError):
    error_code = "TL1006"


class AgentTaskError(CCError):
    error_code = "AG1001"
