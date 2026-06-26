from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from typing import Any


@dataclass(slots=True)
class PermissionDecision:
    status: Literal["allow", "deny", "ask"]
    reason: str
    source: str
    context_snapshot: dict[str, Any] | None = None

    @property
    def is_terminal_denial(self) -> bool:
        return self.status == "deny"
