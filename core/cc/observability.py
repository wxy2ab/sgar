from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
from typing import Any

from .jsonl import append_jsonl_many_sync, append_jsonl_sync


@dataclass(slots=True)
class EventRecord:
    event_type: str
    timestamp: float = field(default_factory=time.time)
    session_id: str | None = None
    turn_id: str | None = None
    task_id: str | None = None
    tool_name: str | None = None
    success: bool | None = None
    error_code: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class JsonlAuditLogger:
    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: EventRecord) -> None:
        append_jsonl_sync(self.file_path, _json_safe(asdict(record)))

    def append_many(self, records: list[EventRecord]) -> None:
        if not records:
            return
        append_jsonl_many_sync(self.file_path, [_json_safe(asdict(record)) for record in records])


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if callable(value):
        name = getattr(value, "__name__", value.__class__.__name__)
        return f"<callable:{name}>"
    return repr(value)
