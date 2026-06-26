from __future__ import annotations

from pathlib import Path


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def normalize_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def resolve_under_cwd(value: str | Path, cwd: str | Path) -> Path:
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = Path(cwd) / p
    return p.resolve()


def is_unc_path(path: Path) -> bool:
    text = str(path)
    return text.startswith("\\\\") or text.startswith("//")


def path_matches_any(path: Path, roots: list[str | Path]) -> bool:
    normalized_path = normalize_path(path)
    return any(is_relative_to(normalized_path, normalize_path(root)) for root in roots)
