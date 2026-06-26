from __future__ import annotations

from pathlib import Path
from typing import Any

from .strategy_common import PATH_RE, contains_target_path, extract_path_tokens


_OUTLINE_HINTS = (
    "architecture",
    "architectural",
    "codebase",
    "component",
    "design",
    "doc",
    "docs",
    "documentation",
    "flow",
    "layout",
    "module",
    "modules",
    "overview",
    "repo",
    "repository",
    "structure",
    "tree",
    "where is",
    "workspace",
    "调用链",
    "在哪",
    "代码库",
    "仓库",
    "关系",
    "子模块",
    "实现在哪里",
    "总览",
    "文档",
    "整体",
    "架构",
    "模块",
    "流程",
    "目录",
    "组织",
    "结构",
    "说明文档",
)
_TARGETED_HINTS = (
    "class ",
    "function",
    "method",
    "single file",
    "函数",
    "单个文件",
    "某个文件",
    "某个函数",
    "某个类",
    "类 ",
)
_IGNORED_NAMES = {
    ".git",
    ".hg",
    ".idea",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".svn",
    ".trae",
    ".venv",
    "__pycache__",
    "env",
    "node_modules",
}
_DEFAULT_INCLUDE_SUFFIXES = {
    ".py",
    ".md",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
    ".ini",
}


def decide_mode_strategy(agent_mode: str, user_input: str | list[dict[str, object]]) -> dict[str, Any]:
    text = user_input if isinstance(user_input, str) else str(user_input)
    lowered = text.lower()
    has_outline_hint = any(hint in lowered for hint in _OUTLINE_HINTS)
    has_targeted_hint = any(hint in lowered for hint in _TARGETED_HINTS) or contains_target_path(text)

    use_repository_outline = False
    reason = "not_applicable"
    if agent_mode == "ask":
        if has_outline_hint:
            use_repository_outline = True
            reason = "ask_outline_keywords"
        elif "where" in lowered and not has_targeted_hint:
            use_repository_outline = True
            reason = "ask_location_question"
        elif "哪里" in text and not has_targeted_hint:
            use_repository_outline = True
            reason = "ask_location_question"
    elif agent_mode == "doc":
        if has_targeted_hint and not has_outline_hint:
            reason = "doc_targeted_request"
        else:
            use_repository_outline = True
            reason = "doc_structure_first"

    detail_level = "high" if agent_mode == "doc" else "medium"
    if not use_repository_outline:
        detail_level = "minimal"
    return {
        "mode": agent_mode,
        "use_repository_outline": use_repository_outline,
        "reason": reason,
        "detail_level": detail_level,
        "has_outline_hint": has_outline_hint,
        "has_targeted_hint": has_targeted_hint,
    }


def build_repository_outline(
    cwd: str | Path,
    *,
    max_depth: int = 3,
    max_entries_per_dir: int = 6,
    include_patterns: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    root = Path(cwd).resolve()
    suffixes = {item.lower() for item in include_patterns or ()}
    if not suffixes:
        suffixes = set(_DEFAULT_INCLUDE_SUFFIXES)

    lines = [f"{root.name}/"]
    _append_tree_lines(
        path=root,
        lines=lines,
        depth=1,
        max_depth=max_depth,
        max_entries_per_dir=max_entries_per_dir,
        suffixes=suffixes,
    )
    return {
        "root": str(root),
        "max_depth": max_depth,
        "max_entries_per_dir": max_entries_per_dir,
        "text": "\n".join(lines),
    }


def _append_tree_lines(
    *,
    path: Path,
    lines: list[str],
    depth: int,
    max_depth: int,
    max_entries_per_dir: int,
    suffixes: set[str],
) -> None:
    if depth > max_depth or not path.is_dir():
        return
    entries = _filtered_entries(path, suffixes=suffixes)
    visible = entries[:max_entries_per_dir]
    omitted = len(entries) - len(visible)
    prefix = "  " * depth

    for entry in visible:
        label = f"{entry.name}/" if entry.is_dir() else entry.name
        lines.append(f"{prefix}- {label}")
        if entry.is_dir():
            _append_tree_lines(
                path=entry,
                lines=lines,
                depth=depth + 1,
                max_depth=max_depth,
                max_entries_per_dir=max_entries_per_dir,
                suffixes=suffixes,
            )
    if omitted > 0:
        lines.append(f"{prefix}- ... ({omitted} more entries)")


def _filtered_entries(path: Path, *, suffixes: set[str]) -> list[Path]:
    directories: list[Path] = []
    files: list[Path] = []
    for entry in sorted(path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
        if entry.name in _IGNORED_NAMES or entry.name.startswith("."):
            continue
        if entry.is_dir():
            directories.append(entry)
            continue
        if entry.suffix.lower() in suffixes:
            files.append(entry)
    return directories + files


# --------------------------------------------------------------------------- #
# Paths-in-request block
# --------------------------------------------------------------------------- #
#
# ``build_repository_outline`` truncates to a few entries per directory.
# A user prompt like "审阅 core/deepstack-agent/stock_rec_v3 ..." can
# easily reference a path the truncated outline omits, leading the LLM
# to falsely conclude "directory not in outline => doesn't exist". The
# block produced here surfaces the user-named paths separately, marks
# them ``[verified]`` / ``[missing]`` against the workspace, and (for
# verified directories) embeds a focused subtree so the LLM has the
# structural context it would otherwise need to glob for.

_FOCUSED_SUBTREE_DEPTH = 3
_FOCUSED_SUBTREE_ENTRIES = 12


def build_paths_in_request_block(
    user_input: str | list[dict[str, object]],
    cwd: str | Path,
    *,
    focused_depth: int = _FOCUSED_SUBTREE_DEPTH,
    focused_entries_per_dir: int = _FOCUSED_SUBTREE_ENTRIES,
) -> str:
    """Render a ``# Paths in this task`` Markdown block, or ``""``.

    Args:
        user_input: Raw text or message-list shape used by the prompt
            pipeline (we coerce to text).
        cwd: Workspace root the paths are interpreted relative to.
        focused_depth / focused_entries_per_dir: Controls the focused
            subtree sample appended for each verified directory. The
            defaults match what ccx does.

    Returns the empty string when no path tokens were found, so callers
    can append unconditionally.
    """
    text = user_input if isinstance(user_input, str) else str(user_input)
    tokens = extract_path_tokens(text)
    if not tokens:
        return ""
    cwd_path = Path(cwd).resolve() if cwd else None
    lines: list[str] = []
    focused_chunks: list[str] = []
    for tok in tokens:
        status = "[?]"
        target: Path | None = None
        if cwd_path is not None:
            cand = (cwd_path / tok) if not Path(tok).is_absolute() else Path(tok)
            try:
                resolved = cand.resolve()
            except OSError:
                resolved = cand
            if resolved.exists():
                status = "[verified]"
                target = resolved
            else:
                status = "[missing in cwd — verify with `glob`]"
        lines.append(f"- {status} `{tok}`")
        if target is not None and target.is_dir():
            try:
                outline = build_repository_outline(
                    target,
                    max_depth=focused_depth,
                    max_entries_per_dir=focused_entries_per_dir,
                )
            except Exception:  # noqa: BLE001
                outline = {"text": ""}
            sub_text = str(outline.get("text") or "")
            if sub_text:
                focused_chunks.append(
                    f"### Focused subtree: `{tok}`\n```\n{sub_text}\n```"
                )
    body_parts = [
        "Paths the user named in this turn. They are real even if "
        "the truncated repository outline below does not list them — "
        "verify with `glob \"<path>/**/*\"` if you need to enumerate.",
        "",
        *lines,
    ]
    if focused_chunks:
        body_parts.append("")
        body_parts.extend(focused_chunks)
    return "\n".join(body_parts)
