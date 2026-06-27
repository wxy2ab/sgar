"""Skill discovery across three roots, with precedence.

Roots (low → high precedence):

1. ``<cwd>/skills``           → ``"repo"``     — committed/shared project skills
   (the existing repo convention: ``skills/<name>/SKILL.md``).
2. ``~/.<root>/skills``        → ``"user"``     — user-global skills. ``<root>`` is
   the same directory name ``setting.ini`` keys off, so this stays in lockstep
   with the settings location and tracks the export dir name after ccx is
   exported to sgar (``~/.llm_dealer/skills`` here, ``~/.sgar/skills`` there).
3. ``<cwd>/.skills``           → ``"project"``  — local (typically gitignored)
   override; wins over the shared and user roots.

A skill is a ``SKILL.md`` (anywhere under a root) or a flat top-level ``*.md``
file, carrying optional YAML frontmatter (``name:`` / ``description:``). On a
name collision a higher-precedence root replaces a lower one; within a single
root the lexicographically-first path wins (deterministic). Missing roots are
skipped — never created — and a single unreadable / malformed file is skipped
rather than breaking the whole registry.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from core.utils.config_setting import Config

from .registry import SkillRegistry
from .types import SkillDefinition


def skill_roots(cwd: str | Path | None = None) -> list[tuple[Path, str]]:
    """Return ``(root, source)`` pairs in ascending precedence order.

    The user root reuses :meth:`Config._user_config_path` so it resolves to the
    exact same ``~/.<root>/`` directory that holds ``setting.ini`` — keeping the
    two consistent by construction (the user's stated requirement).
    """
    base = Path(cwd) if cwd is not None else Path.cwd()
    user_root = Config._user_config_path().parent / "skills"
    return [
        (base / "skills", "repo"),
        (user_root, "user"),
        (base / ".skills", "project"),
    ]


def _parse_frontmatter(default_name: str, content: str) -> tuple[str, str]:
    """Extract ``(name, description)`` from a skill markdown body.

    Parses a leading ``---`` … ``---`` YAML frontmatter block with
    ``yaml.safe_load`` (so folded scalars like ``description: >`` spanning
    multiple lines are read correctly). Falls back to the default name and the
    first non-heading line when there is no usable frontmatter.
    """
    name = default_name
    description = ""

    stripped = content.lstrip()
    if stripped.startswith("---"):
        # Body after the opening fence; close on the next line that is exactly '---'.
        rest = stripped[3:]
        end = rest.find("\n---")
        if end != -1:
            block = rest[:end]
            try:
                meta = yaml.safe_load(block)
            except yaml.YAMLError:
                meta = None
            if isinstance(meta, dict):
                raw_name = meta.get("name")
                if raw_name:
                    name = str(raw_name).strip() or default_name
                raw_desc = meta.get("description")
                if raw_desc:
                    description = " ".join(str(raw_desc).split())

    if not description:
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("---"):
                continue
            description = line[:200]
            break

    if not description:
        description = f"Skill: {name}"
    return name, description


def _read_skill(path: Path, source: str, *, default_name: str, base_dir: Path) -> SkillDefinition | None:
    """Build a :class:`SkillDefinition` from one markdown file, or ``None`` on error."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    name, description = _parse_frontmatter(default_name, content)
    return SkillDefinition(
        name=name,
        description=description,
        content=content,
        source=source,
        path=str(path),
        base_dir=str(base_dir),
    )


def discover_skills(root: Path, source: str) -> list[SkillDefinition]:
    """Discover skills under a single root (empty list if the root is absent).

    Within the root, ``SKILL.md`` files (at any depth) and flat top-level
    ``*.md`` files are collected; duplicate names are resolved by sorted path so
    the result is deterministic regardless of filesystem ordering.
    """
    try:
        if not root.is_dir():
            return []
    except OSError:
        return []

    candidates: list[tuple[Path, str, Path]] = []  # (file, default_name, base_dir)
    try:
        for md in root.rglob("SKILL.md"):
            candidates.append((md, md.parent.name or source, md.parent))
        for md in root.glob("*.md"):
            if md.name == "SKILL.md":  # already covered by rglob
                continue
            candidates.append((md, md.stem, root))
    except OSError:
        return []

    skills: dict[str, SkillDefinition] = {}
    for path, default_name, base_dir in sorted(candidates, key=lambda item: str(item[0])):
        skill = _read_skill(path, source, default_name=default_name, base_dir=base_dir)
        if skill is None:
            continue
        skills.setdefault(skill.name, skill)  # first sorted path wins within a root
    return list(skills.values())


def load_skill_registry(cwd: str | Path | None = None) -> SkillRegistry:
    """Build a registry from all three roots, applying precedence.

    Roots are loaded lowest→highest precedence with ``override=True`` so a skill
    in a higher root replaces a same-named skill from a lower one.
    """
    registry = SkillRegistry()
    for root, source in skill_roots(cwd):
        for skill in discover_skills(root, source):
            registry.register(skill, override=True)
    return registry
