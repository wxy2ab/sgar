"""Skill registry — an in-memory name→skill map.

The registry IS the unified, dynamic skill-access API: anything in-process can
``reg = load_skill_registry(cwd); reg.register(SkillDefinition(...))`` to add a
skill at runtime (no plugin framework, no entry points). Disk discovery
(:func:`core.cc.skills.loader.load_skill_registry`) is just one producer that
calls :meth:`register`; precedence across roots is enforced by the *order* of
those calls (lowest-precedence root first, highest last, last-writer-wins).
"""

from __future__ import annotations

from .types import SkillDefinition


class SkillRegistry:
    """Store loaded skills by name."""

    def __init__(self) -> None:
        self._skills: dict[str, SkillDefinition] = {}

    def register(self, skill: SkillDefinition, *, override: bool = True) -> None:
        """Register one skill.

        ``override`` (default ``True``) makes a later registration win on name
        collision — this is how cross-root precedence is applied by the loader.
        Pass ``override=False`` from a programmatic caller that wants to add a
        skill only if the name is still free.
        """
        if not override and skill.name in self._skills:
            return
        self._skills[skill.name] = skill

    def get(self, name: str) -> SkillDefinition | None:
        """Return a skill by exact name, else ``None``."""
        return self._skills.get(name)

    def list_skills(self) -> list[SkillDefinition]:
        """Return all skills sorted by name."""
        return sorted(self._skills.values(), key=lambda skill: skill.name)

    def names(self) -> list[str]:
        """Return all registered skill names, sorted."""
        return sorted(self._skills)

    def __len__(self) -> int:
        return len(self._skills)

    def __bool__(self) -> bool:
        return bool(self._skills)
