"""Skill data model.

A *skill* is a markdown document (``SKILL.md`` with YAML frontmatter) that
carries prompt-facing instructions the model can pull on demand. Skills are
discovered from several roots and stored in a :class:`SkillRegistry`; the
``skill`` tool returns a skill's body to the model.

Kept deliberately small (a frozen dataclass — pure data, no behaviour) so the
whole subsystem travels cleanly in the ccx export closure without dragging in
extra dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SkillDefinition:
    """A loaded skill.

    ``source`` records which root the skill came from (``"repo"`` / ``"user"`` /
    ``"project"``) and doubles as the precedence label. ``base_dir`` is the
    directory holding the ``SKILL.md`` so the model can read bundled assets
    (scripts, templates) that sit beside it.
    """

    name: str
    description: str
    content: str
    source: str
    path: str | None = None
    base_dir: str | None = None
