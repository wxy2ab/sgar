"""Multi-root skill subsystem for the cc / ccx runtime.

Public API:

* :class:`SkillDefinition` — a loaded skill (data).
* :class:`SkillRegistry`   — name→skill map; ``register`` is the dynamic API.
* :func:`load_skill_registry` — discover skills from the three roots.
* :func:`skill_roots`        — the ``(root, source)`` pairs in precedence order.

Lives under ``core.cc`` (not ``core.ccx``) because the tool registry and
QueryEngine it plugs into live in cc, and cc must not import ccx. The ccx modes
drive cc's QueryEngine, so wiring the ``skill`` tool into cc covers ccx too, and
this package travels in the ccx export closure. It is re-exported from
``core.ccx`` for API ergonomics.
"""

from __future__ import annotations

from .loader import discover_skills, load_skill_registry, skill_roots
from .registry import SkillRegistry
from .types import SkillDefinition

__all__ = [
    "SkillDefinition",
    "SkillRegistry",
    "discover_skills",
    "load_skill_registry",
    "skill_roots",
]
