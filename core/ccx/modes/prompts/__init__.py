"""Externalised mode prompts — TOML files loaded once and cached.

Borrowed from codex's pattern of declaring each builtin agent in a
``.toml`` file (``codex-rs/core/src/agent/builtins/{awaiter,explorer}.toml``).
Before C1, every ccx mode hard-coded its system and user prompt as
module-level string constants, which had two problems:

1. A prompt tweak demanded a Python edit, code review, and rerun of
   the entire test suite even when the change was purely textual.
2. Prompts written by a model engineer (not a code engineer) bled into
   ``.py`` review noise. A/B testing across prompt variants required
   either parallel branches or runtime conditionals.

Moving the prompts to TOML solves both. ``<mode>.toml`` lives next to
this loader. Multi-language variants nest under ``[system.<lang>]``
so future per-language metadata (version, author, last-updated) has a
natural home. ``[user_template].format`` is a Python ``.format()``
string — placeholders like ``{goal}`` are filled by the caller.

Loader contract:

- ``load_mode_prompts(mode_name)`` returns a ``ModePrompts`` dataclass.
- Result is cached forever after first load (start-up cost only).
- Missing files raise ``MissingPromptFileError`` so each mode runner
  can fall back to its compiled-in defaults (kept around during the
  incremental migration window — see ``modes/plan.py`` for the
  pattern).
- Malformed TOML or missing required sections raise
  ``InvalidPromptFileError`` with the offending key path in the
  message so a typo is immediately obvious.

The loader does NOT compile or pre-render ``user_template`` — the
caller does ``prompts.user_template.format(...)`` at call time so
parameters can vary per invocation. Loader returns the raw string.
"""

from __future__ import annotations

import logging
import threading
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PROMPT_DIR = Path(__file__).resolve().parent


class PromptLoadError(Exception):
    """Base for any failure loading a mode prompt TOML."""


class MissingPromptFileError(PromptLoadError):
    """The TOML for a requested mode does not exist on disk."""


class InvalidPromptFileError(PromptLoadError):
    """The TOML parsed but is missing a required key or has the
    wrong shape."""


@dataclass(frozen=True, slots=True)
class ModePrompts:
    """All prompt strings for one mode.

    ``system_en`` / ``system_zh`` are full system prompts (required).
    The language is chosen by the caller (typically based on
    ``CCConfig.prompt_language``).

    ``user_template`` is an OPTIONAL Python ``.format()`` template. It
    only makes sense for modes whose user prompt is a single string
    with simple slots like ``{goal}`` — plan mode is the canonical
    case. Modes that assemble the user prompt dynamically (spec /
    agent build it from conditional ``parts``; watch / doc compose it
    from multiple sources) leave ``user_template`` empty and keep the
    assembly logic in ``.py``. An empty string means "no template;
    caller will build the user prompt itself".
    """

    system_en: str
    system_zh: str
    user_template: str = ""

    def system_for(self, language: str) -> str:
        """Return the system prompt for the given language, defaulting
        to English. Matches the runtime convention used across ccx
        mode runners (``language.startswith("zh")`` → zh, else en)."""
        return self.system_zh if language.startswith("zh") else self.system_en


_cache: dict[str, ModePrompts] = {}
_cache_lock = threading.Lock()


def load_mode_prompts(mode_name: str) -> ModePrompts:
    """Load and cache the ``<mode_name>.toml`` from this package.

    Returns the cached ``ModePrompts`` on every subsequent call.
    Raises ``MissingPromptFileError`` if the TOML is absent and
    ``InvalidPromptFileError`` if it parses but lacks
    ``[system.en].text`` / ``[system.zh].text`` /
    ``[user_template].format``.

    Thread-safe: the first caller wins the parse; concurrent callers
    block on the lock and then see the cached value.
    """
    cached = _cache.get(mode_name)
    if cached is not None:
        return cached
    with _cache_lock:
        cached = _cache.get(mode_name)  # re-check after acquiring lock
        if cached is not None:
            return cached
        prompts = _load_from_disk(mode_name)
        _cache[mode_name] = prompts
        return prompts


def _load_from_disk(mode_name: str) -> ModePrompts:
    path = _PROMPT_DIR / f"{mode_name}.toml"
    if not path.is_file():
        raise MissingPromptFileError(
            f"no prompt file for mode {mode_name!r} at {path}"
        )
    try:
        with path.open("rb") as fp:
            data: dict[str, Any] = tomllib.load(fp)
    except tomllib.TOMLDecodeError as exc:
        raise InvalidPromptFileError(
            f"{path}: TOML parse failed: {exc}"
        ) from exc

    try:
        system_en = data["system"]["en"]["text"]
        system_zh = data["system"]["zh"]["text"]
    except (KeyError, TypeError) as exc:
        raise InvalidPromptFileError(
            f"{path}: missing required key (need [system.en].text "
            f"and [system.zh].text): {exc}"
        ) from exc

    # user_template is optional: modes like spec / agent / watch / doc
    # build the user prompt dynamically and leave this section out.
    # When present, the value must be a non-empty string.
    user_template_section = data.get("user_template")
    if user_template_section is None:
        user_template = ""
    elif isinstance(user_template_section, dict):
        user_template = user_template_section.get("format", "")
    else:
        raise InvalidPromptFileError(
            f"{path}: [user_template] must be a table when present"
        )

    for name, value, required in (
        ("system.en.text", system_en, True),
        ("system.zh.text", system_zh, True),
        ("user_template.format", user_template, False),
    ):
        if not isinstance(value, str):
            raise InvalidPromptFileError(
                f"{path}: {name} must be a string"
            )
        if required and not value.strip():
            raise InvalidPromptFileError(
                f"{path}: {name} must be a non-empty string"
            )
        if not required and user_template_section is not None and not value.strip():
            # Section present but format empty — explicit error so the
            # config author doesn't silently get no template.
            raise InvalidPromptFileError(
                f"{path}: {name} present but empty; omit [user_template] "
                f"section entirely to declare no template"
            )

    return ModePrompts(
        system_en=system_en,
        system_zh=system_zh,
        user_template=user_template,
    )


def fallback_system_prompt(
    mode_name: str,
    language: str,
    exc: PromptLoadError,
    *,
    fallback_en: str,
    fallback_zh: str,
    logger: logging.Logger,
) -> str:
    """Resolve a mode's system prompt from compiled-in constants after the
    TOML load failed — the shared fallback shell for the plan / spec / agent
    prompt builders.

    Each builder keeps its own ``try: load_mode_prompts(mode) ...`` (so the
    happy path, and any test that monkeypatches ``load_mode_prompts``, stay in
    the caller's own namespace) and, in its ``except PromptLoadError as exc``
    branch, delegates here. This centralises the two pieces that were
    copy-pasted verbatim across all three builders:

    1. The single WARN — so a deploy missing the data file gets noticed. The
       loader cache means one load attempt per process, so this never spams.
    2. The ``language.startswith("zh")`` constant selection — the same
       convention :meth:`ModePrompts.system_for` uses for the TOML path.

    The *user* prompt is intentionally NOT handled here: plan formats a
    ``{goal}`` template while spec / agent assemble conditional parts, so each
    builder keeps its own user-prompt fallback (the semantics differ).
    """
    logger.warning(
        "%s prompts TOML unavailable (%s); using fallback constants",
        mode_name,
        exc,
    )
    return fallback_zh if language.startswith("zh") else fallback_en


def _clear_cache_for_tests() -> None:
    """Test-only: drop the prompt cache so a fixture can reload a
    mutated TOML file. Production code never calls this.
    """
    with _cache_lock:
        _cache.clear()


__all__ = [
    "InvalidPromptFileError",
    "MissingPromptFileError",
    "ModePrompts",
    "PromptLoadError",
    "fallback_system_prompt",
    "load_mode_prompts",
]
