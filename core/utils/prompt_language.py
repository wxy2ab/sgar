from __future__ import annotations

import os
from typing import Iterable, List, Optional


DEFAULT_PROMPT_LANGUAGE = "zh"
SUPPORTED_PROMPT_LANGUAGES = ("zh", "en")

_PROMPT_LANGUAGE_ALIASES = {
    "zh": "zh",
    "cn": "zh",
    "zh-cn": "zh",
    "zh-hans": "zh",
    "zh-hant": "zh",
    "chs": "zh",
    "cht": "zh",
    "chinese": "zh",
    "中文": "zh",
    "en": "en",
    "en-us": "en",
    "en-gb": "en",
    "english": "en",
    "英文": "en",
}


def normalize_prompt_language(value: Optional[str]) -> str:
    """Normalize a prompt language option to ``zh`` or ``en``."""
    if value is None:
        return DEFAULT_PROMPT_LANGUAGE

    text = str(value).strip().lower().replace("_", "-")
    if not text:
        return DEFAULT_PROMPT_LANGUAGE

    return _PROMPT_LANGUAGE_ALIASES.get(text, DEFAULT_PROMPT_LANGUAGE)


def choose_prompt_text(prompt_language: Optional[str], *, zh: str, en: str) -> str:
    """Return the language-specific text block."""
    return en if normalize_prompt_language(prompt_language) == "en" else zh


def build_language_variant_candidates(filename: str, prompt_language: Optional[str]) -> List[str]:
    """Return candidate filenames ordered by language-specific fallback."""
    language = normalize_prompt_language(prompt_language)
    stem, ext = os.path.splitext(filename)
    if not ext:
        ext = ".md"
    base = f"{stem}{ext}"
    localized = f"{stem}.{language}{ext}"
    if localized == base:
        return [base]
    return [localized, base]


def find_language_variant_path(
    directory: str,
    filename: str,
    prompt_language: Optional[str],
) -> Optional[str]:
    """Resolve the best prompt file for the requested language."""
    for candidate in build_language_variant_candidates(filename, prompt_language):
        path = os.path.join(directory, candidate)
        if os.path.exists(path):
            return path
    return None


def iter_unique_strings(items: Iterable[str]) -> List[str]:
    """Return a stable de-duplicated list of strings."""
    seen = set()
    result: List[str] = []
    for item in items:
        text = str(item)
        if text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result
