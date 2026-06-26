from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from core.utils.prompt_language import normalize_prompt_language

from ..errors import PromptLanguageError, PromptNotFoundError


@dataclass(slots=True)
class PromptAsset:
    key: str
    zh: str
    en: str
    source: str = "built-in"
    tags: list[str] = field(default_factory=list)

    def resolve(self, language: str) -> str:
        normalized = normalize_prompt_language(language)
        if normalized == "en" and self.en.strip():
            return self.en
        if self.zh.strip():
            return self.zh
        if self.en.strip():
            return self.en
        raise PromptNotFoundError(
            f"Prompt asset {self.key!r} has no content for any language.",
            error_code="CF1005",
        )


class PromptLocaleResolver:
    def normalize(self, language: str | None) -> str:
        normalized = normalize_prompt_language(language)
        if normalized not in {"zh", "en"}:
            raise PromptLanguageError(f"Unsupported prompt language: {language!r}")
        return normalized


class PromptCatalog:
    def __init__(self, resolver: PromptLocaleResolver | None = None) -> None:
        self._resolver = resolver or PromptLocaleResolver()
        self._assets: dict[str, PromptAsset] = {}

    def register(self, asset: PromptAsset) -> None:
        self._assets[asset.key] = asset

    def register_many(self, assets: Iterable[PromptAsset]) -> None:
        for asset in assets:
            self.register(asset)

    def has_language(self, key: str, language: str) -> bool:
        asset = self._assets.get(key)
        if not asset:
            return False
        normalized = self._resolver.normalize(language)
        return bool(asset.en.strip()) if normalized == "en" else bool(asset.zh.strip())

    def resolve(self, key: str, language: str | None) -> str:
        asset = self._assets.get(key)
        if asset is None:
            raise PromptNotFoundError(f"Prompt asset not found: {key}", error_code="CF1006")
        return asset.resolve(self._resolver.normalize(language))

    def get(self, key: str) -> PromptAsset | None:
        return self._assets.get(key)

    def keys(self) -> list[str]:
        return sorted(self._assets)

    @classmethod
    def from_prompt_root(cls, prompt_root: str | Path) -> "PromptCatalog":
        catalog = cls()
        root = Path(prompt_root)
        for language in ("zh", "en"):
            for path in root.rglob(f"*.{language}.md"):
                relative = path.relative_to(root).with_suffix("")
                last = relative.parts[-1].rsplit(".", 1)[0]
                key = ".".join(relative.parts[:-1] + (last,))
                existing = catalog.get(key)
                text = path.read_text(encoding="utf-8")
                if existing:
                    if language == "zh":
                        existing.zh = text
                    else:
                        existing.en = text
                    continue
                catalog.register(
                    PromptAsset(
                        key=key,
                        zh=text if language == "zh" else "",
                        en=text if language == "en" else "",
                        source=str(path),
                    )
                )
        return catalog
