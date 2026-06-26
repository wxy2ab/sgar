import configparser
import os
from pathlib import Path
from typing import Iterable, Optional

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        key = cls
        if hasattr(cls, "_singleton_key"):
            key = (cls, cls._singleton_key(*args, **kwargs))
        if key not in cls._instances:
            cls._instances[key] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[key]

class Config(metaclass=Singleton):
    DEFAULT_PATH = "setting.ini"

    @classmethod
    def _singleton_key(cls, path="setting.ini"):
        return str(path or cls.DEFAULT_PATH)

    def __init__(self, path="setting.ini"):
        self.requested_path = path or self.DEFAULT_PATH
        self.config = configparser.ConfigParser()
        self.file_name = str(self._preferred_write_path())
        self.loaded_path: Optional[Path] = self._resolve_existing_path()

        if self.loaded_path is not None:
            self.config.read(self.loaded_path, encoding='utf-8')
            self.file_name = str(self.loaded_path)
            self.use_file = True
        else:
            self.use_file = False

    @classmethod
    def _project_root(cls) -> Path:
        return Path(__file__).resolve().parents[2]

    @classmethod
    def _root_dir_name(cls) -> str:
        return cls._project_root().name

    @classmethod
    def _user_config_path(cls) -> Path:
        return Path.home() / f".{cls._root_dir_name()}" / cls.DEFAULT_PATH

    def _requested_path_obj(self) -> Path:
        return Path(self.requested_path).expanduser()

    def _default_search_paths(self) -> list[Path]:
        return [Path(self.DEFAULT_PATH), self._user_config_path()]

    def _candidate_paths(self) -> list[Path]:
        requested = self._requested_path_obj()
        candidates: list[Path] = [requested]
        for fallback in self._default_search_paths():
            if fallback not in candidates:
                candidates.append(fallback)
        return candidates

    def _resolve_existing_path(self) -> Optional[Path]:
        for candidate in self._candidate_paths():
            if candidate.exists():
                return candidate
        return None

    def _preferred_write_path(self) -> Path:
        requested = self._requested_path_obj()
        if requested != Path(self.DEFAULT_PATH):
            return requested
        return Path(self.DEFAULT_PATH)

    @staticmethod
    def _normalize_value(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        return normalized if normalized != "" else None

    def _get_from_env(self, key: str) -> Optional[str]:
        return self._normalize_value(os.environ.get(key.upper()))

    def _get_from_file(self, key: str, section: str = "Default") -> Optional[str]:
        if not self.use_file or not self.config.has_section(section):
            return None
        return self._normalize_value(self.config[section].get(key))

    def get(self, key: str = "token", /, section: str = "Default"):
        env_value = self._get_from_env(key)
        if env_value is not None:
            return env_value
        return self._get_from_file(key, section)

    def get_with_fallback(
        self,
        keys: Iterable[str],
        default: Optional[str] = None,
        /,
        section: str = "Default",
    ) -> Optional[str]:
        for key in keys:
            value = self.get(key, section=section)
            if value is not None:
                return value
        return default

    def resolve_value(
        self,
        explicit_value: Optional[str],
        keys: Iterable[str],
        default: Optional[str] = None,
        /,
        section: str = "Default",
    ) -> Optional[str]:
        explicit = self._normalize_value(explicit_value)
        if explicit is not None:
            return explicit
        return self.get_with_fallback(keys, default, section=section)

    def has_key(self, key: str = "token", /, section: str = "Default"):
        return self.get(key, section=section) is not None

    def set(self, key: str = "token", value: str = "", /, section: str = "Default"):
        target_path = Path(self.file_name).expanduser()
        if not target_path.parent.exists() and target_path.parent != Path("."):
            target_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.config.has_section(section):
            self.config.add_section(section)
        self.config[section][key] = value
        with open(target_path, "w", encoding="utf-8") as configfile:
            self.config.write(configfile)
        self.file_name = str(target_path)
        self.loaded_path = target_path
        self.use_file = True
