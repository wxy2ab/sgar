import importlib
from pathlib import Path

from core.utils.config_setting import Config


def test_user_config_path_does_not_derive_from_package_root(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(
        Config,
        "_project_root",
        classmethod(lambda cls: Path("/tmp/site-packages")),
    )

    assert Config._user_config_path() == tmp_path / ".sgar" / "setting.ini"


def test_config_cli_uses_runtime_user_config_path(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))

    import sgar.config_cli as config_cli

    config_cli = importlib.reload(config_cli)

    assert config_cli.USER_CONFIG_PATH == Config._user_config_path()
