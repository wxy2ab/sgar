from __future__ import annotations

from types import SimpleNamespace

import sgar.cli as cli


def _result(*, final_text: str = "ok", failed: bool = False):
    return SimpleNamespace(
        final_text=final_text,
        session_snapshot={},
        failed=failed,
        error_code=None,
        error_message=None,
    )


def test_runtime_commands_still_forward_to_core(monkeypatch):
    seen: dict[str, object] = {}

    def _fake_forward(argv: list[str]) -> int:
        seen["argv"] = argv
        return 7

    monkeypatch.setattr(cli, "_forward_to_core", _fake_forward)

    assert cli.main(["status"]) == 7
    assert seen["argv"] == ["status"]


def test_run_command_dispatches_selected_mode_and_metadata(monkeypatch):
    seen: dict[str, object] = {}

    def _fake_run_code_agent(**kwargs):
        seen.update(kwargs)
        return _result(final_text="completed")

    monkeypatch.setattr(cli, "_run_code_agent", _fake_run_code_agent)

    exit_code = cli.main([
        "run",
        "--mode",
        "sgar",
        "--cwd",
        "/tmp/repo",
        "--metadata-json",
        '{"ccx_contract": {"kind": "demo"}}',
        "--docs-output-path",
        "docs/out.md",
        "repair flaky tests",
    ])

    assert exit_code == 0
    assert seen["mode"] == "sgar"
    assert seen["instruction"] == "repair flaky tests"
    assert seen["cwd"] == "/tmp/repo"
    assert seen["metadata"] == {
        "ccx_contract": {"kind": "demo"},
        "docs_output_path": "docs/out.md",
    }


def test_mode_shortcut_uses_expected_mode(monkeypatch):
    seen: dict[str, object] = {}

    def _fake_run_code_agent(**kwargs):
        seen.update(kwargs)
        return _result(final_text="planned")

    monkeypatch.setattr(cli, "_run_code_agent", _fake_run_code_agent)

    exit_code = cli.main(["plan", "design auth migration"])

    assert exit_code == 0
    assert seen["mode"] == "plan"
    assert seen["instruction"] == "design auth migration"
