from core.ccx.sgar.store import SgarStore


def test_initialize_appends_sgar_entries_to_existing_gitignore(tmp_path):
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("__pycache__/\n", encoding="utf-8")

    store = SgarStore(tmp_path)
    store.initialize(project_name="demo")

    assert gitignore.read_text(encoding="utf-8") == (
        "__pycache__/\n"
        ".sgar/\n"
        ".sgarx/\n"
    )


def test_initialize_does_not_duplicate_gitignore_entries(tmp_path):
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text(".sgar/\nnode_modules/\n.sgarx/\n", encoding="utf-8")

    store = SgarStore(tmp_path)
    store.initialize(project_name="demo")

    assert gitignore.read_text(encoding="utf-8") == ".sgar/\nnode_modules/\n.sgarx/\n"


def test_initialize_does_not_create_gitignore_when_absent(tmp_path):
    store = SgarStore(tmp_path)
    store.initialize(project_name="demo")

    assert not (tmp_path / ".gitignore").exists()
