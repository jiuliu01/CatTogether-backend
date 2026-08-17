from core.workspace_changes import changed_workspace_paths, snapshot_workspace


def test_detects_created_modified_and_deleted_files(tmp_path):
    existing = tmp_path / "existing.txt"
    removed = tmp_path / "removed.txt"
    existing.write_text("before", encoding="utf-8")
    removed.write_text("remove me", encoding="utf-8")
    before = snapshot_workspace(str(tmp_path))

    existing.write_text("after is longer", encoding="utf-8")
    removed.unlink()
    (tmp_path / "created.txt").write_text("new", encoding="utf-8")

    assert changed_workspace_paths(str(tmp_path), before) == [
        "created.txt",
        "existing.txt",
        "removed.txt",
    ]


def test_ignores_dependency_and_runtime_directories(tmp_path):
    ignored = tmp_path / "node_modules" / "package"
    ignored.mkdir(parents=True)
    file = ignored / "index.js"
    file.write_text("before", encoding="utf-8")
    before = snapshot_workspace(str(tmp_path))
    file.write_text("after", encoding="utf-8")

    assert changed_workspace_paths(str(tmp_path), before) == []
