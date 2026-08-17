import json

from memory import import_legacy


def test_project_agent_migration_preserves_project_id(tmp_path, monkeypatch):
    project_id = "a" * 32
    memory_dir = tmp_path / "data" / "projects" / project_id / "agents" / "memory"
    memory_dir.mkdir(parents=True)
    (memory_dir / "coordinator.json").write_text(
        json.dumps({
            "entries": [
                {"id": "history-1", "kind": "decision", "text": "old result"},
                {"id": "skill-1", "kind": "practice", "text": "repeatable method"},
            ]
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(import_legacy, "_BACKEND_DIR", tmp_path)

    scanner = import_legacy.LegacyScanner()
    scanner._scan_project_agent_memory()

    assert scanner.history[0].channel_id == project_id
    assert scanner.history[0].thread_id == "legacy-agent:coordinator"
    assert scanner.skills[0].scope_id == project_id
