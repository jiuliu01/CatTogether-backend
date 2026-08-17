import pytest

from core.run_event_store import RunEventStore


@pytest.fixture(autouse=True)
def isolate_run_progress_events(tmp_path, monkeypatch):
    """Tests must not append progress JSONL into the user's runtime data."""
    from integrations.feishu import progress as progress_module

    monkeypatch.setattr(
        progress_module,
        "run_event_store",
        RunEventStore(tmp_path / "run_events"),
    )


@pytest.fixture(autouse=True)
def isolate_default_memory_database(tmp_path, monkeypatch):
    """Orchestrator tests must never write to the runtime memory.db."""
    import memory.db as memory_db_module
    import memory.context_builder as context_builder_module
    import memory.memory_api as memory_api_module

    test_db = memory_db_module.MemoryDB(tmp_path / "runtime_memory.db")
    monkeypatch.setattr(memory_db_module, "memory_db", test_db)
    monkeypatch.setattr(context_builder_module, "memory_db", test_db)
    monkeypatch.setattr(memory_api_module, "memory_db", test_db)
    yield
    test_db.close()
