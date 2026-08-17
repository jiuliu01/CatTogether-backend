"""Stage 1 tests: SQLite backend, BM25 retrieval, ADD-only semantics,
Hot Memory, History, Skill stores, and JSON→SQLite migration.

All tests use an in-memory or temp-file SQLite database, not the production
``memory.db``.  The v3 schema DDL is applied on connect by ``MemoryDB``.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Ensure backend is on sys.path for test isolation.
_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from memory.db import MemoryDB
from memory.models import (
    Actor,
    Fact,
    FactRelation,
    HotMemoryItem,
    ProvenanceInfo,
    RetrievalQuery,
    TemporalInfo,
)
from memory.scope import MemoryScope


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path) -> MemoryDB:
    """A MemoryDB pointing at a temp file, isolated from production."""
    db_path = tmp_path / "test_memory.db"
    db = MemoryDB(db_path=db_path)
    yield db
    db.close()


@pytest.fixture
def backend(tmp_db):
    from memory.backends.sqlite_backend import SQLiteStorageBackend
    return SQLiteStorageBackend(db=tmp_db)


def _make_fact(
    fid: str = "f1",
    tenant: str = "default",
    domain: str = "project",
    scope_id: str = "p1",
    kind: str = "decision",
    text: str = "Use SQLite for memory storage",
    tags: list[str] | None = None,
    importance: float = 0.7,
    confidence: float = 0.8,
    status: str = "active",
    search_text: str | None = None,
) -> Fact:
    return Fact(
        id=fid, tenant_id=tenant, domain=domain, scope_id=scope_id,  # type: ignore[arg-type]
        kind=kind, text=text, search_text=search_text or text, tags=tags or [],  # type: ignore[arg-type]
        importance=importance, confidence=confidence, status=status,  # type: ignore[arg-type]
    )


# ===========================================================================
# §1  SQLite schema initialization
# ===========================================================================

class TestSchemaInit:
    def test_schema_applied_on_connect(self, tmp_db):
        conn = tmp_db.connect()
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "facts" in tables
        assert "events" in tables
        assert "entities" in tables
        assert "hot_memories" in tables
        assert "session_messages" in tables
        assert "skill_registry" in tables
        assert "scope_registry" in tables
        assert "transaction_outbox" in tables
        assert "memory_audit_log" in tables
        assert "schema_migrations" in tables

    def test_fts5_tables_exist(self, tmp_db):
        conn = tmp_db.connect()
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "facts_fts" in tables
        assert "session_messages_fts" in tables

    def test_schema_version_recorded(self, tmp_db):
        tmp_db.connect()
        row = tmp_db.query_one("SELECT version FROM schema_migrations WHERE version=4")
        assert row is not None

    def test_idempotent_connect(self, tmp_db):
        conn1 = tmp_db.connect()
        conn2 = tmp_db.connect()
        assert conn1 is conn2

    def test_wal_mode_enabled(self, tmp_db):
        conn = tmp_db.connect()
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"


# ===========================================================================
# §2  Fact CRUD + ADD-only semantics (R4)
# ===========================================================================

class TestFactCRUD:
    def test_insert_fact(self, backend):
        fact = _make_fact()
        result = backend.insert_fact(fact)
        assert result.id == "f1"
        retrieved = backend.get_fact("f1", "default")
        assert retrieved is not None
        assert retrieved.text == "Use SQLite for memory storage"

    def test_insert_fact_with_temporal_provenance(self, backend):
        fact = Fact(
            id="f2", tenant_id="default", domain="project", scope_id="p1",
            kind="decision", text="Use PostgreSQL for production",
            search_text="PostgreSQL production database",
            temporal=TemporalInfo(event_time=datetime(2026, 8, 10, tzinfo=timezone.utc)),
            provenance=ProvenanceInfo(source_type="agent_result", extractor="llm"),
        )
        backend.insert_fact(fact)
        retrieved = backend.get_fact("f2", "default")
        assert retrieved is not None
        assert retrieved.search_text == "PostgreSQL production database"
        assert retrieved.temporal is not None
        assert retrieved.provenance.extractor == "llm"

    def test_add_only_insert_does_not_overwrite(self, backend):
        """R4: insert_fact always creates new rows. Inserting with an existing
        id should fail (PRIMARY KEY constraint), not silently overwrite."""
        fact1 = _make_fact(fid="f1", text="original text")
        backend.insert_fact(fact1)
        fact2 = _make_fact(fid="f1", text="different text")
        with pytest.raises(Exception):
            backend.insert_fact(fact2)
        # Original is unchanged.
        retrieved = backend.get_fact("f1", "default")
        assert retrieved.text == "original text"

    def test_patch_metadata_only(self, backend):
        """R4: patch_fact_metadata changes only metadata, never text."""
        fact = _make_fact(fid="f1", text="original")
        backend.insert_fact(fact)
        updated = backend.patch_fact_metadata(
            "f1", "default", tags=["new", "tags"], importance=0.9,
        )
        assert updated is not None
        assert updated.text == "original"  # text unchanged
        assert updated.tags == ["new", "tags"]
        assert updated.importance == 0.9

    def test_soft_delete(self, backend):
        """R4: delete defaults to soft delete (status=archived)."""
        fact = _make_fact()
        backend.insert_fact(fact)
        assert backend.soft_delete("f1", "default") is True
        retrieved = backend.get_fact("f1", "default")
        assert retrieved.status == "archived"

    def test_hard_delete(self, backend):
        fact = _make_fact()
        backend.insert_fact(fact)
        assert backend.hard_delete("f1", "default") is True
        assert backend.get_fact("f1", "default") is None

    def test_list_facts_by_domain(self, backend):
        backend.insert_fact(_make_fact(fid="f1", domain="project", scope_id="p1"))
        backend.insert_fact(_make_fact(fid="f2", domain="user", scope_id="alice"))
        facts = backend.list_facts("default", domain="project")
        assert len(facts) == 1
        assert facts[0].id == "f1"

    def test_list_facts_excludes_inactive(self, backend):
        backend.insert_fact(_make_fact(fid="f1", status="active"))
        backend.insert_fact(_make_fact(fid="f2", status="archived"))
        facts = backend.list_facts("default", domain="project")
        assert len(facts) == 1
        assert facts[0].id == "f1"

    def test_tenant_isolation(self, backend):
        """R5: tenant_id hard isolation."""
        backend.insert_fact(_make_fact(fid="f1", tenant="acme", text="acme fact"))
        backend.insert_fact(_make_fact(fid="f2", tenant="other", text="other fact"))
        acme_facts = backend.list_facts("acme", domain="project")
        assert len(acme_facts) == 1
        assert acme_facts[0].text == "acme fact"
        other_facts = backend.list_facts("other", domain="project")
        assert len(other_facts) == 1
        assert other_facts[0].text == "other fact"


# ===========================================================================
# §3  BM25 retrieval via FTS5
# ===========================================================================

class TestBM25Retrieval:
    def test_basic_search(self, backend):
        backend.insert_fact(_make_fact(fid="f1", text="Use SQLite for memory storage"))
        backend.insert_fact(_make_fact(fid="f2", text="Use Redis for caching"))
        results = backend.search_bm25("SQLite", "default", domain="project")
        assert len(results) >= 1
        assert results[0][0].text == "Use SQLite for memory storage"

    def test_search_returns_scored_results(self, backend):
        backend.insert_fact(_make_fact(fid="f1", text="SQLite SQLite SQLite storage"))
        backend.insert_fact(_make_fact(fid="f2", text="SQLite mentioned once"))
        results = backend.search_bm25("SQLite", "default", domain="project", top_k=2)
        assert len(results) == 2
        # Higher term frequency → better (more negative) BM25 score.
        # bm25() returns negative values; lower = better match.
        assert results[0][1] <= results[1][1]

    def test_search_filters_by_tenant(self, backend):
        backend.insert_fact(_make_fact(fid="f1", tenant="a", text="SQLite storage"))
        backend.insert_fact(_make_fact(fid="f2", tenant="b", text="SQLite caching"))
        results_a = backend.search_bm25("SQLite", "a", domain="project")
        results_b = backend.search_bm25("SQLite", "b", domain="project")
        assert len(results_a) == 1
        assert results_a[0][0].tenant_id == "a"
        assert len(results_b) == 1
        assert results_b[0][0].tenant_id == "b"

    def test_search_filters_by_scope(self, backend):
        backend.insert_fact(_make_fact(fid="f1", scope_id="p1", text="SQLite"))
        backend.insert_fact(_make_fact(fid="f2", scope_id="p2", text="SQLite"))
        results = backend.search_bm25("SQLite", "default", domain="project", scope_id="p1")
        assert len(results) == 1
        assert results[0][0].scope_id == "p1"

    def test_search_excludes_archived(self, backend):
        backend.insert_fact(_make_fact(fid="f1", text="SQLite active", status="active"))
        backend.insert_fact(_make_fact(fid="f2", text="SQLite archived", status="archived"))
        results = backend.search_bm25("SQLite", "default", domain="project")
        assert len(results) == 1
        assert results[0][0].text == "SQLite active"

    def test_empty_query_returns_empty(self, backend):
        backend.insert_fact(_make_fact(fid="f1", text="SQLite"))
        assert backend.search_bm25("", "default") == []
        assert backend.search_bm25("   ", "default") == []

    def test_search_bm25_as_results(self, backend):
        backend.insert_fact(_make_fact(fid="f1", text="Use SQLite for storage"))
        query = RetrievalQuery(text="SQLite", tenant_id="default", domain="project", scope_id="p1")
        results = backend.search_bm25_as_results(query)
        assert len(results) >= 1
        assert "bm25" in results[0].signals

    def test_chinese_text_search(self, backend):
        backend.insert_fact(_make_fact(fid="f1", text="使用SQLite作为记忆存储后端"))
        backend.insert_fact(_make_fact(fid="f2", text="使用Redis作为缓存"))
        results = backend.search_bm25("SQLite", "default", domain="project")
        assert len(results) >= 1
        assert "SQLite" in results[0][0].text


# ===========================================================================
# §4  v1 StorageBackend ABC compatibility
# ===========================================================================

class TestStorageBackendCompat:
    def test_upsert_and_list(self, backend):
        from models.schemas import MemoryEntry
        scope = MemoryScope("project", "p1")
        entry = MemoryEntry(id="e1", domain="project", scope_id="p1", kind="decision",
                            text="test entry", confidence=0.8)
        asyncio.run(backend.upsert(scope, entry))
        entries = asyncio.run(backend.list(scope))
        assert len(entries) == 1
        assert entries[0].text == "test entry"

    def test_recall_returns_matches(self, backend):
        from models.schemas import MemoryEntry
        scope = MemoryScope("project", "p1")
        entry = MemoryEntry(id="e1", domain="project", scope_id="p1", kind="decision",
                            text="Use SQLite for storage", confidence=0.8)
        asyncio.run(backend.upsert(scope, entry))
        results = asyncio.run(backend.recall(scope, "SQLite", 5))
        assert len(results) >= 1
        assert any("SQLite" in e.text for e in results)

    def test_delete_soft_deletes(self, backend):
        from models.schemas import MemoryEntry
        scope = MemoryScope("project", "p1")
        entry = MemoryEntry(id="e1", domain="project", scope_id="p1", kind="decision",
                            text="to be deleted")
        asyncio.run(backend.upsert(scope, entry))
        assert asyncio.run(backend.delete(scope, "e1")) is True
        # list excludes inactive (archived) entries.
        entries = asyncio.run(backend.list(scope))
        assert len(entries) == 0

    def test_workspace_domain_normalized(self, backend):
        from models.schemas import MemoryEntry
        scope = MemoryScope("workspace", "ws-1")
        entry = MemoryEntry(id="e1", domain="workspace", scope_id="ws-1",
                            kind="decision", text="workspace entry")
        asyncio.run(backend.upsert(scope, entry))
        # Should be retrievable via project domain.
        facts = backend.list_facts("default", domain="project", scope_id="ws-1")
        assert len(facts) == 1


# ===========================================================================
# §5  Hot Memory store (R7)
# ===========================================================================

class TestHotMemory:
    def test_add_hot_auto_approve(self, tmp_db):
        from memory.hot_memory_store import HotMemoryStore
        store = HotMemoryStore(db=tmp_db)
        item = store.add_hot("default", "user", "alice", "prefers dark mode",
                            category="preference", importance=0.95)
        assert item.status == "active"  # auto-approved (>= 0.9 threshold)

    def test_add_hot_pending_approval(self, tmp_db):
        from memory.hot_memory_store import HotMemoryStore
        store = HotMemoryStore(db=tmp_db)
        item = store.add_hot("default", "user", "alice", "maybe likes tea",
                            category="note", importance=0.5)
        assert item.status == "pending_approval"

    def test_list_hot_active_only(self, tmp_db):
        from memory.hot_memory_store import HotMemoryStore
        store = HotMemoryStore(db=tmp_db)
        store.add_hot("default", "user", "alice", "active item", importance=0.95)
        store.add_hot("default", "user", "alice", "pending item", importance=0.5)
        items = store.list_hot("default", "user", "alice")
        assert all(i.status == "active" for i in items)
        assert len(items) == 1

    def test_approve_hot(self, tmp_db):
        from memory.hot_memory_store import HotMemoryStore
        store = HotMemoryStore(db=tmp_db)
        item = store.add_hot("default", "user", "alice", "pending", importance=0.5)
        assert store.approve_hot(item.id, "admin") is True
        items = store.list_hot("default", "user", "alice")
        assert len(items) == 1
        assert items[0].text == "pending"

    def test_archive_hot(self, tmp_db):
        from memory.hot_memory_store import HotMemoryStore
        store = HotMemoryStore(db=tmp_db)
        item = store.add_hot("default", "user", "alice", "to archive", importance=0.95)
        assert store.archive_hot(item.id) is True
        assert store.list_hot("default", "user", "alice") == []

    def test_capacity_eviction(self, tmp_db, monkeypatch):
        from memory.hot_memory_store import HotMemoryStore
        from config import settings
        monkeypatch.setattr(settings, "memory_hot_max_per_scope", 3)
        store = HotMemoryStore(db=tmp_db)
        # Add 5 items, all auto-approved.
        for i in range(5):
            store.add_hot("default", "user", "alice", f"item {i}",
                        importance=0.9 + i * 0.01)
        items = store.list_hot("default", "user", "alice")
        assert len(items) == 3  # capacity enforced
        # Lowest importance evicted.
        texts = {i.text for i in items}
        assert "item 0" not in texts
        assert "item 1" not in texts


# ===========================================================================
# §6  History store (R8)
# ===========================================================================

class TestHistoryStore:
    def test_append_and_list(self, tmp_db):
        from memory.history_store import HistoryStore
        store = HistoryStore(db=tmp_db)
        msg = store.append_message("default", "ch1", "user", "Hello world")
        assert msg.seq == 1
        msg2 = store.append_message("default", "ch1", "agent", "Hi there")
        assert msg2.seq == 2
        messages = store.list_messages("default", "ch1")
        assert len(messages) == 2

    def test_search_history(self, tmp_db):
        from memory.history_store import HistoryStore
        store = HistoryStore(db=tmp_db)
        store.append_message("default", "ch1", "user", "How to configure SQLite?")
        store.append_message("default", "ch1", "agent", "Use WAL mode for SQLite")
        store.append_message("default", "ch1", "user", "What about Redis?")
        results = store.search_history("default", "SQLite", channel_id="ch1")
        assert len(results) >= 1
        assert any("SQLite" in m.content for m in results)

    def test_thread_isolation(self, tmp_db):
        from memory.history_store import HistoryStore
        store = HistoryStore(db=tmp_db)
        store.append_message("default", "ch1", "user", "thread 1 msg", thread_id="t1")
        store.append_message("default", "ch1", "user", "thread 2 msg", thread_id="t2")
        msgs = store.list_messages("default", "ch1", thread_id="t1")
        assert len(msgs) == 1
        assert msgs[0].content == "thread 1 msg"


# ===========================================================================
# §7  Skill store (R6)
# ===========================================================================

class TestSkillStore:
    def test_create_and_get(self, tmp_db):
        from memory.skill_store import SkillStore
        store = SkillStore(db=tmp_db)
        skill = store.create_skill("default", "agent", "coding",
                                   "run_tests", "Always run pytest after changes",
                                   description="testing practice")
        assert skill.version == 1
        retrieved = store.get_skill("default", "agent", "coding", "run_tests")
        assert retrieved is not None
        assert retrieved.body == "Always run pytest after changes"

    def test_update_creates_new_version(self, tmp_db):
        from memory.skill_store import SkillStore
        store = SkillStore(db=tmp_db)
        skill = store.create_skill("default", "agent", "coding",
                                   "deploy", "Step 1: build")
        updated = store.update_skill(skill.id, body="Step 1: build\nStep 2: push")
        assert updated.version == 2
        assert updated.previous_version_id == skill.id
        current = store.get_skill("default", "agent", "coding", "deploy")
        assert current.version == 2
        assert "Step 2" in current.body

    def test_rollback(self, tmp_db):
        from memory.skill_store import SkillStore
        store = SkillStore(db=tmp_db)
        v1 = store.create_skill("default", "agent", "coding",
                                "rollback_test", "version 1 body")
        v2 = store.update_skill(v1.id, body="version 2 body")
        v3 = store.update_skill(v2.id, body="version 3 body")
        # Rollback to version 1.
        rolled = store.rollback_skill(v3.id, to_version=1)
        assert rolled.version == 4  # new version number
        assert rolled.body == "version 1 body"

    def test_duplicate_name_rejected(self, tmp_db):
        from memory.skill_store import SkillStore
        store = SkillStore(db=tmp_db)
        store.create_skill("default", "agent", "coding", "unique", "body 1")
        with pytest.raises(ValueError, match="already exists"):
            store.create_skill("default", "agent", "coding", "unique", "body 2")

    def test_list_skills(self, tmp_db):
        from memory.skill_store import SkillStore
        store = SkillStore(db=tmp_db)
        store.create_skill("default", "agent", "coding", "skill_a", "body a")
        store.create_skill("default", "agent", "coding", "skill_b", "body b")
        skills = store.list_skills("default", "agent", "coding")
        assert len(skills) == 2
        names = {s.name for s in skills}
        assert names == {"skill_a", "skill_b"}


# ===========================================================================
# §8  JSON → SQLite migration — moved to test_import_legacy.py
# ===========================================================================


# ===========================================================================
# §9  Four-layer physical separation
# ===========================================================================

class TestFourLayerSeparation:
    def test_hot_and_fact_in_separate_tables(self, tmp_db):
        from memory.hot_memory_store import HotMemoryStore
        from memory.backends.sqlite_backend import SQLiteStorageBackend
        backend = SQLiteStorageBackend(db=tmp_db)
        hot = HotMemoryStore(db=tmp_db)
        # Insert into both layers.
        backend.insert_fact(_make_fact(fid="f1", text="a fact"))
        hot.add_hot("default", "project", "p1", "a hot item", importance=0.95)
        # Verify they're in separate tables.
        facts = tmp_db.query_all("SELECT * FROM facts")
        hots = tmp_db.query_all("SELECT * FROM hot_memories")
        assert len(facts) == 1
        assert len(hots) == 1
        assert facts[0]["text"] == "a fact"
        assert hots[0]["text"] == "a hot item"

    def test_history_in_separate_table(self, tmp_db):
        from memory.history_store import HistoryStore
        from memory.backends.sqlite_backend import SQLiteStorageBackend
        backend = SQLiteStorageBackend(db=tmp_db)
        history = HistoryStore(db=tmp_db)
        backend.insert_fact(_make_fact(fid="f1"))
        history.append_message("default", "ch1", "user", "hello")
        facts = tmp_db.query_all("SELECT * FROM facts")
        msgs = tmp_db.query_all("SELECT * FROM session_messages")
        assert len(facts) == 1
        assert len(msgs) == 1

    def test_skill_in_separate_table(self, tmp_db):
        from memory.skill_store import SkillStore
        from memory.backends.sqlite_backend import SQLiteStorageBackend
        backend = SQLiteStorageBackend(db=tmp_db)
        skills = SkillStore(db=tmp_db)
        backend.insert_fact(_make_fact(fid="f1"))
        skills.create_skill("default", "agent", "coding", "test_skill", "body")
        facts = tmp_db.query_all("SELECT * FROM facts")
        skill_rows = tmp_db.query_all("SELECT * FROM skill_registry")
        assert len(facts) == 1
        assert len(skill_rows) == 1
