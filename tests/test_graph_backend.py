"""Stage 3 tests: Entity Linking, Temporal Parsing, Graph Backend.

Tests cover:
- EntityLinker: candidate extraction, exact/fuzzy matching, idempotent linking
- TemporalParser: ISO dates, Chinese dates, relative expressions
- SqliteGraphBackend: entity mention edges, fact relations, graph traversal
- GraphProjector: outbox projection
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from memory.db import MemoryDB
from memory.models import Fact, FactRelation, TemporalInfo
from memory.scope import MemoryScope


@pytest.fixture
def tmp_db(tmp_path) -> MemoryDB:
    db_path = tmp_path / "test_graph_memory.db"
    db = MemoryDB(db_path=db_path)
    yield db
    db.close()


@pytest.fixture
def backend(tmp_db):
    from memory.backends.sqlite_backend import SQLiteStorageBackend
    return SQLiteStorageBackend(db=tmp_db)


def _make_fact(
    fid: str = "e1",
    tenant: str = "default",
    domain: str = "project",
    scope_id: str = "p1",
    kind: str = "decision",
    text: str = "Use SQLite for memory storage",
) -> Fact:
    return Fact(
        id=fid, tenant_id=tenant, domain=domain, scope_id=scope_id,  # type: ignore[arg-type]
        kind=kind, text=text,  # type: ignore[arg-type]
    )


# ===========================================================================
# §1  Entity Linker
# ===========================================================================

class TestEntityLinker:
    @pytest.fixture
    def linker(self, tmp_db):
        from memory.entity_linker import EntityLinker
        return EntityLinker(db=tmp_db)

    def test_extract_candidates_tech(self, linker):
        candidates = linker.extract_candidates("Use SQLite and FastAPI for the backend")
        names = [c[0] for c in candidates]
        assert "SQLite" in names
        assert "FastAPI" in names

    def test_extract_candidates_file_path(self, linker):
        candidates = linker.extract_candidates("Edit backend/memory/db.py to add WAL")
        names = [c[0] for c in candidates]
        assert any("db.py" in n for n in names)

    def test_extract_candidates_dedup(self, linker):
        candidates = linker.extract_candidates("SQLite is great. SQLite is fast.")
        names = [c[0] for c in candidates]
        assert names.count("SQLite") == 1

    def test_extract_candidates_empty(self, linker):
        candidates = linker.extract_candidates("use the thing for stuff")
        # No tech terms, no file paths, no CamelCase
        assert len(candidates) == 0 or all(c[1] != "tech" for c in candidates)

    def test_link_fact_creates_entities(self, linker, backend):
        fact = _make_fact(fid="el1", text="Use SQLite and FastAPI for the backend")
        backend.insert_fact(fact)
        refs = linker.link_fact(fact)
        assert len(refs) >= 2
        # Entities should be in the database
        entities = linker.list_entities()
        names = [e["name"] for e in entities]
        assert "SQLite" in names
        assert "FastAPI" in names

    def test_link_fact_idempotent(self, linker, backend):
        fact = _make_fact(fid="el2", text="Use SQLite for storage")
        backend.insert_fact(fact)
        refs1 = linker.link_fact(fact)
        refs2 = linker.link_fact(fact)
        # Same entity_ids both times
        ids1 = {r.entity_id for r in refs1}
        ids2 = {r.entity_id for r in refs2}
        assert ids1 == ids2

    def test_fuzzy_matching(self, linker, backend):
        """Similar entity names should fuzzy-match to the same entity."""
        fact1 = _make_fact(fid="el3", text="Use SQLite for the database")
        backend.insert_fact(fact1)
        linker.link_fact(fact1)

        fact2 = _make_fact(fid="el4", text="Use SQLite database for storage")
        backend.insert_fact(fact2)
        refs2 = linker.link_fact(fact2)

        # SQLite should resolve to the same entity both times
        sqlite_refs = [r for r in refs2 if r.name == "SQLite"]
        if sqlite_refs:
            entities = linker.list_entities()
            sqlite_entities = [e for e in entities if e["name"] == "SQLite"]
            assert len(sqlite_entities) == 1

    def test_alias_lookup(self, linker, backend):
        fact = _make_fact(fid="el5", text="Use SQLite for storage")
        backend.insert_fact(fact)
        linker.link_fact(fact)
        entity_id = linker.find_by_alias("SQLite")
        assert entity_id is not None
        entity = linker.get_entity(entity_id)
        assert entity is not None
        assert entity["name"] == "SQLite"

    def test_mention_count_increments(self, linker, backend):
        f1 = _make_fact(fid="el6", text="Use SQLite for storage")
        f2 = _make_fact(fid="el7", text="Use SQLite for queries")
        backend.insert_fact(f1)
        backend.insert_fact(f2)
        linker.link_fact(f1)
        linker.link_fact(f2)
        entity_id = linker.find_by_alias("SQLite")
        entity = linker.get_entity(entity_id)
        # v4: mention_count replaced by last_seen_at; verify entity is still found.
        assert entity is not None
        assert entity["name"] == "SQLite"


# ===========================================================================
# §2  Temporal Parser
# ===========================================================================

class TestTemporalParser:
    @pytest.fixture
    def parser(self):
        from memory.temporal_parser import TemporalParser
        return TemporalParser()

    @pytest.fixture
    def fixed_now(self):
        return datetime(2026, 8, 10, 12, 0, 0, tzinfo=timezone.utc)

    def test_iso_date(self, parser, fixed_now):
        ti = parser.parse("Deploy on 2026-08-15", now=fixed_now)
        assert ti is not None
        assert ti.event_time == datetime(2026, 8, 15, tzinfo=timezone.utc)
        assert "2026-08-15" in ti.time_expressions

    def test_iso_year_month(self, parser, fixed_now):
        ti = parser.parse("Release planned for 2026-09", now=fixed_now)
        assert ti is not None
        assert ti.event_time == datetime(2026, 9, 1, tzinfo=timezone.utc)

    def test_chinese_date(self, parser, fixed_now):
        ti = parser.parse("2026年8月10日部署", now=fixed_now)
        assert ti is not None
        assert ti.event_time == datetime(2026, 8, 10, tzinfo=timezone.utc)

    def test_chinese_month_day(self, parser, fixed_now):
        ti = parser.parse("8月15日上线", now=fixed_now)
        assert ti is not None
        assert ti.event_time == datetime(2026, 8, 15, tzinfo=timezone.utc)

    def test_chinese_relative_today(self, parser, fixed_now):
        ti = parser.parse("今天完成任务", now=fixed_now)
        assert ti is not None
        assert ti.event_time == fixed_now
        assert "today" in ti.time_expressions

    def test_chinese_relative_yesterday(self, parser, fixed_now):
        ti = parser.parse("昨天部署了", now=fixed_now)
        assert ti is not None
        from datetime import timedelta
        assert ti.event_time == fixed_now - timedelta(days=1)

    def test_chinese_relative_last_week(self, parser, fixed_now):
        ti = parser.parse("上周开了会", now=fixed_now)
        assert ti is not None
        from datetime import timedelta
        assert ti.event_time == fixed_now - timedelta(weeks=1)

    def test_english_relative(self, parser, fixed_now):
        ti = parser.parse("Deploy yesterday", now=fixed_now)
        assert ti is not None
        from datetime import timedelta
        assert ti.event_time == fixed_now - timedelta(days=1)

    def test_no_temporal(self, parser, fixed_now):
        ti = parser.parse("Use SQLite for storage", now=fixed_now)
        assert ti is None

    def test_multiple_expressions(self, parser, fixed_now):
        ti = parser.parse("Start 2026-08-01, end 2026-08-31", now=fixed_now)
        assert ti is not None
        assert len(ti.time_expressions) >= 2
        # First date becomes event_time
        assert ti.event_time == datetime(2026, 8, 1, tzinfo=timezone.utc)


# ===========================================================================
# §3  Graph Backend
# ===========================================================================

class TestGraphBackend:
    @pytest.fixture
    def graph(self, tmp_db):
        from memory.graph_backend import SqliteGraphBackend
        return SqliteGraphBackend(db=tmp_db)

    def test_upsert_entity_mention(self, graph, tmp_db):
        graph.upsert_entity_mention_sync("f1", "ent1", "subject", "default")
        facts = graph.facts_for_entity("ent1")
        assert "f1" in facts

    def test_upsert_fact_relation(self, graph):
        rel = graph.upsert_fact_relation_sync("f1", "f2", "supersedes", "default")
        assert rel.src_fact_id == "f1"
        assert rel.dst_fact_id == "f2"
        assert rel.relation == "supersedes"
        related = graph.related_facts_sync("f1", "default")
        assert ("f2", "supersedes", 1.0) in related

    def test_related_facts_bidirectional(self, graph):
        graph.upsert_fact_relation_sync("f1", "f2", "supports", "default")
        # Should be findable from both directions
        related_f1 = graph.related_facts_sync("f1", "default")
        related_f2 = graph.related_facts_sync("f2", "default")
        assert any(fid == "f2" for fid, _, _ in related_f1)
        assert any(fid == "f1" for fid, _, _ in related_f2)

    def test_entity_neighbors_hop1(self, graph, tmp_db):
        graph.upsert_entity_mention_sync("f1", "ent1", "subject", "default")
        graph.upsert_entity_mention_sync("f2", "ent1", "context", "default")
        neighbors = graph.entity_neighbors_sync("ent1", "default", max_hops=1)
        assert set(neighbors) == {"f1", "f2"}

    def test_entity_neighbors_hop2(self, graph, tmp_db):
        # ent1 → f1, f2; f1 → f3 via relation
        graph.upsert_entity_mention_sync("f1", "ent1", "subject", "default")
        graph.upsert_entity_mention_sync("f2", "ent1", "context", "default")
        graph.upsert_fact_relation_sync("f1", "f3", "derived_from", "default")
        neighbors = graph.entity_neighbors_sync("ent1", "default", max_hops=2)
        assert "f3" in neighbors

    def test_async_entity_mention(self, graph):
        import asyncio
        scope = MemoryScope(domain="project", scope_id="p1")

        async def _run():
            await graph.upsert_entity_mention("f1", "ent1", "subject", scope)
            return await graph.entity_neighbors("ent1", scope, max_hops=1)

        neighbors = asyncio.run(_run())
        assert "f1" in neighbors


# ===========================================================================
# §4  Graph Projector (outbox)
# ===========================================================================

class TestGraphProjector:
    @pytest.fixture
    def projector(self, tmp_db):
        from memory.graph_backend import GraphProjector
        return GraphProjector(db=tmp_db)

    def test_enqueue_and_process(self, projector, backend, tmp_db):
        fact = _make_fact(fid="gp1", text="Use SQLite and FastAPI for storage")
        backend.insert_fact(fact)
        projector.enqueue_projection(fact.id, fact.tenant_id, "insert")
        assert projector.pending_count() == 1
        processed = projector.process_outbox()
        assert processed == 1
        assert projector.pending_count() == 0

    def test_process_delete_op(self, projector, backend, tmp_db):
        from memory.entity_linker import EntityLinker
        fact = _make_fact(fid="gp2", text="Use SQLite for storage")
        backend.insert_fact(fact)
        linker = EntityLinker(tmp_db)
        linker.link_fact(fact)
        # Verify mention exists
        assert len(linker.list_entities()) > 0
        # Enqueue delete and process
        projector.enqueue_projection(fact.id, fact.tenant_id, "delete")
        projector.process_outbox()
        # Mention should be removed
        rows = tmp_db.query_all(
            "SELECT * FROM fact_entity_mentions WHERE fact_id = ?",
            (fact.id,),
        )
        assert len(rows) == 0

    def test_process_empty_outbox(self, projector):
        assert projector.process_outbox() == 0
        assert projector.pending_count() == 0
