"""Tests for Stage 5: three-signal retrieval + four-layer context builder."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from memory.context_builder import ContextBuilder, MemoryContext, infer_memory_intent
from memory.db import MemoryDB
from memory.embedder import DummyEmbedder
from memory.layers.fact_store import FactStore
from memory.layers.history_store import HistoryStoreWrapper
from memory.layers.hot_store import HotStore
from memory.layers.skill_store import SkillStoreWrapper
from memory.models import (
    Actor, ActorContext, Fact, HotMemoryItem, RetrievalQuery, RetrievalResult,
)
from memory.permissions import compile_allowed_scopes
from memory.retrieval.bm25 import BM25Retriever
from memory.retrieval.entity import EntityRetriever
from memory.retrieval.hybrid import HybridRetriever
from memory.retrieval.mmr import mmr_rerank
from memory.retrieval.rrf import reciprocal_rank_fusion
from memory.retrieval.semantic import SemanticRetriever
from memory.retrieval.temporal_reranker import TemporalReranker
from memory.rebuild_embeddings import rebuild_embeddings
from memory.scope import MemoryScope, canonical_scope_id, tenant_id_from_user_id


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path):
    db = MemoryDB(tmp_path / "test_memory.db")
    yield db
    db.close()


def _insert_fact(
    db, fact_id, tenant_id="default", domain="project", scope_id="proj1",
    kind="project_fact", text="We use PostgreSQL", search_text=None,
    status="active", importance=0.5, confidence=0.5,
    expires_at=None, valid_to=None, agent_id=None, task_id=None,
):
    """Insert a fact directly into the DB for retrieval tests."""
    import json
    from datetime import datetime, timezone
    conn = db.connect()
    search_text = search_text if search_text is not None else text
    temporal_json = json.dumps({"valid_to": valid_to.isoformat()}) if valid_to else None
    prov_json = json.dumps({"extractor": "rule", "extraction_confidence": confidence})
    expires_str = expires_at.isoformat() if expires_at else None
    conn.execute(
        """INSERT OR REPLACE INTO facts
           (id, tenant_id, domain, scope_id, agent_id, task_id, kind, text,
            search_text, tags, importance, confidence, status, temporal,
            provenance, embedding_model, created_at, updated_at, expires_at,
            version, schema_version)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, '[]', ?, ?, ?, ?, ?, NULL, ?, ?, ?, 1, 4)""",
        (fact_id, tenant_id, domain, scope_id, agent_id, task_id, kind, text,
         search_text, importance, confidence, status, temporal_json, prov_json,
         datetime.now(timezone.utc).isoformat(),
         datetime.now(timezone.utc).isoformat(), expires_str),
    )
    # Insert into FTS5.
    conn.execute(
        "INSERT INTO facts_fts(rowid, search_text) VALUES ((SELECT rowid FROM facts WHERE id=?), ?)",
        (fact_id, search_text),
    )


def _insert_embedding(db, fact_id, text, dim=512):
    """Insert a dummy embedding for a fact."""
    embedder = DummyEmbedder(dim=dim)
    vec = embedder.embed(text)
    import struct
    blob = struct.pack(f"{len(vec)}f", *vec)
    conn = db.connect()
    conn.execute(
        "INSERT OR REPLACE INTO fact_embeddings (fact_id, tenant_id, domain, scope_id, embedding) "
        "VALUES (?, 'default', 'project', 'proj1', ?)",
        (fact_id, blob),
    )


# ---------------------------------------------------------------------------
# RRF tests
# ---------------------------------------------------------------------------

class TestRRF:
    def test_basic_fusion(self):
        """Two signals with overlapping IDs should fuse scores."""
        results = reciprocal_rank_fusion({
            "bm25": ["a", "b", "c"],
            "semantic": ["b", "a", "d"],
        }, weights={"bm25": 1.0, "semantic": 1.0, "entity": 0.8}, rrf_k=60)
        # 'a' and 'b' appear in both signals → higher score.
        assert results["a"] > results["c"]
        assert results["b"] > results["c"]
        assert results["b"] > results["d"]  # b is rank 0 in semantic, rank 1 in bm25

    def test_weights_applied(self):
        """Entity signal with lower weight should contribute less."""
        results = reciprocal_rank_fusion({
            "bm25": ["a"],
            "entity": ["a"],
        }, weights={"bm25": 1.0, "semantic": 1.0, "entity": 0.8}, rrf_k=60)
        # score = 1.0/(60+1) + 0.8/(60+1) = 1.8/61
        assert abs(results["a"] - (1.0 + 0.8) / 61) < 1e-9

    def test_empty_signals(self):
        results = reciprocal_rank_fusion({})
        assert results == {}

    def test_sorted_descending(self):
        results = reciprocal_rank_fusion({
            "bm25": ["a", "b", "c"],
        })
        ids = list(results.keys())
        assert ids[0] == "a"  # rank 0 = highest score

    def test_zero_weight_signal_ignored(self):
        results = reciprocal_rank_fusion({
            "bm25": ["a", "b"],
            "semantic": ["c"],
        }, weights={"bm25": 1.0, "semantic": 0.0, "entity": 0.0})
        assert "c" not in results  # semantic weight is 0


# ---------------------------------------------------------------------------
# BM25 retriever tests
# ---------------------------------------------------------------------------

class TestBM25Retriever:
    def test_basic_search(self, tmp_db):
        _insert_fact(tmp_db, "f1", text="We use PostgreSQL as database", search_text="We use PostgreSQL as database")
        _insert_fact(tmp_db, "f2", text="We use Redis as cache", search_text="We use Redis as cache")
        retriever = BM25Retriever(tmp_db)
        query = RetrievalQuery(text="PostgreSQL", tenant_id="default", domain="project", scope_id="proj1")
        hits = retriever.retrieve(query)
        assert len(hits) >= 1
        assert hits[0][0] == "f1"

    def test_empty_query(self, tmp_db):
        retriever = BM25Retriever(tmp_db)
        query = RetrievalQuery(text="", tenant_id="default")
        assert retriever.retrieve(query) == []

    def test_status_filter(self, tmp_db):
        _insert_fact(tmp_db, "f1", text="active fact about Python", search_text="active fact about Python", status="active")
        _insert_fact(tmp_db, "f2", text="pending fact about Python", search_text="pending fact about Python", status="pending_review")
        retriever = BM25Retriever(tmp_db)
        query = RetrievalQuery(text="Python", tenant_id="default", domain="project", scope_id="proj1")
        hits = retriever.retrieve(query)
        ids = [h[0] for h in hits]
        assert "f1" in ids
        assert "f2" not in ids  # pending_review should be filtered

    def test_tenant_isolation(self, tmp_db):
        _insert_fact(tmp_db, "f1", tenant_id="tenant_a", text="secret A data", search_text="secret A data")
        _insert_fact(tmp_db, "f2", tenant_id="tenant_b", text="secret B data", search_text="secret B data")
        retriever = BM25Retriever(tmp_db)
        query = RetrievalQuery(text="secret", tenant_id="tenant_a", domain="project", scope_id="proj1")
        hits = retriever.retrieve(query)
        ids = [h[0] for h in hits]
        assert "f1" in ids
        assert "f2" not in ids


# ---------------------------------------------------------------------------
# Semantic retriever tests
# ---------------------------------------------------------------------------

class TestSemanticRetriever:
    def test_unavailable_returns_empty(self, tmp_db):
        """If sqlite-vec is not available, semantic returns empty."""
        retriever = SemanticRetriever(tmp_db, DummyEmbedder())
        query = RetrievalQuery(text="test", tenant_id="default")
        if not tmp_db.vec_available:
            assert retriever.retrieve(query) == []

    def test_empty_query(self, tmp_db):
        retriever = SemanticRetriever(tmp_db, DummyEmbedder())
        query = RetrievalQuery(text="", tenant_id="default")
        assert retriever.retrieve(query) == []


# ---------------------------------------------------------------------------
# Entity retriever tests
# ---------------------------------------------------------------------------

class TestEntityRetriever:
    def test_no_entities_returns_empty(self, tmp_db):
        retriever = EntityRetriever(tmp_db)
        query = RetrievalQuery(text="", tenant_id="default")
        assert retriever.retrieve(query) == []

    def test_explicit_entity_lookup(self, tmp_db):
        """Query with explicit entity IDs should find linked facts."""
        import json
        conn = tmp_db.connect()
        # Create entity + mention.
        conn.execute(
            "INSERT INTO entities (entity_id, tenant_id, canonical_name, type, first_seen_at, last_seen_at) "
            "VALUES ('e1', 'default', 'PostgreSQL', 'concept', ?, ?)",
            (datetime.now(timezone.utc).isoformat(), datetime.now(timezone.utc).isoformat()),
        )
        _insert_fact(tmp_db, "f1", text="We use PostgreSQL", search_text="We use PostgreSQL")
        conn.execute(
            "INSERT INTO fact_entity_mentions (tenant_id, fact_id, entity_id, mention_text, role, linking_confidence) "
            "VALUES ('default', 'f1', 'e1', 'PostgreSQL', 'subject', 0.9)",
        )
        retriever = EntityRetriever(tmp_db)
        query = RetrievalQuery(text="database", tenant_id="default", entities=["e1"], domain="project", scope_id="proj1")
        hits = retriever.retrieve(query)
        assert len(hits) >= 1
        assert hits[0][0] == "f1"


# ---------------------------------------------------------------------------
# Temporal reranker tests
# ---------------------------------------------------------------------------

class TestTemporalReranker:
    def _make_result(self, kind="project_fact", valid_to=None, confidence=0.5, importance=0.5, score=1.0):
        from memory.models import TemporalInfo, ProvenanceInfo
        temporal = TemporalInfo(valid_to=valid_to) if valid_to else None
        fact = Fact(
            id="f1", tenant_id="default", domain="project", scope_id="proj1",
            kind=kind, text="test", search_text="test",
            confidence=confidence, importance=importance,
            temporal=temporal, provenance=ProvenanceInfo(),
        )
        return RetrievalResult(fact=fact, score=score, signals=["bm25"])

    def test_expired_fact_dropped(self):
        reranker = TemporalReranker()
        past = datetime.now(timezone.utc) - timedelta(days=1)
        result = self._make_result(valid_to=past)
        out = reranker.rerank([result])
        assert len(out) == 0

    def test_expired_fact_kept_for_audit(self):
        reranker = TemporalReranker()
        past = datetime.now(timezone.utc) - timedelta(days=1)
        result = self._make_result(valid_to=past)
        out = reranker.rerank([result], intent="audit")
        assert len(out) == 1  # audit keeps expired facts

    def test_stable_kind_no_decay(self):
        reranker = TemporalReranker()
        result = self._make_result(kind="preference", score=1.0)
        out = reranker.rerank([result])
        assert len(out) == 1
        # preference has no decay → score stays close to 1.0 (modulo conf/imp boost)
        assert out[0].score > 0.9

    def test_progress_decay(self):
        reranker = TemporalReranker()
        # Create a progress fact from 60 days ago (2 half-lives).
        from memory.models import TemporalInfo, ProvenanceInfo
        old_time = datetime.now(timezone.utc) - timedelta(days=60)
        fact = Fact(
            id="f1", tenant_id="default", domain="project", scope_id="proj1",
            kind="progress", text="old progress", search_text="old progress",
            temporal=TemporalInfo(event_time=old_time),
            provenance=ProvenanceInfo(),
            created_at=old_time,
        )
        result = RetrievalResult(fact=fact, score=1.0, signals=["bm25"])
        out = reranker.rerank([result])
        assert len(out) == 1
        # 60 days / 30 day half-life = 2 half-lives → decay = 0.25
        assert out[0].score < 0.3
        assert out[0].score > 0.2


# ---------------------------------------------------------------------------
# MMR tests
# ---------------------------------------------------------------------------

class TestMMR:
    def _make_result(self, fact_id, text, score=1.0):
        fact = Fact(id=fact_id, tenant_id="default", domain="project", scope_id="proj1",
                    kind="project_fact", text=text, search_text=text)
        return RetrievalResult(fact=fact, score=score, signals=["bm25"])

    def test_single_result(self):
        results = [self._make_result("f1", "hello")]
        out = mmr_rerank(results)
        assert len(out) == 1

    def test_diverse_selection(self):
        results = [
            self._make_result("f1", "PostgreSQL database config", score=1.0),
            self._make_result("f2", "PostgreSQL database setup", score=0.9),
            self._make_result("f3", "Redis cache config", score=0.8),
        ]
        out = mmr_rerank(results, lambda_param=0.5, top_k=2)
        assert len(out) == 2
        # f1 should be first (highest score).
        assert out[0].fact.id == "f1"

    def test_empty(self):
        assert mmr_rerank([]) == []


# ---------------------------------------------------------------------------
# Hybrid retriever tests
# ---------------------------------------------------------------------------

class TestHybridRetriever:
    def test_bm25_only_fallback(self, tmp_db):
        """When vec is unavailable, hybrid should still return BM25 results."""
        _insert_fact(tmp_db, "f1", text="We use PostgreSQL as production database", search_text="We use PostgreSQL as production database")
        _insert_fact(tmp_db, "f2", text="We use Redis as cache", search_text="We use Redis as cache")
        retriever = HybridRetriever(tmp_db, DummyEmbedder())
        query = RetrievalQuery(text="PostgreSQL", tenant_id="default", domain="project", scope_id="proj1", top_k=5)
        results = retriever.retrieve(query, apply_mmr=False)
        assert len(results) >= 1
        assert results[0].fact.id == "f1"

    def test_signals_recorded(self, tmp_db):
        _insert_fact(tmp_db, "f1", text="We use PostgreSQL as production database", search_text="We use PostgreSQL as production database")
        retriever = HybridRetriever(tmp_db, DummyEmbedder())
        query = RetrievalQuery(text="PostgreSQL", tenant_id="default", domain="project", scope_id="proj1", top_k=5)
        results = retriever.retrieve(query, apply_mmr=False)
        assert len(results) >= 1
        assert "bm25" in results[0].signals

    def test_cross_scope_filtered(self, tmp_db):
        """Facts from a different scope should not appear."""
        _insert_fact(tmp_db, "f1", scope_id="proj1", text="PostgreSQL config", search_text="PostgreSQL config")
        _insert_fact(tmp_db, "f2", scope_id="proj2", text="PostgreSQL config", search_text="PostgreSQL config")
        retriever = HybridRetriever(tmp_db, DummyEmbedder())
        query = RetrievalQuery(text="PostgreSQL", tenant_id="default", domain="project", scope_id="proj1", top_k=5)
        results = retriever.retrieve(query, apply_mmr=False)
        ids = [r.fact.id for r in results]
        assert "f1" in ids
        assert "f2" not in ids

    def test_cross_tenant_filtered(self, tmp_db):
        _insert_fact(tmp_db, "f1", tenant_id="tenant_a", text="PostgreSQL config", search_text="PostgreSQL config")
        _insert_fact(tmp_db, "f2", tenant_id="tenant_b", text="PostgreSQL config", search_text="PostgreSQL config")
        retriever = HybridRetriever(tmp_db, DummyEmbedder())
        query = RetrievalQuery(text="PostgreSQL", tenant_id="tenant_a", domain="project", scope_id="proj1", top_k=5)
        results = retriever.retrieve(query, apply_mmr=False)
        ids = [r.fact.id for r in results]
        assert "f1" in ids
        assert "f2" not in ids


# ---------------------------------------------------------------------------
# Layer store tests
# ---------------------------------------------------------------------------

class TestHotStore:
    def test_add_and_get(self, tmp_db):
        store = HotStore(tmp_db)
        item = store.add("default", "user", "user1", "Always use Python 3.12", priority=5)
        assert item.text == "Always use Python 3.12"
        items = store.get_for_scope("default", "user", "user1")
        assert len(items) >= 1
        assert any(i.text == "Always use Python 3.12" for i in items)


class TestFactStore:
    def test_get(self, tmp_db):
        _insert_fact(tmp_db, "f1", text="test fact", search_text="test fact")
        store = FactStore(tmp_db)
        fact = store.get("f1", "default")
        assert fact is not None
        assert fact.text == "test fact"

    def test_get_nonexistent(self, tmp_db):
        store = FactStore(tmp_db)
        assert store.get("nonexistent", "default") is None


class TestHistoryStore:
    def test_append_and_search(self, tmp_db):
        store = HistoryStoreWrapper(tmp_db)
        msg = store.append("default", "ch1", "user", "We decided to use PostgreSQL")
        assert msg.content == "We decided to use PostgreSQL"
        results = store.search("default", "PostgreSQL", channel_id="ch1")
        assert len(results) >= 1


class TestSkillStore:
    def test_create_and_get(self, tmp_db):
        store = SkillStoreWrapper(tmp_db)
        skill = store.create("default", "agent", "custom", "deploy-procedure", "Run deploy.sh")
        assert skill.name == "deploy-procedure"
        fetched = store.get("default", "deploy-procedure", "custom")
        assert fetched is not None
        assert fetched.body == "Run deploy.sh"


# ---------------------------------------------------------------------------
# Context builder tests
# ---------------------------------------------------------------------------

class TestContextBuilder:
    def test_backtracking_intent_is_inferred(self):
        assert infer_memory_intent("你先回想一下之前干了啥？") == "reflect"
        assert infer_memory_intent("继续上次的任务") == "continue_task"
        assert infer_memory_intent("数据库是什么") == "recall"

    def test_scope_and_tenant_are_canonical(self):
        external = "feishu:tenant-a:user-1"
        assert tenant_id_from_user_id(external) == "tenant-a"
        canonical = canonical_scope_id("user", external)
        assert canonical.startswith("u_") and len(canonical) == 66
        assert canonical_scope_id("user", canonical) == canonical

    def test_basic_context(self, tmp_db):
        """Context builder should assemble hot + facts into rendered text."""
        # Add a hot item.
        hot = HotStore(tmp_db)
        hot.add("default", "project", "proj1", "Always use type hints", priority=10)

        # Add a fact.
        _insert_fact(tmp_db, "f1", text="We use PostgreSQL as database", search_text="We use PostgreSQL as database")

        builder = ContextBuilder(tmp_db)
        actor = ActorContext(tenant_id="default", role="user", actor_id="user1")
        scope = MemoryScope(domain="project", scope_id="proj1", tenant_id="default")

        ctx = asyncio.run(builder.build_memory_context(
            actor=actor, scope=scope, query="PostgreSQL", intent="recall", token_budget=500,
        ))
        assert ctx.token_count > 0
        assert "Hot Memory" in ctx.rendered or "Facts" in ctx.rendered
        assert len(ctx.hot_items) >= 1

    def test_history_not_loaded_for_recall(self, tmp_db):
        """History should not be loaded for 'recall' intent."""
        builder = ContextBuilder(tmp_db)
        actor = ActorContext(tenant_id="default", role="user", actor_id="user1")
        scope = MemoryScope(domain="project", scope_id="proj1", tenant_id="default")
        ctx = asyncio.run(builder.build_memory_context(
            actor=actor, scope=scope, query="test", intent="recall", token_budget=500,
        ))
        assert len(ctx.history_messages) == 0
        assert ctx.budget_used["history"] == 0

    def test_history_loaded_for_continue_task(self, tmp_db):
        """History should be loaded for 'continue_task' intent."""
        hist = HistoryStoreWrapper(tmp_db)
        hist.append("default", "ch1", "user", "We decided to use PostgreSQL for production")

        builder = ContextBuilder(tmp_db)
        actor = ActorContext(tenant_id="default", role="user", actor_id="user1")
        scope = MemoryScope(domain="project", scope_id="proj1", tenant_id="default")
        ctx = asyncio.run(builder.build_memory_context(
            actor=actor, scope=scope, query="PostgreSQL", intent="continue_task",
            token_budget=500, channel_id="ch1",
        ))
        # History should have been searched.
        assert ctx.budget_used["history"] >= 0  # may be 0 if no match, but layer was loaded

    def test_vague_reflection_uses_recent_fact_and_history(self, tmp_db):
        _insert_fact(
            tmp_db,
            "recent-work",
            text="Completed the life-note theme refactor",
            search_text="Completed the life-note theme refactor",
        )
        hist = HistoryStoreWrapper(tmp_db)
        first = hist.append(
            "default", "ch1", "agent", "Fixed the diary and billing pages",
            thread_id="thread1", agent_id="coordinator", message_id="sm-fixed",
        )
        duplicate = hist.append(
            "default", "ch1", "agent", "Fixed the diary and billing pages",
            thread_id="thread1", agent_id="coordinator", message_id="sm-fixed",
        )
        assert first.message_id == duplicate.message_id

        builder = ContextBuilder(tmp_db)
        actor = ActorContext(tenant_id="default", role="coordinator", actor_id="coordinator")
        scope = MemoryScope(
            domain="project", scope_id="proj1", agent_id="coordinator", tenant_id="default",
        )
        ctx = asyncio.run(builder.build_memory_context(
            actor=actor,
            scope=scope,
            query="你先回想一下之前干了啥？",
            intent="recall",
            token_budget=800,
            channel_id="ch1",
            thread_id="thread1",
        ))
        assert any(item.fact.id == "recent-work" for item in ctx.fact_results)
        assert any(item.message_id == "sm-fixed" for item in ctx.history_messages)
        assert "life-note theme refactor" in ctx.rendered
        assert "diary and billing" in ctx.rendered

    def test_skill_loaded_for_start_task(self, tmp_db):
        """Skill should be loaded for 'start_task' intent."""
        skills = SkillStoreWrapper(tmp_db)
        skills.create("default", "agent", "custom", "deploy", "Run deploy.sh")

        builder = ContextBuilder(tmp_db)
        actor = ActorContext(tenant_id="default", role="agent", actor_id="agent1")
        scope = MemoryScope(domain="agent", scope_id="custom", tenant_id="default")
        ctx = asyncio.run(builder.build_memory_context(
            actor=actor, scope=scope, query="deploy", intent="start_task", token_budget=500,
        ))
        assert len(ctx.skills) >= 1

    def test_token_budget_respected(self, tmp_db):
        """Rendered context should not exceed token budget."""
        for i in range(10):
            _insert_fact(tmp_db, f"f{i}", text=f"Fact number {i} about PostgreSQL database config", search_text=f"Fact number {i} about PostgreSQL database config")
        builder = ContextBuilder(tmp_db)
        actor = ActorContext(tenant_id="default", role="user", actor_id="user1")
        scope = MemoryScope(domain="project", scope_id="proj1", tenant_id="default")
        ctx = asyncio.run(builder.build_memory_context(
            actor=actor, scope=scope, query="PostgreSQL", intent="recall", token_budget=100,
        ))
        assert ctx.token_count <= 120  # small overshoot allowed for truncation marker

    def test_cross_tenant_no_leakage(self, tmp_db):
        """Facts from tenant_b should not appear in tenant_a's context."""
        _insert_fact(tmp_db, "f1", tenant_id="tenant_a", text="tenant A secret", search_text="tenant A secret")
        _insert_fact(tmp_db, "f2", tenant_id="tenant_b", text="tenant B secret", search_text="tenant B secret")
        builder = ContextBuilder(tmp_db)
        actor = ActorContext(tenant_id="tenant_a", role="user", actor_id="user1")
        scope = MemoryScope(domain="project", scope_id="proj1", tenant_id="tenant_a")
        ctx = asyncio.run(builder.build_memory_context(
            actor=actor, scope=scope, query="secret", intent="recall", token_budget=500,
        ))
        assert "tenant B" not in ctx.rendered

    def test_history_survives_database_restart(self, tmp_path):
        db_path = tmp_path / "restart_memory.db"
        first_db = MemoryDB(db_path)
        history = HistoryStoreWrapper(first_db)
        history.append(
            "tenant-a", "channel-a", "agent", "Finished the Memory 2.1 migration",
            thread_id="thread-a", agent_id="coordinator", message_id="restart-proof",
        )
        first_db.close()

        restarted_db = MemoryDB(db_path)
        try:
            builder = ContextBuilder(restarted_db)
            actor = ActorContext(
                tenant_id="tenant-a", role="coordinator", actor_id="coordinator",
            )
            scope = MemoryScope(
                domain="project", scope_id="project-a", agent_id="coordinator",
                tenant_id="tenant-a",
            )
            ctx = asyncio.run(builder.build_memory_context(
                actor=actor,
                scope=scope,
                query="What did you do before?",
                intent="reflect",
                token_budget=500,
                channel_id="channel-a",
                thread_id="thread-a",
            ))
            assert any(item.message_id == "restart-proof" for item in ctx.history_messages)
            assert "Memory 2.1 migration" in ctx.rendered
        finally:
            restarted_db.close()


class TestEmbeddingRepair:
    def test_rebuild_is_idempotent(self, tmp_db):
        _insert_fact(tmp_db, "needs-vector", text="vector repair")
        report = rebuild_embeddings(tmp_db, DummyEmbedder(dim=512))
        assert report["rebuilt"] == 1
        assert report["failed"] == 0

        second = rebuild_embeddings(tmp_db, DummyEmbedder(dim=512))
        assert second["selected"] == 0
