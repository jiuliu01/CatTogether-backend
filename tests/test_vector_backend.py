"""Stage 2 tests: sqlite-vec VectorBackend and Embedder.

Tests are structured so that:
- Embedder tests (DummyEmbedder) always run — no external deps.
- Vector backend tests skip gracefully if sqlite-vec is not installed.

Hybrid retrieval is tested in ``test_retrieval.py`` (three-signal RRF).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# Ensure backend is on sys.path for test isolation.
_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from memory.db import MemoryDB
from memory.embedder import DummyEmbedder, Embedder, get_embedder
from memory.models import Fact, RetrievalQuery
from memory.scope import MemoryScope


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path) -> MemoryDB:
    db_path = tmp_path / "test_vec_memory.db"
    db = MemoryDB(db_path=db_path)
    yield db
    db.close()


@pytest.fixture
def backend(tmp_db):
    from memory.backends.sqlite_backend import SQLiteStorageBackend
    return SQLiteStorageBackend(db=tmp_db)


@pytest.fixture
def vec_backend(tmp_db):
    from memory.backends.sqlite_vec_backend import SqliteVecBackend
    return SqliteVecBackend(db=tmp_db)


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
) -> Fact:
    return Fact(
        id=fid, tenant_id=tenant, domain=domain, scope_id=scope_id,  # type: ignore[arg-type]
        kind=kind, text=text, search_text=text, tags=tags or [],  # type: ignore[arg-type]
        importance=importance, confidence=confidence,  # type: ignore[arg-type]
    )


VEC_AVAILABLE = False
try:
    import sqlite_vec  # noqa: F401
    VEC_AVAILABLE = True
except ImportError:
    pass

skip_if_no_vec = pytest.mark.skipif(
    not VEC_AVAILABLE, reason="sqlite-vec not installed",
)


# ===========================================================================
# §1  Embedder
# ===========================================================================

class TestEmbedder:
    def test_dummy_embedder_dim(self):
        emb = DummyEmbedder(dim=128)
        assert emb.dim == 128
        assert emb.model_name == "dummy-hash"

    def test_dummy_embedder_deterministic(self):
        """Same text → same embedding."""
        emb = DummyEmbedder(dim=64)
        v1 = emb.embed("hello world")
        v2 = emb.embed("hello world")
        assert v1 == v2

    def test_dummy_embedder_different_text(self):
        """Different text → different embedding."""
        emb = DummyEmbedder(dim=64)
        v1 = emb.embed("hello world")
        v2 = emb.embed("goodbye world")
        assert v1 != v2

    def test_dummy_embedder_unit_norm(self):
        """Embedding should be unit-normalised."""
        emb = DummyEmbedder(dim=128)
        v = emb.embed("test text for norm")
        norm = math.sqrt(sum(x * x for x in v))
        assert abs(norm - 1.0) < 1e-6

    def test_dummy_embedder_batch(self):
        emb = DummyEmbedder(dim=64)
        texts = ["alpha", "beta", "gamma"]
        batch = emb.embed_batch(texts)
        assert len(batch) == 3
        for text, vec in zip(texts, batch):
            assert vec == emb.embed(text)

    def test_get_embedder_returns_embedder(self):
        emb = get_embedder()
        assert isinstance(emb, Embedder)
        assert emb.dim > 0


# ===========================================================================
# §2  VectorBackend (sqlite-vec)
# ===========================================================================

@skip_if_no_vec
class TestVectorBackend:
    def test_vec_extension_loaded(self, tmp_db):
        assert tmp_db.vec_available is True

    def test_upsert_and_count(self, vec_backend, backend):
        fact = _make_fact(fid="v1", text="vector test fact")
        backend.insert_fact(fact)
        emb = DummyEmbedder(dim=512)
        vec_backend.upsert_embedding_sync(
            "v1", "default", "project", "p1", emb.embed(fact.text),
        )
        assert vec_backend.count_embeddings("default") == 1

    def test_search_returns_relevant(self, vec_backend, backend):
        """Search with the same embedding should return the fact with high cosine."""
        fact = _make_fact(fid="v2", text="semantic search test")
        backend.insert_fact(fact)
        emb = DummyEmbedder(dim=512)
        vec_emb = emb.embed(fact.text)
        vec_backend.upsert_embedding_sync(
            "v2", "default", "project", "p1", vec_emb,
        )
        results = vec_backend.search_sync(
            vec_emb, "default", domain="project", scope_id="p1", top_k=5,
        )
        assert len(results) >= 1
        assert results[0][0] == "v2"
        # Same vector → cosine similarity ≈ 1.0
        assert results[0][1] > 0.99

    def test_search_filters_tenant(self, vec_backend, backend):
        fact = _make_fact(fid="v3", text="tenant isolated fact")
        backend.insert_fact(fact)
        emb = DummyEmbedder(dim=512)
        vec_backend.upsert_embedding_sync(
            "v3", "default", "project", "p1", emb.embed(fact.text),
        )
        # Search under a different tenant → 0 results
        results = vec_backend.search_sync(
            emb.embed(fact.text), "other_tenant", top_k=5,
        )
        assert len(results) == 0

    def test_search_filters_domain(self, vec_backend, backend):
        fact = _make_fact(fid="v4", text="domain scoped fact", domain="user")
        backend.insert_fact(fact)
        emb = DummyEmbedder(dim=512)
        vec_backend.upsert_embedding_sync(
            "v4", "default", "user", "p1", emb.embed(fact.text),
        )
        # Search in project domain → 0 results
        results = vec_backend.search_sync(
            emb.embed(fact.text), "default", domain="project", top_k=5,
        )
        assert len(results) == 0

    def test_delete_embedding(self, vec_backend, backend):
        fact = _make_fact(fid="v5", text="to be deleted")
        backend.insert_fact(fact)
        emb = DummyEmbedder(dim=512)
        vec_backend.upsert_embedding_sync(
            "v5", "default", "project", "p1", emb.embed(fact.text),
        )
        assert vec_backend.count_embeddings("default") == 1
        assert vec_backend.delete_embedding_sync("v5") is True
        assert vec_backend.count_embeddings("default") == 0

    def test_upsert_replaces_existing(self, vec_backend, backend):
        """Re-upserting the same fact_id replaces the old embedding."""
        fact = _make_fact(fid="v6", text="original text")
        backend.insert_fact(fact)
        emb = DummyEmbedder(dim=512)
        vec_backend.upsert_embedding_sync(
            "v6", "default", "project", "p1", emb.embed("original text"),
        )
        vec_backend.upsert_embedding_sync(
            "v6", "default", "project", "p1", emb.embed("updated text"),
        )
        assert vec_backend.count_embeddings("default") == 1

    def test_async_upsert_and_search(self, vec_backend, backend):
        import asyncio
        fact = _make_fact(fid="v7", text="async vector test")
        backend.insert_fact(fact)
        emb = DummyEmbedder(dim=512)
        scope = MemoryScope(domain="project", scope_id="p1")

        async def _run():
            await vec_backend.upsert_embedding("v7", scope, emb.embed(fact.text))
            return await vec_backend.search(emb.embed(fact.text), scope, top_k=5)

        results = asyncio.run(_run())
        assert len(results) >= 1
        assert results[0][0] == "v7"


# ===========================================================================
# §3  Hybrid Retrieval — moved to test_retrieval.py (three-signal RRF)
# ===========================================================================
