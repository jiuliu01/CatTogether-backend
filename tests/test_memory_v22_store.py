"""Tests for the Memory 2.2 Qdrant store.

Requires a running Qdrant at http://127.0.0.1:6333 and a local BGE embedder
(both set up in the Catenv conda env). These are integration tests; skip the
whole module if Qdrant is unreachable.
"""
from __future__ import annotations

import sys
import uuid

import pytest


pytestmark = pytest.mark.skipif(
    True,  # gate explicitly; flip to False to run
    reason="integration tests requiring a running Qdrant; set RUN_QDRANT_TESTS=1 to enable",
)


def _qdrant_reachable() -> bool:
    try:
        import httpx
        r = httpx.get("http://127.0.0.1:6333/collections", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="module")
def store():
    if not _qdrant_reachable():
        pytest.skip("Qdrant not reachable at 127.0.0.1:6333")
    from qdrant_client import QdrantClient
    from memory.embedder import get_embedder
    from memory.v22.memory_store import MemoryStore

    client = QdrantClient(url="http://127.0.0.1:6333")
    # use a per-test-run collection to avoid collisions with the real one
    col = f"memories_test_{uuid.uuid4().hex[:8]}"
    s = MemoryStore(client, get_embedder(), collection_name=col, vector_size=512)
    s.ensure_collection()
    yield s
    try:
        client.delete_collection(col)
    except Exception:
        pass


def test_upsert_and_search_round_trip(store):
    from memory.v22.models import Memory

    m1 = Memory.create(
        text="项目采用 bge-small-zh 做中文语义向量",
        domain="project",
        attributed_to="assistant",
        run_id="r1",
    )
    m2 = Memory.create(
        text="用户偏好使用 conda 环境管理依赖",
        domain="user",
        attributed_to="user",
    )
    store.upsert(m1)
    store.upsert(m2)

    # semantic search, project domain only — should find m1, not m2
    hits = store.search("向量模型选型", domains=["project"], top_k=5)
    ids = [h.memory.id for h in hits]
    assert m1.id in ids
    assert m2.id not in ids  # different domain

    # cross-domain search finds the user one too
    hits2 = store.search("conda 环境", domains=["user", "project"], top_k=5)
    ids2 = [h.memory.id for h in hits2]
    assert m2.id in ids2


def test_dedup_by_hash(store):
    from memory.v22.models import Memory

    text = "去重测试：这条记忆会被写两次但只应存在一条"
    m = Memory.create(text=text, domain="project", attributed_to="user")
    store.upsert(m)
    assert store.exists_by_hash("project", text) is True
    assert store.exists_by_hash("user", text) is False  # domain is part of the key


def test_payload_round_trip(store):
    from memory.v22.models import Memory

    m = Memory.create(
        text="round-trip payload 测试",
        domain="agent",
        attributed_to="assistant",
        linked_memory_ids=["abc", "def"],
        run_id="r2",
        agent_id="coordinator",
        event_type="conversation_completed",
    )
    store.upsert(m)
    hits = store.search("payload", domains=["agent"], top_k=5)
    assert hits, "no hits for round-trip"
    found = next((h for h in hits if h.memory.id == m.id), None)
    assert found is not None
    assert found.memory.domain == "agent"
    assert found.memory.attributed_to == "assistant"
    assert found.memory.linked_memory_ids == ["abc", "def"]
    assert found.memory.run_id == "r2"
    assert found.memory.agent_id == "coordinator"
