"""Qdrant-backed memory store (Memory 2.2, §3) — dense + BM25 sparse.

Each Memory maps to one Qdrant Point that carries TWO vectors in named slots:

- ``""``   (default) : dense 512-dim BGE embedding  → semantic KNN
- ``"bm25"``         : BM25 sparse vector (FastEmbed Qdrant/bm25) → lexical scoring

plus the payload (text/domain/attributed_to/linked_memory_ids/...).

This is the Mem0 route (§3.4 方案 a, refined): BM25 is a real scored sparse
retrieval with Qdrant's ``Modifier.IDF``, not the ``MatchText`` boolean filter.
Both paths are fused with RRF in ``search``.

Qdrant sparse vectors exist since v1.7; the collection is created with
``sparse_vectors_config={"bm25": SparseVectorParams(modifier=Modifier.IDF)}``.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Protocol

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    Modifier,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from memory.v22.models import Memory, MemorySearchResult, memory_hash


logger = logging.getLogger(__name__)


DEFAULT_COLLECTION = "memories"
DEFAULT_EMBED_DIM = 512
BM25_SLOT = "bm25"          # named sparse vector slot in the collection
DENSE_SLOT = ""             # Qdrant default (unnamed) dense vector slot


class DenseEmbedderLike(Protocol):
    """Dense embedder (BGE). Matches ``memory.embedder.Embedder``."""
    def embed(self, text: str) -> list[float]: ...
    @property
    def dim(self) -> int: ...


class SparseEmbedderLike(Protocol):
    """BM25 sparse embedder. Matches ``fastembed.SparseTextEmbedding``."""
    def embed(self, text: str):  # -> Iterable[SparseEmbedding]
        ...
    @property
    def model_name(self) -> str: ...


def _to_sparse_vector(sparse_output) -> SparseVector:
    """Normalize a FastEmbed ``SparseEmbedding`` (generator → first item) to a
    Qdrant ``SparseVector``. Accepts either an iterable or a single value."""
    if isinstance(sparse_output, SparseVector):
        return sparse_output
    # fastembed returns a generator over SparseEmbedding; take the first.
    try:
        first = next(iter(sparse_output))
    except StopIteration:
        return SparseVector(indices=[], values=[])
    indices = [int(i) for i in first.indices]
    values = [float(v) for v in first.values]
    return SparseVector(indices=indices, values=values)


class MemoryStore:
    """Qdrant store for flat Memory records, dual-vector per point.

    Lifecycle: ``ensure_collection()`` creates the collection with a dense
    (Cosine, 512-dim) + sparse (``bm25`` slot, IDF modifier) configuration if
    missing. After that, ``upsert`` / ``search`` / ``exists_by_hash`` are the
    hot-path calls.
    """

    def __init__(
        self,
        client: QdrantClient,
        dense_embedder: DenseEmbedderLike,
        sparse_embedder: SparseEmbedderLike | None,
        *,
        collection_name: str = DEFAULT_COLLECTION,
        vector_size: int = DEFAULT_EMBED_DIM,
    ) -> None:
        self._client = client
        self._dense = dense_embedder
        self._sparse = sparse_embedder
        self._collection = collection_name
        self._vector_size = vector_size
        self._ready = False

    # ------------------------------------------------------------------
    # collection setup
    # ------------------------------------------------------------------
    def ensure_collection(self) -> None:
        """Create the collection (dense + bm25 sparse slots) if missing."""
        if self._ready:
            return
        if not self._client.collection_exists(self._collection):
            sparse_cfg = (
                {BM25_SLOT: SparseVectorParams(modifier=Modifier.IDF)}
                if self._sparse is not None else None
            )
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._vector_size, distance=Distance.COSINE,
                ),
                sparse_vectors_config=sparse_cfg,
            )
            logger.info(
                "created Qdrant collection %s (dense %dd + bm25 sparse)",
                self._collection, self._vector_size,
            )
        self._ready = True

    # ------------------------------------------------------------------
    # writes
    # ------------------------------------------------------------------
    def upsert(self, memory: Memory) -> Memory:
        """Write one Memory as a Qdrant Point carrying dense + bm25 vectors."""
        self.ensure_collection()
        dense_vec = self._dense.embed(memory.text)
        if len(dense_vec) != self._vector_size:
            raise ValueError(
                f"dense embedding dim {len(dense_vec)} != collection size {self._vector_size}"
            )
        vectors: dict[str, Any] = {DENSE_SLOT: dense_vec}
        if self._sparse is not None:
            vectors[BM25_SLOT] = _to_sparse_vector(self._sparse.embed(memory.text))
        point = PointStruct(
            id=memory.id,
            vector=vectors,
            payload=memory.to_payload(),
        )
        self._client.upsert(collection_name=self._collection, points=[point])
        return memory

    def exists_by_hash(self, domain: str, text: str) -> bool:
        """True if a Memory with this ``(domain, hash)`` already exists."""
        self.ensure_collection()
        h = memory_hash(text)
        filt = Filter(must=[
            FieldCondition(key="domain", match=MatchAny(any=[domain])),
            FieldCondition(key="hash", match=MatchAny(any=[h])),
        ])
        hits, _ = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=filt,
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return len(hits) > 0

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        *,
        domains: list[str] | None = None,
        top_k: int = 8,
        weight_vector: float = 0.5,
        weight_bm25: float = 0.5,
    ) -> list[MemorySearchResult]:
        """Dual-path retrieval fused with RRF.

        1. Dense vector KNN on the query embedding, filtered by domain.
        2. BM25 sparse scoring on the ``bm25`` slot, filtered by domain.
        3. RRF-fuse the two ranked lists, fetch payloads for the top ids.

        If ``sparse_embedder`` was not provided at construction, the BM25 path
        is skipped and only dense KNN runs (its weight is renormalized).
        """
        self.ensure_collection()
        if not domains:
            domains = ["project", "user", "task", "agent"]
        domain_filter = Filter(must=[
            FieldCondition(key="domain", match=MatchAny(any=domains))
        ])

        ranked: list[tuple[list[str], float]] = []
        vec_ids: list[str] = []
        bm25_ids: list[str] = []

        # --- dense vector KNN ---
        if weight_vector > 0:
            try:
                qv = self._dense.embed(query)
                if len(qv) == self._vector_size:
                    res = self._client.query_points(
                        collection_name=self._collection,
                        query=qv,
                        query_filter=domain_filter,
                        limit=top_k * 2,
                        with_payload=False,
                        with_vectors=False,
                    )
                    vec_ids = [str(p.id) for p in res.points]
            except Exception:
                logger.exception("dense KNN failed during memory search")
            if vec_ids:
                ranked.append((vec_ids, weight_vector))

        # --- BM25 sparse scoring (real scored retrieval, not boolean) ---
        if self._sparse is not None and weight_bm25 > 0:
            try:
                q_sparse = _to_sparse_vector(self._sparse.embed(query))
                if q_sparse.indices:
                    res = self._client.query_points(
                        collection_name=self._collection,
                        query=q_sparse,
                        using=BM25_SLOT,
                        query_filter=domain_filter,
                        limit=top_k * 2,
                        with_payload=False,
                        with_vectors=False,
                    )
                    bm25_ids = [str(p.id) for p in res.points]
            except Exception:
                logger.exception("BM25 sparse query failed during memory search")
            if bm25_ids:
                ranked.append((bm25_ids, weight_bm25))

        if not ranked:
            return []

        # --- fuse + fetch ---
        fused = _rrf_fuse(ranked)[:top_k]
        if not fused:
            return []
        ids = [mid for mid, _ in fused]
        score_map = {mid: score for mid, score in fused}

        points = self._client.retrieve(
            collection_name=self._collection,
            ids=ids,
            with_payload=True,
            with_vectors=False,
        )
        signal_map: dict[str, list[str]] = {mid: [] for mid in ids}
        for mid in vec_ids:
            signal_map.setdefault(mid, []).append("vector")
        for mid in bm25_ids:
            signal_map.setdefault(mid, []).append("bm25")

        results: list[MemorySearchResult] = []
        for p in points:
            pid = str(p.id)
            mem = Memory.from_payload(pid, p.payload or {})
            results.append(MemorySearchResult(
                memory=mem,
                score=score_map.get(pid, 0.0),
                signals=signal_map.get(pid, []),
            ))
        order = {mid: i for i, (mid, _) in enumerate(fused)}
        results.sort(key=lambda r: order.get(r.memory.id, 999))
        return results


def _rrf_fuse(
    ranked_lists: list[tuple[list[str], float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion. Each input is (ordered ids, weight). Returns
    (id, fused_score) sorted desc."""
    scores: dict[str, float] = {}
    for ids, weight in ranked_lists:
        for rank, mid in enumerate(ids):
            scores[mid] = scores.get(mid, 0.0) + weight / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])
