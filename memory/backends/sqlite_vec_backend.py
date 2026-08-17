"""sqlite-vec implementation of VectorBackend (Stage 2, §5 库2, §7 Semantic).

Stores fact embeddings in a ``vec0`` virtual table and provides approximate
nearest-neighbor search.  The vec0 module is loaded by ``MemoryDB.connect()``
when sqlite-vec is available; if not, this backend raises on use.

The vec0 table schema (from ``v3_vector_schema.sql``)::

    CREATE VIRTUAL TABLE fact_embeddings USING vec0(
      embedding FLOAT[512],
      fact_id    TEXT,
      tenant_id  TEXT,
      domain     TEXT,
      scope_id   TEXT
    );

KNN query syntax::

    SELECT fact_id, distance
    FROM fact_embeddings
    WHERE embedding MATCH ?   -- query vector (serialized)
      AND k = ?                -- top-k
      AND tenant_id = ?        -- metadata filter
    ORDER BY distance

sqlite-vec returns L2 (Euclidean) distance.  For unit-normalised embeddings
L2 distance is monotonically related to cosine similarity:
    cosine = 1 - L2² / 2
We convert to cosine similarity in the result mapping.
"""
from __future__ import annotations

import struct
from typing import Any

from memory.db import MemoryDB, memory_db
from memory.models import VectorBackend
from memory.scope import MemoryScope


def _serialize_vec(vec: list[float]) -> bytes:
    """Serialize a float list to the little-endian float32 blob sqlite-vec expects."""
    return struct.pack(f"<{len(vec)}f", *vec)


def _l2_to_cosine(l2_distance: float) -> float:
    """Convert L2 distance to cosine similarity for unit-normalised vectors.

    For unit vectors: ||a - b||² = 2 - 2·cos(θ), so cos(θ) = 1 - L2²/2.
    If vectors aren't unit-normalised this is an approximation.
    """
    return 1.0 - (l2_distance * l2_distance) / 2.0


class SqliteVecBackend(VectorBackend):
    """Vector storage and ANN search backed by sqlite-vec (vec0)."""

    def __init__(self, db: MemoryDB | None = None) -> None:
        self._db = db or memory_db

    @property
    def db(self) -> MemoryDB:
        return self._db

    def _check_available(self) -> None:
        if not self._db.vec_available:
            raise RuntimeError(
                "sqlite-vec extension is not loaded; "
                "pip install sqlite-vec and reconnect"
            )

    # ----- Synchronous core operations -----

    def upsert_embedding_sync(
        self,
        fact_id: str,
        tenant_id: str,
        domain: str,
        scope_id: str,
        embedding: list[float],
    ) -> None:
        """Insert or replace an embedding for a fact.

        vec0 tables don't support UPDATE; we delete-then-insert.
        """
        self._check_available()
        conn = self._db.connect()
        blob = _serialize_vec(embedding)
        conn.execute("BEGIN")
        try:
            conn.execute(
                "DELETE FROM fact_embeddings WHERE fact_id = ?",
                (fact_id,),
            )
            conn.execute(
                """INSERT INTO fact_embeddings
                   (embedding, fact_id, tenant_id, domain, scope_id)
                   VALUES (?, ?, ?, ?, ?)""",
                (blob, fact_id, tenant_id, domain, scope_id),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def search_sync(
        self,
        embedding: list[float],
        tenant_id: str,
        domain: str | None = None,
        scope_id: str | None = None,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """KNN search returning (fact_id, cosine_similarity) pairs.

        sqlite-vec vec0 MATCH returns rows ordered by distance ascending.
        We convert L2 distance → cosine similarity for unit-normalised vectors.
        """
        self._check_available()
        conn = self._db.connect()
        blob = _serialize_vec(embedding)

        # Build metadata filter clause.  vec0 supports `column = ?` in WHERE
        # alongside the MATCH/k constraints.
        extra_clauses: list[str] = []
        params: list[Any] = [blob, top_k, tenant_id]
        # MATCH and k must come first; metadata filters after.
        # vec0 syntax: WHERE embedding MATCH ? AND k = ? AND tenant_id = ?
        base_where = "embedding MATCH ? AND k = ? AND tenant_id = ?"
        if domain is not None:
            extra_clauses.append("domain = ?")
            params.append(domain)
        if scope_id is not None:
            extra_clauses.append("scope_id = ?")
            params.append(scope_id)
        where = base_where
        if extra_clauses:
            where = base_where + " AND " + " AND ".join(extra_clauses)

        sql = (
            "SELECT fact_id, distance "
            "FROM fact_embeddings "
            f"WHERE {where} "
            "ORDER BY distance"
        )
        rows = conn.execute(sql, params).fetchall()
        return [(row["fact_id"], _l2_to_cosine(row["distance"])) for row in rows]

    def delete_embedding_sync(self, fact_id: str) -> bool:
        """Delete an embedding by fact_id."""
        self._check_available()
        conn = self._db.connect()
        cur = conn.execute(
            "DELETE FROM fact_embeddings WHERE fact_id = ?",
            (fact_id,),
        )
        return cur.rowcount > 0

    def count_embeddings(self, tenant_id: str | None = None) -> int:
        """Count stored embeddings (for diagnostics / tests)."""
        self._check_available()
        if tenant_id is None:
            row = self._db.query_one("SELECT COUNT(*) AS n FROM fact_embeddings")
        else:
            row = self._db.query_one(
                "SELECT COUNT(*) AS n FROM fact_embeddings WHERE tenant_id = ?",
                (tenant_id,),
            )
        return row["n"] if row else 0

    # ----- Async VectorBackend ABC -----

    async def upsert_embedding(
        self, fact_id: str, scope: MemoryScope, embedding: list[float],
    ) -> None:
        tenant = scope.tenant_id or "default"
        domain = scope.domain
        if domain == "workspace":
            domain = "project"
        self.upsert_embedding_sync(
            fact_id, tenant, domain, scope.scope_id, embedding,
        )

    async def search(
        self, embedding: list[float], scope: MemoryScope, top_k: int = 10,
    ) -> list[tuple[str, float]]:
        tenant = scope.tenant_id or "default"
        domain = scope.domain
        if domain == "workspace":
            domain = "project"
        return self.search_sync(
            embedding, tenant, domain=domain, scope_id=scope.scope_id, top_k=top_k,
        )

    async def delete_embedding(self, fact_id: str) -> bool:
        return self.delete_embedding_sync(fact_id)
