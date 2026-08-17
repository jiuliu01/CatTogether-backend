"""SQLite implementation of the memory storage backend (Stage 1, §6.5).

This backend stores Facts in the ``facts`` table with an FTS5 external-content
index (``facts_fts``) for BM25 retrieval.  It implements the v1
``StorageBackend`` ABC (list / recall / upsert / delete) so it can drop in
behind the existing ``memory_manager`` without changes, plus v3-native Fact
operations (insert_fact / get_fact / patch_fact_metadata / search_bm25) that
the nine-stage write pipeline and three-signal retrieval will use directly.

ADD-only semantics (R4):
- ``insert_fact`` always INSERTs a new row (never overwrites).
- ``patch_fact_metadata`` changes only metadata columns (tags / importance /
  confidence / status / expires_at), never ``text``.
- ``delete`` is a soft delete: ``status = 'archived'``.

BM25 retrieval (§7, Stage 1 weight = 1.0):
    SELECT f.*, bm25(facts_fts) AS score
    FROM facts_fts
    JOIN facts f ON f.rowid = facts_fts.rowid
    WHERE facts_fts MATCH :query
      AND f.tenant_id = :tenant
      AND f.domain = :domain
      AND f.status = 'active'
    ORDER BY score
    LIMIT :top_k

The FTS5 virtual table and sync triggers are created by ``v3_schema.sql``;
we never touch them directly.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from memory.db import MemoryDB, memory_db
from memory.models import (
    Fact,
    FactDomain,
    FactStatus,
    ProvenanceInfo,
    RetrievalQuery,
    RetrievalResult,
    StorageBackend,
    TemporalInfo,
)
from memory.scope import MemoryScope
from models.schemas import MemoryEntry


# ---------------------------------------------------------------------------
# Row <-> Fact conversion
# ---------------------------------------------------------------------------

def _dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def _json_loads(value: str | None, default: Any = None) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def _row_to_fact(row: sqlite3.Row) -> Fact:
    """Convert a ``facts`` row to a ``Fact`` model (v4 schema)."""
    temporal_raw = row["temporal"]
    temporal = TemporalInfo.model_validate(_json_loads(temporal_raw, {})) if temporal_raw else None
    provenance_raw = _json_loads(row["provenance"], {})
    provenance = ProvenanceInfo.model_validate(provenance_raw) if provenance_raw else ProvenanceInfo()
    return Fact(
        id=row["id"],
        tenant_id=row["tenant_id"],
        domain=row["domain"],
        scope_id=row["scope_id"],
        agent_id=row["agent_id"],
        task_id=row["task_id"],
        kind=row["kind"],
        text=row["text"],
        search_text=row["search_text"],
        tags=_json_loads(row["tags"], []),
        importance=row["importance"],
        confidence=row["confidence"],
        status=row["status"],
        temporal=temporal,
        provenance=provenance,
        embedding_model=row["embedding_model"],
        created_at=_dt(row["created_at"]) or datetime.now(timezone.utc),
        updated_at=_dt(row["updated_at"]) or datetime.now(timezone.utc),
        expires_at=_dt(row["expires_at"]),
        version=row["version"],
        schema_version=row["schema_version"],
    )


def _fact_to_row_params(fact: Fact) -> tuple[Any, ...]:
    """Convert a ``Fact`` to a parameter tuple for INSERT (v4 schema)."""
    return (
        fact.id,
        fact.tenant_id,
        fact.domain,
        fact.scope_id,
        fact.agent_id,
        fact.task_id,
        fact.kind,
        fact.text,
        fact.search_text,
        json.dumps(fact.tags, ensure_ascii=False),
        fact.importance,
        fact.confidence,
        fact.status,
        json.dumps(fact.temporal.model_dump(mode="json"), ensure_ascii=False) if fact.temporal else None,
        json.dumps(fact.provenance.model_dump(mode="json"), ensure_ascii=False),
        fact.embedding_model,
        fact.created_at.isoformat(),
        fact.updated_at.isoformat(),
        fact.expires_at.isoformat() if fact.expires_at else None,
        fact.version,
        fact.schema_version,
    )


_INSERT_SQL = """
INSERT INTO facts (
  id, tenant_id, domain, scope_id, agent_id, task_id, kind, text, search_text, tags,
  importance, confidence, status, temporal, provenance, embedding_model,
  created_at, updated_at, expires_at, version, schema_version
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


# ---------------------------------------------------------------------------
# SQLiteStorageBackend
# ---------------------------------------------------------------------------

class SQLiteStorageBackend(StorageBackend):
    """Fact storage backed by SQLite + FTS5 (Stage 1).

    Implements both the v1 ``StorageBackend`` ABC (for backward compatibility
    with ``memory_manager``) and v3-native Fact operations used by the
    nine-stage pipeline.
    """

    def __init__(self, db: MemoryDB | None = None) -> None:
        self._db = db or memory_db

    @property
    def db(self) -> MemoryDB:
        return self._db

    # ----- v3-native Fact operations -----

    def insert_fact(self, fact: Fact) -> Fact:
        """INSERT a new fact row (ADD-only, R4). Never overwrites."""
        conn = self._db.connect()
        conn.execute("BEGIN")
        try:
            conn.execute(_INSERT_SQL, _fact_to_row_params(fact))
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        return fact

    def get_fact(self, fact_id: str, tenant_id: str = "default") -> Fact | None:
        row = self._db.query_one(
            "SELECT * FROM facts WHERE id = ? AND tenant_id = ?",
            (fact_id, tenant_id),
        )
        return _row_to_fact(row) if row else None

    def list_facts(
        self,
        tenant_id: str,
        domain: FactDomain | None = None,
        scope_id: str | None = None,
        *,
        include_inactive: bool = False,
        limit: int = 1000,
    ) -> list[Fact]:
        """List facts by tenant/domain/scope, filtered by status."""
        clauses = ["tenant_id = ?"]
        params: list[Any] = [tenant_id]
        if domain is not None:
            clauses.append("domain = ?")
            params.append(domain)
        if scope_id is not None:
            clauses.append("scope_id = ?")
            params.append(scope_id)
        if not include_inactive:
            clauses.append("status = 'active'")
        where = " AND ".join(clauses)
        rows = self._db.query_all(
            f"SELECT * FROM facts WHERE {where} ORDER BY updated_at DESC LIMIT ?",
            (*params, limit),
        )
        return [_row_to_fact(r) for r in rows]

    def patch_fact_metadata(
        self,
        fact_id: str,
        tenant_id: str,
        *,
        tags: list[str] | None = None,
        importance: float | None = None,
        confidence: float | None = None,
        status: FactStatus | None = None,
        expires_at: datetime | None = None,
    ) -> Fact | None:
        """Update only metadata columns (R4: never changes text)."""
        sets: list[str] = []
        params: list[Any] = []
        if tags is not None:
            sets.append("tags = ?")
            params.append(json.dumps(tags, ensure_ascii=False))
        if importance is not None:
            sets.append("importance = ?")
            params.append(importance)
        if confidence is not None:
            sets.append("confidence = ?")
            params.append(confidence)
        if status is not None:
            sets.append("status = ?")
            params.append(status)
        if expires_at is not None:
            sets.append("expires_at = ?")
            params.append(expires_at.isoformat())
        if not sets:
            return self.get_fact(fact_id, tenant_id)
        sets.append("updated_at = ?")
        params.append(datetime.now(timezone.utc).isoformat())
        params.extend([fact_id, tenant_id])
        conn = self._db.connect()
        conn.execute("BEGIN")
        try:
            conn.execute(
                f"UPDATE facts SET {', '.join(sets)} WHERE id = ? AND tenant_id = ?",
                params,
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        return self.get_fact(fact_id, tenant_id)

    def soft_delete(self, fact_id: str, tenant_id: str) -> bool:
        """Soft delete: set status='archived' (R4)."""
        result = self.patch_fact_metadata(fact_id, tenant_id, status="archived")
        return result is not None

    def hard_delete(self, fact_id: str, tenant_id: str) -> bool:
        """Hard delete: remove the row entirely. Use only for GDPR forget."""
        conn = self._db.connect()
        conn.execute("BEGIN")
        try:
            cur = conn.execute(
                "DELETE FROM facts WHERE id = ? AND tenant_id = ?",
                (fact_id, tenant_id),
            )
            conn.execute("COMMIT")
            return cur.rowcount > 0
        except Exception:
            conn.execute("ROLLBACK")
            raise

    # ----- BM25 search (FTS5) -----

    def search_bm25(
        self,
        query: str,
        tenant_id: str,
        domain: FactDomain | None = None,
        scope_id: str | None = None,
        top_k: int = 10,
    ) -> list[tuple[Fact, float]]:
        """BM25 search via FTS5. Returns (Fact, bm25_score) pairs.

        The FTS5 ``MATCH`` uses the query text directly; ``unicode61``
        tokenizer handles CJK by treating each character as a token.  We
        wrap the query in double quotes to treat it as a phrase, falling
        back to bare terms on syntax error.
        """
        if not query.strip():
            return []
        # Build the WHERE clause for tenant/domain/scope filtering on the
        # base table, joining through rowid.
        clauses = ["f.tenant_id = ?", "f.status = 'active'"]
        params: list[Any] = [tenant_id]
        if domain is not None:
            clauses.append("f.domain = ?")
            params.append(domain)
        if scope_id is not None:
            clauses.append("f.scope_id = ?")
            params.append(scope_id)
        where = " AND ".join(clauses)
        sql = (
            "SELECT f.*, bm25(facts_fts) AS score "
            "FROM facts_fts "
            "JOIN facts f ON f.rowid = facts_fts.rowid "
            f"WHERE facts_fts MATCH ? AND {where} "
            "ORDER BY score LIMIT ?"
        )
        results = self._execute_fts_query(sql, query, params, top_k)
        return [(_row_to_fact(row), row["score"]) for row in results]

    def _execute_fts_query(
        self,
        sql: str,
        query: str,
        params: list[Any],
        top_k: int,
    ) -> list[sqlite3.Row]:
        """Try phrase-quoted MATCH first, fall back to bare-term AND query.

        With the trigram tokenizer a phrase query ``"a b"`` only matches when
        the literal substring ``a b`` appears contiguously — which is rarely
        what we want for multi-word recall.  So when the phrase query returns
        zero rows (or errors) we fall back to bare terms joined by implicit
        AND, letting each word trigram-match independently.
        """
        conn = self._db.connect()
        # FTS5 phrase query: wrap in double quotes, escaping internal quotes.
        safe_query = query.replace('"', '""')
        try:
            rows = conn.execute(sql, (f'"{safe_query}"', *params, top_k)).fetchall()
            if rows:
                return rows
        except sqlite3.OperationalError:
            pass
        # Fallback: bare terms (FTS5 treats as implicit AND).
        # Escape characters that FTS5 treats as operators.
        bare = query.replace('"', "").replace("*", "").replace(":", " ").strip()
        if not bare:
            return []
        try:
            return conn.execute(sql, (bare, *params, top_k)).fetchall()
        except sqlite3.OperationalError:
            return []

    def search_bm25_as_results(
        self,
        query: RetrievalQuery,
    ) -> list[RetrievalResult]:
        """BM25 search returning RetrievalResult objects (for the fusion layer)."""
        hits = self.search_bm25(
            query.text,
            query.tenant_id,
            domain=query.domain,
            scope_id=query.scope_id,
            top_k=query.top_k,
        )
        return [
            RetrievalResult(fact=fact, score=float(score), signals=["bm25"])
            for fact, score in hits
        ]

    # ----- StorageBackend ABC (v1 compat) -----

    async def list(self, scope: MemoryScope, *, include_inactive: bool = False) -> list[MemoryEntry]:
        """List facts as MemoryEntry (v1 compat)."""
        tenant = scope.tenant_id or "default"
        domain = scope.domain
        if domain == "workspace":
            domain = "project"
        facts = self.list_facts(
            tenant,
            domain=domain,  # type: ignore[arg-type]
            scope_id=scope.scope_id,
            include_inactive=include_inactive,
        )
        return [self._fact_to_entry(f) for f in facts]

    async def recall(self, scope: MemoryScope, query: str, top_k: int) -> list[MemoryEntry]:
        """BM25 recall returning MemoryEntry (v1 compat)."""
        tenant = scope.tenant_id or "default"
        domain = scope.domain
        if domain == "workspace":
            domain = "project"
        hits = self.search_bm25(
            query,
            tenant,
            domain=domain,  # type: ignore[arg-type]
            scope_id=scope.scope_id,
            top_k=top_k,
        )
        return [self._fact_to_entry(f) for f, _ in hits]

    async def upsert(self, scope: MemoryScope, entry: MemoryEntry) -> MemoryEntry:
        """Upsert via INSERT OR REPLACE (v1 compat, NOT ADD-only).

        The v1 ``upsert`` semantics replace an existing entry with the same id.
        This is used by the v1 write pipeline and the migration script.  The v3
        pipeline uses ``insert_fact`` (ADD-only) instead.
        """
        fact = self._entry_to_fact(entry, scope)
        conn = self._db.connect()
        conn.execute("BEGIN")
        try:
            # Delete existing row with same id (upsert semantics).
            conn.execute(
                "DELETE FROM facts WHERE id = ? AND tenant_id = ?",
                (fact.id, fact.tenant_id),
            )
            conn.execute(_INSERT_SQL, _fact_to_row_params(fact))
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        return entry

    async def delete(self, scope: MemoryScope, entry_id: str) -> bool:
        """Soft delete (v1 compat, R4)."""
        tenant = scope.tenant_id or "default"
        return self.soft_delete(entry_id, tenant)

    # ----- Conversion helpers -----

    @staticmethod
    def _entry_to_fact(entry: MemoryEntry, scope: MemoryScope) -> Fact:
        """Convert a v1 MemoryEntry to a v4 Fact for storage."""
        domain = scope.domain
        if domain == "workspace":
            domain = "project"
        kind = entry.kind
        # v4 removed "practice" and "working_note"; map to "convention".
        if kind in ("practice", "working_note"):
            kind = "convention"
        tenant = entry.tenant_id or scope.tenant_id or "default"
        return Fact(
            id=entry.id,
            tenant_id=tenant,
            domain=domain,  # type: ignore[arg-type]
            scope_id=scope.scope_id,
            kind=kind,  # type: ignore[arg-type]
            text=entry.text,
            search_text=entry.text,
            tags=entry.tags,
            importance=entry.importance,
            confidence=entry.confidence,
            status=entry.status,
            created_at=entry.created_at,
            updated_at=entry.updated_at,
            expires_at=entry.expires_at,
            version=entry.version,
            schema_version=4,
        )

    @staticmethod
    def _fact_to_entry(fact: Fact) -> MemoryEntry:
        """Convert a v4 Fact back to a v1 MemoryEntry for compat consumers."""
        kind = fact.kind
        # MemoryKind still includes "practice" for compat; keep as-is.
        return MemoryEntry(
            id=fact.id,
            domain=fact.domain,
            scope_id=fact.scope_id,
            kind=kind,  # type: ignore[arg-type]
            text=fact.text,
            tags=fact.tags,
            importance=fact.importance,
            confidence=fact.confidence,
            status=fact.status,
            source_type=fact.provenance.source_type,
            source_ids=fact.provenance.source_ids,
            created_at=fact.created_at,
            updated_at=fact.updated_at,
            expires_at=fact.expires_at,
            version=fact.version,
            tenant_id=fact.tenant_id,
            temporal=fact.temporal.model_dump(mode="json") if fact.temporal else None,
            provenance=fact.provenance.model_dump(mode="json"),
            schema_version=4,
        )
