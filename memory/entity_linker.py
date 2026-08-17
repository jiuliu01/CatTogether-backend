"""Entity Linking for Memory 2.0 (§6 环4, Stage 3).

Resolves entity mentions in fact text to canonical entity records.  The linker
populates three tables:

- ``entities`` — canonical entity main table (entity_id, name, type, aliases)
- ``entity_aliases`` — alias → entity_id mapping for fast lookup
- ``fact_entity_mentions`` — fact_id ↔ entity_id link table

Linking algorithm (§6 环4):
1. Extract candidate mentions from text (regex for proper nouns, tech terms,
   file paths, etc.).
2. For each candidate, check the alias table for an exact match.
3. If no exact match, compute bigram Jaccard similarity against existing entity
   names/aliases; if above threshold (default 0.45), link to the best match.
4. If no match, create a new canonical entity with the mention as its name.

The linker is idempotent: re-linking the same fact produces the same result.
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from memory.db import MemoryDB, memory_db
from memory.models import EntityRef, EntityType, Fact


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _entity_id(name: str, tenant_id: str) -> str:
    """Deterministic entity_id from name + tenant."""
    digest = hashlib.sha256(f"{tenant_id}:{name}".encode("utf-8")).hexdigest()
    return f"ent_{digest[:16]}"


def _bigrams(text: str) -> set[str]:
    """Character bigrams for fuzzy matching (CJK + Latin)."""
    text = text.strip().lower()
    if len(text) < 2:
        return {text} if text else set()
    return {text[i : i + 2] for i in range(len(text) - 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


# Regex patterns for candidate entity extraction.
# These are intentionally simple — LLM-based extraction is a future enhancement.
_PATTERNS: list[tuple[re.Pattern[str], EntityType]] = [
    # File paths: src/main.py, backend/memory/db.py
    (re.compile(r"\b[\w/]+\.\w{1,5}\b"), "file"),
    # Tech terms: CamelCase or known tech keywords
    (re.compile(r"\b(?:SQLite|Python|FastAPI|React|TypeScript|JavaScript|Docker|Redis|PostgreSQL|Node\.js|Vue|Angular)\b"), "tech"),
    # CamelCase identifiers (likely tech/project names)
    (re.compile(r"\b[A-Z][a-z]+[A-Z][a-zA-Z]*\b"), "tech"),
    # All-caps acronyms (2-6 chars)
    (re.compile(r"\b[A-Z]{2,6}\b"), "concept"),
    # Chinese proper-noun-like sequences (2-6 CJK chars between 《》 or after 项目/使用)
    (re.compile(r"《([^》]{1,20})》"), "project"),
]


class EntityLinker:
    """Entity linking: resolve mentions to canonical entities (§6 环4).

    Parameters:
        db: MemoryDB instance (defaults to the module singleton).
        fuzzy_threshold: Jaccard bigram similarity threshold for fuzzy matching.
    """

    def __init__(
        self,
        db: MemoryDB | None = None,
        fuzzy_threshold: float = 0.45,
    ) -> None:
        self._db = db or memory_db
        self._fuzzy_threshold = fuzzy_threshold

    @property
    def db(self) -> MemoryDB:
        return self._db

    # ----- Candidate extraction -----

    def extract_candidates(self, text: str) -> list[tuple[str, EntityType]]:
        """Extract candidate entity mentions from text.

        Returns (name, type) pairs, deduplicated, in order of first appearance.
        """
        seen: set[str] = set()
        candidates: list[tuple[str, EntityType]] = []
        for pattern, etype in _PATTERNS:
            for m in pattern.finditer(text):
                name = m.group(1) if m.lastindex else m.group(0)
                name = name.strip()
                if not name or name.lower() in seen:
                    continue
                seen.add(name.lower())
                candidates.append((name, etype))
        return candidates

    # ----- Linking -----

    def link_fact(self, fact: Fact) -> list[EntityRef]:
        """Link entities in a fact's text and populate edge tables.

        Returns the list of EntityRef objects linked to this fact.
        Idempotent: re-linking the same fact produces the same result.
        """
        candidates = self.extract_candidates(fact.text)
        if not candidates:
            return []

        conn = self._db.connect()
        conn.execute("BEGIN")
        try:
            # Clear existing mentions for this fact (idempotent re-link).
            conn.execute(
                "DELETE FROM fact_entity_mentions WHERE fact_id = ? AND tenant_id = ?",
                (fact.id, fact.tenant_id),
            )

            refs: list[EntityRef] = []
            for name, etype in candidates:
                entity_id = self._resolve_or_create(
                    conn, name, etype, fact.tenant_id,
                )
                role = "subject" if refs == [] else "mentions"
                # Insert mention link.
                conn.execute(
                    """INSERT OR IGNORE INTO fact_entity_mentions
                       (tenant_id, fact_id, entity_id, mention_text, role, linking_confidence)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (fact.tenant_id, fact.id, entity_id, name, role, 0.8),
                )
                refs.append(EntityRef(
                    entity_id=entity_id,
                    name=name,
                    type=etype,
                    role=role,  # type: ignore[arg-type]
                ))

            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        return refs

    def _resolve_or_create(
        self,
        conn: Any,
        name: str,
        etype: EntityType,
        tenant_id: str,
    ) -> str:
        """Resolve a mention to an existing entity, or create a new one.

        Strategy:
        1. Exact match on alias table.
        2. Fuzzy match (bigram Jaccard) against existing entity names.
        3. Create new canonical entity.
        """
        # 1. Exact alias match.
        row = conn.execute(
            "SELECT entity_id FROM entity_aliases WHERE alias = ? AND tenant_id = ?",
            (name, tenant_id),
        ).fetchone()
        if row:
            self._bump_last_seen(conn, row["entity_id"], tenant_id)
            return row["entity_id"]

        # 2. Fuzzy match against existing entity names.
        existing = conn.execute(
            "SELECT entity_id, canonical_name FROM entities WHERE tenant_id = ? AND type = ?",
            (tenant_id, etype),
        ).fetchall()
        name_bigrams = _bigrams(name)
        best_id: str | None = None
        best_sim = 0.0
        for erow in existing:
            sim = _jaccard(name_bigrams, _bigrams(erow["canonical_name"]))
            if sim > best_sim:
                best_sim = sim
                best_id = erow["entity_id"]
        if best_id is not None and best_sim >= self._fuzzy_threshold:
            # Add this name as an alias of the matched entity.
            conn.execute(
                """INSERT OR IGNORE INTO entity_aliases
                   (entity_id, alias, tenant_id) VALUES (?, ?, ?)""",
                (best_id, name, tenant_id),
            )
            self._bump_last_seen(conn, best_id, tenant_id)
            return best_id

        # 3. Create new canonical entity.
        entity_id = _entity_id(name, tenant_id)
        now_iso = _utc_now_iso()
        conn.execute(
            """INSERT OR IGNORE INTO entities
               (entity_id, tenant_id, canonical_name, type, aliases, first_seen_at, last_seen_at)
               VALUES (?, ?, ?, ?, '[]', ?, ?)""",
            (entity_id, tenant_id, name, etype, now_iso, now_iso),
        )
        conn.execute(
            """INSERT OR IGNORE INTO entity_aliases
               (entity_id, alias, tenant_id) VALUES (?, ?, ?)""",
            (entity_id, name, tenant_id),
        )
        return entity_id

    def _bump_last_seen(self, conn: Any, entity_id: str, tenant_id: str) -> None:
        conn.execute(
            "UPDATE entities SET last_seen_at = ? WHERE entity_id = ? AND tenant_id = ?",
            (_utc_now_iso(), entity_id, tenant_id),
        )

    # ----- Query helpers -----

    def get_entity(self, entity_id: str, tenant_id: str = "default") -> dict[str, Any] | None:
        row = self._db.query_one(
            "SELECT * FROM entities WHERE entity_id = ? AND tenant_id = ?",
            (entity_id, tenant_id),
        )
        if not row:
            return None
        return {
            "entity_id": row["entity_id"],
            "name": row["canonical_name"],
            "canonical_name": row["canonical_name"],
            "type": row["type"],
            "aliases": json.loads(row["aliases"]) if row["aliases"] else [],
            "last_seen_at": row["last_seen_at"],
        }

    def list_entities(self, tenant_id: str = "default", limit: int = 100) -> list[dict[str, Any]]:
        rows = self._db.query_all(
            "SELECT * FROM entities WHERE tenant_id = ? ORDER BY last_seen_at DESC LIMIT ?",
            (tenant_id, limit),
        )
        return [
            {
                "entity_id": r["entity_id"],
                "name": r["canonical_name"],
                "canonical_name": r["canonical_name"],
                "type": r["type"],
                "last_seen_at": r["last_seen_at"],
            }
            for r in rows
        ]

    def find_by_alias(self, alias: str, tenant_id: str = "default") -> str | None:
        """Look up an entity_id by exact alias match."""
        row = self._db.query_one(
            "SELECT entity_id FROM entity_aliases WHERE alias = ? AND tenant_id = ?",
            (alias, tenant_id),
        )
        return row["entity_id"] if row else None
