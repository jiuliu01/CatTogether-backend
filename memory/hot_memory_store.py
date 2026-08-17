"""Hot Memory store (first layer, R7).

Small, stable, always-injected facts that have been promoted to the hot layer
for instant context.  Capacity-limited per scope (≤ ``memory_hot_max_per_scope``,
default 50).  Items above the auto-approve threshold are immediately active;
below it they enter ``pending_approval`` status until a human approves.

Promotion path: a Fact with high importance/confidence is promoted to Hot
Memory via ``add_hot``.  The context builder reads hot items first, before any
Fact/History/Skill retrieval.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Literal

from config import settings
from memory.db import MemoryDB, memory_db
from memory.models import HotMemoryItem, HotCategory, HotStatus, FactDomain
from memory.scope import MemoryScope
from models.schemas import MemoryEntry


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


class HotMemoryStore:
    """Hot Memory layer: small, stable, always-injected (R7).

    Capacity: ≤ ``settings.memory_hot_max_per_scope`` per (tenant, domain, scope).
    When capacity is exceeded, the lowest-importance ``active`` item is evicted
    (set to ``archived``).
    """

    def __init__(self, db: MemoryDB | None = None) -> None:
        self._db = db or memory_db

    @property
    def max_per_scope(self) -> int:
        return settings.memory_hot_max_per_scope

    @property
    def auto_approve_threshold(self) -> float:
        return settings.memory_hot_auto_approve_threshold

    # ----- Core operations -----

    def add_hot(
        self,
        tenant_id: str,
        domain: FactDomain,
        scope_id: str,
        text: str,
        category: HotCategory = "note",
        importance: float = 0.8,
        source_fact_id: str | None = None,
    ) -> HotMemoryItem:
        """Add a hot-memory item. Auto-approves if importance >= threshold."""
        item_id = f"hot_{uuid.uuid4().hex[:16]}"
        now = _utc_now_iso()
        # Auto-approve if importance is high enough.
        if importance >= self.auto_approve_threshold:
            status: HotStatus = "active"
        else:
            status = "pending_approval"
        # v4: category maps to priority; source_fact_id maps to source_fact_ids.
        priority_map = {"note": 0, "decision": 1, "reminder": 2, "summary": 3}
        priority = priority_map.get(category, 0)
        source_fact_ids = json.dumps([source_fact_id] if source_fact_id else [])
        conn = self._db.connect()
        conn.execute("BEGIN")
        try:
            conn.execute(
                """INSERT INTO hot_memories
                   (id, tenant_id, domain, scope_id, text, priority,
                    source_fact_ids, status, proposed_by, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (item_id, tenant_id, domain, scope_id, text, priority,
                 source_fact_ids, status, "system", now, now),
            )
            self._enforce_capacity(conn, tenant_id, domain, scope_id)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        return HotMemoryItem(
            id=item_id,
            tenant_id=tenant_id,
            domain=domain,
            scope_id=scope_id,
            text=text,
            priority=priority,
            source_fact_ids=[source_fact_id] if source_fact_id else [],
            status=status,
            proposed_by="system",
            created_at=_utc_now(),
            updated_at=_utc_now(),
        )

    def _enforce_capacity(
        self,
        conn,
        tenant_id: str,
        domain: str,
        scope_id: str,
    ) -> None:
        """Evict lowest-priority active items if over capacity."""
        count = conn.execute(
            """SELECT COUNT(*) FROM hot_memories
               WHERE tenant_id=? AND domain=? AND scope_id=? AND status='active'""",
            (tenant_id, domain, scope_id),
        ).fetchone()[0]
        if count <= self.max_per_scope:
            return
        excess = count - self.max_per_scope
        conn.execute(
            """UPDATE hot_memories SET status='archived', updated_at=?
               WHERE id IN (
                 SELECT id FROM hot_memories
                 WHERE tenant_id=? AND domain=? AND scope_id=? AND status='active'
                 ORDER BY priority DESC, updated_at ASC
                 LIMIT ?
               )""",
            (_utc_now_iso(), tenant_id, domain, scope_id, excess),
        )

    def list_hot(
        self,
        tenant_id: str,
        domain: FactDomain | None = None,
        scope_id: str | None = None,
        *,
        include_pending: bool = False,
    ) -> list[HotMemoryItem]:
        """List active hot items, ordered by importance desc."""
        clauses = ["tenant_id = ?"]
        params: list = [tenant_id]
        if domain is not None:
            clauses.append("domain = ?")
            params.append(domain)
        if scope_id is not None:
            clauses.append("scope_id = ?")
            params.append(scope_id)
        if include_pending:
            clauses.append("status IN ('active', 'pending_approval')")
        else:
            clauses.append("status = 'active'")
        where = " AND ".join(clauses)
        rows = self._db.query_all(
            f"""SELECT * FROM hot_memories WHERE {where}
                ORDER BY priority ASC, updated_at DESC""",
            tuple(params),
        )
        return [self._row_to_item(r) for r in rows]

    def approve_hot(self, item_id: str, approved_by: str) -> bool:
        """Approve a pending item → status='active'."""
        now = _utc_now_iso()
        conn = self._db.connect()
        conn.execute("BEGIN")
        try:
            cur = conn.execute(
                """UPDATE hot_memories
                   SET status='active', approved_by=?, updated_at=?
                   WHERE id=? AND status='pending_approval'""",
                (approved_by, now, item_id),
            )
            conn.execute("COMMIT")
            return cur.rowcount > 0
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def archive_hot(self, item_id: str) -> bool:
        """Archive a hot item (soft delete)."""
        now = _utc_now_iso()
        conn = self._db.connect()
        conn.execute("BEGIN")
        try:
            cur = conn.execute(
                "UPDATE hot_memories SET status='archived', updated_at=? WHERE id=?",
                (now, item_id),
            )
            conn.execute("COMMIT")
            return cur.rowcount > 0
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def _row_to_item(self, row) -> HotMemoryItem:
        return HotMemoryItem(
            id=row["id"],
            tenant_id=row["tenant_id"],
            domain=row["domain"],
            scope_id=row["scope_id"],
            text=row["text"],
            priority=row["priority"],
            source_fact_ids=json.loads(row["source_fact_ids"]) if row["source_fact_ids"] else [],
            status=row["status"],
            proposed_by=row["proposed_by"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            approved_by=row["approved_by"],
        )

    # ----- HotMemoryAPI Protocol (scope-based) -----

    async def list_hot_api(self, scope: MemoryScope) -> list[MemoryEntry]:
        """HotMemoryAPI.list_hot — returns MemoryEntry for compat."""
        tenant = scope.tenant_id or "default"
        domain = scope.domain
        if domain == "workspace":
            domain = "project"
        items = self.list_hot(tenant, domain, scope.scope_id)  # type: ignore[arg-type]
        return [self._item_to_entry(item) for item in items]

    async def add_hot_api(
        self, scope: MemoryScope, text: str, category: str, importance: float = 0.8,
    ) -> MemoryEntry:
        """HotMemoryAPI.add_hot."""
        tenant = scope.tenant_id or "default"
        domain = scope.domain
        if domain == "workspace":
            domain = "project"
        item = self.add_hot(
            tenant, domain, scope.scope_id, text,  # type: ignore[arg-type]
            category=category,  # type: ignore[arg-type]
            importance=importance,
        )
        return self._item_to_entry(item)

    async def approve_hot_api(self, scope: MemoryScope, item_id: str, approved_by: str) -> bool:
        return self.approve_hot(item_id, approved_by)

    async def archive_hot_api(self, scope: MemoryScope, item_id: str) -> bool:
        return self.archive_hot(item_id)

    @staticmethod
    def _item_to_entry(item: HotMemoryItem) -> MemoryEntry:
        # v4: HotMemoryItem has priority (int) instead of importance (float).
        # Map priority → importance for legacy MemoryEntry compat.
        importance = min(1.0, item.priority / 3.0) if item.priority > 0 else 0.5
        return MemoryEntry(
            id=item.id,
            domain=item.domain,
            scope_id=item.scope_id,
            kind="preference",
            text=item.text,
            importance=importance,
            status="active" if item.status == "active" else "archived",
            tenant_id=item.tenant_id,
            schema_version=4,
        )
