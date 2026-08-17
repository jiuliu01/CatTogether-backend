"""Hot layer store — small, stable, always-injected (§5.8).

Thin wrapper over the existing ``HotMemoryStore`` providing the final
interface used by the context builder.

Hot items are capacity-limited per scope and have an approval workflow:
items below the auto-approve threshold enter ``pending_approval`` until
a human (or coordinator) approves them.
"""
from __future__ import annotations

from typing import Any

from memory.db import MemoryDB, memory_db
from memory.hot_memory_store import HotMemoryStore
from memory.models import HotMemoryItem
from memory.scope import MemoryScope


class HotStore:
    """Hot memory layer (§5.8).

    Wraps ``HotMemoryStore`` with the final interface used by the
    context builder and the API layer.
    """

    def __init__(self, db: MemoryDB | None = None) -> None:
        self._db = db or memory_db
        self._store = HotMemoryStore(self._db)

    def get_for_scope(
        self,
        tenant_id: str,
        domain: str,
        scope_id: str,
        *,
        limit: int | None = None,
    ) -> list[HotMemoryItem]:
        """Get active hot items for a scope, ordered by priority then recency."""
        items = self._store.list_hot(
            tenant_id=tenant_id,
            domain=domain,  # type: ignore[arg-type]
            scope_id=scope_id,
        )
        if limit is not None:
            items = items[:limit]
        return items

    def add(
        self,
        tenant_id: str,
        domain: str,
        scope_id: str,
        text: str,
        *,
        priority: int = 0,
        source_fact_ids: list[str] | None = None,
        proposed_by: str = "system",
    ) -> HotMemoryItem:
        """Add a hot memory item.

        ``priority`` is mapped to an importance value for the underlying
        store (higher priority → higher importance → auto-approve).
        """
        importance = min(1.0, 0.5 + priority * 0.1)
        source_fact_id = source_fact_ids[0] if source_fact_ids else None
        return self._store.add_hot(
            tenant_id=tenant_id,
            domain=domain,  # type: ignore[arg-type]
            scope_id=scope_id,
            text=text,
            importance=importance,
            source_fact_id=source_fact_id,
        )

    def approve(self, item_id: str, approved_by: str) -> bool:
        """Approve a pending hot item."""
        return self._store.approve_hot(item_id, approved_by)

    def archive(self, item_id: str) -> bool:
        """Archive (soft-delete) a hot item."""
        return self._store.archive_hot(item_id)

    def delete(self, item_id: str, tenant_id: str) -> bool:
        """Delete a hot item (hard delete)."""
        conn = self._db.connect()
        cur = conn.execute(
            "DELETE FROM hot_memories WHERE id = ? AND tenant_id = ?",
            (item_id, tenant_id),
        )
        return cur.rowcount > 0
