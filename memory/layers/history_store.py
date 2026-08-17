"""History layer store — session messages, backtracking (§5.9).

Thin wrapper over the existing ``HistoryStore`` providing the final
interface used by the context builder.  History is only loaded for
backtracking intents (``continue_task``, ``reflect``, ``audit``).
"""
from __future__ import annotations

from memory.db import MemoryDB, memory_db
from memory.history_store import HistoryStore as _HistoryStoreImpl
from memory.models import SessionMessage


class HistoryStoreWrapper:
    """History layer (§5.9).

    Wraps ``HistoryStore`` with the final interface used by the context
    builder.  Default budget is 0 — only loaded when intent requires it.
    """

    def __init__(self, db: MemoryDB | None = None) -> None:
        self._db = db or memory_db
        self._store = _HistoryStoreImpl(self._db)

    def append(
        self,
        tenant_id: str,
        channel_id: str,
        role: str,
        content: str,
        *,
        thread_id: str | None = None,
        agent_id: str | None = None,
        search_text: str | None = None,
        message_id: str | None = None,
    ) -> SessionMessage:
        """Append a message to the history layer."""
        return self._store.append_message(
            tenant_id=tenant_id,
            channel_id=channel_id,
            role=role,  # type: ignore[arg-type]
            content=content,
            thread_id=thread_id,
            agent_id=agent_id,
            message_id=message_id,
        )

    def search(
        self,
        tenant_id: str,
        query: str,
        *,
        channel_id: str | None = None,
        thread_id: str | None = None,
        limit: int = 20,
    ) -> list[SessionMessage]:
        """FTS5 search over session messages."""
        msgs = self._store.search_history(
            tenant_id=tenant_id,
            query=query,
            channel_id=channel_id,
            top_k=limit,
        )
        if thread_id is not None:
            msgs = [m for m in msgs if getattr(m, "thread_id", None) == thread_id]
        return msgs

    def list_messages(
        self,
        tenant_id: str,
        channel_id: str,
        *,
        thread_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[SessionMessage]:
        """List messages in order (for scrolling)."""
        return self._store.list_messages(
            tenant_id=tenant_id,
            channel_id=channel_id,
            thread_id=thread_id,
            limit=limit,
            offset=offset,
        )


# Alias for the __init__.py export name.
HistoryStore = HistoryStoreWrapper
