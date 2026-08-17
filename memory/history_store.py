"""History store (third layer, R8).

Complete session records: every user message, agent response, tool call, and
file diff.  Stored in ``session_messages`` with an FTS5 index
(``session_messages_fts``) for on-demand search.  History is NOT
auto-injected into context — the context builder pulls it only when a search
query is issued, keeping the hot path cheap.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from memory.db import MemoryDB, memory_db
from memory.models import SessionMessage, SessionRole
from memory.scope import MemoryScope
from models.schemas import MemoryEntry


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


class HistoryStore:
    """History layer: complete session records, FTS5 on-demand (R8)."""

    def __init__(self, db: MemoryDB | None = None) -> None:
        self._db = db or memory_db

    # ----- Core operations -----

    def append_message(
        self,
        tenant_id: str,
        channel_id: str,
        role: SessionRole,
        content: str,
        *,
        thread_id: str | None = None,
        agent_id: str | None = None,
        tool_call: dict[str, Any] | None = None,
        file_diff: dict[str, Any] | None = None,
        seq: int | None = None,
        message_id: str | None = None,
    ) -> SessionMessage:
        """Append a session message to history."""
        msg_id = message_id or f"sm_{uuid.uuid4().hex[:16]}"
        now = _utc_now_iso()
        if seq is None:
            seq = self._next_seq(tenant_id, channel_id, thread_id)
        conn = self._db.connect()
        conn.execute("BEGIN")
        try:
            conn.execute(
                """INSERT OR IGNORE INTO session_messages
                   (message_id, tenant_id, channel_id, thread_id, seq, role, agent_id,
                    content, search_text, tool_call, file_diff, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    msg_id, tenant_id, channel_id, thread_id, seq, role, agent_id,
                    content, content,
                    json.dumps(tool_call, ensure_ascii=False) if tool_call else None,
                    json.dumps(file_diff, ensure_ascii=False) if file_diff else None,
                    now,
                ),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        existing = self._db.query_one(
            "SELECT * FROM session_messages WHERE tenant_id=? AND message_id=?",
            (tenant_id, msg_id),
        )
        if existing is not None:
            return self._row_to_msg(existing)
        return SessionMessage(
            message_id=msg_id,
            tenant_id=tenant_id,
            channel_id=channel_id,
            thread_id=thread_id,
            seq=seq,
            role=role,
            agent_id=agent_id,
            content=content,
            tool_call=tool_call,
            file_diff=file_diff,
            created_at=_utc_now(),
        )

    def _next_seq(self, tenant_id: str, channel_id: str, thread_id: str | None) -> int:
        row = self._db.query_one(
            """SELECT COALESCE(MAX(seq), 0) + 1 AS next
               FROM session_messages
               WHERE tenant_id=? AND channel_id=? AND
                     thread_id IS ?""",
            (tenant_id, channel_id, thread_id),
        )
        return row["next"] if row else 1

    def list_messages(
        self,
        tenant_id: str,
        channel_id: str,
        *,
        thread_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[SessionMessage]:
        """List messages in chronological order."""
        rows = self._db.query_all(
            """SELECT * FROM session_messages
               WHERE tenant_id=? AND channel_id=? AND thread_id IS ?
               ORDER BY seq DESC LIMIT ? OFFSET ?""",
            (tenant_id, channel_id, thread_id, limit, offset),
        )
        return [self._row_to_msg(r) for r in rows]

    def search_history(
        self,
        tenant_id: str,
        query: str,
        *,
        channel_id: str | None = None,
        top_k: int = 10,
    ) -> list[SessionMessage]:
        """FTS5 search over session_messages content."""
        if not query.strip():
            return []
        clauses = ["sm.tenant_id = ?"]
        params: list = [tenant_id]
        if channel_id is not None:
            clauses.append("sm.channel_id = ?")
            params.append(channel_id)
        where = " AND ".join(clauses)
        sql = (
            "SELECT sm.*, bm25(session_messages_fts) AS score "
            "FROM session_messages_fts "
            "JOIN session_messages sm ON sm.rowid = session_messages_fts.rowid "
            f"WHERE session_messages_fts MATCH ? AND {where} "
            "ORDER BY score LIMIT ?"
        )
        conn = self._db.connect()
        safe_query = query.replace('"', '""')
        try:
            rows = conn.execute(sql, (f'"{safe_query}"', *params, top_k)).fetchall()
        except Exception:
            bare = query.replace('"', "").replace("*", "").replace(":", " ").strip()
            if not bare:
                return []
            try:
                rows = conn.execute(sql, (bare, *params, top_k)).fetchall()
            except Exception:
                return []
        return [self._row_to_msg(r) for r in rows]

    def _row_to_msg(self, row) -> SessionMessage:
        tool_call = json.loads(row["tool_call"]) if row["tool_call"] else None
        file_diff = json.loads(row["file_diff"]) if row["file_diff"] else None
        return SessionMessage(
            message_id=row["message_id"],
            tenant_id=row["tenant_id"],
            channel_id=row["channel_id"],
            thread_id=row["thread_id"],
            seq=row["seq"],
            role=row["role"],
            agent_id=row["agent_id"],
            content=row["content"],
            tool_call=tool_call,
            file_diff=file_diff,
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    # ----- HistoryAPI Protocol (scope-based) -----

    async def append_message_api(self, scope: MemoryScope, message: MemoryEntry) -> None:
        """HistoryAPI.append_message."""
        tenant = scope.tenant_id or "default"
        self.append_message(
            tenant_id=tenant,
            channel_id=scope.scope_id,
            role="user",
            content=message.text,
        )

    async def search_history_api(
        self, scope: MemoryScope, query: str, top_k: int = 10,
    ) -> list[MemoryEntry]:
        """HistoryAPI.search_history — returns MemoryEntry for compat."""
        tenant = scope.tenant_id or "default"
        msgs = self.search_history(tenant, query, channel_id=scope.scope_id, top_k=top_k)
        return [
            MemoryEntry(
                id=m.id,
                domain="agent",
                scope_id=scope.scope_id,
                kind="convention",
                text=m.content,
                tenant_id=m.tenant_id,
                schema_version=4,
                created_at=m.created_at,
                updated_at=m.created_at,
            )
            for m in msgs
        ]
