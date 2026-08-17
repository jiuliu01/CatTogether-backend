"""Audit logging for Memory 2.0 (§9, Stage 4).

Every memory operation (write / update / delete / forget / approve / supersede)
is recorded in the ``memory_audit_log`` table for compliance and debugging.

Security constraint (§9): the audit log records WHO did WHAT to WHICH fact,
never the fact text itself.  The ``reason`` field stores short categorical
labels (e.g. "dedup", "conflict", "gdpr"), never free-form fact content.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal

from memory.db import MemoryDB, memory_db


AuditAction = Literal[
    "write", "search", "update", "delete", "forget",
    "approve", "supersede", "hot_promote", "hot_archive",
    "skill_create", "skill_update", "skill_rollback",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AuditLogger:
    """Append-only audit log writer for memory operations.

    All methods are synchronous (the audit INSERT is part of the same
    logical operation as the data mutation, so it runs inline).
    """

    def __init__(self, db: MemoryDB | None = None) -> None:
        self._db = db or memory_db

    def log(
        self,
        *,
        tenant_id: str,
        action: AuditAction,
        actor: str,
        domain: str | None = None,
        fact_id: str | None = None,
        reason: str = "",
    ) -> str:
        """Write a single audit log entry. Returns the audit log id."""
        audit_id = f"aud_{uuid.uuid4().hex[:16]}"
        self._db.execute(
            """INSERT INTO memory_audit_log
               (id, tenant_id, action, domain, fact_id, actor, reason, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                audit_id,
                tenant_id,
                action,
                domain,
                fact_id,
                actor,
                reason,
                _utc_now_iso(),
            ),
        )
        return audit_id

    def query(
        self,
        tenant_id: str,
        *,
        action: str | None = None,
        fact_id: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query audit log entries (newest first)."""
        clauses = ["tenant_id = ?"]
        params: list = [tenant_id]
        if action is not None:
            clauses.append("action = ?")
            params.append(action)
        if fact_id is not None:
            clauses.append("fact_id = ?")
            params.append(fact_id)
        where = " AND ".join(clauses)
        rows = self._db.query_all(
            f"""SELECT * FROM memory_audit_log
                WHERE {where}
                ORDER BY created_at DESC LIMIT ?""",
            (*params, limit),
        )
        return [
            {
                "id": r["id"],
                "tenant_id": r["tenant_id"],
                "action": r["action"],
                "domain": r["domain"],
                "fact_id": r["fact_id"],
                "actor": r["actor"],
                "reason": r["reason"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]


# Module-level singleton (shares the default MemoryDB).
audit_logger = AuditLogger()
