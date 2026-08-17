"""Rebuild missing Memory 2.1 fact embeddings.

This command is intentionally idempotent: by default it only processes active
facts that either have no vector or have a pending ``reindex`` outbox record.
An outbox record is marked complete only after the vector was stored.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any

from memory.backends.sqlite_vec_backend import SqliteVecBackend
from memory.db import MemoryDB, memory_db
from memory.embedder import Embedder, get_embedder


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def rebuild_embeddings(
    db: MemoryDB | None = None,
    embedder: Embedder | None = None,
    *,
    limit: int | None = None,
    rebuild_all: bool = False,
) -> dict[str, Any]:
    """Build vectors and return a compact maintenance report."""
    target_db = db or memory_db
    conn = target_db.connect()
    if not target_db.vec_available:
        raise RuntimeError("sqlite-vec is unavailable; install sqlite-vec and reconnect")

    vector_store = SqliteVecBackend(target_db)
    generator = embedder or get_embedder()
    existing = {
        (row["tenant_id"], row["fact_id"])
        for row in conn.execute(
            "SELECT tenant_id, fact_id FROM fact_embeddings"
        ).fetchall()
    }
    pending = {
        (row["tenant_id"], row["fact_id"])
        for row in conn.execute(
            """SELECT tenant_id, fact_id FROM transaction_outbox
               WHERE op = 'reindex' AND processed_at IS NULL AND fact_id IS NOT NULL"""
        ).fetchall()
    }
    rows = conn.execute(
        """SELECT id, tenant_id, domain, scope_id, text, search_text
           FROM facts WHERE status = 'active' ORDER BY updated_at, id"""
    ).fetchall()
    selected = [
        row for row in rows
        if rebuild_all
        or (row["tenant_id"], row["id"]) not in existing
        or (row["tenant_id"], row["id"]) in pending
    ]
    if limit is not None:
        selected = selected[:max(0, limit)]

    rebuilt = 0
    failures: list[dict[str, str]] = []
    now = _utc_now_iso()
    for row in selected:
        try:
            text = row["search_text"] or row["text"]
            vector = generator.embed(text)
            if len(vector) != generator.dim:
                raise ValueError(
                    f"embedding dimension {len(vector)} does not match configured {generator.dim}"
                )
            vector_store.upsert_embedding_sync(
                row["id"], row["tenant_id"], row["domain"], row["scope_id"], vector,
            )
            conn.execute(
                """UPDATE facts SET embedding_model = ?, updated_at = ?
                   WHERE tenant_id = ? AND id = ?""",
                (generator.model_name, now, row["tenant_id"], row["id"]),
            )
            conn.execute(
                """UPDATE transaction_outbox SET processed_at = ?
                   WHERE tenant_id = ? AND fact_id = ?
                     AND op = 'reindex' AND processed_at IS NULL""",
                (now, row["tenant_id"], row["id"]),
            )
            rebuilt += 1
        except Exception as exc:  # keep the remaining repair work moving
            failures.append({"fact_id": row["id"], "error": str(exc)})

    return {
        "selected": len(selected),
        "rebuilt": rebuilt,
        "failed": len(failures),
        "model": generator.model_name,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild Memory 2.1 fact embeddings")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--all", action="store_true", dest="rebuild_all")
    args = parser.parse_args()
    report = rebuild_embeddings(limit=args.limit, rebuild_all=args.rebuild_all)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
