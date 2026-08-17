"""Migrate the 2.1 SQLite ``facts`` table into the 2.2 Qdrant ``memories``
collection via the real MemoryExtractor (the same path live extraction uses).

Each old fact is wrapped as a fake "agent invocation completed" event and fed
to ``MemoryExtractor.extract`` — same memory agent, same JSONL contract, same
domain derivation, same dedup, same Qdrant write. The migration therefore
doubles as an end-to-end test of the extraction pipeline.

Usage::

    # Dry-run: run the memory agent, print what it would write, do NOT touch Qdrant.
    python backend/scripts/migrate_memory_v22.py

    # Apply: actually write to Qdrant.
    python backend/scripts/migrate_memory_v22.py --apply

    # Limit to the first N facts (for debugging).
    python backend/scripts/migrate_memory_v22.py --apply --limit 5

Requires: CT_MEMORY_V22_ENABLED=true, Qdrant running, Catenv (Claude CLI available).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sqlite3
import sys
import os
from datetime import datetime, timezone
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("migrate_v22")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_old_facts(db_path: Path) -> list[dict]:
    """Read all active facts from the 2.1 SQLite DB."""
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT id, tenant_id, domain, scope_id, agent_id, kind, text, "
            "provenance, created_at FROM facts WHERE status = 'active' "
            "ORDER BY created_at ASC"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _author_to_attributed(author: str) -> str:
    """Map the 2.1 provenance.author to a 2.2 attributed_to hint.

    Used only to pick the ExtractionContext.agent_id; the final
    attributed_to still comes from the memory agent's JSONL output.
    """
    a = (author or "").strip().lower()
    if a in ("coder", "coordinator", "reviewer", "researcher"):
        return "assistant"
    return "user"


def _extract_mutation_paths(provenance: dict) -> list[str]:
    """Pull file-like strings out of provenance.source_ids, if any."""
    ids = provenance.get("source_ids") or []
    out: list[str] = []
    for sid in ids:
        s = str(sid or "")
        if "/" in s or "\\" in s or s.endswith((".py", ".js", ".wxss", ".json", ".md", ".html")):
            out.append(s)
    return out[:10]


def _build_context(fact: dict) -> "ExtractionContext":  # type: ignore[name-defined]
    """Wrap one old fact as a fake agent-invocation ExtractionContext."""
    from memory.v22.extractor import ExtractionContext

    prov = json.loads(fact["provenance"]) if fact.get("provenance") else {}
    author = prov.get("author", "")
    attributed_hint = _author_to_attributed(author)
    mutation_paths = _extract_mutation_paths(prov)

    domain = fact.get("domain", "project")
    scope_id = fact.get("scope_id", "")

    # project facts: set project_id + mutation_paths so _derive_domain → project.
    # user facts: leave project_id unset so _derive_domain → user (when agent
    # emits attributed_to=user) or agent. We accept whatever the system derives.
    project_id = scope_id if domain == "project" else None

    return ExtractionContext(
        event_type="agent_invocation",
        agent_id=author or "migration",
        agent_name=author or "migration",
        user_text=fact.get("text", ""),
        final_text=fact.get("text", ""),
        mutation_paths=mutation_paths,
        succeeded=True,
        run_id="migration-v22",
        channel_id=None,
        thread_id=None,
        project_id=project_id,
        user_id="feishu:migration:user",
    )


async def migrate(*, db_path: Path, apply: bool, limit: int | None) -> dict:
    """Run the migration. Returns a report dict."""
    from config import settings
    if not settings.memory_v22_enabled:
        logger.error("CT_MEMORY_V22_ENABLED is off; set it to true to migrate.")
        return {"error": "v22 disabled"}

    from memory.v22 import get_memory_extractor, reset_for_tests
    from memory.v22.memory_store import MemoryStore
    reset_for_tests()
    extractor = get_memory_extractor()
    if extractor is None:
        logger.error("memory extractor unavailable (Qdrant / agent not ready)")
        return {"error": "extractor unavailable"}

    # In dry-run mode, swap the store's upsert for a no-op that records instead
    # of writing to Qdrant.
    dry_run_store: list = []
    real_upsert = extractor._store.upsert
    real_exists = extractor._store.exists_by_hash

    if not apply:
        def fake_upsert(mem):
            dry_run_store.append(mem)
            return mem
        def fake_exists(domain, text):
            # In dry run, report duplicates among already-dry-written.
            import hashlib
            h = hashlib.sha256(text.encode()).hexdigest()
            return any(m.hash == h and m.domain == domain for m in dry_run_store)
        extractor._store.upsert = fake_upsert  # type: ignore
        extractor._store.exists_by_hash = fake_exists  # type: ignore

    facts = _load_old_facts(db_path)
    if limit:
        facts = facts[:limit]
    logger.info("loaded %d active facts from %s", len(facts), db_path)

    report: list[dict] = []
    total_written = 0
    total_skipped_dup = 0
    total_dropped = 0

    for i, fact in enumerate(facts):
        fid = fact["id"][:16]
        ctx = _build_context(fact)
        logger.info("[%d/%d] fact %s | domain=%s text=%s",
                    i + 1, len(facts), fid, fact.get("domain"),
                    (fact.get("text") or "")[:60])
        try:
            result = await extractor.extract(ctx)
        except Exception as exc:
            logger.exception("extract failed for fact %s", fid)
            report.append({
                "fact_id": fid, "domain": fact.get("domain"),
                "outcome": "error", "error": str(exc),
                "written": [],
            })
            continue

        entry = {
            "fact_id": fid,
            "domain": fact.get("domain"),
            "kind": fact.get("kind"),
            "original_text": (fact.get("text") or "")[:200],
            "outcome": "ok" if result.written else ("skipped" if result.skipped_dup else "empty"),
            "skipped_dup": result.skipped_dup,
            "skipped_invalid": result.skipped_invalid,
            "failed": result.failed,
            "written": [
                {
                    "id": m.id,
                    "text": (m.text or "")[:200],
                    "domain": m.domain,
                    "attributed_to": m.attributed_to,
                    "linked_memory_ids": list(m.linked_memory_ids),
                }
                for m in result.written
            ],
        }
        report.append(entry)
        total_written += len(result.written)
        total_skipped_dup += result.skipped_dup
        if not result.written and not result.skipped_dup:
            total_dropped += 1
        logger.info("  → written=%d dup=%d invalid=%d failed=%s",
                    len(result.written), result.skipped_dup,
                    result.skipped_invalid, result.failed)
        for m in result.written:
            logger.info("    + [%s|%s] %s", m.domain, m.attributed_to, (m.text or "")[:80])

    # Restore the real upsert (in case the process keeps running).
    extractor._store.upsert = real_upsert  # type: ignore
    extractor._store.exists_by_hash = real_exists  # type: ignore

    summary = {
        "mode": "apply" if apply else "dry-run",
        "total_facts": len(facts),
        "total_written": total_written,
        "total_skipped_dup": total_skipped_dup,
        "total_dropped_empty": total_dropped,
        "timestamp": _utc_now_iso(),
        "per_fact": report,
    }

    # Persist the report.
    report_dir = Path("data/memory_migrations")
    report_dir.mkdir(parents=True, exist_ok=True)
    tag = "apply" if apply else "dryrun"
    report_path = report_dir / f"v22_migration_report_{tag}.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("report saved to %s", report_path)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Migrate 2.1 facts → 2.2 Qdrant memories")
    ap.add_argument("--apply", action="store_true",
                    help="Actually write to Qdrant (default: dry-run, no writes).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only migrate the first N facts.")
    ap.add_argument("--db", type=Path,
                    default=Path(__file__).resolve().parent.parent / "data" / "memory.db")
    args = ap.parse_args()

    os.environ.setdefault("CT_MEMORY_V22_ENABLED", "true")
    # Make sure the backend is importable.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    # Register built-in agents (memory agent) before building the extractor.
    from bootstrap import register_builtins
    register_builtins()

    summary = asyncio.run(migrate(
        db_path=args.db, apply=args.apply, limit=args.limit,
    ))

    print("\n" + "=" * 60)
    print(f"mode: {summary.get('mode')}")
    print(f"facts processed: {summary.get('total_facts')}")
    print(f"memories written: {summary.get('total_written')}")
    print(f"skipped (dup): {summary.get('total_skipped_dup')}")
    print(f"dropped (empty): {summary.get('total_dropped_empty')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
