"""One-time legacy JSON → Memory 2.1 / Schema v4 migration (§13).

Scans all legacy JSON memory sources, classifies entries into the four
final layers (Fact / History / Skill Draft / Excluded), validates in
memory, then writes to the v4 database in a single atomic transaction.

Migration rules (§13.2):
    Project Memory  → Fact/project   (workspace → project)
    User Memory     → Fact/user      (keep existing u_<sha256> scope key)
    项目 Agent Memory:
        kind=practice → Skill Draft (status=draft, not active)
        everything else → History (channel_id=project_id, thread_id=legacy-agent:<agent>)
    全局 Agent Memory:
        longmemeval-sut.json → all excluded
        coordinator.json (final-channel tag) → excluded
        coder.json (c1 tag) → excluded
        claude-code.json, codex.json → History (channel_id=legacy-global:<agent>)

ID rule (§13.3):  legacy_id = sha256(source_relative_path + old_entry_id + target_layer)

Usage::
    python -m memory.import_legacy              # dry-run (no writes)
    python -m memory.import_legacy --execute     # actual migration
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure backend/ is on sys.path when run as a script.
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from memory.db import MemoryDB, memory_db


TENANT_ID = "default"

# Expected counts (from §13.1, adjusted for actual data scan).
# The plan's original counts were based on an earlier scan; the actual
# data may differ slightly.  We report actual counts and verify the
# classification is correct, not that numbers match a stale plan.
EXPECTED_FACT = 11        # 5 project + 6 user
EXPECTED_HISTORY = 236    # 196 project-agent + 19 claude-code + 21 codex
EXPECTED_SKILL_DRAFT = 25  # practice entries from project-agent
EXPECTED_EXCLUDED = 660   # 542 longmemeval + 30 coordinator + 88 coder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _legacy_id(source_rel: str, old_id: str, target_layer: str) -> str:
    """Deterministic ID: sha256(source_relative_path + old_entry_id + target_layer)."""
    raw = f"{source_rel}|{old_id}|{target_layer}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _rel_path(path: str) -> str:
    """Return path relative to backend/."""
    try:
        return str(Path(path).resolve().relative_to(_BACKEND_DIR))
    except ValueError:
        return path


def _parse_created_at(entry: dict[str, Any]) -> str:
    """Parse created_at from a legacy entry, falling back to now()."""
    raw = entry.get("created_at")
    if raw:
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).isoformat()
        except Exception:
            pass
    return _utc_now_iso()


def _kind_map(kind: str) -> str:
    """Map legacy kinds to v4 FactKind.

    v4 removed ``practice`` and ``working_note``; map to ``convention``.
    """
    if kind in ("practice", "working_note"):
        return "convention"
    if kind in (
        "preference", "correction", "project_fact", "decision",
        "progress", "convention", "goal", "constraint", "outcome", "relation",
    ):
        return kind
    return "convention"  # safe default


def _build_provenance(source_rel: str, entry: dict[str, Any]) -> str:
    """Build a ProvenanceInfo JSON string for a migrated fact."""
    prov = {
        "source_type": "migration",
        "source_ids": [source_rel],
        "author": entry.get("source_type", "agent_result"),
        "extractor": "migration",
        "extraction_confidence": entry.get("confidence", 0.5),
        "verified_by": None,
        "verified_at": None,
        "evidence_refs": [],
    }
    return json.dumps(prov, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Classification data classes
# ---------------------------------------------------------------------------

class FactRecord:
    """A classified fact to insert."""
    __slots__ = ("id", "domain", "scope_id", "kind", "text", "tags",
                 "importance", "confidence", "created_at", "updated_at",
                 "provenance", "source_rel")

    def __init__(self, *, id, domain, scope_id, kind, text, tags,
                 importance, confidence, created_at, updated_at,
                 provenance, source_rel):
        self.id = id
        self.domain = domain
        self.scope_id = scope_id
        self.kind = kind
        self.text = text
        self.tags = tags
        self.importance = importance
        self.confidence = confidence
        self.created_at = created_at
        self.updated_at = updated_at
        self.provenance = provenance
        self.source_rel = source_rel


class HistoryRecord:
    """A classified history message to insert."""
    __slots__ = ("message_id", "channel_id", "thread_id", "role",
                 "agent_id", "content", "created_at", "source_rel")

    def __init__(self, *, message_id, channel_id, thread_id, role,
                 agent_id, content, created_at, source_rel):
        self.message_id = message_id
        self.channel_id = channel_id
        self.thread_id = thread_id
        self.role = role
        self.agent_id = agent_id
        self.content = content
        self.created_at = created_at
        self.source_rel = source_rel


class SkillDraftRecord:
    """A classified skill draft to insert."""
    __slots__ = ("id", "scope_id", "name", "body", "tags",
                 "created_at", "source_rel")

    def __init__(self, *, id, scope_id, name, body, tags,
                 created_at, source_rel):
        self.id = id
        self.scope_id = scope_id
        self.name = name
        self.body = body
        self.tags = tags
        self.created_at = created_at
        self.source_rel = source_rel


class ExcludedRecord:
    """A record that was excluded from migration."""
    __slots__ = ("source_rel", "old_id", "reason")

    def __init__(self, *, source_rel, old_id, reason):
        self.source_rel = source_rel
        self.old_id = old_id
        self.reason = reason


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

class LegacyScanner:
    """Scans and classifies all legacy JSON memory sources."""

    def __init__(self) -> None:
        self.facts: list[FactRecord] = []
        self.history: list[HistoryRecord] = []
        self.skills: list[SkillDraftRecord] = []
        self.excluded: list[ExcludedRecord] = []
        self._seq_counter: dict[str, int] = {}  # channel_id → next seq

    def _next_seq(self, channel_id: str) -> int:
        seq = self._seq_counter.get(channel_id, 0) + 1
        self._seq_counter[channel_id] = seq
        return seq

    def scan(self) -> None:
        """Scan all sources and classify entries."""
        self._scan_project_memory()
        self._scan_user_memory()
        self._scan_project_agent_memory()
        self._scan_global_agent_memory()

    # ----- Project Memory → Fact/project -----

    def _scan_project_memory(self) -> None:
        for f in sorted(glob.glob(str(_BACKEND_DIR / "data" / "projects" / "*" / "memory.json"))):
            source_rel = _rel_path(f)
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            scope = data.get("scope", {})
            project_id = scope.get("scope_id", "")
            for entry in data.get("entries", []):
                fact_id = _legacy_id(source_rel, entry["id"], "fact")
                self.facts.append(FactRecord(
                    id=fact_id,
                    domain="project",
                    scope_id=project_id,
                    kind=_kind_map(entry.get("kind", "project_fact")),
                    text=entry["text"],
                    tags=entry.get("tags", []),
                    importance=entry.get("importance", 0.5),
                    confidence=entry.get("confidence", 0.5),
                    created_at=_parse_created_at(entry),
                    updated_at=_parse_created_at(entry),
                    provenance=_build_provenance(source_rel, entry),
                    source_rel=source_rel,
                ))

    # ----- User Memory → Fact/user -----

    def _scan_user_memory(self) -> None:
        for f in sorted(glob.glob(str(_BACKEND_DIR / "data" / "users" / "*" / "memory.json"))):
            source_rel = _rel_path(f)
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            scope = data.get("scope", {})
            user_scope_id = scope.get("scope_id", "")
            for entry in data.get("entries", []):
                fact_id = _legacy_id(source_rel, entry["id"], "fact")
                self.facts.append(FactRecord(
                    id=fact_id,
                    domain="user",
                    scope_id=user_scope_id,
                    kind=_kind_map(entry.get("kind", "preference")),
                    text=entry["text"],
                    tags=entry.get("tags", []),
                    importance=entry.get("importance", 0.5),
                    confidence=entry.get("confidence", 0.5),
                    created_at=_parse_created_at(entry),
                    updated_at=_parse_created_at(entry),
                    provenance=_build_provenance(source_rel, entry),
                    source_rel=source_rel,
                ))

    # ----- 项目 Agent Memory → History + Skill Draft -----

    def _scan_project_agent_memory(self) -> None:
        pattern = str(_BACKEND_DIR / "data" / "projects" / "*" / "agents" / "memory" / "*.json")
        for f in sorted(glob.glob(pattern)):
            source_rel = _rel_path(f)
            # Extract project_id and agent_id from path.
            parts = Path(f).parts
            # .../data/projects/<project_id>/agents/memory/<agent>.json
            project_id = parts[-4] if len(parts) >= 4 else "unknown"
            agent_name = Path(f).stem  # e.g. "coder", "coordinator"

            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            for entry in data.get("entries", []):
                old_id = entry["id"]
                kind = entry.get("kind")

                if kind == "practice":
                    # → Skill Draft
                    skill_id = _legacy_id(source_rel, old_id, "skill")
                    self.skills.append(SkillDraftRecord(
                        id=skill_id,
                        scope_id=project_id,
                        name=f"legacy-practice-{old_id[:8]}",
                        body=entry["text"],
                        tags=entry.get("tags", []),
                        created_at=_parse_created_at(entry),
                        source_rel=source_rel,
                    ))
                else:
                    # → History (original agent output)
                    msg_id = _legacy_id(source_rel, old_id, "history")
                    channel_id = project_id
                    thread_id = f"legacy-agent:{agent_name}"
                    self.history.append(HistoryRecord(
                        message_id=msg_id,
                        channel_id=channel_id,
                        thread_id=thread_id,
                        role="agent",
                        agent_id=agent_name,
                        content=entry["text"],
                        created_at=_parse_created_at(entry),
                        source_rel=source_rel,
                    ))

    # ----- 全局 Agent Memory → History + Excluded -----

    def _scan_global_agent_memory(self) -> None:
        pattern = str(_BACKEND_DIR / "data" / "memory" / "*.json")
        for f in sorted(glob.glob(pattern)):
            source_rel = _rel_path(f)
            filename = Path(f).name

            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            # longmemeval-sut.json → all excluded
            if filename == "longmemeval-sut.json":
                for entry in data.get("entries", []):
                    self.excluded.append(ExcludedRecord(
                        source_rel=source_rel,
                        old_id=entry["id"],
                        reason="longmemeval_benchmark",
                    ))
                continue

            for entry in data.get("entries", []):
                old_id = entry["id"]
                tags = entry.get("tags", [])

                # Test data exclusion
                if "c1" in tags:
                    self.excluded.append(ExcludedRecord(
                        source_rel=source_rel, old_id=old_id, reason="test_data_c1",
                    ))
                    continue
                if "final-channel" in tags:
                    self.excluded.append(ExcludedRecord(
                        source_rel=source_rel, old_id=old_id, reason="test_data_final_channel",
                    ))
                    continue

                # Real agent output → History
                agent_name = Path(f).stem
                msg_id = _legacy_id(source_rel, old_id, "history")
                channel_id = f"legacy-global:{agent_name}"
                self.history.append(HistoryRecord(
                    message_id=msg_id,
                    channel_id=channel_id,
                    thread_id=None,
                    role="agent",
                    agent_id=agent_name,
                    content=entry["text"],
                    created_at=_parse_created_at(entry),
                    source_rel=source_rel,
                ))

    # ----- Validation -----

    def validate(self) -> None:
        """Validate all classified records before writing.

        Raises ``ValueError`` on any issue, ensuring the migration
        stops before touching the database.
        """
        errors: list[str] = []

        # Check for duplicate IDs within each layer.
        fact_ids = [f.id for f in self.facts]
        if len(fact_ids) != len(set(fact_ids)):
            errors.append("duplicate fact IDs detected")

        history_ids = [h.message_id for h in self.history]
        if len(history_ids) != len(set(history_ids)):
            errors.append("duplicate history message IDs detected")

        skill_ids = [s.id for s in self.skills]
        if len(skill_ids) != len(set(skill_ids)):
            errors.append("duplicate skill IDs detected")

        # Check required fields.
        for fact in self.facts:
            if not fact.text:
                errors.append(f"fact {fact.id} has empty text")
            if not fact.scope_id:
                errors.append(f"fact {fact.id} has empty scope_id")

        for msg in self.history:
            if not msg.content:
                errors.append(f"history {msg.message_id} has empty content")
            if not msg.channel_id:
                errors.append(f"history {msg.message_id} has empty channel_id")

        for skill in self.skills:
            if not skill.body:
                errors.append(f"skill {skill.id} has empty body")

        if errors:
            raise ValueError(
                "validation failed with {} error(s):\n  {}".format(
                    len(errors), "\n  ".join(errors[:20])
                )
            )

    def summary(self) -> dict[str, Any]:
        """Return a summary dict of classified counts."""
        return {
            "facts": len(self.facts),
            "history": len(self.history),
            "skill_drafts": len(self.skills),
            "excluded": len(self.excluded),
            "total": len(self.facts) + len(self.history) + len(self.skills) + len(self.excluded),
        }


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class LegacyWriter:
    """Writes classified records to the v4 database in one transaction."""

    def __init__(self, db: MemoryDB, scanner: LegacyScanner) -> None:
        self._db = db
        self._scanner = scanner

    def write(self) -> None:
        """Write all records in a single BEGIN IMMEDIATE transaction."""
        conn = self._db.connect()
        conn.execute("BEGIN IMMEDIATE")
        try:
            self._write_facts(conn)
            self._write_history(conn)
            self._write_skills(conn)
            self._write_audit(conn)
            # Verify counts inside the transaction.
            self._verify_counts(conn)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def _write_facts(self, conn) -> None:
        for fact in self._scanner.facts:
            conn.execute(
                """INSERT INTO facts
                   (id, tenant_id, domain, scope_id, agent_id, task_id, kind,
                    text, search_text, tags, importance, confidence, status,
                    temporal, provenance, embedding_model, created_at, updated_at,
                    expires_at, version, schema_version)
                   VALUES (?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?, ?, ?, 'active',
                           NULL, ?, NULL, ?, ?, NULL, 1, 4)""",
                (
                    fact.id,
                    TENANT_ID,
                    fact.domain,
                    fact.scope_id,
                    fact.kind,
                    fact.text,
                    fact.text,  # search_text = text
                    json.dumps(fact.tags, ensure_ascii=False),
                    fact.importance,
                    fact.confidence,
                    fact.provenance,
                    fact.created_at,
                    fact.updated_at,
                ),
            )

    def _write_history(self, conn) -> None:
        # Track seq per channel within this transaction.
        seq_map: dict[str, int] = {}
        for msg in self._scanner.history:
            seq = seq_map.get(msg.channel_id, 0) + 1
            seq_map[msg.channel_id] = seq
            conn.execute(
                """INSERT INTO session_messages
                   (message_id, tenant_id, channel_id, thread_id, seq, role,
                    agent_id, content, search_text, tool_call, file_diff,
                    redaction_status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, 'clean', ?)""",
                (
                    msg.message_id,
                    TENANT_ID,
                    msg.channel_id,
                    msg.thread_id,
                    seq,
                    msg.role,
                    msg.agent_id,
                    msg.content,
                    msg.content,  # search_text = content
                    msg.created_at,
                ),
            )

    def _write_skills(self, conn) -> None:
        now = _utc_now_iso()
        for skill in self._scanner.skills:
            conn.execute(
                """INSERT INTO skill_registry
                   (id, tenant_id, domain, scope_id, name, description, body,
                    version, previous_version_id, tags, status, created_at, updated_at)
                   VALUES (?, ?, 'agent', ?, ?, '', ?, 1, NULL, ?, 'draft', ?, ?)""",
                (
                    skill.id,
                    TENANT_ID,
                    skill.scope_id,
                    skill.name,
                    skill.body,
                    json.dumps(skill.tags, ensure_ascii=False),
                    skill.created_at,
                    now,
                ),
            )

    def _write_audit(self, conn) -> None:
        """Write audit log entries for the migration."""
        now = _utc_now_iso()
        # One summary audit entry.
        audit_id = f"aud_{uuid.uuid4().hex[:16]}"
        conn.execute(
            """INSERT INTO memory_audit_log
               (id, tenant_id, action, domain, fact_id, actor, reason, created_at)
               VALUES (?, ?, 'write', NULL, NULL, 'system', 'legacy_migration', ?)""",
            (audit_id, TENANT_ID, now),
        )

    def _verify_counts(self, conn) -> None:
        """Verify that the written counts match the classified counts."""
        fact_count = conn.execute(
            "SELECT COUNT(*) FROM facts WHERE tenant_id = ?", (TENANT_ID,)
        ).fetchone()[0]
        history_count = conn.execute(
            "SELECT COUNT(*) FROM session_messages WHERE tenant_id = ?", (TENANT_ID,)
        ).fetchone()[0]
        skill_count = conn.execute(
            "SELECT COUNT(*) FROM skill_registry WHERE tenant_id = ? AND status = 'draft'",
            (TENANT_ID,),
        ).fetchone()[0]

        expected_facts = len(self._scanner.facts)
        expected_history = len(self._scanner.history)
        expected_skills = len(self._scanner.skills)

        if fact_count != expected_facts:
            raise RuntimeError(
                f"fact count mismatch: DB={fact_count}, expected={expected_facts}"
            )
        if history_count != expected_history:
            raise RuntimeError(
                f"history count mismatch: DB={history_count}, expected={expected_history}"
            )
        if skill_count != expected_skills:
            raise RuntimeError(
                f"skill count mismatch: DB={skill_count}, expected={expected_skills}"
            )


# ---------------------------------------------------------------------------
# Post-migration verification
# ---------------------------------------------------------------------------

def verify_integrity(db: MemoryDB) -> None:
    """Run FK check, FTS sync check, and count verification."""
    conn = db.connect()

    # FK integrity check.
    fk_issues = conn.execute("PRAGMA foreign_key_check").fetchall()
    if fk_issues:
        raise RuntimeError(f"foreign key check failed: {fk_issues}")

    # FTS sync check: facts_fts row count should match facts.
    fts_count = conn.execute("SELECT COUNT(*) FROM facts_fts").fetchone()[0]
    facts_count = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    if fts_count != facts_count:
        raise RuntimeError(
            f"facts_fts out of sync: fts={fts_count}, facts={facts_count}"
        )

    # FTS sync check: session_messages_fts row count should match session_messages.
    sm_fts_count = conn.execute("SELECT COUNT(*) FROM session_messages_fts").fetchone()[0]
    sm_count = conn.execute("SELECT COUNT(*) FROM session_messages").fetchone()[0]
    if sm_fts_count != sm_count:
        raise RuntimeError(
            f"session_messages_fts out of sync: fts={sm_fts_count}, msgs={sm_count}"
        )


def delete_legacy_json() -> list[str]:
    """Delete all legacy JSON memory files.

    Returns the list of deleted file paths.
    """
    deleted: list[str] = []
    patterns = [
        str(_BACKEND_DIR / "data" / "projects" / "*" / "memory.json"),
        str(_BACKEND_DIR / "data" / "users" / "*" / "memory.json"),
        str(_BACKEND_DIR / "data" / "projects" / "*" / "agents" / "memory" / "*.json"),
        str(_BACKEND_DIR / "data" / "memory" / "*.json"),
    ]
    for pattern in patterns:
        for f in sorted(glob.glob(pattern)):
            os.remove(f)
            deleted.append(f)
    return deleted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Legacy JSON → Memory 2.1 migration")
    parser.add_argument(
        "--execute", action="store_true",
        help="Execute the migration (default: dry-run only)",
    )
    parser.add_argument(
        "--delete-json", action="store_true",
        help="Delete legacy JSON files after successful migration (requires --execute)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Memory 2.1 Legacy Migration (§13)")
    print("=" * 60)

    # Phase 1: Scan and classify.
    print("\n[1] Scanning legacy JSON sources...")
    scanner = LegacyScanner()
    scanner.scan()
    summary = scanner.summary()
    print(f"    Facts:       {summary['facts']}")
    print(f"    History:     {summary['history']}")
    print(f"    Skill Drafts:{summary['skill_drafts']}")
    print(f"    Excluded:    {summary['excluded']}")
    print(f"    Total:       {summary['total']}")

    # Phase 2: Validate.
    print("\n[2] Validating classified records...")
    try:
        scanner.validate()
        print("    OK — all records valid")
    except ValueError as exc:
        print(f"    FAILED — {exc}")
        return 1

    if not args.execute:
        print("\n[DRY RUN] No database writes performed.")
        print("          Re-run with --execute to perform the migration.")
        return 0

    # Phase 3: Write to database.
    print("\n[3] Writing to database (single transaction)...")
    writer = LegacyWriter(memory_db, scanner)
    try:
        writer.write()
        print("    OK — transaction committed")
    except Exception as exc:
        print(f"    FAILED — {exc}")
        print("    Transaction rolled back, no data written.")
        return 1

    # Phase 4: Verify integrity.
    print("\n[4] Verifying database integrity...")
    try:
        verify_integrity(memory_db)
        print("    OK — FK check passed, FTS in sync")
    except RuntimeError as exc:
        print(f"    FAILED — {exc}")
        return 1

    # Phase 5: Delete legacy JSON.
    if args.delete_json:
        print("\n[5] Deleting legacy JSON files...")
        deleted = delete_legacy_json()
        print(f"    Deleted {len(deleted)} files")
    else:
        print("\n[5] Skipping JSON deletion (use --delete-json to enable)")

    print("\n" + "=" * 60)
    print("Migration complete.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
