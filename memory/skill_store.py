"""Skill store (fourth layer, R6).

Reusable procedures and experience documents with independent versioning.
Each skill has a name unique within (tenant, domain, scope).  Updating a skill
creates a new version row linked to the previous version via
``previous_version_id``.  Rollback restores an earlier version as the current
active row.

The ``practice`` kind from v1/v2 is migrated here: each ``practice`` MemoryEntry
becomes a Skill with ``name`` derived from the entry text.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from memory.db import MemoryDB, memory_db
from memory.models import SkillMetadata, FactDomain, SkillStatus
from memory.scope import MemoryScope
from models.schemas import MemoryEntry


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


class SkillStore:
    """Skill layer: reusable procedures with independent versioning (R6)."""

    def __init__(self, db: MemoryDB | None = None) -> None:
        self._db = db or memory_db

    # ----- Core operations -----

    def create_skill(
        self,
        tenant_id: str,
        domain: FactDomain,
        scope_id: str,
        name: str,
        body: str,
        description: str = "",
        tags: list[str] | None = None,
    ) -> SkillMetadata:
        """Create a new skill (version 1). Fails if name already exists."""
        skill_id = f"skill_{uuid.uuid4().hex[:16]}"
        now = _utc_now_iso()
        conn = self._db.connect()
        conn.execute("BEGIN")
        try:
            existing = conn.execute(
                """SELECT id FROM skill_registry
                   WHERE tenant_id=? AND domain=? AND scope_id=? AND name=?
                     AND status='active'""",
                (tenant_id, domain, scope_id, name),
            ).fetchone()
            if existing:
                conn.execute("ROLLBACK")
                raise ValueError(f"skill '{name}' already exists in this scope")
            conn.execute(
                """INSERT INTO skill_registry
                   (id, tenant_id, domain, scope_id, name, description, body,
                    version, previous_version_id, tags, status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 1, NULL, ?, 'active', ?, ?)""",
                (skill_id, tenant_id, domain, scope_id, name, description, body,
                 json.dumps(tags or [], ensure_ascii=False), now, now),
            )
            conn.execute("COMMIT")
        except ValueError:
            raise
        except Exception:
            conn.execute("ROLLBACK")
            raise
        return SkillMetadata(
            id=skill_id, tenant_id=tenant_id, domain=domain, scope_id=scope_id,
            name=name, description=description, body=body, version=1,
            tags=tags or [], status="active",
            created_at=_utc_now(), updated_at=_utc_now(),
        )

    def get_skill(
        self,
        tenant_id: str,
        domain: FactDomain,
        scope_id: str,
        name: str,
    ) -> SkillMetadata | None:
        """Get the current active version of a skill by name."""
        row = self._db.query_one(
            """SELECT * FROM skill_registry
               WHERE tenant_id=? AND domain=? AND scope_id=? AND name=?
                 AND status='active'
               ORDER BY version DESC LIMIT 1""",
            (tenant_id, domain, scope_id, name),
        )
        return self._row_to_skill(row) if row else None

    def get_skill_by_id(self, skill_id: str) -> SkillMetadata | None:
        row = self._db.query_one("SELECT * FROM skill_registry WHERE id=?", (skill_id,))
        return self._row_to_skill(row) if row else None

    def list_skills(
        self,
        tenant_id: str,
        domain: FactDomain | None = None,
        scope_id: str | None = None,
    ) -> list[SkillMetadata]:
        """List all active skills in a scope."""
        clauses = ["tenant_id = ?", "status = 'active'"]
        params: list = [tenant_id]
        if domain is not None:
            clauses.append("domain = ?")
            params.append(domain)
        if scope_id is not None:
            clauses.append("scope_id = ?")
            params.append(scope_id)
        where = " AND ".join(clauses)
        rows = self._db.query_all(
            f"""SELECT * FROM skill_registry WHERE {where}
                ORDER BY name ASC, version DESC""",
            tuple(params),
        )
        # Deduplicate by name, keeping highest version.
        seen: set[str] = set()
        skills: list[SkillMetadata] = []
        for row in rows:
            if row["name"] in seen:
                continue
            seen.add(row["name"])
            skills.append(self._row_to_skill(row))
        return skills

    def update_skill(
        self,
        skill_id: str,
        body: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
    ) -> SkillMetadata:
        """Update a skill by creating a new version (ADD-only versioning)."""
        current = self.get_skill_by_id(skill_id)
        if current is None:
            raise ValueError(f"skill {skill_id} not found")
        if current.status != "active":
            raise ValueError(f"skill {skill_id} is not active")
        new_id = f"skill_{uuid.uuid4().hex[:16]}"
        new_version = current.version + 1
        new_body = body if body is not None else current.body
        new_description = description if description is not None else current.description
        new_tags = tags if tags is not None else current.tags
        now = _utc_now_iso()
        conn = self._db.connect()
        conn.execute("BEGIN")
        try:
            # Archive the old version.
            conn.execute(
                "UPDATE skill_registry SET status='archived', updated_at=? WHERE id=?",
                (now, skill_id),
            )
            # Insert the new version.
            conn.execute(
                """INSERT INTO skill_registry
                   (id, tenant_id, domain, scope_id, name, description, body,
                    version, previous_version_id, tags, status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)""",
                (new_id, current.tenant_id, current.domain, current.scope_id,
                 current.name, new_description, new_body, new_version, skill_id,
                 json.dumps(new_tags, ensure_ascii=False), now, now),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        return SkillMetadata(
            id=new_id, tenant_id=current.tenant_id, domain=current.domain,
            scope_id=current.scope_id, name=current.name, description=new_description,
            body=new_body, version=new_version, previous_version_id=skill_id,
            tags=new_tags, status="active",
            created_at=_utc_now(), updated_at=_utc_now(),
        )

    def rollback_skill(self, skill_id: str, to_version: int) -> SkillMetadata:
        """Rollback a skill to a specific version.

        Archives the current active version and reactivates the specified
        historical version as a new current row.
        """
        current = self.get_skill_by_id(skill_id)
        if current is None:
            raise ValueError(f"skill {skill_id} not found")
        # Find the historical version.
        row = self._db.query_one(
            """SELECT * FROM skill_registry
               WHERE tenant_id=? AND domain=? AND scope_id=? AND name=?
                 AND version=?""",
            (current.tenant_id, current.domain, current.scope_id, current.name, to_version),
        )
        if row is None:
            raise ValueError(f"version {to_version} not found for skill {skill_id}")
        historical = self._row_to_skill(row)
        now = _utc_now_iso()
        new_id = f"skill_{uuid.uuid4().hex[:16]}"
        new_version = current.version + 1
        conn = self._db.connect()
        conn.execute("BEGIN")
        try:
            conn.execute(
                "UPDATE skill_registry SET status='archived', updated_at=? WHERE id=?",
                (now, skill_id),
            )
            conn.execute(
                """INSERT INTO skill_registry
                   (id, tenant_id, domain, scope_id, name, description, body,
                    version, previous_version_id, tags, status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)""",
                (new_id, historical.tenant_id, historical.domain, historical.scope_id,
                 historical.name, historical.description, historical.body, new_version,
                 skill_id, json.dumps(historical.tags, ensure_ascii=False), now, now),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        return SkillMetadata(
            id=new_id, tenant_id=historical.tenant_id, domain=historical.domain,
            scope_id=historical.scope_id, name=historical.name,
            description=historical.description, body=historical.body,
            version=new_version, previous_version_id=skill_id,
            tags=historical.tags, status="active",
            created_at=_utc_now(), updated_at=_utc_now(),
        )

    def _row_to_skill(self, row) -> SkillMetadata:
        return SkillMetadata(
            id=row["id"],
            tenant_id=row["tenant_id"],
            domain=row["domain"],
            scope_id=row["scope_id"],
            name=row["name"],
            description=row["description"],
            body=row["body"],
            version=row["version"],
            previous_version_id=row["previous_version_id"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            status=row["status"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    # ----- SkillAPI Protocol (scope-based) -----

    async def create_skill_api(
        self, scope: MemoryScope, name: str, body: str, description: str = "",
    ) -> MemoryEntry:
        tenant = scope.tenant_id or "default"
        domain = scope.domain
        if domain == "workspace":
            domain = "project"
        skill = self.create_skill(
            tenant, domain, scope.scope_id, name, body, description,  # type: ignore[arg-type]
        )
        return self._skill_to_entry(skill)

    async def get_skill_api(self, scope: MemoryScope, name: str) -> MemoryEntry | None:
        tenant = scope.tenant_id or "default"
        domain = scope.domain
        if domain == "workspace":
            domain = "project"
        skill = self.get_skill(tenant, domain, scope.scope_id, name)  # type: ignore[arg-type]
        return self._skill_to_entry(skill) if skill else None

    async def update_skill_api(self, scope: MemoryScope, skill_id: str, body: str) -> MemoryEntry:
        skill = self.update_skill(skill_id, body=body)
        return self._skill_to_entry(skill)

    async def rollback_skill_api(self, scope: MemoryScope, skill_id: str, to_version: int) -> MemoryEntry:
        skill = self.rollback_skill(skill_id, to_version)
        return self._skill_to_entry(skill)

    @staticmethod
    def _skill_to_entry(skill: SkillMetadata) -> MemoryEntry:
        return MemoryEntry(
            id=skill.id,
            domain=skill.domain,
            scope_id=skill.scope_id,
            kind="convention",  # v1 compat: practice kind maps to Skill
            text=skill.body,
            tags=skill.tags,
            importance=0.8,
            confidence=1.0,
            tenant_id=skill.tenant_id,
            schema_version=4,
            created_at=skill.created_at,
            updated_at=skill.updated_at,
            version=skill.version,
        )
