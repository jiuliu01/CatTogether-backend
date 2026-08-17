"""Skill layer store — reusable procedures, versioned (§5.10).

Thin wrapper over the existing ``SkillStore`` providing the final
interface used by the context builder.  Skills are only loaded when the
task requires procedural guidance.
"""
from __future__ import annotations

from memory.db import MemoryDB, memory_db
from memory.models import SkillMetadata
from memory.skill_store import SkillStore as _SkillStoreImpl


class SkillStoreWrapper:
    """Skill layer (§5.10).

    Wraps ``SkillStore`` with the final interface.  Skills have an
    approval workflow (create → pending → approved) and versioning
    (update creates a new version, rollback restores a previous one).
    """

    def __init__(self, db: MemoryDB | None = None) -> None:
        self._db = db or memory_db
        self._store = _SkillStoreImpl(self._db)

    def create(
        self,
        tenant_id: str,
        domain: str,
        scope_id: str,
        name: str,
        body: str,
        *,
        description: str = "",
        tags: list[str] | None = None,
    ) -> SkillMetadata:
        """Create a new skill (enters pending_review status)."""
        return self._store.create_skill(
            tenant_id=tenant_id,
            domain=domain,  # type: ignore[arg-type]
            scope_id=scope_id,
            name=name,
            body=body,
            description=description,
            tags=tags or [],
        )

    def get(self, tenant_id: str, name: str, scope_id: str = "default", *, domain: str = "agent") -> SkillMetadata | None:
        """Get a skill by name within a scope."""
        return self._store.get_skill(tenant_id=tenant_id, domain=domain, scope_id=scope_id, name=name)  # type: ignore[arg-type]

    def get_by_id(self, skill_id: str) -> SkillMetadata | None:
        """Get a skill by its ID."""
        return self._store.get_skill_by_id(skill_id)

    def list(
        self,
        tenant_id: str,
        domain: str | None = None,
        scope_id: str | None = None,
        *,
        limit: int = 50,
    ) -> list[SkillMetadata]:
        """List skills, optionally filtered by domain/scope."""
        skills = self._store.list_skills(
            tenant_id=tenant_id,
            domain=domain,  # type: ignore[arg-type]
            scope_id=scope_id,
        )
        return skills[:limit]

    def update(
        self,
        skill_id: str,
        body: str,
        *,
        change_summary: str = "",
    ) -> SkillMetadata:
        """Update a skill's body (creates a new version)."""
        return self._store.update_skill(skill_id, body)

    def rollback(self, skill_id: str, to_version: int) -> SkillMetadata:
        """Rollback a skill to a previous version."""
        return self._store.rollback_skill(skill_id, to_version)


# Alias for the __init__.py export name.
SkillStore = SkillStoreWrapper