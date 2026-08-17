"""Memory 2.1 / Schema v4 canonical data models.

Final converged types defined in ``word/4.2.Memory2.1最终版实施规划与架构.md`` §5.
These replace the v3 types in ``fact_models.py``.  Key changes from v3:

- ``FactKind``: removed ``working_note``; added ``goal/constraint/outcome/relation``.
- ``FactStatus``: added ``pending_review``.
- ``ProvenanceExtractor``: added ``"migration"``.
- ``Fact``: removed top-level ``entities``/``supersedes_id``; added
  ``search_text``/``agent_id``/``task_id``; ``schema_version=4``.
- ``Entity``: ``canonical_name``/``last_seen_at``/``merged_into_id`` replace
  ``name``/``mention_count``/``metadata``.
- ``EntityMention``: ``mention_text``/``linking_confidence`` replace simplified
  ``role``.
- ``FactRelation``: adds ``confidence``/``created_by``/``created_at``; relation
  types add ``"related"``.
- ``HotMemoryItem``: ``priority``/``source_fact_ids``/``proposed_by``.
- ``SessionMessage``: adds ``search_text``/``redaction_status``.
- ``SkillMetadata`` + ``SkillVersion``: independent versioning.
- ``RetrievalResult``: adds ``explanation``.
- ``Role``: removed ``readonly``; added ``coordinator``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from memory.scope import MemoryScope
from models.schemas import MemoryEntry, MemoryWriteEvent, now


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

MemoryDomain = Literal["user", "project", "task", "agent"]
"""Four-domain scope (2.1 final)."""

FactDomain = MemoryDomain  # Backward-compatible alias.

FactKind = Literal[
    "preference", "correction", "project_fact", "decision",
    "progress", "convention", "goal", "constraint", "outcome", "relation",
]
"""Fact kinds. ``practice`` and ``working_note`` are removed."""

FactStatus = Literal[
    "active", "pending_review", "superseded", "archived", "disputed",
]
"""Fact lifecycle status. ``pending_review`` is new in v4."""

EntityType = Literal[
    "person", "project", "tech", "file", "concept", "tool", "org", "metric",
]

EventStatus = Literal["pending", "processed", "failed", "skipped"]
EventType = Literal[
    "conversation_completed", "agent_invocation", "agent_tool_call",
    "file_changed", "manual_write",
]

ActorKind = Literal["user", "agent", "system"]

Role = Literal["user", "agent", "coordinator", "admin", "system"]
"""Five roles (2.1). ``readonly`` removed; ``coordinator`` added."""

HotStatus = Literal["pending_approval", "active", "archived"]

HotCategory = Literal["note", "decision", "reminder", "summary"]
"""Hot memory category (kept for compat with v3 callers)."""

SessionRole = Literal["user", "agent", "system", "tool"]
RedactionStatus = Literal["clean", "redacted", "encrypted", "blocked"]

SkillStatus = Literal["draft", "pending_approval", "active", "archived"]
SkillVisibility = Literal["private", "project", "tenant"]

FactRelationType = Literal[
    "supersedes", "supports", "contradicts", "derived_from", "related",
]

ProvenanceSourceType = Literal[
    "user_statement", "agent_result", "workspace_change",
    "manual", "migration",
]
ProvenanceExtractor = Literal["rule", "llm", "manual", "migration"]

MemoryIntent = Literal[
    "recall", "continue_task", "start_task", "reflect", "audit",
]


# ---------------------------------------------------------------------------
# Sub-models (embedded JSONB columns)
# ---------------------------------------------------------------------------

class EvidenceRef(BaseModel):
    """A piece of evidence backing a fact (file / commit / diff)."""
    kind: Literal["file", "commit", "diff", "tool_result"] = "file"
    ref: str
    summary: str = ""


class TemporalInfo(BaseModel):
    """Temporal metadata embedded in a fact (§5.5)."""
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    event_time: datetime | None = None
    time_expressions: list[str] = Field(default_factory=list)
    is_snapshot: bool = False


class ProvenanceInfo(BaseModel):
    """Provenance metadata embedded in a fact (§5.6)."""
    source_type: ProvenanceSourceType = "agent_result"
    source_ids: list[str] = Field(default_factory=list)
    author: str = ""
    extractor: ProvenanceExtractor = "rule"
    extraction_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    verified_by: str | None = None
    verified_at: datetime | None = None
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)


class Actor(BaseModel):
    """The actor that triggered a memory event."""
    kind: ActorKind = "system"
    id: str = "system"
    display_name: str = ""


class AllowedScope(BaseModel):
    """A scope that an actor is permitted to read/write (§4.3)."""
    tenant_id: str = "default"
    domain: MemoryDomain = "user"
    scope_id: str = "default"


class ActorContext(BaseModel):
    """Actor identity + resolved permissions for retrieval."""
    actor: Actor = Field(default_factory=Actor)
    tenant_id: str = "default"
    role: Role = "user"
    allowed_scopes: list[AllowedScope] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Event (§5.1)
# ---------------------------------------------------------------------------

class Event(BaseModel):
    """An immutable record submitted to the write pipeline."""
    event_id: str
    tenant_id: str = "default"
    event_type: EventType = "conversation_completed"
    actor: Actor = Field(default_factory=Actor)
    scope_hint: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    allowed_domains: list[MemoryDomain] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=now)
    processed_at: datetime | None = None
    processing_stage: str | None = None
    attempt_count: int = 0
    status: EventStatus = "pending"
    error: str | None = None
    fact_ids: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Fact (§5.2)
# ---------------------------------------------------------------------------

class Fact(BaseModel):
    """A long-term factual knowledge unit (§5.2).

    ADD-only write semantics (R4): new facts always INSERT new rows.
    Supersession is tracked via ``fact_relations``, not a top-level field.
    Entity mentions are tracked via ``fact_entity_mentions``, not a top-level
    field.
    """
    id: str
    tenant_id: str = "default"
    domain: MemoryDomain = "agent"
    scope_id: str = "legacy"
    agent_id: str | None = None
    task_id: str | None = None
    kind: FactKind = "project_fact"
    text: str
    search_text: str = ""
    tags: list[str] = Field(default_factory=list)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    status: FactStatus = "active"
    temporal: TemporalInfo | None = None
    provenance: ProvenanceInfo = Field(default_factory=ProvenanceInfo)
    embedding_model: str | None = None
    created_at: datetime = Field(default_factory=now)
    updated_at: datetime = Field(default_factory=now)
    expires_at: datetime | None = None
    version: int = 1
    schema_version: int = 4


# ---------------------------------------------------------------------------
# Entity & Mention (§5.4)
# ---------------------------------------------------------------------------

class Entity(BaseModel):
    """A canonical entity record (§5.4)."""
    entity_id: str
    tenant_id: str = "default"
    canonical_name: str
    type: EntityType = "concept"
    aliases: list[str] = Field(default_factory=list)
    first_seen_at: datetime = Field(default_factory=now)
    last_seen_at: datetime = Field(default_factory=now)
    merged_into_id: str | None = None


class EntityMention(BaseModel):
    """A row in ``fact_entity_mentions`` linking a fact to an entity (§5.4)."""
    tenant_id: str = "default"
    fact_id: str
    entity_id: str | None = None
    mention_text: str = ""
    role: Literal["mentions", "subject", "object"] = "mentions"
    linking_confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class EntityAlias(BaseModel):
    """A row in ``entity_aliases``."""
    entity_id: str
    alias: str
    tenant_id: str = "default"


class EntityRef(BaseModel):
    """Legacy v3 embedded entity reference.

    In v4, entity mentions are stored in the ``fact_entity_mentions`` table
    and the ``Fact`` model no longer has a top-level ``entities`` field.
    This type is kept for the entity linker's return value.
    """
    entity_id: str
    name: str
    type: EntityType = "concept"
    role: Literal["subject", "object", "context", "mentions"] = "subject"


# ---------------------------------------------------------------------------
# FactRelation (§5.7)
# ---------------------------------------------------------------------------

class FactRelation(BaseModel):
    """A directed relationship between two facts (§5.7)."""
    tenant_id: str = "default"
    src_fact_id: str
    dst_fact_id: str
    relation: FactRelationType
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_by: str = "system"
    created_at: datetime = Field(default_factory=now)


# ---------------------------------------------------------------------------
# Hot Memory (§5.8)
# ---------------------------------------------------------------------------

class HotMemoryItem(BaseModel):
    """A hot-memory entry: small, stable, always-injected (§5.8)."""
    id: str
    tenant_id: str = "default"
    domain: MemoryDomain = "user"
    scope_id: str = "default"
    text: str
    priority: int = 0
    source_fact_ids: list[str] = Field(default_factory=list)
    status: HotStatus = "active"
    proposed_by: str = "system"
    approved_by: str | None = None
    created_at: datetime = Field(default_factory=now)
    updated_at: datetime = Field(default_factory=now)


# ---------------------------------------------------------------------------
# History (§5.9)
# ---------------------------------------------------------------------------

class SessionMessage(BaseModel):
    """A complete session message record in the History layer (§5.9)."""
    message_id: str
    tenant_id: str = "default"
    channel_id: str
    thread_id: str | None = None
    seq: int = 0
    role: SessionRole = "user"
    agent_id: str | None = None
    content: str = ""
    search_text: str = ""
    tool_call: dict[str, Any] | None = None
    file_diff: dict[str, Any] | None = None
    redaction_status: RedactionStatus = "clean"
    created_at: datetime = Field(default_factory=now)


# ---------------------------------------------------------------------------
# Skill (§5.10)
# ---------------------------------------------------------------------------

class SkillMetadata(BaseModel):
    """A reusable skill/procedure with independent versioning (§5.10).

    During Stage 2, this keeps the v3-era fields that ``skill_store.py`` and
    tests depend on (``id``, ``domain``, ``scope_id``, ``body``, ``version``,
    ``previous_version_id``, ``tags``).  The final split-metadata design
    (``skill_id``/``visibility``/``owner_scope_id``/``path``/``current_version``)
    will be layered in Stage 5.
    """
    id: str
    tenant_id: str = "default"
    domain: FactDomain = "agent"
    scope_id: str = "default"
    name: str
    description: str = ""
    body: str = ""
    version: int = 1
    previous_version_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    status: SkillStatus = "active"
    created_at: datetime = Field(default_factory=now)
    updated_at: datetime = Field(default_factory=now)


class SkillVersion(BaseModel):
    """An immutable snapshot of a skill at a specific version (§5.10)."""
    skill_id: str
    version: int
    snapshot_path: str
    checksum: str
    created_by: str = "system"
    created_at: datetime = Field(default_factory=now)
    change_summary: str = ""


# ---------------------------------------------------------------------------
# Retrieval query / result (§5.11)
# ---------------------------------------------------------------------------

class RetrievalQuery(BaseModel):
    """A hybrid retrieval query (three-signal + optional second stage)."""
    text: str
    actor: ActorContext | None = None
    tenant_id: str = "default"
    allowed_scopes: list[AllowedScope] = Field(default_factory=list)
    intent: MemoryIntent = "recall"
    domain: MemoryDomain | None = None
    scope_id: str | None = None
    entities: list[str] = Field(default_factory=list)
    as_of: datetime | None = None
    time_range: tuple[datetime | None, datetime | None] | None = None
    top_k: int = 10


class RetrievalResult(BaseModel):
    """A single retrieval hit with fused score and source signal."""
    fact: Fact
    score: float = 0.0
    signals: list[Literal["semantic", "bm25", "entity", "graph"]] = Field(default_factory=list)
    explanation: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Write response
# ---------------------------------------------------------------------------

class WriteResponse(BaseModel):
    """Response from a write-propose submission."""
    event_id: str
    fact_ids: list[str] = Field(default_factory=list)
    status: Literal["accepted", "skipped", "failed"] = "accepted"
    reason: str = ""


# ---------------------------------------------------------------------------
# Backend ABCs (merged from contracts.py — Memory 2.1 Stage 8)
# ---------------------------------------------------------------------------

class StorageBackend(ABC):
    """Fact storage backend (SQLite in v3, JSON in v1/v2)."""

    @abstractmethod
    async def list(self, scope: MemoryScope, *, include_inactive: bool = False) -> list[MemoryEntry]:
        raise NotImplementedError

    @abstractmethod
    async def recall(self, scope: MemoryScope, query: str, top_k: int) -> list[MemoryEntry]:
        raise NotImplementedError

    @abstractmethod
    async def upsert(self, scope: MemoryScope, entry: MemoryEntry) -> MemoryEntry:
        raise NotImplementedError

    @abstractmethod
    async def delete(self, scope: MemoryScope, entry_id: str) -> bool:
        raise NotImplementedError


class VectorBackend(ABC):
    """Embedding storage and ANN search (sqlite-vec in v3 stage 2)."""

    @abstractmethod
    async def upsert_embedding(self, fact_id: str, scope: MemoryScope, embedding: list[float]) -> None:
        raise NotImplementedError

    @abstractmethod
    async def search(self, embedding: list[float], scope: MemoryScope, top_k: int) -> list[tuple[str, float]]:
        """Return (fact_id, cosine_similarity) pairs."""
        raise NotImplementedError

    @abstractmethod
    async def delete_embedding(self, fact_id: str) -> bool:
        raise NotImplementedError


class GraphBackend(ABC):
    """Entity/relation edge storage and traversal (SQLite edge tables in v3 stage 3)."""

    @abstractmethod
    async def upsert_entity_mention(self, fact_id: str, entity_id: str, role: str, scope: MemoryScope) -> None:
        raise NotImplementedError

    @abstractmethod
    async def upsert_fact_relation(self, src_fact_id: str, dst_fact_id: str, relation: str, weight: float, scope: MemoryScope) -> None:
        raise NotImplementedError

    @abstractmethod
    async def entity_neighbors(self, entity_id: str, scope: MemoryScope, max_hops: int = 2) -> list[str]:
        """Return fact_ids reachable from the entity within max_hops."""
        raise NotImplementedError

    @abstractmethod
    async def related_facts(self, fact_id: str, scope: MemoryScope) -> list[tuple[str, str, float]]:
        """Return (fact_id, relation, weight) for facts related to the given fact."""
        raise NotImplementedError


class HotMemoryAPI(Protocol):
    """Hot Memory layer: small, stable, always-injected (R7)."""

    async def list_hot(self, scope: MemoryScope) -> list[MemoryEntry]:
        ...

    async def add_hot(self, scope: MemoryScope, text: str, category: str, importance: float = 0.8) -> MemoryEntry:
        ...

    async def approve_hot(self, scope: MemoryScope, item_id: str, approved_by: str) -> bool:
        ...

    async def archive_hot(self, scope: MemoryScope, item_id: str) -> bool:
        ...


class HistoryAPI(Protocol):
    """History layer: complete session records, FTS5 on-demand (R8)."""

    async def append_message(self, scope: MemoryScope, message: MemoryEntry) -> None:
        ...

    async def search_history(self, scope: MemoryScope, query: str, top_k: int = 10) -> list[MemoryEntry]:
        ...


class SkillAPI(Protocol):
    """Skill layer: reusable procedures with independent versioning (R6)."""

    async def create_skill(self, scope: MemoryScope, name: str, body: str, description: str = "") -> MemoryEntry:
        ...

    async def get_skill(self, scope: MemoryScope, name: str) -> MemoryEntry | None:
        ...

    async def update_skill(self, scope: MemoryScope, skill_id: str, body: str) -> MemoryEntry:
        ...

    async def rollback_skill(self, scope: MemoryScope, skill_id: str, to_version: int) -> MemoryEntry:
        ...


class TokenCounter(Protocol):
    def count(self, text: str) -> int:
        ...

    def truncate(self, text: str, budget: int) -> str:
        ...
