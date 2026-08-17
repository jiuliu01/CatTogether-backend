"""Memory 2.2 flat data model.

Replaces the heavy v4 ``Fact`` (kind/status/supersede/schema_version/nine-
stage) with a flat record that maps 1:1 to a Qdrant Point:

    id   = Point ID (UUID, Qdrant requires unsigned int or UUID string)
    text = payload field indexed for BM25
    domain/attributed_to/linked_memory_ids/... = payload metadata

See ``word/5.Memory2.2实施方案.md`` §4.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


MemoryDomain = Literal["user", "project", "task", "agent"]
AttributedTo = Literal["user", "assistant"]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def memory_hash(text: str) -> str:
    """Content fingerprint for dedup. ``sha256(text)`` — domain is checked at
    the store layer (same text in different domains is allowed to coexist)."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


class Memory(BaseModel):
    """One memory record. Maps to one Qdrant Point.

    Fields are intentionally flat — no nested provenance/temporal objects.
    ``attributed_to`` carries the only provenance signal (who said it);
    everything else is metadata for filtering and correlation.
    """
    id: str
    text: str
    domain: MemoryDomain
    attributed_to: AttributedTo
    linked_memory_ids: list[str] = Field(default_factory=list)
    run_id: str | None = None
    channel_id: str | None = None
    thread_id: str | None = None
    agent_id: str | None = None
    event_type: str | None = None
    hash: str
    created_at: datetime = Field(default_factory=_utc_now)

    @classmethod
    def create(
        cls,
        *,
        text: str,
        domain: MemoryDomain,
        attributed_to: AttributedTo,
        linked_memory_ids: list[str] | None = None,
        run_id: str | None = None,
        channel_id: str | None = None,
        thread_id: str | None = None,
        agent_id: str | None = None,
        event_type: str | None = None,
        memory_id: str | None = None,
    ) -> "Memory":
        """Build a Memory with a fresh UUID and content hash.

        ``memory_id`` may be supplied (e.g. when migrating an existing record);
        otherwise a UUID4 hex string is minted, which Qdrant accepts as a
        Point ID.
        """
        import uuid

        # Use the dashed UUID form (str(uuid4)) so the id we hold matches what
        # Qdrant returns on retrieval — Qdrant normalizes 32-char hex to the
        # canonical 8-4-4-4-12 dashed form, which would break id equality
        # checks (e.g. linked_memory_ids validation) if we stored the hex form.
        return cls(
            id=memory_id or str(uuid.uuid4()),
            text=text,
            domain=domain,
            attributed_to=attributed_to,
            linked_memory_ids=list(linked_memory_ids or []),
            run_id=run_id,
            channel_id=channel_id,
            thread_id=thread_id,
            agent_id=agent_id,
            event_type=event_type,
            hash=memory_hash(text),
        )

    def to_payload(self) -> dict:
        """Serialize to the Qdrant Point payload dict.

        Everything except ``id`` and the vector lives in the payload. ``text``
        is stored here so BM25 (via the FTS5 mirror) and the ``memory_search``
        tool can return it without a second lookup.
        """
        return {
            "text": self.text,
            "domain": self.domain,
            "attributed_to": self.attributed_to,
            "linked_memory_ids": self.linked_memory_ids,
            "run_id": self.run_id,
            "channel_id": self.channel_id,
            "thread_id": self.thread_id,
            "agent_id": self.agent_id,
            "event_type": self.event_type,
            "hash": self.hash,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_payload(cls, point_id: str, payload: dict) -> "Memory":
        """Reconstruct a Memory from a Qdrant Point id + payload."""
        created = payload.get("created_at")
        try:
            created_dt = datetime.fromisoformat(created) if created else _utc_now()
        except (TypeError, ValueError):
            created_dt = _utc_now()
        return cls(
            id=str(point_id),
            text=payload.get("text", ""),
            domain=payload.get("domain", "agent"),  # type: ignore[arg-type]
            attributed_to=payload.get("attributed_to", "assistant"),  # type: ignore[arg-type]
            linked_memory_ids=list(payload.get("linked_memory_ids") or []),
            run_id=payload.get("run_id"),
            channel_id=payload.get("channel_id"),
            thread_id=payload.get("thread_id"),
            agent_id=payload.get("agent_id"),
            event_type=payload.get("event_type"),
            hash=payload.get("hash", memory_hash(payload.get("text", ""))),
            created_at=created_dt,
        )


class MemorySearchResult(BaseModel):
    """One hit from a memory search. ``score`` is the fused RRF score."""
    memory: Memory
    score: float
    signals: list[str] = Field(default_factory=list)
