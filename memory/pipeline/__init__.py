"""Nine-stage write pipeline for Memory 2.1 (§7).

Stages:
    1 Event Store → 2 Candidate Extraction → 3 Fact Structuring
    → 4 Scope & Policy → 5 Provenance → 6 Entity & Temporal
    → 7 Embedding → 8 Dedup & Relations → 9 Transaction Commit + Outbox

ADD-only semantics (§7.2): new facts always INSERT; similarity finds
candidates but never overwrites; conflicts build ``contradicts`` relations;
trusted corrections build ``supersedes`` relations.

Usage::

    from memory.pipeline import WritePipeline
    from memory.models import Event

    pipeline = WritePipeline()
    response = await pipeline.process(event)

The pipeline is idempotent (same event_id + content hash → skipped),
checkpointable (event status flows pending→processed/failed/skipped), and
backpressure-aware (queue full → reject).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from memory.db import MemoryDB, memory_db
from memory.models import (
    Event, Fact, Entity, EntityMention, FactRelation, WriteResponse,
)
from memory.scope import MemoryScope


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline-internal types
# ---------------------------------------------------------------------------

@dataclass
class FactDraft:
    """Intermediate representation carried through stages 3-9.

    Stage 3 creates a draft with a partially-built ``Fact``.  Each subsequent
    stage enriches the draft.  Stage 9 commits the draft to the database.
    """
    fact: Fact
    entities: list[Entity] = field(default_factory=list)
    mentions: list[EntityMention] = field(default_factory=list)
    relations: list[FactRelation] = field(default_factory=list)
    embedding: list[float] | None = None
    scope: MemoryScope | None = None
    dedup_action: str = "add"
    """``add`` — insert as new Fact.
    ``skip`` — discard (true dedup: same source + content hash).
    ``relation_only`` — skip the Fact insert but commit relations."""
    skipped_reason: str = ""
    needs_reindex: bool = False
    """Set when embedding failed; commit stage writes a reindex outbox entry."""


@dataclass
class DedupDecision:
    """Output of stage 8 (Dedup & Relations)."""
    action: str = "add"
    relations: list[FactRelation] = field(default_factory=list)
    reason: str = ""


def _content_hash(text: str, kind: str, domain: str, scope_id: str) -> str:
    """Deterministic content hash for idempotency checking."""
    raw = f"{domain}|{scope_id}|{kind}|{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# WritePipeline — orchestrator
# ---------------------------------------------------------------------------

class WritePipeline:
    """Orchestrates the nine-stage write pipeline (§7).

    Each stage is a separate module; the orchestrator runs them in order,
    updating the event's ``processing_stage`` checkpoint after each stage.
    If any stage raises, the event is marked ``failed`` and can be retried.
    """

    def __init__(
        self,
        db: MemoryDB | None = None,
        *,
        queue_size: int | None = None,
    ) -> None:
        from config import settings
        self._db = db or memory_db
        self._queue: asyncio.Queue[Event] = asyncio.Queue(
            maxsize=queue_size or settings.memory_write_queue_size
        )
        self._worker: asyncio.Task | None = None
        self.stats: dict[str, int] = {
            "submitted": 0, "written": 0, "skipped": 0,
            "failed": 0, "deduplicated": 0,
        }

        # Lazily constructed stage instances.
        self._event_store: Any = None
        self._candidate_extractor: Any = None
        self._fact_structurer: Any = None
        self._scope_policy: Any = None
        self._provenance_builder: Any = None
        self._entity_temporal: Any = None
        self._embedding_stage: Any = None
        self._dedup_relations: Any = None
        self._commit_stage: Any = None

    # ----- Lazy stage accessors -----

    @property
    def event_store(self) -> Any:
        if self._event_store is None:
            from memory.pipeline.event_store import EventStore
            self._event_store = EventStore(self._db)
        return self._event_store

    @property
    def candidate_extractor(self) -> Any:
        if self._candidate_extractor is None:
            from memory.pipeline.candidate_extractor import CandidateExtractor
            self._candidate_extractor = CandidateExtractor()
        return self._candidate_extractor

    @property
    def fact_structurer(self) -> Any:
        if self._fact_structurer is None:
            from memory.pipeline.fact_structurer import FactStructurer
            self._fact_structurer = FactStructurer()
        return self._fact_structurer

    @property
    def scope_policy(self) -> Any:
        if self._scope_policy is None:
            from memory.pipeline.scope_policy import ScopePolicy
            self._scope_policy = ScopePolicy()
        return self._scope_policy

    @property
    def provenance_builder(self) -> Any:
        if self._provenance_builder is None:
            from memory.pipeline.provenance import ProvenanceBuilder
            self._provenance_builder = ProvenanceBuilder()
        return self._provenance_builder

    @property
    def entity_temporal(self) -> Any:
        if self._entity_temporal is None:
            from memory.pipeline.entity_temporal import EntityTemporalProcessor
            self._entity_temporal = EntityTemporalProcessor(self._db)
        return self._entity_temporal

    @property
    def embedding_stage(self) -> Any:
        if self._embedding_stage is None:
            from memory.pipeline.embedding import EmbeddingStage
            self._embedding_stage = EmbeddingStage()
        return self._embedding_stage

    @property
    def dedup_relations(self) -> Any:
        if self._dedup_relations is None:
            from memory.pipeline.dedup_relations import DedupRelations
            self._dedup_relations = DedupRelations(self._db)
        return self._dedup_relations

    @property
    def commit_stage(self) -> Any:
        if self._commit_stage is None:
            from memory.pipeline.commit import CommitStage
            self._commit_stage = CommitStage(self._db)
        return self._commit_stage

    # ----- Queue / worker -----

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()

    @property
    def running(self) -> bool:
        return self._worker is not None and not self._worker.done()

    async def start(self) -> None:
        if not self.running:
            self._worker = asyncio.create_task(
                self._run(), name="memory-write-pipeline-worker"
            )

    async def stop(self, timeout: float = 5.0) -> None:
        if not self._worker:
            return
        try:
            await asyncio.wait_for(self._queue.join(), timeout=timeout)
        except TimeoutError:
            pass
        self._worker.cancel()
        await asyncio.gather(self._worker, return_exceptions=True)
        self._worker = None

    async def submit(self, event: Event) -> bool:
        """Enqueue an event for background processing.

        Returns ``False`` if the queue is full (backpressure).
        """
        self.stats["submitted"] += 1
        # Stage 1: persist event immediately (even before queueing).
        self.event_store.put(event)
        try:
            self._queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            self.stats["failed"] += 1
            logger.warning("memory write queue is full; event %s rejected", event.event_id)
            return False

    async def _run(self) -> None:
        while True:
            event = await self._queue.get()
            try:
                await self.process(event)
            except Exception:
                self.stats["failed"] += 1
                logger.exception("memory write event failed: %s", event.event_id)
            finally:
                self._queue.task_done()

    # ----- Main processing -----

    async def process(self, event: Event) -> WriteResponse:
        """Run all nine stages for *event*.

        This is the synchronous (non-queued) entry point — call directly for
        testing or when you need the result immediately.  Use ``submit()`` for
        fire-and-forget background processing.
        """
        # Ensure the event is persisted (submit() does this too, but process()
        # may be called directly for sync testing).
        self.event_store.put(event)

        # Idempotency: skip if already processed.
        if self.event_store.is_processed(event.event_id, event.tenant_id):
            self.stats["skipped"] += 1
            return WriteResponse(
                event_id=event.event_id, status="skipped",
                reason="event already processed",
            )

        try:
            fact_ids = await self._run_stages(event)
            self.event_store.mark_status(
                event.event_id, event.tenant_id, "processed",
                fact_ids=fact_ids,
            )
            if fact_ids:
                self.stats["written"] += len(fact_ids)
            else:
                self.stats["skipped"] += 1
            return WriteResponse(
                event_id=event.event_id, fact_ids=fact_ids, status="accepted",
            )
        except Exception as exc:
            self.event_store.mark_status(
                event.event_id, event.tenant_id, "failed",
                error=str(exc),
            )
            logger.exception("pipeline failed for event %s", event.event_id)
            return WriteResponse(
                event_id=event.event_id, status="failed", reason=str(exc),
            )

    async def _run_stages(self, event: Event) -> list[str]:
        """Run stages 2-9 and return committed fact_ids."""
        # Stage 2: Candidate Extraction
        self.event_store.mark_stage(event.event_id, event.tenant_id, "candidate_extraction")
        candidates = await self.candidate_extractor.extract(event)
        if not candidates:
            self.event_store.mark_status(
                event.event_id, event.tenant_id, "skipped",
                error="no candidates",
            )
            return []

        # Stage 3: Fact Structuring
        self.event_store.mark_stage(event.event_id, event.tenant_id, "fact_structuring")
        drafts = self.fact_structurer.structure(candidates, event)

        # Stage 4: Scope & Policy
        self.event_store.mark_stage(event.event_id, event.tenant_id, "scope_policy")
        actor = self._build_actor_context(event)
        drafts = [
            d for d in
            (self.scope_policy.resolve(d, actor) for d in drafts)
            if d is not None
        ]
        if not drafts:
            return []

        # Stages 5-8 per draft
        for draft in drafts:
            # Stage 5: Provenance
            self.provenance_builder.build(draft, event)

            # Stage 6: Entity & Temporal
            self.entity_temporal.process(draft)

            # Stage 7: Embedding
            self.embedding_stage.embed(draft)

            # Stage 8: Dedup & Relations
            existing = self.dedup_relations.fetch_candidates(draft)
            decision = self.dedup_relations.evaluate(draft, existing)
            draft.dedup_action = decision.action
            draft.relations = decision.relations
            if decision.action == "skip":
                draft.skipped_reason = decision.reason
                self.stats["deduplicated"] += 1

        # Stage 9: Commit
        self.event_store.mark_stage(event.event_id, event.tenant_id, "commit")
        fact_ids = self.commit_stage.commit(drafts, event)
        return fact_ids

    @staticmethod
    def _build_actor_context(event: Event) -> Any:
        """Build an ActorContext from the event's actor + tenant."""
        from memory.permissions import ActorContext as PermActorContext

        # The Event.actor is a models.Actor; build the permissions ActorContext.
        actor = event.actor
        # Map actor kind to role: agent→agent, system→system, user→user.
        role = actor.kind if actor.kind in ("user", "agent", "system") else "user"
        return PermActorContext(
            tenant_id=event.tenant_id,
            role=role,  # type: ignore[arg-type]
            actor_id=actor.id,
            display_name=actor.display_name,
        )
