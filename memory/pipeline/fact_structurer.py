"""Stage 3: Fact Structuring — Candidate → FactDraft[] (§7.1).

Structures each ``MemoryCandidate`` into a ``FactDraft`` with a partially-built
``Fact``.  Key responsibilities:

- Set ``search_text = text`` (critical for FTS5 BM25 search).
- Determine ``status``: agent-proposed facts default to ``pending_review``;
  user/system facts default to ``active``.
- Map v1 domain names (``workspace`` → ``project``).
- Generate deterministic fact IDs.
- Structural errors → ``pending_review`` status.

Failure handling: structurally invalid candidates → pending_review.
"""
from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any

from core.output_safety import sanitize_agent_text
from memory.models import (
    Event, Fact, FactKind, ProvenanceInfo, TemporalInfo,
)
from memory.pipeline import FactDraft
from memory.scope import normalize_domain
from models.schemas import MemoryCandidate


# v1 kind → v4 FactKind mapping.  ``practice`` and ``working_note`` are removed
# in v4; they map to the closest surviving kind.
_KIND_MAP: dict[str, str] = {
    "preference": "preference",
    "correction": "correction",
    "project_fact": "project_fact",
    "decision": "decision",
    "progress": "progress",
    "convention": "convention",
    "practice": "convention",      # v1 practice → v4 convention
    "working_note": "progress",    # v1 working_note → v4 progress
    "goal": "goal",
    "constraint": "constraint",
    "outcome": "outcome",
    "relation": "relation",
}


def _fact_id(tenant_id: str, candidate: MemoryCandidate, event: Event) -> str:
    """Deterministic fact ID from tenant + content + event."""
    raw = f"{tenant_id}|{candidate.domain}|{candidate.kind}|{candidate.text}|{event.event_id}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"fact_{digest[:24]}"


class FactStructurer:
    """Stage 3: structure candidates into FactDrafts."""

    def structure(
        self,
        candidates: list[MemoryCandidate],
        event: Event,
    ) -> list[FactDraft]:
        """Convert candidates to FactDrafts.

        Each candidate becomes one draft.  Unsafe text is sanitised; if
        sanitisation removes all content, the candidate is dropped.
        """
        drafts: list[FactDraft] = []
        actor_kind = event.actor.kind

        for candidate in candidates:
            # Sanitise text (strip secrets / PII).
            safety = sanitize_agent_text(candidate.text)
            safe_text = safety.text.strip()
            if not safe_text:
                continue

            # Map domain (workspace → project).
            try:
                domain = normalize_domain(candidate.domain)
            except ValueError:
                continue

            # Map kind.
            kind = _KIND_MAP.get(candidate.kind, "project_fact")

            # Determine status: agent proposals → pending_review.
            is_agent = actor_kind == "agent"
            status = "pending_review" if is_agent else "active"

            # Build the Fact.
            fact_id = _fact_id(event.tenant_id, candidate, event)
            now = datetime.now(timezone.utc)

            p = event.payload or {}
            # Resolve scope_id from the event context.  The payload does not
            # carry a literal "scope_id"; instead the scope is derived from
            # the domain-specific identifier (workspace_id for project, task_id
            # for task, agent_id for agent).  event.scope_hint is the fallback
            # set by the API layer from proposal.scope_hint.
            scope_id = (
                p.get("scope_id")
                or (p.get("workspace_id") if domain == "project" else None)
                or (p.get("task_id") if domain == "task" else None)
                or (p.get("agent_id") if domain == "agent" else None)
                or getattr(event, "scope_hint", None)
                or "default"
            )
            from memory.scope import canonical_scope_id
            scope_id = canonical_scope_id(domain, str(scope_id))
            fact = Fact(
                id=fact_id,
                tenant_id=event.tenant_id,
                domain=domain,
                scope_id=scope_id,
                agent_id=p.get("agent_id") if is_agent else None,
                task_id=p.get("task_id"),
                kind=kind,  # type: ignore[arg-type]
                text=safe_text,
                search_text=safe_text,  # FTS5 indexes search_text, not text.
                tags=list(candidate.tags),
                importance=candidate.importance,
                confidence=candidate.confidence,
                status=status,  # type: ignore[arg-type]
                temporal=None,
                provenance=ProvenanceInfo(
                    source_type=candidate.source_type,  # type: ignore[arg-type]
                    source_ids=list(candidate.source_ids),
                    author=event.actor.id,
                    extractor="rule",
                    extraction_confidence=candidate.confidence,
                ),
                created_at=now,
                updated_at=now,
                schema_version=4,
            )

            drafts.append(FactDraft(fact=fact))

        return drafts
