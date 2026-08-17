"""Stage 5: Provenance — Draft + Event → complete ProvenanceInfo (§7.1).

Enriches the draft's ``fact.provenance`` with information from the event:
- ``author`` from ``event.actor``
- ``extractor`` from the event payload (rule/llm/manual/migration)
- ``source_ids`` from the event's source message IDs
- ``evidence_refs`` from mutation paths in the event payload

If provenance is missing a source (no source_ids and no author), the fact
is demoted to ``pending_review`` — we never commit a fact with no provenance.
"""
from __future__ import annotations

import logging
from typing import Any

from memory.models import Event, EvidenceRef, ProvenanceInfo
from memory.pipeline import FactDraft


logger = logging.getLogger(__name__)


class ProvenanceBuilder:
    """Stage 5: complete the provenance info on a draft."""

    def build(self, draft: FactDraft, event: Event) -> None:
        """Enrich ``draft.fact.provenance`` from *event*.

        Mutates the draft in place.  If provenance is incomplete (no source
        and no author), the fact status is demoted to ``pending_review``.
        """
        fact = draft.fact
        p = event.payload or {}
        prov = fact.provenance

        # Author from event actor.
        if not prov.author:
            prov = prov.model_copy(update={"author": event.actor.id})

        # Extractor: check payload for an explicit extractor, else keep existing.
        extractor = p.get("extractor", prov.extractor)
        if extractor != prov.extractor:
            prov = prov.model_copy(update={"extractor": extractor})  # type: ignore[arg-type]

        # Source IDs: merge event source_message_ids with existing.
        event_source_ids = p.get("source_message_ids", [])
        if event_source_ids:
            merged = list(dict.fromkeys(
                list(prov.source_ids) + list(event_source_ids)
            ))
            prov = prov.model_copy(update={"source_ids": merged})

        # Evidence refs from mutation paths.
        mutation_paths = p.get("mutation_paths", [])
        if mutation_paths:
            existing_refs = list(prov.evidence_refs)
            for path in mutation_paths:
                # Avoid duplicates.
                if not any(er.ref == path for er in existing_refs):
                    existing_refs.append(EvidenceRef(
                        kind="file", ref=path, summary="",
                    ))
            prov = prov.model_copy(update={"evidence_refs": existing_refs})

        # Extraction confidence from payload if present.
        extraction_conf = p.get("extraction_confidence")
        if extraction_conf is not None and extraction_conf != prov.extraction_confidence:
            prov = prov.model_copy(
                update={"extraction_confidence": float(extraction_conf)}
            )

        # Write back the completed provenance.
        fact.provenance = prov

        # Completeness check: a fact with no source and no author is quarantined.
        if not prov.source_ids and not prov.author:
            logger.warning(
                "provenance: fact %s has no source_ids and no author; "
                "demoting to pending_review",
                fact.id,
            )
            fact.status = "pending_review"  # type: ignore[assignment]
