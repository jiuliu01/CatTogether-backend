"""Stage 7: Embedding — Draft → 512-dim vector (§7.1).

Generates an embedding vector for the fact's text using the configured
embedder (DummyEmbedder for tests, SentenceTransformerEmbedder for production).

Failure handling: if embedding generation fails, the draft's ``embedding``
is set to ``None`` and ``needs_reindex`` is flagged.  The commit stage will
write a reindex outbox entry so the embedding can be regenerated later.
The fact is still committed — BM25 search will work without an embedding.
"""
from __future__ import annotations

import logging
from typing import Any

from memory.pipeline import FactDraft


logger = logging.getLogger(__name__)


class EmbeddingStage:
    """Stage 7: generate an embedding vector for a draft.

    The embedder is lazily loaded on first use.  If it fails (model not
    available, OOM, etc.), the draft is flagged for reindex and processing
    continues without an embedding.
    """

    def __init__(self, embedder: Any = None) -> None:
        self._embedder = embedder

    @property
    def embedder(self) -> Any:
        if self._embedder is None:
            from memory.embedder import get_embedder
            self._embedder = get_embedder()
        return self._embedder

    def embed(self, draft: FactDraft) -> None:
        """Generate an embedding for ``draft.fact.text``.

        On success, sets ``draft.embedding`` and ``draft.fact.embedding_model``.
        On failure, sets ``draft.needs_reindex = True`` and leaves
        ``draft.embedding = None``.
        """
        fact = draft.fact
        try:
            vector = self.embedder.embed(fact.text)
            draft.embedding = vector
            fact.embedding_model = self.embedder.model_name
        except Exception:
            logger.exception(
                "embedding failed for fact %s; flagging for reindex",
                fact.id,
            )
            draft.embedding = None
            draft.needs_reindex = True
            fact.embedding_model = None
