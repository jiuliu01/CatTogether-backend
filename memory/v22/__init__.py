"""Memory 2.2 package — LLM extraction + Qdrant storage.

Public surface:
- ``Memory`` / ``MemorySearchResult`` (models)
- ``MemoryStore`` (Qdrant façade)
- ``MemoryExtractor`` (orchestrator hook target)
- ``get_memory_store`` / ``get_memory_extractor`` (lazy singletons)

The singletons are lazy and failure-tolerant: if Qdrant or the embedder cannot
be reached, the getters return ``None`` and the orchestrator skips v22 work
without breaking the main task.
"""
from __future__ import annotations

import logging
from typing import Optional

from memory.v22.models import Memory, MemorySearchResult  # noqa: F401  (re-export)
from memory.v22.memory_store import MemoryStore
from memory.v22.extractor import MemoryExtractor, ExtractionContext, ExtractionResult  # noqa: F401


logger = logging.getLogger(__name__)


_store: Optional[MemoryStore] = None
_extractor: Optional[MemoryExtractor] = None
_init_failed = False


def get_memory_store() -> Optional[MemoryStore]:
    """Lazy singleton for the Qdrant-backed MemoryStore.

    Returns ``None`` if Qdrant is unreachable, the embedder fails to load, or
    v22 is disabled. Callers must tolerate ``None``.
    """
    global _store, _init_failed
    if _init_failed:
        return None
    if _store is not None:
        return _store
    from config import settings
    if not settings.memory_v22_enabled:
        return None
    try:
        from qdrant_client import QdrantClient
        from fastembed import SparseTextEmbedding
        from memory.embedder import get_embedder

        client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=30,
        )
        # Probe the connection once so a misconfigured URL fails fast here
        # rather than on every agent invocation. Use a bounded timeout so a
        # transient Qdrant stall (GC pause, disk flush) doesn't permanently
        # mark the store as failed for the whole process.
        client.get_collections()
        dense = get_embedder()
        sparse = None
        try:
            sparse = SparseTextEmbedding(model_name="Qdrant/bm25")
        except Exception:
            logger.warning("BM25 sparse embedder unavailable; v22 will run dense-only")
        store = MemoryStore(
            client=client,
            dense_embedder=dense,
            sparse_embedder=sparse,
            collection_name=settings.qdrant_collection,
            vector_size=settings.memory_embedding_dim,
        )
        store.ensure_collection()
        _store = store
        logger.info("Memory 2.2 store ready (Qdrant %s, collection %s)",
                    settings.qdrant_url, settings.qdrant_collection)
        return _store
    except Exception:
        # Transient failures (timeout, connection reset) must NOT permanently
        # disable the store — return None for this call but let the next call
        # retry. Only hard config errors (bad URL/embedder) naturally persist.
        logger.warning("Memory 2.2 store init failed this attempt (will retry next call): %s", exc_info=True)
        return None


def get_memory_extractor() -> Optional[MemoryExtractor]:
    """Lazy singleton for the MemoryExtractor.

    Returns ``None`` if the store is unavailable or the memory agent is not
    registered. The memory agent is registered in ``bootstrap.register_builtins``.
    """
    global _extractor
    if _extractor is not None:
        return _extractor
    store = get_memory_store()
    if store is None:
        return None
    from core.registry import registry
    agent = registry.get("memory")
    if agent is None:
        logger.warning("memory agent not registered; v22 extraction unavailable")
        return None
    from config import settings
    _extractor = MemoryExtractor(
        store=store,
        memory_agent=agent,
        recent_turns=settings.memory_extract_recent_turns,
        related_top_k=settings.memory_extract_top_k,
    )
    return _extractor


def reset_for_tests() -> None:
    """Drop cached singletons (tests only)."""
    global _store, _extractor, _init_failed
    _store = None
    _extractor = None
    _init_failed = False
