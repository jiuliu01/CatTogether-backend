"""Embedding generation for Memory 2.0 semantic retrieval (§6 环9, §7 Semantic).

The ``Embedder`` ABC decouples embedding generation from the vector backend.
Two implementations are provided:

- ``DummyEmbedder`` — deterministic hash-based pseudo-embedding for tests and
  development environments without a model download.  Produces a fixed-dimension
  vector from the SHA-256 of the text, normalised to unit length.  Not
  semantically meaningful but stable and fast.
- ``SentenceTransformerEmbedder`` — lazy-loaded ``sentence-transformers`` wrapper
  for the configured local model (default ``bge-small-zh-v1.5``).  Only imported
  when actually used so the heavy dependency stays optional.

The embedder is a *pure function* — it has no side effects and does not touch
the database.  The write pipeline calls ``embed(text)`` and passes the result
to ``VectorBackend.upsert_embedding``.
"""
from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from typing import Any

from config import settings


class Embedder(ABC):
    """Abstract embedding generator."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimensionality."""
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier for provenance tracking."""
        raise NotImplementedError

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for *text*."""
        raise NotImplementedError

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Default implementation loops; subclasses can override for efficiency.
        """
        return [self.embed(t) for t in texts]


class DummyEmbedder(Embedder):
    """Deterministic hash-based pseudo-embedding for tests / dev.

    Maps text → SHA-256 → dimension-d vector of +1/-1 signs, normalised to
    unit length.  Two identical texts always produce the same vector; similar
    texts are *not* semantically close — this is for plumbing tests only.
    """

    def __init__(self, dim: int | None = None) -> None:
        self._dim = dim or settings.memory_embedding_dim

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return "dummy-hash"

    def embed(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # Expand hash bytes to cover dim dimensions by re-hashing.
        signs: list[float] = []
        block = h
        while len(signs) < self._dim:
            for b in block:
                if len(signs) >= self._dim:
                    break
                signs.append(1.0 if (b & 1) else -1.0)
            block = hashlib.sha256(block).digest()
        norm = math.sqrt(float(len(signs)))
        return [s / norm for s in signs]


class SentenceTransformerEmbedder(Embedder):
    """Local sentence-transformers embedding (e.g. bge-small-zh-v1.5).

    The ``sentence-transformers`` package is imported lazily so it remains an
    optional dependency.  If the package or model is unavailable, a
    ``RuntimeError`` is raised on first ``embed()`` call.
    """

    def __init__(
        self,
        model_name: str | None = None,
        dim: int | None = None,
    ) -> None:
        self._model_name = model_name or settings.memory_embedding_model
        self._dim = dim or settings.memory_embedding_dim
        self._model: Any = None

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed; "
                "pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, text: str) -> list[float]:
        model = self._load()
        vec = model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        vecs = model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vecs]


def get_embedder() -> Embedder:
    """Factory: return the configured embedder.

    - ``provider="local"`` → SentenceTransformerEmbedder
    - ``provider="dummy"`` → DummyEmbedder
    - Otherwise → DummyEmbedder (safe default)
    """
    provider = settings.memory_embedding_provider
    model = settings.memory_embedding_model
    if provider == "local":
        return SentenceTransformerEmbedder()
    if provider == "dummy":
        return DummyEmbedder()
    # Safe fallback for tests / dev without model downloads.
    return DummyEmbedder()
