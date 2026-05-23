"""Abstract base class for vector stores.

Qdrant adapter lands in Phase 2. The interface is deliberately minimal
so a FAISS or Chroma fallback could be added later without touching the
retrieval layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class VectorHit:
    """A single retrieval result with score and payload."""

    chunk_id: str
    score: float
    payload: dict[str, Any]


class BaseVectorStore(ABC):
    """Persistent vector store with metadata filtering."""

    @abstractmethod
    def upsert(
        self,
        chunk_ids: list[str],
        vectors: np.ndarray,
        payloads: list[dict[str, Any]],
    ) -> None:
        """Insert or update vectors with associated payloads."""
        ...

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        filter_: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        """Return the top-k nearest neighbours, optionally filtered."""
        ...

    @abstractmethod
    def count(self, filter_: dict[str, Any] | None = None) -> int:
        """Count items, optionally filtered."""
        ...
