"""Abstract base class for embedding backends.

Concrete implementations (BGE-large, MedCPT, Qwen3-Embedding) land in
Phase 2 alongside the guideline retrieval tool.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np


class BaseEmbedder(ABC):
    """Embed texts into a fixed-dimensional vector space."""

    model_name: ClassVar[str]
    embedding_dim: ClassVar[int]

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return a ``(len(texts), embedding_dim)`` float32 array."""
        ...

    def embed_one(self, text: str) -> np.ndarray:
        """Convenience wrapper for single-text embedding."""
        return self.embed([text])[0]
