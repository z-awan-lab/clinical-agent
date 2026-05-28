"""BGE-large-en-v1.5 embedder.

The model is loaded lazily on first ``embed`` call. This keeps
``import clinical_agent`` fast for users who only need the lightweight
parts (e.g. running tests, exploring the schema), and means the heavy
ML dependencies are optional at install time.

The default model is ``BAAI/bge-large-en-v1.5`` (1024-dim, English).
Switch to a smaller model (e.g. ``BAAI/bge-small-en-v1.5``, 384-dim)
for faster indexing during development; rebuild the index when you
switch since dimensions change.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from .base import BaseEmbedder

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


# Known dimensions for BGE family models. Used for ahead-of-time
# validation against the vector store schema; the actual dimension is
# verified against the loaded model on first embed.
_BGE_DIMS: dict[str, int] = {
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-small-en-v1.5": 384,
}


class BGEEmbedder(BaseEmbedder):
    """BGE embedder via the ``sentence-transformers`` library.

    BGE family models expect a "Represent this sentence for searching
    relevant passages: " prefix on *queries* but not on passages. The
    ``embed`` method below is for passages; use ``embed_query`` when
    embedding a user question for retrieval.
    """

    model_name: ClassVar[str] = "BAAI/bge-large-en-v1.5"
    embedding_dim: ClassVar[int] = 1024

    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: str | None = None,
        batch_size: int = 32,
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name  # type: ignore[misc]
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self._model: SentenceTransformer | None = None
        # Surface the expected dimension early when we know it.
        if model_name in _BGE_DIMS:
            self.embedding_dim = _BGE_DIMS[model_name]  # type: ignore[misc]

    # ---- lazy model loading -------------------------------------------

    def _ensure_loaded(self) -> SentenceTransformer:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "BGEEmbedder requires sentence-transformers. "
                    "Install with: pip install -e '.[ml]'"
                ) from exc
            logger.info("loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name, device=self.device)
            # Verify the dim we advertised matches reality.
            actual_dim = int(self._model.get_sentence_embedding_dimension())
            if actual_dim != self.embedding_dim:
                logger.warning(
                    "advertised dim %d != actual dim %d for %s; updating",
                    self.embedding_dim,
                    actual_dim,
                    self.model_name,
                )
                self.embedding_dim = actual_dim  # type: ignore[misc]
        return self._model

    # ---- public API ---------------------------------------------------

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed passages. Returns ``(len(texts), embedding_dim)`` float32."""
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        model = self._ensure_loaded()
        vectors = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query string with the BGE retrieval prefix."""
        prefixed = self.QUERY_PREFIX + query
        return self.embed([prefixed])[0]
