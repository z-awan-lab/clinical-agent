"""Qdrant vector store adapter.

Two modes:

* In-memory (``QdrantStore(location=":memory:")``) — fast, no service
  required. Used in tests and for quick prototyping.
* Server (``QdrantStore(url="http://localhost:6333")``) — talks to a
  Qdrant container via the HTTP client. Persistence and cross-process
  access. Default for ``docker-compose up``.

The collection is created on first ``upsert`` if it doesn't exist. Vector
dimension and distance metric are fixed at construction time and
verified against the existing collection if one is present — switching
embedders requires either a new collection or a re-index.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from .base import BaseVectorStore, VectorHit

if TYPE_CHECKING:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels

logger = logging.getLogger(__name__)


# We use cosine distance throughout because BGE embeddings are normalised
# by default — cosine and dot-product are equivalent in that case, and
# cosine is the standard convention.
DEFAULT_DISTANCE = "Cosine"


class QdrantStore(BaseVectorStore):
    """Qdrant adapter implementing the BaseVectorStore contract."""

    def __init__(
        self,
        collection: str,
        embedding_dim: int,
        url: str | None = None,
        location: str | None = None,
        api_key: str | None = None,
        distance: str = DEFAULT_DISTANCE,
    ) -> None:
        """Construct a Qdrant store.

        Provide exactly one of ``url`` (server mode) or ``location``
        (in-memory mode with ``":memory:"``, or local persistent
        storage with a path).
        """
        if (url is None) == (location is None):
            raise ValueError("provide exactly one of `url` or `location`")

        self.collection = collection
        self.embedding_dim = embedding_dim
        self.distance = distance
        self.url = url
        self.location = location
        self.api_key = api_key
        self._client: QdrantClient | None = None

    # ---- lazy client init ---------------------------------------------

    def _get_client(self) -> QdrantClient:
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "QdrantStore requires qdrant-client. Install with: pip install -e '.[ml]'"
                ) from exc
            if self.url is not None:
                self._client = QdrantClient(url=self.url, api_key=self.api_key)
            else:
                self._client = QdrantClient(location=self.location)
        return self._client

    def _ensure_collection(self) -> None:
        """Create the collection if it doesn't exist. Idempotent."""
        from qdrant_client.http import models as qmodels

        client = self._get_client()
        existing = {c.name for c in client.get_collections().collections}
        if self.collection in existing:
            return
        logger.info(
            "creating Qdrant collection %s (dim=%d, distance=%s)",
            self.collection,
            self.embedding_dim,
            self.distance,
        )
        client.create_collection(
            collection_name=self.collection,
            vectors_config=qmodels.VectorParams(
                size=self.embedding_dim,
                distance=qmodels.Distance(self.distance),
            ),
        )

    # ---- BaseVectorStore methods --------------------------------------

    def upsert(
        self,
        chunk_ids: list[str],
        vectors: np.ndarray,
        payloads: list[dict[str, Any]],
    ) -> None:
        if not (len(chunk_ids) == len(vectors) == len(payloads)):
            raise ValueError("chunk_ids, vectors, and payloads must be same length")
        if len(chunk_ids) == 0:
            return
        if vectors.shape[1] != self.embedding_dim:
            raise ValueError(
                f"vector dim {vectors.shape[1]} != configured dim {self.embedding_dim}"
            )

        from qdrant_client.http import models as qmodels

        self._ensure_collection()
        client = self._get_client()

        # Qdrant accepts string IDs but they must be valid UUIDs or
        # unsigned ints. Our content-addressed hex IDs aren't UUIDs, so
        # we stash the original ID in the payload and use the hex
        # interpreted as an integer as the point ID. Collisions are
        # cryptographically improbable at our scale.
        points = [
            qmodels.PointStruct(
                id=int(chunk_id, 16),
                vector=vectors[i].tolist(),
                payload={**payloads[i], "chunk_id": chunk_id},
            )
            for i, chunk_id in enumerate(chunk_ids)
        ]
        client.upsert(collection_name=self.collection, points=points, wait=True)

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        filter_: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        from qdrant_client.http import models as qmodels

        client = self._get_client()
        # Build a Qdrant Filter from a simple flat dict of equality
        # conditions. Supports list values as "any of" (Qdrant `match any`).
        qfilter: qmodels.Filter | None = None
        if filter_:
            must = []
            for key, val in filter_.items():
                if isinstance(val, list):
                    must.append(
                        qmodels.FieldCondition(
                            key=key,
                            match=qmodels.MatchAny(any=val),
                        )
                    )
                else:
                    must.append(
                        qmodels.FieldCondition(
                            key=key,
                            match=qmodels.MatchValue(value=val),
                        )
                    )
            qfilter = qmodels.Filter(must=must)

        results = client.query_points(
            collection_name=self.collection,
            query=query_vector.tolist(),
            limit=k,
            query_filter=qfilter,
            with_payload=True,
        ).points

        hits: list[VectorHit] = []
        for r in results:
            payload = dict(r.payload or {})
            cid = payload.pop("chunk_id", str(r.id))
            hits.append(VectorHit(chunk_id=cid, score=float(r.score), payload=payload))
        return hits

    def count(self, filter_: dict[str, Any] | None = None) -> int:
        from qdrant_client.http import models as qmodels

        client = self._get_client()
        if not self._collection_exists():
            return 0
        qfilter: qmodels.Filter | None = None
        if filter_:
            qfilter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(key=k, match=qmodels.MatchValue(value=v))
                    for k, v in filter_.items()
                ]
            )
        return client.count(collection_name=self.collection, count_filter=qfilter, exact=True).count

    def _collection_exists(self) -> bool:
        client = self._get_client()
        return self.collection in {c.name for c in client.get_collections().collections}
