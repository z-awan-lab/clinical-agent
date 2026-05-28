"""Tests for the Qdrant store using in-memory mode.

These run without any external service. They DO require the
``qdrant-client`` package, so they're gated: if qdrant-client isn't
installed the whole file is skipped. CI installs the ``ml`` extras
when running the full suite; the base CI run skips them and only
exercises the lightweight tests.
"""

from __future__ import annotations

import numpy as np
import pytest

qdrant_client = pytest.importorskip("qdrant_client")

from clinical_agent.vectorstore import QdrantStore  # noqa: E402


@pytest.fixture
def store() -> QdrantStore:
    return QdrantStore(
        collection="test_collection",
        embedding_dim=4,
        location=":memory:",
    )


def _random_vectors(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    # Normalise so cosine scores are well-behaved.
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


class TestUpsert:
    def test_must_provide_one_of_url_or_location(self) -> None:
        with pytest.raises(ValueError):
            QdrantStore(collection="c", embedding_dim=4)
        with pytest.raises(ValueError):
            QdrantStore(collection="c", embedding_dim=4, url="x", location=":memory:")

    def test_upsert_then_count(self, store: QdrantStore) -> None:
        vecs = _random_vectors(3, 4)
        store.upsert(
            chunk_ids=["aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb", "cccccccccccccccc"],
            vectors=vecs,
            payloads=[
                {"text": "a", "source_type": "guideline"},
                {"text": "b", "source_type": "guideline"},
                {"text": "c", "source_type": "research_abstract"},
            ],
        )
        assert store.count() == 3

    def test_upsert_empty_is_noop(self, store: QdrantStore) -> None:
        store.upsert(chunk_ids=[], vectors=np.zeros((0, 4)), payloads=[])
        assert store.count() == 0

    def test_mismatched_lengths_raise(self, store: QdrantStore) -> None:
        with pytest.raises(ValueError):
            store.upsert(
                chunk_ids=["a" * 16],
                vectors=_random_vectors(2, 4),
                payloads=[{"x": 1}, {"x": 2}],
            )

    def test_wrong_dim_raises(self, store: QdrantStore) -> None:
        with pytest.raises(ValueError):
            store.upsert(
                chunk_ids=["a" * 16],
                vectors=_random_vectors(1, 8),  # collection is dim 4
                payloads=[{"x": 1}],
            )


class TestSearch:
    def _populate(self, store: QdrantStore) -> np.ndarray:
        # Use deterministic vectors so we can predict the nearest neighbour.
        vectors = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        store.upsert(
            chunk_ids=[f"{i:016x}" for i in range(1, 5)],
            vectors=vectors,
            payloads=[
                {
                    "text": "guideline_a",
                    "source_type": "guideline",
                    "evidence_tier": "guideline_national",
                },
                {
                    "text": "guideline_b",
                    "source_type": "guideline",
                    "evidence_tier": "guideline_national",
                },
                {
                    "text": "review_a",
                    "source_type": "research_abstract",
                    "evidence_tier": "systematic_review",
                },
                {
                    "text": "consumer_a",
                    "source_type": "consumer_health",
                    "evidence_tier": "consumer_health",
                },
            ],
        )
        return vectors

    def test_search_returns_nearest_first(self, store: QdrantStore) -> None:
        self._populate(store)
        query = np.array([0.99, 0.01, 0.0, 0.0], dtype=np.float32)
        hits = store.search(query_vector=query, k=2)
        assert len(hits) == 2
        # The first vector [1, 0, 0, 0] should win.
        assert hits[0].payload["text"] == "guideline_a"
        # Scores should be monotonically non-increasing.
        assert hits[0].score >= hits[1].score

    def test_filter_by_source_type(self, store: QdrantStore) -> None:
        self._populate(store)
        query = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        hits = store.search(query_vector=query, k=5, filter_={"source_type": "guideline"})
        assert len(hits) == 2
        assert all(h.payload["source_type"] == "guideline" for h in hits)

    def test_filter_by_evidence_tier_list_is_any_of(self, store: QdrantStore) -> None:
        self._populate(store)
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        hits = store.search(
            query_vector=query,
            k=5,
            filter_={"evidence_tier": ["guideline_national", "systematic_review"]},
        )
        assert len(hits) == 3
        tiers = {h.payload["evidence_tier"] for h in hits}
        assert tiers == {"guideline_national", "systematic_review"}

    def test_hit_chunk_id_round_trips(self, store: QdrantStore) -> None:
        self._populate(store)
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        hits = store.search(query_vector=query, k=1)
        assert hits[0].chunk_id == f"{1:016x}"


class TestCount:
    def test_count_on_missing_collection_returns_zero(self) -> None:
        s = QdrantStore(collection="never_created", embedding_dim=4, location=":memory:")
        assert s.count() == 0

    def test_count_with_filter(self, store: QdrantStore) -> None:
        store.upsert(
            chunk_ids=[f"{i:016x}" for i in range(1, 4)],
            vectors=_random_vectors(3, 4),
            payloads=[
                {"source_type": "guideline"},
                {"source_type": "guideline"},
                {"source_type": "research_abstract"},
            ],
        )
        assert store.count() == 3
        assert store.count(filter_={"source_type": "guideline"}) == 2
        assert store.count(filter_={"source_type": "research_abstract"}) == 1
