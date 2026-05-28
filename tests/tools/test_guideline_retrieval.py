"""Tests for the GuidelineRetrievalTool.

We use a deterministic fake embedder so retrieval is predictable without
loading a real model. The vector store is the real ``QdrantStore`` in
in-memory mode when ``qdrant-client`` is available, otherwise the test
is skipped (matches the gating in ``test_qdrant_store.py``).
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

qdrant_client = pytest.importorskip("qdrant_client")

from clinical_agent.embeddings.base import BaseEmbedder  # noqa: E402
from clinical_agent.ingestion import (  # noqa: E402
    EvidenceTier,
    RawDocument,
    SimpleTextIngester,
    SourceType,
    chunks_to_payloads,
)
from clinical_agent.tools import GuidelineRetrievalTool  # noqa: E402
from clinical_agent.vectorstore import QdrantStore  # noqa: E402


class FakeEmbedder(BaseEmbedder):
    """Deterministic 4-dim embedder for tests.

    Maps each keyword to an axis: sepsis -> [1,0,0,0], qsofa -> [0,1,0,0],
    pain -> [0,0,1,0], fever -> [0,0,0,1]. Anything else is the zero
    vector (which will retrieve poorly — by design).
    """

    model_name: ClassVar[str] = "fake-4d"
    embedding_dim: ClassVar[int] = 4

    KEYWORDS: ClassVar[list[str]] = ["sepsis", "qsofa", "pain", "fever"]

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), 4), dtype=np.float32)
        for i, t in enumerate(texts):
            lower = t.lower()
            for j, kw in enumerate(self.KEYWORDS):
                if kw in lower:
                    out[i, j] = 1.0
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]


@pytest.fixture
def populated_tool() -> GuidelineRetrievalTool:
    docs = [
        RawDocument(
            text="Sepsis is a life-threatening organ dysfunction.",
            source_url="https://example.org/sepsis",
            title="Sepsis overview",
            source_type=SourceType.GUIDELINE,
            evidence_tier=EvidenceTier.GUIDELINE_NATIONAL,
        ),
        RawDocument(
            text="qSOFA is a bedside score for sepsis screening.",
            source_url="https://example.org/qsofa",
            title="qSOFA",
            source_type=SourceType.RESEARCH_ABSTRACT,
            evidence_tier=EvidenceTier.SYSTEMATIC_REVIEW,
        ),
        RawDocument(
            text="Fever may indicate infection.",
            source_url="https://example.org/fever",
            title="Fever",
            source_type=SourceType.CONSUMER_HEALTH,
            evidence_tier=EvidenceTier.CONSUMER_HEALTH,
        ),
    ]
    chunks = SimpleTextIngester(docs).ingest()
    embedder = FakeEmbedder()
    vectors = embedder.embed([c.text for c in chunks])
    store = QdrantStore(
        collection="test_guidelines",
        embedding_dim=embedder.embedding_dim,
        location=":memory:",
    )
    ids, payloads = chunks_to_payloads(chunks)
    store.upsert(chunk_ids=ids, vectors=vectors, payloads=payloads)
    return GuidelineRetrievalTool(embedder=embedder, store=store)


class TestSpec:
    def test_spec_well_formed(self) -> None:
        tool = GuidelineRetrievalTool(embedder=FakeEmbedder(), store=None)  # type: ignore[arg-type]
        spec = tool.to_spec()
        assert spec["name"] == "guideline_retrieval"
        assert spec["input_schema"]["required"] == ["query"]


class TestRetrieval:
    def test_query_returns_most_relevant_chunk_first(
        self, populated_tool: GuidelineRetrievalTool
    ) -> None:
        r = populated_tool.run(query="sepsis", k=3)
        assert r.success
        assert len(r.data["results"]) == 3
        assert "sepsis" in r.data["results"][0]["text"].lower()

    def test_empty_query_rejected(self, populated_tool: GuidelineRetrievalTool) -> None:
        r = populated_tool.run(query="   ")
        assert not r.success
        assert "query" in (r.error or "").lower()

    def test_k_is_clamped(self, populated_tool: GuidelineRetrievalTool) -> None:
        r = populated_tool.run(query="sepsis", k=999)
        # At most as many as we indexed.
        assert r.success
        assert len(r.data["results"]) <= 3

    def test_filter_by_source_type(self, populated_tool: GuidelineRetrievalTool) -> None:
        r = populated_tool.run(query="sepsis", k=5, source_type="guideline")
        assert r.success
        assert all(x["source_type"] == "guideline" for x in r.data["results"])

    def test_filter_by_evidence_tier_list(self, populated_tool: GuidelineRetrievalTool) -> None:
        r = populated_tool.run(
            query="sepsis",
            k=5,
            evidence_tier=["guideline_national", "systematic_review"],
        )
        assert r.success
        tiers = {x["evidence_tier"] for x in r.data["results"]}
        assert tiers <= {"guideline_national", "systematic_review"}

    def test_min_score_filters_low_quality_hits(
        self, populated_tool: GuidelineRetrievalTool
    ) -> None:
        # A query that doesn't match any keyword produces a zero query
        # vector; cosine against any real vector is undefined or 0.
        r = populated_tool.run(query="completely_unrelated_topic_xyz", k=5, min_score=0.5)
        assert r.success
        assert r.data["results"] == []

    def test_metadata_reports_filter_and_count(
        self, populated_tool: GuidelineRetrievalTool
    ) -> None:
        r = populated_tool.run(query="sepsis", k=2, source_type="guideline")
        assert r.metadata["filter_applied"] == {"source_type": "guideline"}
        assert r.metadata["n_results"] == len(r.data["results"])

    def test_results_contain_provenance(self, populated_tool: GuidelineRetrievalTool) -> None:
        r = populated_tool.run(query="sepsis", k=1)
        first = r.data["results"][0]
        assert first["source_url"].startswith("https://example.org/")
        assert first["chunk_id"]
        assert first["score"] is not None
