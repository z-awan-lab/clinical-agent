"""Tests for the BGE embedder.

The model load is heavy (~1.5GB download for bge-large) so the full
embed test is gated by ``CLINICAL_AGENT_ML_TESTS=1``. The unit tests
below cover the parts that don't require the model.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from clinical_agent.embeddings import BGEEmbedder


class TestBGEEmbedder:
    def test_construct_with_known_model_sets_dim(self) -> None:
        e = BGEEmbedder(model_name="BAAI/bge-large-en-v1.5")
        assert e.embedding_dim == 1024
        e_small = BGEEmbedder(model_name="BAAI/bge-small-en-v1.5")
        assert e_small.embedding_dim == 384

    def test_construct_does_not_load_model(self) -> None:
        # If construction loaded the model, this test would take 30+ seconds.
        e = BGEEmbedder(model_name="BAAI/bge-large-en-v1.5")
        assert e._model is None

    def test_embed_empty_returns_empty_array(self) -> None:
        e = BGEEmbedder(model_name="BAAI/bge-large-en-v1.5")
        result = e.embed([])
        assert isinstance(result, np.ndarray)
        assert result.shape == (0, 1024)


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("CLINICAL_AGENT_ML_TESTS") != "1",
    reason="set CLINICAL_AGENT_ML_TESTS=1 to run tests that load embedding models",
)
class TestBGEEmbedderWithModel:
    """Full model-load tests. Slow; gated by env var."""

    def test_embed_returns_correct_shape(self) -> None:
        # Use bge-small for speed — same interface, 384-dim, 130MB download.
        e = BGEEmbedder(model_name="BAAI/bge-small-en-v1.5")
        out = e.embed(["sepsis is a medical emergency", "qSOFA scoring"])
        assert out.shape == (2, 384)
        assert out.dtype == np.float32

    def test_embed_query_uses_prefix(self) -> None:
        e = BGEEmbedder(model_name="BAAI/bge-small-en-v1.5")
        # Embedding a raw string vs the prefixed form should differ.
        plain = e.embed(["sepsis criteria"])[0]
        as_query = e.embed_query("sepsis criteria")
        assert not np.allclose(plain, as_query)

    def test_embeddings_are_normalised_by_default(self) -> None:
        e = BGEEmbedder(model_name="BAAI/bge-small-en-v1.5")
        out = e.embed(["one", "two", "three"])
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-4)
