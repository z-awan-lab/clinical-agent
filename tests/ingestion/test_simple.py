"""Tests for ingestion primitives and the SimpleTextIngester."""

from __future__ import annotations

from clinical_agent.ingestion import (
    Chunk,
    EvidenceTier,
    RawDocument,
    RecursiveCharacterChunker,
    SimpleTextIngester,
    SourceType,
    chunks_to_payloads,
    make_chunk_id,
)


class TestMakeChunkId:
    def test_deterministic(self) -> None:
        a = make_chunk_id("https://example.org/x", 0, "hello world")
        b = make_chunk_id("https://example.org/x", 0, "hello world")
        assert a == b

    def test_different_for_different_inputs(self) -> None:
        a = make_chunk_id("https://example.org/x", 0, "hello world")
        b = make_chunk_id("https://example.org/x", 1, "hello world")
        c = make_chunk_id("https://example.org/y", 0, "hello world")
        d = make_chunk_id("https://example.org/x", 0, "goodbye world")
        assert len({a, b, c, d}) == 4

    def test_length(self) -> None:
        cid = make_chunk_id("u", 0, "t")
        assert len(cid) == 16
        assert all(c in "0123456789abcdef" for c in cid)


class TestChunk:
    def _make(self) -> Chunk:
        return Chunk(
            chunk_id="abcdef0123456789",
            text="example text",
            source_type=SourceType.GUIDELINE,
            evidence_tier=EvidenceTier.GUIDELINE_NATIONAL,
            source_url="https://example.org/x",
            retrieved_at="2026-05-01T00:00:00Z",
            title="Example",
            section="Intro",
            chunk_index=0,
            extra={"flag": True},
        )

    def test_to_payload_flattens(self) -> None:
        p = self._make().to_payload()
        assert p["text"] == "example text"
        assert p["source_type"] == "guideline"
        assert p["evidence_tier"] == "guideline_national"
        assert p["source_url"] == "https://example.org/x"
        assert p["title"] == "Example"
        assert p["section"] == "Intro"
        assert p["chunk_index"] == 0
        assert p["extra.flag"] is True

    def test_to_dict_round_trips_enum_values(self) -> None:
        d = self._make().to_dict()
        assert d["source_type"] == "guideline"
        assert d["evidence_tier"] == "guideline_national"


class TestSimpleTextIngester:
    def test_emits_one_or_more_chunks_per_doc(self) -> None:
        docs = [
            RawDocument(text="A short document.", source_url="u1"),
            RawDocument(text="Another short doc.", source_url="u2"),
        ]
        ingester = SimpleTextIngester(docs)
        chunks = ingester.ingest()
        assert len(chunks) >= 2
        urls = {c.source_url for c in chunks}
        assert urls == {"u1", "u2"}

    def test_chunk_index_increments_within_doc(self) -> None:
        long_text = "Sentence number {i}. " * 500
        docs = [RawDocument(text=long_text, source_url="u1")]
        chunker = RecursiveCharacterChunker(chunk_size=300, chunk_overlap=30)
        chunks = SimpleTextIngester(docs, chunker=chunker).ingest()
        assert len(chunks) > 1
        for i, c in enumerate(chunks):
            assert c.chunk_index == i

    def test_propagates_source_type_and_evidence_tier(self) -> None:
        docs = [
            RawDocument(
                text="text",
                source_url="u1",
                source_type=SourceType.RESEARCH_ABSTRACT,
                evidence_tier=EvidenceTier.RCT,
            )
        ]
        chunks = SimpleTextIngester(docs).ingest()
        assert chunks[0].source_type == SourceType.RESEARCH_ABSTRACT
        assert chunks[0].evidence_tier == EvidenceTier.RCT

    def test_chunks_to_payloads_returns_parallel_lists(self) -> None:
        docs = [RawDocument(text="some text", source_url="u1")]
        chunks = SimpleTextIngester(docs).ingest()
        ids, payloads = chunks_to_payloads(chunks)
        assert len(ids) == len(payloads) == len(chunks)
        for cid, p in zip(ids, payloads, strict=True):
            assert isinstance(cid, str)
            assert "text" in p
