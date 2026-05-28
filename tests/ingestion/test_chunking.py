"""Tests for the recursive character chunker."""

from __future__ import annotations

import pytest

from clinical_agent.ingestion.chunking import RecursiveCharacterChunker


class TestConstruction:
    def test_default_constructs(self) -> None:
        c = RecursiveCharacterChunker()
        assert c.chunk_size > 0
        assert c.chunk_overlap >= 0

    def test_chunk_overlap_must_be_less_than_size(self) -> None:
        with pytest.raises(ValueError):
            RecursiveCharacterChunker(chunk_size=100, chunk_overlap=100)

    def test_negative_overlap_rejected(self) -> None:
        with pytest.raises(ValueError):
            RecursiveCharacterChunker(chunk_size=100, chunk_overlap=-1)

    def test_zero_chunk_size_rejected(self) -> None:
        with pytest.raises(ValueError):
            RecursiveCharacterChunker(chunk_size=0)

    def test_empty_separators_rejected(self) -> None:
        with pytest.raises(ValueError):
            RecursiveCharacterChunker(separators=())


class TestSplit:
    def test_empty_input_returns_empty_list(self) -> None:
        assert RecursiveCharacterChunker().split("") == []
        assert RecursiveCharacterChunker().split("   ") == []

    def test_short_text_returns_single_chunk(self) -> None:
        text = "Sepsis is a medical emergency."
        chunks = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=10).split(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_splits_into_multiple_chunks(self) -> None:
        # Build a text with clear paragraph boundaries so we can predict
        # how it splits.
        paragraphs = [f"Paragraph {i}. " * 30 for i in range(5)]
        text = "\n\n".join(paragraphs)
        chunks = RecursiveCharacterChunker(chunk_size=400, chunk_overlap=50).split(text)
        assert len(chunks) > 1

    def test_chunks_respect_size_bound_approximately(self) -> None:
        text = "Lorem ipsum dolor sit amet. " * 200
        size = 500
        chunks = RecursiveCharacterChunker(chunk_size=size, chunk_overlap=50).split(text)
        # Allow some slack — the chunker prefers natural boundaries over
        # exact size, and overlap material is added to the start of each
        # subsequent chunk.
        for c in chunks:
            assert len(c) <= size * 1.5 + 50

    def test_overlap_creates_shared_prefix_between_consecutive_chunks(self) -> None:
        from itertools import pairwise

        text = ". ".join(f"sentence number {i}" for i in range(200)) + "."
        chunker = RecursiveCharacterChunker(chunk_size=200, chunk_overlap=80)
        chunks = chunker.split(text)
        assert len(chunks) >= 2
        # The tail of chunk[i] should appear at the start of chunk[i+1]
        # for at least one pair (greedy merge means it might not happen
        # at every boundary but should happen somewhere).
        any_overlap = False
        for a, b in pairwise(chunks):
            tail = a[-30:]
            if tail and tail in b[:120]:
                any_overlap = True
                break
        assert any_overlap, "expected overlap between at least one consecutive pair"

    def test_no_content_loss_on_simple_paragraph_split(self) -> None:
        paragraphs = ["First paragraph here.", "Second paragraph here.", "Third paragraph here."]
        text = "\n\n".join(paragraphs)
        chunks = RecursiveCharacterChunker(chunk_size=2048).split(text)
        joined = " ".join(chunks)
        for p in paragraphs:
            assert p in joined

    def test_text_with_no_separators_at_all(self) -> None:
        # Pathological case: a single 1000-char token. The final ""
        # separator must kick in.
        text = "a" * 1000
        chunks = RecursiveCharacterChunker(chunk_size=200, chunk_overlap=20).split(text)
        assert len(chunks) >= 5
        # Total characters across chunks (minus overlap) should at least
        # match the input length.
        assert sum(len(c) for c in chunks) >= len(text)
