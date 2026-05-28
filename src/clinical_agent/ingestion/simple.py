"""Simple in-memory ingester.

Takes a list of ``RawDocument`` records and produces ``Chunk`` objects
via a configurable chunker. Used in tests and for quick prototyping
before the source-specific HTTP ingesters land.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from .base import (
    BaseIngester,
    Chunk,
    EvidenceTier,
    SourceType,
    make_chunk_id,
    now_iso,
)
from .chunking import RecursiveCharacterChunker


@dataclass
class RawDocument:
    """A single document waiting to be chunked."""

    text: str
    source_url: str
    title: str | None = None
    section: str | None = None
    source_type: SourceType = SourceType.GUIDELINE
    evidence_tier: EvidenceTier = EvidenceTier.UNCLASSIFIED
    extra: dict[str, Any] = field(default_factory=dict)


class SimpleTextIngester(BaseIngester):
    """Chunk a list of in-memory documents.

    The default chunker is suitable for clinical guideline text. Override
    via the ``chunker`` argument for different chunking behaviour.
    """

    source_name = "simple_text"
    default_source_type = SourceType.GUIDELINE
    default_evidence_tier = EvidenceTier.UNCLASSIFIED

    def __init__(
        self,
        documents: list[RawDocument],
        chunker: RecursiveCharacterChunker | None = None,
    ) -> None:
        self.documents = documents
        self.chunker = chunker or RecursiveCharacterChunker()

    def iter_chunks(self) -> Iterator[Chunk]:
        retrieved_at = now_iso()
        for doc in self.documents:
            for i, text in enumerate(self.chunker.split(doc.text)):
                yield Chunk(
                    chunk_id=make_chunk_id(doc.source_url, i, text),
                    text=text,
                    source_type=doc.source_type,
                    evidence_tier=doc.evidence_tier,
                    source_url=doc.source_url,
                    retrieved_at=retrieved_at,
                    title=doc.title,
                    section=doc.section,
                    chunk_index=i,
                    extra=doc.extra,
                )
