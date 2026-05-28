"""Ingestion: chunking, source-specific loaders, and the Chunk dataclass."""

from .base import (
    BaseIngester,
    Chunk,
    EvidenceTier,
    SourceType,
    chunks_to_payloads,
    make_chunk_id,
    now_iso,
)
from .chunking import RecursiveCharacterChunker
from .simple import RawDocument, SimpleTextIngester

__all__ = [
    "BaseIngester",
    "Chunk",
    "EvidenceTier",
    "RawDocument",
    "RecursiveCharacterChunker",
    "SimpleTextIngester",
    "SourceType",
    "chunks_to_payloads",
    "make_chunk_id",
    "now_iso",
]
