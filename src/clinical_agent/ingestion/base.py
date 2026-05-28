"""Ingestion primitives — the contract every source-specific ingester implements.

A ``Chunk`` is the atomic unit indexed by the vector store. Every chunk
carries provenance (``source_url``, ``retrieved_at``), a coarse type
(``source_type``), and an evidence tier so retrieval can filter and the
agent can surface why it trusts (or distrusts) a piece of evidence.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from typing import Any


class SourceType(StrEnum):
    """Coarse-grained classification surfaced to the agent and the UI."""

    GUIDELINE = "guideline"
    CONSUMER_HEALTH = "consumer_health"
    RESEARCH_ABSTRACT = "research_abstract"


class EvidenceTier(StrEnum):
    """Manually assigned per-source, propagated to every chunk.

    Ordering reflects the typical clinical evidence hierarchy. The agent
    can prefer higher tiers when consolidating retrieved evidence, and
    the demo UI surfaces the tier to the reader.
    """

    GUIDELINE_NATIONAL = "guideline_national"  # NICE, CDC, NIH, WHO
    SYSTEMATIC_REVIEW = "systematic_review"
    RCT = "rct"
    OBSERVATIONAL = "observational"
    NARRATIVE = "narrative"
    CONSUMER_HEALTH = "consumer_health"
    UNCLASSIFIED = "unclassified"


@dataclass(frozen=True)
class Chunk:
    """A single retrievable unit of text with provenance.

    The ``chunk_id`` is content-addressed (sha256 prefix of source_url +
    chunk_index + text) so the same source ingested twice yields the same
    IDs and re-indexing is idempotent.
    """

    chunk_id: str
    text: str
    source_type: SourceType
    evidence_tier: EvidenceTier
    source_url: str
    retrieved_at: str
    title: str | None = None
    section: str | None = None
    chunk_index: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        """Flatten to a vector-store payload dict."""
        return {
            "text": self.text,
            "source_type": self.source_type.value,
            "evidence_tier": self.evidence_tier.value,
            "source_url": self.source_url,
            "retrieved_at": self.retrieved_at,
            "title": self.title,
            "section": self.section,
            "chunk_index": self.chunk_index,
            **{f"extra.{k}": v for k, v in self.extra.items()},
        }

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["source_type"] = self.source_type.value
        d["evidence_tier"] = self.evidence_tier.value
        return d


def make_chunk_id(source_url: str, chunk_index: int, text: str) -> str:
    """Deterministic ID for a chunk: 16-char sha256 prefix of its provenance."""
    h = sha256()
    h.update(source_url.encode("utf-8"))
    h.update(b"\x00")
    h.update(str(chunk_index).encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()[:16]


def now_iso() -> str:
    """Current UTC timestamp in ISO-8601 form, suitable for ``retrieved_at``."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class BaseIngester(ABC):
    """Produce a stream of ``Chunk`` objects from a source.

    Subclasses implement ``iter_chunks``. The default ``ingest`` method
    materialises them into a list — fine for the sepsis-scale corpora
    we're working with. For larger corpora the streaming form keeps
    memory bounded.
    """

    source_name: str
    default_source_type: SourceType
    default_evidence_tier: EvidenceTier

    @abstractmethod
    def iter_chunks(self) -> Iterator[Chunk]:
        """Yield ``Chunk`` objects one at a time."""
        ...

    def ingest(self) -> list[Chunk]:
        """Materialise all chunks into a list."""
        return list(self.iter_chunks())


def chunks_to_payloads(chunks: Iterable[Chunk]) -> tuple[list[str], list[dict[str, Any]]]:
    """Split an iterable of chunks into parallel (ids, payloads) lists.

    Convenience for the indexing path: the vector store needs IDs and
    payloads in parallel arrays alongside the embedded vectors.
    """
    ids: list[str] = []
    payloads: list[dict[str, Any]] = []
    for c in chunks:
        ids.append(c.chunk_id)
        payloads.append(c.to_payload())
    return ids, payloads
