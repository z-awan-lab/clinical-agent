"""Build the vector index for guideline retrieval.

Phase 2 entry point. Wires a chunker + embedder + Qdrant store and
upserts a small example corpus. Source-specific ingesters (CDC, NIH,
WHO, NICE, PubMed bulk) land in the next phase; this script proves the
pipeline end to end and serves as the template they'll plug into.

Usage:
    python scripts/build_index.py --in-memory      # quick smoke test
    python scripts/build_index.py --qdrant-url http://localhost:6333
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from clinical_agent.embeddings import BGEEmbedder
from clinical_agent.ingestion import (
    EvidenceTier,
    RawDocument,
    RecursiveCharacterChunker,
    SimpleTextIngester,
    SourceType,
    chunks_to_payloads,
)
from clinical_agent.utils import setup_logging
from clinical_agent.vectorstore import QdrantStore

# Tiny in-repo example corpus. Real ingesters will replace this in the
# next session. Three documents, three distinct evidence tiers, so the
# filtering paths get exercised end to end.
EXAMPLE_DOCS: list[RawDocument] = [
    RawDocument(
        text=(
            "Sepsis is a life-threatening organ dysfunction caused by a "
            "dysregulated host response to infection. Septic shock is a "
            "subset of sepsis with circulatory and metabolic abnormalities "
            "that substantially increase mortality. Early recognition "
            "and prompt administration of broad-spectrum antibiotics "
            "within one hour of recognition is associated with improved "
            "outcomes."
        ),
        source_url="https://example.org/sepsis-guideline",
        title="Sepsis: recognition and early management (example)",
        section="Introduction",
        source_type=SourceType.GUIDELINE,
        evidence_tier=EvidenceTier.GUIDELINE_NATIONAL,
    ),
    RawDocument(
        text=(
            "qSOFA is a bedside score using respiratory rate >= 22, "
            "altered mentation, and systolic blood pressure <= 100 mmHg. "
            "A qSOFA of 2 or more identifies patients with suspected "
            "infection at greater risk of poor outcome. qSOFA is a "
            "screening tool, not a diagnostic criterion for sepsis."
        ),
        source_url="https://example.org/qsofa-review",
        title="qSOFA in clinical practice (example)",
        section="Scoring",
        source_type=SourceType.RESEARCH_ABSTRACT,
        evidence_tier=EvidenceTier.SYSTEMATIC_REVIEW,
    ),
    RawDocument(
        text=(
            "If you think you or someone you love has sepsis, get medical "
            "help right away. Sepsis is a medical emergency. Symptoms can "
            "include fever, chills, fast heart rate, confusion, and "
            "shortness of breath."
        ),
        source_url="https://example.org/sepsis-patient-info",
        title="Sepsis: what to know (example consumer page)",
        source_type=SourceType.CONSUMER_HEALTH,
        evidence_tier=EvidenceTier.CONSUMER_HEALTH,
    ),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection", default="clinical_agent_sepsis", help="Qdrant collection name"
    )
    parser.add_argument("--model", default="BAAI/bge-large-en-v1.5", help="embedding model")
    parser.add_argument(
        "--in-memory",
        action="store_true",
        help="use Qdrant in-memory mode (no server required)",
    )
    parser.add_argument(
        "--qdrant-url",
        default="http://localhost:6333",
        help="Qdrant server URL (ignored when --in-memory)",
    )
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--chunk-overlap", type=int, default=256)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(level="DEBUG" if args.verbose else "INFO")

    # 1) Chunk the example corpus.
    chunker = RecursiveCharacterChunker(
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )
    ingester = SimpleTextIngester(documents=EXAMPLE_DOCS, chunker=chunker)
    chunks = ingester.ingest()
    if not chunks:
        print("no chunks produced — nothing to index", file=sys.stderr)
        return 1
    print(f"chunked {len(EXAMPLE_DOCS)} documents into {len(chunks)} chunks")

    # 2) Embed.
    embedder = BGEEmbedder(model_name=args.model)
    vectors = embedder.embed([c.text for c in chunks])
    print(f"embedded {len(vectors)} chunks (dim={embedder.embedding_dim})")
    assert isinstance(vectors, np.ndarray)

    # 3) Upsert into Qdrant.
    store = QdrantStore(
        collection=args.collection,
        embedding_dim=embedder.embedding_dim,
        url=None if args.in_memory else args.qdrant_url,
        location=":memory:" if args.in_memory else None,
    )
    ids, payloads = chunks_to_payloads(chunks)
    store.upsert(chunk_ids=ids, vectors=vectors, payloads=payloads)
    total = store.count()
    print(f"indexed {total} chunks into collection '{args.collection}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
