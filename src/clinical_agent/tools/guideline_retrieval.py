"""Guideline retrieval tool.

Wraps an ``BaseEmbedder`` and a ``BaseVectorStore`` to give the agent
a single tool call that returns top-k retrieved chunks. Surfaces both
hard and soft filters:

* Hard filters (``source_type``, ``evidence_tier``) restrict candidates
  before scoring.
* Soft signals (``min_score``) drop low-confidence hits post-hoc.

The tool returns enough provenance (source URL, title, section, tier)
for the agent to cite specific chunks in its final answer.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from ..embeddings.base import BaseEmbedder
from ..vectorstore.base import BaseVectorStore
from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class GuidelineRetrievalTool(BaseTool):
    """Retrieve top-k chunks from the indexed clinical corpus."""

    name: ClassVar[str] = "guideline_retrieval"
    description: ClassVar[str] = (
        "Search indexed clinical guidelines and curated literature for "
        "passages relevant to a clinical question. Returns ranked chunks "
        "with source, evidence tier, and citation metadata. Prefer this "
        "tool over pubmed_search when the question concerns established "
        "clinical recommendations rather than primary research findings."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language clinical question.",
            },
            "k": {
                "type": "integer",
                "description": "Number of chunks to return (1-20).",
                "default": 5,
                "minimum": 1,
                "maximum": 20,
            },
            "source_type": {
                "type": "string",
                "description": (
                    "Restrict to one source type: guideline, consumer_health, or research_abstract."
                ),
            },
            "evidence_tier": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Restrict to one or more evidence tiers, "
                    "e.g. ['guideline_national', 'systematic_review']."
                ),
            },
            "min_score": {
                "type": "number",
                "description": "Drop hits with similarity below this threshold.",
            },
        },
        "required": ["query"],
    }

    def __init__(
        self,
        embedder: BaseEmbedder,
        store: BaseVectorStore,
    ) -> None:
        self.embedder = embedder
        self.store = store

    def run(self, **kwargs: Any) -> ToolResult:
        query = kwargs.get("query")
        if not isinstance(query, str) or not query.strip():
            return ToolResult(
                success=False, error="`query` is required and must be a non-empty string."
            )

        k = int(kwargs.get("k", 5))
        k = max(1, min(k, 20))
        min_score = kwargs.get("min_score")

        # Build the hard filter from optional args.
        filter_: dict[str, Any] = {}
        if (st := kwargs.get("source_type")) is not None:
            filter_["source_type"] = st
        if (et := kwargs.get("evidence_tier")) is not None:
            filter_["evidence_tier"] = et

        # BGE-style embedders expose embed_query with the retrieval
        # prefix; fall back to plain embed for other backends.
        if hasattr(self.embedder, "embed_query"):
            query_vec = self.embedder.embed_query(query)
        else:
            query_vec = self.embedder.embed_one(query)

        try:
            hits = self.store.search(
                query_vector=query_vec,
                k=k,
                filter_=filter_ or None,
            )
        except Exception as exc:  # surface as tool failure, not a crash
            logger.exception("vector store search failed")
            return ToolResult(success=False, error=f"retrieval failed: {exc}")

        if min_score is not None:
            hits = [h for h in hits if h.score >= float(min_score)]

        return ToolResult(
            success=True,
            data={
                "query": query,
                "results": [
                    {
                        "chunk_id": h.chunk_id,
                        "score": h.score,
                        "text": h.payload.get("text", ""),
                        "title": h.payload.get("title"),
                        "section": h.payload.get("section"),
                        "source_url": h.payload.get("source_url"),
                        "source_type": h.payload.get("source_type"),
                        "evidence_tier": h.payload.get("evidence_tier"),
                    }
                    for h in hits
                ],
            },
            metadata={
                "n_results": len(hits),
                "filter_applied": filter_ or None,
                "min_score": min_score,
            },
        )
