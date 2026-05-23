"""PubMed search tool backed by the NCBI E-utilities API.

Two-stage query: ``eSearch`` returns PMIDs matching the query; ``eFetch``
returns the full records for those PMIDs as XML, which we parse into a
flat dict per article.

References:
    https://www.ncbi.nlm.nih.gov/books/NBK25501/

Set the ``NCBI_API_KEY`` environment variable for the 10 req/sec rate
limit (vs 3 req/sec without). Get a key at
https://www.ncbi.nlm.nih.gov/account/.
"""

from __future__ import annotations

import logging
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, ClassVar

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# NCBI rate limits: 3 req/sec without API key, 10 req/sec with. We stay well
# under both so concurrent tool calls (Phase 4) don't trip the limiter.
_MIN_INTERVAL_NO_KEY = 0.34
_MIN_INTERVAL_WITH_KEY = 0.11


@dataclass
class PubMedArticle:
    """A single PubMed record, flattened for downstream use."""

    pmid: str
    title: str
    abstract: str
    authors: list[str]
    journal: str
    year: str | None
    mesh_terms: list[str]
    doi: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pmid": self.pmid,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "journal": self.journal,
            "year": self.year,
            "mesh_terms": self.mesh_terms,
            "doi": self.doi,
        }


class PubMedSearchTool(BaseTool):
    """Search PubMed and return structured article records.

    Designed to be called by the agent orchestrator. The ``query`` argument
    is passed verbatim to PubMed (Entrez supports its own query syntax,
    including MeSH terms and field tags — see the NCBI docs).
    """

    name: ClassVar[str] = "pubmed_search"
    description: ClassVar[str] = (
        "Search PubMed for biomedical literature. Returns the most relevant "
        "abstracts for a query. Use this when the question requires evidence "
        "from primary research rather than clinical guidelines. Query may use "
        "PubMed syntax (e.g. 'sepsis[MeSH] AND mortality')."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "PubMed search query.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of articles to return (1-20).",
                "default": 5,
                "minimum": 1,
                "maximum": 20,
            },
        },
        "required": ["query"],
    }

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("NCBI_API_KEY")
        self.timeout = timeout
        self.session = session or requests.Session()
        self._last_request_at = 0.0
        self._min_interval = _MIN_INTERVAL_WITH_KEY if self.api_key else _MIN_INTERVAL_NO_KEY

    # ---- public ----------------------------------------------------------

    def run(self, **kwargs: Any) -> ToolResult:
        query = kwargs.get("query")
        if not query or not isinstance(query, str):
            return ToolResult(success=False, error="`query` is required and must be a string.")
        max_results = int(kwargs.get("max_results", 5))
        max_results = max(1, min(max_results, 20))

        try:
            pmids = self._esearch(query, max_results)
        except requests.RequestException as exc:
            return ToolResult(success=False, error=f"eSearch failed: {exc}")

        if not pmids:
            return ToolResult(
                success=True,
                data={"articles": [], "query": query},
                metadata={"n_results": 0},
            )

        try:
            articles = self._efetch(pmids)
        except requests.RequestException as exc:
            return ToolResult(success=False, error=f"eFetch failed: {exc}")

        return ToolResult(
            success=True,
            data={
                "articles": [a.to_dict() for a in articles],
                "query": query,
            },
            metadata={
                "n_results": len(articles),
                "pmids_returned": [a.pmid for a in articles],
                "api_key_used": self.api_key is not None,
            },
        )

    # ---- E-utilities calls ----------------------------------------------

    @retry(
        retry=retry_if_exception_type(requests.RequestException),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _esearch(self, query: str, max_results: int) -> list[str]:
        self._throttle()
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": str(max_results),
            "retmode": "xml",
            "sort": "relevance",
        }
        if self.api_key:
            params["api_key"] = self.api_key
        resp = self.session.get(ESEARCH_URL, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return parse_esearch_xml(resp.content)

    @retry(
        retry=retry_if_exception_type(requests.RequestException),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _efetch(self, pmids: list[str]) -> list[PubMedArticle]:
        self._throttle()
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        }
        if self.api_key:
            params["api_key"] = self.api_key
        resp = self.session.get(EFETCH_URL, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return parse_efetch_xml(resp.content)

    # ---- helpers ---------------------------------------------------------

    def _throttle(self) -> None:
        """Block as needed to respect NCBI's rate limit."""
        now = time.monotonic()
        wait = self._min_interval - (now - self._last_request_at)
        if wait > 0:
            time.sleep(wait)
        self._last_request_at = time.monotonic()


# ---- XML parsing (kept module-level for testability) --------------------


def parse_esearch_xml(xml_bytes: bytes) -> list[str]:
    """Extract PMIDs from an eSearch response."""
    root = ET.fromstring(xml_bytes)
    return [el.text for el in root.findall(".//IdList/Id") if el.text]


def parse_efetch_xml(xml_bytes: bytes) -> list[PubMedArticle]:
    """Parse an eFetch response into ``PubMedArticle`` objects.

    PubMed XML is verbose and the schema has many optional fields. We
    defensively use ``findtext`` and silently skip missing elements rather
    than failing on edge-case records.
    """
    root = ET.fromstring(xml_bytes)
    articles: list[PubMedArticle] = []
    for art in root.findall(".//PubmedArticle"):
        pmid = art.findtext(".//MedlineCitation/PMID") or ""
        title = (art.findtext(".//Article/ArticleTitle") or "").strip()

        # Abstracts may have multiple <AbstractText> elements (structured
        # abstracts: BACKGROUND, METHODS, RESULTS, CONCLUSIONS). Join them
        # with section labels when present.
        abstract_parts: list[str] = []
        for at in art.findall(".//Article/Abstract/AbstractText"):
            label = at.get("Label")
            text = (at.text or "").strip()
            abstract_parts.append(f"{label}: {text}" if label else text)
        abstract = " ".join(p for p in abstract_parts if p)

        authors: list[str] = []
        for auth in art.findall(".//Article/AuthorList/Author"):
            last = auth.findtext("LastName") or ""
            initials = auth.findtext("Initials") or ""
            collective = auth.findtext("CollectiveName")
            if collective:
                authors.append(collective)
            elif last:
                authors.append(f"{last} {initials}".strip())

        journal = (art.findtext(".//Article/Journal/Title") or "").strip()
        year = art.findtext(".//Article/Journal/JournalIssue/PubDate/Year")

        mesh_terms = [
            (el.text or "").strip()
            for el in art.findall(".//MeshHeadingList/MeshHeading/DescriptorName")
            if el.text
        ]

        doi = None
        for aid in art.findall(".//ArticleId"):
            if aid.get("IdType") == "doi":
                doi = (aid.text or "").strip() or None

        articles.append(
            PubMedArticle(
                pmid=pmid,
                title=title,
                abstract=abstract,
                authors=authors,
                journal=journal,
                year=year,
                mesh_terms=mesh_terms,
                doi=doi,
            )
        )
    return articles
