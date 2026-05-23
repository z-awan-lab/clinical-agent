"""Tests for the PubMed search tool.

Unit tests mock the HTTP layer with pytest's monkeypatch so CI runs
offline. A single integration test hits the real NCBI API and is
skipped unless ``CLINICAL_AGENT_INTEGRATION_TESTS=1``.
"""

from __future__ import annotations

import os
import time
from typing import Any
from unittest.mock import MagicMock

import pytest
import requests

from clinical_agent.tools.pubmed_search import (
    PubMedSearchTool,
    parse_efetch_xml,
    parse_esearch_xml,
)

# ----- XML parsing ------------------------------------------------------


class TestParseEsearch:
    def test_extracts_pmids(self, esearch_xml: bytes) -> None:
        pmids = parse_esearch_xml(esearch_xml)
        assert pmids == ["38000001", "38000002", "38000003"]

    def test_empty_result_returns_empty_list(self, esearch_empty_xml: bytes) -> None:
        assert parse_esearch_xml(esearch_empty_xml) == []


class TestParseEfetch:
    def test_returns_one_article_per_pubmed_article(self, efetch_xml: bytes) -> None:
        articles = parse_efetch_xml(efetch_xml)
        assert len(articles) == 3

    def test_basic_fields(self, efetch_xml: bytes) -> None:
        articles = parse_efetch_xml(efetch_xml)
        first = articles[0]
        assert first.pmid == "38000001"
        assert first.title.startswith("Early identification of sepsis")
        assert first.journal == "Intensive care medicine"
        assert first.year == "2023"
        assert first.doi == "10.1007/s00134-023-00001-x"

    def test_structured_abstract_concatenated_with_labels(self, efetch_xml: bytes) -> None:
        first = parse_efetch_xml(efetch_xml)[0]
        assert "BACKGROUND:" in first.abstract
        assert "METHODS:" in first.abstract
        assert "RESULTS:" in first.abstract
        assert "CONCLUSIONS:" in first.abstract

    def test_unstructured_abstract_has_no_label_prefix(self, efetch_xml: bytes) -> None:
        second = parse_efetch_xml(efetch_xml)[1]
        assert second.abstract.startswith("Lactate clearance")
        assert ":" not in second.abstract.split(".")[0]  # no label before first sentence

    def test_authors_with_last_and_initials(self, efetch_xml: bytes) -> None:
        first = parse_efetch_xml(efetch_xml)[0]
        assert first.authors == ["Smith J", "Johnson AB"]

    def test_collective_author_handled(self, efetch_xml: bytes) -> None:
        third = parse_efetch_xml(efetch_xml)[2]
        assert third.authors == ["Surviving Sepsis Campaign Working Group"]

    def test_mesh_terms_extracted(self, efetch_xml: bytes) -> None:
        first = parse_efetch_xml(efetch_xml)[0]
        assert "Sepsis" in first.mesh_terms

    def test_missing_optional_fields_dont_crash(self, efetch_xml: bytes) -> None:
        # Third article has no abstract, no MeSH, no DOI, no year.
        third = parse_efetch_xml(efetch_xml)[2]
        assert third.abstract == ""
        assert third.mesh_terms == []
        assert third.doi is None
        assert third.year is None


# ----- Tool.run() with mocked HTTP --------------------------------------


def _mock_response(content: bytes, status: int = 200) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.content = content
    resp.status_code = status
    if status >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(f"{status} error")
    else:
        resp.raise_for_status.return_value = None
    return resp


class TestRun:
    def test_happy_path_returns_articles(self, esearch_xml: bytes, efetch_xml: bytes) -> None:
        session = MagicMock(spec=requests.Session)
        session.get.side_effect = [
            _mock_response(esearch_xml),
            _mock_response(efetch_xml),
        ]
        tool = PubMedSearchTool(session=session)
        result = tool.run(query="sepsis", max_results=3)

        assert result.success is True
        assert result.error is None
        assert len(result.data["articles"]) == 3
        assert result.data["query"] == "sepsis"
        assert result.metadata["n_results"] == 3
        assert result.metadata["pmids_returned"] == [
            "38000001",
            "38000002",
            "38000003",
        ]

    def test_empty_search_returns_success_with_no_articles(self, esearch_empty_xml: bytes) -> None:
        session = MagicMock(spec=requests.Session)
        session.get.side_effect = [_mock_response(esearch_empty_xml)]
        tool = PubMedSearchTool(session=session)
        result = tool.run(query="qwertyuiopasdfgh_no_hits", max_results=5)

        assert result.success is True
        assert result.data["articles"] == []
        assert result.metadata["n_results"] == 0
        # eFetch is not called when there are no PMIDs.
        assert session.get.call_count == 1

    def test_missing_query_returns_error(self) -> None:
        tool = PubMedSearchTool(session=MagicMock(spec=requests.Session))
        result = tool.run(max_results=5)
        assert result.success is False
        assert "query" in (result.error or "").lower()

    def test_non_string_query_rejected(self) -> None:
        tool = PubMedSearchTool(session=MagicMock(spec=requests.Session))
        result = tool.run(query=12345)
        assert result.success is False

    def test_esearch_http_error_caught(self) -> None:
        session = MagicMock(spec=requests.Session)
        # All retries fail with the same error — tenacity reraises after attempts.
        session.get.side_effect = requests.ConnectionError("dns failure")
        tool = PubMedSearchTool(session=session)
        result = tool.run(query="sepsis")
        assert result.success is False
        assert "eSearch failed" in (result.error or "")

    def test_max_results_clamped(self, esearch_xml: bytes, efetch_xml: bytes) -> None:
        session = MagicMock(spec=requests.Session)
        session.get.side_effect = [
            _mock_response(esearch_xml),
            _mock_response(efetch_xml),
        ]
        tool = PubMedSearchTool(session=session)
        tool.run(query="sepsis", max_results=9999)

        # retmax sent to PubMed should be clamped to 20.
        esearch_params = session.get.call_args_list[0].kwargs["params"]
        assert esearch_params["retmax"] == "20"

    def test_api_key_passed_in_query_params(self, esearch_xml: bytes, efetch_xml: bytes) -> None:
        session = MagicMock(spec=requests.Session)
        session.get.side_effect = [
            _mock_response(esearch_xml),
            _mock_response(efetch_xml),
        ]
        tool = PubMedSearchTool(api_key="fake-key-abc123", session=session)
        result = tool.run(query="sepsis")

        assert result.success is True
        for call in session.get.call_args_list:
            assert call.kwargs["params"].get("api_key") == "fake-key-abc123"
        assert result.metadata["api_key_used"] is True

    def test_no_api_key_means_slower_throttle(self) -> None:
        tool_no_key = PubMedSearchTool(session=MagicMock())
        tool_with_key = PubMedSearchTool(api_key="k", session=MagicMock())
        assert tool_no_key._min_interval > tool_with_key._min_interval


# ----- Throttling -------------------------------------------------------


class TestThrottle:
    def test_throttle_enforces_min_interval(self, monkeypatch: Any) -> None:
        sleeps: list[float] = []
        monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

        # Force a fixed monotonic so the throttle math is deterministic.
        clock = [1000.0]
        monkeypatch.setattr(time, "monotonic", lambda: clock[0])

        tool = PubMedSearchTool(session=MagicMock())
        tool._throttle()  # first call: no wait
        # Pretend almost no time has passed.
        clock[0] = 1000.05
        tool._throttle()  # second call: should sleep close to _min_interval

        assert len(sleeps) == 1
        assert sleeps[0] > 0


# ----- Tool spec --------------------------------------------------------


class TestSpec:
    def test_spec_has_required_fields(self) -> None:
        tool = PubMedSearchTool(session=MagicMock())
        spec = tool.to_spec()
        assert spec["name"] == "pubmed_search"
        assert "description" in spec
        assert spec["input_schema"]["required"] == ["query"]


# ----- Live integration (skipped by default) ---------------------------


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("CLINICAL_AGENT_INTEGRATION_TESTS") != "1",
    reason="set CLINICAL_AGENT_INTEGRATION_TESTS=1 to run live API tests",
)
def test_live_pubmed_query_returns_results() -> None:
    """Sanity check against the real PubMed API. Off by default."""
    tool = PubMedSearchTool()
    result = tool.run(query="sepsis[MeSH] AND 2024[PDAT]", max_results=2)
    assert result.success is True
    assert result.metadata["n_results"] > 0
    first = result.data["articles"][0]
    assert first["pmid"]
    assert first["title"]
