"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def esearch_xml() -> bytes:
    return (FIXTURES_DIR / "pubmed_esearch.xml").read_bytes()


@pytest.fixture
def efetch_xml() -> bytes:
    return (FIXTURES_DIR / "pubmed_efetch.xml").read_bytes()


@pytest.fixture
def esearch_empty_xml() -> bytes:
    return (FIXTURES_DIR / "pubmed_esearch_empty.xml").read_bytes()
