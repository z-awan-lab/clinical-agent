"""Smoke test — the package and its public exports import cleanly."""

from __future__ import annotations


def test_package_imports() -> None:
    import clinical_agent

    assert clinical_agent.__version__


def test_tools_exports() -> None:
    from clinical_agent.tools import BaseTool, PubMedSearchTool, ToolResult

    assert issubclass(PubMedSearchTool, BaseTool)
    assert ToolResult(success=True).success is True


def test_base_classes_present() -> None:
    from clinical_agent.embeddings import BaseEmbedder
    from clinical_agent.generation import BaseGenerator
    from clinical_agent.vectorstore import BaseVectorStore

    assert BaseEmbedder is not None
    assert BaseVectorStore is not None
    assert BaseGenerator is not None
