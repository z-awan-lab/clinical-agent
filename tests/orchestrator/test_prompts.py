"""Tests for system prompt assembly."""

from __future__ import annotations

from clinical_agent.orchestrator.prompts import build_system_prompt, render_tool_specs
from clinical_agent.tools import ClinicalCalculatorTool, PubMedSearchTool


def test_tool_specs_render_includes_each_tool() -> None:
    out = render_tool_specs([PubMedSearchTool(), ClinicalCalculatorTool()])
    assert "pubmed_search" in out
    assert "clinical_calculator" in out
    # Alphabetical ordering so prompts are deterministic across runs.
    assert out.index("clinical_calculator") < out.index("pubmed_search")


def test_system_prompt_contains_key_sections() -> None:
    prompt = build_system_prompt([PubMedSearchTool()], max_steps=8)
    assert "pubmed_search" in prompt
    assert "<tool_call" in prompt
    assert "<final_answer>" in prompt
    assert "<refuse" in prompt
    assert "Refusal" in prompt
    assert "evidence tier" in prompt.lower()
    assert "8" in prompt  # max_steps surfaced


def test_calculator_distractor_guidance_present() -> None:
    prompt = build_system_prompt([ClinicalCalculatorTool()], max_steps=8)
    # The distractor guidance is what makes Phase 5 tool-selection
    # honesty meaningful — verify it's actually in the prompt.
    assert "CHA2DS2-VASc" in prompt
    assert "Wells DVT" in prompt
