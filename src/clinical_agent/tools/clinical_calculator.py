"""Clinical calculator tool.

Dispatches to a registry of deterministic clinical scores. The agent
calls this with a ``score`` name and the score's inputs; the tool
validates and computes, returning a structured ``ScoreResult``.

Design principle (design.md): the calculator returns the number; the
agent must surface that number rather than computing its own. The tool
description makes the available scores explicit so the agent can choose
correctly — and so the distractor scores (CHA2DS2-VASc, Wells DVT) are
visibly off-topic for sepsis questions.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..calculators import REGISTRY
from .base import BaseTool, ToolResult


class ClinicalCalculatorTool(BaseTool):
    name: ClassVar[str] = "clinical_calculator"
    description: ClassVar[str] = (
        "Compute a validated clinical score deterministically. Available "
        "scores: 'qsofa' (sepsis bedside screen), 'sofa' (organ dysfunction, "
        "all six systems required), 'sepsis3' (Sepsis-3 criteria check), "
        "'cha2ds2vasc' (AF stroke risk — not for sepsis), 'wells_dvt' (DVT "
        "probability — not for sepsis). Returns the score with a per-"
        "component breakdown. Use this rather than computing scores yourself; "
        "if required inputs are missing the tool reports exactly which."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "score": {
                "type": "string",
                "enum": sorted(REGISTRY.keys()),
                "description": "Which score to compute.",
            },
            "inputs": {
                "type": "object",
                "description": "Score-specific inputs (see the score's contract).",
            },
        },
        "required": ["score", "inputs"],
    }

    def run(self, **kwargs: Any) -> ToolResult:
        score_name = kwargs.get("score")
        if not isinstance(score_name, str) or score_name not in REGISTRY:
            return ToolResult(
                success=False,
                error=(
                    f"Unknown score '{score_name}'. "
                    f"Available: {', '.join(sorted(REGISTRY.keys()))}."
                ),
            )

        inputs = kwargs.get("inputs")
        if not isinstance(inputs, dict):
            return ToolResult(
                success=False,
                error="`inputs` must be an object of score-specific parameters.",
            )

        scorer = REGISTRY[score_name]
        try:
            result = scorer.score(**inputs)
        except (ValueError, TypeError) as exc:
            return ToolResult(
                success=False,
                error=f"invalid inputs for {score_name}: {exc}",
            )

        # A strict refusal (missing inputs) is a *successful* tool call
        # that returns ok=False — the agent needs to see what's missing,
        # not treat it as an error.
        return ToolResult(
            success=True,
            data=result.to_dict(),
            metadata={
                "score": score_name,
                "computed": result.ok,
                "missing": result.missing,
            },
        )
