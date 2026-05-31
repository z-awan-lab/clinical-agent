"""End-to-end orchestrator tests with a scripted generator.

The ``ScriptedGenerator`` returns canned model outputs in sequence. This
lets us drive the full state machine through any trajectory shape
without needing a GPU, while still exercising real LangGraph wiring,
the real parser, and real tool execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pytest

from clinical_agent.calculators import REGISTRY as CALC_REGISTRY
from clinical_agent.generation.base import BaseGenerator, GenerationResult
from clinical_agent.orchestrator import ClinicalAgent, ToolRegistry
from clinical_agent.tools import ClinicalCalculatorTool
from clinical_agent.tools.base import BaseTool, ToolResult


@dataclass
class ScriptedGenerator(BaseGenerator):
    """Returns pre-scripted responses in order. Raises if exhausted."""

    responses: list[str]
    calls: int = 0
    model_name: ClassVar[str] = "scripted"

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> GenerationResult:
        if self.calls >= len(self.responses):
            raise RuntimeError(
                f"ScriptedGenerator exhausted: {self.calls + 1} calls but "
                f"only {len(self.responses)} scripted responses"
            )
        text = self.responses[self.calls]
        self.calls += 1
        return GenerationResult(
            text=text,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
            finish_reason="stop",
        )


class EchoTool(BaseTool):
    """Test tool: returns whatever args you pass it."""

    name: ClassVar[str] = "echo"
    description: ClassVar[str] = "Echo the input arguments back."
    input_schema: ClassVar[dict] = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def run(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, data={"echoed": kwargs.get("text", "")})


# ---- happy paths ------------------------------------------------------


class TestSingleToolThenAnswer:
    def test_one_tool_call_then_answer(self) -> None:
        gen = ScriptedGenerator(
            responses=[
                '<tool_call name="echo">{"text": "hello"}</tool_call>',
                "<final_answer>The echo said hello.<citations>none</citations></final_answer>",
            ]
        )
        agent = ClinicalAgent(generator=gen, tools=ToolRegistry([EchoTool()]))
        final = agent.run("test question")
        assert final.finished
        assert not final.refused
        assert final.final_answer == "The echo said hello."
        assert final.termination_reason == "final_answer"
        # Trajectory: plan, tool_result, plan.
        kinds = [s.kind for s in final.trajectory]
        assert kinds == ["plan", "tool_result", "plan"]

    def test_immediate_answer_no_tool(self) -> None:
        gen = ScriptedGenerator(
            responses=["<final_answer>Sepsis is a medical emergency.</final_answer>"]
        )
        agent = ClinicalAgent(generator=gen, tools=ToolRegistry([EchoTool()]))
        final = agent.run("what is sepsis?")
        assert final.finished
        assert final.final_answer == "Sepsis is a medical emergency."
        assert len(final.trajectory) == 1


class TestRefusal:
    def test_explicit_refuse(self) -> None:
        gen = ScriptedGenerator(responses=['<refuse reason="outside scope of sepsis tooling"/>'])
        agent = ClinicalAgent(generator=gen, tools=ToolRegistry([EchoTool()]))
        final = agent.run("recommend a wine pairing")
        assert final.finished
        assert final.refused
        assert final.refuse_reason == "outside scope of sepsis tooling"
        assert final.termination_reason == "model_refused"


# ---- safety guards ---------------------------------------------------


class TestStepBudget:
    def test_budget_exceeded_forces_refuse(self) -> None:
        # Endless tool calls. Budget of 3 should cut after 3 plans —
        # but the loop guard catches the identical-args case first, so
        # vary the args.
        gen = ScriptedGenerator(
            responses=[
                '<tool_call name="echo">{"text": "1"}</tool_call>',
                '<tool_call name="echo">{"text": "2"}</tool_call>',
                '<tool_call name="echo">{"text": "3"}</tool_call>',
                '<tool_call name="echo">{"text": "4"}</tool_call>',
                '<tool_call name="echo">{"text": "5"}</tool_call>',
                '<tool_call name="echo">{"text": "6"}</tool_call>',
                '<tool_call name="echo">{"text": "7"}</tool_call>',
            ]
        )
        agent = ClinicalAgent(generator=gen, tools=ToolRegistry([EchoTool()]), max_steps=3)
        final = agent.run("loop forever")
        assert final.finished
        assert final.refused
        assert final.termination_reason == "step_budget_exceeded"


class TestLoopGuard:
    def test_identical_call_twice_triggers_loop_guard(self) -> None:
        gen = ScriptedGenerator(
            responses=[
                '<tool_call name="echo">{"text": "same"}</tool_call>',
                '<tool_call name="echo">{"text": "same"}</tool_call>',
            ]
        )
        agent = ClinicalAgent(generator=gen, tools=ToolRegistry([EchoTool()]))
        final = agent.run("looping question")
        assert final.finished
        assert final.refused
        assert final.termination_reason == "loop_guard"

    def test_same_tool_different_args_is_fine(self) -> None:
        gen = ScriptedGenerator(
            responses=[
                '<tool_call name="echo">{"text": "first"}</tool_call>',
                '<tool_call name="echo">{"text": "second"}</tool_call>',
                "<final_answer>Done.</final_answer>",
            ]
        )
        agent = ClinicalAgent(generator=gen, tools=ToolRegistry([EchoTool()]))
        final = agent.run("varied call")
        assert final.finished and not final.refused


# ---- parse error handling --------------------------------------------


class TestParseRetry:
    def test_one_retry_then_success(self) -> None:
        gen = ScriptedGenerator(
            responses=[
                "Just prose, no action block.",  # parse fails
                "<final_answer>Recovered.</final_answer>",  # retry succeeds
            ]
        )
        agent = ClinicalAgent(generator=gen, tools=ToolRegistry([EchoTool()]))
        final = agent.run("test")
        assert final.finished and not final.refused
        assert final.final_answer == "Recovered."
        # Trajectory should have one parse_error and one successful plan.
        kinds = [s.kind for s in final.trajectory]
        assert "parse_error" in kinds
        assert kinds[-1] == "plan"

    def test_parse_fails_twice_hard_fails_to_refuse(self) -> None:
        gen = ScriptedGenerator(
            responses=[
                "First malformed response.",
                "Second malformed response.",
            ]
        )
        agent = ClinicalAgent(generator=gen, tools=ToolRegistry([EchoTool()]))
        final = agent.run("test")
        assert final.finished
        assert final.refused
        assert final.termination_reason == "parse_failed_after_retry"


# ---- unknown tool ----------------------------------------------------


class TestUnknownTool:
    def test_unknown_tool_is_feedback_not_crash(self) -> None:
        gen = ScriptedGenerator(
            responses=[
                '<tool_call name="nonexistent">{}</tool_call>',
                "<final_answer>Recovered.</final_answer>",
            ]
        )
        agent = ClinicalAgent(generator=gen, tools=ToolRegistry([EchoTool()]))
        final = agent.run("test")
        assert final.finished and not final.refused
        # The trajectory should record the failed tool result.
        tool_results = [s for s in final.trajectory if s.kind == "tool_result"]
        assert len(tool_results) == 1
        assert tool_results[0].tool_success is False
        assert "unknown tool" in (tool_results[0].tool_error or "").lower()


# ---- realistic clinical trajectory -----------------------------------


class TestClinicalTrajectory:
    """An end-to-end qSOFA trajectory using the real calculator tool."""

    def test_full_qsofa_trajectory(self) -> None:
        gen = ScriptedGenerator(
            responses=[
                "<thinking>The patient has tachypnoea, hypotension, and altered "
                "mentation — qSOFA is the appropriate bedside screen.</thinking>"
                '<tool_call name="clinical_calculator">'
                '{"score": "qsofa", "inputs": {"respiratory_rate": 24, '
                '"systolic_bp": 90, "gcs": 13}}'
                "</tool_call>",
                "<final_answer>qSOFA score is 3/3, indicating higher risk of "
                "poor outcome in suspected infection. Full SOFA assessment is "
                "warranted.<citations>none</citations></final_answer>",
            ]
        )
        agent = ClinicalAgent(generator=gen, tools=ToolRegistry([ClinicalCalculatorTool()]))
        final = agent.run(
            "Patient with suspected pneumonia: RR 24, BP 90/60, GCS 13. Sepsis screening?"
        )
        assert final.finished and not final.refused
        assert "qSOFA" in (final.final_answer or "")
        tool_steps = [s for s in final.trajectory if s.kind == "tool_result"]
        assert len(tool_steps) == 1
        assert tool_steps[0].tool_data["score"] == 3
        # Thinking carried through to the trajectory.
        plan_steps = [s for s in final.trajectory if s.kind == "plan"]
        assert plan_steps[0].parsed_action.thinking


# ---- registry --------------------------------------------------------


class TestToolRegistry:
    def test_duplicate_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            ToolRegistry([EchoTool(), EchoTool()])

    def test_unknown_lookup_raises(self) -> None:
        reg = ToolRegistry([EchoTool()])
        with pytest.raises(KeyError, match="unknown tool"):
            _ = reg["nothing"]


# ---- serialisation ---------------------------------------------------


class TestStateSerialisation:
    def test_to_dict_contains_trajectory(self) -> None:
        gen = ScriptedGenerator(
            responses=[
                '<tool_call name="echo">{"text": "hi"}</tool_call>',
                "<final_answer>done</final_answer>",
            ]
        )
        agent = ClinicalAgent(generator=gen, tools=ToolRegistry([EchoTool()]))
        final = agent.run("test")
        d = final.to_dict()
        assert d["finished"] is True
        assert len(d["trajectory"]) == 3
        # Each step should have at least step_index and kind.
        for s in d["trajectory"]:
            assert "step_index" in s
            assert "kind" in s


# ---- ensure all five calculators are dispatchable via the orchestrator
# (sanity that Phase 3 + Phase 4 wire together)


def test_calculator_registry_visible_in_prompt() -> None:
    """Every Phase 3 score is exposed via the calculator tool."""
    tool = ClinicalCalculatorTool()
    enum = tool.input_schema["properties"]["score"]["enum"]
    assert set(enum) == set(CALC_REGISTRY.keys())
