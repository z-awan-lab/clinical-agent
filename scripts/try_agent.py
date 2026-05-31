"""Manual try-out for the orchestrator using a scripted generator.

Usage:
    python scripts/try_agent.py

Runs the agent on a worked sepsis trajectory without requiring a GPU,
to demonstrate the orchestrator mechanics end to end. Phase 4.5 will
swap in the real MedGemma generator on the HPC.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import ClassVar

from clinical_agent.generation.base import BaseGenerator, GenerationResult
from clinical_agent.orchestrator import ClinicalAgent, ToolRegistry
from clinical_agent.tools import ClinicalCalculatorTool
from clinical_agent.utils import setup_logging


@dataclass
class ScriptedGenerator(BaseGenerator):
    responses: list[str]
    calls: int = 0
    model_name: ClassVar[str] = "scripted-demo"

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> GenerationResult:
        text = self.responses[self.calls]
        self.calls += 1
        return GenerationResult(
            text=text,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
            finish_reason="stop",
        )


SCRIPT = [
    (
        "<thinking>Suspected pneumonia with tachypnoea, hypotension, and altered "
        "mentation. qSOFA is the appropriate bedside screen for sepsis "
        "risk.</thinking>"
        '<tool_call name="clinical_calculator">'
        '{"score": "qsofa", "inputs": {"respiratory_rate": 24, '
        '"systolic_bp": 92, "gcs": 13}}'
        "</tool_call>"
    ),
    (
        "<thinking>qSOFA is 3/3 — high risk. Recommend escalation and full SOFA "
        "assessment. The calculator output is the authoritative number."
        "</thinking>"
        "<final_answer>This patient has qSOFA 3/3 (RR ≥22, SBP ≤100, GCS <15), "
        "indicating substantially higher risk of poor outcome in suspected "
        "infection. Recommend full SOFA assessment, urgent sepsis workup "
        "(blood cultures, lactate), and broad-spectrum antibiotics within 1 "
        "hour per sepsis bundle recommendations. qSOFA is a screening tool, "
        "not diagnostic — clinical judgement and SOFA-based organ "
        "dysfunction assessment remain required for Sepsis-3 diagnosis."
        "<citations>none</citations></final_answer>"
    ),
]


def main() -> int:
    setup_logging("INFO")

    gen = ScriptedGenerator(responses=SCRIPT)
    agent = ClinicalAgent(
        generator=gen,
        tools=ToolRegistry([ClinicalCalculatorTool()]),
        max_steps=8,
    )

    question = (
        "68-year-old man with suspected community-acquired pneumonia. "
        "Vitals: RR 24, BP 92/58, HR 110, GCS 13, T 38.6 °C. "
        "Should I be worried about sepsis?"
    )
    print(f"=== Question ===\n{question}\n")

    state = agent.run(question)

    print("=== Trajectory ===")
    for step in state.trajectory:
        print(f"\n[step {step.step_index}] {step.kind}")
        if step.kind == "plan" and step.parsed_action:
            for thought in step.parsed_action.thinking:
                print(f"  thinking: {thought}")
            if step.parsed_action.tool_name:
                print(f"  → call {step.parsed_action.tool_name}({step.parsed_action.tool_args})")
            elif step.parsed_action.answer_text:
                preview = step.parsed_action.answer_text[:120]
                print(f"  → final_answer: {preview}...")
        if step.kind == "tool_result":
            print(f"  tool: {step.tool_name}")
            print(f"  success: {step.tool_success}")
            print(f"  data: {json.dumps(step.tool_data, indent=2)[:400]}")

    print("\n=== Termination ===")
    print(f"reason: {state.termination_reason}")
    print(f"finished: {state.finished}")
    print(f"refused: {state.refused}")
    if state.final_answer:
        print(f"\n=== Final answer ===\n{state.final_answer}")
        if state.final_citations:
            print(f"citations: {state.final_citations}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
