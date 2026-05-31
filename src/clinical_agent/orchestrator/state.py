"""Agent state passed through the orchestrator graph.

Every node in the state machine reads and returns an updated copy of
this state. Keeping it a plain typed dataclass (rather than the
TypedDict pattern LangGraph examples often use) gives us proper
attribute access, mypy support, and easy serialisation for the
trajectory pane.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .protocol import ParsedAction


@dataclass
class TrajectoryStep:
    """A single step in the agent's trajectory.

    Steps are append-only; we never mutate a prior step. The trajectory
    pane renders the list in order.
    """

    step_index: int
    kind: str  # "plan" | "tool_result" | "answer" | "refuse" | "parse_error"
    raw_model_output: str | None = None
    parsed_action: ParsedAction | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    tool_success: bool | None = None
    tool_data: dict[str, Any] = field(default_factory=dict)
    tool_error: str | None = None
    parse_error: str | None = None
    elapsed_ms: float | None = None


@dataclass
class AgentState:
    """State threaded through the orchestrator graph."""

    question: str
    max_steps: int = 8
    # Conversation-like message history sent to the model on each turn.
    # We keep it as a list of (role, content) tuples — generic across
    # generators that may or may not have a native chat format.
    messages: list[tuple[str, str]] = field(default_factory=list)
    trajectory: list[TrajectoryStep] = field(default_factory=list)
    # The single next action to execute, set by the plan node and
    # consumed by the dispatch node.
    pending_action: ParsedAction | None = None
    # When a parse fails we carry the error so the next plan node can
    # decide whether to retry with a corrective prompt.
    last_parse_error: str | None = None
    retry_pending: bool = False
    # Terminal state.
    finished: bool = False
    final_answer: str | None = None
    final_citations: list[str] = field(default_factory=list)
    refused: bool = False
    refuse_reason: str | None = None
    # Diagnostic.
    termination_reason: str | None = None

    @property
    def step_count(self) -> int:
        return len(self.trajectory)

    def append(self, step: TrajectoryStep) -> None:
        self.trajectory.append(step)

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the trajectory pane / JSON dumps."""
        return {
            "question": self.question,
            "max_steps": self.max_steps,
            "finished": self.finished,
            "refused": self.refused,
            "refuse_reason": self.refuse_reason,
            "final_answer": self.final_answer,
            "final_citations": self.final_citations,
            "termination_reason": self.termination_reason,
            "trajectory": [
                {
                    "step_index": s.step_index,
                    "kind": s.kind,
                    "tool_name": s.tool_name,
                    "tool_args": s.tool_args,
                    "tool_success": s.tool_success,
                    "tool_data": s.tool_data,
                    "tool_error": s.tool_error,
                    "parse_error": s.parse_error,
                    "thinking": (s.parsed_action.thinking if s.parsed_action else []),
                    "elapsed_ms": s.elapsed_ms,
                }
                for s in self.trajectory
            ],
        }
