"""LangGraph orchestrator for the clinical agent.

Topology:

         ┌────────────────────────────────────────────┐
         │                                            │
    ┌────▼───┐    parsed OK     ┌──────────┐          │
    │  plan  │─────────────────▶│ dispatch │──────────┘ (back to plan)
    │ (LLM)  │                  └─────┬────┘
    └───┬─┬──┘                        │
        │ │                           ▼
        │ │                       (final answer / refuse) ─▶ end
        │ │
        │ └─ parse error, retry remaining ─▶ plan again (with corrective)
        │
        └─── parse error, retry exhausted ─▶ refuse ─▶ end

Termination conditions:
    * <final_answer> emitted → state.finished = True
    * <refuse/> emitted → state.refused = True
    * Step budget exceeded → forced refuse
    * Loop guard tripped (same tool + identical args twice running) →
      forced refuse
    * Parse error after one retry → forced refuse
"""

from __future__ import annotations

import logging
import time
from typing import Any

from langgraph.graph import END, StateGraph

from ..generation.base import BaseGenerator
from .prompts import build_system_prompt
from .protocol import (
    ActionKind,
    ParsedAction,
    ProtocolError,
    format_corrective_prompt,
    parse,
)
from .registry import ToolRegistry
from .state import AgentState, TrajectoryStep

logger = logging.getLogger(__name__)


# Node name constants — used both when wiring and when routing.
NODE_PLAN = "plan"
NODE_DISPATCH = "dispatch"
NODE_FINALISE = "finalise"


def _format_tool_result_message(step: TrajectoryStep) -> str:
    """Render a tool result as the user-turn message fed back to the model."""
    if step.tool_success:
        return (
            f'<tool_result name="{step.tool_name}" success="true">\n'
            f"{step.tool_data}\n"
            f"</tool_result>"
        )
    return (
        f'<tool_result name="{step.tool_name}" success="false">\n'
        f"error: {step.tool_error}\n"
        f"</tool_result>"
    )


class ClinicalAgent:
    """Builds and runs the orchestrator graph.

    The graph itself is constructed at ``__init__`` time so the same
    agent instance can be reused across many ``run`` calls. State is
    *not* shared between runs — each ``run`` constructs a fresh
    ``AgentState``.
    """

    def __init__(
        self,
        generator: BaseGenerator,
        tools: ToolRegistry,
        max_steps: int = 8,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> None:
        self.generator = generator
        self.tools = tools
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._graph = self._build_graph()

    # ---- public ------------------------------------------------------

    def run(self, question: str) -> AgentState:
        """Run the agent on a single question and return the final state."""
        system = build_system_prompt(self.tools.all(), self.max_steps)
        state = AgentState(
            question=question,
            max_steps=self.max_steps,
            messages=[("system", system), ("user", question)],
        )
        result = self._graph.invoke(state)
        # LangGraph returns the state as a dict-like; coerce back to our
        # dataclass type for downstream typing.
        if isinstance(result, AgentState):
            return result
        return AgentState(**result)

    # ---- graph wiring ------------------------------------------------

    def _build_graph(self) -> Any:
        g: StateGraph = StateGraph(AgentState)
        g.add_node(NODE_PLAN, self._plan)
        g.add_node(NODE_DISPATCH, self._dispatch)
        g.add_node(NODE_FINALISE, self._finalise)
        g.set_entry_point(NODE_PLAN)

        g.add_conditional_edges(
            NODE_PLAN,
            self._after_plan,
            {
                NODE_DISPATCH: NODE_DISPATCH,
                NODE_FINALISE: NODE_FINALISE,
                NODE_PLAN: NODE_PLAN,  # retry path
            },
        )
        g.add_conditional_edges(
            NODE_DISPATCH,
            self._after_dispatch,
            {
                NODE_PLAN: NODE_PLAN,
                NODE_FINALISE: NODE_FINALISE,
            },
        )
        g.add_edge(NODE_FINALISE, END)
        return g.compile()

    # ---- nodes -------------------------------------------------------

    def _plan(self, state: AgentState) -> AgentState:
        """Generate one model turn, parse it, populate ``pending_action``."""
        if state.step_count >= state.max_steps:
            state.termination_reason = "step_budget_exceeded"
            state.refused = True
            state.refuse_reason = (
                f"Step budget of {state.max_steps} exceeded without a final answer."
            )
            state.finished = True
            return state

        prompt = _flatten_messages(state.messages)
        t0 = time.monotonic()
        gen = self.generator.generate(
            prompt=prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            stop=["</tool_call>", "</final_answer>", "</refuse>"],
        )
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        # If we stopped on a closing tag the tag itself was consumed —
        # paste it back so the parser sees a complete block.
        raw = gen.text
        raw = _restore_stop_tag(raw)

        try:
            action = parse(raw)
        except ProtocolError as exc:
            # Decide retry vs hard-fail.
            if state.retry_pending:
                # Already retried once — hard fail to refusal.
                step = TrajectoryStep(
                    step_index=state.step_count,
                    kind="parse_error",
                    raw_model_output=raw,
                    parse_error=exc.reason,
                    elapsed_ms=elapsed_ms,
                )
                state.append(step)
                state.refused = True
                state.refuse_reason = (
                    f"Model output could not be parsed after one retry ({exc.reason})."
                )
                state.termination_reason = "parse_failed_after_retry"
                state.finished = True
                return state
            # First parse failure — queue a corrective prompt and let
            # the router send us back through the plan node.
            step = TrajectoryStep(
                step_index=state.step_count,
                kind="parse_error",
                raw_model_output=raw,
                parse_error=exc.reason,
                elapsed_ms=elapsed_ms,
            )
            state.append(step)
            state.messages.append(("assistant", raw))
            state.messages.append(("user", format_corrective_prompt(exc)))
            state.retry_pending = True
            state.last_parse_error = exc.reason
            return state

        # Successful parse — store the action and the model's raw output
        # in the trajectory, and append to the message history.
        state.retry_pending = False
        state.last_parse_error = None
        state.pending_action = action
        state.messages.append(("assistant", raw))
        state.append(
            TrajectoryStep(
                step_index=state.step_count,
                kind="plan",
                raw_model_output=raw,
                parsed_action=action,
                elapsed_ms=elapsed_ms,
            )
        )
        return state

    def _dispatch(self, state: AgentState) -> AgentState:
        """Execute the pending tool call or finalise an answer/refusal."""
        action = state.pending_action
        assert action is not None, "dispatch called without pending action"

        if action.kind is ActionKind.FINAL_ANSWER:
            state.finished = True
            state.final_answer = action.answer_text
            state.final_citations = list(action.citations)
            state.termination_reason = "final_answer"
            return state

        if action.kind is ActionKind.REFUSE:
            state.finished = True
            state.refused = True
            state.refuse_reason = action.refuse_reason or "model refused"
            state.termination_reason = "model_refused"
            return state

        # Tool call. Loop guard: if the previous step was a tool call
        # with the same name and identical args, treat as a loop and
        # force refuse.
        if _is_repeat_tool_call(state, action):
            state.finished = True
            state.refused = True
            state.refuse_reason = (
                f"Loop detected: tool '{action.tool_name}' called twice "
                "with identical arguments. Aborting to avoid infinite loop."
            )
            state.termination_reason = "loop_guard"
            return state

        if action.tool_name is None or action.tool_name not in self.tools:
            # Unknown tool — treat as an execution error and feed back.
            err = f"unknown tool '{action.tool_name}'. Available: {', '.join(self.tools.names())}."
            step = TrajectoryStep(
                step_index=state.step_count,
                kind="tool_result",
                tool_name=action.tool_name,
                tool_args=action.tool_args,
                tool_success=False,
                tool_error=err,
            )
            state.append(step)
            state.messages.append(("user", _format_tool_result_message(step)))
            state.pending_action = None
            return state

        tool = self.tools[action.tool_name]
        t0 = time.monotonic()
        try:
            result = tool(**action.tool_args)
        except (TypeError, ValueError) as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            step = TrajectoryStep(
                step_index=state.step_count,
                kind="tool_result",
                tool_name=action.tool_name,
                tool_args=action.tool_args,
                tool_success=False,
                tool_error=f"tool raised: {exc}",
                elapsed_ms=elapsed_ms,
            )
            state.append(step)
            state.messages.append(("user", _format_tool_result_message(step)))
            state.pending_action = None
            return state

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        step = TrajectoryStep(
            step_index=state.step_count,
            kind="tool_result",
            tool_name=action.tool_name,
            tool_args=action.tool_args,
            tool_success=result.success,
            tool_data=result.data,
            tool_error=result.error,
            elapsed_ms=elapsed_ms,
        )
        state.append(step)
        state.messages.append(("user", _format_tool_result_message(step)))
        state.pending_action = None
        return state

    def _finalise(self, state: AgentState) -> AgentState:
        """No-op terminal node. Exists so the graph has a clean END edge."""
        return state

    # ---- routers -----------------------------------------------------

    def _after_plan(self, state: AgentState) -> str:
        if state.finished:
            return NODE_FINALISE
        if state.retry_pending:
            # Loop back through plan for the corrective retry.
            return NODE_PLAN
        return NODE_DISPATCH

    def _after_dispatch(self, state: AgentState) -> str:
        if state.finished:
            return NODE_FINALISE
        return NODE_PLAN


# ---- helpers ----------------------------------------------------------


def _flatten_messages(messages: list[tuple[str, str]]) -> str:
    """Render the (role, content) message list as a flat prompt.

    Concrete generators (the MedGemma wrapper) may override this with a
    chat template. The fallback here gives a sensible-looking prompt
    that's easy to inspect.
    """
    blocks = []
    for role, content in messages:
        blocks.append(f"<|{role}|>\n{content}")
    blocks.append("<|assistant|>\n")
    return "\n".join(blocks)


def _restore_stop_tag(text: str) -> str:
    """Reattach the closing tag we stopped on, if needed.

    Generators that honour the ``stop`` argument typically truncate
    *before* the stop sequence. Re-adding it makes the parser's life
    easier.
    """
    for tag in ("</tool_call>", "</final_answer>", "</refuse>"):
        # If the response contains the opener but not the closer, add it.
        opener = tag.replace("</", "<").split(">")[0] + ">"
        # opener is "<tool_call>" etc. — match without requiring exact whitespace.
        bare = opener[1:-1]  # "tool_call"
        if f"<{bare}" in text and tag not in text:
            return text + tag
    return text


def _is_repeat_tool_call(state: AgentState, action: ParsedAction) -> bool:
    """Loop guard: catch immediate repeats."""
    for step in reversed(state.trajectory):
        if step.kind == "tool_result":
            return step.tool_name == action.tool_name and step.tool_args == action.tool_args
        if step.kind == "plan":
            continue
    return False
