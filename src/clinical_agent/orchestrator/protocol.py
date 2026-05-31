"""Parser for the tagged-block agent protocol.

The model emits one or more blocks in its response. The orchestrator
only acts on the *last* action-bearing block (``<tool_call>``,
``<final_answer>``, or ``<refuse>``). ``<thinking>`` blocks are kept
for the trajectory pane but not acted upon.

Why tags rather than JSON-as-whole-response: small instruction-tuned
models reliably emit prose around any structured output, and brittle
"the whole response must be valid JSON" parsing fails far more often
than tag extraction. Tags also let us interleave reasoning with action.

Grammar (informal):

    <thinking> free text </thinking>      # zero or more
    <tool_call name="...">                 # exactly one when tool-calling
        { ...json args... }
    </tool_call>

    -- OR --

    <final_answer>                         # exactly one when answering
        free text
        <citations>chunk_id_1, pmid:38000001</citations>   # optional
    </final_answer>

    -- OR --

    <refuse reason="..."/>                 # explicit refusal
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# Tag regex — tolerant of whitespace, multi-line content, and case.
# We use re.DOTALL so . matches newlines inside the block. Anchoring on
# explicit tag names rather than a generic <tag>...</tag> pattern lets
# us treat unknown tags as text rather than failing the parse.
_THINKING_RE = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL | re.IGNORECASE)
_TOOL_CALL_RE = re.compile(
    r'<tool_call\s+name=["\']([^"\']+)["\']\s*>(.*?)</tool_call>',
    re.DOTALL | re.IGNORECASE,
)
_FINAL_ANSWER_RE = re.compile(r"<final_answer>(.*?)</final_answer>", re.DOTALL | re.IGNORECASE)
_CITATIONS_RE = re.compile(r"<citations>(.*?)</citations>", re.DOTALL | re.IGNORECASE)
_REFUSE_RE = re.compile(
    r'<refuse(?:\s+reason=["\']([^"\']*)["\'])?\s*/?>(?:</refuse>)?',
    re.DOTALL | re.IGNORECASE,
)


class ActionKind(StrEnum):
    TOOL_CALL = "tool_call"
    FINAL_ANSWER = "final_answer"
    REFUSE = "refuse"


@dataclass
class ParsedAction:
    """The action the orchestrator should execute next."""

    kind: ActionKind
    # For TOOL_CALL:
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    # For FINAL_ANSWER:
    answer_text: str | None = None
    citations: list[str] = field(default_factory=list)
    # For REFUSE:
    refuse_reason: str | None = None
    # Always carried — useful for the trajectory pane.
    thinking: list[str] = field(default_factory=list)


class ProtocolError(Exception):
    """The model's output couldn't be parsed into a valid action.

    The orchestrator catches this and either retries with a corrective
    prompt or hard-fails to refusal, per the design (one retry then
    hard-fail).
    """

    def __init__(self, reason: str, raw: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.raw = raw


def parse(text: str) -> ParsedAction:
    """Parse a model response into a single ``ParsedAction``.

    Raises ``ProtocolError`` when no action block is present, when
    multiple conflicting actions are present, or when a tool call's
    arguments aren't valid JSON.
    """
    if not text or not text.strip():
        raise ProtocolError("empty model output", text)

    thinking = [m.group(1).strip() for m in _THINKING_RE.finditer(text)]

    refuse_match = _REFUSE_RE.search(text)
    final_match = _FINAL_ANSWER_RE.search(text)
    tool_match = _TOOL_CALL_RE.search(text)

    # Action priority is decided by which is *last* in the text: the
    # model may "think out loud" about calling a tool then change its
    # mind and answer. Last action wins.
    candidates: list[tuple[int, ActionKind, re.Match[str]]] = []
    if refuse_match:
        candidates.append((refuse_match.start(), ActionKind.REFUSE, refuse_match))
    if final_match:
        candidates.append((final_match.start(), ActionKind.FINAL_ANSWER, final_match))
    if tool_match:
        candidates.append((tool_match.start(), ActionKind.TOOL_CALL, tool_match))

    if not candidates:
        raise ProtocolError("no <tool_call>, <final_answer>, or <refuse> block found", text)

    # Pick the latest-positioned action block.
    candidates.sort(key=lambda c: c[0])
    _, kind, match = candidates[-1]

    if kind is ActionKind.REFUSE:
        return ParsedAction(
            kind=ActionKind.REFUSE,
            refuse_reason=(match.group(1) or "").strip() or None,
            thinking=thinking,
        )

    if kind is ActionKind.FINAL_ANSWER:
        inner = match.group(1)
        cite_match = _CITATIONS_RE.search(inner)
        citations: list[str] = []
        if cite_match:
            raw = cite_match.group(1)
            citations = [c.strip() for c in raw.split(",") if c.strip()]
            inner = _CITATIONS_RE.sub("", inner)
        return ParsedAction(
            kind=ActionKind.FINAL_ANSWER,
            answer_text=inner.strip(),
            citations=citations,
            thinking=thinking,
        )

    # ToolCall
    tool_name = match.group(1).strip()
    body = match.group(2).strip()
    try:
        args = json.loads(body) if body else {}
    except json.JSONDecodeError as exc:
        raise ProtocolError(
            f"tool_call for '{tool_name}' has invalid JSON body: {exc}", text
        ) from exc
    if not isinstance(args, dict):
        raise ProtocolError(f"tool_call for '{tool_name}' must be a JSON object", text)
    return ParsedAction(
        kind=ActionKind.TOOL_CALL,
        tool_name=tool_name,
        tool_args=args,
        thinking=thinking,
    )


def format_corrective_prompt(error: ProtocolError) -> str:
    """Build the corrective message used on the one allowed retry.

    Deliberately short — small models lose the plot if the corrective
    prompt itself is long.
    """
    return (
        "Your previous response could not be parsed: "
        f"{error.reason}.\n"
        "Respond with exactly ONE of:\n"
        '  <tool_call name="TOOL"> {"arg": "value"} </tool_call>\n'
        "  <final_answer> ... <citations>id1, id2</citations> </final_answer>\n"
        '  <refuse reason="why"/>\n'
        "Do not include any other action blocks."
    )
