"""Tests for the orchestrator protocol parser."""

from __future__ import annotations

import pytest

from clinical_agent.orchestrator.protocol import (
    ActionKind,
    ProtocolError,
    format_corrective_prompt,
    parse,
)


class TestToolCall:
    def test_basic_tool_call(self) -> None:
        text = '<tool_call name="qsofa">{"respiratory_rate": 24, "systolic_bp": 90, "gcs": 13}</tool_call>'
        a = parse(text)
        assert a.kind is ActionKind.TOOL_CALL
        assert a.tool_name == "qsofa"
        assert a.tool_args == {"respiratory_rate": 24, "systolic_bp": 90, "gcs": 13}

    def test_tool_call_with_thinking(self) -> None:
        text = (
            "<thinking>This question is about sepsis screening, qSOFA fits.</thinking>\n"
            '<tool_call name="qsofa">{"respiratory_rate": 24, "systolic_bp": 90, "gcs": 13}</tool_call>'
        )
        a = parse(text)
        assert a.kind is ActionKind.TOOL_CALL
        assert len(a.thinking) == 1
        assert "qSOFA fits" in a.thinking[0]

    def test_tool_call_multiline_args(self) -> None:
        text = """<tool_call name="clinical_calculator">
{
  "score": "sofa",
  "inputs": {"pao2_fio2": 200}
}
</tool_call>"""
        a = parse(text)
        assert a.kind is ActionKind.TOOL_CALL
        assert a.tool_args["score"] == "sofa"

    def test_invalid_json_raises(self) -> None:
        text = '<tool_call name="qsofa">{not valid json}</tool_call>'
        with pytest.raises(ProtocolError, match="invalid JSON"):
            parse(text)

    def test_non_object_json_raises(self) -> None:
        text = '<tool_call name="qsofa">[1, 2, 3]</tool_call>'
        with pytest.raises(ProtocolError, match="must be a JSON object"):
            parse(text)

    def test_empty_body_is_empty_args(self) -> None:
        text = '<tool_call name="pubmed_search"></tool_call>'
        a = parse(text)
        assert a.tool_args == {}


class TestFinalAnswer:
    def test_basic_final_answer(self) -> None:
        a = parse("<final_answer>The patient meets sepsis criteria.</final_answer>")
        assert a.kind is ActionKind.FINAL_ANSWER
        assert a.answer_text == "The patient meets sepsis criteria."
        assert a.citations == []

    def test_citations_parsed(self) -> None:
        text = (
            "<final_answer>Per NICE NG51, give antibiotics within 1 hour."
            "<citations>abc123, def456, pmid:38000001</citations>"
            "</final_answer>"
        )
        a = parse(text)
        assert a.kind is ActionKind.FINAL_ANSWER
        assert a.citations == ["abc123", "def456", "pmid:38000001"]
        assert "<citations>" not in a.answer_text

    def test_empty_citations_block(self) -> None:
        text = "<final_answer>Short answer.<citations></citations></final_answer>"
        a = parse(text)
        assert a.citations == []


class TestRefuse:
    def test_self_closing(self) -> None:
        a = parse('<refuse reason="insufficient evidence"/>')
        assert a.kind is ActionKind.REFUSE
        assert a.refuse_reason == "insufficient evidence"

    def test_no_reason(self) -> None:
        a = parse("<refuse/>")
        assert a.kind is ActionKind.REFUSE
        assert a.refuse_reason is None

    def test_paired_form(self) -> None:
        a = parse('<refuse reason="x"></refuse>')
        assert a.kind is ActionKind.REFUSE


class TestPriority:
    def test_latest_block_wins(self) -> None:
        # Model thought about a tool then decided to answer — answer wins.
        text = (
            '<tool_call name="qsofa">{}</tool_call>'
            "Actually I have enough already.\n"
            "<final_answer>qSOFA cannot be computed without inputs.</final_answer>"
        )
        a = parse(text)
        assert a.kind is ActionKind.FINAL_ANSWER

    def test_thinking_not_treated_as_action(self) -> None:
        # A <thinking> block mentioning <tool_call> text doesn't count.
        text = "<thinking>I could use tool_call.</thinking><final_answer>done</final_answer>"
        assert parse(text).kind is ActionKind.FINAL_ANSWER


class TestErrors:
    def test_empty_input(self) -> None:
        with pytest.raises(ProtocolError, match="empty"):
            parse("")

    def test_whitespace_only(self) -> None:
        with pytest.raises(ProtocolError, match="empty"):
            parse("   \n  ")

    def test_no_action_block(self) -> None:
        with pytest.raises(ProtocolError, match=r"no .* block found"):
            parse("Just some prose with no action block.")


class TestCorrectivePrompt:
    def test_format_includes_reason(self) -> None:
        err = ProtocolError("invalid JSON body", "raw")
        msg = format_corrective_prompt(err)
        assert "invalid JSON body" in msg
        assert "<tool_call" in msg
        assert "<final_answer>" in msg
        assert "<refuse" in msg
