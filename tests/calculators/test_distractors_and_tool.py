"""Tests for distractor scorers and the ClinicalCalculatorTool dispatch."""

from __future__ import annotations

from clinical_agent.calculators import Cha2ds2VascScorer, WellsDvtScorer
from clinical_agent.tools import ClinicalCalculatorTool


class TestCha2ds2Vasc:
    def test_max_score(self) -> None:
        r = Cha2ds2VascScorer().score(
            chf=True,
            hypertension=True,
            age=80,
            diabetes=True,
            prior_stroke=True,
            vascular_disease=True,
            female=True,
        )
        # 1 + 1 + 2(age>=75) + 1 + 2(stroke) + 1 + 1 = 9
        assert r.score == 9

    def test_age_bands(self) -> None:
        base = dict(
            chf=False,
            hypertension=False,
            diabetes=False,
            prior_stroke=False,
            vascular_disease=False,
            female=False,
        )
        assert Cha2ds2VascScorer().score(age=64, **base).score == 0
        assert Cha2ds2VascScorer().score(age=65, **base).score == 1
        assert Cha2ds2VascScorer().score(age=75, **base).score == 2

    def test_missing_refuses(self) -> None:
        r = Cha2ds2VascScorer().score(chf=True)
        assert not r.ok
        assert r.missing


class TestWellsDvt:
    def test_high_probability(self) -> None:
        r = WellsDvtScorer().score(
            active_cancer=True,
            paralysis_or_immobilisation=True,
            bedridden_or_recent_surgery=True,
            localised_tenderness=False,
            entire_leg_swollen=False,
            calf_swelling_gt3cm=False,
            pitting_oedema=False,
            collateral_superficial_veins=False,
            previous_dvt=False,
            alternative_diagnosis_likely=False,
        )
        assert r.score == 3
        assert "high" in r.interpretation.lower()

    def test_alternative_diagnosis_subtracts_two(self) -> None:
        r = WellsDvtScorer().score(
            active_cancer=True,
            paralysis_or_immobilisation=False,
            bedridden_or_recent_surgery=False,
            localised_tenderness=False,
            entire_leg_swollen=False,
            calf_swelling_gt3cm=False,
            pitting_oedema=False,
            collateral_superficial_veins=False,
            previous_dvt=False,
            alternative_diagnosis_likely=True,
        )
        assert r.score == -1  # 1 - 2

    def test_missing_refuses(self) -> None:
        r = WellsDvtScorer().score(active_cancer=True)
        assert not r.ok
        assert r.missing


class TestClinicalCalculatorTool:
    def test_dispatch_qsofa(self) -> None:
        tool = ClinicalCalculatorTool()
        r = tool.run(
            score="qsofa",
            inputs={"respiratory_rate": 24, "systolic_bp": 90, "gcs": 13},
        )
        assert r.success
        assert r.data["score"] == 3
        assert r.metadata["computed"] is True

    def test_unknown_score_errors(self) -> None:
        r = ClinicalCalculatorTool().run(score="apache2", inputs={})
        assert not r.success
        assert "unknown score" in (r.error or "").lower()

    def test_inputs_must_be_dict(self) -> None:
        r = ClinicalCalculatorTool().run(score="qsofa", inputs="not a dict")
        assert not r.success

    def test_strict_refusal_is_successful_call_with_computed_false(self) -> None:
        # Missing inputs => the tool call succeeds, but ok/computed is False
        # and the missing list is populated. The agent must see this, not
        # treat it as a hard error.
        tool = ClinicalCalculatorTool()
        r = tool.run(score="sofa", inputs={"pao2_fio2": 450})
        assert r.success is True
        assert r.data["ok"] is False
        assert r.metadata["computed"] is False
        assert r.metadata["missing"]

    def test_spec_lists_all_scores(self) -> None:
        tool = ClinicalCalculatorTool()
        enum = tool.input_schema["properties"]["score"]["enum"]
        assert set(enum) == {"qsofa", "sofa", "sepsis3", "cha2ds2vasc", "wells_dvt"}
