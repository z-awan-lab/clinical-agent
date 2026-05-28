"""Tests for qSOFA and Sepsis-3."""

from __future__ import annotations

from clinical_agent.calculators import QSofaScorer, Sepsis3Scorer


class TestQSofa:
    def test_all_three_criteria_met(self) -> None:
        r = QSofaScorer().score(respiratory_rate=24, systolic_bp=90, gcs=13)
        assert r.ok
        assert r.score == 3

    def test_none_met(self) -> None:
        r = QSofaScorer().score(respiratory_rate=16, systolic_bp=120, gcs=15)
        assert r.ok
        assert r.score == 0

    def test_rr_boundary_22_scores(self) -> None:
        # >= 22 scores; 21 does not.
        assert QSofaScorer().score(respiratory_rate=22, systolic_bp=120, gcs=15).score == 1
        assert QSofaScorer().score(respiratory_rate=21, systolic_bp=120, gcs=15).score == 0

    def test_sbp_boundary_100_scores(self) -> None:
        # <= 100 scores; 101 does not.
        assert QSofaScorer().score(respiratory_rate=16, systolic_bp=100, gcs=15).score == 1
        assert QSofaScorer().score(respiratory_rate=16, systolic_bp=101, gcs=15).score == 0

    def test_gcs_15_is_not_altered(self) -> None:
        assert QSofaScorer().score(respiratory_rate=16, systolic_bp=120, gcs=15).score == 0
        assert QSofaScorer().score(respiratory_rate=16, systolic_bp=120, gcs=14).score == 1

    def test_explicit_altered_mentation_bool(self) -> None:
        r = QSofaScorer().score(respiratory_rate=16, systolic_bp=120, altered_mentation=True)
        assert r.score == 1

    def test_high_risk_interpretation_at_2(self) -> None:
        r = QSofaScorer().score(respiratory_rate=24, systolic_bp=90, gcs=15)
        assert r.score == 2
        assert "higher risk" in r.interpretation.lower()

    def test_missing_rr_refuses(self) -> None:
        r = QSofaScorer().score(systolic_bp=90, gcs=13)
        assert not r.ok
        assert "respiratory_rate" in r.missing

    def test_missing_mentation_source_refuses(self) -> None:
        r = QSofaScorer().score(respiratory_rate=24, systolic_bp=90)
        assert not r.ok
        assert any("mentation" in m or "gcs" in m for m in r.missing)


class TestSepsis3:
    def test_meets_sepsis(self) -> None:
        r = Sepsis3Scorer().score(suspected_infection=True, sofa_score=4, sofa_baseline=0)
        assert r.ok
        assert r.components["meets_sepsis"] == 1
        assert "SEPSIS" in r.interpretation

    def test_delta_below_2_does_not_meet(self) -> None:
        r = Sepsis3Scorer().score(suspected_infection=True, sofa_score=1, sofa_baseline=0)
        assert r.components["meets_sepsis"] == 0

    def test_no_infection_does_not_meet(self) -> None:
        r = Sepsis3Scorer().score(suspected_infection=False, sofa_score=6, sofa_baseline=0)
        assert r.components["meets_sepsis"] == 0

    def test_delta_exactly_2_meets(self) -> None:
        r = Sepsis3Scorer().score(suspected_infection=True, sofa_score=5, sofa_baseline=3)
        assert r.components["meets_sepsis"] == 1

    def test_septic_shock_when_vaso_and_lactate(self) -> None:
        r = Sepsis3Scorer().score(
            suspected_infection=True,
            sofa_score=6,
            sofa_baseline=0,
            vasopressors_for_map65=True,
            lactate=3.0,
        )
        assert r.components["meets_septic_shock"] == 1
        assert "SEPTIC SHOCK" in r.interpretation

    def test_lactate_boundary_2_is_not_shock(self) -> None:
        # Lactate must be > 2, not >= 2.
        r = Sepsis3Scorer().score(
            suspected_infection=True,
            sofa_score=6,
            sofa_baseline=0,
            vasopressors_for_map65=True,
            lactate=2.0,
        )
        assert r.components["meets_septic_shock"] == 0

    def test_sepsis_without_shock_inputs_notes_it(self) -> None:
        r = Sepsis3Scorer().score(suspected_infection=True, sofa_score=4, sofa_baseline=0)
        assert r.components["meets_sepsis"] == 1
        assert any("shock" in n.lower() for n in r.notes)

    def test_missing_input_refuses(self) -> None:
        r = Sepsis3Scorer().score(suspected_infection=True, sofa_score=4)
        assert not r.ok
        assert "sofa_baseline" in r.missing
