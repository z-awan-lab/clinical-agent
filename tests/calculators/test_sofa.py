"""Tests for the SOFA score.

Per-organ-system boundary tests plus a worked multi-system case. Band
edges are taken from Vincent et al. 1996 / Singer et al. 2016.
"""

from __future__ import annotations

import pytest

from clinical_agent.calculators import SofaScorer
from clinical_agent.calculators.sofa import (
    _cardiovascular_points,
    _cns_points,
    _coagulation_points,
    _liver_points,
    _renal_points,
    _respiration_points,
)


class TestRespiration:
    @pytest.mark.parametrize(
        "ratio,vent,expected",
        [
            (450, False, 0),
            (400, False, 0),
            (399, False, 1),
            (300, False, 1),
            (299, False, 2),
            (200, False, 2),
            # < 200 and < 100 bands need ventilation.
            (150, True, 3),
            (150, False, 2),  # not ventilated: capped at the < 300 band
            (90, True, 4),
            (90, False, 2),
        ],
    )
    def test_bands(self, ratio: float, vent: bool, expected: int) -> None:
        assert _respiration_points(ratio, vent) == expected


class TestCoagulation:
    @pytest.mark.parametrize(
        "plt,expected",
        [(200, 0), (150, 0), (149, 1), (100, 1), (99, 2), (50, 2), (49, 3), (20, 3), (19, 4)],
    )
    def test_bands(self, plt: float, expected: int) -> None:
        assert _coagulation_points(plt) == expected


class TestLiver:
    @pytest.mark.parametrize(
        "bili,expected",
        [(1.0, 0), (1.2, 1), (1.9, 1), (2.0, 2), (5.9, 2), (6.0, 3), (11.9, 3), (12.0, 4)],
    )
    def test_bands(self, bili: float, expected: int) -> None:
        assert _liver_points(bili) == expected


class TestCardiovascular:
    def test_map_only_no_vaso(self) -> None:
        assert _cardiovascular_points(80, "none") == 0
        assert _cardiovascular_points(69, "none") == 1

    def test_vasopressor_bands(self) -> None:
        assert _cardiovascular_points(60, "dopamine_le5") == 2
        assert _cardiovascular_points(60, "dopamine_gt5") == 3
        assert _cardiovascular_points(60, "norepinephrine_le01") == 3
        assert _cardiovascular_points(60, "epinephrine_gt01") == 4

    def test_vasopressor_overrides_map(self) -> None:
        # Even a normal MAP scores by the pressor requirement.
        assert _cardiovascular_points(85, "norepinephrine_gt01") == 4


class TestCns:
    @pytest.mark.parametrize(
        "gcs,expected",
        [(15, 0), (14, 1), (13, 1), (12, 2), (10, 2), (9, 3), (6, 3), (5, 4), (3, 4)],
    )
    def test_bands(self, gcs: float, expected: int) -> None:
        assert _cns_points(gcs) == expected


class TestRenal:
    def test_creatinine_bands(self) -> None:
        assert _renal_points(1.0, None) == 0
        assert _renal_points(1.2, None) == 1
        assert _renal_points(2.0, None) == 2
        assert _renal_points(3.5, None) == 3
        assert _renal_points(5.0, None) == 4

    def test_urine_output_bands(self) -> None:
        assert _renal_points(None, 600) == 0
        assert _renal_points(None, 499) == 3
        assert _renal_points(None, 199) == 4

    def test_higher_of_creatinine_or_urine_wins(self) -> None:
        # Creatinine says 1 point, urine output says 4 — take 4.
        assert _renal_points(1.2, 150) == 4


class TestSofaIntegration:
    def test_all_normal_is_zero(self) -> None:
        r = SofaScorer().score(
            pao2_fio2=450,
            on_ventilation=False,
            platelets=250,
            bilirubin=0.8,
            map=85,
            vasopressor="none",
            gcs=15,
            creatinine=0.9,
        )
        assert r.ok
        assert r.score == 0
        assert all(v == 0 for v in r.components.values())

    def test_worked_multisystem_case(self) -> None:
        # Respiration 150 ventilated = 3; platelets 80 = 2; bilirubin 7 = 3;
        # norepinephrine_le01 = 3; gcs 11 = 2; creatinine 2.5 = 2. Total 15.
        r = SofaScorer().score(
            pao2_fio2=150,
            on_ventilation=True,
            platelets=80,
            bilirubin=7.0,
            map=60,
            vasopressor="norepinephrine_le01",
            gcs=11,
            creatinine=2.5,
        )
        assert r.ok
        assert r.components == {
            "respiration": 3,
            "coagulation": 2,
            "liver": 3,
            "cardiovascular": 3,
            "cns": 2,
            "renal": 2,
        }
        assert r.score == 15
        assert "Sepsis-3" in r.interpretation

    def test_renal_via_urine_output_only(self) -> None:
        r = SofaScorer().score(
            pao2_fio2=450,
            on_ventilation=False,
            platelets=250,
            bilirubin=0.8,
            map=85,
            vasopressor="none",
            gcs=15,
            urine_output=150,  # no creatinine
        )
        assert r.ok
        assert r.components["renal"] == 4

    def test_missing_organ_system_refuses(self) -> None:
        r = SofaScorer().score(
            pao2_fio2=450,
            on_ventilation=False,
            platelets=250,
            bilirubin=0.8,
            map=85,
            vasopressor="none",
            # gcs missing
            creatinine=0.9,
        )
        assert not r.ok
        assert "gcs" in r.missing

    def test_missing_both_renal_inputs_refuses(self) -> None:
        r = SofaScorer().score(
            pao2_fio2=450,
            on_ventilation=False,
            platelets=250,
            bilirubin=0.8,
            map=85,
            vasopressor="none",
            gcs=15,
            # neither creatinine nor urine_output
        )
        assert not r.ok
        assert "creatinine_or_urine_output" in r.missing
