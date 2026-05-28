"""SOFA (Sequential Organ Failure Assessment) score.

Six organ systems, each scored 0-4 (total range 0-24). Higher scores
indicate more severe organ dysfunction. Band definitions follow Vincent
et al., Intensive Care Med 1996, as used in the Sepsis-3 definition
(Singer et al., JAMA 2016).

Strict mode: all six systems are required. Partial SOFA is clinically
misleading (a missing system silently scored 0 understates severity),
so this scorer refuses unless every component is supplied.

Required inputs:
    pao2_fio2:        PaO2/FiO2 ratio in mmHg (respiration)
    on_ventilation:   bool, whether respiratory support is mechanical
                      (affects the <= 200 and <= 100 bands)
    platelets:        x10^3 / microlitre (coagulation)
    bilirubin:        mg/dL (liver)
    map:              mean arterial pressure, mmHg (cardiovascular)
    vasopressor:      one of None/"none", "dopamine_le5", "dopamine_gt5",
                      "dopamine_gt15", "epinephrine_le01",
                      "epinephrine_gt01", "norepinephrine_le01",
                      "norepinephrine_gt01" (cardiovascular)
    gcs:              Glasgow Coma Scale total 3-15 (CNS)
    creatinine:       mg/dL (renal)  -- OR --
    urine_output:     mL/day (renal); if both given, the higher band wins
"""

from __future__ import annotations

from typing import Any

from .base import BaseScorer, ScoreResult


def _respiration_points(pao2_fio2: float, ventilated: bool) -> int:
    # The <= 200 and <= 100 bands require respiratory support.
    if pao2_fio2 < 100 and ventilated:
        return 4
    if pao2_fio2 < 200 and ventilated:
        return 3
    if pao2_fio2 < 300:
        return 2
    if pao2_fio2 < 400:
        return 1
    return 0


def _coagulation_points(platelets: float) -> int:
    if platelets < 20:
        return 4
    if platelets < 50:
        return 3
    if platelets < 100:
        return 2
    if platelets < 150:
        return 1
    return 0


def _liver_points(bilirubin: float) -> int:
    if bilirubin >= 12.0:
        return 4
    if bilirubin >= 6.0:
        return 3
    if bilirubin >= 2.0:
        return 2
    if bilirubin >= 1.2:
        return 1
    return 0


_VASOPRESSOR_POINTS = {
    "none": 0,
    "dopamine_le5": 2,  # dopamine <= 5 ug/kg/min
    "dopamine_gt5": 3,  # dopamine > 5
    "dopamine_gt15": 4,  # dopamine > 15
    "epinephrine_le01": 3,  # epinephrine <= 0.1
    "epinephrine_gt01": 4,  # epinephrine > 0.1
    "norepinephrine_le01": 3,
    "norepinephrine_gt01": 4,
}


def _cardiovascular_points(map_mmhg: float, vasopressor: str) -> int:
    vp = _VASOPRESSOR_POINTS.get(vasopressor, 0)
    if vp > 0:
        return vp
    # No vasopressors: score on MAP alone.
    return 1 if map_mmhg < 70 else 0


def _cns_points(gcs: float) -> int:
    if gcs < 6:
        return 4
    if gcs < 10:
        return 3
    if gcs < 13:
        return 2
    if gcs < 15:
        return 1
    return 0


def _renal_points(creatinine: float | None, urine_output: float | None) -> int:
    cr_pts = 0
    if creatinine is not None:
        if creatinine >= 5.0:
            cr_pts = 4
        elif creatinine >= 3.5:
            cr_pts = 3
        elif creatinine >= 2.0:
            cr_pts = 2
        elif creatinine >= 1.2:
            cr_pts = 1
    uo_pts = 0
    if urine_output is not None:
        if urine_output < 200:
            uo_pts = 4
        elif urine_output < 500:
            uo_pts = 3
    return max(cr_pts, uo_pts)


class SofaScorer(BaseScorer):
    name = "SOFA"
    # Renal is satisfied by creatinine OR urine_output, handled separately.
    required = (
        "pao2_fio2",
        "platelets",
        "bilirubin",
        "map",
        "vasopressor",
        "gcs",
    )

    def score(self, **inputs: Any) -> ScoreResult:
        # Renal needs special handling: either creatinine or urine_output.
        base_missing = [k for k in self.required if k not in inputs or inputs[k] is None]
        renal_missing = inputs.get("creatinine") is None and inputs.get("urine_output") is None
        if base_missing or renal_missing:
            miss = list(base_missing)
            if renal_missing:
                miss.append("creatinine_or_urine_output")
            return ScoreResult(
                name=self.name,
                ok=False,
                missing=sorted(miss),
                interpretation=(
                    f"Cannot compute SOFA: missing {', '.join(sorted(miss))}. "
                    "All six organ systems are required."
                ),
            )
        return self._compute(**inputs)

    def _compute(self, **inputs: Any) -> ScoreResult:
        ventilated = bool(inputs.get("on_ventilation", False))
        resp = _respiration_points(float(inputs["pao2_fio2"]), ventilated)
        coag = _coagulation_points(float(inputs["platelets"]))
        liver = _liver_points(float(inputs["bilirubin"]))
        cardio = _cardiovascular_points(float(inputs["map"]), str(inputs["vasopressor"]))
        cns = _cns_points(float(inputs["gcs"]))
        cr = inputs.get("creatinine")
        uo = inputs.get("urine_output")
        renal = _renal_points(
            float(cr) if cr is not None else None,
            float(uo) if uo is not None else None,
        )

        total = resp + coag + liver + cardio + cns + renal

        # Mortality association is non-linear; we report the score and a
        # qualitative band rather than a spurious precise mortality %.
        if total >= 12:
            band = "high organ dysfunction burden"
        elif total >= 6:
            band = "moderate organ dysfunction"
        else:
            band = "low organ dysfunction"

        return ScoreResult(
            name=self.name,
            ok=True,
            score=total,
            components={
                "respiration": resp,
                "coagulation": coag,
                "liver": liver,
                "cardiovascular": cardio,
                "cns": cns,
                "renal": renal,
            },
            interpretation=(
                f"SOFA {total}/24 ({band}). An acute rise of >= 2 points "
                "from baseline in suspected infection meets the Sepsis-3 "
                "organ-dysfunction criterion for sepsis."
            ),
        )
