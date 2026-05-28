"""Wells score for deep vein thrombosis (DVT) pretest probability.

Second *distractor* tool — clinically valid, irrelevant to sepsis.

Each present feature scores +1, except the final item which subtracts 2:
    active cancer; paralysis/paresis/recent immobilisation of legs;
    recently bedridden >= 3 days or major surgery within 12 weeks;
    localised tenderness along deep venous system; entire leg swollen;
    calf swelling > 3 cm vs asymptomatic side; pitting oedema confined to
    symptomatic leg; collateral superficial veins (non-varicose);
    previously documented DVT; **alternative diagnosis at least as likely
    as DVT: -2**.

Bands: >= 3 high, 1-2 moderate, <= 0 low probability.
"""

from __future__ import annotations

from typing import Any

from .base import BaseScorer, ScoreResult

_FEATURES = (
    "active_cancer",
    "paralysis_or_immobilisation",
    "bedridden_or_recent_surgery",
    "localised_tenderness",
    "entire_leg_swollen",
    "calf_swelling_gt3cm",
    "pitting_oedema",
    "collateral_superficial_veins",
    "previous_dvt",
)


class WellsDvtScorer(BaseScorer):
    name = "Wells DVT"
    required = (*_FEATURES, "alternative_diagnosis_likely")

    def _compute(self, **inputs: Any) -> ScoreResult:
        components = {f: (1 if inputs[f] else 0) for f in _FEATURES}
        alt = -2 if inputs["alternative_diagnosis_likely"] else 0
        components["alternative_diagnosis_likely"] = alt
        total = sum(components.values())

        if total >= 3:
            band = "high"
        elif total >= 1:
            band = "moderate"
        else:
            band = "low"

        return ScoreResult(
            name=self.name,
            ok=True,
            score=total,
            components=components,
            interpretation=(
                f"Wells DVT {total} ({band} pretest probability). Guides "
                "D-dimer / ultrasound decisions for suspected DVT. Not "
                "relevant to sepsis assessment."
            ),
        )
