"""CHA2DS2-VASc score — stroke risk in non-valvular atrial fibrillation.

Included as a *distractor* tool: it is clinically valid but irrelevant to
sepsis. Phase 5 evaluation uses it to measure whether the agent correctly
declines to invoke an off-topic calculator on a sepsis question.

Points:
    C  Congestive heart failure / LV dysfunction      1
    H  Hypertension                                    1
    A2 Age >= 75                                        2
    D  Diabetes mellitus                                1
    S2 Prior stroke / TIA / thromboembolism             2
    V  Vascular disease (MI, PAD, aortic plaque)        1
    A  Age 65-74                                        1
    Sc Sex category (female)                            1
"""

from __future__ import annotations

from typing import Any

from .base import BaseScorer, ScoreResult


class Cha2ds2VascScorer(BaseScorer):
    name = "CHA2DS2-VASc"
    required = (
        "chf",
        "hypertension",
        "age",
        "diabetes",
        "prior_stroke",
        "vascular_disease",
        "female",
    )

    def _compute(self, **inputs: Any) -> ScoreResult:
        age = float(inputs["age"])
        age_pts = 2 if age >= 75 else (1 if age >= 65 else 0)

        components = {
            "chf": 1 if inputs["chf"] else 0,
            "hypertension": 1 if inputs["hypertension"] else 0,
            "age": age_pts,
            "diabetes": 1 if inputs["diabetes"] else 0,
            "prior_stroke": 2 if inputs["prior_stroke"] else 0,
            "vascular_disease": 1 if inputs["vascular_disease"] else 0,
            "female": 1 if inputs["female"] else 0,
        }
        total = sum(components.values())

        return ScoreResult(
            name=self.name,
            ok=True,
            score=total,
            components=components,
            interpretation=(
                f"CHA2DS2-VASc {total}. Used to guide anticoagulation in "
                "atrial fibrillation. Not relevant to sepsis assessment."
            ),
        )
