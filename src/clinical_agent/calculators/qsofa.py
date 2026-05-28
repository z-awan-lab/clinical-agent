"""qSOFA (quick SOFA) bedside screening score.

Three criteria, one point each (range 0-3):
    * Respiratory rate >= 22 breaths/min
    * Altered mentation (GCS < 15)
    * Systolic blood pressure <= 100 mmHg

A score of >= 2 identifies patients with suspected infection at greater
risk of poor outcome. qSOFA is a *screening* tool, not a diagnostic
criterion for sepsis (Sepsis-3, Singer et al. JAMA 2016).

Inputs:
    respiratory_rate: breaths per minute
    systolic_bp: mmHg
    gcs: Glasgow Coma Scale total (3-15); altered mentation is GCS < 15
        (alternatively pass ``altered_mentation`` as a bool directly)
"""

from __future__ import annotations

from typing import Any

from .base import BaseScorer, ScoreResult


class QSofaScorer(BaseScorer):
    name = "qSOFA"
    required = ("respiratory_rate", "systolic_bp")

    def _compute(self, **inputs: Any) -> ScoreResult:
        rr = float(inputs["respiratory_rate"])
        sbp = float(inputs["systolic_bp"])

        # Mentation: accept either an explicit bool or derive from GCS.
        altered: bool
        notes: list[str] = []
        if inputs.get("altered_mentation") is not None:
            altered = bool(inputs["altered_mentation"])
        elif inputs.get("gcs") is not None:
            altered = float(inputs["gcs"]) < 15
        else:
            return ScoreResult(
                name=self.name,
                ok=False,
                missing=["gcs_or_altered_mentation"],
                interpretation=(
                    "Cannot compute qSOFA: provide either `gcs` or `altered_mentation`."
                ),
            )

        rr_pt = 1 if rr >= 22 else 0
        ment_pt = 1 if altered else 0
        sbp_pt = 1 if sbp <= 100 else 0
        total = rr_pt + ment_pt + sbp_pt

        if total >= 2:
            interp = (
                "qSOFA >= 2: higher risk of poor outcome in suspected "
                "infection. Consider assessment for organ dysfunction "
                "(full SOFA) and escalation. Screening tool, not diagnostic."
            )
        else:
            interp = (
                "qSOFA < 2: lower risk by this screen. Does not exclude "
                "sepsis; use clinical judgement."
            )

        return ScoreResult(
            name=self.name,
            ok=True,
            score=total,
            components={
                "respiratory_rate>=22": rr_pt,
                "altered_mentation": ment_pt,
                "systolic_bp<=100": sbp_pt,
            },
            interpretation=interp,
            notes=notes,
        )
