"""Sepsis-3 criteria check (Singer et al., JAMA 2016).

Not a numeric score — a logical determination:

* **Sepsis** = suspected/documented infection AND an acute rise in SOFA
  of >= 2 points from baseline (baseline assumed 0 if the patient was
  not known to have pre-existing organ dysfunction).
* **Septic shock** = sepsis AND, despite adequate fluid resuscitation,
  both (a) vasopressors required to maintain MAP >= 65 mmHg and
  (b) serum lactate > 2 mmol/L.

This check is deliberately conservative: it reports what the inputs
support and flags what would be needed to escalate the determination.
"""

from __future__ import annotations

from typing import Any

from .base import BaseScorer, ScoreResult


class Sepsis3Scorer(BaseScorer):
    name = "Sepsis-3 criteria"
    required = ("suspected_infection", "sofa_score", "sofa_baseline")

    def _compute(self, **inputs: Any) -> ScoreResult:
        suspected = bool(inputs["suspected_infection"])
        sofa = float(inputs["sofa_score"])
        baseline = float(inputs["sofa_baseline"])
        delta = sofa - baseline

        notes: list[str] = []
        meets_sepsis = suspected and delta >= 2

        # Optional septic-shock escalation.
        on_vaso = inputs.get("vasopressors_for_map65")
        lactate = inputs.get("lactate")
        shock_assessable = on_vaso is not None and lactate is not None
        septic_shock = False
        if meets_sepsis and shock_assessable:
            septic_shock = bool(on_vaso) and float(lactate) > 2.0
        elif meets_sepsis and not shock_assessable:
            notes.append(
                "Septic shock not assessed: provide `vasopressors_for_map65` "
                "(bool) and `lactate` (mmol/L) to evaluate."
            )

        if septic_shock:
            interp = (
                "Meets Sepsis-3 criteria for SEPSIS and SEPTIC SHOCK "
                f"(SOFA rise {delta:+.0f} >= 2; vasopressors + lactate > 2). "
                "Septic shock carries substantially higher mortality."
            )
        elif meets_sepsis:
            interp = (
                f"Meets Sepsis-3 criteria for SEPSIS (SOFA rise {delta:+.0f} "
                ">= 2 with suspected infection)."
            )
        elif suspected and delta < 2:
            interp = (
                f"Does not meet Sepsis-3 sepsis criterion: SOFA rise "
                f"{delta:+.0f} < 2 despite suspected infection. Reassess; "
                "organ dysfunction may evolve."
            )
        else:
            interp = (
                "Does not meet Sepsis-3 criteria: infection not suspected/"
                "documented. Sepsis requires infection plus organ dysfunction."
            )

        return ScoreResult(
            name=self.name,
            ok=True,
            score=None,
            components={
                "suspected_infection": int(suspected),
                "sofa_delta": delta,
                "meets_sepsis": int(meets_sepsis),
                "meets_septic_shock": int(septic_shock),
            },
            interpretation=interp,
            notes=notes,
        )
