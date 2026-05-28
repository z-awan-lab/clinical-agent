"""Manual try-out for the clinical calculators.

Usage:
    python scripts/try_calculator.py            # runs a worked sepsis example
    python scripts/try_calculator.py --list     # list available scores

Demonstrates qSOFA, SOFA, and the Sepsis-3 check on a single worked
patient, plus a strict-refusal example.
"""

from __future__ import annotations

import argparse
import json

from clinical_agent.calculators import REGISTRY
from clinical_agent.tools import ClinicalCalculatorTool


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list available scores")
    args = parser.parse_args()

    if args.list:
        for key, scorer in REGISTRY.items():
            print(f"{key:14s} {scorer.name}")
        return 0

    tool = ClinicalCalculatorTool()

    # A worked example: 68yo with pneumonia, deteriorating.
    print("=== qSOFA (RR 24, SBP 92, GCS 13) ===")
    r = tool.run(score="qsofa", inputs={"respiratory_rate": 24, "systolic_bp": 92, "gcs": 13})
    print(json.dumps(r.data, indent=2))

    print("\n=== SOFA (multi-system) ===")
    r = tool.run(
        score="sofa",
        inputs={
            "pao2_fio2": 220,
            "on_ventilation": False,
            "platelets": 90,
            "bilirubin": 2.4,
            "map": 64,
            "vasopressor": "norepinephrine_le01",
            "gcs": 13,
            "creatinine": 1.8,
        },
    )
    print(json.dumps(r.data, indent=2))
    sofa_total = r.data["score"]

    print("\n=== Sepsis-3 check (suspected infection, SOFA rise from 0) ===")
    r = tool.run(
        score="sepsis3",
        inputs={
            "suspected_infection": True,
            "sofa_score": sofa_total,
            "sofa_baseline": 0,
            "vasopressors_for_map65": True,
            "lactate": 3.1,
        },
    )
    print(json.dumps(r.data, indent=2))

    print("\n=== Strict refusal example: SOFA with only one input ===")
    r = tool.run(score="sofa", inputs={"pao2_fio2": 300})
    print(
        json.dumps({"computed": r.metadata["computed"], "missing": r.metadata["missing"]}, indent=2)
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
