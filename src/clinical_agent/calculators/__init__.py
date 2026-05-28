"""Clinical calculators — deterministic scoring functions.

The ``REGISTRY`` maps score name to scorer instance. The
``ClinicalCalculatorTool`` dispatches to these by name.
"""

from .base import BaseScorer, ScoreResult
from .cha2ds2vasc import Cha2ds2VascScorer
from .qsofa import QSofaScorer
from .sepsis3 import Sepsis3Scorer
from .sofa import SofaScorer
from .wells_dvt import WellsDvtScorer

# Canonical registry. Keys are lowercase, hyphen-free identifiers used in
# tool calls; the scorer's .name carries the human-readable label.
REGISTRY: dict[str, BaseScorer] = {
    "qsofa": QSofaScorer(),
    "sofa": SofaScorer(),
    "sepsis3": Sepsis3Scorer(),
    "cha2ds2vasc": Cha2ds2VascScorer(),
    "wells_dvt": WellsDvtScorer(),
}

__all__ = [
    "REGISTRY",
    "BaseScorer",
    "Cha2ds2VascScorer",
    "QSofaScorer",
    "ScoreResult",
    "Sepsis3Scorer",
    "SofaScorer",
    "WellsDvtScorer",
]
