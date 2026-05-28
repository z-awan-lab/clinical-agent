"""Clinical score primitives.

Every score is a ``BaseScorer`` subclass with an explicit input contract.
The design principle from design.md: **calculators return numbers; the
LLM never recomputes them.** So these are pure, deterministic Python,
exhaustively tested against published reference cases.

Strict input handling: a scorer refuses to produce a score if any
required input is missing, returning a ``ScoreResult`` with
``ok=False`` and the list of missing fields. It never assumes a missing
value is normal — silently treating absent data as normal can
understate a sick patient, which is exactly the failure mode a clinical
tool must avoid.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScoreResult:
    """Outcome of a scoring attempt.

    On success: ``ok=True``, ``score`` set, ``components`` carries the
    per-item breakdown, ``interpretation`` a short human-readable note.
    On refusal: ``ok=False``, ``missing`` lists the absent required
    inputs and ``score`` is None.
    """

    name: str
    ok: bool
    score: int | float | None = None
    components: dict[str, int | float] = field(default_factory=dict)
    interpretation: str | None = None
    missing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "score": self.score,
            "components": self.components,
            "interpretation": self.interpretation,
            "missing": self.missing,
            "notes": self.notes,
        }


class BaseScorer(ABC):
    """A single clinical score.

    Subclasses declare ``name``, ``required`` (the input keys that must
    be present), and implement ``_compute`` which is only called once all
    required inputs are confirmed present.
    """

    name: str
    required: tuple[str, ...]

    def score(self, **inputs: Any) -> ScoreResult:
        """Validate inputs strictly, then compute.

        Missing or ``None`` required inputs cause a refusal — no
        assume-normal fallback.
        """
        missing = [key for key in self.required if key not in inputs or inputs[key] is None]
        if missing:
            return ScoreResult(
                name=self.name,
                ok=False,
                missing=sorted(missing),
                interpretation=(
                    f"Cannot compute {self.name}: missing required inputs "
                    f"{', '.join(sorted(missing))}."
                ),
            )
        return self._compute(**inputs)

    @abstractmethod
    def _compute(self, **inputs: Any) -> ScoreResult:
        """Compute the score. All required inputs are guaranteed present."""
        ...


def in_range(value: float, low: float | None, high: float | None) -> bool:
    """Half-open-friendly range check used by banded scores.

    ``low``/``high`` of None mean unbounded on that side. Both bounds
    inclusive — clinical score bands are conventionally written with
    inclusive edges and the per-score logic picks the correct band order.
    """
    if low is not None and value < low:
        return False
    return not (high is not None and value > high)
