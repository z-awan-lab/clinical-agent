"""Generation backends."""

from .base import BaseGenerator, GenerationResult
from .medgemma import MedGemmaGenerator

__all__ = ["BaseGenerator", "GenerationResult", "MedGemmaGenerator"]
