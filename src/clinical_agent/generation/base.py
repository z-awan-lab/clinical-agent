"""Abstract base class for generation models (the agent's reasoning LLM).

Concrete wrappers around MedGemma 1.5 27B-IT and Qwen 2.5 72B-Instruct
land in Phase 4 alongside the LangGraph orchestrator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar


@dataclass
class GenerationResult:
    """A single generation call's output."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    raw: dict[str, Any] | None = None


class BaseGenerator(ABC):
    """Generate text from a prompt. Tool-calling is handled at the
    orchestrator layer via structured prompting, not here — keeps the
    interface uniform across models that lack native function-calling
    APIs.
    """

    model_name: ClassVar[str]

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> GenerationResult:
        """Generate a completion for ``prompt``."""
        ...
