"""Abstract base class for agent tools.

Every tool the agent can call inherits from ``BaseTool`` and returns a
``ToolResult``. This keeps the orchestrator (Phase 4) decoupled from
individual tool implementations and makes mocking easy in tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Standard envelope for tool outputs.

    Tools should never raise on expected failures (e.g. an empty search
    result, a non-2xx HTTP response). They return ``success=False`` and a
    populated ``error`` field. Only unexpected, programmer-fault errors
    should bubble up as exceptions.
    """

    success: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaseTool(ABC):
    """Abstract base class for agent tools.

    Subclasses must define ``name``, ``description``, and ``input_schema``
    (a JSON schema describing the expected arguments). The orchestrator
    will surface ``name`` + ``description`` + ``input_schema`` to the LLM
    in its system prompt so the model can decide when to call the tool.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    input_schema: ClassVar[dict[str, Any]]

    @abstractmethod
    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with keyword arguments. Must not raise on
        expected failures — return ``ToolResult(success=False, error=...)``
        instead.
        """
        ...

    def __call__(self, **kwargs: Any) -> ToolResult:
        return self.run(**kwargs)

    def to_spec(self) -> dict[str, Any]:
        """Return a serialisable spec for the orchestrator's prompt."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
