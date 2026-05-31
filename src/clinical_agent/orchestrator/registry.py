"""Tool registry used by the orchestrator.

A thin wrapper over a dict so the dispatch node has one place to look up
tools by name. Constructed once per agent run.
"""

from __future__ import annotations

from ..tools.base import BaseTool


class ToolRegistry:
    """Name → BaseTool lookup with friendly errors."""

    def __init__(self, tools: list[BaseTool]) -> None:
        # Detect duplicate registrations early — silent overwriting in
        # an agent registry is a notorious source of debugging pain.
        seen: dict[str, BaseTool] = {}
        for t in tools:
            if t.name in seen:
                raise ValueError(f"duplicate tool name: {t.name}")
            seen[t.name] = t
        self._tools = seen

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __getitem__(self, name: str) -> BaseTool:
        if name not in self._tools:
            raise KeyError(f"unknown tool '{name}'. Available: {', '.join(sorted(self._tools))}")
        return self._tools[name]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def all(self) -> list[BaseTool]:
        return [self._tools[n] for n in self.names()]
