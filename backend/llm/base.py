"""Language model interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

class LanguageModel(ABC):
    @abstractmethod
    def generate(self, system: str, prompt: str) -> str:
        """Generate one response."""

    def select_tools(self, system: str, prompt: str, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Optionally return native function calls.

        Providers without function-calling support return an empty list so the
        planner can use its deterministic Plan-and-Execute fallback.
        """

        return []
