"""OpenAI-compatible chat adapter for cloud and local servers."""
from __future__ import annotations

import json
import os
from typing import Any
from backend.llm.base import LanguageModel

class OpenAIChatModel(LanguageModel):
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None) -> None:
        from openai import OpenAI
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY", "local"),
            base_url=base_url,
            timeout=45.0,
            max_retries=1,
        )
        self.model = model

    def generate(self, system: str, prompt: str) -> str:
        lightweight = "expand search queries" in system or "You are the intent router" in system
        client = self.client.with_options(timeout=15.0, max_retries=0) if lightweight else self.client
        response = client.chat.completions.create(model=self.model, temperature=0.1, messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}])
        return response.choices[0].message.content or ""

    def select_tools(self, system: str, prompt: str, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Ask an OpenAI-compatible model for a bounded set of native tool calls."""

        response = self.client.with_options(timeout=20.0, max_retries=0).chat.completions.create(
            model=self.model,
            temperature=0.0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            tools=tools,
            tool_choice="auto",
        )
        calls = []
        for call in response.choices[0].message.tool_calls or []:
            if getattr(call, "type", "function") != "function":
                continue
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            calls.append({"call_id": call.id, "name": call.function.name, "arguments": arguments})
        return calls
