"""Thin wrapper around Anthropic's async client."""

from __future__ import annotations

from typing import Any

import anthropic


class LLMClient:
    """Async Claude API client.

    Thin wrapper that holds the Anthropic client and exposes a single
    ``create_message`` method.  Designed to be injected into agents so
    that tests can substitute a mock.
    """

    def __init__(
        self,
        *,
        model: str = "claude-sonnet-4-20250514",
        client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        self._client = client or anthropic.AsyncAnthropic()
        self.model = model

    async def create_message(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int = 4096,
    ) -> anthropic.types.Message:
        """Send a single tool-use request to Claude."""
        return await self._client.messages.create(
            model=self.model,
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            tool_choice={"type": "any"},
        )
