"""Thin wrapper around Anthropic-compatible async clients (Claude, MiniMax)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import anthropic
import httpx

logger = logging.getLogger(__name__)


class LLMClient:
    """Async LLM client for Anthropic-compatible APIs.

    Works with any provider that exposes the Anthropic Messages API
    (Claude, MiniMax, etc.).  Designed to be injected into agents so
    that tests can substitute a mock.
    """

    def __init__(
        self,
        *,
        model: str = "claude-sonnet-4-20250514",
        base_url: str | None = None,
        api_key: str | None = None,
        client: anthropic.AsyncAnthropic | None = None,
        extra_body: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> None:
        if client:
            self._client = client
        else:
            # Use a custom httpx client to handle proxies with self-signed certs
            http_client = httpx.AsyncClient(verify=False) if base_url else None
            kwargs: dict[str, Any] = {}
            if base_url:
                kwargs["base_url"] = base_url
            if api_key:
                kwargs["api_key"] = api_key
            if http_client:
                kwargs["http_client"] = http_client
            self._client = anthropic.AsyncAnthropic(**kwargs)
        self.model = model
        self._extra_body = extra_body or {}
        self._temperature = temperature

    async def create_message(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int = 16384,
    ) -> anthropic.types.Message:
        """Send a single tool-use request to an Anthropic-compatible API."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "system": system,
            "messages": messages,
            "tools": tools,
            "max_tokens": max_tokens,
            "temperature": self._temperature,
        }
        # Force specific tool if exactly one tool is provided
        if len(tools) == 1:
            kwargs["tool_choice"] = {"type": "tool", "name": tools[0]["name"]}
        elif tools:
            kwargs["tool_choice"] = {"type": "any"}
        # Provider-specific parameters (e.g., MiniMax context_window, effort)
        if self._extra_body:
            kwargs["extra_body"] = self._extra_body
        # Retry with backoff on rate limit (429) and overload (529)
        # Short waits for transient limits, long waits for quota exhaustion
        _BACKOFF = [10, 30, 60, 300, 1800]  # 10s, 30s, 1m, 5m, 30m
        for attempt, wait in enumerate(_BACKOFF):
            try:
                return await self._client.messages.create(**kwargs)
            except (anthropic.RateLimitError, anthropic.APIStatusError) as e:
                is_last = attempt == len(_BACKOFF) - 1
                if isinstance(e, anthropic.APIStatusError) and e.status_code not in (429, 529):
                    raise  # Not a rate limit error
                if is_last:
                    raise
                logger.warning(
                    "Rate limit %s, waiting %ds (attempt %d/%d)",
                    e.status_code if hasattr(e, "status_code") else 429,
                    wait, attempt + 1, len(_BACKOFF),
                )
                await asyncio.sleep(wait)
        raise RuntimeError("Unreachable")
