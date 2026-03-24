"""Tests for investagent.llm."""

from unittest.mock import MagicMock

from investagent.llm import LLMClient


def test_llm_client_default_model():
    client = LLMClient(client=MagicMock())
    assert client.model == "claude-sonnet-4-20250514"


def test_llm_client_custom_model():
    client = LLMClient(model="claude-haiku-4-5-20251001", client=MagicMock())
    assert client.model == "claude-haiku-4-5-20251001"
