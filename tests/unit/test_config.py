"""Tests for investagent.config."""

import pytest

from investagent.config import Settings


def _clean_env(monkeypatch):
    """Remove all provider-related env vars to start from a clean slate."""
    for var in (
        "INVESTAGENT_PROVIDER",
        "INVESTAGENT_MODEL",
        "INVESTAGENT_MAX_TOKENS",
        "MINIMAX_API_KEY",
        "MINIMAX_BASE_URL",
    ):
        monkeypatch.delenv(var, raising=False)


def test_default_provider_is_claude(monkeypatch):
    _clean_env(monkeypatch)
    s = Settings()
    assert s.provider == "claude"
    assert s.model_name == "claude-sonnet-4-20250514"
    assert s.api_base_url is None
    assert s.api_key is None


def test_minimax_provider_defaults(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("INVESTAGENT_PROVIDER", "minimax")
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key-123")
    s = Settings()
    assert s.provider == "minimax"
    assert s.model_name == "MiniMax-M2.7"
    assert s.api_base_url == "https://api.minimaxi.com/anthropic"
    assert s.api_key == "test-key-123"


def test_minimax_custom_base_url(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("INVESTAGENT_PROVIDER", "minimax")
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
    monkeypatch.setenv("MINIMAX_BASE_URL", "https://api.minimax.io/anthropic")
    s = Settings()
    assert s.api_base_url == "https://api.minimax.io/anthropic"


def test_minimax_missing_api_key_raises(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("INVESTAGENT_PROVIDER", "minimax")
    with pytest.raises(ValueError, match="MINIMAX_API_KEY"):
        Settings()


def test_explicit_model_overrides_provider_default(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("INVESTAGENT_PROVIDER", "minimax")
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
    monkeypatch.setenv("INVESTAGENT_MODEL", "MiniMax-M2.5")
    s = Settings()
    assert s.model_name == "MiniMax-M2.5"


def test_unknown_provider_raises(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("INVESTAGENT_PROVIDER", "openai")
    with pytest.raises(ValueError, match="Unknown provider"):
        Settings()
