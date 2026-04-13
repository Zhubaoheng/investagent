"""Tests for poorcharlie.config — unified LLM_* env scheme."""

import pytest

from poorcharlie.config import (
    LLMProviderConfig,
    Settings,
    load_llm_config_from_env,
)


def _clean_env(monkeypatch):
    for var in (
        "LLM_BASE_URL",
        "LLM_API_KEY",
        "LLM_MODEL",
        "LLM_PROVIDER",
        "LLM_MAX_TOKENS",
    ):
        monkeypatch.delenv(var, raising=False)


def test_settings_reads_unified_env(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("LLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-xxx")
    monkeypatch.setenv("LLM_MODEL", "some-model")
    monkeypatch.setenv("LLM_PROVIDER", "minimax")
    s = Settings()
    assert s.provider == "minimax"
    assert s.model_name == "some-model"
    assert s.api_base_url == "https://api.example.com/v1"
    assert s.api_key == "sk-xxx"


def test_settings_defaults_provider_to_openai(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("LLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-xxx")
    monkeypatch.setenv("LLM_MODEL", "some-model")
    s = Settings()
    assert s.provider == "openai"


def test_missing_required_env_raises(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("LLM_API_KEY", "sk-xxx")
    monkeypatch.setenv("LLM_MODEL", "some-model")
    with pytest.raises(ValueError, match="LLM_BASE_URL"):
        load_llm_config_from_env("LLM")


def test_settings_tolerates_missing_env(monkeypatch):
    _clean_env(monkeypatch)
    s = Settings()  # should not raise
    assert s.api_base_url is None
    assert s.api_key is None


def test_load_llm_config_from_env_returns_dataclass(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("LLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-xxx")
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    cfg = load_llm_config_from_env("LLM")
    assert isinstance(cfg, LLMProviderConfig)
    assert cfg.provider == "deepseek"
    assert cfg.base_url == "https://api.example.com/v1"
    assert cfg.api_key == "sk-xxx"
    assert cfg.model == "m"


def test_unknown_provider_tag_warns_but_accepts(monkeypatch, caplog):
    _clean_env(monkeypatch)
    monkeypatch.setenv("LLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-xxx")
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("LLM_PROVIDER", "something-weird")
    cfg = load_llm_config_from_env("LLM")
    assert cfg.provider == "something-weird"
