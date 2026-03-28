"""Settings: model names, hurdle rates, thresholds."""

from __future__ import annotations

import os

_PROVIDER_DEFAULTS: dict[str, dict[str, str | None]] = {
    "claude": {
        "default_model": "claude-sonnet-4-20250514",
        "base_url": None,
        "api_key_env": None,  # SDK reads ANTHROPIC_API_KEY internally
    },
    "minimax": {
        "default_model": "MiniMax-M2.7-highspeed",
        "base_url_default": "https://api.minimaxi.com/anthropic",
        "base_url_env": "MINIMAX_BASE_URL",
        "api_key_env": "MINIMAX_API_KEY",
    },
}


class Settings:
    """Project-wide configuration, overridable via environment variables."""

    hurdle_rate: float = 0.10  # fallback; actual hurdle = 2× risk-free rate per currency

    # Risk-free rates by currency (approximate, updated periodically)
    risk_free_rates: dict[str, float] = {
        "CNY": 0.022,   # 中国10年期国债 ~2.2%
        "HKD": 0.038,   # 香港10年期政府债券 ~3.8%
        "USD": 0.042,   # 美国10年期国债 ~4.2%
    }

    def get_hurdle_rate(self, currency: str = "USD") -> float:
        """Return 2× risk-free rate for the given currency."""
        rfr = self.risk_free_rates.get(currency, 0.04)
        return round(rfr * 2, 4)
    net_cash_watch_threshold: float = 0.5
    net_cash_priority_threshold: float = 1.0
    net_cash_high_priority_threshold: float = 1.5

    def __init__(self) -> None:
        self.provider: str = os.getenv("INVESTAGENT_PROVIDER", "claude")
        if self.provider not in _PROVIDER_DEFAULTS:
            raise ValueError(
                f"Unknown provider {self.provider!r}. "
                f"Supported: {list(_PROVIDER_DEFAULTS)}"
            )

        prov = _PROVIDER_DEFAULTS[self.provider]
        self.model_name: str = os.getenv(
            "INVESTAGENT_MODEL", prov["default_model"]  # type: ignore[arg-type]
        )
        self.max_tokens: int = int(os.getenv("INVESTAGENT_MAX_TOKENS", "4096"))

        # Base URL
        base_url_env = prov.get("base_url_env")
        if base_url_env:
            self.api_base_url: str | None = os.getenv(
                base_url_env, prov.get("base_url_default")
            )
        else:
            self.api_base_url = prov.get("base_url")

        # API key
        api_key_env = prov.get("api_key_env")
        if api_key_env:
            self.api_key: str | None = os.getenv(api_key_env)
            if not self.api_key:
                raise ValueError(
                    f"Provider {self.provider!r} requires the "
                    f"{api_key_env} environment variable"
                )
        else:
            self.api_key = None
