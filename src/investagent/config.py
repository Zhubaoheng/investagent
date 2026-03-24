"""Settings: model names, hurdle rates, thresholds."""

import os


class Settings:
    """Project-wide configuration, overridable via environment variables."""

    model_name: str = os.getenv("INVESTAGENT_MODEL", "claude-sonnet-4-20250514")
    max_tokens: int = int(os.getenv("INVESTAGENT_MAX_TOKENS", "4096"))
    hurdle_rate: float = 0.10
    net_cash_watch_threshold: float = 0.5
    net_cash_priority_threshold: float = 1.0
    net_cash_high_priority_threshold: float = 1.5
