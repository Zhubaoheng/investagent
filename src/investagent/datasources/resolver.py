"""Exchange-to-fetcher resolver.

Maps a CompanyIntake's exchange field to the correct FilingFetcher and
yfinance ticker format.
"""

from __future__ import annotations

from investagent.datasources.base import FilingFetcher, MarketDataFetcher
from investagent.datasources.cninfo import CninfoFetcher
from investagent.datasources.edgar import EdgarFetcher
from investagent.datasources.hkex import HKEXFetcher
from investagent.datasources.market_data import YFinanceFetcher

# Canonical exchange identifiers used in CompanyIntake.exchange
# Grouped by market for clarity.
_EXCHANGE_TO_MARKET: dict[str, str] = {
    # A-shares
    "SSE": "A_SHARE",
    "SZSE": "A_SHARE",
    "BSE": "A_SHARE",
    "上交所": "A_SHARE",
    "深交所": "A_SHARE",
    "北交所": "A_SHARE",
    # Hong Kong
    "HKEX": "HK",
    "港交所": "HK",
    # US ADR
    "NYSE": "US_ADR",
    "NASDAQ": "US_ADR",
    "纽交所": "US_ADR",
    "纳斯达克": "US_ADR",
}

# yfinance ticker suffixes by exchange
_YFINANCE_SUFFIX: dict[str, str] = {
    "SSE": ".SS",
    "上交所": ".SS",
    "SZSE": ".SZ",
    "深交所": ".SZ",
    "BSE": ".BJ",
    "北交所": ".BJ",
    "HKEX": ".HK",
    "港交所": ".HK",
    # US exchanges — no suffix needed
    "NYSE": "",
    "NASDAQ": "",
    "纽交所": "",
    "纳斯达克": "",
}


def resolve_market(exchange: str) -> str:
    """Return canonical market string: 'A_SHARE', 'HK', or 'US_ADR'."""
    market = _EXCHANGE_TO_MARKET.get(exchange)
    if market is None:
        raise ValueError(
            f"Unknown exchange '{exchange}'. "
            f"Valid values: {sorted(_EXCHANGE_TO_MARKET.keys())}"
        )
    return market


def resolve_filing_fetcher(exchange: str) -> FilingFetcher:
    """Return the appropriate FilingFetcher for the given exchange."""
    market = resolve_market(exchange)
    if market == "A_SHARE":
        return CninfoFetcher()
    elif market == "HK":
        return HKEXFetcher()
    elif market == "US_ADR":
        return EdgarFetcher()
    raise ValueError(f"No filing fetcher for market '{market}'")


def resolve_market_data_fetcher() -> MarketDataFetcher:
    """Return the market data fetcher (yfinance covers all three markets)."""
    return YFinanceFetcher()


def to_yfinance_ticker(ticker: str, exchange: str) -> str:
    """Convert a raw ticker + exchange to a yfinance-compatible ticker.

    Examples:
        to_yfinance_ticker("600519", "SSE")   -> "600519.SS"
        to_yfinance_ticker("0700", "HKEX")    -> "0700.HK"
        to_yfinance_ticker("BABA", "NYSE")    -> "BABA"
    """
    suffix = _YFINANCE_SUFFIX.get(exchange, "")
    # Avoid double-suffix (e.g., "0700.HK" + ".HK")
    if suffix and not ticker.endswith(suffix):
        return f"{ticker}{suffix}"
    return ticker
