"""Market data fetcher using yfinance.

Covers all three target markets via Yahoo Finance tickers:
- A-shares: 600519.SS (Shanghai), 000858.SZ (Shenzhen)
- HK stocks: 0700.HK, 9988.HK
- US ADR: BABA, JD, PDD
"""

from __future__ import annotations

import asyncio
import logging

import yfinance as yf

from investagent.datasources.base import MarketDataFetcher, MarketQuote

logger = logging.getLogger(__name__)

# Currency mapping by exchange suffix
_CURRENCY_MAP: dict[str, str] = {
    ".SS": "CNY",
    ".SZ": "CNY",
    ".BJ": "CNY",
    ".HK": "HKD",
}


def _detect_currency(ticker: str) -> str:
    """Detect currency from ticker suffix."""
    for suffix, currency in _CURRENCY_MAP.items():
        if ticker.upper().endswith(suffix):
            return currency
    return "USD"


def _fetch_quote_sync(ticker: str) -> MarketQuote:
    """Fetch a single quote via yfinance (synchronous)."""
    yticker = yf.Ticker(ticker)
    info = yticker.info

    if not info or info.get("regularMarketPrice") is None:
        # Try fast_info as fallback
        fi = yticker.fast_info
        return MarketQuote(
            ticker=ticker,
            name=ticker,
            currency=_detect_currency(ticker),
            price=getattr(fi, "last_price", None),
            market_cap=getattr(fi, "market_cap", None),
        )

    return MarketQuote(
        ticker=ticker,
        name=info.get("shortName", info.get("longName", ticker)),
        currency=info.get("currency", _detect_currency(ticker)),
        price=info.get("regularMarketPrice") or info.get("currentPrice"),
        market_cap=info.get("marketCap"),
        enterprise_value=info.get("enterpriseValue"),
        pe_ratio=info.get("trailingPE"),
        pb_ratio=info.get("priceToBook"),
        dividend_yield=info.get("dividendYield"),
        shares_outstanding=info.get("sharesOutstanding"),
    )


class YFinanceFetcher(MarketDataFetcher):
    """Fetch market data from Yahoo Finance across all three markets."""

    async def get_quote(self, ticker: str) -> MarketQuote:
        return await asyncio.to_thread(_fetch_quote_sync, ticker)

    async def get_quotes(self, tickers: list[str]) -> list[MarketQuote]:
        tasks = [asyncio.to_thread(_fetch_quote_sync, t) for t in tickers]
        results: list[MarketQuote] = []
        for coro in asyncio.as_completed(tasks):
            try:
                quote = await coro
                results.append(quote)
            except Exception:
                logger.warning("Failed to fetch quote", exc_info=True)
        return results
