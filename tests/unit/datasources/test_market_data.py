"""Tests for market data fetcher — all external calls mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from poorcharlie.datasources.base import MarketQuote
from poorcharlie.datasources.market_data import YFinanceFetcher, _detect_currency


# ---------------------------------------------------------------------------
# Currency detection
# ---------------------------------------------------------------------------

def test_detect_currency_us():
    assert _detect_currency("BABA") == "USD"
    assert _detect_currency("JD") == "USD"


def test_detect_currency_hk():
    assert _detect_currency("0700.HK") == "HKD"
    assert _detect_currency("9988.HK") == "HKD"


def test_detect_currency_cn():
    assert _detect_currency("600519.SS") == "CNY"
    assert _detect_currency("000858.SZ") == "CNY"
    assert _detect_currency("430047.BJ") == "CNY"


# ---------------------------------------------------------------------------
# YFinanceFetcher.get_quote
# ---------------------------------------------------------------------------

@pytest.fixture
def fetcher():
    return YFinanceFetcher()


@patch("poorcharlie.datasources.market_data.yf")
async def test_get_quote_full(mock_yf, fetcher):
    mock_ticker = MagicMock()
    mock_ticker.info = {
        "shortName": "Alibaba Group",
        "currency": "USD",
        "regularMarketPrice": 85.50,
        "marketCap": 210_000_000_000,
        "enterpriseValue": 195_000_000_000,
        "trailingPE": 12.5,
        "priceToBook": 1.8,
        "dividendYield": 0.025,
        "sharesOutstanding": 2_500_000_000,
    }
    mock_yf.Ticker.return_value = mock_ticker

    quote = await fetcher.get_quote("BABA")

    assert isinstance(quote, MarketQuote)
    assert quote.ticker == "BABA"
    assert quote.name == "Alibaba Group"
    assert quote.price == 85.50
    assert quote.market_cap == 210_000_000_000
    assert quote.pe_ratio == 12.5


@patch("poorcharlie.datasources.market_data.yf")
async def test_get_quote_fallback_fast_info(mock_yf, fetcher):
    mock_ticker = MagicMock()
    mock_ticker.info = {"regularMarketPrice": None}  # Triggers fallback
    mock_ticker.fast_info = MagicMock(last_price=85.0, market_cap=200_000_000_000)
    mock_yf.Ticker.return_value = mock_ticker

    quote = await fetcher.get_quote("BABA")

    assert quote.price == 85.0
    assert quote.market_cap == 200_000_000_000


@patch("poorcharlie.datasources.market_data.yf")
async def test_get_quote_empty_info(mock_yf, fetcher):
    mock_ticker = MagicMock()
    mock_ticker.info = {}  # Empty info triggers fallback
    mock_ticker.fast_info = MagicMock(last_price=None, market_cap=None)
    mock_yf.Ticker.return_value = mock_ticker

    quote = await fetcher.get_quote("UNKNOWN")

    assert quote.price is None
    assert quote.market_cap is None


# ---------------------------------------------------------------------------
# YFinanceFetcher.get_quotes
# ---------------------------------------------------------------------------

@patch("poorcharlie.datasources.market_data.yf")
async def test_get_quotes_multiple(mock_yf, fetcher):
    def make_ticker(name, price, currency):
        t = MagicMock()
        t.info = {
            "shortName": name,
            "currency": currency,
            "regularMarketPrice": price,
            "marketCap": price * 1_000_000,
        }
        return t

    mock_yf.Ticker.side_effect = [
        make_ticker("Alibaba", 85.0, "USD"),
        make_ticker("Tencent", 380.0, "HKD"),
    ]

    quotes = await fetcher.get_quotes(["BABA", "0700.HK"])

    assert len(quotes) == 2
    tickers = {q.ticker for q in quotes}
    assert "BABA" in tickers
    assert "0700.HK" in tickers


@patch("poorcharlie.datasources.market_data.yf")
async def test_get_quotes_partial_failure(mock_yf, fetcher):
    good = MagicMock()
    good.info = {
        "shortName": "Alibaba",
        "currency": "USD",
        "regularMarketPrice": 85.0,
    }

    def ticker_factory(symbol):
        if symbol == "FAIL":
            raise Exception("Network error")
        return good

    mock_yf.Ticker.side_effect = ticker_factory

    quotes = await fetcher.get_quotes(["BABA", "FAIL"])

    # At least the successful one should come through
    assert len(quotes) >= 1
    assert any(q.ticker == "BABA" for q in quotes)
