"""Tests for exchange-to-fetcher resolver."""

import pytest

from investagent.datasources.cninfo import CninfoFetcher
from investagent.datasources.edgar import EdgarFetcher
from investagent.datasources.hkex import HKEXFetcher
from investagent.datasources.market_data import YFinanceFetcher
from investagent.datasources.resolver import (
    resolve_filing_fetcher,
    resolve_market,
    resolve_market_data_fetcher,
    to_yfinance_ticker,
)


def test_resolve_market_a_share():
    assert resolve_market("SSE") == "A_SHARE"
    assert resolve_market("SZSE") == "A_SHARE"
    assert resolve_market("BSE") == "A_SHARE"
    assert resolve_market("上交所") == "A_SHARE"
    assert resolve_market("深交所") == "A_SHARE"


def test_resolve_market_hk():
    assert resolve_market("HKEX") == "HK"
    assert resolve_market("港交所") == "HK"


def test_resolve_market_us():
    assert resolve_market("NYSE") == "US_ADR"
    assert resolve_market("NASDAQ") == "US_ADR"


def test_resolve_market_unknown():
    with pytest.raises(ValueError, match="Unknown exchange"):
        resolve_market("MARS")


def test_resolve_filing_fetcher_types():
    assert isinstance(resolve_filing_fetcher("SSE"), CninfoFetcher)
    assert isinstance(resolve_filing_fetcher("HKEX"), HKEXFetcher)
    assert isinstance(resolve_filing_fetcher("NYSE"), EdgarFetcher)


def test_resolve_market_data_fetcher():
    assert isinstance(resolve_market_data_fetcher(), YFinanceFetcher)


def test_to_yfinance_ticker_sse():
    assert to_yfinance_ticker("600519", "SSE") == "600519.SS"


def test_to_yfinance_ticker_szse():
    assert to_yfinance_ticker("000858", "SZSE") == "000858.SZ"


def test_to_yfinance_ticker_hk():
    assert to_yfinance_ticker("0700", "HKEX") == "0700.HK"
    # Already has suffix
    assert to_yfinance_ticker("0700.HK", "HKEX") == "0700.HK"


def test_to_yfinance_ticker_us():
    assert to_yfinance_ticker("BABA", "NYSE") == "BABA"
    assert to_yfinance_ticker("JD", "NASDAQ") == "JD"
