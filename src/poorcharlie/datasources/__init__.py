"""Data source layer — fetchers for filings, market data, and company info."""

from poorcharlie.datasources.base import (
    FilingDocument,
    FilingFetcher,
    MarketDataFetcher,
    MarketQuote,
)
from poorcharlie.datasources.cninfo import CninfoFetcher
from poorcharlie.datasources.edgar import EdgarFetcher
from poorcharlie.datasources.hkex import HKEXFetcher
from poorcharlie.datasources.market_data import YFinanceFetcher

__all__ = [
    "CninfoFetcher",
    "EdgarFetcher",
    "FilingDocument",
    "FilingFetcher",
    "HKEXFetcher",
    "MarketDataFetcher",
    "MarketQuote",
    "YFinanceFetcher",
]
