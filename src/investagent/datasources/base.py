"""Abstract base classes and shared models for data sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Shared models
# ---------------------------------------------------------------------------

class FilingDocument(BaseModel, frozen=True):
    """A single filing document retrieved from a data source."""

    market: str                        # "A_SHARE" | "HK" | "US_ADR"
    ticker: str
    company_name: str
    filing_type: str                   # "20-F" | "6-K" | "年报" | "半年报" | etc.
    fiscal_year: str                   # "2024"
    fiscal_period: str                 # "FY" | "H1" | "Q1" | ...
    filing_date: date
    source_url: str                    # URL where the filing can be accessed
    content_type: str                  # "html" | "pdf" | "xbrl"
    raw_content: bytes | None = None   # Downloaded content (optional, can be large)
    text_content: str | None = None    # Extracted text if available
    metadata: dict[str, str] = {}      # Extra metadata per source


class MarketQuote(BaseModel, frozen=True):
    """Current market data snapshot for a company."""

    ticker: str
    name: str
    currency: str                      # "USD" | "HKD" | "CNY"
    price: float | None = None
    market_cap: float | None = None
    enterprise_value: float | None = None
    pe_ratio: float | None = None
    pb_ratio: float | None = None
    dividend_yield: float | None = None
    shares_outstanding: float | None = None


# ---------------------------------------------------------------------------
# Abstract fetchers
# ---------------------------------------------------------------------------

class FilingFetcher(ABC):
    """Fetch financial filings for a given company."""

    @property
    @abstractmethod
    def market(self) -> str:
        """Which market this fetcher covers: 'A_SHARE', 'HK', or 'US_ADR'."""

    @abstractmethod
    async def search_filings(
        self,
        ticker: str,
        filing_types: list[str] | None = None,
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> list[FilingDocument]:
        """Search for filings matching criteria.  Returns metadata only (no content)."""

    @abstractmethod
    async def download_filing(self, filing: FilingDocument) -> FilingDocument:
        """Download full content for a filing.  Returns a new FilingDocument with content populated."""


class MarketDataFetcher(ABC):
    """Fetch current market data for a given ticker."""

    @abstractmethod
    async def get_quote(self, ticker: str) -> MarketQuote:
        """Return current market snapshot."""

    @abstractmethod
    async def get_quotes(self, tickers: list[str]) -> list[MarketQuote]:
        """Return market snapshots for multiple tickers."""
