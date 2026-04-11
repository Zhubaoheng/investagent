"""CachedFilingFetcher — transparent cache wrapper for any FilingFetcher."""

from __future__ import annotations

import logging

from poorcharlie.datasources.base import FilingDocument, FilingFetcher
from poorcharlie.datasources.cache import FilingCache

logger = logging.getLogger(__name__)


class CachedFilingFetcher(FilingFetcher):
    """Wraps any FilingFetcher with transparent PDF caching.

    search_filings() passes through unchanged.
    download_filing() checks cache first; on miss, delegates to the
    wrapped fetcher and stores the result.
    """

    def __init__(self, inner: FilingFetcher, cache: FilingCache) -> None:
        self._inner = inner
        self._cache = cache

    @property
    def market(self) -> str:
        return self._inner.market

    async def search_filings(
        self,
        ticker: str,
        filing_types: list[str] | None = None,
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> list[FilingDocument]:
        return await self._inner.search_filings(
            ticker, filing_types, start_year, end_year,
        )

    async def download_filing(self, filing: FilingDocument) -> FilingDocument:
        # Check cache
        cached = self._cache.get_pdf(filing)
        if cached is not None:
            logger.info(
                "Filing cache hit: %s %s %s_%s (%d bytes)",
                filing.market, filing.ticker,
                filing.fiscal_period, filing.fiscal_year,
                len(cached),
            )
            return filing.model_copy(update={"raw_content": cached})

        # Cache miss — download from source
        logger.info(
            "Filing cache miss: %s %s %s_%s — downloading",
            filing.market, filing.ticker,
            filing.fiscal_period, filing.fiscal_year,
        )
        result = await self._inner.download_filing(filing)

        # Store in cache
        if result.raw_content:
            self._cache.put_pdf(result, result.raw_content)

        return result
