"""SEC EDGAR data source for US-listed Chinese ADRs (20-F, 6-K).

Uses the ``edgartools`` library for filing search, download, and XBRL extraction.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date
from typing import Any

from poorcharlie.datasources.base import (
    FilingDocument,
    FilingFetcher,
)

logger = logging.getLogger(__name__)

# Mapping from our canonical filing_type to edgartools form codes
_FORM_MAP: dict[str, str] = {
    "20-F": "20-F",
    "6-K": "6-K",
}

# Mapping from edgartools form codes back to our fiscal_period heuristic
_PERIOD_HEURISTIC: dict[str, str] = {
    "20-F": "FY",
    "6-K": "H1",  # best guess; 6-K can be anything
}


def _ensure_identity(identity: str) -> None:
    """Set SEC EDGAR identity (required by fair-access policy)."""
    import edgar  # noqa: E402 — deferred import

    edgar.set_identity(identity)


def _filing_to_document(filing: Any, ticker: str) -> FilingDocument:
    """Convert an edgartools Filing object to our FilingDocument model."""
    form = filing.form
    filing_date_raw = filing.filing_date

    # Parse filing_date
    if isinstance(filing_date_raw, str):
        fd = date.fromisoformat(filing_date_raw)
    elif isinstance(filing_date_raw, date):
        fd = filing_date_raw
    else:
        fd = date.today()

    # Determine fiscal year from period_of_report or filing_date
    period_of_report = getattr(filing, "period_of_report", None)
    if period_of_report:
        if isinstance(period_of_report, str):
            fiscal_year = period_of_report[:4]
        else:
            fiscal_year = str(period_of_report.year)
    else:
        fiscal_year = str(fd.year)

    return FilingDocument(
        market="US_ADR",
        ticker=ticker,
        company_name=getattr(filing, "company", ticker),
        filing_type=form,
        fiscal_year=fiscal_year,
        fiscal_period=_PERIOD_HEURISTIC.get(form, "FY"),
        filing_date=fd,
        source_url=getattr(filing, "homepage_url", ""),
        content_type="html",
        metadata={
            "accession_no": getattr(filing, "accession_no", ""),
            "period_of_report": str(period_of_report) if period_of_report else "",
        },
    )


class EdgarFetcher(FilingFetcher):
    """Fetch 20-F and 6-K filings from SEC EDGAR."""

    def __init__(self, identity: str = "PoorCharlie research@poorcharlie.dev") -> None:
        self._identity = identity
        self._identity_set = False

    @property
    def market(self) -> str:
        return "US_ADR"

    def _ensure_ready(self) -> None:
        if not self._identity_set:
            _ensure_identity(self._identity)
            self._identity_set = True

    # ------------------------------------------------------------------
    # Sync helpers (edgartools is synchronous)
    # ------------------------------------------------------------------

    def _search_sync(
        self,
        ticker: str,
        filing_types: list[str] | None = None,
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> list[FilingDocument]:
        self._ensure_ready()
        from edgar import Company

        company = Company(ticker)

        forms = filing_types or ["20-F", "6-K"]
        edgar_forms = [_FORM_MAP.get(f, f) for f in forms]

        kwargs: dict[str, Any] = {"form": edgar_forms, "amendments": False}

        if start_year and end_year:
            kwargs["filing_date"] = f"{start_year}-01-01:{end_year}-12-31"
        elif start_year:
            kwargs["filing_date"] = f"{start_year}-01-01:"
        elif end_year:
            kwargs["filing_date"] = f":{end_year}-12-31"

        filings = company.get_filings(**kwargs)
        results: list[FilingDocument] = []
        for filing in filings:
            try:
                results.append(_filing_to_document(filing, ticker))
            except Exception:
                logger.warning("Failed to parse filing: %s", filing, exc_info=True)
        return results

    def _download_sync(self, filing: FilingDocument) -> FilingDocument:
        self._ensure_ready()
        from edgar import Company

        company = Company(filing.ticker)
        accession = filing.metadata.get("accession_no", "")
        if not accession:
            raise ValueError(f"No accession_no in filing metadata for {filing.ticker}")

        matches = company.get_filings(accession_number=accession)
        if len(matches) == 0:
            raise ValueError(f"Filing not found: {accession}")

        edgar_filing = matches[0]

        html_content = edgar_filing.html()
        text_content = edgar_filing.text()

        return FilingDocument(
            market=filing.market,
            ticker=filing.ticker,
            company_name=filing.company_name,
            filing_type=filing.filing_type,
            fiscal_year=filing.fiscal_year,
            fiscal_period=filing.fiscal_period,
            filing_date=filing.filing_date,
            source_url=filing.source_url,
            content_type=filing.content_type,
            raw_content=html_content.encode("utf-8") if html_content else None,
            text_content=text_content,
            metadata=filing.metadata,
        )

    # ------------------------------------------------------------------
    # Async interface (wraps sync calls)
    # ------------------------------------------------------------------

    async def search_filings(
        self,
        ticker: str,
        filing_types: list[str] | None = None,
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> list[FilingDocument]:
        return await asyncio.to_thread(
            self._search_sync, ticker, filing_types, start_year, end_year,
        )

    async def download_filing(self, filing: FilingDocument) -> FilingDocument:
        return await asyncio.to_thread(self._download_sync, filing)

    # ------------------------------------------------------------------
    # XBRL financial data extraction (bonus — not in base class)
    # ------------------------------------------------------------------

    def _get_financials_sync(
        self, ticker: str, periods: int = 5,
    ) -> dict[str, Any]:
        """Extract structured financials from XBRL for a ticker.

        Returns a dict with keys: income_statement, balance_sheet, cash_flow,
        each containing a list of dicts (one per period).
        """
        self._ensure_ready()
        from edgar import Company

        company = Company(ticker)
        result: dict[str, Any] = {
            "income_statement": [],
            "balance_sheet": [],
            "cash_flow": [],
        }

        try:
            income = company.income_statement(periods=periods, as_dataframe=True)
            if income is not None:
                result["income_statement"] = income.to_dict(orient="records")
        except Exception:
            logger.warning("Failed to extract income statement for %s", ticker, exc_info=True)

        try:
            bs = company.balance_sheet(periods=periods, as_dataframe=True)
            if bs is not None:
                result["balance_sheet"] = bs.to_dict(orient="records")
        except Exception:
            logger.warning("Failed to extract balance sheet for %s", ticker, exc_info=True)

        try:
            cf = company.cashflow_statement(periods=periods, as_dataframe=True)
            if cf is not None:
                result["cash_flow"] = cf.to_dict(orient="records")
        except Exception:
            logger.warning("Failed to extract cash flow for %s", ticker, exc_info=True)

        return result

    async def get_financials(
        self, ticker: str, periods: int = 5,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(self._get_financials_sync, ticker, periods)
