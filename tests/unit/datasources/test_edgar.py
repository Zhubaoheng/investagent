"""Tests for SEC EDGAR fetcher — all external calls mocked."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from investagent.datasources.base import FilingDocument
from investagent.datasources.edgar import EdgarFetcher, _filing_to_document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_filing(
    form: str = "20-F",
    filing_date: str = "2024-07-19",
    period_of_report: str = "2024-03-31",
    company: str = "Alibaba Group",
    homepage_url: str = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany",
    accession_no: str = "0001014473-24-012345",
) -> SimpleNamespace:
    return SimpleNamespace(
        form=form,
        filing_date=filing_date,
        period_of_report=period_of_report,
        company=company,
        homepage_url=homepage_url,
        accession_no=accession_no,
    )


# ---------------------------------------------------------------------------
# _filing_to_document
# ---------------------------------------------------------------------------

def test_filing_to_document_20f():
    mock = _make_mock_filing()
    doc = _filing_to_document(mock, "BABA")
    assert doc.market == "US_ADR"
    assert doc.ticker == "BABA"
    assert doc.filing_type == "20-F"
    assert doc.fiscal_year == "2024"
    assert doc.fiscal_period == "FY"
    assert doc.filing_date == date(2024, 7, 19)
    assert doc.metadata["accession_no"] == "0001014473-24-012345"


def test_filing_to_document_6k():
    mock = _make_mock_filing(form="6-K", period_of_report="2024-09-30")
    doc = _filing_to_document(mock, "JD")
    assert doc.filing_type == "6-K"
    assert doc.fiscal_period == "H1"


def test_filing_to_document_date_object():
    mock = _make_mock_filing()
    mock.filing_date = date(2024, 7, 19)
    mock.period_of_report = date(2024, 3, 31)
    doc = _filing_to_document(mock, "BABA")
    assert doc.filing_date == date(2024, 7, 19)
    assert doc.fiscal_year == "2024"


# ---------------------------------------------------------------------------
# EdgarFetcher.search_filings
# ---------------------------------------------------------------------------

@pytest.fixture
def fetcher():
    return EdgarFetcher(identity="Test test@test.com")


def _mock_company_with_filings(mock_filings: list) -> MagicMock:
    company = MagicMock()
    company.get_filings.return_value = mock_filings
    return company


@patch("investagent.datasources.edgar._ensure_identity")
@patch("edgar.Company")
async def test_search_filings_returns_documents(mock_company_cls, mock_identity, fetcher):
    mock_filings = [
        _make_mock_filing(form="20-F", filing_date="2024-07-19"),
        _make_mock_filing(form="20-F", filing_date="2023-07-20"),
    ]
    mock_company_cls.return_value = _mock_company_with_filings(mock_filings)

    results = await fetcher.search_filings("BABA", filing_types=["20-F"])

    assert len(results) == 2
    assert all(isinstance(r, FilingDocument) for r in results)
    assert results[0].filing_type == "20-F"
    mock_company_cls.return_value.get_filings.assert_called_once()


@patch("investagent.datasources.edgar._ensure_identity")
@patch("edgar.Company")
async def test_search_filings_with_year_range(mock_company_cls, mock_identity, fetcher):
    mock_company_cls.return_value = _mock_company_with_filings([
        _make_mock_filing(filing_date="2023-07-20"),
    ])

    results = await fetcher.search_filings(
        "BABA", filing_types=["20-F"], start_year=2020, end_year=2024,
    )

    call_kwargs = mock_company_cls.return_value.get_filings.call_args[1]
    assert "filing_date" in call_kwargs
    assert "2020-01-01" in call_kwargs["filing_date"]
    assert "2024-12-31" in call_kwargs["filing_date"]


@patch("investagent.datasources.edgar._ensure_identity")
@patch("edgar.Company")
async def test_search_filings_default_types(mock_company_cls, mock_identity, fetcher):
    mock_company_cls.return_value = _mock_company_with_filings([])

    await fetcher.search_filings("BABA")

    call_kwargs = mock_company_cls.return_value.get_filings.call_args[1]
    assert call_kwargs["form"] == ["20-F", "6-K"]


@patch("investagent.datasources.edgar._ensure_identity")
@patch("edgar.Company")
async def test_search_filings_skips_bad_filings(mock_company_cls, mock_identity, fetcher):
    """A filing that raises during parsing should be skipped, not crash."""
    good = _make_mock_filing()
    bad = SimpleNamespace(form="20-F", filing_date="not-a-date")  # will fail

    mock_company_cls.return_value = _mock_company_with_filings([good, bad])

    results = await fetcher.search_filings("BABA")
    # At least the good one should come through
    assert len(results) >= 1


# ---------------------------------------------------------------------------
# EdgarFetcher.download_filing
# ---------------------------------------------------------------------------

@patch("investagent.datasources.edgar._ensure_identity")
@patch("edgar.Company")
async def test_download_filing(mock_company_cls, mock_identity, fetcher):
    mock_edgar_filing = MagicMock()
    mock_edgar_filing.html.return_value = "<html>20-F content</html>"
    mock_edgar_filing.text.return_value = "20-F text content"

    mock_company_cls.return_value.get_filings.return_value = [mock_edgar_filing]

    filing = FilingDocument(
        market="US_ADR",
        ticker="BABA",
        company_name="Alibaba",
        filing_type="20-F",
        fiscal_year="2024",
        fiscal_period="FY",
        filing_date=date(2024, 7, 19),
        source_url="https://sec.gov/example",
        content_type="html",
        metadata={"accession_no": "0001014473-24-012345", "period_of_report": "2024-03-31"},
    )

    result = await fetcher.download_filing(filing)

    assert result.raw_content == b"<html>20-F content</html>"
    assert result.text_content == "20-F text content"
    assert result.ticker == "BABA"


@patch("investagent.datasources.edgar._ensure_identity")
@patch("edgar.Company")
async def test_download_filing_no_accession_raises(mock_company_cls, mock_identity, fetcher):
    filing = FilingDocument(
        market="US_ADR",
        ticker="BABA",
        company_name="Alibaba",
        filing_type="20-F",
        fiscal_year="2024",
        fiscal_period="FY",
        filing_date=date(2024, 7, 19),
        source_url="https://sec.gov/example",
        content_type="html",
        metadata={},  # No accession_no
    )

    with pytest.raises(ValueError, match="No accession_no"):
        await fetcher.download_filing(filing)


# ---------------------------------------------------------------------------
# EdgarFetcher.get_financials
# ---------------------------------------------------------------------------

@patch("investagent.datasources.edgar._ensure_identity")
@patch("edgar.Company")
async def test_get_financials(mock_company_cls, mock_identity, fetcher):
    mock_df = MagicMock()
    mock_df.to_dict.return_value = [{"revenue": 100}]

    company = mock_company_cls.return_value
    company.income_statement.return_value = mock_df
    company.balance_sheet.return_value = mock_df
    company.cashflow_statement.return_value = mock_df

    result = await fetcher.get_financials("BABA", periods=3)

    assert "income_statement" in result
    assert "balance_sheet" in result
    assert "cash_flow" in result
    company.income_statement.assert_called_once_with(periods=3, as_dataframe=True)


@patch("investagent.datasources.edgar._ensure_identity")
@patch("edgar.Company")
async def test_get_financials_handles_failure(mock_company_cls, mock_identity, fetcher):
    company = mock_company_cls.return_value
    company.income_statement.side_effect = Exception("XBRL parse error")
    company.balance_sheet.return_value = None
    company.cashflow_statement.return_value = None

    result = await fetcher.get_financials("BABA")
    # Should not raise — returns empty lists
    assert result["income_statement"] == []
    assert result["balance_sheet"] == []
    assert result["cash_flow"] == []
