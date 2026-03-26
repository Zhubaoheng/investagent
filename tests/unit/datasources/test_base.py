"""Tests for datasource base models."""

from datetime import date

from investagent.datasources.base import FilingDocument, MarketQuote


def test_filing_document_creation():
    doc = FilingDocument(
        market="US_ADR",
        ticker="BABA",
        company_name="Alibaba Group",
        filing_type="20-F",
        fiscal_year="2024",
        fiscal_period="FY",
        filing_date=date(2024, 7, 19),
        source_url="https://sec.gov/example",
        content_type="html",
    )
    assert doc.market == "US_ADR"
    assert doc.ticker == "BABA"
    assert doc.raw_content is None
    assert doc.text_content is None
    assert doc.metadata == {}


def test_filing_document_with_content():
    doc = FilingDocument(
        market="A_SHARE",
        ticker="600519",
        company_name="贵州茅台",
        filing_type="年报",
        fiscal_year="2023",
        fiscal_period="FY",
        filing_date=date(2024, 3, 28),
        source_url="https://static.cninfo.com.cn/example.pdf",
        content_type="pdf",
        raw_content=b"%PDF-1.4 fake content",
        metadata={"org_id": "123", "title": "贵州茅台2023年年度报告"},
    )
    assert doc.raw_content == b"%PDF-1.4 fake content"
    assert doc.metadata["org_id"] == "123"


def test_filing_document_frozen():
    doc = FilingDocument(
        market="HK",
        ticker="0700.HK",
        company_name="Tencent",
        filing_type="Annual Report",
        fiscal_year="2023",
        fiscal_period="FY",
        filing_date=date(2024, 3, 20),
        source_url="https://hkexnews.hk/example.pdf",
        content_type="pdf",
    )
    try:
        doc.ticker = "9988.HK"  # type: ignore[misc]
        assert False, "Should have raised"
    except Exception:
        pass


def test_market_quote_creation():
    quote = MarketQuote(
        ticker="BABA",
        name="Alibaba Group",
        currency="USD",
        price=85.50,
        market_cap=210_000_000_000,
        enterprise_value=195_000_000_000,
        pe_ratio=12.5,
        pb_ratio=1.8,
        dividend_yield=0.025,
        shares_outstanding=2_500_000_000,
    )
    assert quote.price == 85.50
    assert quote.currency == "USD"


def test_market_quote_minimal():
    quote = MarketQuote(ticker="0700.HK", name="Tencent", currency="HKD")
    assert quote.price is None
    assert quote.market_cap is None
