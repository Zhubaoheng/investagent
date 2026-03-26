"""Tests for cninfo fetcher — all external calls mocked."""

from __future__ import annotations

import json
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from investagent.datasources.base import FilingDocument
from investagent.datasources.cninfo import CninfoFetcher, _detect_column


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_detect_column_shanghai():
    assert _detect_column("600519") == "sse"
    assert _detect_column("600519.SH") == "sse"
    assert _detect_column("900001") == "sse"


def test_detect_column_shenzhen():
    assert _detect_column("000858") == "szse"
    assert _detect_column("300750") == "szse"
    assert _detect_column("002594.SZ") == "szse"


def test_detect_column_beijing():
    assert _detect_column("430047") == "bse"
    assert _detect_column("830001") == "bse"


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _make_mock_response(status: int = 200, body: bytes = b""):
    page = MagicMock()
    page.status = status
    page.body = body
    return page


def _org_id_body(code: str = "600519", org_id: str = "gssh0600519") -> bytes:
    return json.dumps([{"code": code, "orgId": org_id, "zwjc": "贵州茅台"}]).encode()


def _search_body(
    announcements: list[dict] | None = None,
    total: int = 1,
) -> bytes:
    if announcements is None:
        announcements = [
            {
                "adjunctUrl": "finalpage/2024-03-28/1234567.PDF",
                "announcementTitle": "贵州茅台酒股份有限公司2023年<em>年报</em>",
                "announcementTime": 1711584000000,
                "secName": "贵州茅台",
                "announcementId": "123456",
            }
        ]
    return json.dumps({
        "announcements": announcements,
        "totalAnnouncement": total,
    }).encode()


# ---------------------------------------------------------------------------
# CninfoFetcher.search_filings
# ---------------------------------------------------------------------------

@pytest.fixture
def fetcher():
    return CninfoFetcher()


@patch("investagent.datasources.cninfo.FetcherSession")
async def test_search_filings_annual(mock_session_cls, fetcher):
    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_session_cls.return_value = mock_session

    # get() for homepage, post() for org_id lookup + search
    mock_session.get.return_value = _make_mock_response(200, b"<html>home</html>")
    mock_session.post.side_effect = [
        _make_mock_response(200, _org_id_body()),
        _make_mock_response(200, _search_body()),
    ]

    results = await fetcher.search_filings("600519", filing_types=["年报"])

    assert len(results) == 1
    doc = results[0]
    assert doc.market == "A_SHARE"
    assert doc.ticker == "600519"
    assert doc.filing_type == "年报"
    assert doc.fiscal_period == "FY"
    assert doc.content_type == "pdf"
    assert "static.cninfo.com.cn" in doc.source_url
    assert doc.metadata["title"] == "贵州茅台酒股份有限公司2023年年报"


@patch("investagent.datasources.cninfo.FetcherSession")
async def test_search_filings_multiple_types(mock_session_cls, fetcher):
    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_session_cls.return_value = mock_session

    mock_session.get.return_value = _make_mock_response(200, b"<html>home</html>")
    mock_session.post.side_effect = [
        _make_mock_response(200, _org_id_body()),
        _make_mock_response(200, _search_body(
            announcements=[
                {
                    "adjunctUrl": "finalpage/2024-03-28/001.PDF",
                    "announcementTitle": "2023年年报",
                    "announcementTime": 1711584000000,
                    "secName": "贵州茅台",
                    "announcementId": "1",
                },
                {
                    "adjunctUrl": "finalpage/2024-08-30/002.PDF",
                    "announcementTitle": "2024年半年报",
                    "announcementTime": 1724976000000,
                    "secName": "贵州茅台",
                    "announcementId": "2",
                },
            ],
            total=2,
        )),
    ]

    results = await fetcher.search_filings("600519", filing_types=["年报", "半年报"])
    assert len(results) == 2
    types = {r.filing_type for r in results}
    assert "年报" in types
    assert "半年报" in types


@patch("investagent.datasources.cninfo.FetcherSession")
async def test_search_filings_empty(mock_session_cls, fetcher):
    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_session_cls.return_value = mock_session

    mock_session.get.return_value = _make_mock_response(200, b"home")
    mock_session.post.side_effect = [
        _make_mock_response(200, _org_id_body()),
        _make_mock_response(200, _search_body(announcements=[], total=0)),
    ]

    results = await fetcher.search_filings("600519")
    assert results == []


@patch("investagent.datasources.cninfo.FetcherSession")
async def test_search_filings_with_year_range(mock_session_cls, fetcher):
    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_session_cls.return_value = mock_session

    mock_session.get.return_value = _make_mock_response(200, b"home")
    mock_session.post.side_effect = [
        _make_mock_response(200, _org_id_body()),
        _make_mock_response(200, _search_body(announcements=[], total=0)),
    ]

    await fetcher.search_filings("600519", start_year=2020, end_year=2024)

    # Second post call is the search call
    search_call = mock_session.post.call_args_list[1]
    data = search_call[1]["data"] if "data" in search_call[1] else search_call[0][1] if len(search_call[0]) > 1 else {}
    assert "2020-01-01" in data.get("seDate", "")
    assert "2024-12-31" in data.get("seDate", "")


@patch("investagent.datasources.cninfo.FetcherSession")
async def test_search_filings_org_id_cached(mock_session_cls, fetcher):
    """Second search for same ticker should not re-lookup orgId."""
    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_session_cls.return_value = mock_session

    mock_session.get.return_value = _make_mock_response(200, b"home")

    # First search: orgId lookup + search
    # Second search: only search (orgId cached)
    mock_session.post.side_effect = [
        _make_mock_response(200, _org_id_body()),
        _make_mock_response(200, _search_body(announcements=[], total=0)),
        # Second call — no orgId lookup needed
        _make_mock_response(200, _search_body(announcements=[], total=0)),
    ]

    await fetcher.search_filings("600519")
    await fetcher.search_filings("600519")

    # 3 post calls: 1 orgId + 2 searches
    assert mock_session.post.call_count == 3


@patch("investagent.datasources.cninfo.FetcherSession")
async def test_search_filings_org_id_failure(mock_session_cls, fetcher):
    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_session_cls.return_value = mock_session

    mock_session.get.return_value = _make_mock_response(200, b"home")
    mock_session.post.return_value = _make_mock_response(403)

    with pytest.raises(ValueError, match="failed with status"):
        await fetcher.search_filings("600519")


# ---------------------------------------------------------------------------
# CninfoFetcher.download_filing
# ---------------------------------------------------------------------------

@patch("investagent.datasources.cninfo.Fetcher")
async def test_download_filing_pdf(mock_fetcher_cls, fetcher):
    mock_page = MagicMock()
    mock_page.status = 200
    mock_page.body = b"%PDF-1.4 fake"
    mock_fetcher_cls.get.return_value = mock_page

    filing = FilingDocument(
        market="A_SHARE",
        ticker="600519",
        company_name="贵州茅台",
        filing_type="年报",
        fiscal_year="2023",
        fiscal_period="FY",
        filing_date=date(2024, 3, 28),
        source_url="https://static.cninfo.com.cn/finalpage/2024-03-28/1234567.PDF",
        content_type="pdf",
        metadata={"org_id": "gssh0600519"},
    )

    result = await fetcher.download_filing(filing)
    assert result.raw_content == b"%PDF-1.4 fake"
    assert result.text_content is None


@patch("investagent.datasources.cninfo.Fetcher")
async def test_download_filing_http_error(mock_fetcher_cls, fetcher):
    mock_page = MagicMock()
    mock_page.status = 403
    mock_fetcher_cls.get.return_value = mock_page

    filing = FilingDocument(
        market="A_SHARE",
        ticker="600519",
        company_name="贵州茅台",
        filing_type="年报",
        fiscal_year="2023",
        fiscal_period="FY",
        filing_date=date(2024, 3, 28),
        source_url="https://static.cninfo.com.cn/example.PDF",
        content_type="pdf",
    )

    with pytest.raises(ValueError, match="status 403"):
        await fetcher.download_filing(filing)
