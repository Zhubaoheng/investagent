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

def _make_response(status_code: int = 200, json_data: Any = None, content: bytes = b""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = content
    resp.text = content.decode("utf-8", errors="replace") if content else ""
    if json_data is not None:
        resp.json.return_value = json_data
        resp.text = json.dumps(json_data)
    else:
        resp.json.side_effect = ValueError("No JSON")
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        from requests.exceptions import HTTPError
        resp.raise_for_status.side_effect = HTTPError(response=resp)
    return resp


from typing import Any


def _org_id_json(code: str = "600519", org_id: str = "gssh0600519") -> list:
    return [{"code": code, "orgId": org_id, "zwjc": "贵州茅台"}]


def _search_json(announcements: list[dict] | None = None, total: int = 1) -> dict:
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
    return {"announcements": announcements, "totalAnnouncement": total}


# ---------------------------------------------------------------------------
# CninfoFetcher.search_filings
# ---------------------------------------------------------------------------

@pytest.fixture
def fetcher():
    return CninfoFetcher()


@patch("investagent.datasources.cninfo._requests")
async def test_search_filings_annual(mock_req, fetcher):
    mock_req.post.side_effect = [
        _make_response(200, json_data=_org_id_json()),
        _make_response(200, json_data=_search_json()),
    ]

    results = await fetcher.search_filings("600519", filing_types=["年报"])

    assert len(results) == 1
    doc = results[0]
    assert doc.market == "A_SHARE"
    assert doc.ticker == "600519"
    assert doc.filing_type == "年报"
    assert doc.fiscal_period == "FY"
    assert "static.cninfo.com.cn" in doc.source_url
    assert doc.metadata["title"] == "贵州茅台酒股份有限公司2023年年报"


@patch("investagent.datasources.cninfo._requests")
async def test_search_filings_multiple_types(mock_req, fetcher):
    mock_req.post.side_effect = [
        _make_response(200, json_data=_org_id_json()),
        _make_response(200, json_data=_search_json(
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


@patch("investagent.datasources.cninfo._requests")
async def test_search_filings_empty(mock_req, fetcher):
    mock_req.post.side_effect = [
        _make_response(200, json_data=_org_id_json()),
        _make_response(200, json_data=_search_json(announcements=[], total=0)),
    ]

    results = await fetcher.search_filings("600519")
    assert results == []


@patch("investagent.datasources.cninfo._requests")
async def test_download_filing_pdf(mock_req, fetcher):
    mock_req.get.return_value = _make_response(200, content=b"%PDF-1.4 fake")

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
