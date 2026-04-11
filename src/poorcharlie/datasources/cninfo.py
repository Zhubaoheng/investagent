"""巨潮资讯网 (cninfo.com.cn) data source for A-share filings.

Uses plain ``requests`` — cninfo's AJAX API endpoints have no JavaScript
anti-scraping protection. Only basic headers (Referer, X-Requested-With)
are needed. No scrapling, no V8, no TLS fingerprinting, fully thread-safe.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import date, datetime
from typing import Any

import requests as _requests

from poorcharlie.datasources.base import FilingDocument, FilingFetcher

logger = logging.getLogger(__name__)

# cninfo API endpoints
_SEARCH_URL = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
_STATIC_BASE = "https://static.cninfo.com.cn/"
_COMPANY_SEARCH_URL = "http://www.cninfo.com.cn/new/information/topSearch/query"

# Headers required for cninfo AJAX endpoints
_HEADERS: dict[str, str] = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Content-Type": "application/x-www-form-urlencoded",
    "Referer": "http://www.cninfo.com.cn/",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
}

# Filing category codes on cninfo
_CATEGORY_MAP: dict[str, str] = {
    "年报": "category_ndbg_szsh;",
    "半年报": "category_bndbg_szsh;",
    "一季报": "category_yjdbg_szsh;",
    "三季报": "category_sjdbg_szsh;",
}

_PERIOD_MAP: dict[str, str] = {
    "年报": "FY",
    "半年报": "H1",
    "一季报": "Q1",
    "三季报": "Q3",
}


def _detect_column(ticker: str) -> str:
    """Detect exchange column from ticker prefix."""
    code = re.sub(r"[^\d]", "", ticker.split(".")[0])
    if code.startswith(("6", "9")):
        return "sse"
    elif code.startswith(("0", "3", "2")):
        return "szse"
    elif code.startswith(("4", "8")):
        return "bse"
    return "szse"


class CninfoFetcher(FilingFetcher):
    """Fetch annual, semi-annual, and quarterly reports from cninfo.

    Thread-safe: uses plain requests, no V8/JS engine, no shared session state.
    """

    def __init__(self) -> None:
        self._org_id_cache: dict[str, tuple[str, str]] = {}

    @property
    def market(self) -> str:
        return "A_SHARE"

    def _lookup_org_id(self, ticker: str) -> tuple[str, str]:
        """Look up cninfo orgId for a stock code."""
        code = re.sub(r"[^\d]", "", ticker.split(".")[0])

        resp = _requests.post(
            _COMPANY_SEARCH_URL,
            data={"keyWord": code, "maxSecNum": 10, "maxListNum": 5},
            headers=_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()

        data = resp.json()
        if isinstance(data, list) and data:
            for item in data:
                if item.get("code") == code:
                    return code, item["orgId"]
            return data[0].get("code", code), data[0].get("orgId", "")

        raise ValueError(f"No results found on cninfo for ticker {ticker}")

    def _get_org_id(self, ticker: str) -> tuple[str, str]:
        """Get (stock_code, orgId) with caching."""
        if ticker not in self._org_id_cache:
            self._org_id_cache[ticker] = self._lookup_org_id(ticker)
        return self._org_id_cache[ticker]

    def _search_sync(
        self,
        ticker: str,
        filing_types: list[str] | None = None,
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> list[FilingDocument]:
        code, org_id = self._get_org_id(ticker)
        column = _detect_column(ticker)

        if filing_types is None:
            filing_types = ["年报", "半年报"]

        categories = "".join(
            _CATEGORY_MAP.get(ft, "") for ft in filing_types
        )

        se_date = ""
        if start_year and end_year:
            se_date = f"{start_year}-01-01~{end_year}-12-31"
        elif start_year:
            se_date = f"{start_year}-01-01~"
        elif end_year:
            se_date = f"~{end_year}-12-31"

        results: list[FilingDocument] = []
        page_num = 1
        max_pages = 5

        while page_num <= max_pages:
            try:
                resp = _requests.post(
                    _SEARCH_URL,
                    data={
                        "stock": f"{code},{org_id}",
                        "tabName": "fulltext",
                        "column": column,
                        "category": categories,
                        "pageNum": str(page_num),
                        "pageSize": "30",
                        "seDate": se_date,
                        "sortName": "",
                        "sortType": "",
                        "isHLtitle": "true",
                    },
                    headers=_HEADERS,
                    timeout=15,
                )

                if resp.status_code != 200:
                    logger.warning("cninfo search returned status %d", resp.status_code)
                    break

                try:
                    data = resp.json()
                except (json.JSONDecodeError, ValueError):
                    logger.warning("cninfo returned invalid JSON on page %d", page_num)
                    break

                announcements = data.get("announcements", [])
                if not announcements:
                    break

                for ann in announcements:
                    adjunct_url = ann.get("adjunctUrl", "")
                    title = ann.get("announcementTitle", "").replace("<em>", "").replace("</em>", "")
                    ann_date = ann.get("announcementTime")

                    if not adjunct_url:
                        continue

                    # Skip summaries and English versions
                    if "摘要" in title:
                        continue
                    if "英文" in title or "英文版" in title:
                        continue

                    # Parse date
                    if isinstance(ann_date, (int, float)):
                        fd = datetime.fromtimestamp(ann_date / 1000).date()
                    elif isinstance(ann_date, str):
                        fd = date.fromisoformat(ann_date[:10])
                    else:
                        fd = date.today()

                    # Determine filing type
                    if "半年" in title:
                        detected_type = "半年报"
                    elif "三季" in title or "第三季" in title:
                        detected_type = "三季报"
                    elif "一季" in title or "第一季" in title:
                        detected_type = "一季报"
                    elif "年报" in title or "年度报告" in title:
                        detected_type = "年报"
                    else:
                        detected_type = "年报"

                    fy_match = re.search(r"(\d{4})\s*年", title)
                    fiscal_year = fy_match.group(1) if fy_match else str(fd.year)

                    source_url = f"{_STATIC_BASE}{adjunct_url}"

                    results.append(
                        FilingDocument(
                            market="A_SHARE",
                            ticker=ticker,
                            company_name=ann.get("secName", ticker),
                            filing_type=detected_type,
                            fiscal_year=fiscal_year,
                            fiscal_period=_PERIOD_MAP.get(detected_type, "FY"),
                            filing_date=fd,
                            source_url=source_url,
                            content_type="pdf",
                            metadata={
                                "org_id": org_id,
                                "announcement_id": str(ann.get("announcementId", "")),
                                "title": title,
                            },
                        )
                    )

                total = data.get("totalAnnouncement", 0)
                if page_num * 30 >= total:
                    break
                page_num += 1

            except Exception:
                logger.warning(
                    "Failed to search cninfo page %d for %s",
                    page_num, ticker, exc_info=True,
                )
                break

        return results

    def _download_sync(self, filing: FilingDocument) -> FilingDocument:
        resp = _requests.get(
            filing.source_url,
            headers={"Referer": "http://www.cninfo.com.cn/"},
            timeout=60,
        )
        resp.raise_for_status()

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
            raw_content=resp.content,
            text_content=None,
            metadata=filing.metadata,
        )

    # ------------------------------------------------------------------
    # Async interface
    # ------------------------------------------------------------------

    async def search_filings(
        self,
        ticker: str,
        filing_types: list[str] | None = None,
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> list[FilingDocument]:
        from poorcharlie.executors import io_pool
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            io_pool(), self._search_sync, ticker, filing_types, start_year, end_year,
        )

    async def download_filing(self, filing: FilingDocument) -> FilingDocument:
        from poorcharlie.executors import io_pool
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(io_pool(), self._download_sync, filing)
