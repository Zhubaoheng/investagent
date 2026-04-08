"""Tests for investagent.datasources.cache."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from investagent.datasources.base import FilingDocument
from investagent.datasources.cache import AkShareCache, FilingCache


def _make_doc(**overrides) -> FilingDocument:
    defaults = {
        "market": "A_SHARE",
        "ticker": "600519",
        "company_name": "贵州茅台",
        "filing_type": "年报",
        "fiscal_year": "2023",
        "fiscal_period": "FY",
        "filing_date": date(2024, 3, 28),
        "source_url": "https://static.cninfo.com.cn/example.pdf",
        "content_type": "pdf",
    }
    defaults.update(overrides)
    return FilingDocument(**defaults)


class TestFilingCache:
    def test_pdf_miss_returns_none(self, tmp_path: Path):
        cache = FilingCache(tmp_path)
        doc = _make_doc()
        assert cache.get_pdf(doc) is None

    def test_pdf_round_trip(self, tmp_path: Path):
        cache = FilingCache(tmp_path)
        doc = _make_doc()
        content = b"fake pdf bytes"
        cache.put_pdf(doc, content)
        assert cache.get_pdf(doc) == content

    def test_pdf_html_content_type(self, tmp_path: Path):
        cache = FilingCache(tmp_path)
        doc = _make_doc(content_type="html")
        content = b"<html>fake</html>"
        cache.put_pdf(doc, content)
        assert cache.get_pdf(doc) == content
        # Check file extension is .html
        path = tmp_path / "A_SHARE" / "600519" / "FY_2023.html"
        assert path.exists()

    def test_markdown_round_trip(self, tmp_path: Path):
        cache = FilingCache(tmp_path)
        doc = _make_doc()
        text = "# 年度报告\n\n营业收入 1000亿"
        cache.put_markdown(doc, text)
        assert cache.get_markdown(doc) == text

    def test_markdown_miss_returns_none(self, tmp_path: Path):
        cache = FilingCache(tmp_path)
        doc = _make_doc()
        assert cache.get_markdown(doc) is None

    def test_sections_round_trip(self, tmp_path: Path):
        cache = FilingCache(tmp_path)
        doc = _make_doc()
        sections = {"mda": "管理层讨论", "income_statement": "利润表"}
        cache.put_sections(doc, sections)
        result = cache.get_sections(doc)
        assert result == sections

    def test_sections_miss_returns_none(self, tmp_path: Path):
        cache = FilingCache(tmp_path)
        doc = _make_doc()
        assert cache.get_sections(doc) is None

    def test_manifest_created_on_put_pdf(self, tmp_path: Path):
        cache = FilingCache(tmp_path)
        doc = _make_doc()
        cache.put_pdf(doc, b"content")
        manifest_path = tmp_path / "A_SHARE" / "600519" / "_manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert "FY_2023" in manifest
        assert manifest["FY_2023"]["source_url"] == doc.source_url
        assert "sha256" in manifest["FY_2023"]

    def test_different_tickers_isolated(self, tmp_path: Path):
        cache = FilingCache(tmp_path)
        doc1 = _make_doc(ticker="600519")
        doc2 = _make_doc(ticker="000858")
        cache.put_pdf(doc1, b"maotai")
        cache.put_pdf(doc2, b"wuliangye")
        assert cache.get_pdf(doc1) == b"maotai"
        assert cache.get_pdf(doc2) == b"wuliangye"

    def test_different_years_isolated(self, tmp_path: Path):
        cache = FilingCache(tmp_path)
        doc2023 = _make_doc(fiscal_year="2023")
        doc2022 = _make_doc(fiscal_year="2022")
        cache.put_markdown(doc2023, "2023 report")
        cache.put_markdown(doc2022, "2022 report")
        assert cache.get_markdown(doc2023) == "2023 report"
        assert cache.get_markdown(doc2022) == "2022 report"

    def test_hk_market_directory(self, tmp_path: Path):
        cache = FilingCache(tmp_path)
        doc = _make_doc(market="HK", ticker="00700")
        cache.put_pdf(doc, b"tencent")
        assert (tmp_path / "HK" / "00700" / "FY_2023.pdf").exists()


class TestAkShareCache:
    def test_miss_returns_none(self, tmp_path: Path):
        cache = AkShareCache(tmp_path)
        assert cache.get("600519", "A_SHARE") is None

    def test_round_trip(self, tmp_path: Path):
        cache = AkShareCache(tmp_path)
        data = {"income_statement": [{"revenue": 1000}], "source": "akshare_sina"}
        cache.put("600519", "A_SHARE", data)
        result = cache.get("600519", "A_SHARE")
        assert result == data

    def test_ttl_expiry(self, tmp_path: Path):
        cache = AkShareCache(tmp_path, max_age_days=0)  # immediately expire
        data = {"income_statement": []}
        cache.put("600519", "A_SHARE", data)

        # Manually backdate the fetched_at
        path = tmp_path / "A_SHARE" / "600519.json"
        raw = json.loads(path.read_text())
        old_time = (datetime.now(tz=timezone.utc) - timedelta(days=1)).isoformat()
        raw["fetched_at"] = old_time
        path.write_text(json.dumps(raw))

        assert cache.get("600519", "A_SHARE") is None

    def test_fresh_data_not_expired(self, tmp_path: Path):
        cache = AkShareCache(tmp_path, max_age_days=30)
        data = {"income_statement": []}
        cache.put("600519", "A_SHARE", data)
        assert cache.get("600519", "A_SHARE") == data

    def test_different_markets_isolated(self, tmp_path: Path):
        cache = AkShareCache(tmp_path)
        cache.put("600519", "A_SHARE", {"market": "A"})
        cache.put("00700", "HK", {"market": "HK"})
        assert cache.get("600519", "A_SHARE") == {"market": "A"}
        assert cache.get("00700", "HK") == {"market": "HK"}
