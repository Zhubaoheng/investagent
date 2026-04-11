"""Shared persistent cache for filings, markdown, sections, and AkShare data.

Filing cache stores:
- Raw PDF/HTML bytes (keyed by market/ticker/period_year)
- Extracted markdown text (deterministic from PDF)
- Extracted sections (deterministic from markdown + market)

AkShare cache stores structured financial data with TTL.

All writes are atomic (write-to-temp + os.rename) to prevent corruption.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from poorcharlie.datasources.base import FilingDocument

logger = logging.getLogger(__name__)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes atomically: temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        os.write(fd, data)
        os.close(fd)
        os.rename(tmp, path)
    except Exception:
        os.close(fd) if not os.get_inheritable(fd) else None
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically: temp file + rename."""
    _atomic_write_bytes(path, text.encode("utf-8"))


class FilingCache:
    """Shared cache for filing PDFs, markdown, and sections.

    Directory layout:
        {cache_root}/{market}/{ticker}/{period}_{year}.pdf
        {cache_root}/{market}/{ticker}/{period}_{year}.md
        {cache_root}/{market}/{ticker}/{period}_{year}.sections.json
        {cache_root}/{market}/{ticker}/_manifest.json
    """

    def __init__(self, cache_root: Path) -> None:
        self._root = cache_root

    def _filing_dir(self, doc: FilingDocument) -> Path:
        return self._root / doc.market / doc.ticker

    def _filing_stem(self, doc: FilingDocument) -> str:
        return f"{doc.fiscal_period}_{doc.fiscal_year}"

    # ------------------------------------------------------------------
    # PDF raw content
    # ------------------------------------------------------------------

    def get_pdf(self, doc: FilingDocument) -> bytes | None:
        ext = "html" if doc.content_type == "html" else "pdf"
        path = self._filing_dir(doc) / f"{self._filing_stem(doc)}.{ext}"
        if path.exists():
            logger.debug("Cache hit (PDF): %s", path)
            return path.read_bytes()
        return None

    def put_pdf(self, doc: FilingDocument, content: bytes) -> None:
        ext = "html" if doc.content_type == "html" else "pdf"
        path = self._filing_dir(doc) / f"{self._filing_stem(doc)}.{ext}"
        _atomic_write_bytes(path, content)
        self._update_manifest(doc, content)
        logger.debug("Cache store (PDF): %s (%d bytes)", path, len(content))

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------

    def get_markdown(self, doc: FilingDocument) -> str | None:
        path = self._filing_dir(doc) / f"{self._filing_stem(doc)}.md"
        if path.exists():
            logger.debug("Cache hit (markdown): %s", path)
            return path.read_text(encoding="utf-8")
        return None

    def put_markdown(self, doc: FilingDocument, text: str) -> None:
        path = self._filing_dir(doc) / f"{self._filing_stem(doc)}.md"
        _atomic_write_text(path, text)
        logger.debug("Cache store (markdown): %s", path)

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------

    def get_sections(self, doc: FilingDocument) -> dict[str, str] | None:
        path = self._filing_dir(doc) / f"{self._filing_stem(doc)}.sections.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                logger.debug("Cache hit (sections): %s", path)
                return data
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def put_sections(self, doc: FilingDocument, sections: dict[str, str]) -> None:
        path = self._filing_dir(doc) / f"{self._filing_stem(doc)}.sections.json"
        _atomic_write_text(path, json.dumps(sections, ensure_ascii=False, indent=2))
        logger.debug("Cache store (sections): %s", path)

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def _update_manifest(self, doc: FilingDocument, content: bytes) -> None:
        manifest_path = self._filing_dir(doc) / "_manifest.json"
        manifest: dict = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass

        key = self._filing_stem(doc)
        manifest[key] = {
            "source_url": doc.source_url,
            "filing_date": doc.filing_date.isoformat(),
            "content_type": doc.content_type,
            "sha256": hashlib.sha256(content).hexdigest(),
            "cached_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        _atomic_write_text(
            manifest_path,
            json.dumps(manifest, ensure_ascii=False, indent=2),
        )


class AkShareCache:
    """Cache for AkShare structured financial data with TTL.

    Directory layout:
        {cache_root}/{market}/{ticker}.json

    Each file contains:
        {"fetched_at": "...", "data": {...}}
    """

    def __init__(self, cache_root: Path, max_age_days: int = 30) -> None:
        self._root = cache_root
        self._max_age = timedelta(days=max_age_days)

    def _cache_path(self, ticker: str, market: str) -> Path:
        return self._root / market / f"{ticker}.json"

    def get(self, ticker: str, market: str) -> dict | None:
        path = self._cache_path(ticker, market)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            fetched_at = datetime.fromisoformat(raw["fetched_at"])
            if datetime.now(tz=timezone.utc) - fetched_at > self._max_age:
                logger.debug("Cache expired (AkShare): %s", path)
                return None
            logger.debug("Cache hit (AkShare): %s", path)
            return raw["data"]
        except (json.JSONDecodeError, KeyError, OSError, ValueError):
            return None

    def put(self, ticker: str, market: str, data: dict) -> None:
        path = self._cache_path(ticker, market)
        payload = {
            "fetched_at": datetime.now(tz=timezone.utc).isoformat(),
            "data": data,
        }
        _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))
        logger.debug("Cache store (AkShare): %s", path)
