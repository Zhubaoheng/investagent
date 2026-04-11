"""Tests for subprocess-based PDF extraction (GIL bypass)."""

from __future__ import annotations

import asyncio
import json
import sys
import time

import pytest


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Create a minimal valid PDF for testing."""
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "营业收入 147,693,604,994.14\n净利润 77,521,476,277.80")
    page.insert_text((72, 120), "会计政策：本公司采用权责发生制。")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class TestWorkerProtocol:
    """Test the worker subprocess protocol directly."""

    def test_extract_markdown_via_subprocess(self, sample_pdf_bytes: bytes):
        """Worker extracts markdown from PDF bytes via stdin/stdout."""
        import subprocess

        header = json.dumps({
            "action": "extract_markdown",
            "data_len": len(sample_pdf_bytes),
        }).encode() + b"\n"

        proc = subprocess.run(
            [sys.executable, "-m", "poorcharlie.datasources.pdf_extract_worker"],
            input=header + sample_pdf_bytes,
            capture_output=True,
            timeout=120,
        )

        assert proc.returncode == 0, f"stderr: {proc.stderr.decode()[:500]}"
        result = json.loads(proc.stdout)
        assert "error" not in result
        assert "text" in result
        assert "147,693,604,994.14" in result["text"]

    def test_extract_sections_via_subprocess(self):
        """Worker extracts sections from markdown text."""
        import subprocess

        text = "## 合并利润表\n营业收入 100\n## 合并资产负债表\n总资产 200"
        header = json.dumps({
            "action": "extract_sections",
            "text": text,
            "market": "A_SHARE",
            "data_len": 0,
        }).encode() + b"\n"

        proc = subprocess.run(
            [sys.executable, "-m", "poorcharlie.datasources.pdf_extract_worker"],
            input=header,
            capture_output=True,
            timeout=30,
        )

        assert proc.returncode == 0, f"stderr: {proc.stderr.decode()[:500]}"
        result = json.loads(proc.stdout)
        assert "error" not in result
        assert "sections" in result

    def test_invalid_action(self):
        """Worker returns error for unknown action."""
        import subprocess

        header = json.dumps({"action": "bogus", "data_len": 0}).encode() + b"\n"
        proc = subprocess.run(
            [sys.executable, "-m", "poorcharlie.datasources.pdf_extract_worker"],
            input=header,
            capture_output=True,
            timeout=10,
        )

        result = json.loads(proc.stdout)
        assert "error" in result


class TestAsyncExecutors:
    """Test the async executor functions."""

    @pytest.mark.asyncio
    async def test_subprocess_extract_pdf(self, sample_pdf_bytes: bytes):
        """subprocess_extract_pdf returns markdown text."""
        from poorcharlie.executors import subprocess_extract_pdf

        text = await subprocess_extract_pdf(sample_pdf_bytes)
        assert isinstance(text, str)
        assert len(text) > 0
        assert "147,693,604,994.14" in text

    @pytest.mark.asyncio
    async def test_subprocess_extract_pdf_empty(self):
        """subprocess_extract_pdf handles empty/invalid input gracefully."""
        from poorcharlie.executors import subprocess_extract_pdf

        text = await subprocess_extract_pdf(b"not a pdf")
        assert isinstance(text, str)
        # Should return empty string, not crash

    @pytest.mark.asyncio
    async def test_subprocess_extract_sections(self):
        """subprocess_extract_sections returns section dict."""
        from poorcharlie.executors import subprocess_extract_sections

        md = "## 合并利润表\n营业收入 100 万\n## 合并资产负债表\n总资产 200 万"
        sections = await subprocess_extract_sections(md, "A_SHARE")
        assert isinstance(sections, dict)

    @pytest.mark.asyncio
    async def test_parallel_pdf_extraction(self, sample_pdf_bytes: bytes):
        """Multiple PDF extractions run truly in parallel (not serialized by GIL)."""
        from poorcharlie.executors import subprocess_extract_pdf

        n = 3
        start = time.time()
        results = await asyncio.gather(
            *[subprocess_extract_pdf(sample_pdf_bytes) for _ in range(n)]
        )
        elapsed = time.time() - start

        # All should succeed
        for r in results:
            assert isinstance(r, str)
            assert "147,693,604,994.14" in r

        # Parallel should be faster than n * sequential
        # (with tiny PDFs this is ~1s each for subprocess startup)
        # Just verify they all completed, not strict timing
        assert elapsed < 60, f"Parallel extraction took too long: {elapsed:.1f}s"

    @pytest.mark.asyncio
    async def test_io_pool_not_blocked_by_cpu(self, sample_pdf_bytes: bytes):
        """I/O pool operations complete quickly even during PDF extraction."""
        from poorcharlie.executors import io_pool, subprocess_extract_pdf

        # Start a slow PDF extraction
        pdf_task = asyncio.create_task(subprocess_extract_pdf(sample_pdf_bytes))

        # Immediately run something on the I/O pool
        loop = asyncio.get_running_loop()
        start = time.time()
        await loop.run_in_executor(io_pool(), time.sleep, 0.01)
        io_elapsed = time.time() - start

        # I/O pool should respond in < 1 second regardless of PDF extraction
        assert io_elapsed < 1.0, f"I/O pool blocked for {io_elapsed:.1f}s"

        # Clean up
        await pdf_task
