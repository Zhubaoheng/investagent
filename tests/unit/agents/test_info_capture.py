"""Tests for poorcharlie.agents.info_capture — hybrid agent with real fetchers."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest

from poorcharlie.agents.base import AgentOutputError
from poorcharlie.agents.info_capture import InfoCaptureAgent
from poorcharlie.datasources.base import FilingDocument, MarketQuote
from poorcharlie.llm import LLMClient
from poorcharlie.schemas.company import CompanyIntake


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _intake() -> CompanyIntake:
    return CompanyIntake(
        ticker="600519", name="贵州茅台", exchange="SSE", sector="白酒", notes="龙头",
    )


def _mock_llm() -> LLMClient:
    return LLMClient(client=MagicMock())


def _mock_response(tool_input: dict) -> MagicMock:
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.input = tool_input
    response = MagicMock()
    response.content = [tool_block]
    response.model = "claude-sonnet-4-20250514"
    response.usage = MagicMock()
    response.usage.input_tokens = 100
    response.usage.output_tokens = 200
    return response


def _mock_filing_fetcher(filings: list[FilingDocument] | None = None) -> MagicMock:
    fetcher = MagicMock()
    fetcher.market = "A_SHARE"
    if filings is None:
        filings = [
            FilingDocument(
                market="A_SHARE",
                ticker="600519",
                company_name="贵州茅台",
                filing_type="年报",
                fiscal_year="2023",
                fiscal_period="FY",
                filing_date=date(2024, 3, 28),
                source_url="https://static.cninfo.com.cn/2023.PDF",
                content_type="pdf",
            ),
            FilingDocument(
                market="A_SHARE",
                ticker="600519",
                company_name="贵州茅台",
                filing_type="年报",
                fiscal_year="2022",
                fiscal_period="FY",
                filing_date=date(2023, 3, 30),
                source_url="https://static.cninfo.com.cn/2022.PDF",
                content_type="pdf",
            ),
        ]
    fetcher.search_filings = AsyncMock(return_value=filings)
    return fetcher


def _mock_market_fetcher(quote: MarketQuote | None = None) -> MagicMock:
    fetcher = MagicMock()
    if quote is None:
        quote = MarketQuote(
            ticker="600519.SS",
            name="KWEICHOW MOUTAI",
            currency="CNY",
            price=1680.0,
            market_cap=2_110_000_000_000.0,
            enterprise_value=2_050_000_000_000.0,
            pe_ratio=25.0,
            pb_ratio=8.0,
        )
    fetcher.get_quote = AsyncMock(return_value=quote)
    return fetcher


def _llm_tool_input() -> dict:
    """LLM output — only the fields the LLM is responsible for."""
    return {
        "company_profile": {
            "full_name": "贵州茅台酒股份有限公司",
            "listing": "SSE:600519",
            "main_business": "高端白酒生产与销售",
        },
        # These will be overridden by fetcher data, but LLM may still emit them
        "filing_manifest": [],
        "market_snapshot": {},
        "official_sources": [
            "贵州茅台官方投资者关系页面",
            "上交所公告系统",
        ],
        "trusted_third_party_sources": [
            "Wind金融终端",
            "东方财富Choice",
        ],
        "missing_items": ["2019年年报：数据源回溯范围不足"],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_info_capture_output():
    """Full flow: fetchers provide filings + market data, LLM provides profile."""
    llm = _mock_llm()
    llm.create_message = AsyncMock(
        return_value=_mock_response(_llm_tool_input())
    )
    agent = InfoCaptureAgent(
        llm,
        filing_fetcher=_mock_filing_fetcher(),
        market_fetcher=_mock_market_fetcher(),
    )
    result = await agent.run(_intake())

    assert result.meta.agent_name == "info_capture"
    # Filing manifest comes from fetcher (2 filings), not LLM
    assert len(result.filing_manifest) == 2
    assert result.filing_manifest[0].filing_type == "年报"
    assert result.filing_manifest[0].fiscal_year == "2023"
    assert "cninfo.com.cn" in result.filing_manifest[0].source_url
    # Market snapshot comes from fetcher, not LLM
    assert result.market_snapshot.price == 1680.0
    assert result.market_snapshot.currency == "CNY"
    # LLM-generated fields
    assert "full_name" in result.company_profile
    assert len(result.official_sources) > 0
    assert result.meta.token_usage == 300


async def test_info_capture_meta_is_server_generated():
    tool_input = _llm_tool_input()
    tool_input["meta"] = {
        "agent_name": "hacked",
        "timestamp": "2020-01-01T00:00:00Z",
        "model_used": "fake",
        "token_usage": 0,
    }
    llm = _mock_llm()
    llm.create_message = AsyncMock(
        return_value=_mock_response(tool_input)
    )
    agent = InfoCaptureAgent(
        llm,
        filing_fetcher=_mock_filing_fetcher(),
        market_fetcher=_mock_market_fetcher(),
    )
    result = await agent.run(_intake())
    assert result.meta.agent_name == "info_capture"
    assert result.meta.model_used == "claude-sonnet-4-20250514"


async def test_info_capture_no_tool_use_raises():
    text_block = MagicMock()
    text_block.type = "text"
    response = MagicMock()
    response.content = [text_block]
    response.model = "claude-sonnet-4-20250514"
    response.usage = MagicMock(input_tokens=50, output_tokens=100)

    llm = _mock_llm()
    llm.create_message = AsyncMock(return_value=response)
    agent = InfoCaptureAgent(
        llm,
        filing_fetcher=_mock_filing_fetcher(),
        market_fetcher=_mock_market_fetcher(),
    )
    with pytest.raises(AgentOutputError, match="no tool_use block"):
        await agent.run(_intake())


async def test_info_capture_malformed_output_raises():
    llm = _mock_llm()
    llm.create_message = AsyncMock(
        return_value=_mock_response({"invalid": "data"})
    )
    agent = InfoCaptureAgent(
        llm,
        filing_fetcher=_mock_filing_fetcher(),
        market_fetcher=_mock_market_fetcher(),
    )
    with pytest.raises(AgentOutputError, match="failed to validate"):
        await agent.run(_intake())


async def test_info_capture_fetcher_failure_graceful():
    """If a fetcher fails, agent should still produce output (empty filings)."""
    llm = _mock_llm()
    llm.create_message = AsyncMock(
        return_value=_mock_response(_llm_tool_input())
    )

    bad_filing_fetcher = MagicMock()
    bad_filing_fetcher.market = "A_SHARE"
    bad_filing_fetcher.search_filings = AsyncMock(side_effect=Exception("Network down"))

    bad_market_fetcher = MagicMock()
    bad_market_fetcher.get_quote = AsyncMock(side_effect=Exception("API error"))

    agent = InfoCaptureAgent(
        llm,
        filing_fetcher=bad_filing_fetcher,
        market_fetcher=bad_market_fetcher,
    )
    result = await agent.run(_intake())

    # Should succeed with empty filings and default snapshot
    assert result.filing_manifest == []
    assert result.market_snapshot.price is None


async def test_info_capture_stores_filing_docs_in_context():
    """Filing documents should be stored in PipelineContext for downstream agents."""
    from poorcharlie.workflow.context import PipelineContext

    llm = _mock_llm()
    llm.create_message = AsyncMock(
        return_value=_mock_response(_llm_tool_input())
    )
    agent = InfoCaptureAgent(
        llm,
        filing_fetcher=_mock_filing_fetcher(),
        market_fetcher=_mock_market_fetcher(),
    )
    ctx = PipelineContext(_intake())
    result = await agent.run(_intake(), ctx=ctx)

    # Raw FilingDocuments stored for Filing agent
    docs = ctx.get_data("filing_documents")
    assert len(docs) == 2
    assert docs[0].ticker == "600519"


async def test_info_capture_no_fetchers_uses_resolver():
    """If no fetchers passed, agent resolves them from exchange."""
    from unittest.mock import patch

    llm = _mock_llm()
    llm.create_message = AsyncMock(
        return_value=_mock_response(_llm_tool_input())
    )

    mock_filing = _mock_filing_fetcher()
    mock_market = _mock_market_fetcher()

    with (
        patch(
            "poorcharlie.agents.info_capture.resolve_filing_fetcher",
            return_value=mock_filing,
        ),
        patch(
            "poorcharlie.agents.info_capture.resolve_market_data_fetcher",
            return_value=mock_market,
        ),
    ):
        agent = InfoCaptureAgent(llm)  # No fetchers passed
        result = await agent.run(_intake())

    assert len(result.filing_manifest) == 2
    assert result.market_snapshot.price == 1680.0
