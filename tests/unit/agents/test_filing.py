"""Tests for investagent.agents.filing."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from investagent.agents.base import AgentOutputError
from investagent.agents.filing import FilingAgent
from investagent.llm import LLMClient
from investagent.schemas.company import CompanyIntake


def _intake() -> CompanyIntake:
    return CompanyIntake(ticker="600519", name="贵州茅台", exchange="SSE")


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


def _filing_tool_input() -> dict:
    return {
        "filing_meta": {
            "market": "A_SHARE",
            "accounting_standard": "CAS",
            "fiscal_years_covered": ["2021", "2022", "2023"],
            "filing_types": ["年报"],
            "currency": "CNY",
            "reporting_language": "zh-CN",
        },
        "income_statement": [
            {
                "fiscal_year": "2023",
                "fiscal_period": "FY",
                "revenue": 150056000000.0,
                "net_income": 74734000000.0,
                "net_income_to_parent": 74734000000.0,
                "eps_basic": 59.49,
            },
        ],
        "balance_sheet": [
            {
                "fiscal_year": "2023",
                "cash_and_equivalents": 57200000000.0,
                "total_assets": 256800000000.0,
                "total_liabilities": 74500000000.0,
                "shareholders_equity": 175000000000.0,
            },
        ],
        "cash_flow": [
            {
                "fiscal_year": "2023",
                "operating_cash_flow": 82000000000.0,
                "capex": -5800000000.0,
                "free_cash_flow": 76200000000.0,
                "dividends_paid": -38000000000.0,
            },
        ],
        "segments": [
            {
                "fiscal_year": "2023",
                "segment_name": "茅台酒",
                "revenue": 140000000000.0,
            },
        ],
        "accounting_policies": [
            {
                "category": "revenue_recognition",
                "fiscal_year": "2023",
                "method": "在控制权转移时点确认收入",
                "raw_text": "公司在将商品控制权转移给客户时确认收入...",
                "changed_from_prior": False,
            },
        ],
        "debt_schedule": [],
        "covenant_status": [],
        "special_items": [],
        "concentration": {
            "top_customer_pct": None,
            "top5_customers_pct": None,
            "customer_losses": [],
            "major_supplier_dependencies": ["高粱供应"],
            "top5_suppliers_pct": None,
            "geographic_revenue_split": {"国内": 0.95, "海外": 0.05},
        },
        "buyback_history": [],
        "acquisition_history": [],
        "dividend_per_share_history": [
            {"fiscal_year": "2023", "dividend_per_share": 30.226},
        ],
        "footnote_extracts": [],
        "risk_factors": [
            {
                "category": "policy",
                "description": "白酒行业消费税政策不确定性",
                "raw_text": "白酒消费税采用从价加从量复合计税...",
                "materiality": "high",
            },
        ],
    }


@pytest.mark.asyncio
async def test_filing_output():
    llm = _mock_llm()
    llm.create_message = AsyncMock(
        return_value=_mock_response(_filing_tool_input())
    )
    agent = FilingAgent(llm)
    result = await agent.run(_intake())
    assert result.meta.agent_name == "filing"
    assert result.filing_meta.market == "A_SHARE"
    assert len(result.income_statement) == 1
    assert result.income_statement[0].revenue == 150056000000.0
    assert result.meta.token_usage == 300


@pytest.mark.asyncio
async def test_filing_meta_is_server_generated():
    tool_input = _filing_tool_input()
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
    agent = FilingAgent(llm)
    result = await agent.run(_intake())
    assert result.meta.agent_name == "filing"
    assert result.meta.model_used == "claude-sonnet-4-20250514"


@pytest.mark.asyncio
async def test_filing_no_tool_use_raises():
    text_block = MagicMock()
    text_block.type = "text"
    response = MagicMock()
    response.content = [text_block]
    response.model = "claude-sonnet-4-20250514"
    response.usage = MagicMock()
    response.usage.input_tokens = 50
    response.usage.output_tokens = 100

    llm = _mock_llm()
    llm.create_message = AsyncMock(return_value=response)
    agent = FilingAgent(llm)
    with pytest.raises(AgentOutputError, match="no tool_use block"):
        await agent.run(_intake())


@pytest.mark.asyncio
async def test_filing_malformed_output_raises():
    llm = _mock_llm()
    llm.create_message = AsyncMock(
        return_value=_mock_response({"invalid": "data"})
    )
    agent = FilingAgent(llm)
    with pytest.raises(AgentOutputError, match="failed to validate"):
        await agent.run(_intake())
