"""Tests for investagent.agents.portfolio_strategy."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from investagent.agents.portfolio_strategy import (
    PortfolioStrategyAgent,
    PortfolioStrategyInput,
    StrategyHoldingInfo,
)
from investagent.llm import LLMClient


def _mock_llm() -> LLMClient:
    return LLMClient(client=MagicMock())


def _mock_response(
    position_decisions: list[dict],
    cash_weight: float,
    industry_distribution: dict | None = None,
    portfolio_quality_summary: str = "",
    risk_notes: list[str] | None = None,
    rebalance_summary: list[str] | None = None,
) -> MagicMock:
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.input = {
        "position_decisions": position_decisions,
        "cash_weight": cash_weight,
        "industry_distribution": industry_distribution or {},
        "portfolio_quality_summary": portfolio_quality_summary,
        "risk_notes": risk_notes or [],
        "rebalance_summary": rebalance_summary or [],
    }
    resp = MagicMock()
    resp.content = [tool_block]
    resp.model = "test-model"
    resp.usage = MagicMock(input_tokens=500, output_tokens=300)
    return resp


def _sample_ranked() -> list[dict]:
    return [
        {
            "ticker": "000858", "name": "五粮液", "rank": 1,
            "conviction_score": 9,
            "strengths_vs_peers": ["估值更低"],
            "weaknesses_vs_peers": ["渠道库存"],
            "portfolio_fit_notes": "",
        },
        {
            "ticker": "000333", "name": "美的集团", "rank": 2,
            "conviction_score": 7,
            "strengths_vs_peers": ["行业分散"],
            "weaknesses_vs_peers": ["周期敞口"],
            "portfolio_fit_notes": "",
        },
    ]


def _sample_details() -> dict[str, dict]:
    return {
        "000858": {
            "industry": "食品饮料",
            "enterprise_quality": "GREAT",
            "price_vs_value": "CHEAP",
            "margin_of_safety_pct": 0.30,
            "thesis": "白酒第二",
        },
        "000333": {
            "industry": "家用电器",
            "enterprise_quality": "GREAT",
            "price_vs_value": "FAIR",
            "margin_of_safety_pct": 0.10,
            "thesis": "家电龙头",
        },
    }


class TestPortfolioStrategyAgent:
    def test_instantiation(self):
        agent = PortfolioStrategyAgent(llm=_mock_llm())
        assert agent.name == "portfolio_strategy"

    def test_build_context_with_rankings(self):
        agent = PortfolioStrategyAgent(llm=_mock_llm())
        inp = PortfolioStrategyInput(
            ranked_candidates=_sample_ranked(),
            candidate_details=_sample_details(),
            available_cash_pct=1.0,
        )
        ctx = agent._build_user_context(inp)

        assert len(ctx["ranked_candidates"]) == 2
        assert ctx["ranked_candidates"][0]["ticker"] == "000858"
        assert ctx["ranked_candidates"][0]["conviction_score"] == 9
        assert ctx["ranked_candidates"][0]["industry"] == "食品饮料"
        assert ctx["ranked_candidates"][0]["margin_of_safety_pct"] == "30%"
        assert ctx["has_holdings"] is False
        assert ctx["available_cash_pct"] == "100%"

    def test_build_context_with_holdings(self):
        agent = PortfolioStrategyAgent(llm=_mock_llm())
        inp = PortfolioStrategyInput(
            ranked_candidates=_sample_ranked(),
            candidate_details=_sample_details(),
            current_holdings=[
                StrategyHoldingInfo(
                    ticker="600519", name="贵州茅台",
                    weight=0.25, industry="食品饮料",
                ),
            ],
            available_cash_pct=0.75,
        )
        ctx = agent._build_user_context(inp)

        assert ctx["has_holdings"] is True
        assert len(ctx["current_holdings"]) == 1
        assert ctx["current_holdings"][0]["weight"] == "25%"
        assert ctx["available_cash_pct"] == "75%"

    def test_build_context_empty(self):
        agent = PortfolioStrategyAgent(llm=_mock_llm())
        inp = PortfolioStrategyInput()
        ctx = agent._build_user_context(inp)
        assert ctx["ranked_candidates"] == []
        assert ctx["has_holdings"] is False

    @pytest.mark.asyncio
    async def test_run_produces_decisions(self):
        llm = _mock_llm()
        llm.create_message = AsyncMock(return_value=_mock_response(
            position_decisions=[
                {
                    "ticker": "000858", "name": "五粮液", "action": "BUY",
                    "target_weight": 0.25, "current_weight": 0.0,
                    "conviction_score": 9,
                    "reason": "GREAT+CHEAP，安全边际30%",
                    "sizing_rationale": "确信度9，安全边际大，给予上限仓位",
                },
                {
                    "ticker": "000333", "name": "美的集团", "action": "BUY",
                    "target_weight": 0.15, "current_weight": 0.0,
                    "conviction_score": 7,
                    "reason": "GREAT+FAIR，行业分散",
                    "sizing_rationale": "确信度7，安全边际一般，中等仓位",
                },
            ],
            cash_weight=0.60,
            industry_distribution={"食品饮料": 0.25, "家用电器": 0.15},
            portfolio_quality_summary="GREAT企业为主的集中组合",
            rebalance_summary=["买入 000858 五粮液 25%", "买入 000333 美的集团 15%"],
        ))

        agent = PortfolioStrategyAgent(llm=llm)
        inp = PortfolioStrategyInput(
            ranked_candidates=_sample_ranked(),
            candidate_details=_sample_details(),
        )
        result = await agent.run(inp)

        assert len(result.position_decisions) == 2
        assert result.position_decisions[0].ticker == "000858"
        assert result.position_decisions[0].action.value == "BUY"
        assert result.position_decisions[0].target_weight == 0.25
        assert result.cash_weight == 0.60
        assert len(result.rebalance_summary) == 2

    @pytest.mark.asyncio
    async def test_run_all_cash(self):
        llm = _mock_llm()
        llm.create_message = AsyncMock(return_value=_mock_response(
            position_decisions=[],
            cash_weight=1.0,
            rebalance_summary=["无合适标的，全部持有现金"],
        ))

        agent = PortfolioStrategyAgent(llm=llm)
        inp = PortfolioStrategyInput()
        result = await agent.run(inp)

        assert len(result.position_decisions) == 0
        assert result.cash_weight == 1.0
