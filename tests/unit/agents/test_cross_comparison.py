"""Tests for poorcharlie.agents.cross_comparison."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from poorcharlie.agents.cross_comparison import (
    CrossComparisonAgent,
    CrossComparisonInput,
)
from poorcharlie.llm import LLMClient


def _mock_llm() -> LLMClient:
    return LLMClient(client=MagicMock())


def _mock_response(
    ranked_candidates: list[dict],
    pairwise_insights: list[dict] | None = None,
    concentration_warnings: list[str] | None = None,
) -> MagicMock:
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.input = {
        "ranked_candidates": ranked_candidates,
        "pairwise_insights": pairwise_insights or [],
        "concentration_warnings": concentration_warnings or [],
    }
    resp = MagicMock()
    resp.content = [tool_block]
    resp.model = "test-model"
    resp.usage = MagicMock(input_tokens=500, output_tokens=300)
    return resp


def _sample_candidates() -> list[dict]:
    return [
        {
            "ticker": "600519",
            "name": "贵州茅台",
            "industry": "食品饮料",
            "final_label": "INVESTABLE",
            "enterprise_quality": "GREAT",
            "price_vs_value": "FAIR",
            "margin_of_safety_pct": 0.15,
            "meets_hurdle_rate": True,
            "thesis": "高ROE消费品龙头，护城河深厚",
            "anti_thesis": "增长放缓",
            "largest_unknowns": ["消费降级影响"],
            "expected_return_summary": "年化15%",
            "why_now": "估值合理",
        },
        {
            "ticker": "000858",
            "name": "五粮液",
            "industry": "食品饮料",
            "final_label": "INVESTABLE",
            "enterprise_quality": "GREAT",
            "price_vs_value": "CHEAP",
            "margin_of_safety_pct": 0.30,
            "meets_hurdle_rate": True,
            "thesis": "白酒行业第二，估值有吸引力",
            "anti_thesis": "渠道库存偏高",
            "largest_unknowns": ["行业竞争格局"],
            "expected_return_summary": "年化20%",
            "why_now": "估值低位",
        },
        {
            "ticker": "000333",
            "name": "美的集团",
            "industry": "家用电器",
            "enterprise_quality": "GREAT",
            "price_vs_value": "FAIR",
            "margin_of_safety_pct": 0.10,
            "meets_hurdle_rate": True,
            "thesis": "家电龙头，海外扩张顺利",
            "anti_thesis": "地产周期影响",
        },
    ]


class TestCrossComparisonAgent:
    def test_instantiation(self):
        agent = CrossComparisonAgent(llm=_mock_llm())
        assert agent.name == "cross_comparison"

    def test_build_context_with_candidates(self):
        agent = CrossComparisonAgent(llm=_mock_llm())
        inp = CrossComparisonInput(candidates=_sample_candidates())
        ctx = agent._build_user_context(inp)

        assert ctx["candidate_count"] == 3
        assert len(ctx["candidates"]) == 3
        assert ctx["candidates"][0]["ticker"] == "600519"
        assert ctx["candidates"][0]["enterprise_quality"] == "GREAT"
        assert ctx["candidates"][0]["margin_of_safety_pct"] == "15%"
        # Industry counts
        assert ctx["industry_counts"]["食品饮料"] == 2
        assert ctx["industry_counts"]["家用电器"] == 1

    def test_build_context_truncates_thesis(self):
        agent = CrossComparisonAgent(llm=_mock_llm())
        long_thesis = "A" * 500
        inp = CrossComparisonInput(candidates=[{
            "ticker": "600519", "thesis": long_thesis,
            "final_label": "INVESTABLE",
        }])
        ctx = agent._build_user_context(inp)
        assert len(ctx["candidates"][0]["thesis"]) == 300

    def test_build_context_empty(self):
        agent = CrossComparisonAgent(llm=_mock_llm())
        inp = CrossComparisonInput()
        ctx = agent._build_user_context(inp)
        assert ctx["candidates"] == []
        assert ctx["candidate_count"] == 0

    @pytest.mark.asyncio
    async def test_run_produces_rankings(self):
        llm = _mock_llm()
        llm.create_message = AsyncMock(return_value=_mock_response(
            ranked_candidates=[
                {
                    "ticker": "000858", "name": "五粮液", "rank": 1,
                    "conviction_score": 9,
                    "strengths_vs_peers": ["估值更低", "安全边际更大"],
                    "weaknesses_vs_peers": ["渠道库存风险"],
                    "portfolio_fit_notes": "",
                },
                {
                    "ticker": "600519", "name": "贵州茅台", "rank": 2,
                    "conviction_score": 8,
                    "strengths_vs_peers": ["护城河最深"],
                    "weaknesses_vs_peers": ["估值较高"],
                    "portfolio_fit_notes": "与五粮液同行业",
                },
                {
                    "ticker": "000333", "name": "美的集团", "rank": 3,
                    "conviction_score": 7,
                    "strengths_vs_peers": ["行业分散"],
                    "weaknesses_vs_peers": ["周期敞口"],
                    "portfolio_fit_notes": "",
                },
            ],
            pairwise_insights=[{
                "ticker_a": "000858",
                "ticker_b": "600519",
                "comparison": "五粮液安全边际更大，同等质量下估值更优",
                "dimension": "valuation",
            }],
            concentration_warnings=["食品饮料行业占候选标的 67%"],
        ))

        agent = CrossComparisonAgent(llm=llm)
        inp = CrossComparisonInput(candidates=_sample_candidates())
        result = await agent.run(inp)

        assert len(result.ranked_candidates) == 3
        assert result.ranked_candidates[0].ticker == "000858"
        assert result.ranked_candidates[0].rank == 1
        assert result.ranked_candidates[0].conviction_score == 9
        assert len(result.pairwise_insights) == 1
        assert len(result.concentration_warnings) == 1
