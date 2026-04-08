"""Tests for investagent.workflow.decision_pipeline."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from investagent.llm import LLMClient
from investagent.schemas.candidate import PortfolioHolding
from investagent.store.candidate_store import CandidateStore
from investagent.workflow.decision_pipeline import run_decision_pipeline


def _mock_llm() -> LLMClient:
    return LLMClient(client=MagicMock())


def _sample_results() -> list[dict]:
    return [
        {
            "ticker": "600519", "name": "贵州茅台", "industry": "食品饮料",
            "final_label": "INVESTABLE",
            "enterprise_quality": "GREAT", "price_vs_value": "FAIR",
            "margin_of_safety_pct": 0.15, "meets_hurdle_rate": True,
            "thesis": "高ROE消费品龙头",
            "anti_thesis": "增长放缓",
        },
        {
            "ticker": "000858", "name": "五粮液", "industry": "食品饮料",
            "final_label": "INVESTABLE",
            "enterprise_quality": "GREAT", "price_vs_value": "CHEAP",
            "margin_of_safety_pct": 0.30, "meets_hurdle_rate": True,
            "thesis": "白酒第二",
            "anti_thesis": "渠道库存",
        },
        {
            "ticker": "000333", "name": "美的集团", "industry": "家用电器",
            "final_label": "INVESTABLE",
            "enterprise_quality": "GREAT", "price_vs_value": "FAIR",
            "margin_of_safety_pct": 0.10, "meets_hurdle_rate": True,
            "thesis": "家电龙头",
        },
    ]


def _mock_comparison_output():
    """Return a mock CrossComparisonOutput."""
    mock = MagicMock()
    r1 = MagicMock()
    r1.model_dump.return_value = {
        "ticker": "000858", "name": "五粮液", "rank": 1,
        "conviction_score": 9, "strengths_vs_peers": ["估值最低"],
        "weaknesses_vs_peers": [], "portfolio_fit_notes": "",
    }
    r2 = MagicMock()
    r2.model_dump.return_value = {
        "ticker": "600519", "name": "贵州茅台", "rank": 2,
        "conviction_score": 8, "strengths_vs_peers": ["护城河最深"],
        "weaknesses_vs_peers": [], "portfolio_fit_notes": "",
    }
    r3 = MagicMock()
    r3.model_dump.return_value = {
        "ticker": "000333", "name": "美的集团", "rank": 3,
        "conviction_score": 7, "strengths_vs_peers": ["行业分散"],
        "weaknesses_vs_peers": [], "portfolio_fit_notes": "",
    }
    mock.ranked_candidates = [r1, r2, r3]
    mock.concentration_warnings = []
    return mock


def _mock_strategy_output():
    """Return a mock PortfolioStrategyOutput."""
    from investagent.schemas.portfolio_strategy import ActionType

    mock = MagicMock()
    d1 = MagicMock()
    d1.ticker = "000858"
    d1.name = "五粮液"
    d1.action = ActionType.BUY
    d1.target_weight = 0.25
    d1.reason = "GREAT+CHEAP"

    d2 = MagicMock()
    d2.ticker = "000333"
    d2.name = "美的集团"
    d2.action = ActionType.BUY
    d2.target_weight = 0.15
    d2.reason = "行业分散"

    mock.position_decisions = [d1, d2]
    return mock


class TestDecisionPipeline:
    @pytest.mark.asyncio
    async def test_empty_candidates_returns_empty(self, tmp_path: Path):
        store = CandidateStore(tmp_path / "store.json")
        llm = _mock_llm()

        result = await run_decision_pipeline(store, llm, scan_date=date(2024, 5, 6))

        assert result == {}
        # Store should be saved
        assert (tmp_path / "store.json").exists()

    @pytest.mark.asyncio
    async def test_single_candidate_skips_comparison(self, tmp_path: Path):
        store = CandidateStore(tmp_path / "store.json")
        store.ingest_scan_results(
            [_sample_results()[0]],  # only maotai
            date(2024, 5, 6),
        )
        llm = _mock_llm()

        with patch(
            "investagent.workflow.decision_pipeline.CrossComparisonAgent"
        ) as mock_cc, patch(
            "investagent.workflow.decision_pipeline.PortfolioStrategyAgent"
        ) as mock_ps:
            # CrossComparison should NOT be called
            mock_ps_instance = MagicMock()
            mock_ps_instance.run = AsyncMock(return_value=_mock_strategy_output())
            mock_ps.return_value = mock_ps_instance

            result = await run_decision_pipeline(store, llm, scan_date=date(2024, 5, 6))

            mock_cc.assert_not_called()
            mock_ps_instance.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_full_flow(self, tmp_path: Path):
        store = CandidateStore(tmp_path / "store.json")
        store.ingest_scan_results(_sample_results(), date(2024, 5, 6))
        llm = _mock_llm()

        with patch(
            "investagent.workflow.decision_pipeline.CrossComparisonAgent"
        ) as mock_cc, patch(
            "investagent.workflow.decision_pipeline.PortfolioStrategyAgent"
        ) as mock_ps:
            mock_cc_instance = MagicMock()
            mock_cc_instance.run = AsyncMock(return_value=_mock_comparison_output())
            mock_cc.return_value = mock_cc_instance

            mock_ps_instance = MagicMock()
            mock_ps_instance.run = AsyncMock(return_value=_mock_strategy_output())
            mock_ps.return_value = mock_ps_instance

            result = await run_decision_pipeline(store, llm, scan_date=date(2024, 5, 6))

            # Both agents called
            mock_cc_instance.run.assert_called_once()
            mock_ps_instance.run.assert_called_once()

            # Result is {ticker: weight}
            assert result == {"000858": 0.25, "000333": 0.15}

            # Store updated with holdings
            assert len(store.get_current_holdings()) == 2
            assert store.to_portfolio_decisions() == {"000858": 0.25, "000333": 0.15}

    @pytest.mark.asyncio
    async def test_comparison_failure_falls_back(self, tmp_path: Path):
        store = CandidateStore(tmp_path / "store.json")
        store.ingest_scan_results(_sample_results(), date(2024, 5, 6))
        llm = _mock_llm()

        with patch(
            "investagent.workflow.decision_pipeline.CrossComparisonAgent"
        ) as mock_cc, patch(
            "investagent.workflow.decision_pipeline.PortfolioStrategyAgent"
        ) as mock_ps:
            # CrossComparison fails
            mock_cc_instance = MagicMock()
            mock_cc_instance.run = AsyncMock(side_effect=Exception("LLM error"))
            mock_cc.return_value = mock_cc_instance

            # Strategy still called with fallback ranking
            mock_ps_instance = MagicMock()
            mock_ps_instance.run = AsyncMock(return_value=_mock_strategy_output())
            mock_ps.return_value = mock_ps_instance

            result = await run_decision_pipeline(store, llm, scan_date=date(2024, 5, 6))

            # Strategy was still called despite comparison failure
            mock_ps_instance.run.assert_called_once()
            assert len(result) > 0
