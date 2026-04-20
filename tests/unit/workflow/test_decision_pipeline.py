"""Tests for poorcharlie.workflow.decision_pipeline."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from poorcharlie.llm import LLMClient
from poorcharlie.schemas.candidate import PortfolioHolding
from poorcharlie.store.candidate_store import CandidateStore
from poorcharlie.workflow.decision_pipeline import run_decision_pipeline


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
    from poorcharlie.schemas.portfolio_strategy import ActionType, PositionTier

    mock = MagicMock()
    d1 = MagicMock()
    d1.ticker = "000858"
    d1.name = "五粮液"
    d1.action = ActionType.BUY
    d1.target_weight = 0.25
    d1.current_weight = 0.0
    d1.conviction_score = 9
    d1.position_tier = PositionTier.CORE
    d1.reason = "GREAT+CHEAP"

    d2 = MagicMock()
    d2.ticker = "000333"
    d2.name = "美的集团"
    d2.action = ActionType.BUY
    d2.target_weight = 0.15
    d2.current_weight = 0.0
    d2.conviction_score = 7
    d2.position_tier = PositionTier.SATELLITE
    d2.reason = "行业分散"

    mock.position_decisions = [d1, d2]
    return mock


def _mock_trader_output(weights_by_ticker: dict[str, float] | None = None):
    """Return a mock TraderOutput mirroring _mock_strategy_output.

    Pass weights_by_ticker to override defaults (e.g., for CORE-blocked scenarios).
    """
    from datetime import datetime, timezone

    from poorcharlie.schemas.common import AgentMeta
    from poorcharlie.schemas.portfolio_strategy import PositionTier
    from poorcharlie.schemas.trader import (
        MarketRegime,
        OrderAction,
        OrderUrgency,
        TraderOrder,
        TraderOutput,
    )

    weights = weights_by_ticker or {"000858": 0.25, "000333": 0.15}
    tiers = {"000858": PositionTier.CORE, "000333": PositionTier.SATELLITE}
    names = {"000858": "五粮液", "000333": "美的集团"}

    orders = [
        TraderOrder(
            ticker=t,
            name=names.get(t, ""),
            action=OrderAction.BUY,
            target_weight=w,
            current_weight=0.0,
            position_tier=tiers.get(t, PositionTier.SATELLITE),
            urgency=OrderUrgency.NORMAL,
            rationale="trader-confirmed",
        )
        for t, w in weights.items()
    ]

    return TraderOutput(
        meta=AgentMeta(
            agent_name="trader",
            timestamp=datetime.now(tz=timezone.utc),
            model_used="test",
            token_usage=0,
        ),
        market_regime=MarketRegime.NORMAL,
        market_assessment="test-normal-regime",
        orders=orders,
        blocked_orders_count=0,
        overall_urgency=OrderUrgency.NORMAL,
        execution_plan_summary="test plan",
    )


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
            "poorcharlie.workflow.decision_pipeline.CrossComparisonAgent"
        ) as mock_cc, patch(
            "poorcharlie.workflow.decision_pipeline.PortfolioStrategyAgent"
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
            "poorcharlie.workflow.decision_pipeline.CrossComparisonAgent"
        ) as mock_cc, patch(
            "poorcharlie.workflow.decision_pipeline.PortfolioStrategyAgent"
        ) as mock_ps, patch(
            "poorcharlie.workflow.decision_pipeline.TraderAgent"
        ) as mock_tr:
            mock_cc_instance = MagicMock()
            mock_cc_instance.run = AsyncMock(return_value=_mock_comparison_output())
            mock_cc.return_value = mock_cc_instance

            mock_ps_instance = MagicMock()
            mock_ps_instance.run = AsyncMock(return_value=_mock_strategy_output())
            mock_ps.return_value = mock_ps_instance

            mock_tr_instance = MagicMock()
            mock_tr_instance.run = AsyncMock(
                return_value=_mock_trader_output({"000858": 0.20, "000333": 0.15}),
            )
            mock_tr.return_value = mock_tr_instance

            result = await run_decision_pipeline(store, llm, scan_date=date(2024, 5, 6))

            # All three agents called
            mock_cc_instance.run.assert_called_once()
            mock_ps_instance.run.assert_called_once()
            mock_tr_instance.run.assert_called_once()

            # Result reflects Trader output (weight capped from 0.25 to 0.20)
            assert result == {"000858": 0.20, "000333": 0.15}

            # Store updated with holdings
            assert len(store.get_current_holdings()) == 2
            assert store.to_portfolio_decisions() == {"000858": 0.20, "000333": 0.15}

    @pytest.mark.asyncio
    async def test_no_change_short_circuits_llm(self, tmp_path: Path):
        """When a held position re-appears unchanged and no new INVESTABLE
        candidates arrive, the pipeline must preserve previous weights
        without calling the LLM agents.
        """
        store = CandidateStore(tmp_path / "store.json")
        # First scan: ingest + seed holdings manually (simulating a prior run)
        store.ingest_scan_results(_sample_results(), date(2024, 5, 6))
        store.update_holdings(
            [
                PortfolioHolding(
                    ticker="000858",
                    name="五粮液",
                    industry="食品饮料",
                    target_weight=0.25,
                    entry_date=date(2024, 5, 6),
                ),
                PortfolioHolding(
                    ticker="000333",
                    name="美的集团",
                    industry="家用电器",
                    target_weight=0.15,
                    entry_date=date(2024, 5, 6),
                ),
            ],
            scan_date=date(2024, 5, 6),
        )
        store.save()

        # Second scan: same data arrives again — no triggers should fire.
        # prev_* fields get populated by the second ingest.
        store.ingest_scan_results(_sample_results(), date(2024, 9, 22))
        llm = _mock_llm()

        with patch(
            "poorcharlie.workflow.decision_pipeline.CrossComparisonAgent"
        ) as mock_cc, patch(
            "poorcharlie.workflow.decision_pipeline.PortfolioStrategyAgent"
        ) as mock_ps, patch(
            "poorcharlie.workflow.decision_pipeline._build_industry_map",
            return_value={},
        ):
            mock_cc_instance = MagicMock()
            mock_cc_instance.run = AsyncMock(return_value=_mock_comparison_output())
            mock_cc.return_value = mock_cc_instance

            mock_ps_instance = MagicMock()
            mock_ps_instance.run = AsyncMock(return_value=_mock_strategy_output())
            mock_ps.return_value = mock_ps_instance

            result = await run_decision_pipeline(
                store, llm, scan_date=date(2024, 9, 22),
            )

            # Neither LLM agent should have been called — short circuit fired
            mock_cc_instance.run.assert_not_called()
            mock_ps_instance.run.assert_not_called()

            # Previous holdings preserved
            assert result == {"000858": 0.25, "000333": 0.15}

    @pytest.mark.asyncio
    async def test_quality_downgrade_triggers_llm(self, tmp_path: Path):
        """When a held position's quality is downgraded, pipeline must
        run the LLM (not short-circuit).
        """
        store = CandidateStore(tmp_path / "store.json")
        store.ingest_scan_results(_sample_results(), date(2024, 5, 6))
        store.update_holdings(
            [
                PortfolioHolding(
                    ticker="000858",
                    name="五粮液",
                    industry="食品饮料",
                    target_weight=0.25,
                    entry_date=date(2024, 5, 6),
                ),
                PortfolioHolding(
                    ticker="000333",
                    name="美的集团",
                    industry="家用电器",
                    target_weight=0.15,
                    entry_date=date(2024, 5, 6),
                ),
            ],
            scan_date=date(2024, 5, 6),
        )
        store.save()

        # Second scan: 000858 downgraded GREAT → AVERAGE
        downgraded_results = _sample_results()
        downgraded_results[1]["enterprise_quality"] = "AVERAGE"
        store.ingest_scan_results(downgraded_results, date(2024, 9, 22))
        llm = _mock_llm()

        with patch(
            "poorcharlie.workflow.decision_pipeline.CrossComparisonAgent"
        ) as mock_cc, patch(
            "poorcharlie.workflow.decision_pipeline.PortfolioStrategyAgent"
        ) as mock_ps, patch(
            "poorcharlie.workflow.decision_pipeline._build_industry_map",
            return_value={},
        ):
            mock_cc_instance = MagicMock()
            mock_cc_instance.run = AsyncMock(return_value=_mock_comparison_output())
            mock_cc.return_value = mock_cc_instance

            mock_ps_instance = MagicMock()
            mock_ps_instance.run = AsyncMock(return_value=_mock_strategy_output())
            mock_ps.return_value = mock_ps_instance

            with patch(
                "poorcharlie.workflow.decision_pipeline.TraderAgent"
            ) as mock_tr:
                mock_tr_instance = MagicMock()
                mock_tr_instance.run = AsyncMock(
                    return_value=_mock_trader_output(),
                )
                mock_tr.return_value = mock_tr_instance

                await run_decision_pipeline(
                    store, llm, scan_date=date(2024, 9, 22),
                )

                # Triggers fired → LLM agents called
                mock_cc_instance.run.assert_called_once()
                mock_ps_instance.run.assert_called_once()
                mock_tr_instance.run.assert_called_once()

                # Verify change_triggers were passed to strategy
                strategy_call_input = mock_ps_instance.run.call_args[0][0]
                assert len(strategy_call_input.change_triggers) >= 1
                trigger_types = {
                    t["trigger_type"] for t in strategy_call_input.change_triggers
                }
                assert "QUALITY_DOWNGRADE" in trigger_types

    @pytest.mark.asyncio
    async def test_tier_flows_from_strategy_to_holdings(self, tmp_path: Path):
        """PortfolioStrategy's position_tier must be written into
        PortfolioHolding so change_detector and Trader can see it later.
        """
        store = CandidateStore(tmp_path / "store.json")
        store.ingest_scan_results(_sample_results(), date(2024, 5, 6))
        llm = _mock_llm()

        with patch(
            "poorcharlie.workflow.decision_pipeline.CrossComparisonAgent"
        ) as mock_cc, patch(
            "poorcharlie.workflow.decision_pipeline.PortfolioStrategyAgent"
        ) as mock_ps, patch(
            "poorcharlie.workflow.decision_pipeline.TraderAgent"
        ) as mock_tr, patch(
            "poorcharlie.workflow.decision_pipeline._build_industry_map",
            return_value={},
        ):
            mock_cc_instance = MagicMock()
            mock_cc_instance.run = AsyncMock(return_value=_mock_comparison_output())
            mock_cc.return_value = mock_cc_instance

            mock_ps_instance = MagicMock()
            mock_ps_instance.run = AsyncMock(return_value=_mock_strategy_output())
            mock_ps.return_value = mock_ps_instance

            mock_tr_instance = MagicMock()
            mock_tr_instance.run = AsyncMock(
                return_value=_mock_trader_output({"000858": 0.20, "000333": 0.15}),
            )
            mock_tr.return_value = mock_tr_instance

            await run_decision_pipeline(store, llm, scan_date=date(2024, 5, 6))

            # Tier flows through Trader to holdings
            holdings = store.get_current_holdings()
            by_ticker = {h.ticker: h.position_tier for h in holdings}
            assert by_ticker["000858"] == "CORE"
            assert by_ticker["000333"] == "SATELLITE"

    @pytest.mark.asyncio
    async def test_trader_blocks_core_sell_preserves_holding(
        self, tmp_path: Path,
    ) -> None:
        """End-to-end: Strategy proposes EXIT on CORE, Trader blocks it,
        the holding is preserved in the store at its previous weight.
        """
        from poorcharlie.schemas.portfolio_strategy import ActionType, PositionTier

        store = CandidateStore(tmp_path / "store.json")
        store.ingest_scan_results(_sample_results(), date(2024, 5, 6))
        # Seed a CORE holding for 600519 茅台
        store.update_holdings(
            [
                PortfolioHolding(
                    ticker="600519",
                    name="贵州茅台",
                    industry="食品饮料",
                    target_weight=0.18,
                    entry_date=date(2024, 5, 6),
                    position_tier="CORE",
                ),
            ],
            scan_date=date(2024, 5, 6),
        )
        store.save()
        # Fresh scan: same data, nothing bad happened
        store.ingest_scan_results(_sample_results(), date(2024, 9, 22))

        # Strategy (hallucinating "estimate too expensive") proposes EXIT
        strategy_output = MagicMock()
        exit_decision = MagicMock()
        exit_decision.ticker = "600519"
        exit_decision.name = "贵州茅台"
        exit_decision.action = ActionType.EXIT
        exit_decision.target_weight = 0.0
        exit_decision.current_weight = 0.18
        exit_decision.conviction_score = 8
        exit_decision.position_tier = PositionTier.CORE
        exit_decision.reason = "估值太贵了"
        strategy_output.position_decisions = [exit_decision]

        # Real TraderAgent runs (not mocked), but its internal LLM call will
        # fail on the mock LLM → falls back to rules-only output.
        # The pre-check should have already converted EXIT → BLOCKED.

        llm = _mock_llm()

        with patch(
            "poorcharlie.workflow.decision_pipeline.CrossComparisonAgent"
        ) as mock_cc, patch(
            "poorcharlie.workflow.decision_pipeline.PortfolioStrategyAgent"
        ) as mock_ps, patch(
            "poorcharlie.workflow.decision_pipeline._build_industry_map",
            return_value={},
        ):
            mock_cc_instance = MagicMock()
            mock_cc_instance.run = AsyncMock(return_value=_mock_comparison_output())
            mock_cc.return_value = mock_cc_instance

            mock_ps_instance = MagicMock()
            mock_ps_instance.run = AsyncMock(return_value=strategy_output)
            mock_ps.return_value = mock_ps_instance

            # No mock for TraderAgent — use the real agent with rules fallback
            result = await run_decision_pipeline(
                store, llm, scan_date=date(2024, 9, 22),
            )

            # CORE holding preserved at its original 0.18 weight
            assert "600519" in result
            assert result["600519"] == 0.18

            holdings = store.get_current_holdings()
            by_ticker = {h.ticker: h for h in holdings}
            assert "600519" in by_ticker
            assert by_ticker["600519"].position_tier == "CORE"

    @pytest.mark.asyncio
    async def test_comparison_failure_falls_back(self, tmp_path: Path):
        store = CandidateStore(tmp_path / "store.json")
        store.ingest_scan_results(_sample_results(), date(2024, 5, 6))
        llm = _mock_llm()

        with patch(
            "poorcharlie.workflow.decision_pipeline.CrossComparisonAgent"
        ) as mock_cc, patch(
            "poorcharlie.workflow.decision_pipeline.PortfolioStrategyAgent"
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
