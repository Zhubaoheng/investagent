"""Tests for PortfolioStrategy CORE/SATELLITE tier post-processing."""

from __future__ import annotations

from datetime import datetime, timezone

from poorcharlie.agents.portfolio_strategy import _enforce_core_tier_rules
from poorcharlie.schemas.portfolio_strategy import (
    ActionType,
    PortfolioStrategyOutput,
    PositionDecision,
    PositionTier,
)
from poorcharlie.schemas.common import AgentMeta


def _meta() -> AgentMeta:
    return AgentMeta(
        agent_name="portfolio_strategy",
        timestamp=datetime.now(tz=timezone.utc),
        model_used="test",
        token_usage=0,
    )


def _output(decisions: list[PositionDecision]) -> PortfolioStrategyOutput:
    return PortfolioStrategyOutput(
        meta=_meta(),
        position_decisions=decisions,
        cash_weight=0.30,
    )


def test_core_kept_when_great_and_investable() -> None:
    decisions = [
        PositionDecision(
            ticker="600519",
            name="贵州茅台",
            action=ActionType.BUY,
            target_weight=0.20,
            position_tier=PositionTier.CORE,
        )
    ]
    details = {
        "600519": {"enterprise_quality": "GREAT", "final_label": "INVESTABLE"}
    }
    output = _enforce_core_tier_rules(_output(decisions), details)
    assert output.position_decisions[0].position_tier == PositionTier.CORE


def test_core_downgraded_when_quality_is_good() -> None:
    decisions = [
        PositionDecision(
            ticker="000858",
            name="五粮液",
            action=ActionType.BUY,
            target_weight=0.15,
            position_tier=PositionTier.CORE,
        )
    ]
    # Not GREAT — must be downgraded
    details = {
        "000858": {"enterprise_quality": "GOOD", "final_label": "INVESTABLE"}
    }
    output = _enforce_core_tier_rules(_output(decisions), details)
    assert output.position_decisions[0].position_tier == PositionTier.SATELLITE


def test_core_downgraded_when_label_is_deep_dive() -> None:
    decisions = [
        PositionDecision(
            ticker="000333",
            name="美的集团",
            action=ActionType.BUY,
            target_weight=0.10,
            position_tier=PositionTier.CORE,
        )
    ]
    details = {
        "000333": {"enterprise_quality": "GREAT", "final_label": "DEEP_DIVE"}
    }
    output = _enforce_core_tier_rules(_output(decisions), details)
    assert output.position_decisions[0].position_tier == PositionTier.SATELLITE


def test_satellite_stays_satellite() -> None:
    decisions = [
        PositionDecision(
            ticker="000651",
            name="格力电器",
            action=ActionType.BUY,
            target_weight=0.10,
            position_tier=PositionTier.SATELLITE,
        )
    ]
    # Even with GREAT+INVESTABLE, SATELLITE stays SATELLITE (post-process
    # only downgrades, never upgrades — upgrade is LLM's decision)
    details = {
        "000651": {"enterprise_quality": "GREAT", "final_label": "INVESTABLE"}
    }
    output = _enforce_core_tier_rules(_output(decisions), details)
    assert output.position_decisions[0].position_tier == PositionTier.SATELLITE


def test_missing_details_forces_downgrade() -> None:
    # If candidate details are missing (unusual), CORE claim must be downgraded
    decisions = [
        PositionDecision(
            ticker="999999",
            name="mystery",
            action=ActionType.BUY,
            target_weight=0.05,
            position_tier=PositionTier.CORE,
        )
    ]
    output = _enforce_core_tier_rules(_output(decisions), {})
    assert output.position_decisions[0].position_tier == PositionTier.SATELLITE


def test_mixed_portfolio_selective_downgrade() -> None:
    decisions = [
        PositionDecision(
            ticker="600519",
            name="贵州茅台",
            action=ActionType.BUY,
            target_weight=0.20,
            position_tier=PositionTier.CORE,
        ),
        PositionDecision(
            ticker="000858",
            name="五粮液",
            action=ActionType.BUY,
            target_weight=0.15,
            position_tier=PositionTier.CORE,  # claimed but will be downgraded
        ),
        PositionDecision(
            ticker="000651",
            name="格力电器",
            action=ActionType.BUY,
            target_weight=0.10,
            position_tier=PositionTier.SATELLITE,
        ),
    ]
    details = {
        "600519": {"enterprise_quality": "GREAT", "final_label": "INVESTABLE"},
        "000858": {"enterprise_quality": "GOOD", "final_label": "INVESTABLE"},
        "000651": {"enterprise_quality": "AVERAGE", "final_label": "DEEP_DIVE"},
    }
    output = _enforce_core_tier_rules(_output(decisions), details)
    by_ticker = {d.ticker: d.position_tier for d in output.position_decisions}
    assert by_ticker["600519"] == PositionTier.CORE
    assert by_ticker["000858"] == PositionTier.SATELLITE
    assert by_ticker["000651"] == PositionTier.SATELLITE
