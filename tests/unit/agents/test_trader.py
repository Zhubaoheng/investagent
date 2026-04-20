"""Tests for the TraderAgent — pre-check, post-check, and integration."""

from __future__ import annotations

from datetime import datetime, timezone

from poorcharlie.agents.trader import (
    _core_sell_has_justification,
    _pre_check_decisions,
    _post_check_output,
)
from poorcharlie.schemas.common import AgentMeta
from poorcharlie.schemas.portfolio_strategy import PositionTier
from poorcharlie.schemas.trader import (
    MarketRegime,
    OrderAction,
    OrderUrgency,
    TraderOrder,
    TraderOutput,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _core_holding(
    ticker: str = "600519", weight: float = 0.18,
) -> dict:
    return {
        "ticker": ticker,
        "name": "贵州茅台",
        "industry": "食品饮料",
        "target_weight": weight,
        "position_tier": "CORE",
    }


def _satellite_holding(
    ticker: str = "000651", weight: float = 0.10,
) -> dict:
    return {
        "ticker": ticker,
        "name": "格力电器",
        "industry": "家用电器",
        "target_weight": weight,
        "position_tier": "SATELLITE",
    }


def _clean_detail(
    quality: str = "GREAT",
    label: str = "INVESTABLE",
    risk: str = "GREEN",
    kill_shots: list[str] | None = None,
) -> dict:
    return {
        "enterprise_quality": quality,
        "final_label": label,
        "accounting_risk_level": risk,
        "kill_shots": kill_shots or [],
    }


def _proposed(
    ticker: str,
    action: str = "EXIT",
    tier: str = "CORE",
    target_weight: float = 0.0,
    current_weight: float = 0.18,
) -> dict:
    return {
        "ticker": ticker,
        "name": "sample",
        "action": action,
        "target_weight": target_weight,
        "current_weight": current_weight,
        "conviction_score": 8,
        "position_tier": tier,
        "reason": "test",
    }


def _meta() -> AgentMeta:
    return AgentMeta(
        agent_name="trader",
        timestamp=datetime.now(tz=timezone.utc),
        model_used="test",
        token_usage=0,
    )


# ---------------------------------------------------------------------------
# _core_sell_has_justification
# ---------------------------------------------------------------------------

def test_core_sell_unjustified_when_all_clean() -> None:
    det = {"600519": _clean_detail()}
    justified, reason = _core_sell_has_justification("600519", det)
    assert justified is False
    assert "not justified" in reason.lower() or "no kill_shots" in reason.lower()


def test_core_sell_justified_by_kill_shots() -> None:
    det = {"600519": _clean_detail(kill_shots=["管理层失信"])}
    justified, reason = _core_sell_has_justification("600519", det)
    assert justified is True
    assert "kill_shot" in reason.lower()


def test_core_sell_justified_by_red_accounting() -> None:
    det = {"600519": _clean_detail(risk="RED")}
    justified, reason = _core_sell_has_justification("600519", det)
    assert justified is True
    assert "red" in reason.lower()


def test_core_sell_justified_by_quality_collapse() -> None:
    det = {"600519": _clean_detail(quality="POOR")}
    justified, reason = _core_sell_has_justification("600519", det)
    assert justified is True
    assert "poor" in reason.lower()


def test_core_sell_justified_by_label_reject() -> None:
    det = {"600519": _clean_detail(label="REJECT")}
    justified, reason = _core_sell_has_justification("600519", det)
    assert justified is True
    assert "reject" in reason.lower()


# ---------------------------------------------------------------------------
# _pre_check_decisions — CORE sell blocking
# ---------------------------------------------------------------------------

def test_pre_check_blocks_core_exit_without_kill_shot() -> None:
    holdings = [_core_holding("600519", weight=0.18)]
    details = {"600519": _clean_detail()}
    proposed = [_proposed("600519", action="EXIT", tier="CORE", target_weight=0.0)]

    enriched, blocked = _pre_check_decisions(proposed, holdings, details)

    assert blocked == ["600519"]
    assert enriched[0]["action"] == "BLOCKED"
    assert enriched[0]["target_weight"] == 0.18  # restored to holding weight
    assert "CORE sell blocked" in enriched[0]["blocked_reason"]


def test_pre_check_blocks_core_reduce_without_kill_shot() -> None:
    holdings = [_core_holding("600519", weight=0.18)]
    details = {"600519": _clean_detail()}
    proposed = [
        _proposed("600519", action="REDUCE", tier="CORE", target_weight=0.05)
    ]

    enriched, blocked = _pre_check_decisions(proposed, holdings, details)

    assert blocked == ["600519"]
    assert enriched[0]["action"] == "BLOCKED"


def test_pre_check_allows_core_exit_with_kill_shot() -> None:
    holdings = [_core_holding("600519", weight=0.18)]
    details = {"600519": _clean_detail(kill_shots=["经销商渠道造假"])}
    proposed = [_proposed("600519", action="EXIT", tier="CORE", target_weight=0.0)]

    enriched, blocked = _pre_check_decisions(proposed, holdings, details)

    assert blocked == []
    assert enriched[0]["action"] == "EXIT"  # unchanged


def test_pre_check_allows_satellite_exit_regardless() -> None:
    holdings = [_satellite_holding("000651", weight=0.10)]
    details = {"000651": _clean_detail()}  # clean, still allow SATELLITE exit
    proposed = [
        _proposed("000651", action="EXIT", tier="SATELLITE", target_weight=0.0)
    ]

    enriched, blocked = _pre_check_decisions(proposed, holdings, details)

    assert blocked == []
    assert enriched[0]["action"] == "EXIT"


# ---------------------------------------------------------------------------
# _pre_check_decisions — weight caps and normalization
# ---------------------------------------------------------------------------

def test_pre_check_caps_oversized_single_position() -> None:
    proposed = [
        _proposed("600519", action="BUY", tier="CORE", target_weight=0.35)
    ]
    enriched, blocked = _pre_check_decisions(proposed, [], {})
    assert enriched[0]["target_weight"] == 0.20


def test_pre_check_normalizes_over_budget_total() -> None:
    # Three positions at 40% each = 120% total, should normalize to 100%
    proposed = [
        _proposed("A", action="BUY", tier="SATELLITE", target_weight=0.40),
        _proposed("B", action="BUY", tier="SATELLITE", target_weight=0.40),
        _proposed("C", action="BUY", tier="SATELLITE", target_weight=0.40),
    ]
    enriched, _ = _pre_check_decisions(proposed, [], {})
    # First capped to 0.20 each by single-pos rule → 60% total, fits
    for d in enriched:
        assert d["target_weight"] == 0.20


# ---------------------------------------------------------------------------
# _post_check_output — LLM output validation
# ---------------------------------------------------------------------------

def test_post_check_reblocks_if_llm_unblocks_core() -> None:
    """LLM output tried to change BLOCKED back to EXIT — post-check re-blocks."""
    pre_checked = [
        {
            "ticker": "600519",
            "name": "贵州茅台",
            "action": "BLOCKED",
            "target_weight": 0.18,
            "current_weight": 0.18,
            "position_tier": "CORE",
            "blocked_reason": "no kill_shots",
        }
    ]
    # LLM tries to change it back to EXIT
    llm_output = TraderOutput(
        meta=_meta(),
        market_regime=MarketRegime.NORMAL,
        orders=[
            TraderOrder(
                ticker="600519",
                name="贵州茅台",
                action=OrderAction.EXIT,  # LLM's attempt
                target_weight=0.0,
                current_weight=0.18,
                position_tier=PositionTier.CORE,
                urgency=OrderUrgency.IMMEDIATE,
                rationale="估值太贵了",
            )
        ],
    )
    fixed = _post_check_output(llm_output, pre_checked)

    assert fixed.orders[0].action == OrderAction.BLOCKED
    assert fixed.orders[0].target_weight == 0.18
    assert fixed.orders[0].blocked_reason == "no kill_shots"


def test_post_check_caps_oversized_llm_weight() -> None:
    pre_checked = [
        {
            "ticker": "600519",
            "name": "贵州茅台",
            "action": "BUY",
            "target_weight": 0.15,
            "current_weight": 0.0,
            "position_tier": "CORE",
        }
    ]
    llm_output = TraderOutput(
        meta=_meta(),
        orders=[
            TraderOrder(
                ticker="600519",
                name="贵州茅台",
                action=OrderAction.BUY,
                target_weight=0.35,  # LLM tried to oversize
                current_weight=0.0,
                position_tier=PositionTier.CORE,
                urgency=OrderUrgency.NORMAL,
                rationale="",
            )
        ],
    )
    fixed = _post_check_output(llm_output, pre_checked)
    assert fixed.orders[0].target_weight == 0.20


def test_post_check_reconstructs_dropped_orders() -> None:
    """LLM dropped a ticker — post-check must reconstruct it."""
    pre_checked = [
        {
            "ticker": "600519",
            "name": "贵州茅台",
            "action": "HOLD",
            "target_weight": 0.18,
            "current_weight": 0.18,
            "position_tier": "CORE",
        },
        {
            "ticker": "000651",
            "name": "格力电器",
            "action": "HOLD",
            "target_weight": 0.10,
            "current_weight": 0.10,
            "position_tier": "SATELLITE",
        },
    ]
    # LLM only output one of the two
    llm_output = TraderOutput(
        meta=_meta(),
        orders=[
            TraderOrder(
                ticker="600519",
                name="贵州茅台",
                action=OrderAction.HOLD,
                target_weight=0.18,
                current_weight=0.18,
                position_tier=PositionTier.CORE,
                urgency=OrderUrgency.NORMAL,
            )
        ],
    )
    fixed = _post_check_output(llm_output, pre_checked)
    tickers = {o.ticker for o in fixed.orders}
    assert tickers == {"600519", "000651"}


def test_post_check_preserves_blocked_count() -> None:
    pre_checked = [
        {
            "ticker": "600519",
            "name": "贵州茅台",
            "action": "BLOCKED",
            "target_weight": 0.18,
            "current_weight": 0.18,
            "position_tier": "CORE",
            "blocked_reason": "no kill_shots",
        }
    ]
    llm_output = TraderOutput(
        meta=_meta(),
        orders=[
            TraderOrder(
                ticker="600519",
                name="贵州茅台",
                action=OrderAction.BLOCKED,
                target_weight=0.18,
                current_weight=0.18,
                position_tier=PositionTier.CORE,
                urgency=OrderUrgency.NORMAL,
                blocked_reason="no kill_shots",
            )
        ],
    )
    fixed = _post_check_output(llm_output, pre_checked)
    assert fixed.blocked_orders_count == 1
