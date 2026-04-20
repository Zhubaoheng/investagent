"""Tests for poorcharlie.workflow.change_detector."""

from __future__ import annotations

from datetime import date

from poorcharlie.schemas.candidate import (
    CandidateSnapshot,
    CandidateState,
    PortfolioHolding,
)
from poorcharlie.workflow.change_detector import (
    TriggerType,
    detect_change_triggers,
    has_cold_start_marker,
)


def _snap(
    ticker: str = "600519",
    name: str = "贵州茅台",
    final_label: str = "INVESTABLE",
    enterprise_quality: str = "GREAT",
    kill_shots: list[str] | None = None,
    accounting_risk_level: str = "GREEN",
    prev_final_label: str = "INVESTABLE",
    prev_enterprise_quality: str = "GREAT",
    prev_kill_shots: list[str] | None = None,
    prev_accounting_risk_level: str = "GREEN",
    has_prior_scan: bool = True,
    state: CandidateState = CandidateState.ANALYZED,
) -> CandidateSnapshot:
    return CandidateSnapshot(
        ticker=ticker,
        name=name,
        industry="食品饮料",
        final_label=final_label,
        enterprise_quality=enterprise_quality,
        kill_shots=kill_shots or [],
        accounting_risk_level=accounting_risk_level,
        prev_final_label=prev_final_label,
        prev_enterprise_quality=prev_enterprise_quality,
        prev_kill_shots=prev_kill_shots or [],
        prev_accounting_risk_level=prev_accounting_risk_level,
        prev_scan_date=date(2024, 5, 20) if has_prior_scan else None,
        scan_date=date(2024, 9, 23),
        state=state,
    )


def _holding(
    ticker: str, weight: float = 0.15, tier: str = "SATELLITE",
) -> PortfolioHolding:
    return PortfolioHolding(
        ticker=ticker,
        name="sample",
        industry="食品饮料",
        target_weight=weight,
        entry_date=date(2024, 1, 1),
        position_tier=tier,
    )


# ---------------------------------------------------------------------------
# No triggers — should short-circuit
# ---------------------------------------------------------------------------

def test_no_triggers_when_holding_unchanged() -> None:
    snap = _snap(state=CandidateState.HELD)
    holdings = [_holding("600519")]
    triggers = detect_change_triggers([snap], holdings)
    assert triggers == []


def test_no_triggers_when_watchlist_candidate_unchanged() -> None:
    # A WATCHLIST candidate, not held, not newly INVESTABLE — nothing to do
    snap = _snap(
        ticker="000858",
        final_label="WATCHLIST",
        prev_final_label="WATCHLIST",
    )
    holdings: list[PortfolioHolding] = []
    triggers = detect_change_triggers([snap], holdings)
    assert triggers == []


# ---------------------------------------------------------------------------
# T1: Quality downgrade on held position
# ---------------------------------------------------------------------------

def test_trigger_quality_downgrade_on_held() -> None:
    snap = _snap(
        enterprise_quality="AVERAGE",
        prev_enterprise_quality="GREAT",
        state=CandidateState.HELD,
    )
    triggers = detect_change_triggers([snap], [_holding("600519")])
    assert len(triggers) == 1
    assert triggers[0].trigger_type == TriggerType.QUALITY_DOWNGRADE


def test_quality_upgrade_does_not_trigger() -> None:
    snap = _snap(
        enterprise_quality="GREAT",
        prev_enterprise_quality="GOOD",
        state=CandidateState.HELD,
    )
    triggers = detect_change_triggers([snap], [_holding("600519")])
    assert triggers == []


def test_quality_downgrade_on_non_held_does_not_trigger() -> None:
    # Non-held candidate whose quality dropped — not our problem to act on
    snap = _snap(
        enterprise_quality="AVERAGE",
        prev_enterprise_quality="GREAT",
        final_label="WATCHLIST",
        prev_final_label="WATCHLIST",
    )
    triggers = detect_change_triggers([snap], [])
    assert triggers == []


# ---------------------------------------------------------------------------
# T2: New kill_shots on held position
# ---------------------------------------------------------------------------

def test_trigger_new_kill_shot_on_held() -> None:
    snap = _snap(
        kill_shots=["供应链集中风险", "关键专利到期"],
        prev_kill_shots=["供应链集中风险"],
        state=CandidateState.HELD,
    )
    triggers = detect_change_triggers([snap], [_holding("600519")])
    assert len(triggers) == 1
    assert triggers[0].trigger_type == TriggerType.NEW_KILL_SHOT


def test_same_kill_shots_do_not_trigger() -> None:
    snap = _snap(
        kill_shots=["供应链集中"],
        prev_kill_shots=["供应链集中"],
        state=CandidateState.HELD,
    )
    triggers = detect_change_triggers([snap], [_holding("600519")])
    assert triggers == []


# ---------------------------------------------------------------------------
# T3: Accounting risk turned RED
# ---------------------------------------------------------------------------

def test_trigger_accounting_red() -> None:
    snap = _snap(
        accounting_risk_level="RED",
        prev_accounting_risk_level="YELLOW",
        state=CandidateState.HELD,
    )
    triggers = detect_change_triggers([snap], [_holding("600519")])
    assert len(triggers) == 1
    assert triggers[0].trigger_type == TriggerType.ACCOUNTING_RED


def test_accounting_still_red_does_not_retrigger() -> None:
    snap = _snap(
        accounting_risk_level="RED",
        prev_accounting_risk_level="RED",
        state=CandidateState.HELD,
    )
    triggers = detect_change_triggers([snap], [_holding("600519")])
    # Same level doesn't re-fire
    assert triggers == []


# ---------------------------------------------------------------------------
# T4: New INVESTABLE candidate
# ---------------------------------------------------------------------------

def test_trigger_new_investable_not_previously_held() -> None:
    snap = _snap(
        ticker="000333",
        name="美的集团",
        final_label="INVESTABLE",
        prev_final_label="WATCHLIST",
    )
    triggers = detect_change_triggers([snap], [_holding("600519")])
    assert len(triggers) == 1
    assert triggers[0].trigger_type == TriggerType.NEW_INVESTABLE
    assert triggers[0].ticker == "000333"


def test_already_investable_held_does_not_retrigger() -> None:
    # Held, still INVESTABLE, no changes — baseline case
    snap = _snap(
        state=CandidateState.HELD,
        final_label="INVESTABLE",
        prev_final_label="INVESTABLE",
    )
    triggers = detect_change_triggers([snap], [_holding("600519")])
    assert triggers == []


def test_was_investable_still_investable_not_held_does_not_retrigger() -> None:
    # Candidate was already INVESTABLE last scan, still is, but isn't held
    # (maybe portfolio was full) — not a NEW trigger
    snap = _snap(
        ticker="000333",
        final_label="INVESTABLE",
        prev_final_label="INVESTABLE",
    )
    triggers = detect_change_triggers([snap], [_holding("600519")])
    assert triggers == []


# ---------------------------------------------------------------------------
# T5: Label reject on held position
# ---------------------------------------------------------------------------

def test_trigger_label_reject_on_held() -> None:
    snap = _snap(
        final_label="REJECT",
        prev_final_label="INVESTABLE",
        state=CandidateState.HELD,
    )
    triggers = detect_change_triggers([snap], [_holding("600519")])
    # REJECT makes candidate non-actionable, but we still track via state=HELD
    types = {t.trigger_type for t in triggers}
    assert TriggerType.LABEL_REJECT in types


def test_trigger_too_hard_on_held() -> None:
    snap = _snap(
        final_label="TOO_HARD",
        prev_final_label="INVESTABLE",
        state=CandidateState.HELD,
    )
    triggers = detect_change_triggers([snap], [_holding("600519")])
    types = {t.trigger_type for t in triggers}
    assert TriggerType.LABEL_REJECT in types


# ---------------------------------------------------------------------------
# Cold start
# ---------------------------------------------------------------------------

def test_cold_start_returns_synthetic_trigger() -> None:
    # No prior scan data anywhere
    snap = _snap(has_prior_scan=False, prev_final_label="", prev_enterprise_quality="")
    triggers = detect_change_triggers([snap], [])
    assert has_cold_start_marker(triggers)


def test_cold_start_marker_detector() -> None:
    from poorcharlie.workflow.change_detector import ChangeTrigger
    triggers = [
        ChangeTrigger(
            ticker="*",
            name="cold_start",
            trigger_type=TriggerType.NEW_INVESTABLE,
            detail="first scan",
        )
    ]
    assert has_cold_start_marker(triggers)

    triggers2 = [
        ChangeTrigger(
            ticker="600519",
            name="贵州茅台",
            trigger_type=TriggerType.NEW_INVESTABLE,
            detail="real trigger",
        )
    ]
    assert not has_cold_start_marker(triggers2)


# ---------------------------------------------------------------------------
# Multiple triggers in one scan
# ---------------------------------------------------------------------------

def test_trigger_detail_annotated_with_core_tier() -> None:
    """Triggers on held positions should include the tier (CORE/SATELLITE)
    in the detail string so downstream consumers can treat CORE differently.
    """
    snap = _snap(
        kill_shots=["新的致命风险"],
        prev_kill_shots=[],
        state=CandidateState.HELD,
    )
    holdings = [_holding("600519", tier="CORE")]
    triggers = detect_change_triggers([snap], holdings)
    assert len(triggers) == 1
    assert "[CORE]" in triggers[0].detail


def test_trigger_detail_annotated_with_satellite_tier() -> None:
    snap = _snap(
        enterprise_quality="AVERAGE",
        prev_enterprise_quality="GREAT",
        state=CandidateState.HELD,
    )
    holdings = [_holding("600519", tier="SATELLITE")]
    triggers = detect_change_triggers([snap], holdings)
    assert len(triggers) == 1
    assert "[SATELLITE]" in triggers[0].detail


def test_new_investable_trigger_not_annotated_with_tier() -> None:
    """Non-held triggers have no position_tier — nothing to annotate."""
    snap = _snap(
        ticker="000333",
        final_label="INVESTABLE",
        prev_final_label="WATCHLIST",
    )
    triggers = detect_change_triggers([snap], [_holding("600519", tier="CORE")])
    new_inv = [t for t in triggers if t.trigger_type == TriggerType.NEW_INVESTABLE]
    assert len(new_inv) == 1
    # No tier bracket for non-held triggers
    assert "[CORE]" not in new_inv[0].detail
    assert "[SATELLITE]" not in new_inv[0].detail


def test_multiple_triggers_accumulate() -> None:
    # Position A: quality downgraded
    snap_a = _snap(
        ticker="600519",
        enterprise_quality="AVERAGE",
        prev_enterprise_quality="GREAT",
        state=CandidateState.HELD,
    )
    # Position B: new kill shot
    snap_b = _snap(
        ticker="000858",
        kill_shots=["新的致命风险"],
        prev_kill_shots=[],
        state=CandidateState.HELD,
    )
    # New INVESTABLE candidate C
    snap_c = _snap(
        ticker="000333",
        final_label="INVESTABLE",
        prev_final_label="WATCHLIST",
    )
    holdings = [_holding("600519"), _holding("000858")]
    triggers = detect_change_triggers([snap_a, snap_b, snap_c], holdings)
    types = [t.trigger_type for t in triggers]
    assert TriggerType.QUALITY_DOWNGRADE in types
    assert TriggerType.NEW_KILL_SHOT in types
    assert TriggerType.NEW_INVESTABLE in types
