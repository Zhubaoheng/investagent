"""Tests for poorcharlie.store.candidate_store."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from poorcharlie.schemas.candidate import (
    CandidateSnapshot,
    CandidateState,
    PortfolioHolding,
)
from poorcharlie.store.candidate_store import CandidateStore


def _sample_results() -> list[dict]:
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
            "thesis": "高ROE消费品龙头",
            "anti_thesis": "增长放缓",
            "largest_unknowns": ["消费降级影响"],
            "expected_return_summary": "年化15%",
            "why_now_or_why_not_now": "估值合理",
        },
        {
            "ticker": "000858",
            "name": "五粮液",
            "industry": "食品饮料",
            "final_label": "WATCHLIST",
            "enterprise_quality": "GOOD",
            "price_vs_value": "CHEAP",
            "margin_of_safety_pct": 0.30,
            "meets_hurdle_rate": True,
            "thesis": "白酒第二",
            "anti_thesis": "渠道库存",
        },
        {
            "ticker": "601398",
            "name": "工商银行",
            "industry": "银行",
            "final_label": "REJECT",
        },
        {
            "ticker": "000001",
            "name": "平安银行",
            "industry": "银行",
            "final_label": "TOO_HARD",
        },
    ]


class TestCandidateStore:
    def test_fresh_store(self, tmp_path: Path):
        store = CandidateStore(tmp_path / "store.json")
        assert store.get_actionable_candidates() == []
        assert store.get_current_holdings() == []
        assert store.to_portfolio_decisions() == {}

    def test_ingest_filters_actionable(self, tmp_path: Path):
        store = CandidateStore(tmp_path / "store.json")
        store.ingest_scan_results(_sample_results(), date(2024, 5, 6))

        # Only INVESTABLE + DEEP_DIVE enter decision pipeline
        # (WATCHLIST monitored via triggers, not cross-comparison)
        actionable = store.get_actionable_candidates()
        tickers = {c.ticker for c in actionable}
        assert tickers == {"600519"}  # 000858 is WATCHLIST → excluded

    def test_ingest_snapshot_fields(self, tmp_path: Path):
        store = CandidateStore(tmp_path / "store.json")
        store.ingest_scan_results(_sample_results(), date(2024, 5, 6))

        actionable = store.get_actionable_candidates()
        maotai = next(c for c in actionable if c.ticker == "600519")
        assert maotai.enterprise_quality == "GREAT"
        assert maotai.margin_of_safety_pct == 0.15
        assert maotai.thesis == "高ROE消费品龙头"
        assert maotai.scan_date == date(2024, 5, 6)
        assert maotai.state == CandidateState.ANALYZED

    def test_ingest_updates_existing(self, tmp_path: Path):
        store = CandidateStore(tmp_path / "store.json")
        store.ingest_scan_results(_sample_results(), date(2024, 5, 6))

        # Update with new scan
        updated = [{
            "ticker": "600519",
            "name": "贵州茅台",
            "final_label": "INVESTABLE",
            "enterprise_quality": "GREAT",
            "price_vs_value": "CHEAP",  # changed
            "margin_of_safety_pct": 0.25,  # changed
        }]
        store.ingest_scan_results(updated, date(2024, 9, 2))

        actionable = store.get_actionable_candidates()
        maotai = next(c for c in actionable if c.ticker == "600519")
        assert maotai.price_vs_value == "CHEAP"
        assert maotai.margin_of_safety_pct == 0.25
        assert maotai.scan_date == date(2024, 9, 2)

    def test_ingest_removes_rejected(self, tmp_path: Path):
        store = CandidateStore(tmp_path / "store.json")
        store.ingest_scan_results(_sample_results(), date(2024, 5, 6))
        # Only INVESTABLE (600519) in decision pipeline; 000858 is WATCHLIST
        assert len(store.get_actionable_candidates()) == 1

        # Reject 茅台 in next scan — should drop from actionable
        store.ingest_scan_results(
            [{"ticker": "600519", "final_label": "REJECT"}],
            date(2024, 9, 2),
        )
        tickers = {c.ticker for c in store.get_actionable_candidates()}
        assert "600519" not in tickers

    def test_update_holdings(self, tmp_path: Path):
        store = CandidateStore(tmp_path / "store.json")
        store.ingest_scan_results(_sample_results(), date(2024, 5, 6))

        holdings = [
            PortfolioHolding(
                ticker="600519", name="贵州茅台", industry="食品饮料",
                target_weight=0.25, entry_date=date(2024, 5, 6),
                entry_reason="GREAT+FAIR",
            ),
        ]
        store.update_holdings(holdings)

        assert len(store.get_current_holdings()) == 1
        assert store.to_portfolio_decisions() == {"600519": 0.25}

    def test_holdings_state_transitions(self, tmp_path: Path):
        store = CandidateStore(tmp_path / "store.json")
        store.ingest_scan_results(_sample_results(), date(2024, 5, 6))

        # Hold 600519
        holdings = [
            PortfolioHolding(
                ticker="600519", target_weight=0.25, entry_date=date(2024, 5, 6),
            ),
        ]
        store.update_holdings(holdings)

        actionable = store.get_actionable_candidates()
        maotai = next(c for c in actionable if c.ticker == "600519")
        assert maotai.state == CandidateState.HELD

        # Exit 600519
        store.update_holdings([])
        # Should not be actionable after exit (state is EXITED)
        # But it should still exist in candidates
        assert store.to_portfolio_decisions() == {}

    def test_held_state_preserved_on_reingest(self, tmp_path: Path):
        """HELD stocks keep HELD state when re-ingested with new analysis."""
        store = CandidateStore(tmp_path / "store.json")
        store.ingest_scan_results(_sample_results(), date(2024, 5, 6))

        # Hold 600519 (INVESTABLE)
        store.update_holdings([
            PortfolioHolding(
                ticker="600519", target_weight=0.25, entry_date=date(2024, 5, 6),
            ),
        ])
        c = next(c for c in store.get_actionable_candidates() if c.ticker == "600519")
        assert c.state == CandidateState.HELD

        # Re-ingest with downgraded label (INVESTABLE → WATCHLIST)
        store.ingest_scan_results(
            [{"ticker": "600519", "name": "贵州茅台", "final_label": "WATCHLIST",
              "enterprise_quality": "GREAT", "price_vs_value": "FAIR"}],
            date(2024, 9, 2),
        )

        # State should still be HELD, but label updated
        c = next(c for c in store.get_actionable_candidates() if c.ticker == "600519")
        assert c.state == CandidateState.HELD
        assert c.final_label == "WATCHLIST"
        assert c.scan_date == date(2024, 9, 2)

    def test_get_candidates_for_rescan(self, tmp_path: Path):
        store = CandidateStore(tmp_path / "store.json")
        store.ingest_scan_results(_sample_results(), date(2023, 5, 6))

        # Hold 600519
        store.update_holdings([
            PortfolioHolding(
                ticker="600519", target_weight=0.25, entry_date=date(2023, 5, 6),
            ),
        ])

        rescan = store.get_candidates_for_rescan(staleness_days=180)
        tickers = {r["ticker"] for r in rescan}
        # Holdings always rescan
        assert "600519" in tickers
        # Stale WATCHLIST should rescan (scan_date is >180 days old)
        assert "000858" in tickers

    def test_persistence_round_trip(self, tmp_path: Path):
        path = tmp_path / "store.json"
        store = CandidateStore(path)
        store.ingest_scan_results(_sample_results(), date(2024, 5, 6))
        store.update_holdings([
            PortfolioHolding(
                ticker="600519", name="贵州茅台", target_weight=0.25,
                entry_date=date(2024, 5, 6),
            ),
        ])
        store.save()

        # Reload
        store2 = CandidateStore(path)
        # 600519 is HELD → included; 000858 is WATCHLIST (not held) → excluded
        assert len(store2.get_actionable_candidates()) == 1
        assert len(store2.get_current_holdings()) == 1
        assert store2.to_portfolio_decisions() == {"600519": 0.25}

    def test_empty_results_no_crash(self, tmp_path: Path):
        store = CandidateStore(tmp_path / "store.json")
        store.ingest_scan_results([], date(2024, 5, 6))
        assert store.get_actionable_candidates() == []

    def test_why_now_field_mapping(self, tmp_path: Path):
        """Both why_now_or_why_not_now and why_now field names are handled."""
        store = CandidateStore(tmp_path / "store.json")
        store.ingest_scan_results(
            [{"ticker": "600519", "final_label": "INVESTABLE",
              "why_now_or_why_not_now": "入场窗口"}],
            date(2024, 5, 6),
        )
        c = store.get_actionable_candidates()[0]
        assert c.why_now == "入场窗口"
