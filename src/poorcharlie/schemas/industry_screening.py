"""Industry screening agent output schema (Layer 1 of layered cross-comparison)."""

from __future__ import annotations

from pydantic import BaseModel

from poorcharlie.schemas.common import BaseAgentOutput


class ScreenedCandidate(BaseModel, frozen=True):
    ticker: str
    name: str = ""
    rank: int
    conviction_score: int  # 1-10
    strengths_in_group: list[str] = []
    weaknesses_in_group: list[str] = []
    advance: bool = True  # whether this candidate advances to Layer 2


class IndustryScreeningOutput(BaseAgentOutput):
    ranked_candidates: list[ScreenedCandidate]
    group_insight: str = ""  # industry-level judgment, e.g. "产能过剩周期底部"
    cycle_assessment: str = ""  # PEAK / NEUTRAL / TROUGH / NOT_CYCLICAL
    hard_exclusions: list[str] = []  # tickers excluded for permanent loss risk
