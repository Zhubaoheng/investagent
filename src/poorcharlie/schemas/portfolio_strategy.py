"""Portfolio strategy agent output schema."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from poorcharlie.schemas.common import BaseAgentOutput


class ActionType(str, Enum):
    BUY = "BUY"
    HOLD = "HOLD"
    ADD = "ADD"
    REDUCE = "REDUCE"
    EXIT = "EXIT"


class PositionTier(str, Enum):
    """Holding tier — governs how defensively a position is protected.

    CORE: "compounding machines" (GREAT + INVESTABLE). Only sold when the
          business is permanently damaged (kill_shot, RED accounting, quality
          → POOR). Never sold due to short-term valuation being rich.
          Munger: "If you can buy a few great companies, then you can sit
          on your ass — that's a good thing."

    SATELLITE: everything else. Normal rotation rules apply — may be
          trimmed or exited when a clearly better opportunity appears or
          when valuation becomes obviously stretched.
    """

    CORE = "CORE"
    SATELLITE = "SATELLITE"


class PositionDecision(BaseModel, frozen=True):
    ticker: str
    name: str = ""
    action: ActionType
    target_weight: float  # 0.0 for EXIT, 0.05-0.30 for others
    current_weight: float = 0.0
    conviction_score: int = 5  # 1-10, from cross-comparison
    position_tier: PositionTier = PositionTier.SATELLITE  # defensive default
    reason: str = ""
    sizing_rationale: str = ""


class PortfolioStrategyOutput(BaseAgentOutput):
    position_decisions: list[PositionDecision]
    cash_weight: float
    industry_distribution: dict[str, float] = {}
    portfolio_quality_summary: str = ""
    risk_notes: list[str] = []
    rebalance_summary: list[str] = []
