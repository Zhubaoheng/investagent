"""Trader Agent schemas.

The Trader Agent is the Layer 4 arbiter between "portfolio intent" (what the
Portfolio Strategy agent wants to own) and "execution reality" (what actually
gets sent to the broker/backtester).

Its responsibilities:
  1. Hard-block CORE position exits that lack fundamental justification
     (no kill_shot, no RED accounting, etc).
  2. Assess market environment (crash? bull? sideways?) to tag order urgency.
  3. Enforce position caps and total-weight invariants.
  4. Produce a concrete order book the executor can follow.

It is NOT responsible for:
  - Picking stocks (that's PortfolioStrategy)
  - Judging business quality (that's the 13 upstream analysis agents)
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from poorcharlie.schemas.common import BaseAgentOutput
from poorcharlie.schemas.portfolio_strategy import PositionTier


class OrderAction(str, Enum):
    """Final order action after Trader review.

    Mirrors ActionType but adds BLOCKED for rules-rejected orders.
    """

    BUY = "BUY"
    ADD = "ADD"
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    EXIT = "EXIT"
    BLOCKED = "BLOCKED"  # the proposed sell was blocked by CORE-protection rules


class OrderUrgency(str, Enum):
    """How aggressively to execute the order.

    Mapped to execution days by the backtester:
      IMMEDIATE: 1-2 trading days (use for risk-off: RED accounting, kill_shots)
      NORMAL:    5 trading days, no-chase (default for ordinary rebalancing)
      PATIENT:   10+ trading days, wait for favorable prices (for CORE adds)
    """

    IMMEDIATE = "IMMEDIATE"
    NORMAL = "NORMAL"
    PATIENT = "PATIENT"


class MarketRegime(str, Enum):
    """Trader's read on overall market environment.

    Informs urgency of new BUYs and aggressiveness of SATELLITE rotation.
    """

    PANIC = "PANIC"             # broad selloff, INVESTABLE candidates suddenly cheap
    NORMAL = "NORMAL"
    EUPHORIA = "EUPHORIA"       # broad rally, be more skeptical of new BUYs
    UNKNOWN = "UNKNOWN"


class TraderOrder(BaseModel, frozen=True):
    """One concrete order after rules + LLM review."""

    ticker: str
    name: str = ""
    action: OrderAction
    target_weight: float  # 0.0 for EXIT/BLOCKED-sell, else 0.05-0.30
    current_weight: float = 0.0
    position_tier: PositionTier = PositionTier.SATELLITE
    urgency: OrderUrgency = OrderUrgency.NORMAL
    rationale: str = ""
    # When action == BLOCKED, describe why the rules rejected the proposal.
    blocked_reason: str = ""


class TraderInput(BaseModel, frozen=True):
    """Input to the Trader Agent: what PortfolioStrategy proposed + context."""

    proposed_decisions: list[dict] = Field(
        default_factory=list,
        description="list of PositionDecision dicts from PortfolioStrategy",
    )
    current_holdings: list[dict] = Field(
        default_factory=list,
        description="current PortfolioHolding dicts (incl. position_tier)",
    )
    candidate_details: dict[str, dict] = Field(
        default_factory=dict,
        description="ticker -> CandidateSnapshot dict (quality, kill_shots, ...)",
    )
    change_triggers: list[dict] = Field(
        default_factory=list,
        description="Same change_triggers that woke up PortfolioStrategy",
    )


class TraderOutput(BaseAgentOutput):
    """Final execution plan validated by rules + LLM."""

    market_regime: MarketRegime = MarketRegime.UNKNOWN
    market_assessment: str = ""          # one-sentence overview
    orders: list[TraderOrder]
    blocked_orders_count: int = 0
    overall_urgency: OrderUrgency = OrderUrgency.NORMAL
    execution_plan_summary: str = ""
