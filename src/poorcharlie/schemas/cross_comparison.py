"""Cross-comparison agent output schema."""

from __future__ import annotations

from pydantic import BaseModel

from poorcharlie.schemas.common import BaseAgentOutput


class PairwiseInsight(BaseModel, frozen=True):
    ticker_a: str
    ticker_b: str
    comparison: str  # e.g. "A优于B因为护城河更深且估值更低"
    dimension: str  # "quality" | "valuation" | "timing" | "risk"


class RankedCandidate(BaseModel, frozen=True):
    ticker: str
    name: str = ""
    rank: int  # 1 = best
    conviction_score: int  # 1-10
    strengths_vs_peers: list[str] = []
    weaknesses_vs_peers: list[str] = []
    portfolio_fit_notes: str = ""


class LollapaloozaCandidate(BaseModel, frozen=True):
    ticker: str
    factors: str  # e.g. "GREAT企业+CHEAP估值+明确催化剂+无kill shot"


class HardExclusion(BaseModel, frozen=True):
    ticker: str
    reason: str  # why permanently excluded


class CrossComparisonOutput(BaseAgentOutput):
    ranked_candidates: list[RankedCandidate]
    pairwise_insights: list[PairwiseInsight] = []
    concentration_warnings: list[str] = []
    lollapalooza_candidates: list[LollapaloozaCandidate] = []
    hard_exclusions: list[HardExclusion] = []
