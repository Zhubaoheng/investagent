"""Portfolio Strategy Agent — produce position decisions with sizing rationale.

Part 2 agent that takes cross-comparison rankings and current holdings,
then outputs concrete BUY/HOLD/ADD/REDUCE/EXIT decisions per position
with conviction-weighted sizing.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from investagent.agents.base import BaseAgent
from investagent.schemas.common import BaseAgentOutput
from investagent.schemas.portfolio_strategy import PortfolioStrategyOutput


class StrategyHoldingInfo(BaseModel, frozen=True):
    ticker: str
    name: str = ""
    weight: float = 0.0
    industry: str = ""
    entry_reason: str = ""


class PortfolioStrategyInput(BaseModel, frozen=True):
    ranked_candidates: list[dict] = []  # RankedCandidate as dicts
    candidate_details: dict[str, dict] = {}  # ticker -> CandidateSnapshot summary
    current_holdings: list[StrategyHoldingInfo] = []
    available_cash_pct: float = 1.0


class PortfolioStrategyAgent(BaseAgent):
    """Produces per-position decisions with conviction-weighted sizing."""

    name: str = "portfolio_strategy"

    def _output_type(self) -> type[BaseAgentOutput]:
        return PortfolioStrategyOutput

    def _agent_role_description(self) -> str:
        return (
            "你是组合策略代理（Portfolio Strategy Agent），负责将横向对比排名转化为具体的持仓决策。"
            "你为每个标的做出 BUY/HOLD/ADD/REDUCE/EXIT 决策，并给出基于确信度的仓位分配理由。"
            "你遵循芒格原则：集中持仓最好的主意，不够好就持现金。"
        )

    def _build_user_context(
        self, input_data: BaseModel, ctx: Any = None,
    ) -> dict[str, Any]:
        data = (
            input_data
            if isinstance(input_data, PortfolioStrategyInput)
            else PortfolioStrategyInput.model_validate(input_data)
        )

        ranked = []
        for r in data.ranked_candidates:
            detail = data.candidate_details.get(r.get("ticker", ""), {})
            mos = detail.get("margin_of_safety_pct")
            mos_str = f"{mos:.0%}" if mos is not None else "N/A"
            ranked.append({
                "ticker": r.get("ticker", ""),
                "name": r.get("name", ""),
                "rank": r.get("rank", 0),
                "conviction_score": r.get("conviction_score", 5),
                "strengths": r.get("strengths_vs_peers") or [],
                "weaknesses": r.get("weaknesses_vs_peers") or [],
                "portfolio_fit_notes": r.get("portfolio_fit_notes", ""),
                "industry": detail.get("industry", "") or "未知",
                "final_label": detail.get("final_label", "") or "未知",
                "enterprise_quality": detail.get("enterprise_quality", "") or "未知",
                "price_vs_value": detail.get("price_vs_value", "") or "未知",
                "margin_of_safety_pct": mos_str,
                "thesis": (detail.get("thesis", "") or "无")[:200],
            })

        holdings = []
        for h in data.current_holdings:
            holdings.append({
                "ticker": h.ticker,
                "name": h.name,
                "weight": f"{h.weight:.0%}",
                "industry": h.industry or "未知",
                "entry_reason": h.entry_reason or "无",
            })

        return {
            "ranked_candidates": ranked,
            "current_holdings": holdings,
            "available_cash_pct": f"{data.available_cash_pct:.0%}",
            "has_holdings": len(holdings) > 0,
        }
