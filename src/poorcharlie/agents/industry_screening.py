"""Industry Screening Agent — Layer 1 of layered cross-comparison.

Compares candidates WITHIN a single industry group. Focuses on
competitive position, cyclicality, and who survives a downturn.
Output feeds into Layer 2 (CrossComparisonAgent) for cross-industry ranking.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from poorcharlie.agents.base import BaseAgent
from poorcharlie.schemas.common import BaseAgentOutput
from poorcharlie.schemas.industry_screening import IndustryScreeningOutput


class IndustryScreeningInput(BaseModel, frozen=True):
    industry_name: str
    candidates: list[dict] = []


class IndustryScreeningAgent(BaseAgent):

    name: str = "industry_screening"

    def _output_type(self) -> type[BaseAgentOutput]:
        return IndustryScreeningOutput

    def _agent_role_description(self) -> str:
        return (
            "你是行业筛选代理（Industry Screening Agent），负责在同一行业内部对候选标的做深度对比。"
            "你特别关注行业周期位置、竞争格局变化、谁能活过行业洗牌。"
            "你的输出将用于跨行业的最终排名——只有行业内最优的标的才会晋级。"
        )

    def _build_user_context(
        self, input_data: BaseModel, ctx: Any = None,
    ) -> dict[str, Any]:
        data = (
            input_data
            if isinstance(input_data, IndustryScreeningInput)
            else IndustryScreeningInput.model_validate(input_data)
        )

        candidates = []
        for c in data.candidates:
            mos = c.get("margin_of_safety_pct")
            mos_str = f"{mos:.0%}" if mos is not None else "N/A"
            candidates.append({
                "ticker": c.get("ticker", ""),
                "name": c.get("name", ""),
                "final_label": c.get("final_label", ""),
                "enterprise_quality": c.get("enterprise_quality", "") or "未知",
                "price_vs_value": c.get("price_vs_value", "") or "未知",
                "margin_of_safety_pct": mos_str,
                "meets_hurdle_rate": "是" if c.get("meets_hurdle_rate") else "否",
                "thesis": (c.get("thesis", "") or "无")[:500],
                "anti_thesis": (c.get("anti_thesis", "") or "无")[:500],
                "largest_unknowns": c.get("largest_unknowns") or [],
                "why_now": c.get("why_now", "") or "无",
            })

        return {
            "industry_name": data.industry_name,
            "candidate_count": len(candidates),
            "candidates": candidates,
        }
