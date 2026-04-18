"""Cross-Comparison Agent — rank candidates against each other.

Part 2 agent that performs horizontal comparison across all actionable
candidates. Produces conviction-scored rankings and pairwise insights
that feed into PortfolioStrategyAgent.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from poorcharlie.agents.base import BaseAgent
from poorcharlie.schemas.common import BaseAgentOutput
from poorcharlie.schemas.cross_comparison import CrossComparisonOutput


class CrossComparisonInput(BaseModel, frozen=True):
    candidates: list[dict] = []  # CandidateSnapshot summaries as dicts
    industry_insights: list[dict] = []  # [{industry, insight, cycle}] from Layer 1


class CrossComparisonAgent(BaseAgent):
    """Ranks investment candidates against each other."""

    name: str = "cross_comparison"

    def _output_type(self) -> type[BaseAgentOutput]:
        return CrossComparisonOutput

    def _agent_role_description(self) -> str:
        return (
            "你是横向对比代理（Cross-Comparison Agent），你的任务是将多个通过分析流水线的候选标的"
            "放在一起横向对比，按投资吸引力排序，给出确信度评分和关键对比洞察。"
            "你不做单独的公司分析——每个候选标的已经由 10 个专业 agent 独立分析过。"
            "你的价值在于：回答'如果只能选 5-10 只，应该选哪些？为什么 A 比 B 更好？'"
        )

    def _build_user_context(
        self, input_data: BaseModel, ctx: Any = None,
    ) -> dict[str, Any]:
        data = (
            input_data
            if isinstance(input_data, CrossComparisonInput)
            else CrossComparisonInput.model_validate(input_data)
        )

        candidates = []
        for c in data.candidates:
            mos = c.get("margin_of_safety_pct")
            mos_str = f"{mos:.0%}" if mos is not None else "N/A"
            candidates.append({
                "ticker": c.get("ticker", ""),
                "name": c.get("name", ""),
                "industry": c.get("industry", "") or "未知",
                "final_label": c.get("final_label", ""),
                "enterprise_quality": c.get("enterprise_quality", "") or "未知",
                "price_vs_value": c.get("price_vs_value", "") or "未知",
                "margin_of_safety_pct": mos_str,
                "meets_hurdle_rate": "是" if c.get("meets_hurdle_rate") else "否",
                "thesis": (c.get("thesis", "") or "无")[:300],
                "anti_thesis": (c.get("anti_thesis", "") or "无")[:300],
                "largest_unknowns": c.get("largest_unknowns") or [],
                "expected_return_summary": c.get("expected_return_summary", "") or "无",
                "why_now": c.get("why_now", "") or "无",
            })

        # Detect industry concentration
        industry_counts: dict[str, int] = {}
        for c in candidates:
            ind = c["industry"]
            industry_counts[ind] = industry_counts.get(ind, 0) + 1

        industry_insights = data.industry_insights or []

        return {
            "candidates": candidates,
            "candidate_count": len(candidates),
            "industry_counts": industry_counts,
            "has_industry_insights": bool(industry_insights),
            "industry_insights": industry_insights,
        }
