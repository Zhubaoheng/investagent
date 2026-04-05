"""Financial Quality Agent — score financial health across six dimensions."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from investagent.agents.base import BaseAgent
from investagent.schemas.common import BaseAgentOutput
from investagent.schemas.company import CompanyIntake
from investagent.schemas.financial_quality import FinancialQualityOutput, FinancialQualityScores

logger = logging.getLogger(__name__)


def compute_enterprise_quality(scores: FinancialQualityScores) -> str:
    """Deterministic enterprise quality from six dimension scores.

    LLM tends to default to AVERAGE. This function computes quality
    objectively from the scores the LLM already assigned (which are
    more granular and reliable than the quality string).
    """
    vals = [
        scores.per_share_growth,
        scores.return_on_capital,
        scores.cash_conversion,
        scores.leverage_safety,
        scores.capital_allocation,
        scores.moat_financial_trace,
    ]
    avg = sum(vals) / len(vals)
    min_s = min(vals)
    count_ge7 = sum(1 for v in vals if v >= 7)
    count_ge6 = sum(1 for v in vals if v >= 6)

    if avg < 3.0 or min_s <= 1:
        return "POOR"
    if avg < 4.5 or min_s < 3:
        return "BELOW_AVERAGE"
    if avg >= 7.5 and min_s >= 5 and count_ge7 >= 4:
        return "GREAT"
    if avg >= 6.0 and min_s >= 4 and count_ge6 >= 3:
        return "GOOD"
    return "AVERAGE"


class FinancialQualityAgent(BaseAgent):
    name: str = "financial_quality"

    def _output_type(self) -> type[BaseAgentOutput]:
        return FinancialQualityOutput

    def _agent_role_description(self) -> str:
        return (
            "You are the Financial Quality Agent. Your role is to evaluate the "
            "financial quality of a company across six scoring dimensions: "
            "per-share growth (EPS/FCF 5-year trends and dilution), return on "
            "capital (ROIC/ROE/ROA and margin stability), cash conversion "
            "(CFO/NI, FCF/NI, capex intensity), leverage safety (net debt/EBIT, "
            "interest coverage, liquidity), capital allocation (buyback quality, "
            "dividend sustainability, M&A track record), and moat financial traces "
            "(stable high margins/ROIC, scale effects). Each dimension is scored "
            "1-10. You determine whether the company passes the minimum quality "
            "standard required for further analysis. If it does not pass, the "
            "pipeline stops. You rely on structured financial data and must "
            "clearly distinguish between fact, inference, and unknown."
        )

    def _build_user_context(self, input_data: BaseModel, ctx: Any = None) -> dict[str, Any]:
        assert isinstance(input_data, CompanyIntake)
        result: dict[str, Any] = {
            "ticker": input_data.ticker,
            "name": input_data.name,
            "exchange": input_data.exchange,
        }
        if ctx is not None:
            from investagent.agents.context_helpers import data_for_financial_quality, format_json
            data = data_for_financial_quality(ctx)
            result["has_filing_data"] = data.get("has_filing", False)
            result["filing_json"] = format_json(data)
        else:
            result["has_filing_data"] = False
            result["filing_json"] = ""
        return result

    async def run(
        self, input_data: BaseModel, ctx: Any = None, *, max_retries: int = 2,
    ) -> FinancialQualityOutput:
        """Run LLM scoring, then override enterprise_quality deterministically."""
        output: FinancialQualityOutput = await super().run(input_data, ctx, max_retries=max_retries)  # type: ignore[assignment]
        computed = compute_enterprise_quality(output.scores)
        if computed != output.enterprise_quality:
            logger.info(
                "Financial quality post-process: %s → %s (scores=%s)",
                output.enterprise_quality, computed, output.scores.model_dump(),
            )
            return output.model_copy(update={"enterprise_quality": computed})
        return output
