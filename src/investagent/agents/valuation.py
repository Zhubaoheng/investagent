"""Valuation & Look-through Return Agent — bear/base/bull expected returns."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from investagent.agents.base import BaseAgent
from investagent.schemas.common import BaseAgentOutput
from investagent.schemas.company import CompanyIntake
from investagent.schemas.valuation import ValuationOutput


class ValuationAgent(BaseAgent):
    name: str = "valuation"

    def _output_type(self) -> type[BaseAgentOutput]:
        return ValuationOutput

    def _agent_role_description(self) -> str:
        return (
            "You are the Valuation & Look-through Return Agent. Your role is to "
            "estimate the expected look-through return of a company under three "
            "scenarios: bear, base, and bull. You calculate normalized earnings "
            "yield and owner earnings / FCF yield, project per-share intrinsic "
            "value growth based on ROIC reinvestment, and subtract friction "
            "(tax, transaction costs) to produce friction-adjusted returns. "
            "You then compare the base-case return against a hurdle rate "
            "(2× risk-free rate for the reporting currency)."
        )

    def _build_user_context(self, input_data: BaseModel, ctx: Any = None) -> dict[str, Any]:
        assert isinstance(input_data, CompanyIntake)
        from investagent.config import Settings

        # Determine currency and hurdle rate
        currency = "USD"
        if ctx is not None:
            try:
                filing = ctx.get_result("filing")
                if hasattr(filing, "filing_meta"):
                    currency = filing.filing_meta.currency or "USD"
            except KeyError:
                pass

        settings = Settings()
        hurdle = settings.get_hurdle_rate(currency)
        rfr = settings.risk_free_rates.get(currency, 0.04)

        result: dict[str, Any] = {
            "ticker": input_data.ticker,
            "name": input_data.name,
            "exchange": input_data.exchange,
            "hurdle_rate": hurdle,
            "hurdle_rate_pct": f"{hurdle * 100:.1f}%",
            "risk_free_rate": rfr,
            "risk_free_rate_pct": f"{rfr * 100:.1f}%",
            "currency": currency,
        }
        if ctx is not None:
            from investagent.agents.context_helpers import data_for_valuation, format_json
            data = data_for_valuation(ctx)
            result["has_filing_data"] = data.get("has_filing", False)
            result["filing_json"] = format_json(data)
            result["market_snapshot"] = data.get("market_snapshot")
        else:
            result["has_filing_data"] = False
            result["filing_json"] = ""
        return result
