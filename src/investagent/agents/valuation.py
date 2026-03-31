"""Valuation & Look-through Return Agent — bear/base/bull expected returns.

LLM outputs per-method IV estimates independently. Python post-processing
takes the median to produce a deterministic intrinsic value, MoS, and
price_vs_value label.
"""

from __future__ import annotations

import logging
import statistics
from typing import Any

from pydantic import BaseModel

from investagent.agents.base import BaseAgent
from investagent.schemas.common import BaseAgentOutput
from investagent.schemas.company import CompanyIntake
from investagent.schemas.valuation import IntrinsicValueRange, ValuationOutput

logger = logging.getLogger(__name__)


def _post_process_valuation(
    output: ValuationOutput,
    price: float | None,
) -> ValuationOutput:
    """Deterministic post-processing: median IV → MoS → price_vs_value.

    Replaces LLM's blended IV with the median of per-method estimates.
    This eliminates the random weighting problem across runs.
    """
    estimates = output.per_method_estimates
    ivs = [e.iv_per_share for e in estimates if e.iv_per_share is not None and e.iv_per_share > 0]

    if not ivs or price is None or price <= 0:
        return output

    median_iv = statistics.median(ivs)
    bear_iv = round(median_iv * 0.75)
    bull_iv = round(median_iv * 1.30)
    mos = round((median_iv - price) / median_iv * 100, 1)

    # Deterministic price_vs_value
    meets = output.meets_hurdle_rate
    if not meets:
        pvv = "EXPENSIVE"
    elif mos > 20:
        pvv = "CHEAP"
    else:
        pvv = "FAIR"

    logger.info(
        "Valuation post-process: %d methods, IVs=%s, median=%.0f, price=%.0f, MoS=%.1f%%, %s",
        len(ivs), [round(v) for v in ivs], median_iv, price, mos, pvv,
    )

    currency = ""
    if output.intrinsic_value_per_share:
        currency = output.intrinsic_value_per_share.currency

    return output.model_copy(update={
        "intrinsic_value_per_share": IntrinsicValueRange(
            bear=bear_iv, base=round(median_iv), bull=bull_iv, currency=currency,
        ),
        "margin_of_safety_pct": mos,
        "price_vs_value": pvv,
    })


class ValuationAgent(BaseAgent):
    name: str = "valuation"

    def _output_type(self) -> type[BaseAgentOutput]:
        return ValuationOutput

    def _agent_role_description(self) -> str:
        return (
            "You are the Valuation & Look-through Return Agent. Your role is to "
            "estimate the expected look-through return of a company under three "
            "scenarios: bear, base, and bull. You use MULTIPLE valuation methods "
            "independently and output each method's IV estimate separately. "
            "The system will take the median of your estimates to produce the "
            "final intrinsic value. You also calculate friction-adjusted returns "
            "and compare against a hurdle rate (2× risk-free rate)."
        )

    def _build_user_context(self, input_data: BaseModel, ctx: Any = None) -> dict[str, Any]:
        assert isinstance(input_data, CompanyIntake)
        from investagent.config import Settings

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
            # Store price for post-processing
            self._current_price = data.get("market_snapshot", {}).get("price") if data.get("market_snapshot") else None
        else:
            result["has_filing_data"] = False
            result["filing_json"] = ""
            self._current_price = None
        return result

    async def run(
        self, input_data: BaseModel, ctx: Any = None, *, max_retries: int = 2,
    ) -> ValuationOutput:
        """Run LLM valuation, then deterministic post-processing."""
        output: ValuationOutput = await super().run(input_data, ctx, max_retries=max_retries)  # type: ignore[assignment]
        price = getattr(self, "_current_price", None)
        return _post_process_valuation(output, price)
