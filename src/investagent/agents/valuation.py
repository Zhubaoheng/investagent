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
    hurdle_rate: float | None = None,
) -> ValuationOutput:
    """Deterministic post-processing: median IV → MoS → meets_hurdle → price_vs_value.

    Replaces LLM's blended IV with the median of per-method estimates.
    Also recomputes meets_hurdle_rate from friction_adjusted_return to
    prevent contradictions (e.g., MoS=-86% but hurdle=True).
    """
    estimates = output.per_method_estimates
    ivs = [e.iv_per_share for e in estimates if e.iv_per_share is not None and e.iv_per_share > 0]

    if not ivs or price is None or price <= 0:
        return output

    # Filter out wildly implausible IV estimates (likely LLM errors)
    sane_ivs = [v for v in ivs if price * 0.05 <= v <= price * 20]
    if not sane_ivs:
        logger.warning(
            "All IV estimates look implausible vs price=%.2f: %s — using raw median",
            price, [round(v, 1) for v in ivs],
        )
        sane_ivs = ivs  # fall back to raw if all filtered out

    median_iv = statistics.median(sane_ivs)
    bear_iv = round(median_iv * 0.75)
    bull_iv = round(median_iv * 1.30)
    mos = round((median_iv - price) / median_iv * 100, 1)
    mos = max(-100.0, min(100.0, mos))  # clamp to [-100%, 100%]

    # Recompute meets_hurdle from friction-adjusted return (not LLM judgment)
    meets = output.meets_hurdle_rate
    base_return = None
    if output.friction_adjusted_return:
        base_return = output.friction_adjusted_return.base
    if base_return is not None and hurdle_rate is not None:
        meets = base_return >= hurdle_rate
    # Failsafe: extreme overvaluation cannot meet hurdle
    if mos < -50 and meets:
        meets = False
        logger.info("Valuation post-process: MoS=%.1f%% override meets_hurdle to False", mos)

    # Deterministic price_vs_value
    if not meets:
        pvv = "EXPENSIVE"
    elif mos > 20:
        pvv = "CHEAP"
    else:
        pvv = "FAIR"

    logger.info(
        "Valuation post-process: %d methods, IVs=%s, median=%.0f, price=%.0f, MoS=%.1f%%, hurdle=%s, %s",
        len(ivs), [round(v) for v in ivs], median_iv, price, mos, meets, pvv,
    )

    currency = ""
    if output.intrinsic_value_per_share:
        currency = output.intrinsic_value_per_share.currency

    return output.model_copy(update={
        "intrinsic_value_per_share": IntrinsicValueRange(
            bear=bear_iv, base=round(median_iv), bull=bull_iv, currency=currency,
        ),
        "margin_of_safety_pct": mos,
        "meets_hurdle_rate": meets,
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
            # Store price + hurdle for post-processing
            self._current_price = data.get("market_snapshot", {}).get("price") if data.get("market_snapshot") else None
            self._current_hurdle_rate = hurdle
        else:
            result["has_filing_data"] = False
            result["filing_json"] = ""
            self._current_price = None
            self._current_hurdle_rate = None
        return result

    async def run(
        self, input_data: BaseModel, ctx: Any = None, *, max_retries: int = 2,
    ) -> ValuationOutput:
        """Run LLM valuation, then deterministic post-processing."""
        output: ValuationOutput = await super().run(input_data, ctx, max_retries=max_retries)  # type: ignore[assignment]
        price = getattr(self, "_current_price", None)
        hurdle = getattr(self, "_current_hurdle_rate", None)
        return _post_process_valuation(output, price, hurdle)
