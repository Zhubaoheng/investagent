"""Math / Compounding Agent — compounding engine, capital returns, sustainability."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from poorcharlie.agents.base import BaseAgent
from poorcharlie.schemas.common import BaseAgentOutput
from poorcharlie.schemas.company import CompanyIntake
from poorcharlie.schemas.mental_models import CompoundingOutput


class CompoundingAgent(BaseAgent):
    name: str = "compounding"

    def _output_type(self) -> type[BaseAgentOutput]:
        return CompoundingOutput

    def _agent_role_description(self) -> str:
        return (
            "You are the Compounding Agent in a Munger-style value investing system. "
            "Your role is to evaluate the long-term compounding potential of a company "
            "by analyzing its return on invested capital (ROIC), reinvestment runway, "
            "incremental returns on capital, and per-share intrinsic value growth rate. "
            "You assess how long high returns can be sustained and whether the compounding "
            "engine is structurally sound. You must clearly distinguish facts from inferences "
            "and flag unknowns."
        )

    def _build_user_context(self, input_data: BaseModel, ctx: Any = None) -> dict[str, Any]:
        assert isinstance(input_data, CompanyIntake)
        result: dict[str, Any] = {
            "ticker": input_data.ticker,
            "name": input_data.name,
            "exchange": input_data.exchange,
        }
        if ctx is not None:
            from poorcharlie.agents.context_helpers import data_for_compounding, format_json
            data = data_for_compounding(ctx)
            result["has_filing_data"] = data.get("has_filing", False)
            result["filing_json"] = format_json(data)
        else:
            result["has_filing_data"] = False
            result["filing_json"] = ""
        return result
