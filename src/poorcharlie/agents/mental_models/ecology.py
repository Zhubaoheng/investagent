"""Ecology / Evolution Agent — ecological niche, adaptability, survival probability."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from poorcharlie.agents.base import BaseAgent
from poorcharlie.schemas.common import BaseAgentOutput
from poorcharlie.schemas.company import CompanyIntake
from poorcharlie.schemas.mental_models import EcologyOutput


class EcologyAgent(BaseAgent):
    name: str = "ecology"

    def _output_type(self) -> type[BaseAgentOutput]:
        return EcologyOutput

    def _agent_role_description(self) -> str:
        return (
            "You are the Ecology Agent in a Munger-style value investing system. "
            "Your role is to evaluate a company's position within its competitive ecosystem "
            "using an evolutionary lens. You assess the company's ecological niche, whether "
            "its adaptability is strengthening or eroding, whether its performance is "
            "cyclical luck or structural advantage, and its long-term survival probability. "
            "You must clearly distinguish facts from inferences and flag unknowns."
        )

    def _build_user_context(self, input_data: BaseModel, ctx: Any = None) -> dict[str, Any]:
        assert isinstance(input_data, CompanyIntake)
        result: dict[str, Any] = {
            "ticker": input_data.ticker,
            "name": input_data.name,
            "exchange": input_data.exchange,
        }
        if ctx is not None:
            from poorcharlie.agents.context_helpers import data_for_ecology, format_json
            data = data_for_ecology(ctx)
            result["has_filing_data"] = data.get("has_filing", False)
            result["filing_json"] = format_json(data)
        else:
            result["has_filing_data"] = False
            result["filing_json"] = ""
        return result
