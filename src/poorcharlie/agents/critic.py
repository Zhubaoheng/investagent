"""Critic Agent — adversarial: find kill shots and permanent loss risks."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from poorcharlie.agents.base import BaseAgent
from poorcharlie.schemas.common import BaseAgentOutput
from poorcharlie.schemas.company import CompanyIntake
from poorcharlie.schemas.critic import CriticOutput


class CriticAgent(BaseAgent):
    name: str = "critic"

    def _output_type(self) -> type[BaseAgentOutput]:
        return CriticOutput

    def _agent_role_description(self) -> str:
        return (
            "You are the Critic Agent — an adversarial devil's advocate whose "
            "sole purpose is to DEMOLISH the investment thesis. You never retell "
            "the bull story. You systematically identify kill shots, permanent "
            "capital loss risks, moat destruction paths, and management failure "
            "modes. You prioritize irreversible harm over temporary setbacks. "
            "You must identify at least 3 thesis-destroying risks and judge "
            "which risks are already priced in versus those that are not. "
            "Your output is the strongest possible bear case."
        )

    def _build_user_context(self, input_data: BaseModel, ctx: Any = None) -> dict[str, Any]:
        assert isinstance(input_data, CompanyIntake)
        result: dict[str, Any] = {
            "ticker": input_data.ticker,
            "name": input_data.name,
            "exchange": input_data.exchange,
        }
        if ctx is not None:
            from poorcharlie.agents.context_helpers import data_for_critic, format_json
            data = data_for_critic(ctx)
            result["has_filing_data"] = data.get("filing", {}).get("has_filing", False)
            result["filing_json"] = format_json(data.get("filing", {}))
            result["upstream_json"] = format_json(data.get("upstream", {}))
        else:
            result["has_filing_data"] = False
            result["filing_json"] = ""
            result["upstream_json"] = ""
        return result
