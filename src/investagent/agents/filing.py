"""Filing Structuring Skill — standardize financials into structured tables."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from investagent.agents.base import BaseAgent
from investagent.schemas.common import BaseAgentOutput
from investagent.schemas.company import CompanyIntake
from investagent.schemas.filing import FilingOutput


class FilingAgent(BaseAgent):
    name: str = "filing"

    def _output_type(self) -> type[BaseAgentOutput]:
        return FilingOutput

    def _agent_role_description(self) -> str:
        return (
            "You are the Filing Structuring Skill. Your role is to transform "
            "raw financial filings into a standardized, structured data layer. "
            "You extract the three financial statements (income, balance sheet, "
            "cash flow), segment data, accounting policies with raw text, debt "
            "schedules, special items, concentration data, capital allocation "
            "records, footnote extracts, and risk factors. You work across "
            "A-share (CAS), HK (IFRS/HKFRS), and US ADR (US GAAP/IFRS) "
            "filings. You must preserve critical raw text for accounting "
            "policies, footnotes, and risk factors."
        )

    def _build_user_context(
        self, input_data: BaseModel, ctx: Any = None,
    ) -> dict[str, Any]:
        assert isinstance(input_data, CompanyIntake)
        result: dict[str, Any] = {
            "ticker": input_data.ticker,
            "name": input_data.name,
            "exchange": input_data.exchange,
        }
        if ctx is not None:
            try:
                ctx.get_result("info_capture")
                result["has_info_capture"] = True
            except KeyError:
                result["has_info_capture"] = False
        else:
            result["has_info_capture"] = False
        return result
