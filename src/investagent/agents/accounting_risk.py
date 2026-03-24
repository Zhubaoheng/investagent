"""Accounting Risk Agent — detect accounting method changes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from investagent.agents.base import BaseAgent
from investagent.schemas.common import BaseAgentOutput


class AccountingRiskAgent(BaseAgent):
    name: str = "accounting_risk"

    def _output_type(self) -> type[BaseAgentOutput]:
        raise NotImplementedError

    def _agent_role_description(self) -> str:
        raise NotImplementedError

    def _build_user_context(self, input_data: BaseModel) -> dict[str, Any]:
        raise NotImplementedError
