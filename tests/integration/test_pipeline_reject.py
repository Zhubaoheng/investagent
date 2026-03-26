"""Integration test: pipeline stops at triage."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from investagent.llm import LLMClient
from investagent.schemas.company import CompanyIntake
from investagent.workflow.orchestrator import run_pipeline


def _mock_response(tool_input: dict) -> MagicMock:
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.input = tool_input
    response = MagicMock()
    response.content = [tool_block]
    response.model = "mock-model"
    response.usage = MagicMock()
    response.usage.input_tokens = 50
    response.usage.output_tokens = 50
    return response


def _triage_reject() -> dict:
    return {
        "decision": "REJECT",
        "explainability_score": {
            "business_model": 2,
            "competition_structure": 3,
            "financial_mapping": 1,
            "key_drivers": 2,
        },
        "fatal_unknowns": [
            "业务模式不透明，无法从公开信息理解核心收入来源",
            "财务报表不可得或严重不完整",
        ],
        "why_it_is_or_is_not_coverable": "该公司信息极度不透明，无法从公开信息构建有意义的分析",
        "next_step": "放弃研究",
    }


@pytest.mark.asyncio
async def test_pipeline_reject_at_triage():
    llm = LLMClient(client=MagicMock())
    llm.create_message = AsyncMock(
        return_value=_mock_response(_triage_reject())
    )

    intake = CompanyIntake(ticker="000001", name="不透明公司", exchange="SZSE")
    ctx = await run_pipeline(intake, llm=llm)

    # Pipeline should be stopped
    assert ctx.is_stopped()
    assert "Triage rejected" in ctx.stop_reason

    # Only triage should have run
    completed = ctx.completed_agents()
    assert completed == ["triage"]

    # LLM called only once
    assert llm.create_message.call_count == 1
