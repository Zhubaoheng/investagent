"""Tests for investagent.agents.triage."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from investagent.agents.base import AgentOutputError
from investagent.agents.triage import TriageAgent
from investagent.llm import LLMClient
from investagent.schemas.company import CompanyIntake
from investagent.schemas.triage import TriageDecision


def _intake() -> CompanyIntake:
    return CompanyIntake(
        ticker="600519",
        name="贵州茅台",
        exchange="SSE",
    )


def _mock_llm() -> LLMClient:
    return LLMClient(client=MagicMock())


def _mock_response(tool_input: dict) -> MagicMock:
    """Create a mock Anthropic Message with a tool_use block."""
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.input = tool_input

    response = MagicMock()
    response.content = [tool_block]
    response.model = "claude-sonnet-4-20250514"
    response.usage = MagicMock()
    response.usage.input_tokens = 100
    response.usage.output_tokens = 200
    return response


def _triage_tool_input(decision: str = "PASS") -> dict:
    return {
        "decision": decision,
        "explainability_score": {
            "business_model": 9,
            "competition_structure": 8,
            "financial_mapping": 8,
            "key_drivers": 9,
        },
        "fatal_unknowns": [],
        "why_it_is_or_is_not_coverable": "茅台商业模式清晰，白酒行业竞争格局稳定",
        "next_step": "进入信息采集阶段",
    }


@pytest.mark.asyncio
async def test_triage_pass():
    llm = _mock_llm()
    llm.create_message = AsyncMock(
        return_value=_mock_response(_triage_tool_input("PASS"))
    )
    agent = TriageAgent(llm)
    result = await agent.run(_intake())
    assert result.decision == TriageDecision.PASS
    assert result.meta.agent_name == "triage"
    assert result.meta.token_usage == 300


@pytest.mark.asyncio
async def test_triage_reject():
    llm = _mock_llm()
    tool_input = _triage_tool_input("REJECT")
    tool_input["fatal_unknowns"] = ["VIE结构不透明"]
    llm.create_message = AsyncMock(
        return_value=_mock_response(tool_input)
    )
    agent = TriageAgent(llm)
    result = await agent.run(_intake())
    assert result.decision == TriageDecision.REJECT
    assert len(result.fatal_unknowns) == 1


@pytest.mark.asyncio
async def test_triage_watch():
    llm = _mock_llm()
    llm.create_message = AsyncMock(
        return_value=_mock_response(_triage_tool_input("WATCH"))
    )
    agent = TriageAgent(llm)
    result = await agent.run(_intake())
    assert result.decision == TriageDecision.WATCH


@pytest.mark.asyncio
async def test_triage_meta_is_server_generated():
    """Server-side meta should override anything the LLM emits."""
    tool_input = _triage_tool_input()
    # Simulate LLM sneaking in a meta (should be overwritten)
    tool_input["meta"] = {
        "agent_name": "hacked",
        "timestamp": "2020-01-01T00:00:00Z",
        "model_used": "fake",
        "token_usage": 0,
    }
    llm = _mock_llm()
    llm.create_message = AsyncMock(
        return_value=_mock_response(tool_input)
    )
    agent = TriageAgent(llm)
    result = await agent.run(_intake())
    assert result.meta.agent_name == "triage"
    assert result.meta.model_used == "claude-sonnet-4-20250514"
    assert result.meta.token_usage == 300


@pytest.mark.asyncio
async def test_triage_no_tool_use_raises():
    """If LLM returns no tool_use block, raise AgentOutputError."""
    text_block = MagicMock()
    text_block.type = "text"

    response = MagicMock()
    response.content = [text_block]
    response.model = "claude-sonnet-4-20250514"
    response.usage = MagicMock()
    response.usage.input_tokens = 50
    response.usage.output_tokens = 100

    llm = _mock_llm()
    llm.create_message = AsyncMock(return_value=response)
    agent = TriageAgent(llm)
    with pytest.raises(AgentOutputError, match="no tool_use block"):
        await agent.run(_intake())


@pytest.mark.asyncio
async def test_triage_malformed_output_raises():
    """If LLM returns invalid data, raise AgentOutputError."""
    bad_input = {"decision": "INVALID_VALUE", "random_field": 42}
    llm = _mock_llm()
    llm.create_message = AsyncMock(
        return_value=_mock_response(bad_input)
    )
    agent = TriageAgent(llm)
    with pytest.raises(AgentOutputError, match="failed to validate"):
        await agent.run(_intake())
