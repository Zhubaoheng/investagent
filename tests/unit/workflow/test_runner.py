"""Tests for investagent.workflow.runner."""

from datetime import datetime, timezone

import pytest

from investagent.agents.base import BaseAgent
from investagent.schemas.common import AgentMeta, BaseAgentOutput, StopSignal
from investagent.schemas.company import CompanyIntake
from investagent.workflow.context import PipelineContext
from investagent.workflow.runner import run_agent


def _meta(name: str = "stub") -> AgentMeta:
    return AgentMeta(
        agent_name=name,
        timestamp=datetime.now(tz=timezone.utc),
        model_used="test",
        token_usage=0,
    )


class StubAgent(BaseAgent):
    """A minimal agent that returns a fixed output."""

    name: str = "stub"

    def __init__(self, output: BaseAgentOutput) -> None:
        self._output = output

    async def run(self, input_data):
        return self._output


class FailingAgent(BaseAgent):
    """An agent that always raises."""

    name: str = "failing"

    async def run(self, input_data):
        raise RuntimeError("LLM call failed")


@pytest.fixture
def ctx():
    intake = CompanyIntake(ticker="AAPL", name="Apple Inc.", exchange="NASDAQ")
    return PipelineContext(intake=intake)


@pytest.mark.asyncio
async def test_run_agent_stores_result(ctx):
    output = BaseAgentOutput(meta=_meta("stub"))
    agent = StubAgent(output)
    result = await run_agent(agent, ctx.intake, ctx)
    assert result is output
    assert ctx.get_result("stub") is output


@pytest.mark.asyncio
async def test_run_agent_propagates_stop_signal(ctx):
    output = BaseAgentOutput(
        meta=_meta("stub"),
        stop_signal=StopSignal(should_stop=True, reason="halt"),
    )
    agent = StubAgent(output)
    await run_agent(agent, ctx.intake, ctx)
    assert ctx.is_stopped() is True
    assert ctx.stop_reason == "halt"


@pytest.mark.asyncio
async def test_run_agent_propagates_exception(ctx):
    agent = FailingAgent()
    with pytest.raises(RuntimeError, match="LLM call failed"):
        await run_agent(agent, ctx.intake, ctx)
