"""Tests for poorcharlie.workflow.context."""

from datetime import datetime, timezone

import pytest

from poorcharlie.schemas.common import AgentMeta, BaseAgentOutput, StopSignal
from poorcharlie.schemas.company import CompanyIntake
from poorcharlie.workflow.context import PipelineContext


def _meta(name: str = "test") -> AgentMeta:
    return AgentMeta(
        agent_name=name,
        timestamp=datetime.now(tz=timezone.utc),
        model_used="test",
        token_usage=0,
    )


def _make_ctx() -> PipelineContext:
    intake = CompanyIntake(ticker="AAPL", name="Apple Inc.", exchange="NASDAQ")
    return PipelineContext(intake=intake)


class TestPipelineContextInit:
    def test_initial_state(self):
        ctx = _make_ctx()
        assert ctx.intake.ticker == "AAPL"
        assert ctx.stopped is False
        assert ctx.stop_reason is None
        assert ctx.completed_agents() == []

    def test_is_stopped_initially_false(self):
        ctx = _make_ctx()
        assert ctx.is_stopped() is False


class TestSetGetResult:
    def test_set_and_get(self):
        ctx = _make_ctx()
        output = BaseAgentOutput(meta=_meta("triage"))
        ctx.set_result("triage", output)
        assert ctx.get_result("triage") is output

    def test_get_missing_raises_key_error(self):
        ctx = _make_ctx()
        with pytest.raises(KeyError):
            ctx.get_result("nonexistent")

    def test_completed_agents_tracks_names(self):
        ctx = _make_ctx()
        ctx.set_result("triage", BaseAgentOutput(meta=_meta("triage")))
        ctx.set_result("filing", BaseAgentOutput(meta=_meta("filing")))
        assert ctx.completed_agents() == ["triage", "filing"]


class TestStopSignalAutoStop:
    def test_stop_signal_halts_pipeline(self):
        ctx = _make_ctx()
        output = BaseAgentOutput(
            meta=_meta("triage"),
            stop_signal=StopSignal(should_stop=True, reason="opaque company"),
        )
        ctx.set_result("triage", output)
        assert ctx.is_stopped() is True
        assert ctx.stop_reason == "opaque company"

    def test_stop_signal_false_does_not_halt(self):
        ctx = _make_ctx()
        output = BaseAgentOutput(
            meta=_meta("triage"),
            stop_signal=StopSignal(should_stop=False, reason="minor concern"),
        )
        ctx.set_result("triage", output)
        assert ctx.is_stopped() is False

    def test_no_stop_signal_does_not_halt(self):
        ctx = _make_ctx()
        output = BaseAgentOutput(meta=_meta("triage"))
        ctx.set_result("triage", output)
        assert ctx.is_stopped() is False


class TestExplicitStop:
    def test_stop_sets_flag_and_reason(self):
        ctx = _make_ctx()
        ctx.stop("Triage rejected")
        assert ctx.is_stopped() is True
        assert ctx.stop_reason == "Triage rejected"
