"""PipelineContext — carries structured outputs between pipeline stages."""

from __future__ import annotations

from typing import Any

from investagent.schemas.common import BaseAgentOutput
from investagent.schemas.company import CompanyIntake


class PipelineContext:
    """Central data bus for the pipeline.

    Each agent writes its output here; downstream agents read from here.
    """

    def __init__(self, intake: CompanyIntake) -> None:
        self.intake = intake
        self._results: dict[str, BaseAgentOutput] = {}
        self._data: dict[str, Any] = {}
        self.stopped: bool = False
        self.stop_reason: str | None = None

    def set_result(self, agent_name: str, output: BaseAgentOutput) -> None:
        """Store agent output. Auto-stops pipeline if StopSignal is set."""
        self._results[agent_name] = output
        if output.stop_signal and output.stop_signal.should_stop:
            self.stop(output.stop_signal.reason)

    def get_result(self, agent_name: str) -> BaseAgentOutput:
        """Return stored output for an agent. Raises KeyError if not found."""
        return self._results[agent_name]

    def set_data(self, key: str, value: Any) -> None:
        """Store arbitrary data (e.g., raw FilingDocuments) for downstream use."""
        self._data[key] = value

    def get_data(self, key: str) -> Any:
        """Return stored data. Raises KeyError if not found."""
        return self._data[key]

    def is_stopped(self) -> bool:
        return self.stopped

    def stop(self, reason: str) -> None:
        """Explicitly halt the pipeline."""
        self.stopped = True
        self.stop_reason = reason

    def completed_agents(self) -> list[str]:
        """Return names of agents that have produced output."""
        return list(self._results.keys())
