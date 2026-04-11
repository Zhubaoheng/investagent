"""Valuation & Look-through Return Agent output schema."""

from __future__ import annotations

from pydantic import BaseModel

from poorcharlie.schemas.common import BaseAgentOutput


class ScenarioReturns(BaseModel, frozen=True):
    bear: float | None = None
    base: float | None = None
    bull: float | None = None


class IntrinsicValueRange(BaseModel, frozen=True):
    """Per-share intrinsic value estimates under three scenarios."""
    bear: float | None = None
    base: float | None = None
    bull: float | None = None
    currency: str = ""


class MethodEstimate(BaseModel, frozen=True):
    """Single valuation method's IV estimate."""
    method: str = ""
    iv_per_share: float | None = None  # base-case IV from this method
    key_assumption: str = ""  # one-line summary of key input


class ValuationOutput(BaseAgentOutput):
    valuation_method: list[str]
    per_method_estimates: list[MethodEstimate] = []  # each method's separate IV
    expected_lookthrough_return: ScenarioReturns
    friction_adjusted_return: ScenarioReturns
    meets_hurdle_rate: bool
    intrinsic_value_per_share: IntrinsicValueRange | None = None
    margin_of_safety_pct: float | None = None
    price_vs_value: str = ""  # "CHEAP" | "FAIR" | "EXPENSIVE"
    key_assumptions: list[str] = []
    sensitivity_drivers: list[str] = []
    notes: list[str] = []
