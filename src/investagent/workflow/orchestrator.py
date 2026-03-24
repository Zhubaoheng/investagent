"""Main pipeline orchestrator: intake -> committee verdict."""

from __future__ import annotations

import asyncio

from investagent.agents.accounting_risk import AccountingRiskAgent
from investagent.agents.committee import CommitteeAgent
from investagent.agents.critic import CriticAgent
from investagent.agents.filing import FilingAgent
from investagent.agents.financial_quality import FinancialQualityAgent
from investagent.agents.info_capture import InfoCaptureAgent
from investagent.agents.mental_models.compounding import CompoundingAgent
from investagent.agents.mental_models.ecology import EcologyAgent
from investagent.agents.mental_models.moat import MoatAgent
from investagent.agents.mental_models.psychology import PsychologyAgent
from investagent.agents.mental_models.systems import SystemsAgent
from investagent.agents.net_cash import NetCashAgent
from investagent.agents.triage import TriageAgent
from investagent.agents.valuation import ValuationAgent
from investagent.config import Settings
from investagent.llm import LLMClient
from investagent.schemas.company import CompanyIntake
from investagent.workflow.context import PipelineContext
from investagent.workflow.gates import (
    check_accounting_risk_gate,
    check_financial_quality_gate,
    check_triage_gate,
)
from investagent.workflow.runner import run_agent


async def run_pipeline(intake: CompanyIntake) -> PipelineContext:
    """Run the full 10-stage analysis pipeline.

    Stages:
    1. Triage -> gate check
    2. Info Capture
    3. Filing Structuring
    4. Accounting Risk -> gate check
    5. Financial Quality -> gate check
    6. Net Cash
    7. Valuation
    8. Mental Models (parallel)
    9. Critic
    10. Investment Committee
    """
    settings = Settings()
    llm = LLMClient(model=settings.model_name)
    ctx = PipelineContext(intake)

    # Stage 1: Triage
    await run_agent(TriageAgent(llm), intake, ctx)
    if ctx.is_stopped():
        return ctx
    proceed, reason = check_triage_gate(ctx)
    if not proceed:
        ctx.stop(reason)
        return ctx

    # Stage 2: Info Capture
    await run_agent(InfoCaptureAgent(llm), intake, ctx)
    if ctx.is_stopped():
        return ctx

    # Stage 3: Filing Structuring
    await run_agent(FilingAgent(llm), intake, ctx)
    if ctx.is_stopped():
        return ctx

    # Stage 4: Accounting Risk
    await run_agent(AccountingRiskAgent(llm), intake, ctx)
    if ctx.is_stopped():
        return ctx
    proceed, reason = check_accounting_risk_gate(ctx)
    if not proceed:
        ctx.stop(reason)
        return ctx

    # Stage 5: Financial Quality
    await run_agent(FinancialQualityAgent(llm), intake, ctx)
    if ctx.is_stopped():
        return ctx
    proceed, reason = check_financial_quality_gate(ctx)
    if not proceed:
        ctx.stop(reason)
        return ctx

    # Stage 6: Net Cash
    await run_agent(NetCashAgent(llm), intake, ctx)
    if ctx.is_stopped():
        return ctx

    # Stage 7: Valuation
    await run_agent(ValuationAgent(llm), intake, ctx)
    if ctx.is_stopped():
        return ctx

    # Stage 8: Mental Models (parallel)
    await asyncio.gather(
        run_agent(MoatAgent(llm), intake, ctx),
        run_agent(CompoundingAgent(llm), intake, ctx),
        run_agent(PsychologyAgent(llm), intake, ctx),
        run_agent(SystemsAgent(llm), intake, ctx),
        run_agent(EcologyAgent(llm), intake, ctx),
    )
    if ctx.is_stopped():
        return ctx

    # Stage 9: Critic
    await run_agent(CriticAgent(llm), intake, ctx)
    if ctx.is_stopped():
        return ctx

    # Stage 10: Investment Committee
    await run_agent(CommitteeAgent(llm), intake, ctx)

    return ctx
