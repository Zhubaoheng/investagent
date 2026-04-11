"""Main pipeline orchestrator: intake -> committee verdict."""

from __future__ import annotations

import asyncio

from poorcharlie.datasources.base import FilingFetcher, MarketDataFetcher

from poorcharlie.agents.accounting_risk import AccountingRiskAgent
from poorcharlie.agents.committee import CommitteeAgent
from poorcharlie.agents.critic import CriticAgent
from poorcharlie.agents.filing import FilingAgent
from poorcharlie.agents.financial_quality import FinancialQualityAgent
from poorcharlie.agents.info_capture import InfoCaptureAgent
from poorcharlie.agents.mental_models.compounding import CompoundingAgent
from poorcharlie.agents.mental_models.ecology import EcologyAgent
from poorcharlie.agents.mental_models.moat import MoatAgent
from poorcharlie.agents.mental_models.psychology import PsychologyAgent
from poorcharlie.agents.mental_models.systems import SystemsAgent
from poorcharlie.agents.net_cash import NetCashAgent
from poorcharlie.agents.triage import TriageAgent
from poorcharlie.agents.valuation import ValuationAgent
from poorcharlie.config import Settings
from poorcharlie.llm import LLMClient
from poorcharlie.schemas.company import CompanyIntake
from poorcharlie.workflow.context import PipelineContext
from poorcharlie.workflow.gates import (
    check_accounting_risk_gate,
    check_financial_quality_gate,
    check_triage_gate,
)
from poorcharlie.workflow.runner import run_agent


async def run_pipeline(
    intake: CompanyIntake,
    *,
    llm: LLMClient | None = None,
    filing_fetcher: FilingFetcher | None = None,
    market_fetcher: MarketDataFetcher | None = None,
    filing_cache: "FilingCache | None" = None,
    akshare_cache: "AkShareCache | None" = None,
) -> PipelineContext:
    """Run the full 10-stage analysis pipeline.

    Stages:
    1. Info Capture (with real data fetching)
    2. Filing Structuring
    3. Triage -> gate check (now with real data)
    4. Accounting Risk -> gate check
    5. Financial Quality -> gate check
    6. Net Cash
    7. Valuation
    8. Mental Models (parallel)
    9. Critic
    10. Investment Committee
    """
    if llm is None:
        settings = Settings()
        # MiniMax-specific parameters
        extra_body = None
        if settings.provider == "minimax":
            extra_body = {
                "context_window_size": 200000,
                "effort": "high",
            }
        llm = LLMClient(
            model=settings.model_name,
            base_url=settings.api_base_url,
            api_key=settings.api_key,
            extra_body=extra_body,
        )
    ctx = PipelineContext(intake)
    cutoff = str(intake.as_of_date) if intake.as_of_date else None

    # Backtest mode: use historical data when as_of_date is set
    if intake.as_of_date and market_fetcher is None:
        from poorcharlie.datasources.historical_market_data import HistoricalMarketDataFetcher
        market_fetcher = HistoricalMarketDataFetcher(
            as_of_date=intake.as_of_date,
            exchange=intake.exchange,
        )

    # Stage 1: Info Capture (with datasource integration)
    info_agent = InfoCaptureAgent(
        llm,
        filing_fetcher=filing_fetcher,
        market_fetcher=market_fetcher,
        as_of_date=cutoff,
    )
    await run_agent(info_agent, intake, ctx)
    if ctx.is_stopped():
        return ctx

    # Stage 2: Filing Structuring (with real content extraction)
    filing_agent = FilingAgent(
        llm, filing_fetcher=filing_fetcher, as_of_date=cutoff,
        filing_cache=filing_cache, akshare_cache=akshare_cache,
    )
    await run_agent(filing_agent, intake, ctx)
    if ctx.is_stopped():
        return ctx

    # Stage 3: Triage (with real data from InfoCapture + Filing)
    await run_agent(TriageAgent(llm, as_of_date=cutoff), intake, ctx)
    if ctx.is_stopped():
        return ctx
    proceed, reason = check_triage_gate(ctx)
    if not proceed:
        ctx.stop(reason)
        return ctx

    # Stage 4-8: Parallel analysis (all depend on Filing, not each other)
    # AccountingRisk + FinancialQuality + NetCash + Valuation + 5 Mental Models
    # = 9 agents running simultaneously
    await asyncio.gather(
        run_agent(AccountingRiskAgent(llm, as_of_date=cutoff), intake, ctx),
        run_agent(FinancialQualityAgent(llm, as_of_date=cutoff), intake, ctx),
        run_agent(NetCashAgent(llm, as_of_date=cutoff), intake, ctx),
        run_agent(ValuationAgent(llm, as_of_date=cutoff), intake, ctx),
        run_agent(MoatAgent(llm, as_of_date=cutoff), intake, ctx),
        run_agent(CompoundingAgent(llm, as_of_date=cutoff), intake, ctx),
        run_agent(PsychologyAgent(llm, as_of_date=cutoff), intake, ctx),
        run_agent(SystemsAgent(llm, as_of_date=cutoff), intake, ctx),
        run_agent(EcologyAgent(llm, as_of_date=cutoff), intake, ctx),
    )
    if ctx.is_stopped():
        return ctx

    # Post-parallel gate checks
    proceed, reason = check_accounting_risk_gate(ctx)
    if not proceed:
        ctx.stop(reason)
        return ctx
    proceed, reason = check_financial_quality_gate(ctx)
    if not proceed:
        ctx.stop(reason)
        return ctx

    # Stage 9: Critic (needs all upstream outputs)
    await run_agent(CriticAgent(llm, as_of_date=cutoff), intake, ctx)
    if ctx.is_stopped():
        return ctx

    # Stage 10: Investment Committee
    await run_agent(CommitteeAgent(llm, as_of_date=cutoff), intake, ctx)

    return ctx
