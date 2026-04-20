"""Part 2 Decision Pipeline — layered cross-comparison → portfolio strategy.

Three-layer architecture:
  Layer 1: Industry screening (parallel) — within-industry deep comparison
  Layer 2: Cross-comparison — cross-industry final ranking
  Layer 3: Portfolio strategy — weight allocation

Chains CandidateStore, IndustryScreeningAgent, CrossComparisonAgent, and
PortfolioStrategyAgent. Output is {ticker: target_weight} compatible with
both overnight reports and backtest replay.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from datetime import date
from typing import Any

from poorcharlie.agents.cross_comparison import (
    CrossComparisonAgent,
    CrossComparisonInput,
)
from poorcharlie.agents.industry_screening import (
    IndustryScreeningAgent,
    IndustryScreeningInput,
)
from poorcharlie.agents.portfolio_strategy import (
    PortfolioStrategyAgent,
    PortfolioStrategyInput,
    StrategyHoldingInfo,
)
from poorcharlie.agents.trader import TraderAgent
from poorcharlie.llm import LLMClient
from poorcharlie.schemas.candidate import CandidateState, PortfolioHolding
from poorcharlie.schemas.portfolio_strategy import ActionType
from poorcharlie.schemas.trader import OrderAction, TraderInput
from poorcharlie.store.candidate_store import CandidateStore
from poorcharlie.workflow.change_detector import (
    ChangeTrigger,
    detect_change_triggers,
    has_cold_start_marker,
)

logger = logging.getLogger(__name__)

# Groups smaller than this skip Layer 1 (auto-advance all)
_MIN_GROUP_FOR_SCREENING = 3


def _build_industry_map() -> dict[str, str]:
    """Fetch Shenwan L1 industry classification: ticker -> industry name."""
    import time
    try:
        import akshare as ak
        ind = ak.sw_index_first_info()
        industry_map: dict[str, str] = {}
        for _, row in ind.iterrows():
            code = str(row["行业代码"])
            name = str(row["行业名称"])
            try:
                cons = ak.sw_index_third_cons(symbol=code)
                for _, c in cons.iterrows():
                    raw = str(c["股票代码"]).split(".")[0].zfill(6)
                    if raw not in industry_map:
                        industry_map[raw] = name
            except Exception:
                pass
            time.sleep(0.15)
        logger.info("Shenwan L1 industry map: %d tickers, %d industries",
                     len(industry_map), len(set(industry_map.values())))
        return industry_map
    except Exception:
        logger.warning("Failed to build industry map, falling back to single-layer",
                       exc_info=True)
        return {}


def _group_by_industry(
    candidates: list, industry_map: dict[str, str],
) -> dict[str, list]:
    """Group candidates by Shenwan L1 industry."""
    groups: dict[str, list] = defaultdict(list)
    for c in candidates:
        ind = industry_map.get(c.ticker, "其他")
        groups[ind].append(c)
    return dict(groups)


async def _run_layer1(
    groups: dict[str, list],
    llm: LLMClient,
) -> tuple[list[dict], list[dict]]:
    """Layer 1: parallel within-industry screening.

    Returns (survivors_as_dicts, industry_insights).
    Small groups (< _MIN_GROUP_FOR_SCREENING) auto-advance.
    """
    survivors: list[dict] = []
    industry_insights: list[dict] = []

    async def _screen_group(industry: str, group_candidates: list) -> None:
        if len(group_candidates) < _MIN_GROUP_FOR_SCREENING:
            # Auto-advance small groups
            for c in group_candidates:
                survivors.append(c.model_dump(mode="json"))
            industry_insights.append({
                "industry": industry,
                "insight": f"{len(group_candidates)} 只标的，数量过少跳过行业筛选",
                "cycle": "UNKNOWN",
            })
            logger.info("Layer 1 [%s]: %d candidates, auto-advance (< %d)",
                         industry, len(group_candidates), _MIN_GROUP_FOR_SCREENING)
            return

        logger.info("Layer 1 [%s]: screening %d candidates", industry, len(group_candidates))
        agent = IndustryScreeningAgent(llm)
        screening_input = IndustryScreeningInput(
            industry_name=industry,
            candidates=[c.model_dump(mode="json") for c in group_candidates],
        )
        try:
            output = await agent.run(screening_input)
        except Exception:
            logger.warning("Layer 1 [%s] failed, auto-advancing all", industry, exc_info=True)
            for c in group_candidates:
                survivors.append(c.model_dump(mode="json"))
            return

        industry_insights.append({
            "industry": industry,
            "insight": output.group_insight,
            "cycle": output.cycle_assessment,
        })

        # Collect survivors: advance=True and not hard-excluded
        excluded = set(output.hard_exclusions)
        advanced = 0
        for rc in output.ranked_candidates:
            if rc.ticker in excluded:
                continue
            if not rc.advance:
                continue
            # Find original candidate data to pass forward
            orig = next((c for c in group_candidates if c.ticker == rc.ticker), None)
            if orig is None:
                continue
            d = orig.model_dump(mode="json")
            d["_layer1_conviction"] = rc.conviction_score
            d["_layer1_strengths"] = rc.strengths_in_group
            survivors.append(d)
            advanced += 1

        logger.info(
            "Layer 1 [%s]: %d/%d advanced | cycle=%s | %s",
            industry, advanced, len(group_candidates),
            output.cycle_assessment, output.group_insight[:80],
        )

    # Run all groups in parallel
    await asyncio.gather(*[
        _screen_group(ind, grp) for ind, grp in groups.items()
    ])

    return survivors, industry_insights


async def run_decision_pipeline(
    store: CandidateStore,
    llm: LLMClient,
    scan_date: date | None = None,
) -> dict[str, float]:
    """Run the full layered decision pipeline.

    Layer 1: Industry screening (parallel within-industry comparison)
    Layer 2: Cross-comparison (cross-industry final ranking)
    Layer 3: Portfolio strategy (weight allocation)
    """
    candidates = store.get_actionable_candidates()
    logger.info("Decision pipeline: %d actionable candidates", len(candidates))

    if not candidates:
        logger.info("No actionable candidates — 100%% cash")
        store.update_holdings([], scan_date=scan_date)
        store.save()
        return {}

    # ------------------------------------------------------------------
    # Change detection: short-circuit if nothing material changed.
    # Munger: "Never interrupt compound interest unnecessarily."
    # ------------------------------------------------------------------
    current_holdings_for_detect = store.get_current_holdings()
    triggers = detect_change_triggers(candidates, current_holdings_for_detect)
    is_cold_start = has_cold_start_marker(triggers)

    if not triggers and current_holdings_for_detect:
        logger.info(
            "Decision pipeline: no change triggers — preserving previous "
            "portfolio (%d holdings) and skipping LLM",
            len(current_holdings_for_detect),
        )
        # Don't touch holdings; just re-save store to refresh last_updated.
        store.save()
        return store.to_portfolio_decisions()

    # If there ARE triggers but no holdings yet, we need to run the pipeline
    # anyway (cold start / first actual allocation).

    candidate_details = {
        c.ticker: c.model_dump(mode="json") for c in candidates
    }

    # Separate HELD positions (bypass Layer 1)
    held_tickers = {h.ticker for h in store.get_current_holdings()}
    new_candidates = [c for c in candidates if c.ticker not in held_tickers]
    held_candidates = [c for c in candidates if c.ticker in held_tickers]

    # ------------------------------------------------------------------
    # Layer 1: Industry screening (parallel)
    # ------------------------------------------------------------------
    industry_map = _build_industry_map()
    if industry_map and len(new_candidates) >= _MIN_GROUP_FOR_SCREENING:
        groups = _group_by_industry(new_candidates, industry_map)
        logger.info("Layer 1: %d new candidates in %d industry groups",
                     len(new_candidates), len(groups))

        survivors, industry_insights = await _run_layer1(groups, llm)
        logger.info("Layer 1 complete: %d survivors from %d candidates",
                     len(survivors), len(new_candidates))
    else:
        # Fallback: skip Layer 1
        if not industry_map:
            logger.warning("No industry map — falling back to single-layer comparison")
        survivors = [c.model_dump(mode="json") for c in new_candidates]
        industry_insights = []

    # Add HELD candidates (bypassed Layer 1)
    for c in held_candidates:
        survivors.append(c.model_dump(mode="json"))
    logger.info("Layer 2 input: %d candidates (%d from Layer 1 + %d HELD)",
                 len(survivors), len(survivors) - len(held_candidates), len(held_candidates))

    # ------------------------------------------------------------------
    # Layer 2: Cross-comparison
    # ------------------------------------------------------------------
    if len(survivors) >= 2:
        logger.info("Running CrossComparisonAgent (Layer 2) on %d candidates", len(survivors))
        comparison_input = CrossComparisonInput(
            candidates=survivors,
            industry_insights=industry_insights,
        )
        comparison_agent = CrossComparisonAgent(llm)
        try:
            comparison_output = await comparison_agent.run(comparison_input)
            ranked = [r.model_dump(mode="json") for r in comparison_output.ranked_candidates]
            if comparison_output.concentration_warnings:
                for w in comparison_output.concentration_warnings:
                    logger.warning("Concentration: %s", w)
        except Exception:
            logger.error(
                "CrossComparison failed — preserving current holdings.",
                exc_info=True,
            )
            store.save()
            return store.to_portfolio_decisions()
    elif len(survivors) == 1:
        c = survivors[0]
        ranked = [{
            "ticker": c.get("ticker", ""),
            "name": c.get("name", ""),
            "rank": 1,
            "conviction_score": 7,
            "strengths_vs_peers": [],
            "weaknesses_vs_peers": [],
            "portfolio_fit_notes": "唯一候选标的",
        }]
    else:
        logger.info("No survivors from Layer 1 — 100%% cash")
        store.update_holdings([], scan_date=scan_date)
        store.save()
        return {}

    # ------------------------------------------------------------------
    # Layer 3: Portfolio strategy
    # ------------------------------------------------------------------
    current_holdings = store.get_current_holdings()
    current_weight = sum(h.target_weight for h in current_holdings)

    # Pass change triggers to the LLM — so it knows WHY it was woken up
    # and which positions deserve attention vs which should stay untouched.
    # Cold-start marker is filtered out; it's a signal to the router, not
    # useful context for the LLM itself.
    trigger_dicts = [
        {
            "ticker": t.ticker,
            "name": t.name,
            "trigger_type": t.trigger_type.value,
            "detail": t.detail,
        }
        for t in triggers
        if not (t.ticker == "*" and t.name == "cold_start")
    ]

    strategy_input = PortfolioStrategyInput(
        ranked_candidates=ranked,
        candidate_details=candidate_details,
        current_holdings=[
            StrategyHoldingInfo(
                ticker=h.ticker,
                name=h.name,
                weight=h.target_weight,
                industry=h.industry,
                entry_reason=h.entry_reason,
                position_tier=h.position_tier,
            )
            for h in current_holdings
        ],
        available_cash_pct=1.0 - current_weight,
        change_triggers=trigger_dicts,
    )

    logger.info("Running PortfolioStrategyAgent (Layer 3)")
    strategy_agent = PortfolioStrategyAgent(llm)
    try:
        strategy_output = await strategy_agent.run(strategy_input)
    except Exception:
        logger.error("PortfolioStrategy failed, keeping current holdings", exc_info=True)
        store.save()
        return store.to_portfolio_decisions()

    # ------------------------------------------------------------------
    # Layer 4: Trader — final execution arbiter.
    # Rules pre-check (CORE sell block, caps) + LLM (market regime, urgency)
    # + rules post-check. Falls back to Strategy output if Trader fails.
    # ------------------------------------------------------------------
    logger.info("Running TraderAgent (Layer 4)")
    trader_agent = TraderAgent(llm)
    holdings_for_trader = [
        {
            "ticker": h.ticker,
            "name": h.name,
            "industry": h.industry,
            "target_weight": h.target_weight,
            "position_tier": h.position_tier,
        }
        for h in current_holdings
    ]
    trader_input = TraderInput(
        proposed_decisions=[
            {
                "ticker": d.ticker,
                "name": d.name,
                "action": d.action.value,
                "target_weight": d.target_weight,
                "current_weight": d.current_weight,
                "conviction_score": d.conviction_score,
                "position_tier": d.position_tier.value,
                "reason": d.reason,
            }
            for d in strategy_output.position_decisions
        ],
        current_holdings=holdings_for_trader,
        candidate_details=candidate_details,
        change_triggers=trigger_dicts,
    )
    try:
        trader_output = await trader_agent.run(trader_input)
        logger.info(
            "Trader: regime=%s, blocked=%d, urgency=%s",
            trader_output.market_regime.value,
            trader_output.blocked_orders_count,
            trader_output.overall_urgency.value,
        )
        for o in trader_output.orders:
            logger.info(
                "  %s %s [%s] %s weight=%.2f urgency=%s",
                o.ticker, o.name, o.position_tier.value,
                o.action.value, o.target_weight, o.urgency.value,
            )
    except Exception:
        # If Trader fully fails (including its LLM fallback), we have no
        # choice but to use Strategy output directly. But this is rare —
        # TraderAgent.run() has an internal rules-only fallback already.
        logger.error(
            "Trader failed entirely — falling back to Strategy output",
            exc_info=True,
        )
        trader_output = None

    # ------------------------------------------------------------------
    # Update store and return decisions.
    # Use Trader output when available; Strategy output as fallback.
    # ------------------------------------------------------------------
    effective_date = scan_date or date.today()
    existing_entries = {
        h.ticker: (h.entry_date, h.entry_price)
        for h in store.get_current_holdings()
    }
    new_holdings = []

    if trader_output is not None:
        # Trader is the final arbiter. BLOCKED and EXIT are skipped (no
        # holding to create); everything else becomes a PortfolioHolding.
        for o in trader_output.orders:
            if o.action in (OrderAction.EXIT, OrderAction.BLOCKED):
                # BLOCKED means "keep current weight" — we still need to
                # preserve the holding at the pre-check target_weight.
                if o.action == OrderAction.BLOCKED and o.target_weight > 0:
                    detail = candidate_details.get(o.ticker, {})
                    prev_date, prev_price = existing_entries.get(
                        o.ticker, (None, None),
                    )
                    entry_date = prev_date or effective_date
                    entry_price = (
                        prev_price if prev_price is not None
                        else detail.get("scan_close_price")
                    )
                    new_holdings.append(PortfolioHolding(
                        ticker=o.ticker,
                        name=o.name,
                        industry=detail.get("industry", ""),
                        target_weight=o.target_weight,
                        entry_date=entry_date,
                        entry_reason=(
                            o.rationale or o.blocked_reason
                            or "preserved by Trader CORE protection"
                        ),
                        entry_price=entry_price,
                        position_tier=o.position_tier.value,
                    ))
                continue
            if o.target_weight <= 0:
                continue
            detail = candidate_details.get(o.ticker, {})
            prev_date, prev_price = existing_entries.get(o.ticker, (None, None))
            entry_date = prev_date or effective_date
            entry_price = (
                prev_price if prev_price is not None
                else detail.get("scan_close_price")
            )
            weight = o.target_weight
            if weight > 1.0:
                weight = weight / 100.0
            new_holdings.append(PortfolioHolding(
                ticker=o.ticker,
                name=o.name,
                industry=detail.get("industry", ""),
                target_weight=weight,
                entry_date=entry_date,
                entry_reason=o.rationale or "trader-confirmed",
                entry_price=entry_price,
                position_tier=o.position_tier.value,
            ))
    else:
        # Strategy-only fallback path (used when Trader fails entirely).
        for d in strategy_output.position_decisions:
            if d.action == ActionType.EXIT or d.target_weight <= 0:
                continue
            detail = candidate_details.get(d.ticker, {})
            prev_date, prev_price = existing_entries.get(d.ticker, (None, None))
            entry_date = prev_date or effective_date
            entry_price = (
                prev_price if prev_price is not None
                else detail.get("scan_close_price")
            )
            weight = d.target_weight
            if weight > 1.0:
                weight = weight / 100.0
            new_holdings.append(PortfolioHolding(
                ticker=d.ticker,
                name=d.name,
                industry=detail.get("industry", ""),
                target_weight=weight,
                entry_date=entry_date,
                entry_reason=d.reason,
                entry_price=entry_price,
                position_tier=d.position_tier.value,
            ))

    store.update_holdings(new_holdings, scan_date=effective_date)
    store.save()

    allocations = {h.ticker: h.target_weight for h in new_holdings}
    logger.info(
        "Decision pipeline complete: %d positions, %.0f%% cash",
        len(allocations),
        (1.0 - sum(allocations.values())) * 100,
    )
    for d in strategy_output.position_decisions:
        logger.info("  %s %s: %s %.0f%% - %s",
                     d.ticker, d.name, d.action.value,
                     d.target_weight * 100, d.reason)

    return allocations
