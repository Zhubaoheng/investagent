"""Single-ticker opportunity trigger: re-run pipeline + re-solve portfolio.

When a WATCHLIST+ stock's price drops below IV × 0.8, this module:
1. Runs the full pipeline for that one ticker with as_of_date = trigger_date
2. Copies the prior scan's CandidateStore to a trigger-specific path
3. Ingests the fresh result into that store snapshot
4. Runs decision_pipeline (CrossComparison + PortfolioStrategy) for coherent rebalance
5. Returns new allocations + metadata

This replaces the hardcoded `ticker = 0.05` trial position with a proper
conviction-weighted decision that sees the opportunity alongside current
holdings.
"""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from poorcharlie.datasources.cache import AkShareCache, FilingCache
from poorcharlie.datasources.cached_fetcher import CachedFilingFetcher
from poorcharlie.datasources.resolver import resolve_filing_fetcher
from poorcharlie.llm import LLMClient
from poorcharlie.schemas.company import CompanyIntake
from poorcharlie.store.candidate_store import CandidateStore
from poorcharlie.workflow.decision_pipeline import run_decision_pipeline
from poorcharlie.workflow.orchestrator import run_pipeline

# Import the result extractor from run_overnight (single source of truth)
from run_overnight import _EXCHANGE_MAP, _extract_result

logger = logging.getLogger("opportunity_trigger")


async def _run_single_stock_pipeline(
    ticker: str,
    name: str,
    industry: str,
    as_of_date: date,
    llm: LLMClient,
    filing_cache: FilingCache,
    akshare_cache: AkShareCache,
    timeout_s: int = 5400,
) -> dict:
    """Run the full Part 1 pipeline on one stock at a specific as_of_date."""
    exchange = _EXCHANGE_MAP.get(ticker[0], "SSE")
    intake = CompanyIntake(
        ticker=ticker,
        name=name,
        exchange=exchange,
        sector=industry,
        as_of_date=as_of_date,
    )
    cached_fetcher = CachedFilingFetcher(resolve_filing_fetcher(exchange), filing_cache)
    stock = {"ticker": ticker, "name": name, "industry": industry}
    try:
        ctx = await asyncio.wait_for(
            run_pipeline(
                intake,
                llm=llm,
                filing_fetcher=cached_fetcher,
                filing_cache=filing_cache,
                akshare_cache=akshare_cache,
            ),
            timeout=timeout_s,
        )
        return _extract_result(ctx, stock)
    except asyncio.TimeoutError:
        logger.error("Opportunity pipeline TIMEOUT for %s", ticker)
        return {**stock, "final_label": "ERROR", "error": "timeout", "stopped": True}
    except BaseException as e:
        logger.error("Opportunity pipeline FAILED for %s: %s", ticker, e)
        return {**stock, "final_label": "ERROR", "error": str(e), "stopped": True}


async def reevaluate_ticker(
    ticker: str,
    trigger_date: date,
    prev_store_path: Path,
    trigger_output_dir: Path,
    llm: LLMClient,
    filing_cache: FilingCache,
    akshare_cache: AkShareCache,
) -> tuple[dict[str, float], dict] | None:
    """Re-evaluate a single opportunity ticker + re-solve portfolio.

    Args:
        ticker: the triggered candidate
        trigger_date: as_of_date for the re-evaluation
        prev_store_path: path to the pre-trigger CandidateStore (e.g. S0 store)
        trigger_output_dir: where to write trigger-specific artifacts
        llm, filing_cache, akshare_cache: pipeline dependencies

    Returns:
        (allocations, metadata) or None on failure.
        allocations: {ticker: weight}
        metadata: {"run_id", "rationale", "trigger_reason", "pipeline_label"}
    """
    trigger_output_dir.mkdir(parents=True, exist_ok=True)

    # Copy the prior store so this trigger doesn't mutate the scan's store
    trigger_store_path = trigger_output_dir / "candidate_store.json"
    shutil.copy2(prev_store_path, trigger_store_path)
    store = CandidateStore(trigger_store_path)

    # Snapshot previous holdings BEFORE the decision — used to enforce
    # "only the triggered ticker may change" post-process semantics.
    prev_holding_weights = {
        h.ticker: h.target_weight for h in store.get_current_holdings()
    }

    # Look up the candidate's prior snapshot for name/industry
    prev = store._state.candidates.get(ticker)
    if prev is None:
        logger.warning("Ticker %s not in prior store; skipping", ticker)
        return None
    name, industry = prev.name, prev.industry

    # Run pipeline for this single ticker at trigger_date
    logger.info(
        "Opportunity trigger: running pipeline for %s %s as_of=%s",
        ticker, name, trigger_date,
    )
    result = await _run_single_stock_pipeline(
        ticker, name, industry, trigger_date, llm, filing_cache, akshare_cache,
    )

    pipeline_label = result.get("final_label", "ERROR")
    # Persist raw pipeline result for audit
    (trigger_output_dir / "pipeline_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    if pipeline_label == "ERROR":
        logger.warning("Opportunity pipeline ERROR for %s; no allocation change", ticker)
        return None

    # Ingest fresh result into store snapshot (updates just this ticker)
    store.ingest_scan_results([result], trigger_date)
    store.save()

    # Run decision pipeline — evaluates this opportunity alongside current holdings
    raw_allocations = await run_decision_pipeline(store, llm, scan_date=trigger_date)

    # --- Enforce trigger semantics -------------------------------------
    # An opportunity trigger is a REACTIVE signal about ONE ticker crossing
    # its own IV * 0.8 threshold. The re-evaluation pipeline only analyzed
    # that ticker — it used stale snapshots for everything else. So the
    # agent shouldn't use this re-eval as a license to rebalance the whole
    # book ("oh cash is available, let me ADD to 000651"): the other
    # holdings haven't hit their own opportunity thresholds.
    #
    # Policy: keep prev weights for all non-trigger holdings; apply the
    # agent's decision only for the triggered ticker. Full-book rebalance
    # is the scheduled scan's job, not the trigger's.
    allocations: dict[str, float] = dict(prev_holding_weights)
    trigger_decision: str = "HOLD"
    if ticker in raw_allocations:
        new_w = raw_allocations[ticker]
        prev_w = prev_holding_weights.get(ticker, 0.0)
        if prev_w == 0 and new_w > 0:
            trigger_decision = f"BUY {new_w*100:.0f}%"
        elif prev_w > 0 and new_w > prev_w + 1e-6:
            trigger_decision = f"ADD {prev_w*100:.0f}→{new_w*100:.0f}%"
        elif prev_w > 0 and new_w < prev_w - 1e-6:
            trigger_decision = f"REDUCE {prev_w*100:.0f}→{new_w*100:.0f}%"
        elif prev_w > 0 and abs(new_w - prev_w) <= 1e-6:
            trigger_decision = f"HOLD {new_w*100:.0f}%"
        else:
            trigger_decision = "SKIP"
        allocations[ticker] = new_w
    else:
        # Agent chose not to include the triggered ticker at all
        if ticker in allocations:
            trigger_decision = "EXIT"
            del allocations[ticker]
        else:
            trigger_decision = "SKIP"

    # Detect and strip incidental rebalancing (agent changed weights on
    # non-trigger tickers). We don't allow this — log for observability
    # then discard.
    stripped: list[tuple[str, float, float]] = []
    for other_ticker, new_w in raw_allocations.items():
        if other_ticker == ticker:
            continue
        prev_w = prev_holding_weights.get(other_ticker, 0.0)
        if abs(new_w - prev_w) > 1e-6:
            stripped.append((other_ticker, prev_w, new_w))
    if stripped:
        logger.info(
            "Trigger %s: stripped %d incidental rebalance action(s) "
            "(only trigger ticker may change on opportunity re-eval)",
            ticker, len(stripped),
        )
        for t, pw, nw in stripped[:5]:
            logger.info("  would-be: %s %.1f%% → %.1f%% (ignored)",
                        t, pw * 100, nw * 100)

    # Clear weight-0 entries (EXIT)
    allocations = {t: w for t, w in allocations.items() if w > 0}

    metadata = {
        "run_id": trigger_output_dir.name,
        "rationale": (
            f"opportunity trigger on {ticker} at {trigger_date}; "
            f"decision: {trigger_decision}"
        ),
        "trigger_reason": (
            f"{ticker} price crossed IV × 0.8; committee={pipeline_label}"
        ),
        "pipeline_label": pipeline_label,
        "trigger_decision": trigger_decision,
    }

    # Also update the trigger's saved store so the cascading baseline
    # reflects the filtered allocations (not the raw LLM output).
    filtered_holdings = []
    for h in store.get_current_holdings():
        if h.ticker == ticker and ticker in allocations:
            filtered_holdings.append(h.model_copy(update={"target_weight": allocations[ticker]}))
        elif h.ticker in allocations:
            # Preserved from prev — keep the holding as-is
            filtered_holdings.append(h)
    # If trigger ticker is a new BUY (not in prev holdings), it won't be in
    # store.get_current_holdings() — but decision_pipeline already called
    # update_holdings which added it. We need to filter that too.
    for h in store.get_current_holdings():
        if h.ticker not in {x.ticker for x in filtered_holdings} and h.ticker in allocations:
            filtered_holdings.append(h)
    store.update_holdings(filtered_holdings, scan_date=trigger_date)
    store.save()

    return allocations, metadata
