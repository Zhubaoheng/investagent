#!/usr/bin/env python3
"""Re-run Phase 5 (decision pipeline) for all scans using existing checkpoints.

Preserves all Part 1 analysis (14-agent pipeline outputs). Only re-runs:
  - Committee label upgrade (post-processing rules applied to checkpoints)
  - Cross-comparison (LLM call)
  - Portfolio strategy (LLM call)

Usage:
    uv run python scripts/backtest/rerun_phase5.py [--pipeline-concurrency 10]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import glob
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

_NO_PROXY = (
    "cninfo.com.cn,static.cninfo.com.cn,"
    "eastmoney.com,push2.eastmoney.com,push2his.eastmoney.com,"
    "10jqka.com.cn,sina.com.cn,finance.sina.com.cn,"
    "csindex.com.cn,legulegu.com,baostock.com,"
    "minimaxi.com,api.minimaxi.com,"
    "deepseek.com,api.deepseek.com"
)
os.environ.setdefault("NO_PROXY", _NO_PROXY)
os.environ.setdefault("no_proxy", _NO_PROXY)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("rerun_phase5")


SCAN_DATES = [
    date(2023, 11, 18),
    date(2024, 5, 20),
    date(2024, 9, 23),
    date(2025, 5, 19),
    date(2025, 9, 22),
]

DATA_DIR = PROJECT_ROOT / "data" / "full_backtest"


def _apply_committee_upgrade(result: dict) -> dict:
    """Apply the same deterministic upgrade rules as committee.py post-processing.

    Mutates and returns the result dict with corrected final_label.
    """
    quality = result.get("enterprise_quality", "")
    pvv = result.get("price_vs_value", "")
    mos = result.get("margin_of_safety_pct")
    hurdle = result.get("meets_hurdle_rate", False)
    label = result.get("final_label", "")
    original = label

    # Downgrades (same as committee.py)
    if quality in ("POOR", "BELOW_AVERAGE"):
        label = "REJECT"
    elif mos is not None and mos < -50 and quality in ("AVERAGE", "POOR", "BELOW_AVERAGE", ""):
        label = "REJECT"
    elif quality == "AVERAGE" and hurdle is False:
        label = "REJECT"

    # Upgrades (new rules)
    RANK = {"REJECT": 0, "STOPPED": 0, "TOO_HARD": 1, "WATCHLIST": 2,
            "DEEP_DIVE": 3, "SPECIAL_SITUATION": 4, "INVESTABLE": 5}
    if label not in ("REJECT", "STOPPED") and hurdle is True and pvv == "CHEAP":
        if quality == "GREAT":
            if RANK.get(label, 0) < 5:
                label = "INVESTABLE"
        elif quality == "GOOD" and mos is not None and mos > 20:
            if RANK.get(label, 0) < 5:
                label = "INVESTABLE"
        elif quality == "GOOD":
            if RANK.get(label, 0) < 3:
                label = "DEEP_DIVE"

    if label != original:
        logger.debug("  %s: %s → %s (q=%s pvv=%s mos=%s)",
                     result.get("ticker"), original, label, quality, pvv, mos)
    result["final_label"] = label
    return result


def _find_run_dir(as_of_date: str) -> Path | None:
    runs_dir = PROJECT_ROOT / "data" / "runs"
    candidates = []
    for d in runs_dir.iterdir():
        if not d.is_dir():
            continue
        rj = d / "run.json"
        if not rj.exists():
            continue
        try:
            meta = json.loads(rj.read_text())
            if meta.get("as_of_date") == as_of_date:
                candidates.append(d)
        except Exception:
            pass
    if not candidates:
        return None
    return sorted(candidates)[-1]


def _load_and_patch_checkpoints(run_dir: Path) -> list[dict]:
    """Load all pipeline checkpoints, apply committee upgrade rules."""
    ckpt_dir = run_dir / "checkpoints" / "pipeline"
    results = []
    upgraded = 0
    for f in sorted(ckpt_dir.glob("*.json")):
        d = json.loads(f.read_text())
        old_label = d.get("final_label", "")
        _apply_committee_upgrade(d)
        if d["final_label"] != old_label:
            upgraded += 1
        results.append(d)
    logger.info("Loaded %d checkpoints, upgraded %d labels", len(results), upgraded)
    return results


def _extract_allocations(store_path: Path) -> dict[str, float]:
    if not store_path.exists():
        return {}
    try:
        data = json.loads(store_path.read_text())
        return {h["ticker"]: h["target_weight"] for h in data.get("holdings", [])}
    except Exception:
        return {}


async def _run_phase5_for_scan(
    scan_idx: int,
    scan_date: date,
    prev_store_path: Path | None,
    llm,
) -> tuple[Path, dict[str, float]]:
    """Run Phase 5 for a single scan. Returns (store_path, allocations)."""
    from poorcharlie.store.candidate_store import CandidateStore
    from poorcharlie.workflow.decision_pipeline import run_decision_pipeline

    run_dir = _find_run_dir(scan_date.isoformat())
    if run_dir is None:
        raise RuntimeError(f"No run found for {scan_date}")

    logger.info("S%d (%s): loading checkpoints from %s", scan_idx, scan_date, run_dir.name)
    results = _load_and_patch_checkpoints(run_dir)

    # Build fresh candidate store
    store_path = run_dir / "candidate_store.json"
    store = CandidateStore(store_path)

    # For incremental scans, seed holdings from previous scan
    if prev_store_path and prev_store_path.exists() and scan_idx > 0:
        prev_store = CandidateStore(prev_store_path)
        prev_holdings = prev_store.get_current_holdings()
        if prev_holdings:
            store.update_holdings(prev_holdings, scan_date=scan_date)
            logger.info("  Seeded %d holdings from previous scan", len(prev_holdings))

    store.ingest_scan_results(results, scan_date)
    actionable = store.get_actionable_candidates()
    logger.info("  %d actionable candidates (INVESTABLE + DEEP_DIVE + HELD)", len(actionable))

    if not actionable:
        logger.info("  No actionable candidates → 100%% cash")
        store.save()
        return store_path, {}

    allocations = await run_decision_pipeline(store, llm, scan_date=scan_date)
    logger.info("  → %d positions, %.0f%% cash",
                len(allocations), (1 - sum(allocations.values())) * 100)
    return store_path, allocations


async def main_async(args):
    from poorcharlie.config import create_llm_client
    from decision_schema import make_record, save_decisions

    llm = create_llm_client()
    logger.info("LLM: %s / %s", llm.provider, llm.model)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    decisions_file = DATA_DIR / "all_decisions.json"
    start_from = getattr(args, 'start_from', 0) or 0

    if start_from > 0 and decisions_file.exists():
        from decision_schema import load_decisions
        all_decisions = load_decisions(decisions_file)
        logger.info("Loaded %d existing decisions (resuming from S%d)", len(all_decisions), start_from)
    else:
        all_decisions = {}

    prev_store_path: Path | None = None
    # Find prev store for the scan before start_from
    if start_from > 0:
        prev_date = SCAN_DATES[start_from - 1]
        prev_run = _find_run_dir(prev_date.isoformat())
        if prev_run:
            prev_store_path = prev_run / "candidate_store.json"

    for i, scan_date in enumerate(SCAN_DATES):
        if i < start_from:
            run_dir = _find_run_dir(scan_date.isoformat())
            if run_dir:
                prev_store_path = run_dir / "candidate_store.json"
            continue

        logger.info("")
        logger.info("=" * 70)
        logger.info("SCAN S%d: %s", i, scan_date)
        logger.info("=" * 70)

        store_path, allocations = await _run_phase5_for_scan(
            i, scan_date, prev_store_path, llm,
        )

        all_decisions[scan_date.isoformat()] = make_record(
            source="scan",
            weights=allocations,
            scan_id=f"S{i}",
            run_id=_find_run_dir(scan_date.isoformat()).name,
            rationale=f"S{i} {'cold start' if i == 0 else 'incremental'} scan (Phase 5 re-run)",
        )

        save_decisions(decisions_file, all_decisions)
        logger.info("Saved %d decisions to %s", len(all_decisions), decisions_file)

        prev_store_path = store_path

    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 5 RE-RUN COMPLETE: %d decision points", len(all_decisions))
    logger.info("=" * 70)
    for d in sorted(all_decisions.keys()):
        r = all_decisions[d]
        logger.info("  %s [%s]: %d pos, %.0f%% cash",
                     d, r["source"], len(r["weights"]), r["cash"] * 100)


def main():
    parser = argparse.ArgumentParser(description="Re-run Phase 5 with new committee rules")
    parser.add_argument("--start-from", type=int, default=0,
                        help="Skip scans before this index (0=S0, 4=S4)")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
