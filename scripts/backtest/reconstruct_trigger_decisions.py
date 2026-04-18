#!/usr/bin/env python3
"""Reconstruct opportunity_trigger decisions from disk artifacts.

Rationale: `run_full_backtest.py --start-from N` skips the between-scan trigger
loop for scans < N, but the opportunity-trigger artifacts on disk (one dir per
trigger under `data/full_backtest/triggers/opp_YYYY-MM-DD_TICKER/`) still hold
the canonical post-trigger allocations from when the triggers originally ran.
This script walks those dirs chronologically and appends missing
`opportunity_trigger` entries to `all_decisions.json`.

Safe to re-run: scan entries are preserved; existing trigger entries are
overwritten with the reconstructed ones (same data, same format).

Usage:
    uv run python scripts/backtest/reconstruct_trigger_decisions.py
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


_OPP_DIR_RE = re.compile(r"^opp_(\d{4}-\d{2}-\d{2})_(\d+)$")


def _extract_allocations(store_path: Path) -> dict[str, float]:
    if not store_path.exists():
        return {}
    try:
        data = json.loads(store_path.read_text())
        return {h["ticker"]: h["target_weight"] for h in data.get("holdings", [])}
    except Exception:
        logger.warning("Failed to parse %s", store_path, exc_info=True)
        return {}


def _extract_metadata(pipeline_path: Path) -> tuple[str, str]:
    """Return (rationale, trigger_reason) from pipeline_result.json."""
    if not pipeline_path.exists():
        return "", ""
    try:
        data = json.loads(pipeline_path.read_text())
        thesis = data.get("thesis") or ""
        next_action = data.get("next_action") or ""
        label = data.get("final_label") or ""
        rationale = f"{label}: {thesis}"[:500] if thesis else label
        return rationale, f"price-vs-value {data.get('price_vs_value') or 'UNKNOWN'}"
    except Exception:
        return "", ""


def main() -> int:
    from decision_schema import load_decisions, make_record, save_decisions

    triggers_dir = PROJECT_ROOT / "data" / "full_backtest" / "triggers"
    decisions_file = PROJECT_ROOT / "data" / "full_backtest" / "all_decisions.json"

    if not triggers_dir.exists():
        logger.error("No triggers dir at %s", triggers_dir)
        return 1

    all_decisions = load_decisions(decisions_file)
    scan_dates = {d for d, rec in all_decisions.items() if rec.get("source") == "scan"}
    logger.info("Loaded %d existing decisions (%d scans)", len(all_decisions), len(scan_dates))

    # Sort by (trigger_date, mtime): for multiple triggers on the same date,
    # the one that ran LAST is the cascade endpoint (its candidate_store.json
    # reflects all prior triggers on that date). Sorting by name breaks this
    # because it's alphanumeric over ticker IDs, not chronological.
    def _sort_key(p: Path) -> tuple[str, float]:
        m = _OPP_DIR_RE.match(p.name)
        date = m.group(1) if m else "0000-00-00"
        return (date, p.stat().st_mtime)

    opp_dirs = sorted(triggers_dir.iterdir(), key=_sort_key)
    reconstructed = 0
    skipped_scan_collision = 0

    for d in opp_dirs:
        if not d.is_dir():
            continue
        m = _OPP_DIR_RE.match(d.name)
        if not m:
            continue
        trigger_date, ticker = m.group(1), m.group(2)

        if trigger_date in scan_dates:
            # A scan already owns this date — never overwrite a scan record.
            skipped_scan_collision += 1
            continue

        alloc = _extract_allocations(d / "candidate_store.json")
        if not alloc:
            logger.warning("No allocations in %s; skipping", d.name)
            continue

        rationale, trigger_reason = _extract_metadata(d / "pipeline_result.json")
        rec = make_record(
            source="opportunity_trigger",
            weights=alloc,
            run_id=d.name,
            rationale=rationale,
            trigger_ticker=ticker,
            trigger_reason=trigger_reason,
        )
        all_decisions[trigger_date] = rec
        reconstructed += 1
        logger.info(
            "  %s  %s → %d positions, cash=%.1f%%",
            trigger_date, ticker, len(alloc), rec["cash"] * 100,
        )

    if skipped_scan_collision:
        logger.info("Skipped %d trigger dirs whose date collides with a scan", skipped_scan_collision)

    save_decisions(decisions_file, all_decisions)
    logger.info(
        "Reconstructed %d trigger decisions; total %d → %s",
        reconstructed, len(all_decisions), decisions_file,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
