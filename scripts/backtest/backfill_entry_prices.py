#!/usr/bin/env python3
"""One-off: backfill entry_price on existing holdings in a candidate_store.json.

For each holding, fetch the qfq close price on its entry_date via baostock
and write it back into the store.

Usage:
    uv run python scripts/backtest/backfill_entry_prices.py \\
        --store data/runs/overnight_XXX/candidate_store.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

_NO_PROXY = "baostock.com,sina.com.cn,eastmoney.com"
os.environ.setdefault("NO_PROXY", _NO_PROXY)
os.environ.setdefault("no_proxy", _NO_PROXY)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backfill")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", required=True)
    args = parser.parse_args()

    from data_feeds import fetch_daily_prices

    store_path = Path(args.store)
    data = json.loads(store_path.read_text(encoding="utf-8"))
    holdings = data.get("holdings", [])
    logger.info("Loaded %d holdings from %s", len(holdings), store_path)

    updated = 0
    for h in holdings:
        ticker = h["ticker"]
        if h.get("entry_price") is not None:
            logger.info("%s already has entry_price=%.2f, skip", ticker, h["entry_price"])
            continue
        entry_date = date.fromisoformat(h["entry_date"])
        # Fetch a small window to handle holidays
        try:
            df = fetch_daily_prices(ticker, entry_date, entry_date + timedelta(days=10))
        except Exception as e:
            logger.warning("%s fetch failed: %s", ticker, e)
            continue
        if df.empty:
            logger.warning("%s no price data around %s", ticker, entry_date)
            continue
        # Use first available close on or after entry_date
        row = df.iloc[0]
        price = float(row["close"])
        h["entry_price"] = price
        logger.info("%s entry_price=%.2f (as of %s)", ticker, price, str(row["date"])[:10])
        updated += 1

    # Also try to backfill scan_close_price on candidates for held tickers
    candidates = data.get("candidates", {})
    for h in holdings:
        t = h["ticker"]
        if t in candidates and candidates[t].get("scan_close_price") is None:
            if h.get("entry_price") is not None:
                candidates[t]["scan_close_price"] = h["entry_price"]

    store_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    logger.info("Updated %d holdings → %s", updated, store_path)


if __name__ == "__main__":
    main()
