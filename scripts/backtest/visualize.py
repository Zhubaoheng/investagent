#!/usr/bin/env python3
"""Generate rich backtest visualizations from fills.json + all_decisions.json.

Outputs:
  - holdings_timeline.png  — stacked area: weight of each stock over time
  - trade_scatter.png      — buy/sell marks on NAV curve
  - sector_allocation.png  — sector weight over time
  - trade_log.csv          — Excel-friendly 交割单

Usage:
    uv run python scripts/backtest/visualize.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np

# Ensure Chinese font works
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

RESULTS_DIR = PROJECT_ROOT / "data" / "backtest_results"
DECISIONS_FILE = PROJECT_ROOT / "data" / "full_backtest" / "all_decisions.json"


def _load_fills() -> list[dict]:
    with open(RESULTS_DIR / "fills.json") as f:
        return json.load(f)["orders"]


def _load_decisions() -> dict:
    with open(DECISIONS_FILE) as f:
        return json.load(f)["decisions"]


def _ticker_name_map() -> dict[str, str]:
    """Build ticker->name from fills + decisions checkpoint data."""
    import glob
    names = {}
    for f in glob.glob(str(PROJECT_ROOT / "data/runs/overnight_*/checkpoints/pipeline/*.json")):
        try:
            d = json.loads(open(f).read())
            names[d["ticker"]] = d.get("name", d["ticker"])[:6]
        except Exception:
            pass
    return names


def plot_holdings_timeline(decisions: dict, output_path: Path, names: dict) -> None:
    """Stacked area chart showing portfolio weight allocation over time."""
    dates = sorted(decisions.keys())
    all_tickers = set()
    for d in dates:
        all_tickers.update(decisions[d]["weights"].keys())

    # Build weight matrix
    ticker_list = sorted(all_tickers)
    rows = []
    for d in dates:
        w = decisions[d]["weights"]
        cash = decisions[d].get("cash", 0)
        row = {"date": pd.Timestamp(d)}
        for t in ticker_list:
            row[t] = w.get(t, 0)
        row["CASH"] = cash
        rows.append(row)

    df = pd.DataFrame(rows).set_index("date")

    # Extend last row to today for visual continuity
    today = pd.Timestamp(date.today())
    if df.index[-1] < today:
        df.loc[today] = df.iloc[-1]

    # Resample to daily for smoother stacked area (forward fill)
    df = df.resample("D").ffill()

    # Sort columns by average weight (largest at bottom)
    avg_weights = df.drop(columns=["CASH"], errors="ignore").mean().sort_values(ascending=False)
    ordered_cols = list(avg_weights.index) + (["CASH"] if "CASH" in df.columns else [])
    df = df[ordered_cols]

    fig, ax = plt.subplots(figsize=(16, 8))
    labels = [f"{names.get(t, t)}" for t in ordered_cols]
    colors = plt.cm.tab20(np.linspace(0, 1, len(ordered_cols)))
    # Make CASH grey
    for i, col in enumerate(ordered_cols):
        if col == "CASH":
            colors[i] = [0.85, 0.85, 0.85, 1.0]

    ax.stackplot(df.index, *[df[c].values for c in ordered_cols],
                 labels=labels, colors=colors)

    # Mark scan dates
    for d in dates:
        ax.axvline(pd.Timestamp(d), color="black", linewidth=0.5, alpha=0.3, linestyle="--")

    ax.set_title("持仓权重变化 (Holdings Timeline)", fontsize=14)
    ax.set_ylabel("Weight")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", ncol=4, fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path / "holdings_timeline.png", dpi=150)
    plt.close(fig)
    print(f"  Saved holdings_timeline.png")


def plot_trade_scatter(fills: list[dict], output_path: Path, names: dict) -> None:
    """NAV-style chart with buy/sell markers."""
    buys = [f for f in fills if f["action"] == "BUY"]
    sells = [f for f in fills if f["action"] == "SELL"]

    fig, ax = plt.subplots(figsize=(16, 6))

    # Group by date for aggregate buy/sell value
    buy_dates = defaultdict(float)
    sell_dates = defaultdict(float)
    for b in buys:
        buy_dates[b["date"]] += abs(b["value"])
    for s in sells:
        sell_dates[s["date"]] += abs(s["value"])

    buy_x = [pd.Timestamp(d) for d in sorted(buy_dates.keys())]
    buy_y = [buy_dates[d.strftime("%Y-%m-%d")] for d in buy_x]
    sell_x = [pd.Timestamp(d) for d in sorted(sell_dates.keys())]
    sell_y = [sell_dates[d.strftime("%Y-%m-%d")] for d in sell_x]

    ax.bar(buy_x, buy_y, color="green", alpha=0.6, width=2, label="买入")
    ax.bar(sell_x, [-v for v in sell_y], color="red", alpha=0.6, width=2, label="卖出")

    ax.set_title("每日交易金额 (Daily Trade Volume)", fontsize=14)
    ax.set_ylabel("交易金额 (¥)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.axhline(0, color="black", linewidth=0.5)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path / "trade_volume.png", dpi=150)
    plt.close(fig)
    print(f"  Saved trade_volume.png")


def plot_sector_allocation(decisions: dict, output_path: Path) -> None:
    """Pie chart of sector allocation at each scan."""
    # Use SW L1 industry map
    try:
        import akshare as ak
        import time
        ind_info = ak.sw_index_first_info()
        industry_map = {}
        for _, row in ind_info.iterrows():
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
    except Exception:
        industry_map = {}

    if not industry_map:
        print("  Skipped sector_allocation.png (no industry map)")
        return

    dates = sorted(decisions.keys())
    n = len(dates)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, d in zip(axes, dates):
        weights = decisions[d]["weights"]
        sector_weights = defaultdict(float)
        for t, w in weights.items():
            sector = industry_map.get(t, "其他")
            sector_weights[sector] += w

        labels = list(sector_weights.keys())
        sizes = list(sector_weights.values())
        ax.pie(sizes, labels=labels, autopct="%1.0f%%", textprops={"fontsize": 8})
        ax.set_title(f"S{dates.index(d)} ({d})", fontsize=10)

    fig.suptitle("行业配置变化 (Sector Allocation)", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path / "sector_allocation.png", dpi=150)
    plt.close(fig)
    print(f"  Saved sector_allocation.png")


def export_trade_log_csv(fills: list[dict], output_path: Path, names: dict) -> None:
    """Export 交割单 as CSV."""
    rows = []
    for o in fills:
        rows.append({
            "日期": o["date"],
            "代码": o["ticker"],
            "名称": names.get(o["ticker"], o["ticker"]),
            "方向": "买入" if o["action"] == "BUY" else "卖出",
            "数量": abs(o["size"]),
            "价格": round(o["price"], 2),
            "金额": round(abs(o["value"]), 2),
            "手续费": round(o["commission"], 2),
        })
    df = pd.DataFrame(rows)
    csv_path = output_path / "trade_log.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  Saved trade_log.csv ({len(rows)} trades)")

    # Also print summary
    total_buy = df[df["方向"] == "买入"]["金额"].sum()
    total_sell = df[df["方向"] == "卖出"]["金额"].sum()
    total_comm = df["手续费"].sum()
    print(f"  买入总额: ¥{total_buy:,.0f}")
    print(f"  卖出总额: ¥{total_sell:,.0f}")
    print(f"  手续费合计: ¥{total_comm:,.0f}")


def main():
    print("Generating visualizations...")
    fills = _load_fills()
    decisions = _load_decisions()
    names = _ticker_name_map()

    plot_holdings_timeline(decisions, RESULTS_DIR, names)
    plot_trade_scatter(fills, RESULTS_DIR, names)
    export_trade_log_csv(fills, RESULTS_DIR, names)

    # Sector allocation is slow (needs akshare fetch), skip by default
    import os
    if os.environ.get("INCLUDE_SECTOR"):
        plot_sector_allocation(decisions, RESULTS_DIR)

    print(f"\nAll outputs in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
