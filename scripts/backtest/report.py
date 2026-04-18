"""Backtest report generation — charts and summary."""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from metrics import compute_metrics

logger = logging.getLogger(__name__)


def plot_nav_curve(
    nav: pd.Series,
    benchmarks: dict[str, pd.Series],
    output_path: Path,
) -> None:
    """Plot NAV curve vs benchmarks."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Normalize all to 1.0 at start
    nav_norm = nav / nav.iloc[0]
    ax.plot(nav_norm.index, nav_norm.values, label="Strategy", linewidth=2, color="darkblue")

    colors = ["grey", "orange", "green"]
    for i, (name, bench) in enumerate(benchmarks.items()):
        if len(bench) > 0:
            bench_norm = bench / bench.iloc[0]
            ax.plot(bench_norm.index, bench_norm.values,
                    label=name, linewidth=1, alpha=0.7, color=colors[i % len(colors)])

    ax.set_title("Portfolio NAV vs Benchmarks", fontsize=14)
    ax.set_ylabel("Normalized Value (start = 1.0)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path / "nav_curve.png", dpi=150)
    plt.close(fig)
    logger.info("Saved NAV curve to %s", output_path / "nav_curve.png")


def plot_drawdown(nav: pd.Series, output_path: Path) -> None:
    """Plot drawdown chart."""
    cummax = nav.cummax()
    drawdown = (nav - cummax) / cummax

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.4, color="red")
    ax.set_title("Drawdown", fontsize=14)
    ax.set_ylabel("Drawdown %")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path / "drawdown.png", dpi=150)
    plt.close(fig)
    logger.info("Saved drawdown chart to %s", output_path / "drawdown.png")


def generate_report(
    nav: pd.Series,
    benchmarks: dict[str, pd.Series],
    decisions: dict[str, dict],
    trades: dict,
    output_path: Path,
    params: dict | None = None,
) -> None:
    """Generate full backtest report: charts + markdown summary."""
    output_path.mkdir(parents=True, exist_ok=True)

    # Charts
    plot_nav_curve(nav, benchmarks, output_path)
    plot_drawdown(nav, output_path)

    # Metrics
    csi300 = benchmarks.get("CSI 300")
    metrics = compute_metrics(nav, csi300)

    # Write markdown report
    lines = ["# Backtest Report\n"]

    if params:
        lines.append("## Parameters\n")
        for k, v in params.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

    lines.append("## Performance Metrics\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    fmt_map = {
        "cumulative_return": ("Cumulative Return", "{:.2%}"),
        "cagr": ("CAGR", "{:.2%}"),
        "volatility": ("Volatility", "{:.2%}"),
        "sharpe_ratio": ("Sharpe Ratio", "{:.2f}"),
        "max_drawdown": ("Max Drawdown", "{:.2%}"),
        "alpha": ("Alpha", "{:.2%}"),
        "beta": ("Beta", "{:.2f}"),
        "information_ratio": ("Information Ratio", "{:.2f}"),
        "benchmark_return": ("CSI 300 Return", "{:.2%}"),
    }
    for key, (label, fmt) in fmt_map.items():
        val = metrics.get(key)
        if val is not None:
            lines.append(f"| {label} | {fmt.format(val)} |")
    lines.append("")

    # Decisions timeline
    lines.append("## Decision Timeline\n")
    for dt, portfolio in sorted(decisions.items()):
        lines.append(f"### {dt}\n")
        if isinstance(portfolio, dict):
            for ticker, weight in portfolio.items():
                lines.append(f"- {ticker}: {weight:.0%}")
        lines.append("")

    # Trade log: fills + closed round-trips
    orders = trades.get("orders", []) if isinstance(trades, dict) else []
    closed = trades.get("closed", []) if isinstance(trades, dict) else []
    summary = trades.get("summary", {}) if isinstance(trades, dict) else {}

    total_commission = sum(o.get("commission", 0.0) for o in orders)
    total_buy_value = sum(o["value"] for o in orders if o.get("action") == "BUY")
    total_sell_value = sum(
        abs(o["value"]) for o in orders if o.get("action") == "SELL"
    )

    lines.append("## Trade Summary\n")
    lines.append(f"- Total fills: {len(orders)} "
                 f"(BUY: {sum(1 for o in orders if o.get('action') == 'BUY')}, "
                 f"SELL: {sum(1 for o in orders if o.get('action') == 'SELL')})")
    lines.append(f"- Total buy turnover: ¥{total_buy_value:,.0f}")
    lines.append(f"- Total sell turnover: ¥{total_sell_value:,.0f}")
    lines.append(f"- Total commission + taxes + slippage: ¥{total_commission:,.0f}")
    lines.append(f"- Closed round-trips: {len(closed)} "
                 f"(analyzer reports closed={summary.get('total_closed', '?')}, "
                 f"open={summary.get('total_open', '?')})")
    if closed:
        total_net_pnl = sum(c["pnl_net"] for c in closed)
        winners = [c for c in closed if c["pnl_net"] > 0]
        losers = [c for c in closed if c["pnl_net"] < 0]
        lines.append(f"- Closed net PnL: ¥{total_net_pnl:,.0f} "
                     f"({len(winners)} winners, {len(losers)} losers)")
    lines.append("")

    if closed:
        lines.append("## Closed Trades\n")
        lines.append("| Date | Ticker | Size | Price | Value | Gross PnL | Net PnL |")
        lines.append("|------|--------|------|-------|-------|-----------|---------|")
        for c in closed:
            lines.append(
                f"| {c['date']} | {c['ticker']} | {c['size']} | "
                f"{c['price']:.2f} | ¥{c['value']:,.0f} | "
                f"¥{c['pnl_gross']:,.0f} | ¥{c['pnl_net']:,.0f} |"
            )
        lines.append("")

    if orders:
        lines.append("## All Fills\n")
        lines.append("| Date | Ticker | Action | Size | Price | Value | Commission |")
        lines.append("|------|--------|--------|------|-------|-------|------------|")
        for o in orders:
            lines.append(
                f"| {o['date']} | {o['ticker']} | {o['action']} | "
                f"{o['size']} | {o['price']:.2f} | ¥{o['value']:,.0f} | "
                f"¥{o['commission']:.2f} |"
            )
        lines.append("")

    report_text = "\n".join(lines)
    (output_path / "report.md").write_text(report_text, encoding="utf-8")
    logger.info("Saved report to %s", output_path / "report.md")

    # Also dump the raw fill log as JSON for external analysis.
    with open(output_path / "fills.json", "w") as f:
        json.dump({"orders": orders, "closed": closed, "summary": summary},
                  f, indent=2, default=str)

    # Also save raw metrics as JSON
    with open(output_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
