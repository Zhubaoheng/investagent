"""Performance metrics calculation for backtesting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(
    nav: pd.Series,
    benchmark: pd.Series | None = None,
    risk_free_rate: float = 0.02,
) -> dict[str, float | None]:
    """Compute performance metrics from a daily NAV series.

    Args:
        nav: Daily net asset value series (indexed by date).
        benchmark: Optional benchmark daily close series (same index).
        risk_free_rate: Annualized risk-free rate for Sharpe calculation.

    Returns dict of metric name -> value.
    """
    if len(nav) < 2:
        return {}

    returns = nav.pct_change().dropna()
    total_days = (nav.index[-1] - nav.index[0]).days
    years = total_days / 365.25

    # Cumulative return
    cumulative_return = (nav.iloc[-1] / nav.iloc[0]) - 1

    # CAGR
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1 if years > 0 else None

    # Volatility (annualized)
    volatility = returns.std() * np.sqrt(252)

    # Sharpe Ratio
    sharpe = (cagr - risk_free_rate) / volatility if volatility > 0 and cagr is not None else None

    # Max drawdown
    cummax = nav.cummax()
    drawdown = (nav - cummax) / cummax
    max_drawdown = drawdown.min()

    # Calmar ratio (CAGR / |MDD|)
    calmar = (cagr / abs(max_drawdown)) if (cagr is not None and max_drawdown < 0) else None

    # Sortino (downside deviation)
    downside = returns[returns < 0]
    downside_dev = downside.std() * np.sqrt(252) if len(downside) > 1 else None
    sortino = (
        (cagr - risk_free_rate) / downside_dev
        if (downside_dev is not None and downside_dev > 0 and cagr is not None)
        else None
    )

    result: dict[str, float | None] = {
        "cumulative_return": cumulative_return,
        "cagr": cagr,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
    }

    # Benchmark-relative metrics
    if benchmark is not None and len(benchmark) > 1:
        # Align indices
        aligned = pd.DataFrame({"nav": nav, "bench": benchmark}).dropna()
        if len(aligned) > 10:
            nav_ret = aligned["nav"].pct_change().dropna()
            bench_ret = aligned["bench"].pct_change().dropna()

            # Beta
            cov = nav_ret.cov(bench_ret)
            bench_var = bench_ret.var()
            beta = cov / bench_var if bench_var > 0 else None

            # Alpha (Jensen's)
            bench_cagr = (aligned["bench"].iloc[-1] / aligned["bench"].iloc[0]) ** (1 / years) - 1
            alpha = cagr - (risk_free_rate + beta * (bench_cagr - risk_free_rate)) if beta is not None and cagr is not None else None

            # Tracking error & information ratio
            excess = nav_ret - bench_ret
            tracking_error = excess.std() * np.sqrt(252)
            info_ratio = (cagr - bench_cagr) / tracking_error if tracking_error > 0 and cagr is not None else None

            result["beta"] = beta
            result["alpha"] = alpha
            result["tracking_error"] = tracking_error
            result["information_ratio"] = info_ratio
            result["benchmark_return"] = (aligned["bench"].iloc[-1] / aligned["bench"].iloc[0]) - 1

    return result


def summarize_decision_sources(decisions: dict[str, dict]) -> dict[str, dict]:
    """Tally decision points by source (schema v1.1 records).

    Returns {source: {count, tickers_touched, avg_positions, avg_cash_pct}}.
    """
    summary: dict[str, dict] = {}
    for _, rec in decisions.items():
        src = rec.get("source", "unknown")
        bucket = summary.setdefault(
            src,
            {"count": 0, "tickers_touched": set(), "positions": [], "cash": []},
        )
        bucket["count"] += 1
        weights = rec.get("weights", {})
        bucket["tickers_touched"].update(weights.keys())
        bucket["positions"].append(len(weights))
        bucket["cash"].append(rec.get("cash", 1.0 - sum(weights.values())))
    # Finalize
    final: dict[str, dict] = {}
    for src, b in summary.items():
        final[src] = {
            "count": b["count"],
            "tickers_touched": len(b["tickers_touched"]),
            "avg_positions": float(np.mean(b["positions"])) if b["positions"] else 0.0,
            "avg_cash_pct": float(np.mean(b["cash"])) if b["cash"] else 0.0,
        }
    return final


def compute_trade_stats(trades: list[dict]) -> dict[str, float | None]:
    """Compute trading statistics from a list of trade records.

    Each trade dict should have: pnl, entry_date, exit_date.
    """
    if not trades:
        return {}

    pnls = [t["pnl"] for t in trades if "pnl" in t]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]

    win_rate = len(winners) / len(pnls) if pnls else None
    avg_win = np.mean(winners) if winners else None
    avg_loss = abs(np.mean(losers)) if losers else None
    profit_factor = (avg_win / avg_loss) if avg_win and avg_loss else None

    # Average holding period
    hold_days = []
    for t in trades:
        if "entry_date" in t and "exit_date" in t:
            delta = (t["exit_date"] - t["entry_date"]).days
            hold_days.append(delta)
    avg_hold = np.mean(hold_days) if hold_days else None

    return {
        "total_trades": len(pnls),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "avg_holding_days": avg_hold,
    }
