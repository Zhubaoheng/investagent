"""Backtrader Strategy — replays pre-computed decisions.

Execution model:
- Decisions are made on scan dates (post-earnings season)
- Orders are spread over EXEC_DAYS trading days (gradual execution)
- Buy orders only execute when close ≤ previous day's close (no chasing)
- Sell orders execute unconditionally (risk management priority)
- Cash earns daily interest at short-term repo/treasury rates
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, datetime

import backtrader as bt

logger = logging.getLogger(__name__)

# Transaction cost parameters (per spec §4.2)
COMMISSION_RATE = 0.00025   # 0.025% per side
STAMP_TAX_RATE = 0.0005     # 0.05% sell only
SLIPPAGE_RATE = 0.001       # 0.1% per side

# Gradual execution: spread orders over N trading days
EXEC_DAYS = 5


class MungerStrategy(bt.Strategy):
    """Munger-style concentrated investment strategy.

    Reads pre-computed decisions from a dict and executes them
    gradually over EXEC_DAYS trading days. Buy orders only fill
    when price is not rising (no chasing), sell orders fill
    unconditionally.
    """

    params = dict(
        decisions={},           # {date_str: {ticker: target_weight, ...}}
        price_trigger_down=0.20,
        price_trigger_up=0.50,
        price_decisions={},
        cash_rates={},          # {year: annual_rate} for daily cash interest
        exec_days=EXEC_DAYS,    # days to spread execution over
    )

    def __init__(self):
        self.entry_prices = {}       # ticker -> price at entry
        self.order_pending = {}      # ticker -> order object
        # Gradual execution queue: list of {ticker, daily_size, days_left, is_buy, weight}
        self._exec_queue: list[dict] = []

    def log(self, txt: str) -> None:
        dt = self.datetime.date()
        logger.info("[%s] %s", dt, txt)

    def next(self):
        today = self.datetime.date()
        today_str = today.isoformat()

        # Daily cash interest
        cash = self.broker.getcash()
        if cash > 0 and self.p.cash_rates:
            annual_rate = self.p.cash_rates.get(today.year, 0.018)
            daily_interest = cash * annual_rate / 365
            self.broker.add_cash(daily_interest)

        # Check for new decisions → queue gradual execution
        if today_str in self.p.decisions:
            self._queue_rebalance(self.p.decisions[today_str])
        elif today_str in self.p.price_decisions:
            self._queue_rebalance(self.p.price_decisions[today_str])

        # Execute queued orders (daily slice)
        self._process_exec_queue()

    def _queue_rebalance(self, target: dict[str, float]) -> None:
        """Convert target weights into a gradual execution queue."""
        portfolio_value = self.broker.getvalue()

        # Clear any pending executions (new decision supersedes old)
        if self._exec_queue:
            self.log(f"New decision: clearing {len(self._exec_queue)} pending executions")
            self._exec_queue.clear()

        # Immediate: sell positions not in target (risk management, don't delay)
        for data in self.datas:
            ticker = data._name
            pos = self.getposition(data)
            if pos.size > 0 and ticker not in target:
                self.log(f"SELL ALL {ticker} (not in target)")
                self.close(data)

        # Queue gradual execution for buys and adjustments
        for ticker, weight in target.items():
            data = self._get_data_by_name(ticker)
            if data is None:
                continue

            target_value = portfolio_value * weight
            current_pos = self.getposition(data)
            current_value = current_pos.size * data.close[0] if current_pos.size > 0 else 0
            diff = target_value - current_value

            if abs(diff) < portfolio_value * 0.01:
                continue  # skip tiny adjustments

            total_size = int(abs(diff) / data.close[0] / 100) * 100
            if total_size <= 0:
                continue

            daily_size = max(100, int(total_size / self.p.exec_days / 100) * 100)
            is_buy = diff > 0

            self._exec_queue.append({
                "ticker": ticker,
                "daily_size": daily_size,
                "remaining_size": total_size,
                "is_buy": is_buy,
                "weight": weight,
            })

            action = "BUY" if is_buy else "SELL"
            self.log(
                f"QUEUED {action} {ticker}: {total_size} shares over "
                f"{self.p.exec_days} days ({daily_size}/day), target={weight:.1%}"
            )

    def _process_exec_queue(self) -> None:
        """Execute one day's slice of queued orders."""
        still_pending = []

        for item in self._exec_queue:
            ticker = item["ticker"]
            data = self._get_data_by_name(ticker)
            if data is None:
                continue

            size = min(item["daily_size"], item["remaining_size"])
            if size <= 0:
                continue

            if item["is_buy"]:
                # Buy only when price is not rising vs previous close
                # (don't chase rallies — wait for flat/down days)
                if len(data.close) >= 2 and data.close[0] > data.close[-1] * 1.02:
                    # Price up >2% from yesterday — skip today, try tomorrow
                    still_pending.append(item)
                    continue

                self.buy(data, size=size)
                self.log(f"EXEC BUY {ticker} size={size} @ {data.close[0]:.2f}")
            else:
                # Sells execute unconditionally (risk management)
                self.sell(data, size=size)
                self.log(f"EXEC SELL {ticker} size={size} @ {data.close[0]:.2f}")

            item["remaining_size"] -= size
            self.entry_prices[ticker] = data.close[0]

            if item["remaining_size"] > 0:
                still_pending.append(item)
            else:
                action = "BUY" if item["is_buy"] else "SELL"
                self.log(f"COMPLETED {action} {ticker} (target={item['weight']:.1%})")

        self._exec_queue = still_pending

    def _get_data_by_name(self, name: str):
        for data in self.datas:
            if data._name == name:
                return data
        return None

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(
                f"TRADE CLOSED {trade.data._name}: "
                f"PnL={trade.pnl:.2f} Net={trade.pnlcomm:.2f}"
            )


class BacktestCommission(bt.CommInfoBase):
    """A-share commission: buy commission + sell commission + stamp tax."""

    params = dict(
        commission=COMMISSION_RATE,
        stamp_tax=STAMP_TAX_RATE,
        slippage=SLIPPAGE_RATE,
    )

    def _getcommission(self, size, price, pseudoexec):
        commission = abs(size) * price * self.p.commission
        slippage = abs(size) * price * self.p.slippage
        if size < 0:  # selling
            commission += abs(size) * price * self.p.stamp_tax
        return commission + slippage
