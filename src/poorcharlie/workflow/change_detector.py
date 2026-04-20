"""Change detector — decide whether a scan warrants portfolio rebalancing.

Philosophy: Munger-style "sit on your ass" investing. A scheduled scan is not,
by itself, a reason to trade. The previous scan already produced a considered
portfolio; we only rebalance when something material has changed.

Five triggers justify re-running PortfolioStrategy (LLM):

  T1. A held position's enterprise_quality was downgraded since last scan.
  T2. A held position has new kill_shots from Critic (vs previous scan).
  T3. A held position's accounting_risk_level turned RED.
  T4. A new candidate earned INVESTABLE (not previously held, not previously
      INVESTABLE).
  T5. A held position's final_label fell to REJECT/TOO_HARD.

Without any trigger, the pipeline returns the previous scan's allocations
unchanged and skips the LLM call entirely — this is the deliberate
"nothing changed, don't trade" path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from poorcharlie.schemas.candidate import CandidateSnapshot, PortfolioHolding

logger = logging.getLogger(__name__)


# Quality ranking for downgrade detection. Higher int = better.
_QUALITY_RANK = {
    "GREAT": 4,
    "GOOD": 3,
    "AVERAGE": 2,
    "BELOW_AVERAGE": 1,
    "POOR": 0,
    "": -1,  # unknown treated as no prior signal
}


class TriggerType(str, Enum):
    QUALITY_DOWNGRADE = "QUALITY_DOWNGRADE"
    NEW_KILL_SHOT = "NEW_KILL_SHOT"
    ACCOUNTING_RED = "ACCOUNTING_RED"
    NEW_INVESTABLE = "NEW_INVESTABLE"
    LABEL_REJECT = "LABEL_REJECT"


@dataclass(frozen=True)
class ChangeTrigger:
    """A single reason why the portfolio needs re-evaluation."""

    ticker: str
    name: str
    trigger_type: TriggerType
    detail: str

    def describe(self) -> str:
        return f"[{self.trigger_type.value}] {self.ticker} {self.name}: {self.detail}"


def _quality_rank(quality: str) -> int:
    return _QUALITY_RANK.get((quality or "").upper(), -1)


def _check_quality_downgrade(snap: CandidateSnapshot) -> ChangeTrigger | None:
    if not snap.prev_enterprise_quality or not snap.enterprise_quality:
        return None
    prev_rank = _quality_rank(snap.prev_enterprise_quality)
    cur_rank = _quality_rank(snap.enterprise_quality)
    if prev_rank < 0 or cur_rank < 0:
        return None
    if cur_rank < prev_rank:
        return ChangeTrigger(
            ticker=snap.ticker,
            name=snap.name,
            trigger_type=TriggerType.QUALITY_DOWNGRADE,
            detail=f"enterprise_quality {snap.prev_enterprise_quality} -> {snap.enterprise_quality}",
        )
    return None


def _check_new_kill_shots(snap: CandidateSnapshot) -> ChangeTrigger | None:
    prev_set = {k.strip() for k in (snap.prev_kill_shots or []) if k.strip()}
    cur_set = {k.strip() for k in (snap.kill_shots or []) if k.strip()}
    new_shots = cur_set - prev_set
    if not new_shots:
        return None
    return ChangeTrigger(
        ticker=snap.ticker,
        name=snap.name,
        trigger_type=TriggerType.NEW_KILL_SHOT,
        detail=f"{len(new_shots)} new kill_shot(s): " + "; ".join(list(new_shots)[:2]),
    )


def _check_accounting_red(snap: CandidateSnapshot) -> ChangeTrigger | None:
    cur = (snap.accounting_risk_level or "").upper()
    prev = (snap.prev_accounting_risk_level or "").upper()
    if cur == "RED" and prev != "RED":
        return ChangeTrigger(
            ticker=snap.ticker,
            name=snap.name,
            trigger_type=TriggerType.ACCOUNTING_RED,
            detail=f"accounting_risk_level {prev or 'UNKNOWN'} -> RED",
        )
    return None


def _check_label_reject(snap: CandidateSnapshot) -> ChangeTrigger | None:
    cur = (snap.final_label or "").upper()
    prev = (snap.prev_final_label or "").upper()
    if cur in ("REJECT", "TOO_HARD") and prev not in ("REJECT", "TOO_HARD", ""):
        return ChangeTrigger(
            ticker=snap.ticker,
            name=snap.name,
            trigger_type=TriggerType.LABEL_REJECT,
            detail=f"final_label {prev} -> {cur}",
        )
    return None


def _check_new_investable(
    snap: CandidateSnapshot, held_tickers: set[str],
) -> ChangeTrigger | None:
    if snap.ticker in held_tickers:
        return None
    if (snap.final_label or "").upper() != "INVESTABLE":
        return None
    # New = was not INVESTABLE before (or was a first-time appearance)
    if (snap.prev_final_label or "").upper() == "INVESTABLE":
        return None
    return ChangeTrigger(
        ticker=snap.ticker,
        name=snap.name,
        trigger_type=TriggerType.NEW_INVESTABLE,
        detail=(
            f"new INVESTABLE candidate "
            f"(prev label: {snap.prev_final_label or 'none'})"
        ),
    )


def detect_change_triggers(
    candidates: list[CandidateSnapshot],
    holdings: list[PortfolioHolding],
) -> list[ChangeTrigger]:
    """Scan candidates + holdings for any trigger justifying portfolio re-evaluation.

    Returns an empty list when nothing material changed — caller should then
    skip the LLM and preserve previous allocations.

    Cold start (no prev_* fields populated anywhere) returns one synthetic
    trigger so the first scan always runs the LLM. The caller detects this
    via has_cold_start_marker().

    Trigger details are annotated with the held position's tier (CORE or
    SATELLITE) when applicable — downstream consumers (Trader, Portfolio
    prompt) use this to apply differentiated treatment: CORE positions
    should only EXIT on permanent damage, never on valuation.
    """
    held_tickers = {h.ticker for h in holdings}
    holdings_by_ticker = {h.ticker: h for h in holdings}
    triggers: list[ChangeTrigger] = []

    # Cold start detection: if NO candidate has prev_scan_date set, this is
    # the first time the store has been written. Always run LLM.
    any_prior = any(c.prev_scan_date is not None for c in candidates)
    if not any_prior and candidates:
        logger.info("change_detector: cold start (no prior scan) — LLM will run")
        return [ChangeTrigger(
            ticker="*",
            name="cold_start",
            trigger_type=TriggerType.NEW_INVESTABLE,
            detail="first scan — no previous portfolio to preserve",
        )]

    def _annotate_with_tier(
        trigger: ChangeTrigger | None, ticker: str,
    ) -> ChangeTrigger | None:
        """Prepend the held position's tier to the trigger detail."""
        if trigger is None:
            return None
        holding = holdings_by_ticker.get(ticker)
        if holding is None:
            return trigger
        tier = holding.position_tier or "SATELLITE"
        return ChangeTrigger(
            ticker=trigger.ticker,
            name=trigger.name,
            trigger_type=trigger.trigger_type,
            detail=f"[{tier}] {trigger.detail}",
        )

    for snap in candidates:
        is_held = snap.ticker in held_tickers

        if is_held:
            for check in (
                _check_quality_downgrade,
                _check_new_kill_shots,
                _check_accounting_red,
                _check_label_reject,
            ):
                t = _annotate_with_tier(check(snap), snap.ticker)
                if t is not None:
                    triggers.append(t)

        # New INVESTABLE candidates apply regardless of held status
        t = _check_new_investable(snap, held_tickers)
        if t is not None:
            triggers.append(t)

    # Also check held tickers that fell out of the candidate pool entirely
    # (ingest_scan_results removes them or marks EXITED). Those are already
    # reflected by label transitions when present; if the ticker has no
    # snapshot at all we rely on downstream pipeline handling.

    if triggers:
        logger.info(
            "change_detector: %d trigger(s) — LLM will run:\n  %s",
            len(triggers),
            "\n  ".join(t.describe() for t in triggers[:10]),
        )
    else:
        logger.info(
            "change_detector: no triggers — preserving previous portfolio (no LLM call)"
        )

    return triggers


def has_cold_start_marker(triggers: list[ChangeTrigger]) -> bool:
    """Detect the synthetic cold-start trigger."""
    return any(t.ticker == "*" and t.name == "cold_start" for t in triggers)
