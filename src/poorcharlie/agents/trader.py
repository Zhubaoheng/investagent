"""Trader Agent — Layer 4 execution arbiter.

Three-layer sandwich architecture:
  1. Pre-check (rules): block CORE sells that lack fundamental justification,
     cap single-position / industry weights, normalize total weight.
  2. LLM review: assess market regime, assign urgency per order, write
     execution plan summary.
  3. Post-check (rules): re-validate LLM output to ensure guardrails
     weren't bypassed. If LLM tried to un-block a CORE sell, re-block it.

The Trader does NOT second-guess stock selection (that's PortfolioStrategy's
job). It only decides HOW and WHEN to execute what's been decided.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from poorcharlie.agents.base import BaseAgent
from poorcharlie.schemas.common import BaseAgentOutput
from poorcharlie.schemas.portfolio_strategy import ActionType, PositionTier
from poorcharlie.schemas.trader import (
    MarketRegime,
    OrderAction,
    OrderUrgency,
    TraderInput,
    TraderOrder,
    TraderOutput,
)

logger = logging.getLogger(__name__)

# Position caps — mirror Munger "single position max 20%" constraint.
_SINGLE_POSITION_MAX = 0.20
_TOTAL_WEIGHT_MAX = 1.0

# Quality rank for classifying "permanent damage" on a CORE position.
_PERMANENT_DAMAGE_QUALITIES = {"POOR", "BELOW_AVERAGE"}


# ---------------------------------------------------------------------------
# Pre-check rules (before LLM)
# ---------------------------------------------------------------------------

def _core_sell_has_justification(
    ticker: str, candidate_details: dict[str, dict],
) -> tuple[bool, str]:
    """Check if a CORE position's sell is justified by permanent damage.

    Returns (is_justified, reason). A CORE sell is allowed only when at
    least one of these is true:
      - Critic reported non-empty kill_shots
      - AccountingRisk level is RED
      - enterprise_quality fell to BELOW_AVERAGE or POOR
      - final_label fell to REJECT or TOO_HARD

    Purely valuation-driven sells on CORE positions are NEVER justified —
    that is the whole point of CORE tier.
    """
    det = candidate_details.get(ticker, {}) or {}
    quality = (det.get("enterprise_quality", "") or "").upper()
    label = (det.get("final_label", "") or "").upper()
    risk = (det.get("accounting_risk_level", "") or "").upper()
    kill_shots = det.get("kill_shots") or []

    if kill_shots:
        return True, f"Critic reports {len(kill_shots)} kill_shot(s)"
    if risk == "RED":
        return True, "AccountingRisk = RED"
    if quality in _PERMANENT_DAMAGE_QUALITIES:
        return True, f"enterprise_quality = {quality}"
    if label in ("REJECT", "TOO_HARD"):
        return True, f"final_label = {label}"
    return False, (
        "no kill_shots, accounting still OK, quality still acceptable, "
        "label still actionable — CORE sell is not justified"
    )


def _pre_check_decisions(
    proposed_decisions: list[dict],
    current_holdings: list[dict],
    candidate_details: dict[str, dict],
) -> tuple[list[dict], list[str]]:
    """Apply hard rules to proposed decisions before the LLM sees them.

    Returns (enriched_decisions, blocked_tickers).
    Each enriched decision gets:
      - `original_action`: the Strategy's proposal
      - `action`: possibly rewritten (e.g., EXIT → BLOCKED)
      - `blocked_reason`: populated when action was changed to BLOCKED
    """
    holdings_by_ticker = {h["ticker"]: h for h in current_holdings}
    blocked: list[str] = []
    enriched: list[dict] = []

    for d in proposed_decisions:
        enriched_d = dict(d)
        ticker = enriched_d.get("ticker", "")
        original_action = (enriched_d.get("action") or "").upper()
        enriched_d["original_action"] = original_action
        enriched_d["blocked_reason"] = ""

        # Identify if this is a CORE-tier position being sold
        tier = (enriched_d.get("position_tier") or "").upper()
        # For held positions, authoritative tier comes from stored holding
        stored_tier = (
            (holdings_by_ticker.get(ticker, {}) or {}).get("position_tier", "")
            or ""
        ).upper()
        effective_tier = stored_tier or tier

        is_sell_action = original_action in ("EXIT", "REDUCE")
        is_core = effective_tier == "CORE"

        if is_sell_action and is_core:
            justified, reason = _core_sell_has_justification(
                ticker, candidate_details,
            )
            if not justified:
                # Block the sell: keep current weight, mark BLOCKED
                holding = holdings_by_ticker.get(ticker, {})
                keep_weight = holding.get("target_weight") or enriched_d.get(
                    "current_weight", 0.0,
                )
                enriched_d["action"] = "BLOCKED"
                enriched_d["target_weight"] = keep_weight
                enriched_d["blocked_reason"] = (
                    f"CORE sell blocked by rules: {reason}"
                )
                blocked.append(ticker)
                logger.info(
                    "Trader pre-check BLOCKED %s (%s): %s",
                    ticker, effective_tier, reason,
                )
                enriched.append(enriched_d)
                continue

        # Cap single position weight
        tw = float(enriched_d.get("target_weight") or 0.0)
        if tw > _SINGLE_POSITION_MAX:
            logger.info(
                "Trader pre-check capped %s weight %.2f -> %.2f (single-pos max)",
                ticker, tw, _SINGLE_POSITION_MAX,
            )
            enriched_d["target_weight"] = _SINGLE_POSITION_MAX

        enriched.append(enriched_d)

    # Normalize total weight if it exceeds 100% (excluding BLOCKED which
    # preserve existing weight and shouldn't drive normalization)
    total = sum(
        float(d.get("target_weight") or 0.0)
        for d in enriched
        if d.get("action") != "BLOCKED" and d.get("action") != "EXIT"
    )
    blocked_sum = sum(
        float(d.get("target_weight") or 0.0)
        for d in enriched
        if d.get("action") == "BLOCKED"
    )
    available = max(0.0, _TOTAL_WEIGHT_MAX - blocked_sum)
    if total > available and total > 0:
        scale = available / total
        logger.info(
            "Trader pre-check scaling active weights by %.3f "
            "(total=%.3f, blocked=%.3f, available=%.3f)",
            scale, total, blocked_sum, available,
        )
        for d in enriched:
            if d.get("action") in ("BLOCKED", "EXIT"):
                continue
            d["target_weight"] = float(d.get("target_weight") or 0.0) * scale

    return enriched, blocked


# ---------------------------------------------------------------------------
# Post-check rules (after LLM)
# ---------------------------------------------------------------------------

def _post_check_output(
    output: TraderOutput,
    pre_checked_decisions: list[dict],
) -> TraderOutput:
    """Re-validate LLM output against pre-check decisions.

    The LLM might try (intentionally or due to hallucination) to:
      - Change a BLOCKED order back to EXIT → re-block it
      - Add/remove orders → re-align with pre-check list
      - Push target_weight beyond 20% → cap it
      - Let total > 100% → normalize
    """
    pre_by_ticker = {d["ticker"]: d for d in pre_checked_decisions}
    fixed_orders: list[TraderOrder] = []
    corrections: list[str] = []

    # Ensure every pre-checked ticker has exactly one order; drop hallucinated
    # tickers from LLM output.
    llm_by_ticker = {o.ticker: o for o in output.orders}

    for pre_d in pre_checked_decisions:
        ticker = pre_d["ticker"]
        pre_action = pre_d.get("action", "")
        pre_blocked_reason = pre_d.get("blocked_reason", "")
        llm_order = llm_by_ticker.get(ticker)

        if llm_order is None:
            # LLM dropped this ticker — reconstruct from pre-check
            fixed_orders.append(TraderOrder(
                ticker=ticker,
                name=pre_d.get("name", ""),
                action=OrderAction(pre_action if pre_action else "HOLD"),
                target_weight=float(pre_d.get("target_weight") or 0.0),
                current_weight=float(pre_d.get("current_weight") or 0.0),
                position_tier=PositionTier(
                    (pre_d.get("position_tier") or "SATELLITE").upper()
                ),
                urgency=OrderUrgency.NORMAL,
                rationale="[reconstructed: LLM dropped this ticker]",
                blocked_reason=pre_blocked_reason,
            ))
            corrections.append(f"{ticker} reconstructed (LLM dropped)")
            continue

        # Re-enforce BLOCKED: if pre-check blocked this, action MUST be BLOCKED
        if pre_action == "BLOCKED":
            if llm_order.action != OrderAction.BLOCKED:
                corrections.append(
                    f"{ticker} re-blocked (LLM tried to set {llm_order.action})"
                )
                llm_order = llm_order.model_copy(update={
                    "action": OrderAction.BLOCKED,
                    "target_weight": float(pre_d.get("target_weight") or 0.0),
                    "blocked_reason": pre_blocked_reason,
                })

        # Cap single-position weight
        if llm_order.target_weight > _SINGLE_POSITION_MAX:
            corrections.append(
                f"{ticker} re-capped {llm_order.target_weight:.3f} "
                f"-> {_SINGLE_POSITION_MAX}"
            )
            llm_order = llm_order.model_copy(update={
                "target_weight": _SINGLE_POSITION_MAX,
            })

        fixed_orders.append(llm_order)

    # Drop hallucinated tickers that weren't in pre-check
    pre_tickers = {d["ticker"] for d in pre_checked_decisions}
    dropped = [t for t in llm_by_ticker if t not in pre_tickers]
    if dropped:
        corrections.append(f"dropped hallucinated tickers: {dropped}")

    # Normalize total weight if still over budget (excl. BLOCKED/EXIT)
    active_total = sum(
        o.target_weight for o in fixed_orders
        if o.action not in (OrderAction.BLOCKED, OrderAction.EXIT)
    )
    blocked_total = sum(
        o.target_weight for o in fixed_orders
        if o.action == OrderAction.BLOCKED
    )
    available = max(0.0, _TOTAL_WEIGHT_MAX - blocked_total)
    if active_total > available and active_total > 0:
        scale = available / active_total
        corrections.append(
            f"normalized active weights by {scale:.3f} "
            f"(active={active_total:.3f}, blocked={blocked_total:.3f})"
        )
        fixed_orders = [
            o.model_copy(update={"target_weight": o.target_weight * scale})
            if o.action not in (OrderAction.BLOCKED, OrderAction.EXIT)
            else o
            for o in fixed_orders
        ]

    if corrections:
        logger.info("Trader post-check corrections: %s", "; ".join(corrections))

    blocked_count = sum(1 for o in fixed_orders if o.action == OrderAction.BLOCKED)
    return output.model_copy(update={
        "orders": fixed_orders,
        "blocked_orders_count": blocked_count,
    })


# ---------------------------------------------------------------------------
# TraderAgent
# ---------------------------------------------------------------------------

class TraderAgent(BaseAgent):
    """Layer 4 execution arbiter: rules + LLM + rules."""

    name: str = "trader"

    def _output_type(self) -> type[BaseAgentOutput]:
        return TraderOutput

    def _agent_role_description(self) -> str:
        return (
            "你是交易员代理（Trader Agent），Layer 4 执行裁决者。"
            "上游 PortfolioStrategy 已给出目标组合，你的职责是：\n"
            "  1. 评估市场环境（PANIC/NORMAL/EUPHORIA）\n"
            "  2. 为每条订单打 urgency 标签（IMMEDIATE/NORMAL/PATIENT）\n"
            "  3. 给出执行计划，不改变 action\n\n"
            "【强硬边界】\n"
            "规则层已经把非法的 CORE 卖出改成了 action=BLOCKED。"
            "**你不能把 BLOCKED 改回 EXIT 或 REDUCE**。如果你这么做了，"
            "post-process 规则会把它改回来。你的工作是尊重规则层的决定，"
            "专注于执行层面的判断。\n\n"
            "芒格：'Never interrupt compound interest unnecessarily.' "
            "你的默认倾向应该是保守执行、减少冲击。"
        )

    def _build_user_context(
        self, input_data: BaseModel, ctx: Any = None,
    ) -> dict[str, Any]:
        data = (
            input_data
            if isinstance(input_data, TraderInput)
            else TraderInput.model_validate(input_data)
        )

        # The caller (decision_pipeline) runs pre-check first and stores the
        # enriched decisions on the agent instance so we can pass them to
        # the LLM prompt with original_action + blocked_reason fields.
        pre_checked = getattr(self, "_pre_checked_decisions", None)
        if pre_checked is None:
            pre_checked, _ = _pre_check_decisions(
                data.proposed_decisions,
                data.current_holdings,
                data.candidate_details,
            )
            self._pre_checked_decisions = pre_checked

        # Keep candidate details compact — just the fields the Trader cares about
        candidate_details_brief = {}
        for ticker, det in data.candidate_details.items():
            candidate_details_brief[ticker] = {
                "enterprise_quality": det.get("enterprise_quality", ""),
                "final_label": det.get("final_label", ""),
                "margin_of_safety_pct": (
                    f"{det['margin_of_safety_pct']:.0%}"
                    if det.get("margin_of_safety_pct") is not None
                    else "N/A"
                ),
                "kill_shots_count": len(det.get("kill_shots") or []),
            }

        holdings_rendered = []
        for h in data.current_holdings:
            holdings_rendered.append({
                "ticker": h.get("ticker", ""),
                "name": h.get("name", ""),
                "weight": f"{h.get('target_weight', 0.0):.0%}",
                "industry": h.get("industry", "未知"),
                "position_tier": h.get("position_tier", "SATELLITE"),
            })

        return {
            "has_holdings": len(holdings_rendered) > 0,
            "current_holdings": holdings_rendered,
            "has_triggers": len(data.change_triggers) > 0,
            "change_triggers": data.change_triggers,
            "proposed_decisions": pre_checked,
            "candidate_details_brief": candidate_details_brief,
        }

    async def run(
        self, input_data: BaseModel, ctx: Any = None, *, max_retries: int = 2,
    ) -> TraderOutput:
        """Execute three-layer sandwich: pre-check → LLM → post-check."""
        data = (
            input_data
            if isinstance(input_data, TraderInput)
            else TraderInput.model_validate(input_data)
        )

        # Layer 1: Pre-check
        pre_checked, blocked_tickers = _pre_check_decisions(
            data.proposed_decisions,
            data.current_holdings,
            data.candidate_details,
        )
        self._pre_checked_decisions = pre_checked
        if blocked_tickers:
            logger.info(
                "Trader pre-check blocked %d order(s): %s",
                len(blocked_tickers), blocked_tickers,
            )

        # Layer 2: LLM
        try:
            llm_output: TraderOutput = await super().run(  # type: ignore[assignment]
                data, ctx, max_retries=max_retries,
            )
        except Exception:
            logger.warning(
                "Trader LLM failed — falling back to rules-only output",
                exc_info=True,
            )
            # Rules-only fallback: build orders directly from pre_checked
            from datetime import datetime, timezone
            from poorcharlie.schemas.common import AgentMeta

            fallback_orders = []
            for d in pre_checked:
                action_str = (d.get("action") or "HOLD").upper()
                try:
                    action = OrderAction(action_str)
                except ValueError:
                    action = OrderAction.HOLD
                try:
                    tier = PositionTier(
                        (d.get("position_tier") or "SATELLITE").upper()
                    )
                except ValueError:
                    tier = PositionTier.SATELLITE
                fallback_orders.append(TraderOrder(
                    ticker=d.get("ticker", ""),
                    name=d.get("name", ""),
                    action=action,
                    target_weight=float(d.get("target_weight") or 0.0),
                    current_weight=float(d.get("current_weight") or 0.0),
                    position_tier=tier,
                    urgency=OrderUrgency.NORMAL,
                    rationale="[rules-only fallback: Trader LLM unavailable]",
                    blocked_reason=d.get("blocked_reason", ""),
                ))
            llm_output = TraderOutput(
                meta=AgentMeta(
                    agent_name=self.name,
                    timestamp=datetime.now(tz=timezone.utc),
                    model_used="rules-only-fallback",
                    token_usage=0,
                ),
                market_regime=MarketRegime.UNKNOWN,
                market_assessment="LLM fallback — no market assessment",
                orders=fallback_orders,
                blocked_orders_count=len(blocked_tickers),
                overall_urgency=OrderUrgency.NORMAL,
                execution_plan_summary="rules-only execution (LLM fallback)",
            )

        # Layer 3: Post-check
        return _post_check_output(llm_output, pre_checked)
