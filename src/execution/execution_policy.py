"""
Execution Policy — P3.17

Not strictly Maker Limit. Correct is Execution Policy where decision depends on:
  setup, urgency, spread, volatility, expected slippage, queue/fill probability, EV after fill probability

Allowed:
  LIMIT_MAKER (post-only, fee saving)
  LIMIT (taker if crosses, maker otherwise)
  MARKET (immediate, urgency high)

Execution must consider EV after fill probability:
  EV = P(fill) * E[net] - (1-P(fill)) * opportunity_cost
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import math


ExecutionType = Literal["LIMIT_MAKER", "LIMIT", "MARKET"]


@dataclass
class ExecutionContext:
    setup: str  # e.g., LONG_BREAKOUT, LONG_PULLBACK
    urgency: float  # 0.0-1.0 (0 = patient, 1 = urgent breakout)
    spread_pct: float  # e.g., 0.0005 = 0.05%
    volatility_atr_pct: float  # atr/close
    expected_slippage: float  # e.g., 0.0002
    queue_probability: float  # P(fill) for limit, 0-1
    expected_net_edge: float  # E[net] from ExpectedValueGate
    is_volatile_regime: bool = False


@dataclass
class ExecutionDecision:
    execution_type: ExecutionType
    reason: str
    expected_ev_after_fill: float
    fill_probability: float
    limit_price: float | None = None


class ExecutionPolicy:
    """
    Chooses execution type based on setup, urgency, spread, volatility, slippage, queue, EV.

    Rules:
      - High urgency (breakout, 0.8+) + high volatility + low queue prob → MARKET
      - Low urgency (mean reversion, pullback) + tight spread + high queue prob → LIMIT_MAKER
      - Otherwise → LIMIT
      - Always compute EV after fill probability: EV = P(fill)*E[net] - (1-P)*opportunity
        If EV < hurdle, may still choose MARKET if urgency high and EV after fill for market > limit
    """

    def __init__(
        self,
        maker_fee: float = -0.0001,  # maker rebate often negative
        taker_fee: float = 0.0004,
        hurdle: float = 0.001,
    ) -> None:
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.hurdle = hurdle

    def decide(
        self,
        ctx: ExecutionContext,
        entry_price: float,
        atr: float,
    ) -> ExecutionDecision:
        # Base fill probabilities
        # LIMIT_MAKER: post-only, lower fill prob than LIMIT
        p_maker = max(0.1, ctx.queue_probability * 0.7)  # maker 30% lower
        p_limit = ctx.queue_probability
        p_market = 0.99  # almost certain

        # Costs
        maker_cost = self.maker_fee + ctx.expected_slippage * 0.5  # maker saves slippage
        taker_cost = self.taker_fee + ctx.expected_slippage

        # EV after fill for each type
        ev_maker = p_maker * (ctx.expected_net_edge - maker_cost) - (1 - p_maker) * 0.002  # opportunity cost 0.2% if missed
        ev_limit = p_limit * (ctx.expected_net_edge - taker_cost * 0.5) - (1 - p_limit) * 0.001
        ev_market = p_market * (ctx.expected_net_edge - taker_cost)

        # Urgency overrides
        if ctx.urgency > 0.8 and ctx.volatility_atr_pct > 0.02:
            # Volatile breakout — market to avoid missing
            if ev_market > self.hurdle * 0.5:  # even if slightly below hurdle, urgency
                return ExecutionDecision(
                    execution_type="MARKET",
                    reason=f"urgency {ctx.urgency:.2f} + vol {ctx.volatility_atr_pct:.3f} → MARKET (ev_market {ev_market:.4f})",
                    expected_ev_after_fill=ev_market,
                    fill_probability=p_market,
                )

        # Tight spread + high queue prob → maker
        if ctx.spread_pct < 0.0008 and p_maker > 0.6 and ctx.urgency < 0.5:
            if ev_maker > self.hurdle:
                return ExecutionDecision(
                    execution_type="LIMIT_MAKER",
                    reason=f"tight spread {ctx.spread_pct:.4f} queue {p_maker:.2f} → LIMIT_MAKER (ev {ev_maker:.4f})",
                    expected_ev_after_fill=ev_maker,
                    fill_probability=p_maker,
                    limit_price=entry_price,
                )

        # Default: choose best EV among the three that exceeds hurdle
        best_ev = max(ev_maker, ev_limit, ev_market)
        if best_ev < self.hurdle:
            # None exceeds hurdle — still choose highest EV but note
            # For mean reversion, even low EV limit may be best due to high fill prob
            pass

        if ev_limit >= ev_maker and ev_limit >= ev_market:
            return ExecutionDecision(
                execution_type="LIMIT",
                reason=f"LIMIT best EV {ev_limit:.4f} (maker {ev_maker:.4f} market {ev_market:.4f}) setup {ctx.setup}",
                expected_ev_after_fill=ev_limit,
                fill_probability=p_limit,
                limit_price=entry_price,
            )
        elif ev_maker >= ev_market:
            return ExecutionDecision(
                execution_type="LIMIT_MAKER",
                reason=f"LIMIT_MAKER best EV {ev_maker:.4f} setup {ctx.setup}",
                expected_ev_after_fill=ev_maker,
                fill_probability=p_maker,
                limit_price=entry_price,
            )
        else:
            return ExecutionDecision(
                execution_type="MARKET",
                reason=f"MARKET best EV {ev_market:.4f} setup {ctx.setup}",
                expected_ev_after_fill=ev_market,
                fill_probability=p_market,
            )
