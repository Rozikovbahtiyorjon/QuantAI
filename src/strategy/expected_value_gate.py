"""
ExpectedValueGate — Mandatory gate for new Entry Engine (P3.14)

Computes:
  P(win) + P(loss) + expected payoff + fees + slippage + estimated execution cost
  = expected net return
  → PASS / FAIL

This is mandatory for every entry, not optional.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class ExpectedValueResult:
    expected_net: float  # E[net] = Pwin*avg_win_net - Ploss*avg_loss_net - costs
    p_win: float
    p_loss: float
    expected_payoff: float  # Pwin*avg_win - Ploss*avg_loss (gross)
    total_costs: float  # fees+slippage+spread+funding+latency
    gross_win: float
    gross_loss: float
    passed: bool
    reason: str
    hurdle: float = 0.0


class ExpectedValueGate:
    """
    Mandatory Expected Value Gate.

    Formula:
      E[net] = P(win)*avg_win - P(loss)*avg_loss - total_costs
      total_costs = commission*2 + slippage*2 + spread + funding + latency_cost + execution_cost

    Trade only if E[net] > hurdle (e.g., 0.001 = 0.1% net edge).
    This is more robust than P(win) alone or RR alone.
    """

    def __init__(
        self,
        hurdle: float = 0.001,  # 0.1% net edge required
        commission: float = 0.0004,  # per side
        slippage: float = 0.0002,  # per side
        spread: float = 0.0001,  # half spread per side
        funding: float = 0.0,  # per horizon
        latency_cost_per_ms: float = 0.0000002,  # 0.02% per 1000ms
    ) -> None:
        self.hurdle = float(hurdle)
        self.commission = float(commission)
        self.slippage = float(slippage)
        self.spread = float(spread)
        self.funding = float(funding)
        self.latency_cost_per_ms = float(latency_cost_per_ms)

    def evaluate(
        self,
        p_win: float,
        avg_win: float,  # gross avg win (e.g., 0.015 = 1.5%)
        avg_loss: float,  # gross avg loss positive value (e.g., 0.008 = 0.8% risk)
        latency_ms: float = 100.0,
        execution_cost: float = 0.0,  # additional estimated execution cost
    ) -> ExpectedValueResult:
        """
        Evaluate expected net return.

        Args:
            p_win: probability of win (0-1) from ML or historical win rate
            avg_win: average win gross % (e.g., 0.015)
            avg_loss: average loss gross % positive (e.g., 0.008)
            latency_ms: estimated latency for cost
            execution_cost: additional execution cost estimate

        Returns:
            ExpectedValueResult with expected_net and PASS/FAIL
        """
        p_win = max(0.0, min(1.0, float(p_win)))
        p_loss = 1.0 - p_win
        avg_win = float(avg_win)
        avg_loss = float(avg_loss)

        # Gross expected payoff
        expected_payoff = p_win * avg_win - p_loss * avg_loss

        # Total costs
        # Round trip: commission*2 + slippage*2 + spread + funding + latency
        latency_cost = float(latency_ms) * self.latency_cost_per_ms
        total_costs = (
            2 * self.commission
            + 2 * self.slippage
            + self.spread
            + self.funding
            + latency_cost
            + float(execution_cost)
        )

        expected_net = expected_payoff - total_costs

        passed = expected_net > self.hurdle
        reason = (
            f"E[net]={expected_net:.4f} (Pwin {p_win:.2%}*win {avg_win:.4f} - Ploss {p_loss:.2%}*loss {avg_loss:.4f} = payoff {expected_payoff:.4f} - costs {total_costs:.4f}) "
            f"{'>' if passed else '<='} hurdle {self.hurdle:.4f} → {'PASS' if passed else 'FAIL'}"
        )

        return ExpectedValueResult(
            expected_net=expected_net,
            p_win=p_win,
            p_loss=p_loss,
            expected_payoff=expected_payoff,
            total_costs=total_costs,
            gross_win=avg_win,
            gross_loss=avg_loss,
            passed=passed,
            reason=reason,
            hurdle=self.hurdle,
        )

    def evaluate_from_ml(
        self,
        p_win: float,  # from ML P(win)
        sl_pct: float,  # stop loss % (e.g., 0.015 = 1.5% risk)
        tp_pct: float,  # take profit % (e.g., 0.03 = 3% reward)
        latency_ms: float = 100.0,
    ) -> ExpectedValueResult:
        """
        Convenience: evaluate from ML P(win) and SL/TP distances.

        Uses SL/TP as avg_win/avg_loss proxies.
        """
        return self.evaluate(
            p_win=p_win,
            avg_win=float(tp_pct),
            avg_loss=float(sl_pct),
            latency_ms=latency_ms,
        )
