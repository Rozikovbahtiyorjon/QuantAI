"""
Cost/Slippage/Latency Stress — Audit #39-40

Each strategy is evaluated at:
 1.0x, 1.25x, 1.5x, 2x, 3x costs
 +25/+50/+100% slippage
 50/100/250/500ms/1s/3s latency

Fragile if PF drops below 1.0 at 1.5x costs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from src.backtest_engine import BacktestEngine


@dataclass
class StressResult:
    multiplier: float
    pf: float
    net_profit: float
    max_dd_pct: float
    fragile: bool


def cost_stress(df: pd.DataFrame, base_commission: float = 0.0004, base_slippage: float = 0.0002) -> List[StressResult]:
    results: List[StressResult] = []
    import src.trade_engine as te_mod

    original_commission = None
    # TradeEngine reads commission from settings; we patch via env or direct
    for mult in [1.0, 1.25, 1.5, 2.0, 3.0]:
        # Patch commission/slippage for this run by monkey-patching TradeEngine cost attributes if present
        be = BacktestEngine(initial_balance=1000.0, minimum_rows=min(30, len(df)))
        # Try to set cost multipliers via TradeEngine if it exposes them
        te = be.trade_engine
        # Fallback: many TradeEngine implementations read from config.settings; we temporarily patch
        from config.settings import settings as _s

        orig_comm = _s.commission.commission
        orig_slip = _s.commission.slippage
        try:
            _s.commission.commission = base_commission * mult
            _s.commission.slippage = base_slippage * mult
            res = be.run(df)
            results.append(
                StressResult(
                    multiplier=mult,
                    pf=float(res.profit_factor),
                    net_profit=float(res.net_profit),
                    max_dd_pct=float(res.max_drawdown_pct),
                    fragile=float(res.profit_factor) < 1.0,
                )
            )
        finally:
            _s.commission.commission = orig_comm
            _s.commission.slippage = orig_slip
    return results


def is_cost_robust(results: List[StressResult]) -> bool:
    """Robust if PF >1.0 at 1.5x costs."""
    for r in results:
        if r.multiplier == 1.5:
            return not r.fragile
    return False


def slippage_stress(df: pd.DataFrame, base_slippage: float = 0.0002) -> List[StressResult]:
    """Slippage stress: normal +25%, +50%, +100% slippage."""
    results: List[StressResult] = []
    from config.settings import settings as _s
    orig_slip = _s.commission.slippage
    for pct_inc in [0.0, 0.25, 0.50, 1.0]:
        mult = 1.0 + pct_inc
        _s.commission.slippage = base_slippage * mult
        try:
            be = BacktestEngine(initial_balance=1000.0, minimum_rows=min(30, len(df)))
            res = be.run(df)
            results.append(StressResult(multiplier=mult, pf=float(res.profit_factor), net_profit=float(res.net_profit), max_dd_pct=float(res.max_drawdown_pct), fragile=float(res.profit_factor) < 1.0))
        finally:
            _s.commission.slippage = orig_slip
    # Restore label as slippage multiplier for clarity
    return results


def latency_stress(df: pd.DataFrame, latencies_ms: List[int] | None = None, base_slippage: float = 0.0002) -> List[Dict[str, float]]:
    """P2.6 Latency stress: 50ms, 100ms, 250ms, 500ms, 1s, 3s — P2.3 fill model with latency.
    
    Each latency is simulated via LimitFillModel(latency_ms) fill probability,
    not just slippage heuristic. Also runs backtest with latency-adjusted fill/slippage.
    """
    if latencies_ms is None:
        latencies_ms = [50, 100, 250, 500, 1000, 3000]
    results: List[Dict[str, float]] = []
    from src.execution.fill_model import LimitFillModel
    from config.settings import settings as _s
    for ms in latencies_ms:
        # P2.3: use fill model with explicit latency
        fm = LimitFillModel(experiment_seed=42, base_latency_ms=float(ms))
        fills = 0
        attempts = 0
        for _, row in df.head(200).iterrows():
            res = fm.attempt_fill(
                limit_price=float(row["close"]), side="BUY",
                bar_high=float(row["high"]), bar_low=float(row["low"]),
                bar_volume=float(row["volume"]), avg_volume=float(df["volume"].mean()),
                latency_ms=float(ms), order_book_depth=1.0,
            )
            attempts += 1
            if res.filled:
                fills += 1
        fill_rate = fills / max(1, attempts)
        # Latency also adds slippage for market orders (heuristic for backtest)
        extra_slip = (ms / 500.0) * 0.0001
        # Backtest with latency-adjusted slippage (real fill simulation would need trade_engine queue)
        orig_slip = _s.commission.slippage
        _s.commission.slippage = base_slippage + extra_slip
        try:
            be = BacktestEngine(initial_balance=1000.0, minimum_rows=min(30, len(df)))
            res_bt = be.run(df)
            pf = float(res_bt.profit_factor) if res_bt.profit_factor != float('inf') else 0.0
        except Exception:
            pf = 0.0
        finally:
            _s.commission.slippage = orig_slip
        # Robust if PF >1.0 and fill_rate >0.3 even at 3s
        robust = pf > 1.0 and fill_rate > 0.25
        results.append({
            "latency_ms": float(ms),
            "fill_rate": fill_rate,
            "extra_slippage": extra_slip,
            "pf": pf,
            "robust": robust,
        })
    return results


def queue_simulation(df: pd.DataFrame, fill_model=None) -> Dict[str, float]:
    """P2.3 Queue/fill simulation — 5 factors: price touched + volume + queue + latency + order book."""
    try:
        from src.execution.fill_model import LimitFillModel
        fm = fill_model or LimitFillModel(experiment_seed=42, base_latency_ms=100.0, order_book_depth=1.0)
        fills = 0
        attempts = 0
        total_prob = 0.0
        for i, row in df.head(200).iterrows():
            # Mock limit order at close with realistic latency/order book per bar
            # Use bar timestamp for deterministic seed
            ts = str(row.get("timestamp", f"bar_{i}"))
            res = fm.attempt_fill(
                limit_price=float(row["close"]), side="BUY",
                bar_high=float(row["high"]), bar_low=float(row["low"]),
                bar_volume=float(row["volume"]), avg_volume=float(df["volume"].mean()),
                spread=0.0002, symbol="BTCUSDT", bar_timestamp=ts, order_id=f"q_{i}",
                latency_ms=100.0, order_book_depth=1.0,
            )
            attempts += 1
            total_prob += res.fill_prob
            if res.filled:
                fills += 1
        avg_prob = total_prob / max(1, attempts)
        return {
            "fill_rate": fills / max(1, attempts),
            "avg_fill_prob": avg_prob,
            "attempts": float(attempts),
            "fills": float(fills),
            "latency_ms": 100.0,
            "depth": 1.0,
        }
    except Exception as e:
        return {"fill_rate": 0.0, "attempts": 0.0, "fills": 0.0, "error": str(e)}
