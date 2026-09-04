"""
====================================================
QuantAI Triple-Barrier Labeling (Phase 1)

Implements López de Prado "Advances in Financial ML" Ch.3
Triple-Barrier method: label by first barrier touched.

- Upper barrier:  close * (1 + pt)  → BUY
- Lower barrier:  close * (1 - sl)  → SELL
- Vertical barrier: time = t + max_holding (future_bars)

Unlike simple future_return ± threshold, this respects PATH:
  - if SL hit before TP → SELL even if future close > current
  - costs are part of label (net-of-cost)

Returns:
  target ∈ {-1,0,1}, barrier_hit ∈ {"upper","lower","vertical","none"},
  t1 (barrier touch index), return_at_t1
====================================================
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class TripleBarrierConfig:
    """Cost-aware dynamic barriers.

    pt/sl as fraction (0.002 = 0.2%). If use_atr True, barriers = ATR*mult.
    max_holding_bars = vertical barrier (time).
    min_return: net-of-cost threshold — if |return| < min_return → HOLD
                even when barrier hit (filters noise).
    """
    # Fixed fractional barriers (fallback when ATR unavailable)
    pt: float = 0.012  # take-profit 1.2%
    sl: float = 0.008  # stop-loss 0.8%  (RR ~1.5, conservative)
    max_holding_bars: int = 5

    # ATR-adaptive (preferred): barrier = ATR * mult
    use_atr: bool = True
    atr_pt_mult: float = 3.0
    atr_sl_mult: float = 1.5
    atr_col: str = "atr"

    # Net-of-cost filter: round-trip cost = 2*(commission+slippage)
    # Default 0.04%+0.02% = 0.06% per side → 0.12% round-trip
    commission: float = 0.0004
    slippage: float = 0.0002
    min_net_return: float | None = None  # if None → 2*(commission+slippage)

    # Time barrier label when neither horizontal hit
    # If hold_is_neutral True: vertical → HOLD (0), else label by sign of return at t1
    hold_is_neutral: bool = True

    # P1 FIX Audit #13: Ambiguous bar handling (OHLC cannot recover intra-bar path)
    # - "discard": both barriers touched same bar → barrier="ambiguous", target=0 (for ML labeling)
    # - "conservative": both touched → lower-first (risk-first) for backtest realism
    ambiguous_mode: Literal["discard", "conservative"] = "discard"

    def net_cost(self) -> float:
        if self.min_net_return is not None:
            return float(self.min_net_return)
        return 2 * (float(self.commission) + float(self.slippage))


def _barriers_for_row(row: pd.Series, cfg: TripleBarrierConfig) -> tuple[float, float]:
    close = float(row["close"])
    if close <= 0:
        return close, close
    if cfg.use_atr and cfg.atr_col in row and pd.notna(row[cfg.atr_col]) and float(row[cfg.atr_col]) > 0:
        atr = float(row[cfg.atr_col])
        upper = close + atr * float(cfg.atr_pt_mult)
        lower = close - atr * float(cfg.atr_sl_mult)
    else:
        upper = close * (1 + float(cfg.pt))
        lower = close * (1 - float(cfg.sl))
    # Guard against inverted barriers on tiny ATR
    if lower >= close:
        lower = close * (1 - float(cfg.sl))
    if upper <= close:
        upper = close * (1 + float(cfg.pt))
    return upper, lower


def triple_barrier_label(
    df: pd.DataFrame,
    idx: int,
    cfg: TripleBarrierConfig,
) -> dict:
    """Label single position at idx by first barrier hit.

    df must have columns: close, high, low, (atr if use_atr).
    Scans high/low of bars (idx+1 .. idx+max_holding) to detect touch.
    """
    close0 = float(df.iloc[idx]["close"])
    if close0 <= 0 or idx + 1 >= len(df):
        return {"target": 0, "barrier": "none", "t1": idx, "ret": 0.0}

    upper, lower = _barriers_for_row(df.iloc[idx], cfg)
    t1 = min(idx + int(cfg.max_holding_bars), len(df) - 1)

    hit_barrier: Literal["upper", "lower", "vertical", "none"] = "vertical"
    hit_idx = t1

    # Scan path — P1 FIX Audit #13: Ambiguous bars (OHLC cannot recover intra-bar order).
    # If both barriers hit same bar, the true first touch is unknowable from OHLC alone.
    # The larger wick distance is NOT a reliable proxy for path order.
    # Hence:
    #   - discard mode (default for ML): mark as "ambiguous" → HOLD (avoids label noise)
    #   - conservative mode (for backtest): lower-first (risk-first) to avoid optimistic bias
    for j in range(idx + 1, t1 + 1):
        high = float(df.iloc[j]["high"])
        low = float(df.iloc[j]["low"])
        hit_up = high >= upper
        hit_lo = low <= lower
        if hit_up and hit_lo:
            if cfg.ambiguous_mode == "discard":
                hit_barrier = "ambiguous"  # type: ignore
                hit_idx = j
                break
            # conservative: risk-first
            up_dist = high - upper
            lo_dist = lower - low
            if lo_dist >= up_dist:
                hit_barrier = "lower"
            else:
                hit_barrier = "upper"
            hit_idx = j
            break
        if hit_lo:
            hit_barrier = "lower"
            hit_idx = j
            break
        if hit_up:
            hit_barrier = "upper"
            hit_idx = j
            break

    # P1 FIX Audit #14: Barrier return must use barrier price, not candle close.
    # Previously: ret = (close_at_barrier_bar - close0)/close0 could be +0.2% even when TP=+1% was hit.
    # Correct: exit at barrier price; vertical/ambiguous use close.
    if hit_barrier == "upper":
        exit_price = upper
    elif hit_barrier == "lower":
        exit_price = lower
    elif hit_barrier == "ambiguous":  # type: ignore
        # No reliable exit price — treat as vertical (no edge) for labeling
        exit_price = float(df.iloc[hit_idx]["close"])
    else:  # vertical
        exit_price = float(df.iloc[hit_idx]["close"])
    # Also keep candle close for diagnostics
    close1 = float(df.iloc[hit_idx]["close"])
    gross_ret = (exit_price - close0) / close0 if close0 else 0.0
    # Ret is net barrier return; keep close_ret for reference if needed
    ret = gross_ret
    close_ret = (close1 - close0) / close0 if close0 else 0.0

    # Net-of-cost filter on BARRIER return (not close return)
    cost = cfg.net_cost()
    # Ambiguous bars are always HOLD (no reliable edge) regardless of cost
    if hit_barrier == "ambiguous":  # type: ignore
        target = 0
    elif abs(ret) < cost and hit_barrier != "vertical":
        # Hit barrier but net after costs is noise → HOLD
        target = 0
    else:
        if hit_barrier == "upper":
            target = 1
        elif hit_barrier == "lower":
            target = -1
        else:  # vertical
            if cfg.hold_is_neutral:
                target = 0
            else:
                target = 1 if ret >= cfg.pt else (-1 if ret <= -cfg.sl else 0)

    return {
        "target": target,
        "barrier": hit_barrier,
        "t1": hit_idx,
        "ret": float(ret),
        "close_ret": float(close_ret),
        "entry_price": float(close0),
        "exit_price": float(exit_price),
        "gross_ret": float(gross_ret),
        "net_ret": float(ret - cost) if hit_barrier in ("upper", "lower") else float(ret),
        "upper": upper,
        "lower": lower,
        "cost": float(cost),
    }


def label_dataset(
    df: pd.DataFrame,
    indices: list[int],
    cfg: TripleBarrierConfig,
) -> pd.DataFrame:
    """Label many indices at once (for DatasetBuilder). Returns DataFrame with target, barrier, t1, ret."""
    rows = []
    for i in indices:
        r = triple_barrier_label(df, i, cfg)
        rows.append({"index": i, **r})
    return pd.DataFrame(rows)


__all__ = ["TripleBarrierConfig", "triple_barrier_label", "label_dataset"]
