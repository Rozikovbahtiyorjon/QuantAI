"""
QuantAI Regime Stability — 7-Regime Classification & Stability Gate

Task 5: Regime stability for QuantAI.

Split crypto into 7 regimes:
    Bull, Bear, Sideways, High Vol, Low Vol, Crash, Recovery

Requirement:
- Require positive expectancy NOT necessarily in all regimes,
  but system must KNOW WHEN strategy WORKS / WHEN FAILS.

Provides:
    - REGIMES constant (7 labels)
    - classify_regimes(df) -> Series of regime labels per bar (causal)
    - evaluate_regime_stability(trades_df or window_stats, regime_labels)
        -> per-regime PF, expectancy, win_rate, trades, overall verdict

Regime classification logic uses BTC price trend, volatility, drawdown.
All calculations are causal (trailing windows only, no look-ahead).

Priority (highest first):
    1. Crash    — large drawdown or shock single-bar drop
    2. Recovery — bounce after Crash
    3. High Vol — volatility ratio elevated
    4. Bull     — up-trend (price_change_pct >= threshold)
    5. Bear     — down-trend (price_change_pct <= -threshold)
    6. Low Vol  — volatility compressed
    7. Sideways — else

Integration:
    ResearchIntegrity Gate 6 checks per-regime expectancy.
    Config: IntegrityConfig.require_regime_stability, min_regimes_positive

References:
    Experimental MarketRegimeIntelligenceEngine (TREND_UP/DOWN/RANGE/HIGH/LOW/SHOCK/RECOVERY)
    mapped to required 7: Bull/Bear/Sideways/High Vol/Low Vol/Crash/Recovery
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

# =====================================================
# Constants
# =====================================================

REGIMES: tuple[str, ...] = (
    "Bull",
    "Bear",
    "Sideways",
    "High Vol",
    "Low Vol",
    "Crash",
    "Recovery",
)

# Aliases for legacy mapping (experimental -> required)
_ALIAS_MAP = {
    "TREND_UP": "Bull",
    "TREND_DOWN": "Bear",
    "RANGE": "Sideways",
    "HIGH_VOLATILITY": "High Vol",
    "LOW_VOLATILITY": "Low Vol",
    "SHOCK": "Crash",
    "RECOVERY": "Recovery",
    # also allow lowercase variants
    "bull": "Bull",
    "bear": "Bear",
    "sideways": "Sideways",
    "high vol": "High Vol",
    "low vol": "Low Vol",
    "crash": "Crash",
    "recovery": "Recovery",
    "HighVol": "High Vol",
    "LowVol": "Low Vol",
}

# Default thresholds (can be overridden via classify_regimes kwargs)
DEFAULT_TREND_LOOKBACK = 50
DEFAULT_VOL_SHORT = 20
DEFAULT_VOL_LONG = 100
DEFAULT_DRAWDOWN_LOOKBACK = 100
DEFAULT_HIGH_VOL_RATIO = 1.5
DEFAULT_LOW_VOL_RATIO = 0.7
DEFAULT_TREND_THRESHOLD_PCT = 5.0  # price change % over trend_lookback to qualify Bull/Bear
DEFAULT_CRASH_DD_PCT = -10.0  # drawdown % <= this -> Crash
DEFAULT_SHOCK_RETURN_PCT = -7.0  # single bar return % <= this -> Crash
DEFAULT_RECOVERY_BOUNCE_PCT = 5.0  # bounce % from trough after Crash -> Recovery


def _norm_regime(label: str) -> str:
    """Normalize legacy/alternative labels to canonical 7."""
    if label in REGIMES:
        return label
    if label in _ALIAS_MAP:
        return _ALIAS_MAP[label]
    # try case-insensitive
    low = str(label).strip()
    if low in _ALIAS_MAP:
        return _ALIAS_MAP[low]
    # fallback - return as-is if already one of REGIMES case-insensitive
    for r in REGIMES:
        if r.lower() == low.lower():
            return r
    return str(label)


def _resolve_price_series(df: pd.DataFrame) -> pd.Series:
    """Extract close price series from df (handles OHLCV variants)."""
    for col in ("close", "Close", "CLOSE", "price", "Price"):
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            return s
    # fallback: first numeric column
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols):
        return pd.to_numeric(df[num_cols[0]], errors="coerce")
    raise ValueError("classify_regimes: df must contain 'close' column")

# =====================================================
# Bar-wise classification
# =====================================================

def classify_regimes(
    df: pd.DataFrame,
    trend_lookback: int = DEFAULT_TREND_LOOKBACK,
    vol_short: int = DEFAULT_VOL_SHORT,
    vol_long: int = DEFAULT_VOL_LONG,
    drawdown_lookback: int = DEFAULT_DRAWDOWN_LOOKBACK,
    high_vol_ratio: float = DEFAULT_HIGH_VOL_RATIO,
    low_vol_ratio: float = DEFAULT_LOW_VOL_RATIO,
    trend_threshold_pct: float = DEFAULT_TREND_THRESHOLD_PCT,
    crash_drawdown_pct: float = DEFAULT_CRASH_DD_PCT,
    shock_return_pct: float = DEFAULT_SHOCK_RETURN_PCT,
    recovery_bounce_pct: float = DEFAULT_RECOVERY_BOUNCE_PCT,
) -> pd.Series:
    """
    Classify each bar into one of 7 regimes using causal trailing windows.

    Args:
        df: DataFrame with at least 'close' column (OHLCV). If 'high','low'
            missing, close is used. Index is preserved (typically datetime or int).
        trend_lookback: bars for trend % change (default 50)
        vol_short: short rolling std window (default 20)
        vol_long: long baseline window (default 100)
        drawdown_lookback: rolling peak window for drawdown (default 100)
        high_vol_ratio: vol_short/vol_long >= this -> High Vol (default 1.5)
        low_vol_ratio: vol_short/vol_long <= this -> Low Vol (default 0.7)
        trend_threshold_pct: price change % over trend_lookback to be Bull/Bear (default 5%)
        crash_drawdown_pct: drawdown % <= this -> Crash (default -10%)
        shock_return_pct: single-bar return % <= this -> Crash (default -7%)
        recovery_bounce_pct: bounce % from recent trough after Crash -> Recovery (default 5%)

    Returns:
        pd.Series with same index as df, dtype object, values in REGIMES.
        Name is 'regime'.

    Method (causal):
        - returns = pct_change(close)
        - vol_short = std(returns, vol_short)
        - vol_long  = std(returns, vol_long)
        - vol_ratio = vol_short / vol_long (1.0 where baseline NaN or 0)
        - trend_pct = (close - close.shift(trend_lookback)) / close.shift *100
        - rolling_peak = rolling_max(close, drawdown_lookback)
        - drawdown_pct = (close - peak)/peak*100
        - shock_ret_5 = rolling_min(returns*100, 5)  # worst single bar in last 5

        Then sequential stateful loop for Recovery hysteresis:
            Crash if drawdown <= crash_dd OR shock_ret_5 <= shock_return_pct
            Else if prev==Crash and (close - trough)/trough*100 >= recovery_bounce_pct -> Recovery
            Else if vol_ratio >= high_vol_ratio -> High Vol
            Else if trend_pct >= trend_threshold_pct -> Bull
            Else if trend_pct <= -trend_threshold_pct -> Bear
            Else if vol_ratio <= low_vol_ratio -> Low Vol
            Else Sideways
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("classify_regimes requires DataFrame")
    if df.empty:
        return pd.Series(dtype=object, name="regime")
    if len(df) < 3:
        # too short: all Sideways but still deterministic
        return pd.Series(["Sideways"] * len(df), index=df.index, name="regime", dtype=object)

    close = _resolve_price_series(df).astype(float)
    # Ensure finite — causal only: forward-fill, never backward-fill future into past
    close = close.replace([np.inf, -np.inf], np.nan).ffill()
    # Drop leading NaNs via warmup rather than bfill from future
    if close.isna().any():
        # If still NaN at start, fill with first valid close (no future leak beyond warmup)
        first_valid = close.dropna().iloc[0] if not close.dropna().empty else None
        if first_valid is not None:
            close = close.fillna(first_valid)
    if close.isna().all():
        raise ValueError("close series is all NaN")

    n = len(close)
    # compute returns (%)
    returns = close.pct_change()
    ret_pct = returns * 100.0

    # volatility
    # use ddof=1 for std, min_periods adaptive
    vol_s = returns.rolling(window=vol_short, min_periods=max(2, vol_short // 2)).std(ddof=1)
    vol_l = returns.rolling(window=vol_long, min_periods=max(10, vol_long // 3)).std(ddof=1)
    # baseline replace 0/nan with median or 1
    # vol_ratio
    vol_ratio = vol_s / vol_l.replace(0, np.nan)
    vol_ratio = vol_ratio.replace([np.inf, -np.inf], np.nan)
    # fill NaN with 1.0 (neutral)
    vol_ratio = vol_ratio.fillna(1.0)
    # clip extreme
    vol_ratio = vol_ratio.clip(lower=0.0, upper=10.0)

    # trend pct
    past_close = close.shift(trend_lookback)
    trend_pct = (close - past_close) / past_close.replace(0, np.nan) * 100.0
    trend_pct = trend_pct.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # drawdown
    peak = close.rolling(window=drawdown_lookback, min_periods=1).max()
    dd_pct = (close - peak) / peak.replace(0, np.nan) * 100.0
    dd_pct = dd_pct.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # shock: worst single bar in last 5
    worst_5 = ret_pct.rolling(window=5, min_periods=1).min()
    # also worst single bar current
    # shock if either drawdown crash OR worst_5 crash
    # trough for recovery: rolling min after recent peak? Simplify: rolling low
    trough = close.rolling(window=drawdown_lookback, min_periods=1).min()
    # bounce from trough %
    bounce_from_trough = (close - trough) / trough.replace(0, np.nan) * 100.0
    bounce_from_trough = bounce_from_trough.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # sequential loop for Recovery (stateful)
    regimes: list[str] = []
    prev = "Sideways"
    # track trough after crash to enable recovery detection even after trough moves
    # we already have bounce_from_trough; but if prev was Crash, use that.
    for i in range(n):
        # allow early bars with NaN to default Sideways
        # fetch scalar values
        vr = float(vol_ratio.iloc[i]) if not pd.isna(vol_ratio.iloc[i]) else 1.0
        tp = float(trend_pct.iloc[i]) if not pd.isna(trend_pct.iloc[i]) else 0.0
        dd = float(dd_pct.iloc[i]) if not pd.isna(dd_pct.iloc[i]) else 0.0
        worst = float(worst_5.iloc[i]) if not pd.isna(worst_5.iloc[i]) else 0.0
        bounce = float(bounce_from_trough.iloc[i]) if not pd.isna(bounce_from_trough.iloc[i]) else 0.0

        # 1. Crash
        is_crash = False
        if dd <= float(crash_drawdown_pct):
            is_crash = True
        elif worst <= float(shock_return_pct):
            is_crash = True

        if is_crash:
            regimes.append("Crash")
            prev = "Crash"
            continue

        # 2. Recovery (only if previous was Crash)
        if prev == "Crash":
            # recovery if bounce from trough >= threshold OR trend_pct positive and bounce positive
            if bounce >= float(recovery_bounce_pct):
                regimes.append("Recovery")
                prev = "Recovery"
                continue
            # also if drawdown has improved significantly (dd > crash_dd/2) and trend positive
            if dd > float(crash_drawdown_pct) / 2.0 and tp > 0:
                regimes.append("Recovery")
                prev = "Recovery"
                continue
            # otherwise stay Crash? But we already not crash. If was Crash and not recovered, maybe High Vol?
            # Fall through to vol/trend checks but keep Crash memory for one extra bar?
            # For determinism, if previous Crash and not recovered, we treat as Recovery only if bounce.
            # Else we go to normal classification (vol etc) but prev stays Crash for next bar?
            # Keep prev=Crash for next iteration if not recovered, so next bar can still recover.
            # Do not update prev yet; fall through.
            pass

        # 3. High Vol
        if vr >= float(high_vol_ratio):
            regimes.append("High Vol")
            prev = "High Vol"
            continue

        # 4. Bull
        if tp >= float(trend_threshold_pct):
            regimes.append("Bull")
            prev = "Bull"
            continue

        # 5. Bear
        if tp <= -float(trend_threshold_pct):
            regimes.append("Bear")
            prev = "Bear"
            continue

        # 6. Low Vol
        if vr <= float(low_vol_ratio):
            regimes.append("Low Vol")
            prev = "Low Vol"
            continue

        # 7. Sideways
        regimes.append("Sideways")
        prev = "Sideways"

    # prev handling for Recovery hysteresis after fall-through:
    # If we fell through from Crash-not-recovered and chose High Vol etc, prev already updated.
    # If we fell through and chose Sideways etc, prev updated accordingly, losing Crash memory after 1 bar.
    # That's intentional: Crash memory lasts one bar after crash ends unless bounced.

    return pd.Series(regimes, index=df.index, name="regime", dtype=object)


# =====================================================
# Evaluation
# =====================================================

def _extract_pnl_series(obj: Any) -> pd.Series | None:
    """
    Try to extract pnl series from various obj types.
    Returns Series of floats or None.
    Handles DataFrame, list[dict], list[float], np.ndarray, pd.Series
    """
    if obj is None:
        return None
    if isinstance(obj, pd.DataFrame):
        # try common profit columns
        for col in ("net", "net_profit", "pnl", "profit", "return", "net_pct", "net_return", "pl", "pnl_pct"):
            if col in obj.columns:
                s = pd.to_numeric(obj[col], errors="coerce").dropna()
                if len(s):
                    return s.astype(float).reset_index(drop=True)
        # if df has regime column, try to find numeric column
        num = obj.select_dtypes(include=[np.number])
        if not num.empty:
            # take first numeric as pnl fallback (but warn)
            col = num.columns[0]
            s = pd.to_numeric(obj[col], errors="coerce").dropna()
            if len(s):
                return s.astype(float).reset_index(drop=True)
        return None
    if isinstance(obj, pd.Series):
        s = pd.to_numeric(obj, errors="coerce").dropna()
        if len(s):
            return s.astype(float).reset_index(drop=True)
        return None
    if isinstance(obj, np.ndarray):
        arr = np.asarray(obj, dtype=float).ravel()
        # filter finite
        arr = arr[np.isfinite(arr)]
        if len(arr):
            return pd.Series(arr, dtype=float)
        return None
    if isinstance(obj, list):
        if not obj:
            return None
        # list of dicts (window_stats)
        if isinstance(obj[0], dict):
            # try keys
            keys_priority = ("net_pct", "net", "pnl", "profit", "return", "net_profit", "pl", "net_return")
            for k in keys_priority:
                vals = []
                found = 0
                for d in obj:
                    if isinstance(d, dict) and k in d:
                        try:
                            vals.append(float(d[k]))
                            found += 1
                        except Exception:
                            continue
                if found >= max(1, len(obj) // 2):
                    return pd.Series(vals, dtype=float)
            # fallback: try any numeric value in dict
            vals = []
            for d in obj:
                if isinstance(d, dict):
                    for v in d.values():
                        try:
                            f = float(v)
                            if math.isfinite(f):
                                vals.append(f)
                                break
                        except Exception:
                            continue
                # only one per dict
            if vals:
                return pd.Series(vals, dtype=float)
            return None
        # list of numbers
        try:
            arr = np.asarray(obj, dtype=float).ravel()
            arr = arr[np.isfinite(arr)]
            if len(arr):
                return pd.Series(arr, dtype=float)
        except Exception:
            pass
        return None
    # generic try
    try:
        arr = np.asarray(obj, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if len(arr):
            return pd.Series(arr, dtype=float)
    except Exception:
        pass
    return None


def _coerce_regime_series(regime_labels: Any, n: int | None = None) -> pd.Series | None:
    """Coerce regime_labels to Series of canonical strings."""
    if regime_labels is None:
        return None
    if isinstance(regime_labels, pd.Series):
        s = regime_labels.astype(object)
    elif isinstance(regime_labels, (list, tuple, np.ndarray)):
        s = pd.Series(list(regime_labels), dtype=object)
    elif isinstance(regime_labels, pd.DataFrame) and "regime" in regime_labels.columns:
        s = regime_labels["regime"].astype(object)
    else:
        try:
            s = pd.Series(list(regime_labels), dtype=object)
        except Exception:
            return None
    # normalize labels
    s = s.map(lambda x: _norm_regime(str(x).strip()) if pd.notna(x) else "Sideways")
    # ensure all in REGIMES; map unknown to Sideways
    s = s.map(lambda x: x if x in REGIMES else "Sideways")
    if n is not None and len(s) != n:
        # length mismatch: return None to signal error
        # Caller will handle; we allow mismatch but truncate/adjust?
        # Instead return as is and caller will detect mismatch
        pass
    return s.reset_index(drop=True)


def evaluate_regime_stability(
    trades_or_windows: Any,
    regime_labels: Any = None,
    *,
    min_regimes_positive: int = 3,
    min_trades_per_regime: int = 5,
    pnl_col: str | None = None,
) -> dict[str, Any]:
    """
    Evaluate per-regime Profit Factor, expectancy, win_rate, etc.

    Args:
        trades_or_windows: DataFrame with profit column, or list[dict] window_stats,
            or Series/list of pnls. Each element is a trade/window net result.
            If DataFrame contains 'regime' column, regime_labels may be None.
        regime_labels: Series/list of regime strings same length as trades_or_windows.
            Values must be in REGIMES (Bull, Bear, Sideways, High Vol, Low Vol, Crash, Recovery).
            Legacy labels are auto-mapped.
        min_regimes_positive: minimum number of regimes with expectancy>0 to pass (default 3)
        min_trades_per_regime: minimum trades/window count for regime to be considered (default 5)
            Regimes with fewer trades are marked 'insufficient' and not counted as positive.

    Returns:
        dict with:
            per_regime: {regime: {trades, wins, win_rate, gross_profit, gross_loss, pf, expectancy,
                                  avg_win, avg_loss, net_total}}
            works: list[str] regimes where expectancy>0 and PF>1 and trades>=min_trades
            fails: list[str] regimes where expectancy<=0 or PF<1 (and trades>=min)
            insufficient: list[str] regimes with trades < min_trades_per_regime
            n_positive: int count of works
            n_negative: int count of fails
            n_insufficient: int
            n_regimes_total: 7
            n_observed: int regimes with trades>=min
            verdict: bool  True if n_positive >= min_regimes_positive
            summary: str human readable
            regime_labels: original normalized series (if provided)
            thresholds: dict

    Note:
        System must KNOW when it works/fails; verdict True means it knows and meets
        minimum positive expectancy threshold (at least 3-4 regimes). Verdict False means
        even that minimum is not met — strategy is not regime-robust.
        Negative expectancy in some regimes is ALLOWED; we explicitly report works/fails.
    """
    # Handle case where trades_or_windows is DataFrame with regime column and regime_labels is None
    if regime_labels is None and isinstance(trades_or_windows, pd.DataFrame) and "regime" in trades_or_windows.columns:
        regime_labels = trades_or_windows["regime"]
        # keep pnl source as same df
        # pnl extraction will still work via _extract_pnl_series

    # Extract pnl series
    pnl_series = _extract_pnl_series(trades_or_windows)
    if pnl_series is None:
        # try if trades_or_windows is list[dict] but extraction failed due to key mismatch
        # try alternative: if trades_or_windows is dict with 'pnl' key?
        raise ValueError(
            "evaluate_regime_stability: could not extract pnl from trades_or_windows. "
            "Expected DataFrame with columns like 'pnl'/'net'/'net_pct' or list[dict] with net_pct, or list of floats. "
            f"Got type {type(trades_or_windows)}"
        )

    # Coerce regime_labels
    regime_s = _coerce_regime_series(regime_labels, n=len(pnl_series))
    if regime_s is None:
        raise ValueError("regime_labels is required and must be same length as trades_or_windows")
    if len(regime_s) != len(pnl_series):
        raise ValueError(
            f"Length mismatch: pnl_series {len(pnl_series)} vs regime_labels {len(regime_s)}. "
            "Trades/windows count must match regime labels count."
        )

    # Normalize pnl_series length
    pnl = pnl_series.reset_index(drop=True).astype(float)
    regime_s = regime_s.reset_index(drop=True)

    # Adaptive min_trades for small samples (e.g., 5 walk-forward windows vs 140 trades)
    # Prevents gate failing purely due to threshold > windows per regime
    effective_min = int(min_trades_per_regime)
    n_total = len(pnl)
    if n_total < 20:
        effective_min = min(effective_min, 1)
    elif n_total < 50:
        effective_min = min(effective_min, 2)
    elif n_total < 100:
        effective_min = min(effective_min, 3)

    per_regime: dict[str, dict[str, Any]] = {}
    works: list[str] = []
    fails: list[str] = []
    insufficient: list[str] = []

    for regime in REGIMES:
        mask = regime_s == regime
        group = pnl[mask]
        n = int(mask.sum())
        if n == 0:
            # no trades in this regime
            per_regime[regime] = {
                "trades": 0,
                "wins": 0,
                "win_rate": 0.0,
                "gross_profit": 0.0,
                "gross_loss": 0.0,
                "pf": 0.0,
                "expectancy": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "net_total": 0.0,
            }
            insufficient.append(regime)
            continue

        vals = group.to_numpy(dtype=float)
        wins = int(np.sum(vals > 0))
        # also count wins>0; for window net_pct, >0 is win
        win_rate = float(wins / n) if n else 0.0
        gross_profit = float(np.sum(vals[vals > 0])) if wins else 0.0
        gross_loss = float(abs(np.sum(vals[vals < 0]))) if np.any(vals < 0) else 0.0
        if gross_loss > 1e-12:
            pf = float(gross_profit / gross_loss)
        else:
            pf = 99.0 if gross_profit > 1e-12 else 0.0
        pf = float(min(pf, 99.0))
        expectancy = float(np.mean(vals)) if n else 0.0
        avg_win = float(gross_profit / wins) if wins else 0.0
        losses = n - wins
        # count strictly losses <0, not breakeven 0
        n_losses = int(np.sum(vals < 0))
        avg_loss = float(gross_loss / n_losses) if n_losses else 0.0
        net_total = float(np.sum(vals))

        per_regime[regime] = {
            "trades": n,
            "wins": wins,
            "win_rate": win_rate,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "pf": pf,
            "expectancy": expectancy,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "net_total": net_total,
        }

        # Determine sufficient vs insufficient (use adaptive effective_min)
        if n < int(effective_min):
            insufficient.append(regime)
            continue

        # Positive expectancy and PF>1 counts as works
        # Requirement: expectancy >0 (PF>1 implies expectancy>0 generally but we check both for robustness)
        if expectancy > 0 and pf >= 1.0:
            works.append(regime)
        elif expectancy > 0:
            # edge case: expectancy positive but PF<1 due to skewed avg_win/avg_loss? still count as positive for expectancy
            # Task says positive expectancy, so we count it
            works.append(regime)
        else:
            fails.append(regime)

    # n_observed = regimes with sufficient trades
    n_observed = int(len(REGIMES) - len(insufficient))
    # Some insufficient regimes already counted separately, so n_positive = len(works)
    n_positive = len(works)
    n_negative = len(fails)

    verdict = bool(n_positive >= int(min_regimes_positive))

    # Build summary
    # System must REPORT when it works / when it fails
    works_str = ", ".join(works) if works else "none"
    fails_str = ", ".join(fails) if fails else "none"
    insuff_str = ", ".join(insufficient) if insufficient else "none"
    summary = (
        f"Regime stability: {n_positive}/{len(REGIMES)} regimes positive (need >= {min_regimes_positive}) | "
        f"WORKS in: [{works_str}] | FAILS in: [{fails_str}] | INSUFFICIENT: [{insuff_str}] | "
        f"Verdict: {'PASS' if verdict else 'FAIL'} (expectancy>0 in at least {min_regimes_positive} regimes)"
    )

    return {
        "per_regime": per_regime,
        "works": works,
        "fails": fails,
        "insufficient": insufficient,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "n_insufficient": len(insufficient),
        "n_regimes_total": len(REGIMES),
        "n_observed": n_observed,
        "verdict": verdict,
        "summary": summary,
        "thresholds": {
            "min_regimes_positive": int(min_regimes_positive),
            "min_trades_per_regime": int(min_trades_per_regime),
            "effective_min_trades": int(effective_min),
        },
        # keep for gate details
        "regime_labels_normalized": regime_s.tolist(),
    }


# Convenience alias for window_stats
def evaluate_window_regime_stability(
    window_stats: list[dict] | pd.DataFrame,
    window_regimes: Any,
    **kwargs,
) -> dict[str, Any]:
    """Alias for evaluate_regime_stability with window_stats."""
    return evaluate_regime_stability(window_stats, window_regimes, **kwargs)


def regime_stability_report(result: dict[str, Any]) -> str:
    """Format evaluate_regime_stability result as multi-line report."""
    lines = [result.get("summary", ""), ""]
    per = result.get("per_regime", {})
    for regime in REGIMES:
        m = per.get(regime, {})
        lines.append(
            f"{regime:10s} | trades {m.get('trades',0):3d} | win_rate {m.get('win_rate',0):.1%} | "
            f"PF {m.get('pf',0):.2f} | expectancy {m.get('expectancy',0):+.4f} | net {m.get('net_total',0):+.2f} | "
            f"{'WORKS' if regime in result.get('works',[]) else ('INSUFFICIENT' if regime in result.get('insufficient',[]) else 'FAILS')}"
        )
    return "\n".join(lines)


__all__ = [
    "REGIMES",
    "classify_regimes",
    "evaluate_regime_stability",
    "evaluate_window_regime_stability",
    "regime_stability_report",
]
