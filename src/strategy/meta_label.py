"""
QuantAI Multi-Timeframe Filter + Meta-Labeling (Phases B + A).

MultiTFConfirm (B):
    Wraps any base generator. Direction allowed only when the higher
    timeframe trend agrees. Higher-TF buckets are computed CAUSALLY:
    only COMPLETED tf_bars-sized buckets are used (forming bucket
    excluded), EMA over bucket closes.

FilteredGenerator (A):
    Wraps base (+ optional MTF gate). At signal time builds the entry
    feature vector and consults a fitted MetaLabelModel; trades below
    P(win) threshold are dropped.

meta labeling helpers:
    harvest_entries()  - run engine gross on train slice, record entries
    build_labeled()    - triple-barrier labels per entry (conservative)
    MetaLabelModel     - XGBoost wrapper with PurgedKFold-friendly fit
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.strategy.signal_generator import SignalResult


# =====================================================
# B. MULTI-TIMEFRAME CONFIRMATION
# =====================================================

@dataclass
class MultiTFConfig:
    # Higher timeframe expressed in base bars (1h data -> 4 = 4h).
    tf_bars: int = 4

    # EMA period measured in HTF buckets.
    htf_ema_period: int = 50

    # Require strict alignment close>htf_ema for LONG (mirrored SHORT).
    require_close_side: bool = True


class MultiTFConfirm:
    """
    Causal higher-timeframe trend gate around a base generator.

    The wrapped generator produces the raw decision; this filter
    blocks counter-HTF-trend directions.
    """

    def __init__(
        self,
        base_generator: Any,
        config: MultiTFConfig | None = None,
    ) -> None:
        self.base = base_generator
        self.config = config or MultiTFConfig()

    def reset(self) -> None:
        try:
            self.base.reset()
        except AttributeError:
            pass

    def _htf_trend(self, df: pd.DataFrame) -> int:
        """
        +1 / -1 / 0 for HTF trend direction.

        Uses completed buckets only: the trailing
        len(df) % tf_bars rows of the CURRENT forming bucket are
        excluded from aggregation input... implemented by bucketing on
        integer indices of CLOSED buckets.
        """

        cfg = self.config
        n = len(df)
        k = cfg.tf_bars

        closed_count = n // k          # fully closed buckets
        if closed_count < cfg.htf_ema_period + 2:
            return 0                    # not enough HTF history -> no trade

        closes = df["close"].astype(float).to_numpy()

        # Close price of each closed bucket = close of its last bar.
        bucket_ends = closes[: closed_count * k].reshape(closed_count, k)[:, -1]

        ema = pd.Series(bucket_ends).ewm(
            span=cfg.htf_ema_period, adjust=False
        ).mean().to_numpy()

        last_close = bucket_ends[-1]
        last_ema = ema[-1]

        if last_close > last_ema:
            return 1
        if last_close < last_ema:
            return -1
        return 0

    def generate(self, df: pd.DataFrame) -> SignalResult:
        result = self.base.generate(df)

        if result.signal == "HOLD":
            return result

        trend = self._htf_trend(df)

        if result.signal == "BUY" and trend < 0:
            result.signal = "HOLD"
            result.trade_approved = False
            result.reasons.append("MTF: blocked BUY vs 4h downtrend")
            return result

        if result.signal == "SELL" and trend > 0:
            result.signal = "HOLD"
            result.trade_approved = False
            result.reasons.append("MTF: blocked SELL vs 4h uptrend")
            return result

        return result


# =====================================================
# A. ENTRY FEATURES
# =====================================================

FEATURE_NAMES = [
    "rsi",
    "adx",
    "atr_pct",
    "vol_ratio",
    "close_vs_ema_trend_atr",
    "dist_20bar_high_atr",
    "dist_20bar_low_atr",
    "side_long",
    "ema_stack_bull",
    "ema_stack_bear",
    "hour_sin",
    "hour_cos",
]


def entry_features(df: pd.DataFrame, side: str) -> dict[str, float]:
    """
    Causal feature vector at the SIGNAL bar (last row of df window).
    All values derive from current/past data only.
    """

    row = df.iloc[-1]
    close = float(row["close"])
    atr = float(row.get("atr", 0.0)) or 1e-9
    ema_trend = float(row.get("ema_trend", close))

    tail20 = df.iloc[-21:-1] if len(df) > 21 else df.iloc[:-1]
    hi20 = float(tail20["high"].max()) if len(tail20) else close
    lo20 = float(tail20["low"].min()) if len(tail20) else close

    ef = float(row.get("ema_fast", close))
    es = float(row.get("ema_slow", close))

    ts = pd.Timestamp(row.get("timestamp"))
    hour = ts.hour + ts.minute / 60.0

    feats = {
        "rsi": float(row.get("rsi", 50.0)),
        "adx": float(row.get("adx", 0.0)),
        "atr_pct": atr / close * 100.0,
        "vol_ratio": float(row.get("volume_ratio", 1.0))
        if "volume_ratio" in df.columns
        else float(row.get("volume", 0.0)) / max(float(df["volume"].tail(20).mean()), 1e-9),
        "close_vs_ema_trend_atr": (close - ema_trend) / atr,
        "dist_20bar_high_atr": (hi20 - close) / atr,
        "dist_20bar_low_atr": (close - lo20) / atr,
        "side_long": 1.0 if side == "BUY" else 0.0,
        "ema_stack_bull": 1.0 if ef > es > ema_trend else 0.0,
        "ema_stack_bear": 1.0 if ef < es < ema_trend else 0.0,
        "hour_sin": float(np.sin(2 * np.pi * hour / 24.0)),
        "hour_cos": float(np.cos(2 * np.pi * hour / 24.0)),
    }
    return feats


# =====================================================
# A. LABELS (triple barrier, conservative)
# =====================================================

@dataclass
class BarrierConfig:
    sl_atr_mult: float = 3.0      # matches V2 exit engine
    tp_r_mult: float = 2.0        # take profit at 2R
    horizon_bars: int = 72        # 3 days on 1h


def label_entry(
    df: pd.DataFrame,
    entry_index: int,
    side: str,
    entry_price: float,
    cfg: BarrierConfig,
) -> int:
    """
    1 = TP hit before SL within horizon; else 0.
    Entry occurs at OPEN of entry_index bar; barriers measured
    from subsequent bars (entry bar itself excluded to avoid using
    the same bar's range against fresh position).
    """

    R = float(df.iloc[entry_index]["atr"]) * cfg.sl_atr_mult
    if R <= 0:
        return 0

    long = side == "BUY"

    tp = entry_price + cfg.tp_r_mult * R if long else entry_price - cfg.tp_r_mult * R
    sl = entry_price - R if long else entry_price + R

    end = min(len(df), entry_index + 1 + cfg.horizon_bars)

    for j in range(entry_index + 1, end):
        high = float(df.iloc[j]["high"])
        low = float(df.iloc[j]["low"])

        if long:
            hit_sl = low <= sl
            hit_tp = high >= tp
        else:
            hit_sl = high >= sl
            hit_tp = low <= tp

        # conservative: SL wins ties
        if hit_sl:
            return 0
        if hit_tp:
            return 1

    return 0


def build_labeled_dataset(
    df: pd.DataFrame,
    entries: list[dict],
    feature_fn: Callable[[pd.DataFrame, str], dict],
    history_window: int,
    barrier: BarrierConfig | None = None,
) -> pd.DataFrame:
    """
    entries: records produced by TradeEngine entry_callback with keys
             signal_index / side / executed...
    Returns DataFrame[FEATURE_NAMES + ['label']].
    """

    barrier = barrier or BarrierConfig()
    rows = []

    for e in entries:
        i = int(e["signal_index"])
        lo = max(0, i - history_window + 1)

        if i - lo + 1 < 120:           # minimal analyzer window
            continue

        window = df.iloc[lo : i + 1]

        # enough forward bars OR partial horizon (still labelable:
        # unresolved -> conservative 0 handled by loop bounds)
        feats = feature_fn(window, e["side"])
        y = label_entry(df, i, e["side"], float(e["entry_price"]), barrier)

        rows.append({**feats, "label": y})

    return pd.DataFrame(rows)


# =====================================================
# A. META MODEL
# =====================================================

class MetaLabelModel:
    """XGBoost probability-of-win filter."""

    def __init__(
        self,
        threshold: float = 0.55,
        params: dict | None = None,
    ) -> None:
        self.threshold = threshold
        self.params = params or {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "eval_metric": "logloss",
            "verbosity": 0,
        }
        self.model: Any = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        from xgboost import XGBClassifier

        pos = max(int(y.sum()), 1)
        neg = max(int(len(y) - pos), 1)

        self.model = XGBClassifier(
            **self.params,
            scale_pos_weight=neg / pos,
            random_state=42,
        )
        self.model.fit(X[FEATURE_NAMES], y)

    def approve(self, features: dict) -> bool:
        import pandas as pd_

        proba = float(
            self.model.predict_proba(
                pd_.DataFrame([features])[FEATURE_NAMES]
            )[0][1]
        )
        return proba >= self.threshold


# =====================================================
# A+B. FILTERED WRAPPER
# =====================================================

class FilteredGenerator:
    """
    base -> [MTF gate inside base chain] -> meta approval.

    Used on TEST slices during walk-forward after fitting
    MetaLabelModel on the TRAIN slice candidates.
    """

    def __init__(
        self,
        base_generator: Any,
        model: MetaLabelModel | None,
        history_window: int,
    ) -> None:
        self.base = base_generator
        self.model = model
        self.history_window = history_window

    def reset(self) -> None:
        try:
            self.base.reset()
        except AttributeError:
            pass

    def generate(self, df: pd.DataFrame) -> SignalResult:
        result = self.base.generate(df)

        if result.signal == "HOLD":
            return result

        if self.model is None:
            return result

        feats = entry_features(df, result.signal)

        if not self.model.approve(feats):
            result.reasons.append(f"MetaFilter: P(win) < {self.model.threshold}")
            result.signal = "HOLD"
            result.trade_approved = False

        return result


__all__ = [
    "MultiTFConfig",
    "MultiTFConfirm",
    "FEATURE_NAMES",
    "entry_features",
    "BarrierConfig",
    "label_entry",
    "build_labeled_dataset",
    "MetaLabelModel",
    "FilteredGenerator",
]
