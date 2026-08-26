"""
QuantAI Market Regime Filter (Stage A)

Deterministic, causal regime classification with hysteresis:

    TREND_UP   - EMA-trend slope up AND ADX confirms trend
    TREND_DOWN - EMA-trend slope down AND ADX confirms trend
    RANGE      - everything else (flat / weak / transitional)

Design constraints (Phase 1):
    - Causal only: uses last row + trailing window of PREPARED data
      (columns ema_trend / adx must already exist).
    - Hysteresis: entering a trend requires adx >= adx_enter;
      leaving it requires adx < adx_exit (< adx_enter). This prevents
      whipsaw at the boundary.
    - No ML, no fitting, zero look-ahead risk.

Integration:
    Optional gate inside SignalGenerator (SignalConfig.use_regime_gate,
    default OFF until validated on long multi-regime data in Phase 3):
        BUY blocked while regime == TREND_DOWN
        SELL blocked while regime == TREND_UP
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


TREND_UP = "TREND_UP"
TREND_DOWN = "TREND_DOWN"
RANGE = "RANGE"


@dataclass
class RegimeConfig:
    """Parameters for the Stage-A regime filter."""

    # Bars back for EMA-trend slope measurement.
    slope_bars: int = 8

    # ADX threshold to ENTER a trend state.
    adx_enter: float = 22.0

    # ADX threshold to EXIT a trend state (hysteresis).
    # Must be < adx_enter.
    adx_exit: float = 18.0

    # Minimum prepared-window length required to classify.
    min_bars: int = 60

    def __post_init__(self) -> None:
        if self.slope_bars < 1:
            raise ValueError("slope_bars must be >= 1")
        if not 0.0 < self.adx_exit < self.adx_enter <= 100.0:
            raise ValueError(
                "adx thresholds must satisfy 0 < adx_exit < adx_enter <= 100"
            )
        if self.min_bars <= self.slope_bars:
            raise ValueError("min_bars must be > slope_bars")


class RegimeFilter:
    """
    Stateful-but-causal regime classifier.

    State carries only the previous label (hysteresis memory),
    never future information.
    """

    def __init__(self, config: RegimeConfig | None = None) -> None:
        self.config = config or RegimeConfig()
        self._state: str = RANGE

    def reset(self) -> None:
        self._state = RANGE

    @property
    def state(self) -> str:
        return self._state

    def classify(self, df: pd.DataFrame) -> str:
        """
        Classify the regime using data up to and including the LAST row.
        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError("RegimeFilter.classify() requires a DataFrame")

        cfg = self.config

        if len(df) < max(cfg.min_bars, cfg.slope_bars + 1):
            self._state = RANGE
            return self._state

        if "ema_trend" not in df.columns or "adx" not in df.columns:
            raise KeyError(
                "RegimeFilter requires prepared columns 'ema_trend' and 'adx'"
            )

        ema = df["ema_trend"].astype(float)
        adx_series = df["adx"].astype(float)

        slope = float(ema.iloc[-1]) - float(ema.iloc[-1 - cfg.slope_bars])
        adx = float(adx_series.iloc[-1])

        prev = self._state

        if prev in (TREND_UP, TREND_DOWN):
            # Hysteresis exit: trend persists until ADX decays.
            if adx < cfg.adx_exit:
                self._state = RANGE
            else:
                # Direction flip requires fresh confirmation via enter rule.
                if slope > 0 and prev == TREND_DOWN:
                    self._state = (
                        TREND_UP if adx >= cfg.adx_enter else prev
                    )
                elif slope < 0 and prev == TREND_UP:
                    self._state = (
                        TREND_DOWN if adx >= cfg.adx_enter else prev
                    )
                else:
                    self._state = prev
        else:
            if adx >= cfg.adx_enter and slope > 0:
                self._state = TREND_UP
            elif adx >= cfg.adx_enter and slope < 0:
                self._state = TREND_DOWN
            else:
                self._state = RANGE

        return self._state

    def allows(self, signal: str) -> bool:
        """
        Counter-trend gate: block trades against an active trend.
        """

        if self._state == TREND_UP:
            return signal != "SELL"

        if self._state == TREND_DOWN:
            return signal != "BUY"

        return True
