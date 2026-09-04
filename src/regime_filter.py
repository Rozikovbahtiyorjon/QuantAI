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
TRANSITION = "TRANSITION"
UNKNOWN = "UNKNOWN"
TREND_WEAKENING = "TREND_WEAKENING"
BREAKOUT_PREPARATION = "BREAKOUT_PREPARATION"


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


@dataclass
class RegimeState:
    regime: str
    strength: float  # 0.0-1.0
    confidence: float  # 0.0-1.0
    age: int  # bars since last regime change
    duration: int  # total bars in current regime
    adx: float
    slope: float


class RegimeFilter:
    """
    Stateful-but-causal regime classifier — ENTRY-06 enhanced.

    Now returns RegimeState with TRANSITION/UNKNOWN and
    regime_strength/confidence/age/duration.
    - TRANSITION: protective regime between trends (do not trade)
    - UNKNOWN: insufficient data (< min_bars or missing columns)
    - strength: normalized ADX + slope magnitude
    - confidence: based on ADX distance from threshold + slope consistency
    - age: bars since last change
    - duration: total bars in regime
    """

    def __init__(self, config: RegimeConfig | None = None) -> None:
        self.config = config or RegimeConfig()
        self._state: str = RANGE
        self._strength: float = 0.0
        self._confidence: float = 0.0
        self._age: int = 0
        self._duration: int = 0
        self._history: list[str] = []

    def reset(self) -> None:
        self._state = RANGE
        self._strength = 0.0
        self._confidence = 0.0
        self._age = 0
        self._duration = 0
        self._history.clear()

    @property
    def state(self) -> str:
        return self._state

    @property
    def strength(self) -> float:
        return self._strength

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def age(self) -> int:
        return self._age

    @property
    def duration(self) -> int:
        return self._duration

    def get_state(self) -> RegimeState:
        return RegimeState(
            regime=self._state,
            strength=self._strength,
            confidence=self._confidence,
            age=self._age,
            duration=self._duration,
            adx=0.0,
            slope=0.0,
        )

    def classify(self, df: pd.DataFrame) -> str:
        """
        Classify the regime using data up to and including the LAST row.
        Enhanced with TRANSITION/UNKNOWN and strength/confidence/age/duration.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("RegimeFilter.classify() requires a DataFrame")

        cfg = self.config

        # UNKNOWN if insufficient data or missing columns
        if len(df) < max(cfg.min_bars, cfg.slope_bars + 1):
            self._state = UNKNOWN
            self._strength = 0.0
            self._confidence = 0.0
            self._age = 0
            return self._state

        if "ema_trend" not in df.columns or "adx" not in df.columns:
            # Missing critical columns -> UNKNOWN (not RANGE)
            self._state = UNKNOWN
            self._strength = 0.0
            self._confidence = 0.0
            return self._state

        ema = df["ema_trend"].astype(float)
        adx_series = df["adx"].astype(float)

        slope = float(ema.iloc[-1]) - float(ema.iloc[-1 - cfg.slope_bars])
        adx = float(adx_series.iloc[-1])
        prev = self._state

        # Compute strength and confidence
        # Strength: normalized ADX (0-1) + slope magnitude
        adx_norm = min(1.0, max(0.0, (adx - cfg.adx_exit) / (cfg.adx_enter - cfg.adx_exit + 1e-9))) if adx >= cfg.adx_exit else 0.0
        slope_norm = min(1.0, abs(slope) / (ema.iloc[-1] * 0.02 + 1e-9))  # 2% move = strong
        strength = 0.6 * adx_norm + 0.4 * slope_norm
        # Confidence: distance from threshold + consistency
        # High confidence when adx far from threshold and slope consistent
        if prev in (TREND_UP, TREND_DOWN):
            conf = adx_norm * 0.7 + (1.0 if (slope > 0 and prev == TREND_UP) or (slope < 0 and prev == TREND_DOWN) else 0.3) * 0.3
        else:
            conf = (1 - adx_norm) * 0.5 + 0.3  # RANGE confidence when adx low

        # Track age/duration
        is_same = False
        new_state = prev

        # === ENTRY-07: Transition Detection ===
        # TREND_UP → TREND_WEAKENING → TRANSITION → RANGE
        # RANGE → BREAKOUT_PREPARATION → TRANSITION → TREND_UP
        # TRANSITION is protective (do not trade)

        # Detect weakening: ADX decaying but still above exit
        is_weakening = False
        if prev in (TREND_UP, TREND_DOWN) and cfg.adx_exit <= adx < cfg.adx_enter:
            # ADX in hysteresis zone -> weakening
            is_weakening = True

        # Detect breakout preparation: RANGE with rising ADX and tightening BB
        is_breakout_prep = False
        if prev == RANGE and adx >= cfg.adx_enter * 0.8 and adx < cfg.adx_enter:
            # Check BB width if available
            try:
                bb_width = float(df["bb_width"].iloc[-1]) if "bb_width" in df.columns else 1.0
                if bb_width < 0.02:  # squeeze
                    is_breakout_prep = True
            except Exception:
                is_breakout_prep = adx >= 18

        if prev in (TREND_UP, TREND_DOWN):
            if adx < cfg.adx_exit:
                # Check if was weakening before -> go to TRANSITION, not directly RANGE
                if is_weakening or len(self._history) >= 2 and self._history[-1] == TREND_WEAKENING:
                    new_state = TRANSITION
                else:
                    # Direct weaken -> transition for one bar as protective
                    new_state = TRANSITION
            elif is_weakening:
                new_state = TREND_WEAKENING
            else:
                # Direction flip requires fresh confirmation
                if slope > 0 and prev == TREND_DOWN:
                    new_state = TREND_UP if adx >= cfg.adx_enter else TREND_WEAKENING if is_weakening else prev
                elif slope < 0 and prev == TREND_UP:
                    new_state = TREND_DOWN if adx >= cfg.adx_enter else TREND_WEAKENING if is_weakening else prev
                else:
                    new_state = prev
        elif prev == TREND_WEAKENING:
            if adx < cfg.adx_exit:
                new_state = TRANSITION
            elif adx >= cfg.adx_enter and slope > 0:
                new_state = TREND_UP
            elif adx >= cfg.adx_enter and slope < 0:
                new_state = TREND_DOWN
            elif is_breakout_prep:
                new_state = BREAKOUT_PREPARATION
            else:
                # Stay weakening for one more bar then transition
                new_state = TRANSITION
        elif prev == TRANSITION:
            # From transition, go to RANGE or new trend or breakout prep
            if adx >= cfg.adx_enter and slope > 0:
                new_state = TREND_UP
            elif adx >= cfg.adx_enter and slope < 0:
                new_state = TREND_DOWN
            elif is_breakout_prep:
                new_state = BREAKOUT_PREPARATION
            else:
                new_state = RANGE
        elif prev == BREAKOUT_PREPARATION:
            if adx >= cfg.adx_enter and slope != 0:
                new_state = TREND_UP if slope > 0 else TREND_DOWN
            elif adx < cfg.adx_exit:
                new_state = RANGE
            else:
                new_state = TRANSITION
        else:  # RANGE, UNKNOWN, etc.
            if adx >= cfg.adx_enter and slope > 0:
                new_state = TREND_UP
            elif adx >= cfg.adx_enter and slope < 0:
                new_state = TREND_DOWN
            elif is_breakout_prep:
                new_state = BREAKOUT_PREPARATION
            elif is_weakening:
                new_state = TREND_WEAKENING
            else:
                new_state = RANGE

        # Update age/duration
        if new_state == prev:
            self._age += 1
            self._duration += 1
            is_same = True
        else:
            self._age = 0
            self._duration += 1
            # Keep duration as total in regime, reset age
            # Actually duration should be bars in current regime, so reset
            self._duration = 1

        self._state = new_state
        self._strength = float(max(0.0, min(1.0, strength)))
        self._confidence = float(max(0.0, min(1.0, conf)))
        self._history.append(new_state)
        if len(self._history) > 100:
            self._history = self._history[-100:]

        return self._state

    def classify_with_context(self, df: pd.DataFrame) -> RegimeState:
        """Classify and return full RegimeState with metadata."""
        regime = self.classify(df)
        # Get last adx/slope for context
        try:
            ema = df["ema_trend"].astype(float)
            adx = float(df["adx"].iloc[-1])
            slope = float(ema.iloc[-1] - ema.iloc[-1 - self.config.slope_bars])
        except Exception:
            adx = 0.0
            slope = 0.0
        return RegimeState(
            regime=regime,
            strength=self._strength,
            confidence=self._confidence,
            age=self._age,
            duration=self._duration,
            adx=adx,
            slope=slope,
        )

    def allows(self, signal: str) -> bool:
        """
        Counter-trend gate: block trades against an active trend.
        """

        if self._state == TREND_UP:
            return signal != "SELL"

        if self._state == TREND_DOWN:
            return signal != "BUY"

        return True
