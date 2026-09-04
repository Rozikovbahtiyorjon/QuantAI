"""
QuantAI Mean Reversion Strategy (Phase 3B candidate)

Family: Bollinger Band squeeze + RSI extremes mean reversion.

Economic hypothesis:
    In range/low-ADX regimes, price reverts from BB extremes.
    RSI confirms oversold/overbought. Low ADX confirms ranging.

Causality guarantees:
    - BB levels computed on PRIOR bars (shift(1)).
    - RSI computed on prior closes.
    - All inputs from prepared indicator columns.

API-compatible with SignalGenerator.generate(df) -> SignalResult.
"""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from src.strategy.signal_generator import SignalResult


@dataclass
class MeanReversionConfig:
    # Bollinger Band params
    bb_period: int = 20
    bb_std: float = 2.0

    # RSI params
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0

    # ADX filter: only trade in range (low ADX)
    max_adx: float = 25.0

    # SL/TP in ATR multiples
    sl_atr_mult: float = 1.5
    tp_atr_mult: float = 3.0

    # Cooldown
    cooldown_bars: int = 8


class MeanReversionSignalGenerator:
    """
    Mean reversion: BB squeeze + RSI extremes.
    Compatible with SignalGenerator.generate(df) -> SignalResult API.
    """

    def __init__(self, config=None):
        self.config = config or MeanReversionConfig()
        self._last_signal_index = None

    def reset(self):
        self._last_signal_index = None

    def generate(self, df):
        cfg = self.config

        if len(df) < max(50, cfg.bb_period + 10):
            return SignalResult()

        row = df.iloc[-1]
        idx = len(df) - 1

        result = SignalResult()
        result.timestamp = row.get("timestamp")
        result.entry = float(row["close"])

        # Cooldown
        if (self._last_signal_index is not None
                and idx - self._last_signal_index < cfg.cooldown_bars):
            return SignalResult()

        close = float(row["close"])
        rsi = float(row.get("rsi", 50.0))
        adx = float(row.get("adx", 0.0))
        atr = float(row.get("atr", 0.0))

        if atr <= 0:
            return SignalResult()

        # BB position (from prepared indicators)
        bb_pos = float(row.get("bb_position", 0.0))  # -0.5..0.5
        bb_width = float(row.get("bb_width", 0.0))
        bb_squeeze = float(row.get("bb_squeeze", 0.0))

        # ADX filter: only trade in range
        if adx > 25.0:
            return SignalResult()

        # Mean reversion logic
        signal = "HOLD"

        # BB position is -0.5..0.5 (lower..upper)
        bb_pos = row.get("bb_position", 0.0)

        # Buy: price at lower BB (near -0.5) + oversold RSI
        if bb_pos <= -0.4 and float(row.get("rsi", 50)) <= cfg.rsi_oversold:
            signal = "BUY"
        # Sell: price at upper BB (near 0.5) + overbought RSI
        elif bb_pos >= 0.4 and float(row.get("rsi", 50)) >= cfg.rsi_overbought:
            signal = "SELL"

        if signal == "HOLD":
            return SignalResult()

        self._last_signal_index = idx

        stop_dist = atr * 1.5
        tp_dist = atr * 3.0

        result = SignalResult()
        result.timestamp = row.get("timestamp")
        result.entry = close
        result.signal = signal
        result.confidence = 65.0
        result.ai_signal = signal
        result.ai_confidence = 65.0
        result.trade_approved = True

        if signal == "BUY":
            result.stop_loss = result.entry - atr * 1.5
            result.take_profit = result.entry + atr * 3.0
        else:
            result.stop_loss = result.entry + atr * 1.5
            result.take_profit = result.entry - atr * 3.0

        result.reasons.append(f"MeanRev: {signal} bb_pos={row.get('bb_position', 0):.2f} rsi={float(row.get('rsi', 50)):.1f}")
        return result


__all__ = ["MeanReversionConfig", "MeanReversionSignalGenerator"]