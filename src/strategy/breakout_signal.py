"""
QuantAI Breakout Strategy (Phase 3B candidate)

Family: volatility breakout with trend filter (Donchian-style).

Economic hypothesis (not curve-fit):
    Crypto trends persist after range expansions. Entering N-bar-high
    breakouts aligned with the higher-EMa trend, risking 3xATR and
    trailing the position, captures fat right tails while the exit
    engine caps left tails.

Causality guarantees:
    - Breakout level = rolling max/high of PRIOR bars (shift(1)).
    - All inputs are prepared indicator columns (ema_*, atr, adx)
      computed upstream on full history -> prefix-stable.

API-compatible with SignalGenerator.generate(df) -> SignalResult,
so TradeEngine.run(signal_generator=...) accepts it directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.strategy.signal_generator import SignalResult


@dataclass
class BreakoutConfig:
    # Breakout lookback in bars (prior highs/lows, excluding current bar).
    channel_bars: int = 96          # ~4 days on 1h

    # Trend filter: EMA alignment required for direction.
    ema_fast_col: str = "ema_fast"
    ema_slow_col: str = "ema_slow"
    ema_trend_col: str = "ema_trend"

    # Minimum ADX to accept any trade.
    min_adx: float = 20.0

    # Initial stop distance in ATR multiples (exit engine trails after).
    sl_atr_mult: float = 3.0

    # Cooldown bars between signals (anti-cluster).
    cooldown_bars: int = 12


class BreakoutSignalGenerator:
    """
    Drop-in replacement for SignalGenerator with identical call surface:

        gen = BreakoutSignalGenerator()
        result = gen.generate(df)   # df = prepared OHLCV window
    """

    def __init__(self, config: BreakoutConfig | None = None) -> None:
        self.config = config or BreakoutConfig()
        self._last_signal_index: int | None = None

    def reset(self) -> None:
        self._last_signal_index = None

    # ------------------------------------------------------------

    def generate(self, df: pd.DataFrame) -> SignalResult:
        cfg = self.config

        if len(df) < max(cfg.channel_bars + 1, 120):
            return SignalResult()

        row = df.iloc[-1]
        idx = len(df) - 1

        result = SignalResult()
        result.timestamp = row.get("timestamp")
        result.entry = float(row["close"])

        # ---- cooldown ----
        if (
            self._last_signal_index is not None
            and idx - self._last_signal_index < cfg.cooldown_bars
        ):
            return result

        close = float(row["close"])
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        # ---- Donchian channel of PRIOR bars (causal) ----
        prior = df.iloc[:-1]
        upper = float(prior["high"].tail(cfg.channel_bars).max())
        lower = float(prior["low"].tail(cfg.channel_bars).min())

        # ---- trend / strength filters (prepared columns) ----
        ema_fast = float(row[cfg.ema_fast_col])
        ema_slow = float(row[cfg.ema_slow_col])
        ema_trend = float(row[cfg.ema_trend_col])
        adx = float(row.get("adx", 0.0))

        atr = float(row.get("atr", 0.0))
        if atr <= 0:
            return result

        long_breakout = close > upper
        short_breakout = close < lower

        up_trend = ema_fast > ema_slow > ema_trend
        down_trend = ema_fast < ema_slow < ema_trend

        strong = adx >= cfg.min_adx

        signal = "HOLD"

        if long_breakout and up_trend and strong:
            signal = "BUY"
        elif short_breakout and down_trend and strong:
            signal = "SELL"

        if signal == "HOLD":
            return result

        self._last_signal_index = idx

        stop_dist = atr * cfg.sl_atr_mult

        result.signal = signal
        result.confidence = 70.0
        result.ai_signal = signal
        result.ai_confidence = 70.0
        result.trade_approved = True

        if signal == "BUY":
            result.stop_loss = result.entry - stop_dist
            result.take_profit = result.entry + 2 * stop_dist  # ignored w/o TP policy
        else:
            result.stop_loss = result.entry + stop_dist
            result.take_profit = result.entry - 2 * stop_dist

        result.reasons.append(
            f"Breakout: {signal} close={close:.2f} "
            f"chan=[{lower:.2f};{upper:.2f}] adx={adx:.1f}"
        )

        return result


__all__ = ["BreakoutConfig", "BreakoutSignalGenerator"]
