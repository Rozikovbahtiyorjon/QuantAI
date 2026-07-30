"""
"""
====================================================
QuantAI Professional v3.1
Strategy Engine
====================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pandas as pd

from config.settings import (
    EMA_FAST,
    EMA_SLOW,
    EMA_TREND,
    RSI_BUY,
    RSI_SELL,
    ADX_MIN,
)

from src.risk_manager import (
    calculate_sl_tp,
)

from src.confidence_engine import (
    ConfidenceEngine,
)

# ====================================================
# WEIGHTS
# ====================================================

TREND_WEIGHT = 3.0
MOMENTUM_WEIGHT = 2.0
VOLUME_WEIGHT = 1.5
VOLATILITY_WEIGHT = 1.0
BREAKOUT_WEIGHT = 2.5

BUY_THRESHOLD = 5.0
SELL_THRESHOLD = -5.0


# ====================================================
# SIGNAL RESULT
# ====================================================

@dataclass
class SignalResult:

    signal: str = "HOLD"

    confidence: float = 0.0

    score: float = 0.0

    entry: float = 0.0

    stop_loss: float = 0.0

    take_profit: float = 0.0

    reasons: List[str] = field(default_factory=list)


# ====================================================
# MARKET ENGINE
# ====================================================

@dataclass
class MarketEngine:

    score: float = 0.0

    trend_score: float = 0.0

    momentum_score: float = 0.0

    volume_score: float = 0.0

    volatility_score: float = 0.0

    breakout_score: float = 0.0

    confidence_result = None

    reasons: List[str] = field(default_factory=list)

# ====================================================
# TREND
# ====================================================

def evaluate_trend(
    df: pd.DataFrame,
    engine: MarketEngine,
) -> None:

    row = df.iloc[-1]

    score = 0.0

    # EMA alignment

    if (
        row["ema_fast"] >
        row["ema_slow"] >
        row["ema_trend"]
    ):
        score += 2.0

    elif (
        row["ema_fast"] <
        row["ema_slow"] <
        row["ema_trend"]
    ):
        score -= 2.0

    # Trend direction

    if row["trend"] == 1:
        score += 1.0

    elif row["trend"] == -1:
        score -= 1.0

    # ADX filter

    if row["adx"] >= ADX_MIN:

        if score > 0:
            score += 0.5

        elif score < 0:
            score -= 0.5

    engine.trend_score = score * TREND_WEIGHT

    engine.score += engine.trend_score

    engine.reasons.append(
        f"Trend Score = {engine.trend_score:.2f}"
    )

    # ====================================================
# MOMENTUM
# ====================================================

def evaluate_momentum(
    df: pd.DataFrame,
    engine: MarketEngine,
) -> None:

    row = df.iloc[-1]

    score = 0.0

    # RSI

    if row["rsi"] >= RSI_BUY:
        score += 1.0

    elif row["rsi"] <= RSI_SELL:
        score -= 1.0

    # MACD

    if row["macd"] > row["macd_signal"]:
        score += 1.0
    else:
        score -= 1.0

    # Histogram

    if row["macd_hist"] > 0:
        score += 0.5
    else:
        score -= 0.5

    engine.momentum_score = score * MOMENTUM_WEIGHT

    engine.score += engine.momentum_score

    engine.reasons.append(
        f"Momentum Score = {engine.momentum_score:.2f}"
    )


# ====================================================
# VOLUME
# ====================================================

def evaluate_volume(
    df: pd.DataFrame,
    engine: MarketEngine,
) -> None:

    row = df.iloc[-1]

    score = 0.0

    # Volume filter

    if row["volume_filter"]:
        score += 1.0

    # OBV trend

    if row["obv"] > row["obv_ema"]:
        score += 1.0
    else:
        score -= 1.0

    engine.volume_score = score * VOLUME_WEIGHT

    engine.score += engine.volume_score

    engine.reasons.append(
        f"Volume Score = {engine.volume_score:.2f}"
    )

    # ====================================================
# VOLATILITY
# ====================================================

def evaluate_volatility(
    df: pd.DataFrame,
    engine: MarketEngine,
) -> None:

    row = df.iloc[-1]

    score = 0.0

    # Volatility filter

    if row["volatility_filter"]:
        score += 1.0
    else:
        score -= 0.5

    # ATR must exist

    if row["atr"] > 0:
        score += 0.5

    engine.volatility_score = (
        score * VOLATILITY_WEIGHT
    )

    engine.score += engine.volatility_score

    engine.reasons.append(
        f"Volatility Score = {engine.volatility_score:.2f}"
    )


# ====================================================
# BREAKOUT
# ====================================================

def evaluate_breakout(
    df: pd.DataFrame,
    engine: MarketEngine,
) -> None:

    row = df.iloc[-1]

    score = 0.0

    if row["breakout_up"]:
        score += 2.0

    if row["breakout_down"]:
        score -= 2.0

    engine.breakout_score = (
        score * BREAKOUT_WEIGHT
    )

    engine.score += engine.breakout_score

    engine.reasons.append(
        f"Breakout Score = {engine.breakout_score:.2f}"
    )

 # ====================================================
# MARKET EVALUATION
# ====================================================

def evaluate_market(
    df: pd.DataFrame,
) -> MarketEngine:
    """
    Выполняет полный анализ рынка.
    """

    engine = MarketEngine()

    confidence = ConfidenceEngine()

    evaluate_trend(
        df,
        engine,
    )

    evaluate_momentum(
        df,
        engine,
    )

    evaluate_volume(
        df,
        engine,
    )

    evaluate_volatility(
        df,
        engine,
    )

    evaluate_breakout(
        df,
        engine,
    )

    confidence.add_component(
        "trend",
        engine.trend_score,
    )

    confidence.add_component(
        "momentum",
        engine.momentum_score,
    )

    confidence.add_component(
        "volume",
        engine.volume_score,
    )

    confidence.add_component(
        "volatility",
        engine.volatility_score,
    )

    confidence.add_component(
        "structure",
        engine.breakout_score,
    )

    engine.confidence_result = confidence.evaluate()

    return engine


# ====================================================
# SIGNAL GENERATION
# ====================================================

def generate_signal_result(
    df: pd.DataFrame,
) -> SignalResult:

    engine = evaluate_market(df)

    row = df.iloc[-1]

    ai = engine.confidence_result

    result = SignalResult()

    result.score = round(ai.score, 2)

    result.confidence = round(ai.confidence, 2)

    result.reasons = list(engine.reasons)

    result.entry = round(
        float(row["close"]),
        2,
    )

    # ==========================================
    # BUY
    # ==========================================

    if ai.decision == "BUY":

        result.signal = "BUY"

        sl, tp = calculate_sl_tp(

            entry_price=result.entry,

            atr=float(row["atr"]),

        )

        result.stop_loss = sl

        result.take_profit = tp

        return result

    # ==========================================
    # SELL
    # ==========================================

    if ai.decision == "SELL":

        result.signal = "SELL"

        sl, tp = calculate_sl_tp(

            entry_price=result.entry,

            atr=float(row["atr"]),

        )

        risk = result.entry - sl

        result.stop_loss = round(
            result.entry + risk,
            2,
        )

        result.take_profit = round(
            result.entry - (tp - result.entry),
            2,
        )

        return result

    # ==========================================
    # HOLD
    # ==========================================

    result.signal = "HOLD"

    result.stop_loss = result.entry

    result.take_profit = result.entry

    return result


# ====================================================
# PRINT SIGNAL
# ====================================================

def print_signal(
    result: SignalResult,
) -> None:

    print()

    print("=" * 60)
    print("STRATEGY SIGNAL")
    print("=" * 60)

    print(f"Signal         : {result.signal}")
    print(f"Confidence     : {result.confidence:.2f}%")
    print(f"Score          : {result.score:.2f}")

    print()

    print(f"Entry          : {result.entry:.2f}")
    print(f"Stop Loss      : {result.stop_loss:.2f}")
    print(f"Take Profit    : {result.take_profit:.2f}")

    print()

    print("Reasons:")

    for reason in result.reasons:
        print(f" • {reason}")

    print("=" * 60)


# ====================================================
# MODULE EXPORT
# ====================================================

__all__ = [
    "SignalResult",
    "MarketEngine",
    "evaluate_market",
    "generate_signal_result",
    "print_signal",
]