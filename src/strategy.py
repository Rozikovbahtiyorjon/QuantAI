"""
QuantAI Professional v5
Strategy Engine

Pipeline:

Market Data
    ↓
Technical Analysis
    ↓
Confidence Engine
    ↓
AI Decision
    ↓
ML Model
    ↓
AI + ML Fusion v2
    ↓
Order Flow Decision Gate
    ↓
Final Signal
    ↓
Risk Manager
    ↓
BUY / SELL / HOLD

AI + ML Fusion v2 rules:

1. AI HOLD -> ML cannot create a trade
2. ML HOLD -> blocks AI BUY/SELL
3. AI + ML agreement -> confirmation
4. AI BUY + ML SELL -> HOLD
5. AI SELL + ML BUY -> HOLD
6. Agreement confidence:
       AI * 0.60 + ML * 0.40
7. Conflict confidence:
       AI * 0.70
8. Minimum trade confidence:
       60%

Order Flow Strategy rules:

1. Strategy HOLD -> Order Flow cannot create a trade
2. Strategy not approved -> Order Flow cannot create a trade
3. BUY + strong ASK pressure -> HOLD
4. SELL + strong BID pressure -> HOLD
5. Neutral BALANCED Order Flow is normalized to:
       pressure = 0.0
       score = 0.5
6. BID/ASK pressure preserves full source precision

This module does NOT execute trades.
It only generates strategy signals and decision diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

import pandas as pd

from src.confidence_engine import ConfidenceEngine
from src.feature_engine import build_features
from src.model_manager import ModelManager
from src.risk_manager import calculate_sl_tp
from src.order_flow_intelligence import OrderFlowSignal


# ============================================================
# CONFIGURATION
# ============================================================

MIN_CONFIDENCE = 60.0

AI_WEIGHT = 0.60
ML_WEIGHT = 0.40

CONFLICT_PENALTY = 0.70

ORDER_FLOW_CONFLICT_THRESHOLD = 0.15


# ============================================================
# SIGNAL RESULT
# ============================================================

@dataclass
class SignalResult:
    """
    Complete result of one Strategy Engine decision.

    Besides the final trading signal, this object contains
    AI/ML/Fusion and Order Flow diagnostics.
    """

    # --------------------------------------------------------
    # FINAL SIGNAL
    # --------------------------------------------------------

    signal: str = "HOLD"

    score: float = 0.0

    confidence: float = 0.0

    entry: float = 0.0

    stop_loss: float = 0.0

    take_profit: float = 0.0

    reasons: List[str] = field(
        default_factory=list
    )

    # --------------------------------------------------------
    # AI DIAGNOSTICS
    # --------------------------------------------------------

    ai_signal: str = "HOLD"

    ai_confidence: float = 0.0

    # --------------------------------------------------------
    # ML DIAGNOSTICS
    # --------------------------------------------------------

    ml_signal: str = "HOLD"

    ml_probability: float = 0.0

    ml_buy_probability: float = 0.0

    ml_sell_probability: float = 0.0

    ml_hold_probability: float = 0.0

    # --------------------------------------------------------
    # FUSION DIAGNOSTICS
    # --------------------------------------------------------

    fusion_signal: str = "HOLD"

    combined_confidence: float = 0.0

    trade_approved: bool = False

    fusion_reason: str = ""

    # --------------------------------------------------------
    # ORDER FLOW DIAGNOSTICS
    # --------------------------------------------------------

    order_flow_signal: str = "HOLD"

    order_flow_enabled: bool = False

    order_flow_approved: bool = False
    
    order_flow_context: str = "UNKNOWN"
    
    order_flow_score: float = 0.5
    
    order_flow_pressure: float = 0.0
    
    order_flow_reason: str = ""

    # --------------------------------------------------------
    # OPTIONAL CONTEXT
    # --------------------------------------------------------

    window_id: int | None = None

    timestamp: Any = None


# ============================================================
# MARKET ENGINE
# ============================================================

@dataclass
class MarketEngine:

    trend_score: float = 0.0

    momentum_score: float = 0.0

    volume_score: float = 0.0

    volatility_score: float = 0.0

    liquidity_score: float = 0.0

    structure_score: float = 0.0

    regime_score: float = 0.0

    confidence_result: object | None = None

    reasons: List[str] = field(
        default_factory=list
    )


# ============================================================
# ML MODEL
# ============================================================

model_manager = ModelManager()

AI_MODEL = model_manager.load()


# ============================================================
# HELPERS
# ============================================================

def last(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Return the last candle from a DataFrame.
    """

    if df.empty:
        raise ValueError(
            "Strategy received empty DataFrame."
        )

    return df.iloc[-1]


def _normalize_signal(
    signal: Any,
) -> str:
    """
    Normalize signal names.
    """

    if signal is None:
        return "HOLD"

    value = str(
        signal
    ).strip().upper()

    aliases = {
        "LONG": "BUY",
        "SHORT": "SELL",
        "NEUTRAL": "HOLD",
        "WAIT": "HOLD",
        "NONE": "HOLD",
    }

    return aliases.get(
        value,
        value,
    )


def _clamp_probability(
    value: Any,
) -> float:
    """
    Convert probability/confidence to [0, 100].

    Examples:

        0.95 -> 95.0
        95.0 -> 95.0
    """

    if value is None:
        return 0.0

    try:
        number = float(value)

    except (
        TypeError,
        ValueError,
    ):
        return 0.0

    if 0.0 <= number <= 1.0:
        number *= 100.0

    return max(
        0.0,
        min(
            100.0,
            number,
        ),
    )


def _normalize_order_flow(
    order_flow_signal: OrderFlowSignal,
) -> tuple[str, float, float]:
    """
    Normalize Order Flow context, pressure and score.

    BALANCED is explicitly treated as neutral because a balanced
    order book can still have a small notional imbalance caused
    only by different bid/ask prices.

    Returns:

        context
        pressure
        score
    """

    if not isinstance(
        order_flow_signal,
        OrderFlowSignal,
    ):
        raise TypeError(
            "order_flow_signal must be an OrderFlowSignal instance."
        )

    context = str(
        order_flow_signal.context
    ).strip().upper()

    if context == "BALANCED":
        return (
            "BALANCED",
            0.0,
            0.5,
        )

    pressure = float(
        order_flow_signal.pressure
    )

    pressure = max(
        -1.0,
        min(
            1.0,
            pressure,
        ),
    )

    score = max(
        0.0,
        min(
            1.0,
            0.5
            + 0.5 * pressure,
        ),
    )

    return (
        context,
        pressure,
        score,
    )


# ============================================================
# TREND ANALYZER
# ============================================================

def evaluate_trend(
    df: pd.DataFrame,
    engine: MarketEngine,
) -> None:

    row = last(df)

    ema_fast = float(
        row["ema_fast"]
    )

    ema_slow = float(
        row["ema_slow"]
    )

    ema_trend = float(
        row["ema_trend"]
    )

    close = float(
        row["close"]
    )

    adx = float(
        row["adx"]
    )

    score = 0.0

    if ema_fast > ema_slow:
        score += 1.5
    else:
        score -= 1.5

    if ema_slow > ema_trend:
        score += 1.0
    else:
        score -= 1.0

    if close > ema_trend:
        score += 1.0
    else:
        score -= 1.0

    if adx >= 40:
        score += 1.5

    elif adx >= 30:
        score += 1.0

    elif adx >= 20:
        score += 0.5

    elif adx < 15:
        score -= 1.0

    engine.trend_score = round(
        score,
        2,
    )

    engine.reasons.append(
        f"Trend Score = "
        f"{engine.trend_score:.2f}"
    )


# ============================================================
# MOMENTUM ANALYZER
# ============================================================

def evaluate_momentum(
    df: pd.DataFrame,
    engine: MarketEngine,
) -> None:

    row = last(df)

    rsi = float(
        row["rsi"]
    )

    macd = float(
        row["macd"]
    )

    macd_signal = float(
        row["macd_signal"]
    )

    macd_hist = float(
        row["macd_hist"]
    )

    score = 0.0

    if rsi >= 70:
        score += 2.0

    elif rsi >= 60:
        score += 1.5

    elif rsi >= 55:
        score += 1.0

    elif rsi <= 30:
        score -= 2.0

    elif rsi <= 40:
        score -= 1.5

    elif rsi <= 45:
        score -= 1.0

    if macd > macd_signal:
        score += 1.0
    else:
        score -= 1.0

    if macd_hist > 0:
        score += 0.5
    else:
        score -= 0.5

    engine.momentum_score = round(
        score,
        2,
    )

    engine.reasons.append(
        f"Momentum Score = "
        f"{engine.momentum_score:.2f}"
    )


# ============================================================
# VOLUME ANALYZER
# ============================================================

def evaluate_volume(
    df: pd.DataFrame,
    engine: MarketEngine,
) -> None:

    row = last(df)

    volume = float(
        row["volume"]
    )

    volume_sma = float(
        row["volume_sma20"]
    )

    score = 0.0

    if volume_sma <= 0:

        engine.volume_score = 0.0

        engine.reasons.append(
            "Volume Score = 0.00"
        )

        return

    relative_volume = (
        volume / volume_sma
    )

    if relative_volume >= 2.0:
        score += 2.0

    elif relative_volume >= 1.5:
        score += 1.5

    elif relative_volume >= 1.2:
        score += 1.0

    elif relative_volume <= 0.6:
        score -= 1.5

    elif relative_volume <= 0.8:
        score -= 1.0

    engine.volume_score = round(
        score,
        2,
    )

    engine.reasons.append(
        f"Volume Score = "
        f"{engine.volume_score:.2f}"
    )


# ============================================================
# VOLATILITY ANALYZER
# ============================================================

def evaluate_volatility(
    df: pd.DataFrame,
    engine: MarketEngine,
) -> None:

    row = last(df)

    atr = float(
        row["atr"]
    )

    close = float(
        row["close"]
    )

    score = 0.0

    if close <= 0:

        engine.volatility_score = 0.0

        engine.reasons.append(
            "Volatility Score = 0.00"
        )

        return

    atr_percent = (
        atr / close * 100.0
    )

    if 0.30 <= atr_percent <= 2.50:
        score += 1.5

    elif 2.50 < atr_percent <= 4.00:
        score += 0.8

    elif atr_percent > 6.00:
        score -= 1.5

    elif atr_percent < 0.15:
        score -= 1.0

    engine.volatility_score = round(
        score,
        2,
    )

    engine.reasons.append(
        f"Volatility Score = "
        f"{engine.volatility_score:.2f}"
    )


# ============================================================
# STRUCTURE ANALYZER
# ============================================================

def evaluate_structure(
    df: pd.DataFrame,
    engine: MarketEngine,
) -> None:

    if len(df) < 21:

        engine.structure_score = 0.0

        engine.reasons.append(
            "Structure Score = 0.00"
        )

        return

    row = last(df)

    score = 0.0

    high20 = (
        df["high"]
        .rolling(20)
        .max()
        .iloc[-2]
    )

    low20 = (
        df["low"]
        .rolling(20)
        .min()
        .iloc[-2]
    )

    close = float(
        row["close"]
    )

    if close > high20:
        score += 2.0

    elif close < low20:
        score -= 2.0

    else:

        price_range = (
            high20 - low20
        )

        if price_range > 0:

            position = (
                close - low20
            ) / price_range

            if position >= 0.80:
                score += 0.8

            elif position <= 0.20:
                score -= 0.8

    engine.structure_score = round(
        score,
        2,
    )

    engine.reasons.append(
        f"Structure Score = "
        f"{engine.structure_score:.2f}"
    )


# ============================================================
# MARKET EVALUATION
# ============================================================

def evaluate_market(
    df: pd.DataFrame,
) -> MarketEngine:

    engine = MarketEngine()

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

    evaluate_structure(
        df,
        engine,
    )

    confidence_engine = ConfidenceEngine()

    confidence_engine.add_component(
        "trend",
        engine.trend_score,
    )

    confidence_engine.add_component(
        "momentum",
        engine.momentum_score,
    )

    confidence_engine.add_component(
        "volume",
        engine.volume_score,
    )

    confidence_engine.add_component(
        "volatility",
        engine.volatility_score,
    )

    confidence_engine.add_component(
        "structure",
        engine.structure_score,
    )

    engine.confidence_result = (
        confidence_engine.evaluate()
    )

    return engine


# ============================================================
# ML PREDICTION
# ============================================================

def predict_ml(
    df: pd.DataFrame,
) -> tuple[str, float, dict[int, float]]:

    if AI_MODEL is None:

        return (
            "HOLD",
            0.0,
            {},
        )

    try:

        features = build_features(
            df
        )

        X = pd.DataFrame(
            [features]
        )

        probabilities = (
            AI_MODEL
            .predict_proba(X)[0]
        )

        classes = (
            AI_MODEL.classes_
        )

        best_index = int(
            probabilities.argmax()
        )

        prediction = classes[
            best_index
        ]

        ml_probability = (
            float(
                probabilities[
                    best_index
                ]
            )
            * 100.0
        )

        if prediction == 0:
            ml_signal = "SELL"

        elif prediction == 1:
            ml_signal = "HOLD"

        elif prediction == 2:
            ml_signal = "BUY"

        else:
            ml_signal = "HOLD"

        ml_probabilities = {
            int(cls): float(prob) * 100.0
            for cls, prob in zip(
                classes,
                probabilities,
            )
        }

        return (
            ml_signal,
            ml_probability,
            ml_probabilities,
        )

    except Exception as exc:

        print()

        print(
            f"ML prediction error: {exc}"
        )

        return (
            "HOLD",
            0.0,
            {},
        )


# ============================================================
# AI + ML FUSION v2
# ============================================================

def fuse_ai_ml(
    ai_signal: str,
    ai_confidence: float,
    ml_signal: str,
    ml_probability: float,
) -> tuple[str, float, bool, str]:

    ai_signal = _normalize_signal(
        ai_signal
    )

    ml_signal = _normalize_signal(
        ml_signal
    )

    ai_confidence = _clamp_probability(
        ai_confidence
    )

    ml_probability = _clamp_probability(
        ml_probability
    )

    if (
        ai_signal == "HOLD"
        and ml_signal == "HOLD"
    ):

        return (
            "HOLD",
            round(
                ai_confidence,
                2,
            ),
            False,
            "AI HOLD + ML HOLD",
        )

    if ai_signal == "HOLD":

        return (
            "HOLD",
            round(
                ai_confidence,
                2,
            ),
            False,
            f"AI HOLD blocks ML {ml_signal}",
        )

    if ml_signal == ai_signal:

        combined = (
            ai_confidence * AI_WEIGHT
            +
            ml_probability * ML_WEIGHT
        )

        combined = round(
            combined,
            2,
        )

        approved = (
            combined >= MIN_CONFIDENCE
        )

        return (
            ai_signal,
            combined,
            approved,
            (
                f"ML confirms {ml_signal} "
                f"({ml_probability:.2f}%)"
            ),
        )

    if ml_signal == "HOLD":

        return (
            "HOLD",
            round(
                ai_confidence,
                2,
            ),
            False,
            (
                f"ML HOLD blocks AI "
                f"{ai_signal} "
                f"({ml_probability:.2f}%)"
            ),
        )

    penalized = (
        ai_confidence
        * CONFLICT_PENALTY
    )

    penalized = round(
        penalized,
        2,
    )

    return (
        "HOLD",
        penalized,
        False,
        (
            f"ML disagreement: "
            f"AI={ai_signal}, "
            f"ML={ml_signal}, "
            f"ML Probability="
            f"{ml_probability:.2f}%"
        ),
    )


# ============================================================
# ORDER FLOW DECISION GATE
# ============================================================

def apply_order_flow_gate(
    result: SignalResult,
    order_flow_signal: OrderFlowSignal | None,
    conflict_threshold: float = ORDER_FLOW_CONFLICT_THRESHOLD,
) -> SignalResult:
    """
    Apply Order Flow as a final Strategy decision gate.

    Strategy remains the primary decision source.

    Order Flow cannot create a trade.

    Order Flow can only:
        - confirm a Strategy trade;
        - leave it unchanged;
        - block a conflicting BUY/SELL.

    If no Order Flow signal is supplied, the original Strategy
    result remains unchanged.
    """

    if not isinstance(
        result,
        SignalResult,
    ):
        raise AttributeError(
            "strategy_result must be a SignalResult instance."
        )

    if order_flow_signal is None:
        result.order_flow_enabled = False
        result.order_flow_approved = True
        result.order_flow_context = "UNKNOWN"
        result.order_flow_score = 0.5
        result.order_flow_pressure = 0.0
        result.order_flow_reason = ""

        return result

    if not isinstance(
        order_flow_signal,
        OrderFlowSignal,
    ):
        raise TypeError(
            "order_flow_signal must be an OrderFlowSignal instance."
        )

    result.order_flow_enabled = True

    order_flow_context = str(
        order_flow_signal.context
    ).strip().upper()

    if order_flow_context == "BALANCED":
        order_flow_pressure = 0.0
        order_flow_score = 0.5
    else:
        order_flow_pressure = float(
            order_flow_signal.pressure
        )

        order_flow_score = max(
            0.0,
            min(
                1.0,
                0.5
                + 0.5 * order_flow_pressure,
            ),
        )

    result.order_flow_context = (
        order_flow_context
    )

    result.order_flow_pressure = (
        order_flow_pressure
    )

    result.order_flow_score = (
        order_flow_score
    )

    strategy_signal = (
        str(
            result.signal
        )
        .strip()
        .upper()
    )

    strategy_approved = bool(
        result.trade_approved
    )

    # ========================================================
    # STRATEGY NOT APPROVED
    # ========================================================

    if not strategy_approved:
        result.signal = "HOLD"
        result.trade_approved = False
        result.order_flow_approved = False

        result.stop_loss = result.entry
        result.take_profit = result.entry

        result.order_flow_reason = (
            "Strategy trade approval is false; "
            "OrderFlow cannot create a trade."
        )

        result.reasons.append(
            f"OrderFlow: "
            f"{result.order_flow_reason}"
        )

        return result

    # ========================================================
    # STRATEGY HOLD
    # ========================================================

    if strategy_signal == "HOLD":
        result.signal = "HOLD"
        result.trade_approved = False
        result.order_flow_approved = False

        result.stop_loss = result.entry
        result.take_profit = result.entry

        result.order_flow_reason = (
            "Strategy HOLD; "
            "OrderFlow cannot create a trade."
        )

        result.reasons.append(
            f"OrderFlow: "
            f"{result.order_flow_reason}"
        )

        return result

    # ========================================================
    # BUY
    # ========================================================

    if strategy_signal == "BUY":

        if (
            order_flow_pressure
            <= -conflict_threshold
        ):
            result.signal = "HOLD"
            result.trade_approved = False
            result.order_flow_approved = False

            result.stop_loss = result.entry
            result.take_profit = result.entry

            result.order_flow_reason = (
                "OrderFlow conflicts with BUY "
                "strategy signal."
            )

            result.reasons.append(
                f"OrderFlow: "
                f"{result.order_flow_reason}"
            )

            return result

        result.order_flow_approved = True

        result.order_flow_reason = (
            "OrderFlow confirms or does not "
            "conflict with BUY strategy signal."
        )

        result.reasons.append(
            f"OrderFlow: "
            f"{result.order_flow_reason}"
        )

        return result

    # ========================================================
    # SELL
    # ========================================================

    if strategy_signal == "SELL":

        if (
            order_flow_pressure
            >= conflict_threshold
        ):
            result.signal = "HOLD"
            result.trade_approved = False
            result.order_flow_approved = False

            result.stop_loss = result.entry
            result.take_profit = result.entry

            result.order_flow_reason = (
                "OrderFlow conflicts with SELL "
                "strategy signal."
            )

            result.reasons.append(
                f"OrderFlow: "
                f"{result.order_flow_reason}"
            )

            return result

        result.order_flow_approved = True

        result.order_flow_reason = (
            "OrderFlow confirms or does not "
            "conflict with SELL strategy signal."
        )

        result.reasons.append(
            f"OrderFlow: "
            f"{result.order_flow_reason}"
        )

        return result

    # ========================================================
    # UNSUPPORTED SIGNAL
    # ========================================================

    result.signal = "HOLD"
    result.trade_approved = False
    result.order_flow_approved = False

    result.stop_loss = result.entry
    result.take_profit = result.entry

    result.order_flow_reason = (
        f"Unsupported strategy signal: "
        f"{strategy_signal}"
    )

    result.reasons.append(
        f"OrderFlow: "
        f"{result.order_flow_reason}"
    )

    return result


# ============================================================
# SIGNAL GENERATION
# ============================================================

def generate_signal_result(
    df: pd.DataFrame,
    order_flow_signal: OrderFlowSignal | None = None,
) -> SignalResult:

    engine = evaluate_market(
        df
    )

    ai = engine.confidence_result

    row = last(df)

    result = SignalResult()

    result.timestamp = row.get(
        "timestamp",
        None,
    )

    result.score = round(
        float(ai.total_score),
        2,
    )

    result.reasons = list(
        engine.reasons
    )

    result.entry = round(
        float(row["close"]),
        2,
    )

    atr = float(
        row["atr"]
    )

    # ========================================================
    # AI SIGNAL
    # ========================================================

    ai_signal = _normalize_signal(
        ai.decision
    )

    ai_confidence = _clamp_probability(
        ai.confidence
    )

    result.ai_signal = ai_signal

    result.ai_confidence = round(
        ai_confidence,
        2,
    )

    # ========================================================
    # ML PREDICTION
    # ========================================================

    (
        ml_signal,
        ml_probability,
        ml_probabilities,
    ) = predict_ml(df)

    ml_signal = _normalize_signal(
        ml_signal
    )

    ml_probability = _clamp_probability(
        ml_probability
    )

    ml_buy_probability = _clamp_probability(
        ml_probabilities.get(
            2,
            0.0,
        )
    )

    ml_sell_probability = _clamp_probability(
        ml_probabilities.get(
            0,
            0.0,
        )
    )

    ml_hold_probability = _clamp_probability(
        ml_probabilities.get(
            1,
            0.0,
        )
    )

    result.ml_signal = ml_signal

    result.ml_probability = round(
        ml_probability,
        2,
    )

    result.ml_buy_probability = round(
        ml_buy_probability,
        2,
    )

    result.ml_sell_probability = round(
        ml_sell_probability,
        2,
    )

    result.ml_hold_probability = round(
        ml_hold_probability,
        2,
    )

    # ========================================================
    # AI + ML FUSION
    # ========================================================

    (
        fusion_signal,
        combined_confidence,
        approved,
        fusion_reason,
    ) = fuse_ai_ml(
        ai_signal=ai_signal,
        ai_confidence=ai_confidence,
        ml_signal=ml_signal,
        ml_probability=ml_probability,
    )

    result.fusion_signal = _normalize_signal(
        fusion_signal
    )

    result.combined_confidence = round(
        _clamp_probability(
            combined_confidence
        ),
        2,
    )

    result.trade_approved = bool(
        approved
    )

    result.fusion_reason = str(
        fusion_reason
    )

    result.confidence = round(
        result.combined_confidence,
        2,
    )

    result.reasons.append(
        fusion_reason
    )

    # ========================================================
    # FINAL APPROVAL FILTER
    # ========================================================

    if not approved:

        result.signal = "HOLD"

        result.stop_loss = result.entry
        result.take_profit = result.entry

        if (
            fusion_signal == "HOLD"
            and "ML disagreement"
            in fusion_reason
        ):

            result.reasons.append(
                "AI/ML disagreement "
                "did not approve trade"
            )

        elif (
            fusion_signal == "HOLD"
            and "ML HOLD"
            in fusion_reason
        ):

            result.reasons.append(
                "ML HOLD did not approve trade"
            )

        elif ai_signal == "HOLD":

            result.reasons.append(
                "AI HOLD cannot create trade"
            )

        else:

            result.reasons.append(
                f"Confidence below "
                f"threshold or fusion "
                f"not approved "
                f"({combined_confidence:.2f}%)"
            )

    # ========================================================
    # BUY
    # ========================================================

    elif fusion_signal == "BUY":

        result.signal = "BUY"

        sl, tp = calculate_sl_tp(
            entry_price=result.entry,
            atr=atr,
        )

        result.stop_loss = round(
            float(sl),
            2,
        )

        result.take_profit = round(
            float(tp),
            2,
        )

        result.reasons.append(
            "BUY approved by AI + ML fusion"
        )

    # ========================================================
    # SELL
    # ========================================================

    elif fusion_signal == "SELL":

        result.signal = "SELL"

        sl, tp = calculate_sl_tp(
            entry_price=result.entry,
            atr=atr,
        )

        risk = abs(
            result.entry
            - float(sl)
        )

        reward = abs(
            float(tp)
            - result.entry
        )

        result.stop_loss = round(
            result.entry + risk,
            2,
        )

        result.take_profit = round(
            result.entry - reward,
            2,
        )

        result.reasons.append(
            "SELL approved by AI + ML fusion"
        )

    # ========================================================
    # FALLBACK
    # ========================================================

    else:

        result.signal = "HOLD"

        result.stop_loss = result.entry
        result.take_profit = result.entry

        result.trade_approved = False

    # ========================================================
    # ORDER FLOW GATE
    # ========================================================

    if order_flow_signal is not None:

        result = apply_order_flow_gate(
            result,
            order_flow_signal,
        )

    return result


# ============================================================
# PRINT SIGNAL
# ============================================================

def print_signal(
    result: SignalResult,
) -> None:

    print()

    print(
        "=" * 60
    )

    print(
        "STRATEGY SIGNAL v2.0"
    )

    print(
        "=" * 60
    )

    print(
        f"Signal         : "
        f"{result.signal}"
    )

    print(
        f"Confidence     : "
        f"{result.confidence:.2f}%"
    )

    print(
        f"Score          : "
        f"{result.score:.2f}"
    )

    print()

    print(
        f"AI Signal      : "
        f"{result.ai_signal}"
    )

    print(
        f"AI Confidence  : "
        f"{result.ai_confidence:.2f}%"
    )

    print(
        f"ML Signal      : "
        f"{result.ml_signal}"
    )

    print(
        f"ML Probability : "
        f"{result.ml_probability:.2f}%"
    )

    print(
        f"Fusion Signal  : "
        f"{result.fusion_signal}"
    )

    print(
        f"Combined Conf. : "
        f"{result.combined_confidence:.2f}%"
    )

    print(
        f"Approved       : "
        f"{result.trade_approved}"
    )

    print()

    print(
        f"Order Flow     : "
        f"{result.order_flow_signal}"
    )

    print(
        f"OF Context     : "
        f"{result.order_flow_context}"
    )

    print(
        f"OF Score       : "
        f"{result.order_flow_score:.6f}"
    )

    print(
        f"OF Pressure    : "
        f"{result.order_flow_pressure:.6f}"
    )

    print(
        f"OF Approved    : "
        f"{result.order_flow_approved}"
    )

    print()

    print(
        f"Entry          : "
        f"{result.entry:.2f}"
    )

    print(
        f"Stop Loss      : "
        f"{result.stop_loss:.2f}"
    )

    print(
        f"Take Profit    : "
        f"{result.take_profit:.2f}"
    )

    print()

    print(
        "Reasons:"
    )

    for reason in result.reasons:

        print(
            f" • {reason}"
        )

    print(
        "=" * 60
    )


# ============================================================
# PUBLIC API
# ============================================================

__all__ = [
    "MIN_CONFIDENCE",
    "AI_WEIGHT",
    "ML_WEIGHT",
    "CONFLICT_PENALTY",
    "ORDER_FLOW_CONFLICT_THRESHOLD",
    "SignalResult",
    "MarketEngine",
    "evaluate_market",
    "predict_ml",
    "fuse_ai_ml",
    "apply_order_flow_gate",
    "generate_signal_result",
    "print_signal",
]