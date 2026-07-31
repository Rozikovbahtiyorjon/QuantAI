"""
====================================================
QuantAI Professional v5.0
Feature Engine
====================================================

Назначение:

Генерация признаков (Features)
для AI и Machine Learning.

Feature Engine НЕ принимает торговых решений.

Он только вычисляет максимально качественные
характеристики рынка.

Эти признаки затем используются:

    Dataset Builder

    ML Engine

    Probability Engine

====================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd


# ====================================================
# FEATURE VECTOR
# ====================================================

@dataclass
class FeatureVector:
    """
    Контейнер всех вычисленных признаков.
    """

    values: Dict[str, float] = field(default_factory=dict)

    def add(
        self,
        name: str,
        value: float,
    ) -> None:

        self.values[name] = float(value)

    def get(
        self,
        name: str,
        default: float = 0.0,
    ) -> float:

        return self.values.get(name, default)

    def to_dict(self) -> Dict[str, float]:

        return dict(self.values)


# ====================================================
# FEATURE ENGINE
# ====================================================

class FeatureEngine:
    """
    Главный генератор признаков.

    Один объект FeatureEngine
    создаёт FeatureVector
    для последней свечи DataFrame.
    """

    def __init__(self):

        self.features = FeatureVector()

    def reset(self):

        self.features = FeatureVector()

            # ====================================================
    # EMA FEATURES
    # ====================================================

    def calculate_ema_features(
        self,
        row: pd.Series,
    ) -> None:

        close = float(row["close"])

        ema_fast = float(row["ema_fast"])
        ema_slow = float(row["ema_slow"])
        ema_trend = float(row["ema_trend"])

        # Distance to EMA

        self.features.add(
            "ema_fast_distance",
            (close - ema_fast) / ema_fast,
        )

        self.features.add(
            "ema_slow_distance",
            (close - ema_slow) / ema_slow,
        )

        self.features.add(
            "ema_trend_distance",
            (close - ema_trend) / ema_trend,
        )

        # EMA Spread

        self.features.add(
            "ema_fast_slow_spread",
            (ema_fast - ema_slow) / ema_slow,
        )

        self.features.add(
            "ema_slow_trend_spread",
            (ema_slow - ema_trend) / ema_trend,
        )

    # ====================================================
    # ATR FEATURES
    # ====================================================

    def calculate_atr_features(
        self,
        row: pd.Series,
    ) -> None:

        atr = float(row["atr"])
        close = float(row["close"])

        atr_percent = atr / close

        self.features.add(
            "atr_percent",
            atr_percent,
        )

    # ====================================================
    # VOLUME FEATURES
    # ====================================================

    def calculate_volume_features(
        self,
        row: pd.Series,
    ) -> None:

        volume = float(row["volume"])
        volume_sma = float(row["volume_sma20"])

        if volume_sma > 0:

            ratio = volume / volume_sma

        else:

            ratio = 1.0

        self.features.add(
            "relative_volume",
            ratio,
        )

    # ====================================================
    # VWAP FEATURES
    # ====================================================

    def calculate_vwap_features(
        self,
        row: pd.Series,
    ) -> None:

        close = float(row["close"])
        vwap = float(row["vwap"])

        self.features.add(
            "vwap_distance",
            (close - vwap) / vwap,
        )

            # ====================================================
    # RSI FEATURES
    # ====================================================

    def calculate_rsi_features(
        self,
        row: pd.Series,
    ) -> None:

        rsi = float(row["rsi"])

        self.features.add(
            "rsi_normalized",
            rsi / 100.0,
        )

        self.features.add(
            "rsi_distance_50",
            (rsi - 50.0) / 50.0,
        )

        self.features.add(
            "rsi_overbought",
            1.0 if rsi > 70 else 0.0,
        )

        self.features.add(
            "rsi_oversold",
            1.0 if rsi < 30 else 0.0,
        )

            # ====================================================
    # MACD FEATURES
    # ====================================================

    def calculate_macd_features(
        self,
        row: pd.Series,
    ) -> None:

        macd = float(row["macd"])

        signal = float(row["macd_signal"])

        hist = float(row["macd_hist"])

        self.features.add(
            "macd_spread",
            macd - signal,
        )

        self.features.add(
            "macd_histogram",
            hist,
        )

        self.features.add(
            "macd_positive",
            1.0 if hist > 0 else 0.0,
        )

            # ====================================================
    # ADX FEATURES
    # ====================================================

    def calculate_adx_features(
        self,
        row: pd.Series,
    ) -> None:

        adx = float(row["adx"])

        plus_di = float(row["plus_di"])

        minus_di = float(row["minus_di"])

        self.features.add(
            "adx_normalized",
            adx / 100.0,
        )

        self.features.add(
            "trend_direction",
            plus_di - minus_di,
        )

        self.features.add(
            "trend_positive",
            1.0 if plus_di > minus_di else 0.0,
        )

        self.features.add(
            "strong_trend",
            1.0 if adx > 25 else 0.0,
        )

            # ====================================================
    # BOLLINGER FEATURES
    # ====================================================

    def calculate_bollinger_features(
        self,
        row: pd.Series,
    ) -> None:

        close = float(row["close"])

        upper = float(row["bb_upper"])

        middle = float(row["bb_middle"])

        lower = float(row["bb_lower"])

        width = upper - lower

        if width > 0:

            position = (close - lower) / width

        else:

            position = 0.5

        self.features.add(
            "bb_position",
            position,
        )

        self.features.add(
            "bb_width",
            width / middle,
        )

        self.features.add(
            "bb_above_middle",
            1.0 if close > middle else 0.0,
        )

        self.features.add(
            "bb_touch_upper",
            1.0 if close >= upper else 0.0,
        )

        self.features.add(
            "bb_touch_lower",
            1.0 if close <= lower else 0.0,
        )

    # ====================================================
    # BUILD FEATURE VECTOR
    # ====================================================

    def build(
        self,
        df: pd.DataFrame,
    ) -> FeatureVector:
        """
        Построение полного Feature Vector
        по последней свече DataFrame.
        """

        self.reset()

        if len(df) == 0:
            return self.features

        row = df.iloc[-1]

        # EMA
        self.calculate_ema_features(row)

        # ATR
        self.calculate_atr_features(row)

        # Volume
        self.calculate_volume_features(row)

        # VWAP
        self.calculate_vwap_features(row)

        # RSI
        self.calculate_rsi_features(row)

        # MACD
        self.calculate_macd_features(row)

        # ADX
        self.calculate_adx_features(row)

        # Bollinger
        self.calculate_bollinger_features(row)

        return self.features

# ====================================================
# PUBLIC API
# ====================================================

def build_features(
    df: pd.DataFrame,
) -> dict:
    """
    Быстрое построение Feature Vector.

    Возвращает обычный словарь.
    """

    engine = FeatureEngine()

    return engine.build(df).to_dict()

    __all__ = [
    "FeatureVector",
    "FeatureEngine",
    "build_features",
]