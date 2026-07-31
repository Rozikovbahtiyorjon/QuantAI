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