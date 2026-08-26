"""
====================================================
QuantAI Professional v5.0
Feature Engine - Core 4 Indicators + Microstructure

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

from src.microstructure_intelligence import (
    VPINCalculator,
    KyleLambdaEstimator,
    LiquidationLevelAnalyzer,
    compute_microstructure_features,
)
from src.alternative_data import (
    AlternativeDataManager,
    LunarCrushClient,
    FundingRateTracker,
    OIDeltaTracker,
)


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

    Использует 4 базовых индикатора + Microstructure Intelligence + Alternative Data:
    1. EMA (fast, slow, trend)
    2. RSI
    3. ATR
    4. Volume Ratio
    5. VPIN (Volume-synchronized PIN) - Toxicity detection
    6. Kyle's Lambda - Market impact estimation
    7. Liquidation Levels - Support/Resistance from liquidation clusters
    8. LunarCrush - Galaxy Score, AltRank, Social Metrics
    9. Funding Rate - Cross-exchange funding rates
    10. OI Delta - Open Interest delta per candle
    """

    def __init__(self):
        self.features = FeatureVector()

        # Microstructure Intelligence components
        self._vpin_calculator = VPINCalculator(bucket_volume=100.0, window_buckets=50)
        self._kyle_lambda_estimator = KyleLambdaEstimator(window_buckets=100, min_buckets=20)
        self._liquidation_analyzer = LiquidationLevelAnalyzer(
            price_bin_size=0.001,
            min_cluster_volume=10.0,
            lookback_candles=100,
        )
        
        # Alternative Data components
        self._alt_data_manager = None  # Initialized lazily with exchange connections

    def reset(self):
        self.features = FeatureVector()
        self._vpin_calculator.reset()
        self._kyle_lambda_estimator.reset()
        self._liquidation_analyzer.clear()
        if self._alt_data_manager:
            self._alt_data_manager.close()

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
        volume_sma = float(row["volume_sma"])

        if volume_sma > 0:

            ratio = volume / volume_sma

        else:

            ratio = 1.0

        self.features.add(
            "relative_volume",
            ratio,
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
    # MICROSTRUCTURE FEATURES
    # ====================================================

    def calculate_vpin_features(
        self,
        row: pd.Series,
    ) -> None:
        """Calculate VPIN features from current row."""
        # VPIN requires trade data, not just OHLCV
        # For now, we'll add placeholder features
        # In production, VPINCalculator.update() would be called with trade data
        self.features.add("vpin", 0.0)
        self.features.add("vpin_toxicity", 0.0)

    def calculate_kyle_lambda_features(
        self,
        row: pd.Series,
    ) -> None:
        """Calculate Kyle's Lambda features from current row."""
        # Kyle's Lambda requires order flow data
        # Placeholder for now
        self.features.add("kyle_lambda", 0.0)
        self.features.add("kyle_lambda_rsq", 0.0)
        self.features.add("kyle_lambda_confidence", 0.0)

    def calculate_liquidation_features(
        self,
        row: pd.Series,
    ) -> None:
        """Calculate liquidation level features from current row."""
        # Liquidation features require liquidation data
        # Placeholder for now
        self.features.add("nearest_support_dist", 100.0)
        self.features.add("nearest_resistance_dist", 100.0)
        self.features.add("support_strength", 0.0)
        self.features.add("resistance_strength", 0.0)

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

        # RSI
        self.calculate_rsi_features(row)

        # Microstructure
        self.calculate_vpin_features(row)
        self.calculate_kyle_lambda_features(row)
        self.calculate_liquidation_features(row)

        # Alternative Data
        self.calculate_alternative_data_features(row)

        return self.features

    def calculate_alternative_data_features(
        self,
        row: pd.Series,
    ) -> None:
        """Calculate alternative data features (LunarCrush, Funding Rate, OI Delta)."""
        # Placeholder for alternative data features
        # In production, these would be populated from AlternativeDataManager
        self.features.add("lunar_galaxy_score", 50.0)
        self.features.add("lunar_alt_rank", 500)
        self.features.add("lunar_social_volume", 0.0)
        self.features.add("lunar_social_engagement", 0.0)
        self.features.add("lunar_social_dominance", 0.0)
        self.features.add("lunar_sentiment", 0.5)
        self.features.add("lunar_price_score", 0.5)
        self.features.add("funding_rate", 0.0)
        self.features.add("funding_rate_8h", 0.0)
        self.features.add("funding_rate_24h", 0.0)
        self.features.add("oi_delta", 0.0)
        self.features.add("oi_delta_pct", 0.0)


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


# ====================================================
# EXPORTS
# ====================================================

__all__ = [
    "FeatureVector",
    "FeatureEngine",
    "build_features",
]