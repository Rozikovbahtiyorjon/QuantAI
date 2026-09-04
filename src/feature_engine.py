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

    def __init__(self, live_logger=None):
        self.features = FeatureVector()
        self.live_logger = live_logger

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
    # TREND / ADX / MACD / BB / SUPERTREND (FeatureGate v2)
    # ====================================================

    def calculate_trend_features(self, row: pd.Series) -> None:
        # trend_score already computed in indicators.add_indicators
        if "trend_score" in row and pd.notna(row["trend_score"]):
            self.features.add("trend_score", float(row["trend_score"]) / 6.0)  # normalize -6..6 -> -1..1
        if "adx" in row and pd.notna(row["adx"]):
            self.features.add("adx_norm", float(row["adx"]) / 100.0)
            self.features.add("adx_strong", 1.0 if float(row["adx"]) > 25 else 0.0)
        if "plus_di" in row and "minus_di" in row and pd.notna(row["plus_di"]):
            self.features.add("di_diff", (float(row["plus_di"]) - float(row["minus_di"])) / 100.0)

    def calculate_macd_features(self, row: pd.Series) -> None:
        if "macd" in row and "macd_signal" in row and pd.notna(row["macd"]):
            self.features.add("macd_norm", float(row["macd"]) / (float(row["close"]) * 0.01 + 1e-9))
            self.features.add("macd_hist_norm", float(row["macd_hist"]) / (float(row["close"]) * 0.01 + 1e-9))
            # macd above signal?
            self.features.add("macd_above_signal", 1.0 if float(row["macd"]) > float(row["macd_signal"]) else 0.0)

    def calculate_bollinger_features(self, row: pd.Series) -> None:
        if "bb_upper" in row and pd.notna(row["bb_upper"]):
            close = float(row["close"])
            upper = float(row["bb_upper"])
            lower = float(row["bb_lower"])
            middle = float(row["bb_middle"])
            width = (upper - lower) / (middle + 1e-9)
            pos = (close - lower) / (upper - lower + 1e-9)  # 0=lower, 1=upper
            self.features.add("bb_width", width)
            self.features.add("bb_position", pos - 0.5)  # -0.5..0.5
            # squeeze?
            self.features.add("bb_squeeze", 1.0 if width < 0.02 else 0.0)

    def calculate_supertrend_features(self, row: pd.Series) -> None:
        if "trend" in row and pd.notna(row["trend"]):
            self.features.add("supertrend_dir", float(row["trend"]))  # -1 / 1
        if "supertrend" in row and pd.notna(row["supertrend"]):
            self.features.add("supertrend_dist", (float(row["close"]) - float(row["supertrend"])) / float(row["close"]))

    def calculate_volume_extra(self, row: pd.Series) -> None:
        if "volume_filter" in row:
            self.features.add("volume_anomaly", 1.0 if bool(row["volume_filter"]) else 0.0)
        if "volatility_filter" in row:
            self.features.add("volatility_high", 1.0 if bool(row["volatility_filter"]) else 0.0)

    # ====================================================
    # MICROSTRUCTURE FEATURES
    # ====================================================

    def calculate_vpin_features(
        self,
        row: pd.Series,
    ) -> None:
        """VPIN — MISSING: skip feature entirely until trade-feed wired, do NOT emit 0.

        Previously emitted 0.0 (fake signal) or NaN (dropped all rows).
        Correct: do not add to FeatureVector — ML trains on core 11 features only.
        When live trade feed is wired, this will compute VPIN from real bucketed volume
        via VPINCalculator, not 0 placeholder. Until then, feature is MISSING.
        Live-derived: bucket_volume from real trades, not OHLC volume.
        """
        # Check if real VPIN data is available (requires trade-by-trade feed)
        # Until then, do not add fake 0 — mark as MISSING for diagnostics
        if hasattr(self._vpin_calculator, 'is_ready') and not self._vpin_calculator.is_ready():
            return
        # If calculator has real data, compute
        try:
            vpin = self._vpin_calculator.get_vpin()
            if vpin is not None and 0 < vpin < 1:
                self.features.add("vpin", float(vpin))
        except Exception:
            pass
        return

    def calculate_kyle_lambda_features(
        self,
        row: pd.Series,
    ) -> None:
        """Kyle Lambda — MISSING until L2 order book wired, do NOT emit 0."""
        if hasattr(self._kyle_lambda_estimator, 'is_ready') and not self._kyle_lambda_estimator.is_ready():
            return
        try:
            kyle = self._kyle_lambda_estimator.get_lambda()
            if kyle is not None and kyle != 0:
                self.features.add("kyle_lambda", float(kyle))
        except Exception:
            pass
        return

    def calculate_liquidation_features(
        self,
        row: pd.Series,
    ) -> None:
        """Liquidation — MISSING until real liquidation feed wired, do NOT emit 100/0 placeholder."""
        # Previously: distances = 100, strengths = 0 (fake)
        # Correct: compute from real LiquidationLevelAnalyzer only if has clusters
        try:
            levels = self._liquidation_analyzer.get_levels()
            if not levels:
                return  # MISSING — do not emit 100
            # If has real levels, compute distances
            close = float(row.get("close", 0))
            # Find nearest support/resistance from real clusters
            # (simplified: would need actual cluster logic)
            pass
        except Exception:
            pass
        return

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

        # FeatureGate v2: trend / momentum / volatility
        self.calculate_trend_features(row)
        self.calculate_macd_features(row)
        self.calculate_bollinger_features(row)
        self.calculate_supertrend_features(row)
        self.calculate_volume_extra(row)

        # Microstructure (skipped — MISSING)
        self.calculate_vpin_features(row)
        self.calculate_kyle_lambda_features(row)
        self.calculate_liquidation_features(row)

        # Alternative Data (skipped — MISSING)
        self.calculate_alternative_data_features(row)

        # Auto-log to Feature Store if live logger is attached (non-blocking)
        if self.live_logger is not None:
            try:
                self.live_logger.log(self.features.to_dict())
            except Exception:
                pass  # never block feature generation

        return self.features

    def calculate_alternative_data_features(
        self,
        row: pd.Series,
    ) -> None:
        """Alternative data — MISSING: skip entirely until live feed."""
        return


# ====================================================
# PUBLIC API
# ====================================================

def build_features(
    df: pd.DataFrame,
    live_logger=None,
) -> dict:
    """
    Быстрое построение Feature Vector.

    Возвращает обычный словарь.
    Если передан live_logger, фичи автоматически логируются
    в Feature Store (живой поток).
    """

    engine = FeatureEngine(live_logger=live_logger)

    return engine.build(df).to_dict()


# ====================================================
# EXPORTS
# ====================================================

__all__ = [
    "FeatureVector",
    "FeatureEngine",
    "build_features",
]