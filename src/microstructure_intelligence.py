"""
====================================================
QuantAI Professional
Microstructure Intelligence - VPIN & Kyle's Lambda
====================================================

VPIN (Volume-synchronized PIN):
- Measures order flow toxicity
- Volume-synchronized sampling instead of time-synchronized
- Detects informed trading activity

Kyle's Lambda:
- Market impact estimation
- Measures price impact per unit of order flow
- Used for slippage estimation and optimal execution

Liquidation Levels:
- Support/Resistance from liquidation clusters
- Heatmap-based S/R levels

====================================================
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional
from collections.abc import Sequence

import numpy as np
import pandas as pd


# ============================================================
# VPIN (Volume-synchronized Probability of Informed Trading)
# ============================================================

@dataclass(frozen=True)
class VPINResult:
    """Result of VPIN calculation."""
    vpin: float
    bucket_count: int
    buy_volume: float
    sell_volume: float
    total_volume: float
    imbalance: float
    toxicity: str  # "LOW", "MODERATE", "HIGH", "EXTREME"


class VPINCalculator:
    """
    VPIN (Volume-synchronized Probability of Informed Trading) Calculator.
    
    VPIN measures order flow toxicity by computing the probability
    of informed trading using volume-synchronized buckets instead
    of time-synchronized bars.
    
    Key concepts:
    - Volume buckets: Fixed volume per bucket instead of fixed time
    - Order flow imbalance: |buy_volume - sell_volume| / total_volume
    - VPIN = E[|buy_vol - sell_vol| / (buy_vol + sell_vol)]
    
    Toxicity thresholds:
    - LOW: VPIN < 0.15
    - MODERATE: 0.15 <= VPIN < 0.35
    - HIGH: 0.35 <= VPIN < 0.55
    - EXTREME: VPIN >= 0.55
    """

    def __init__(
        self,
        bucket_volume: float = 100.0,  # Volume per bucket (e.g., 100 BTC)
        window_buckets: int = 50,       # Rolling window of buckets
    ):
        """
        Args:
            bucket_volume: Target volume per bucket (e.g., 100 BTC per bucket)
            window_buckets: Number of buckets for rolling VPIN calculation
        """
        if bucket_volume <= 0:
            raise ValueError("bucket_volume must be positive")
        if window_buckets < 2:
            raise ValueError("window_buckets must be >= 2")

        self.bucket_volume = float(bucket_volume)
        self.window_buckets = int(window_buckets)

        # Rolling buckets storage
        self._buy_volumes: deque = deque(maxlen=window_buckets)
        self._sell_volumes: deque = deque(maxlen=window_buckets)
        
        # Current bucket accumulation
        self._current_buy_volume = 0.0
        self._current_sell_volume = 0.0
        self._current_bucket_volume = 0.0

    def update(
        self,
        price: float,
        volume: float,
        side: str,  # "BUY" or "SELL"
    ) -> Optional[VPINResult]:
        """
        Update VPIN with a new trade.
        
        Args:
            price: Trade price
            volume: Trade volume
            side: "BUY" or "SELL"
            
        Returns:
            VPINResult if a bucket is completed, None otherwise
        """
        side = side.upper()
        if side not in ("BUY", "SELL"):
            raise ValueError(f"side must be BUY or SELL, got {side}")

        if volume <= 0:
            raise ValueError("volume must be positive")

        # Add volume to current bucket
        remaining_volume = volume
        
        while remaining_volume > 0:
            space_in_bucket = self.bucket_volume - self._current_bucket_volume
            fill_volume = min(remaining_volume, space_in_bucket)
            
            if side == "BUY":
                self._current_buy_volume += fill_volume
            else:
                self._current_sell_volume += fill_volume
                
            self._current_bucket_volume += fill_volume
            remaining_volume -= fill_volume
            
            # If bucket is full, finalize it
            if self._current_bucket_volume >= self.bucket_volume - 1e-10:
                self._finalize_bucket()

        return self._compute_vpin() if len(self._buy_volumes) >= 2 else None

    def _finalize_bucket(self) -> None:
        """Finalize current bucket and start new one."""
        self._buy_volumes.append(self._current_buy_volume)
        self._sell_volumes.append(self._current_sell_volume)
        
        self._current_buy_volume = 0.0
        self._current_sell_volume = 0.0
        self._current_bucket_volume = 0.0

    def _compute_vpin(self) -> VPINResult:
        """Compute VPIN from completed buckets."""
        if len(self._buy_volumes) < 2:
            return VPINResult(
                vpin=0.0,
                bucket_count=len(self._buy_volumes),
                buy_volume=0.0,
                sell_volume=0.0,
                total_volume=0.0,
                imbalance=0.0,
                toxicity="LOW",
            )

        # Use last N buckets for rolling VPIN
        n = min(len(self._buy_volumes), self.window_buckets)
        buy_vols = list(self._buy_volumes)[-n:]
        sell_vols = list(self._sell_volumes)[-n:]

        buy_volume = sum(buy_vols)
        sell_volume = sum(sell_vols)
        total_volume = buy_volume + sell_volume
        
        if total_volume == 0:
            return VPINResult(
                vpin=0.0,
                bucket_count=len(self._buy_volumes),
                buy_volume=buy_volume,
                sell_volume=sell_volume,
                total_volume=0.0,
                imbalance=0.0,
                toxicity="LOW",
            )

        # VPIN = average absolute imbalance across buckets
        imbalances = []
        for b, s in zip(buy_vols, sell_vols):
            total = b + s
            if total > 0:
                imbalances.append(abs(b - s) / total)
        
        vpin = float(np.mean(imbalances)) if imbalances else 0.0
        imbalance = (buy_volume - sell_volume) / total_volume

        # Toxicity classification
        if vpin < 0.15:
            toxicity = "LOW"
        elif vpin < 0.35:
            toxicity = "MODERATE"
        elif vpin < 0.55:
            toxicity = "HIGH"
        else:
            toxicity = "EXTREME"

        return VPINResult(
            vpin=float(vpin),
            bucket_count=len(self._buy_volumes),
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            total_volume=total_volume,
            imbalance=float(imbalance),
            toxicity=toxicity,
        )

    def get_current_vpin(self) -> float:
        """Get current VPIN value without waiting for bucket completion."""
        if len(self._buy_volumes) < 2:
            return 0.0
        
        n = min(len(self._buy_volumes), self.window_buckets)
        buy_vols = list(self._buy_volumes)[-n:]
        sell_vols = list(self._sell_volumes)[-n:]
        
        imbalances = []
        for b, s in zip(buy_vols, sell_vols):
            total = b + s
            if total > 0:
                imbalances.append(abs(b - s) / total)
        
        return float(np.mean(imbalances)) if imbalances else 0.0

    def reset(self) -> None:
        """Reset all buckets and accumulators."""
        self._buy_volumes.clear()
        self._sell_volumes.clear()
        self._current_buy_volume = 0.0
        self._current_sell_volume = 0.0
        self._current_bucket_volume = 0.0


# ============================================================
# Kyle's Lambda (Market Impact)
# ============================================================

@dataclass(frozen=True)
class KyleLambdaResult:
    """Result of Kyle's Lambda estimation."""
    lambda_estimate: float        # Kyle's lambda (price impact per unit volume)
    r_squared: float             # Goodness of fit
    sample_count: int            # Number of observations used
    price_impact_per_unit: float # Price impact per unit volume
    confidence: str              # "HIGH", "MEDIUM", "LOW"


class KyleLambdaEstimator:
    """
    Kyle's Lambda Estimator for Market Impact.
    
    Kyle's Lambda (λ) measures the price impact per unit of order flow:
    ΔP = λ * Q + ε
    
    Where:
    - ΔP = price change
    - λ = Kyle's lambda (market impact coefficient)
    - Q = signed order flow (buy volume - sell volume)
    - ε = noise
    
    Uses rolling OLS regression on volume-synchronized buckets.
    """

    def __init__(
        self,
        window_buckets: int = 100,
        min_buckets: int = 20,
    ):
        """
        Args:
            window_buckets: Rolling window of buckets for regression
            min_buckets: Minimum buckets required for estimation
        """
        if window_buckets < min_buckets:
            raise ValueError("window_buckets must be >= min_buckets")
        if min_buckets < 10:
            raise ValueError("min_buckets must be >= 10")

        self.window_buckets = window_buckets
        self.min_buckets = min_buckets

        # Rolling data for regression
        self._signed_volumes: deque = deque(maxlen=window_buckets)
        self._price_changes: deque = deque(maxlen=window_buckets)

    def update(
        self,
        price_change: float,
        signed_volume: float,  # buy_volume - sell_volume
    ) -> Optional[KyleLambdaResult]:
        """
        Update with new bucket data.
        
        Args:
            price_change: Price change over the bucket (close - open)
            signed_volume: Buy volume - sell volume (signed order flow)
            
        Returns:
            KyleLambdaResult if enough data, None otherwise
        """
        self._price_changes.append(price_change)
        self._signed_volumes.append(signed_volume)

        if len(self._price_changes) < self.min_buckets:
            return None

        return self._estimate_lambda()

    def _estimate_lambda(self) -> KyleLambdaResult:
        """Estimate Kyle's Lambda using OLS regression."""
        x = np.array(list(self._signed_volumes))
        y = np.array(list(self._price_changes))
        
        # OLS: y = λ * x + ε
        # λ = cov(x,y) / var(x)
        
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        cov_xy = np.mean((x - x_mean) * (y - y_mean))
        var_x = np.mean((x - x_mean) ** 2)
        
        if var_x == 0:
            return KyleLambdaResult(
                lambda_estimate=0.0,
                r_squared=0.0,
                sample_count=len(x),
                price_impact_per_unit=0.0,
                confidence="LOW",
            )
        
        lambda_est = cov_xy / var_x
        
        # R-squared
        y_pred = lambda_est * x
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        
        if ss_tot == 0:
            r_squared = 0.0
        else:
            r_squared = 1.0 - (ss_res / ss_tot)
        
        # Confidence assessment
        n = len(x)
        if n >= 50 and r_squared > 0.3:
            confidence = "HIGH"
        elif n >= 20 and r_squared > 0.1:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        # Price impact per unit volume (absolute lambda)
        price_impact_per_unit = abs(float(lambda_est))
        
        return KyleLambdaResult(
            lambda_estimate=float(lambda_est),
            r_squared=float(r_squared),
            sample_count=n,
            price_impact_per_unit=price_impact_per_unit,
            confidence=confidence,
        )

    def reset(self) -> None:
        """Reset all accumulated data."""
        self._signed_volumes.clear()
        self._price_changes.clear()


# ============================================================
# Liquidation Levels (Support/Resistance from Liquidation Clusters)
# ============================================================

@dataclass(frozen=True)
class LiquidationLevel:
    """Single liquidation cluster level."""
    price: float
    volume: float          # Total liquidation volume at this level
    side: str              # "LONG" (long liquidations) or "SHORT" (short liquidations)
    strength: float        # Normalized strength (0-1)
    distance_pct: float    # Distance from current price (%)


class LiquidationLevelAnalyzer:
    """
    Analyzes liquidation data to identify support/resistance levels
    from liquidation clusters.
    
    Uses liquidation heatmap data to find:
    - Long liquidation clusters (potential support)
    - Short liquidation clusters (potential resistance)
    - Cluster strength based on volume concentration
    """

    def __init__(
        self,
        price_bin_size: float = 0.001,  # 0.1% price bins
        min_cluster_volume: float = 10.0,  # Minimum volume for cluster
        lookback_candles: int = 100,
    ):
        """
        Args:
            price_bin_size: Price bin size as fraction (0.001 = 0.1%)
            min_cluster_volume: Minimum volume to form a cluster
            lookback_candles: Number of candles to look back
        """
        self.price_bin_size = price_bin_size
        self.min_cluster_volume = min_cluster_volume
        self.lookback_candles = lookback_candles

        # Storage for liquidation data
        self._long_liq_bins: dict[float, float] = {}   # price_bin -> volume
        self._short_liq_bins: dict[float, float] = {}  # price_bin -> volume

    def update(
        self,
        current_price: float,
        long_liquidations: Sequence[tuple[float, float]],  # (price, volume)
        short_liquidations: Sequence[tuple[float, float]], # (price, volume)
    ) -> list[LiquidationLevel]:
        """
        Update with new liquidation data.
        
        Args:
            current_price: Current mark price
            long_liquidations: List of (price, volume) for long liquidations
            short_liquidations: List of (price, volume) for short liquidations
            
        Returns:
            List of detected liquidation levels (support/resistance)
        """
        # Bin liquidations by price
        for price, volume in long_liquidations:
            bin_price = self._bin_price(price)
            self._long_liq_bins[bin_price] = self._long_liq_bins.get(bin_price, 0.0) + volume
        
        for price, volume in short_liquidations:
            bin_price = self._bin_price(price)
            self._short_liq_bins[bin_price] = self._short_liq_bins.get(bin_price, 0.0) + volume

        # Find clusters
        levels = self._find_clusters(current_price)
        return levels

    def _bin_price(self, price: float) -> float:
        """Bin price to nearest bin."""
        return round(price / self.price_bin_size) * self.price_bin_size

    def _find_clusters(self, current_price: float) -> list:
        """Find liquidation clusters above/below current price."""
        clusters = []
        
        # Long liquidations below current price = potential support
        for bin_price, volume in self._long_liq_bins.items():
            if bin_price < current_price and volume >= self.min_cluster_volume:
                distance_pct = (current_price - bin_price) / current_price * 100
                strength = min(1.0, volume / 1000.0)  # Normalize
                clusters.append(LiquidationLevel(
                    price=bin_price,
                    volume=volume,
                    side="LONG",
                    strength=strength,
                    distance_pct=distance_pct,
                ))
        
        # Short liquidations above current price = potential resistance
        for bin_price, volume in self._short_liq_bins.items():
            if bin_price > current_price and volume >= self.min_cluster_volume:
                distance_pct = (bin_price - current_price) / current_price * 100
                strength = min(1.0, volume / 1000.0)
                clusters.append(LiquidationLevel(
                    price=bin_price,
                    volume=volume,
                    side="SHORT",
                    strength=strength,
                    distance_pct=distance_pct,
                ))
        
        # Sort by distance from current price
        clusters.sort(key=lambda x: abs(x.distance_pct))
        return clusters[:10]  # Top 10 nearest levels

    def get_support_levels(self, current_price: float, max_levels: int = 5) -> list[LiquidationLevel]:
        """Get nearest support levels (long liquidations below price)."""
        levels = self._find_clusters(current_price)
        supports = [l for l in levels if l.side == "LONG"]
        return sorted(supports, key=lambda x: x.distance_pct)[:max_levels]

    def get_resistance_levels(self, current_price: float, max_levels: int = 5) -> list[LiquidationLevel]:
        """Get nearest resistance levels (short liquidations above price)."""
        levels = self._find_clusters(current_price)
        resistances = [l for l in levels if l.side == "SHORT"]
        return sorted(resistances, key=lambda x: x.distance_pct)[:max_levels]

    def clear(self) -> None:
        """Clear all accumulated data."""
        self._long_liq_bins.clear()
        self._short_liq_bins.clear()


# ============================================================
# Microstructure Features for ML
# ============================================================

def compute_microstructure_features(
    vpin_calculator,
    kyle_lambda_estimator,
    liquidation_analyzer,
    current_price: float,
) -> dict:
    """
    Compute microstructure features for ML model.
    
    Returns dict with:
    - vpin features
    - kyle lambda features
    - liquidation level features
    """
    features = {}
    
    # VPIN features
    if vpin_calculator:
        features["vpin"] = vpin_calculator.get_current_vpin()
        features["vpin_toxicity"] = _toxicity_to_numeric(vpin_calculator._compute_vpin().toxicity if hasattr(vpin_calculator, '_compute_vpin') else "LOW")
    
    # Kyle's Lambda features
    if kyle_lambda_estimator:
        result = kyle_lambda_estimator._estimate_lambda() if hasattr(kyle_lambda_estimator, '_estimate_lambda') else None
        if result:
            features["kyle_lambda"] = result.lambda_estimate
            features["kyle_lambda_rsq"] = result.r_squared
            features["kyle_lambda_confidence"] = _confidence_to_numeric(result.confidence)
    
    # Liquidation features
    if liquidation_analyzer:
        supports = liquidation_analyzer.get_support_levels(100000.0, max_levels=3)  # placeholder price
        resistances = liquidation_analyzer.get_resistance_levels(100000.0, max_levels=3)
        
        features["nearest_support_dist"] = supports[0].distance_pct if supports else 100.0
        features["nearest_resistance_dist"] = resistances[0].distance_pct if resistances else 100.0
        features["support_strength"] = supports[0].strength if supports else 0.0
        features["resistance_strength"] = resistances[0].strength if resistances else 0.0
    
    return features


def _toxicity_to_numeric(toxicity: str) -> float:
    """Convert toxicity string to numeric."""
    mapping = {"LOW": 0.0, "MODERATE": 0.33, "HIGH": 0.66, "EXTREME": 1.0}
    return mapping.get(toxicity, 0.0)


def _confidence_to_numeric(confidence: str) -> float:
    """Convert confidence string to numeric."""
    mapping = {"LOW": 0.0, "MEDIUM": 0.5, "HIGH": 1.0}
    return mapping.get(confidence, 0.0)


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "VPINResult",
    "VPINCalculator",
    "KyleLambdaResult",
    "KyleLambdaEstimator",
    "LiquidationLevel",
    "LiquidationLevelAnalyzer",
    "compute_microstructure_features",
]