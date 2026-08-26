"""
QuantAI AI Analyzer

Technical analysis components:
- Trend: EMA alignment, price vs trend EMA
- Momentum: RSI with adaptive thresholds
- Volume: Relative volume vs SMA
- Volatility: ATR percentile regime detection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from config.settings import settings


@dataclass
class MarketComponents:
    """Container for all technical analysis components."""
    
    # Trend
    trend_score: float = 0.0
    ema_fast: float = 0.0
    ema_slow: float = 0.0
    ema_trend: float = 0.0
    close: float = 0.0
    
    # Momentum
    rsi: float = 50.0
    rsi_zscore: float = 0.0
    
    # Volume
    volume_ratio: float = 1.0
    volume_zscore: float = 0.0
    
    # Volatility
    atr_percent: float = 0.0
    atr_percentile: float = 50.0  # Rolling percentile
    
    # Derived
    trend_alignment: Literal["bullish", "bearish", "neutral"] = "neutral"
    momentum_regime: Literal["overbought", "oversold", "neutral"] = "neutral"
    volume_regime: Literal["high", "low", "normal"] = "normal"
    volatility_regime: Literal["high", "low", "normal"] = "normal"
    
    # Raw component scores (for ConfidenceEngine)
    component_scores: dict[str, float] = field(default_factory=dict)
    
    # Diagnostics
    reasons: list[str] = field(default_factory=list)


class AIAnalyzer:
    """
    Analyzes market data into technical components.
    
    All thresholds configurable via settings.
    Uses adaptive/z-score logic where possible.
    """
    
    def __init__(self):
        self.lookback = 100  # For percentile calculations
        
    def analyze(self, df: pd.DataFrame) -> MarketComponents:
        """
        Analyze last candle of DataFrame with indicators.
        
        Requires: ema_fast, ema_slow, ema_trend, rsi, atr, volume_ratio
        """
        if len(df) < 2:
            raise ValueError("Need at least 2 candles for analysis")
            
        row = df.iloc[-1]
        hist = df.tail(self.lookback)
        
        comp = MarketComponents()
        
        # Extract values
        comp.ema_fast = float(row["ema_fast"])
        comp.ema_slow = float(row["ema_slow"])
        comp.ema_trend = float(row["ema_trend"])
        comp.close = float(row["close"])
        comp.rsi = float(row["rsi"])
        comp.atr_percent = float(row["atr"]) / comp.close * 100
        comp.volume_ratio = float(row.get("volume_ratio", 1.0))
        
        # --- Trend Analysis ---
        self._analyze_trend(comp)
        
        # --- Momentum Analysis ---
        self._analyze_momentum(comp, hist)
        
        # --- Volume Analysis ---
        self._analyze_volume(comp, hist)
        
        # --- Volatility Analysis ---
        self._analyze_volatility(comp, hist)
        
        # --- Component Scores for ConfidenceEngine ---
        self._calculate_component_scores(comp)
        
        return comp
    
    def _analyze_trend(self, comp: MarketComponents) -> None:
        """EMA alignment and trend classification."""
        score = 0.0
        
        # Fast vs Slow
        if comp.ema_fast > comp.ema_slow:
            score += 1.5
        else:
            score -= 1.5
            
        # Slow vs Trend
        if comp.ema_slow > comp.ema_trend:
            score += 1.0
        else:
            score -= 1.0
            
        # Price vs Trend EMA
        if comp.close > comp.ema_trend:
            score += 1.0
        else:
            score -= 1.0
            
        comp.trend_score = round(score, 2)
        
        # Classification
        if comp.ema_fast > comp.ema_slow > comp.ema_trend:
            comp.trend_alignment = "bullish"
        elif comp.ema_fast < comp.ema_slow < comp.ema_trend:
            comp.trend_alignment = "bearish"
        else:
            comp.trend_alignment = "neutral"
            
        comp.reasons.append(f"Trend: {comp.trend_alignment} (score={comp.trend_score:.2f})")
    
    def _analyze_momentum(self, comp: MarketComponents, hist: pd.DataFrame) -> None:
        """RSI with adaptive z-score thresholds."""
        rsi = comp.rsi
        
        # Rolling RSI stats for z-score
        if "rsi" in hist.columns:
            rsi_mean = hist["rsi"].mean()
            rsi_std = hist["rsi"].std()
            if rsi_std > 0:
                comp.rsi_zscore = (rsi - rsi_mean) / rsi_std
            else:
                comp.rsi_zscore = 0.0
        else:
            comp.rsi_zscore = 0.0
        
        # Classification using settings thresholds
        if rsi >= settings.indicators.rsi_overbought:
            comp.momentum_regime = "overbought"
        elif rsi <= settings.indicators.rsi_oversold:
            comp.momentum_regime = "oversold"
        else:
            comp.momentum_regime = "neutral"
            
        comp.reasons.append(f"RSI: {rsi:.1f} ({comp.momentum_regime}, z={comp.rsi_zscore:.2f})")
    
    def _analyze_volume(self, comp: MarketComponents, hist: pd.DataFrame) -> None:
        """Relative volume with z-score."""
        vr = comp.volume_ratio
        
        if "volume_ratio" in hist.columns:
            vr_mean = hist["volume_ratio"].mean()
            vr_std = hist["volume_ratio"].std()
            if vr_std > 0:
                comp.volume_zscore = (vr - vr_mean) / vr_std
            else:
                comp.volume_zscore = 0.0
        else:
            comp.volume_zscore = 0.0
            
        # Classification
        if vr >= 2.0:
            comp.volume_regime = "high"
        elif vr <= 0.5:
            comp.volume_regime = "low"
        else:
            comp.volume_regime = "normal"
            
        comp.reasons.append(f"Volume: {vr:.2f}x ({comp.volume_regime}, z={comp.volume_zscore:.2f})")
    
    def _analyze_volatility(self, comp: MarketComponents, hist: pd.DataFrame) -> None:
        """ATR percentile for regime detection."""
        atr_pct = comp.atr_percent
        
        if "atr" in hist.columns and "close" in hist.columns:
            hist_atr_pct = hist["atr"] / hist["close"] * 100
            # Percentile rank
            comp.atr_percentile = (hist_atr_pct < atr_pct).mean() * 100
        else:
            comp.atr_percentile = 50.0
            
        # Classification
        if comp.atr_percentile >= 80:
            comp.volatility_regime = "high"
        elif comp.atr_percentile <= 20:
            comp.volatility_regime = "low"
        else:
            comp.volatility_regime = "normal"
            
        comp.reasons.append(f"Volatility: {atr_pct:.2f}% (pctl={comp.atr_percentile:.0f}%, {comp.volatility_regime})")
    
    def _calculate_component_scores(self, comp: MarketComponents) -> None:
        """Calculate weighted scores for ConfidenceEngine."""
        # Trend score already calculated
        comp.component_scores["trend"] = comp.trend_score
        
        # Momentum score from RSI z-score
        # Map z-score to [-2, 2] range
        momentum_score = np.clip(comp.rsi_zscore * 1.5, -2.0, 2.0)
        comp.component_scores["momentum"] = round(momentum_score, 2)
        
        # Volume score from z-score
        volume_score = np.clip(comp.volume_zscore * 1.0, -2.0, 2.0)
        comp.component_scores["volume"] = round(volume_score, 2)
        
        # Volatility score: penalize extremes
        # Normal vol = 0, high/low = negative
        if comp.volatility_regime == "normal":
            vol_score = 0.5
        elif comp.volatility_regime == "high":
            vol_score = -1.0
        else:  # low
            vol_score = -0.5
        comp.component_scores["volatility"] = round(vol_score, 2)