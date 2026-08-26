"""
====================================================
QuantAI Professional
Portfolio Correlation Risk Management
====================================================

Portfolio-level correlation risk management for multi-asset trading.

Features:
- Rolling correlation matrix computation
- Correlation-based position limits
- Diversification enforcement
- Correlation clustering for portfolio construction
- Correlation risk budgeting

====================================================
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CorrelationPair:
    """Correlation between two assets."""
    asset_a: str
    asset_b: str
    correlation: float
    sample_count: int
    last_updated: datetime


@dataclass(frozen=True)
class CorrelationCluster:
    """Cluster of highly correlated assets."""
    assets: Tuple[str, ...]
    avg_correlation: float
    max_correlation: float
    min_correlation: float
    representative: str  # Representative asset for the cluster


@dataclass(frozen=True)
class CorrelationRiskResult:
    """Result of portfolio correlation risk check."""
    allowed: bool
    reason: str
    new_position_impact: float  # How much new position increases portfolio correlation
    current_max_correlation: float
    cluster_exposure: Dict[str, float]  # Total exposure per cluster
    recommended_max_position: float  # Recommended max position size
    cluster_allocations: Dict[str, float]  # Current allocation per cluster
    metadata: Dict = field(default_factory=dict)


class CorrelationMatrix:
    """
    Rolling correlation matrix for portfolio assets.
    
    Maintains rolling window of returns and computes
    pairwise correlations with configurable lookback.
    """

    def __init__(
        self,
        window_size: int = 100,
        min_periods: int = 30,
        min_correlation: float = 0.7,  # Threshold for "highly correlated"
    ):
        """
        Args:
            window_size: Rolling window size for correlation calculation
            min_periods: Minimum periods required for valid correlation
            min_correlation: Threshold for considering assets "highly correlated"
        """
        if window_size < 10:
            raise ValueError("window_size must be >= 10")
        if min_periods < 5:
            raise ValueError("min_periods must be >= 5")
        if not 0 <= min_correlation <= 1:
            raise ValueError("min_correlation must be in [0, 1]")

        self.window_size = window_size
        self.min_periods = min_periods
        self.min_correlation = min_correlation

        # Rolling returns storage: symbol -> deque of returns
        self._returns: Dict[str, deque] = {}
        self._correlations: Dict[Tuple[str, str], CorrelationPair] = {}
        self._clusters: List[CorrelationCluster] = []

    def update_returns(
        self,
        symbol: str,
        price: float,
        prev_price: float,
    ) -> None:
        """
        Update returns for a symbol with new price.
        
        Args:
            symbol: Asset symbol
            price: Current price
            prev_price: Previous price
        """
        if prev_price <= 0:
            return

        ret = (price - prev_price) / prev_price
        
        if symbol not in self._returns:
            self._returns[symbol] = deque(maxlen=self.window_size)
        
        self._returns[symbol].append(ret)

    def update_batch(
        self,
        prices: Dict[str, float],
        prev_prices: Dict[str, float],
    ) -> None:
        """Update returns for multiple symbols."""
        for symbol, price in prices.items():
            prev_price = prev_prices.get(symbol)
            if prev_price is not None:
                self.update_returns(symbol, price, prev_price)

    def get_correlation(self, asset_a: str, asset_b: str) -> Optional[float]:
        """Get current correlation between two assets."""
        pair = tuple(sorted([asset_a, asset_b]))
        if pair in self._correlations:
            return self._correlations[pair].correlation
        return None

    def get_correlation_matrix(self) -> pd.DataFrame:
        """Get full correlation matrix as DataFrame."""
        symbols = sorted(self._returns.keys())
        if len(symbols) < 2:
            return pd.DataFrame()
        
        matrix = pd.DataFrame(index=symbols, columns=symbols, dtype=float)
        
        for i, a in enumerate(symbols):
            for j, b in enumerate(symbols):
                if i == j:
                    matrix.iloc[i, j] = 1.0
                elif i < j:
                    corr = self.get_correlation(a, b)
                    matrix.iloc[i, j] = corr if corr is not None else 0.0
        
        # Fill symmetric
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                matrix.iloc[j, i] = matrix.iloc[i, j]
        
        return matrix

    def recompute_correlations(self) -> Dict[Tuple[str, str], CorrelationPair]:
        """Recompute all pairwise correlations."""
        symbols = list(self._returns.keys())
        
        if len(symbols) < 2:
            return {}
        
        new_correlations = {}
        
        for i, a in enumerate(symbols):
            for j, b in enumerate(symbols):
                if i >= j:
                    continue
                
                returns_a = np.array(self._returns[a])
                returns_b = np.array(self._returns[b])
                
                if len(returns_a) < self.min_periods or len(returns_b) < self.min_periods:
                    continue
                
                corr = np.corrcoef(returns_a, returns_b)[0, 1]
                
                if np.isnan(corr):
                    corr = 0.0
                
                pair = (a, b)
                new_correlations[pair] = CorrelationPair(
                    asset_a=a,
                    asset_b=b,
                    correlation=float(corr),
                    sample_count=min(len(returns_a), len(returns_b)),
                    last_updated=datetime.now(timezone.utc),
                )
        
        self._correlations = new_correlations
        return new_correlations

    def detect_clusters(self, min_correlation: float = 0.7) -> List[CorrelationCluster]:
        """
        Detect correlation clusters using simple threshold-based clustering.
        
        Assets with correlation >= min_correlation are grouped together.
        """
        correlations = self.recompute_correlations()
        
        # Build adjacency
        adj = defaultdict(set)
        for (a, b), pair in correlations.items():
            if pair.correlation >= min_correlation:
                adj[a].add(b)
                adj[b].add(a)
        
        # Find connected components (clusters)
        visited = set()
        clusters = []
        
        for asset in self._returns.keys():
            if asset not in visited:
                cluster = []
                stack = [asset]
                
                while stack:
                    current = stack.pop()
                    if current in visited:
                        continue
                    visited.add(current)
                    cluster.append(current)
                    
                    for neighbor in adj.get(current, set()):
                        if neighbor not in visited:
                            stack.append(neighbor)
                
                if len(cluster) >= 2:
                    # Calculate cluster stats
                    correlations_in_cluster = []
                    for i, a in enumerate(cluster):
                        for b in cluster[i+1:]:
                            pair = tuple(sorted([a, b]))
                            if pair in correlations:
                                correlations_in_cluster.append(correlations[pair].correlation)
                    
                    avg_corr = np.mean(correlations_in_cluster) if correlations_in_cluster else 0
                    max_corr = max(correlations_in_cluster) if correlations_in_cluster else 0
                    min_corr = min(correlations_in_cluster) if correlations_in_cluster else 0
                    
                    clusters.append(CorrelationCluster(
                        assets=tuple(sorted(cluster)),
                        avg_correlation=avg_corr,
                        max_correlation=max_corr,
                        min_correlation=min_corr,
                        representative=cluster[0],
                    ))
        
        self._clusters = clusters
        return clusters

    def get_cluster_exposures(
        self,
        positions: Dict[str, float],  # symbol -> notional position
    ) -> Dict[str, float]:
        """Get total exposure per correlation cluster."""
        clusters = self.detect_clusters()
        cluster_exposure = {}
        
        for cluster in clusters:
            total = sum(
                positions.get(asset, 0.0) 
                for asset in cluster.assets
            )
            if total > 0:
                cluster_exposure[cluster.representative] = total
        
        return cluster_exposure


class PortfolioCorrelationManager:
    """
    High-level portfolio correlation risk manager.
    
    Enforces:
    - Max correlation between any two positions
    - Max total exposure per correlation cluster
    - Minimum diversification (min number of uncorrelated assets)
    """

    def __init__(
        self,
        max_pair_correlation: float = 0.7,
        max_cluster_exposure_pct: float = 40.0,  # Max % of equity per cluster
        min_diversified_assets: int = 3,
        correlation_matrix: Optional[CorrelationMatrix] = None,
    ):
        """
        Args:
            max_pair_correlation: Max allowed correlation between any two positions
            max_cluster_exposure_pct: Max % of equity in any single correlation cluster
            min_diversified_assets: Minimum number of uncorrelated positions
            correlation_matrix: Optional pre-configured CorrelationMatrix
        """
        if not 0 <= max_pair_correlation <= 1:
            raise ValueError("max_pair_correlation must be in [0, 1]")
        if not 0 < max_cluster_exposure_pct <= 100:
            raise ValueError("max_cluster_exposure_pct must be in (0, 100]")
        if min_diversified_assets < 1:
            raise ValueError("min_diversified_assets must be >= 1")

        self.max_pair_correlation = max_pair_correlation
        self.max_cluster_exposure_pct = max_cluster_exposure_pct
        self.min_diversified_assets = min_diversified_assets

        self.correlation_matrix = correlation_matrix or CorrelationMatrix()

    def update_prices(
        self,
        prices: Dict[str, float],
        prev_prices: Dict[str, float],
    ) -> None:
        """Update price data and recompute correlations."""
        self.correlation_matrix.update_batch(prices, prev_prices)

    def check_new_position(
        self,
        symbol: str,
        position_notional: float,
        current_positions: Dict[str, float],
        equity: float,
    ) -> Tuple[bool, str, float]:
        """
        Check if a new position violates correlation limits.
        
        Args:
            symbol: Symbol of new position
            position_notional: Notional value of new position
            current_positions: Current positions {symbol: notional}
            equity: Total equity
            
        Returns:
            (allowed, reason, max_allowed_notional)
        """
        # Check pair correlations
        for existing_symbol, existing_notional in current_positions.items():
            if existing_symbol == symbol:
                continue
            
            corr = self.correlation_matrix.get_correlation(symbol, existing_symbol)
            if corr is not None and corr > self.max_pair_correlation:
                return (False, 
                    f"Correlation with {existing_symbol} ({corr:.2f}) exceeds limit ({self.max_pair_correlation})",
                    0.0)
        
        # Check cluster exposure
        self.correlation_matrix.update_returns  # Ensure returns are fresh
        clusters = self.correlation_matrix.detect_clusters()
        
        # Find which cluster the new symbol belongs to
        for cluster in self.correlation_matrix._clusters:
            if symbol in cluster.assets:
                # Check cluster exposure limit
                cluster_exposure = sum(
                    current_positions.get(asset, 0.0) 
                    for asset in cluster.assets
                ) + position_notional
                
                max_cluster_notional = equity * self.max_cluster_exposure_pct / 100
                
                if cluster_exposure > max_cluster_notional:
                    max_additional = max(0, max_cluster_notional - (cluster_exposure - position_notional))
                    return (False, 
                        f"Cluster {cluster.representative} exposure would exceed {self.max_cluster_exposure_pct}% limit",
                        max_additional)
        
        return (True, "OK", float('inf'))

    def check_diversification(
        self,
        current_positions: Dict[str, float],
    ) -> Tuple[bool, str]:
        """Check if portfolio meets minimum diversification."""
        active_assets = [s for s, v in current_positions.items() if v > 0]
        
        if len(active_assets) < self.min_diversified_assets:
            return (False, f"Need at least {self.min_diversified_assets} uncorrelated assets, have {len(active_assets)}")
        
        # Check if we have enough uncorrelated pairs
        uncorrelated_pairs = 0
        for i, a in enumerate(active_assets):
            for b in active_assets[i+1:]:
                corr = self.correlation_matrix.get_correlation(a, b)
                if corr is not None and corr < 0.5:  # Low correlation threshold
                    uncorrelated_pairs += 1
        
        # Need at least n-1 uncorrelated pairs for n assets to be diversified
        min_pairs = self.min_diversified_assets - 1
        if uncorrelated_pairs < min_pairs:
            return (False, f"Portfolio not sufficiently diversified: {uncorrelated_pairs} uncorrelated pairs, need {min_pairs}")
        
        return (True, "OK")

    def get_recommended_allocation(
        self,
        signals: Dict[str, float],  # symbol -> signal strength (-1 to 1)
        equity: float,
        max_position_pct: float = 0.1,  # Max 10% per position
    ) -> Dict[str, float]:
        """
        Get recommended position sizes based on correlation and signal strength.
        
        Uses inverse correlation weighting for diversification.
        """
        symbols = [s for s in signals if signals[s] != 0]
        if not symbols:
            return {}
        
        # Build correlation submatrix for active symbols
        n = len(symbols)
        corr_matrix = np.eye(n)
        for i, a in enumerate(symbols):
            for j, b in enumerate(symbols):
                if i != j:
                    corr = self.correlation_matrix.get_correlation(a, b)
                    if corr is not None:
                        corr_matrix[i, j] = corr
        
        # Inverse correlation weighting (higher weight for less correlated)
        weights = np.ones(n)
        for i in range(n):
            for j in range(n):
                if i != j and corr_matrix[i, j] != 0:
                    weights[i] += abs(corr_matrix[i, j])
        
        # Normalize weights by signal strength
        for i, s in enumerate(symbols):
            signal_strength = abs(signals[s])
            weights[i] *= signal_strength
        
        # Normalize to sum to 1
        total = np.sum(weights)
        if total > 0:
            weights = weights / total
        else:
            weights = np.ones(n) / n
        
        # Apply max position limit
        allocations = {}
        for i, s in enumerate(symbols):
            alloc = weights[i] * equity * 0.9  # 90% invested, 10% reserve
            max_alloc = equity * 0.1  # 10% max per position
            allocations[symbols[i]] = min(alloc, max_alloc)
        
        return allocations


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "CorrelationPair",
    "CorrelationCluster",
    "CorrelationRiskResult",
    "CorrelationMatrix",
    "PortfolioCorrelationManager",
]