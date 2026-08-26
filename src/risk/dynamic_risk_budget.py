"""
====================================================
QuantAI Professional
Dynamic Risk Budget Manager
====================================================

Dynamic risk budgeting for multi-asset portfolio management.

Features:
- Per-asset risk budgets with dynamic allocation
- Volatility-based risk budget adjustment
- Correlation-aware risk allocation
- Drawdown-triggered risk reduction
- Risk budget rebalancing
- Per-strategy risk limits

====================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class RiskBudgetMode(str, Enum):
    """Risk budget allocation mode."""
    EQUAL = "EQUAL"                    # Equal risk per asset
    VOLATILITY_ADJUSTED = "VOLATILITY_ADJUSTED"  # Inverse volatility weighting
    CORRELATION_ADJUSTED = "CORRELATION_ADJUSTED"  # Correlation-aware
    KELLY_OPTIMAL = "KELLY_OPTIMAL"    # Kelly-optimal allocation
    SIGNAL_WEIGHTED = "SIGNAL_WEIGHTED"  # Signal-strength weighted


@dataclass(frozen=True)
class RiskBudget:
    """Risk budget allocation for an asset."""
    symbol: str
    allocated_risk: float      # Capital allocated for risk
    max_position_notional: float # Max position size
    max_drawdown_pct: float    # Max drawdown for this asset
    risk_per_trade_pct: float    # Risk per trade %
    max_leverage: float        # Max leverage allowed
    current_drawdown: float = 0.0
    current_exposure: float = 0.0
    trades_today: int = 0
    pnl_today: float = 0.0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class RiskBudgetManager:
    """
    Dynamic risk budget manager for portfolio risk allocation.
    
    Manages risk budgets across multiple assets with
    dynamic rebalancing based on:
    - Volatility changes
    - Correlation shifts
    - Drawdown events
    - Signal strength changes
    """

    def __init__(
        self,
        total_equity: float,
        mode: str = "VOLATILITY_ADJUSTED",
        max_total_risk_pct: float = 0.05,      # Max 5% total risk per day
        max_asset_risk_pct: float = 0.02,      # Max 2% per asset
        min_budget_pct: float = 0.005,         # Min 0.5% per asset
        max_leverage: float = 10.0,            # Max 10x leverage
        drawdown_reduction_factor: float = 0.5, # Reduce budget by 50% on drawdown
        rebalance_threshold: float = 0.1,      # Rebalance if allocation drifts > 10%
    ):
        if total_equity <= 0:
            raise ValueError("total_equity must be positive")
        
        if not 0 < max_total_risk_pct <= 1:
            raise ValueError("max_total_risk_pct must be in (0, 1]")
        
        self.total_equity = float(total_equity)
        self.mode = RiskBudgetMode(mode) if isinstance(mode, str) else mode
        self.max_total_risk_pct = max_total_risk_pct
        self.max_asset_risk_pct = max_asset_risk_pct
        self.min_budget_pct = min_budget_pct
        self.max_leverage = max_leverage
        self.drawdown_reduction_factor = drawdown_reduction_factor
        self.rebalance_threshold = rebalance_threshold
        
        self.budgets: Dict[str, RiskBudget] = {}
        self._total_risk_used = 0.0
        self._last_rebalance = datetime.now(timezone.utc)

    def register_asset(
        self,
        symbol: str,
        initial_budget: Optional[float] = None,
        max_drawdown_pct: float = 0.10,  # 10% max drawdown
        risk_per_trade_pct: float = 0.01,  # 1% risk per trade
        max_leverage: float = 10.0,
    ) -> RiskBudget:
        """Register a new asset for risk budgeting."""
        if symbol in self.budgets:
            raise ValueError(f"Asset {symbol} already registered")

        if initial_budget is None:
            # Equal allocation by default
            initial_budget = self.total_equity * 0.1  # 10% of equity per asset
        
        budget = RiskBudget(
            symbol=symbol,
            allocated_risk=initial_budget,
            max_position_notional=initial_budget * 5,  # 5x risk budget
            max_drawdown_pct=max_drawdown_pct,
            risk_per_trade_pct=risk_per_trade_pct,
            max_leverage=max_leverage,
        )
        
        self.budgets[symbol] = budget
        return budget

    def remove_asset(self, symbol: str) -> bool:
        """Remove asset from risk budgeting."""
        if symbol in self.budgets:
            del self.budgets[symbol]
            return True
        return False

    def update_equity(self, equity: float) -> None:
        """Update total equity and rebalance if needed."""
        self.total_equity = float(equity)
        self._rebalance_if_needed()

    def _rebalance_if_needed(self) -> bool:
        """Check if rebalancing is needed and execute if so."""
        # Check allocation drift
        current_allocations = {s: b.allocated_risk / self.total_equity for s, b in self.budgets.items()}
        
        # Target allocations based on mode
        target_allocations = self._calculate_target_allocations()
        
        # Check drift
        max_drift = 0.0
        for symbol, current_pct in current_allocations.items():
            target_pct = target_allocations.get(symbol, 0.0)
            drift = abs(current_pct - target_pct)
            if drift > max_drift:
                max_drift = drift
        
        if max_drift > self.rebalance_threshold:
            self._rebalance_allocations(target_allocations)
            return True
        return False

    def _calculate_target_allocations(self) -> Dict[str, float]:
        """Calculate target allocations based on current mode."""
        allocations = {}
        
        if self.mode == RiskBudgetMode.EQUAL:
            # Equal allocation
            n = len(self.budgets)
            if n > 0:
                per_asset = 1.0 / n
                return {s: per_asset for s in self.budgets}
        
        elif self.mode == RiskBudgetMode.VOLATILITY_ADJUSTED:
            # Inverse volatility weighting
            inv_vols = {}
            for symbol, budget in self.budgets.items():
                vol = getattr(budget, 'current_volatility', 0.02)
                inv_vols[symbol] = 1.0 / max(vol, 0.001)
            total = sum(inv_vols.values())
            if total > 0:
                return {s: v / total for s, v in inv_vols.items()}
            else:
                return {s: 1.0 / len(self.budgets) for s in self.budgets}
        
        elif self.mode == RiskBudgetMode.CORRELATION_ADJUSTED:
            # Correlation-adjusted (inverse correlation weighting)
            # Requires correlation matrix - simplified version
            return {s: 1.0 / len(self.budgets) for s in self.budgets}
        
        elif self.mode == RiskBudgetMode.KELLY_OPTIMAL:
            # Kelly-optimal (requires win rate, payoff data)
            return {s: 1.0 / len(self.budgets) for s in self.budgets}
        
        elif self.mode == RiskBudgetMode.SIGNAL_WEIGHTED:
            # Signal-weighted (requires signal strengths)
            return {s: 1.0 / len(self.budgets) for s in self.budgets}
        
        else:
            return {s: 1.0 / len(self.budgets) for s in self.budgets}

    def update_asset_metrics(
        self,
        symbol: str,
        current_drawdown: float = 0.0,
        current_exposure: float = 0.0,
        current_volatility: float = 0.0,
        pnl_today: float = 0.0,
        trades_today: int = 0,
    ) -> Optional[RiskBudget]:
        """Update asset metrics and check for budget adjustments."""
        if symbol not in self.budgets:
            return None
        
        budget = self.budgets[symbol]
        
        # Update metrics (create new RiskBudget since it's frozen)
        updated = RiskBudget(
            symbol=budget.symbol,
            allocated_risk=budget.allocated_risk,
            max_position_notional=budget.max_position_notional,
            max_drawdown_pct=budget.max_drawdown_pct,
            risk_per_trade_pct=budget.risk_per_trade_pct,
            max_leverage=budget.max_leverage,
            current_drawdown=current_drawdown,
            current_exposure=current_exposure,
            trades_today=budget.trades_today + 1,
            pnl_today=pnl_today,
            last_updated=datetime.now(timezone.utc),
        )
        
        self.budgets[symbol] = updated
        
        # Check for drawdown breach
        if current_drawdown > budget.max_drawdown_pct:
            self._handle_drawdown_breach(symbol)
        
        # Check for risk budget overallocation
        self._check_overallocation()
        
        return updated

    def _handle_drawdown_breach(self, symbol: str) -> None:
        """Handle drawdown breach - reduce risk budget."""
        budget = self.budgets[symbol]
        reduction = self.drawdown_reduction_factor
        new_budget = budget.allocated_risk * (1 - reduction)
        
        self.budgets[symbol] = RiskBudget(
            symbol=budget.symbol,
            allocated_risk=max(new_budget, self.total_equity * 0.005),  # Min 0.5%
            max_position_notional=budget.max_position_notional,
            max_drawdown_pct=budget.max_drawdown_pct,
            risk_per_trade_pct=budget.risk_per_trade_pct,
            max_leverage=budget.max_leverage,
            current_drawdown=budget.current_drawdown,
            current_exposure=budget.current_exposure,
            trades_today=budget.trades_today,
            pnl_today=budget.pnl_today,
            last_updated=datetime.now(timezone.utc),
        )

    def _check_overallocation(self) -> None:
        """Check if total allocated risk exceeds limit."""
        total_allocated = sum(b.allocated_risk for b in self.budgets.values())
        max_allowed = self.total_equity * self.max_total_risk_pct
        
        if total_allocated > max_allowed:
            # Scale down proportionally
            scale = max_allowed / total_allocated
            for symbol, budget in self.budgets.items():
                self.budgets[symbol] = RiskBudget(
                    symbol=budget.symbol,
                    allocated_risk=budget.allocated_risk * scale,
                    max_position_notional=budget.max_position_notional,
                    max_drawdown_pct=budget.max_drawdown_pct,
                    risk_per_trade_pct=budget.risk_per_trade_pct,
                    max_leverage=budget.max_leverage,
                    current_drawdown=budget.current_drawdown,
                    current_exposure=budget.current_exposure,
                    trades_today=budget.trades_today,
                    pnl_today=budget.pnl_today,
                    last_updated=datetime.now(timezone.utc),
                )

    def get_budget(self, symbol: str) -> Optional[RiskBudget]:
        """Get risk budget for symbol."""
        return self.budgets.get(symbol)

    def get_all_budgets(self) -> Dict[str, RiskBudget]:
        """Get all risk budgets."""
        return self.budgets.copy()

    def get_total_risk_used(self) -> float:
        """Get total risk capital allocated."""
        return sum(b.allocated_risk for b in self.budgets.values())

    def get_available_risk(self) -> float:
        """Get remaining risk capital available."""
        total = self.get_total_risk_used()
        max_allowed = self.total_equity * self.max_total_risk_pct
        return max(0.0, max_allowed - total)

    def can_open_position(
        self,
        symbol: str,
        position_notional: float,
        risk_pct: float = 0.01,
    ) -> Tuple[bool, float, str]:
        """
        Check if a position can be opened within risk budget.
        
        Returns:
            (allowed, max_allowed_notional, reason)
        """
        if symbol not in self.budgets:
            return (False, 0.0, f"Asset {symbol} not registered in risk budget")
        
        budget = self.budgets[symbol]
        
        risk_amount = position_notional * risk_pct
        
        # Check if within asset budget
        if budget.current_exposure + risk_amount > budget.allocated_risk:
            return (False, 0.0, f"Would exceed asset risk budget ({budget.allocated_risk:.2f})")
        
        # Check total risk limit
        total_risk = self.get_total_risk_used()
        max_total = self.total_equity * self.max_total_risk_pct
        
        if self._total_risk_used + risk_amount > max_total:
            return (False, 0.0, f"Would exceed total risk limit ({max_total:.2f})")
        
        # Check per-asset limit
        max_asset_risk = self.total_equity * self.max_asset_risk_pct
        if budget.current_exposure + risk_amount > max_asset_risk:
            return (False, 0.0, f"Would exceed per-asset risk limit")
        
        # Check drawdown
        max_dd = self.total_equity * 0.1  # 10% overall max drawdown
        if budget.current_drawdown > max_dd:
            return (False, 0.0, f"Asset drawdown exceeded limit ({budget.current_drawdown:.2%})")
        
        # Allowed
        max_allowed = min(
            budget.allocated_risk - budget.current_exposure,
            self.total_equity * self.max_total_risk_pct - self._total_risk_used,
            budget.max_position_notional,
        )
        max_allowed = max(0.0, max_allowed)
        
        return (True, max_allowed, "OK")

    def allocate_for_signal(
        self,
        symbol: str,
        signal_strength: float,  # -1 to 1
        equity: float,
    ) -> float:
        """Allocate position size based on signal strength."""
        if symbol not in self.budgets:
            return 0.0
        
        budget = self.budgets[symbol]
        
        # Base allocation from budget
        base_alloc = min(
            budget.allocated_risk - budget.current_exposure,
            budget.max_position_notional,
        )
        
        # Scale by signal strength
        alloc = base_alloc * abs(signal_strength)
        
        return max(0.0, alloc)

    def get_allocation_summary(self) -> Dict:
        """Get summary of all risk budgets."""
        return {
            "total_equity": self.total_equity,
            "total_allocated": sum(b.allocated_risk for b in self.budgets.values()),
            "total_risk_used": sum(b.allocated_risk for b in self.budgets.values()),
            "available": self.get_available_risk(),
            "assets": {
                s: {
                    "allocated": b.allocated_risk,
                    "exposure": b.current_exposure,
                    "drawdown": b.current_drawdown,
                    "utilization": b.current_exposure / b.allocated_risk if b.allocated_risk > 0 else 0,
                }
                for s, b in self.budgets.items()
            },
        }
    
    def set_mode(self, mode: str) -> None:
        """Change allocation mode."""
        self.mode = mode if isinstance(mode, RiskBudgetMode) else RiskBudgetMode(mode)
        self._rebalance_allocations(self._calculate_target_allocations())

    def set_total_equity(self, equity: float) -> None:
        """Update total equity and rebalance."""
        if equity <= 0:
            raise ValueError("Equity must be positive")
        self.total_equity = float(equity)
        self._rebalance_if_needed()


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "RiskBudgetMode",
    "RiskBudget",
    "RiskBudgetManager",
]