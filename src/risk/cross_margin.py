"""
====================================================
QuantAI Professional
Cross-Margin Management
====================================================

Cross-margin management for multi-asset futures trading.

Features:
- Unified margin across all positions
- Cross-margin vs isolated margin simulation
- Auto-liquidation risk assessment
- Margin optimization across positions
- Liquidation price calculation
- Margin call warnings

====================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class MarginMode(str, Enum):
    """Margin mode types."""
    ISOLATED = "ISOLATED"      # Each position has separate margin
    CROSS = "CROSS"            # Shared margin across all positions


@dataclass(frozen=True)
class PositionMargin:
    """Margin info for a single position."""
    symbol: str
    side: str              # "LONG" or "SHORT"
    size: float            # Position size (contracts)
    entry_price: float
    mark_price: float
    leverage: float
    isolated_margin: float   # Margin allocated to this position
    maintenance_margin: float # Maintenance margin requirement
    liquidation_price: float  # Estimated liquidation price
    unrealized_pnl: float
    margin_ratio: float      # Current margin ratio (equity / maintenance_margin)
    is_isolated: bool


@dataclass(frozen=True)
class CrossMarginAccount:
    """Cross-margin account summary."""
    total_equity: float
    total_margin_used: float
    total_maintenance_margin: float
    available_margin: float
    total_unrealized_pnl: float
    margin_ratio: float      # total_equity / total_maintenance_margin
    positions: Dict[str, PositionMargin]
    margin_mode: MarginMode
    
    def get_liquidation_risk(self) -> str:
        """Get overall liquidation risk level."""
        if self.margin_ratio >= 5.0:
            return "LOW"
        elif self.margin_ratio >= 2.5:
            return "MEDIUM"
        elif self.margin_ratio >= 1.5:
            return "HIGH"
        elif self.margin_ratio >= 1.0:
            return "CRITICAL"
        else:
            return "BANKRUPT"


class CrossMarginManager:
    """
    Cross-margin manager for unified margin management.
    
    Supports both isolated and cross-margin modes with
    seamless switching and risk monitoring.
    """

    def __init__(
        self,
        initial_equity: float,
        margin_mode: MarginMode = MarginMode.CROSS,
        maintenance_margin_rate: float = 0.005,  # 0.5% maintenance margin
        initial_margin_rate: float = 0.01,      # 1% initial margin
    ):
        """
        Args:
            initial_equity: Starting equity
            margin_mode: ISOLATED or CROSS
            maintenance_margin_rate: Maintenance margin rate (e.g., 0.5%)
            initial_margin_rate: Initial margin rate (e.g., 1%)
        """
        if initial_equity <= 0:
            raise ValueError("initial_equity must be positive")
        
        if not 0 < maintenance_margin_rate < 1:
            raise ValueError("maintenance_margin_rate must be in (0, 1)")
        
        if not 0 < initial_margin_rate < 1:
            raise ValueError("initial_margin_rate must be in (0, 1)")

        self.initial_equity = float(initial_equity)
        self.equity = float(initial_equity)
        self.margin_mode = margin_mode
        self.maintenance_margin_rate = maintenance_margin_rate
        self.initial_margin_rate = initial_margin_rate

        self.positions: Dict[str, PositionMargin] = {}

    def set_margin_mode(self, mode: MarginMode) -> None:
        """Switch between ISOLATED and CROSS margin modes."""
        if mode == self.margin_mode:
            return
        
        if mode == MarginMode.CROSS:
            # Switching to cross: consolidate margins
            self._consolidate_to_cross()
        else:
            # Switching to isolated: distribute equity
            self._distribute_to_isolated()
        
        self.margin_mode = mode

    def _consolidate_to_cross(self) -> None:
        """Consolidate all isolated margins to cross margin pool."""
        # In cross mode, all margins are pooled
        pass  # Equity is already pooled

    def _distribute_to_isolated(self) -> None:
        """Distribute cross margin to isolated positions."""
        # Allocate equity proportionally to positions
        total_notional = sum(
            pos.size * pos.mark_price for pos in self.positions.values()
        )
        
        if total_notional > 0:
            for symbol, position in self.positions.items():
                position_notional = position.size * position.mark_price
                allocation = position_notional / total_notional if total_notional > 0 else 0
                position.isolated_margin = self.equity * allocation
                position.maintenance_margin = position.isolated_margin * 0.5  # 50% of isolated

    def update_mark_prices(self, prices: Dict[str, float]) -> None:
        """Update mark prices for all positions."""
        for symbol, price in prices.items():
            if symbol in self.positions:
                pos = self.positions[symbol]
                pos.mark_price = price
                pos.unrealized_pnl = self._calc_unrealized_pnl(pos)
                pos.margin_ratio = self._calc_margin_ratio(pos)

    def _calc_unrealized_pnl(self, position: PositionMargin) -> float:
        if position.side == "LONG":
            return (position.mark_price - position.entry_price) * position.size
        else:
            return (position.entry_price - position.mark_price) * position.size

    def _calc_margin_ratio(self, position: PositionMargin) -> float:
        if position.maintenance_margin <= 0:
            return float('inf')
        return (position.isolated_margin + position.unrealized_pnl) / position.maintenance_margin

    def open_position(
        self,
        symbol: str,
        side: str,
        size: float,
        entry_price: float,
        leverage: float,
        mark_price: Optional[float] = None,
    ) -> PositionMargin:
        """
        Open a new position with margin allocation.
        
        Args:
            symbol: Trading symbol
            side: "LONG" or "SHORT"
            size: Position size (contracts)
            entry_price: Entry price
            leverage: Leverage to use
            mark_price: Current mark price (optional)
            
        Returns:
            PositionMargin with allocated margin
        """
        if size <= 0:
            raise ValueError("size must be positive")
        if leverage <= 0:
            raise ValueError("leverage must be positive")
        if leverage > 100:
            raise ValueError("leverage too high (max 100x)")

        mark_price = mark_price or entry_price
        
        # Calculate notional value
        notional = size * entry_price
        
        # Calculate margin requirements
        initial_margin = notional / leverage  # Initial margin
        maintenance_margin = notional * 0.005   # 0.5% maintenance margin
        
        if self.margin_mode == MarginMode.CROSS:
            # In cross margin, use shared equity
            isolated_margin = self.equity  # Full equity available
        else:
            # In isolated, allocate from equity
            isolated_margin = notional * 0.1  # 10% of equity per position (configurable)
            if isolated_margin > self.equity:
                isolated_margin = self.equity

        position = PositionMargin(
            symbol=symbol,
            side=side.upper(),
            size=size,
            entry_price=entry_price,
            mark_price=mark_price,
            leverage=leverage,
            isolated_margin=isolated_margin,
            maintenance_margin=maintenance_margin,
            liquidation_price=self._calc_liquidation_price(
                side=side.upper(),
                entry_price=entry_price,
                maintenance_margin=maintenance_margin,
                mark_price=mark_price,
            ),
            unrealized_pnl=0.0,
            margin_ratio=float('inf'),
            is_isolated=(self.margin_mode == MarginMode.ISOLATED),
        )

        self.positions[symbol] = position
        return self.positions[symbol]

    def close_position(self, symbol: str, exit_price: float) -> float:
        """
        Close position and return realized PnL.
        """
        if symbol not in self.positions:
            raise ValueError(f"No position for {symbol}")

        position = self.positions[symbol]
        position.mark_price = exit_price
        realized_pnl = self._calc_unrealized_pnl(position)
        
        # Update equity
        self.equity += realized_pnl
        
        del self.positions[symbol]
        return realized_pnl

    def update_position(
        self,
        symbol: str,
        size: Optional[float] = None,
        leverage: Optional[float] = None,
    ) -> Optional[PositionMargin]:
        """Update position size or leverage."""
        if symbol not in self.positions:
            return None

        position = self.positions[symbol]
        
        if size is not None and size > 0:
            position.size = size
        if leverage is not None and leverage > 0:
            position.leverage = leverage

        position.mark_price = position.mark_price  # Keep current
        position.unrealized_pnl = self._calc_unrealized_pnl(position)
        position.margin_ratio = self._calc_margin_ratio(position)
        position.maintenance_margin = position.size * position.mark_price * 0.005
        
        return position

    def get_account_summary(self) -> CrossMarginAccount:
        """Get complete cross-margin account summary."""
        total_margin_used = sum(p.isolated_margin for p in self.positions.values())
        total_maint = sum(p.maintenance_margin for p in self.positions.values())
        total_pnl = sum(p.unrealized_pnl for p in self.positions.values())
        
        equity = self.equity
        margin_used = total_margin_used if self.margin_mode == MarginMode.ISOLATED else max(total_margin_used, sum(p.size * p.mark_price * 0.01 for p in self.positions.values()))
        
        margin_ratio = equity / margin_used if margin_used > 0 else float('inf')
        
        return CrossMarginAccount(
            total_equity=equity,
            total_margin_used=margin_used,
            total_maintenance_margin=total_maint,
            available_margin=equity - margin_used,
            total_unrealized_pnl=total_pnl,
            margin_ratio=margin_ratio,
            positions=self.positions.copy(),
            margin_mode=self.margin_mode,
        )

    def get_liquidation_risk_by_symbol(self) -> Dict[str, str]:
        """Get liquidation risk for each position."""
        return {
            symbol: self._get_position_risk_level(pos)
            for symbol, pos in self.positions.items()
        }

    def _get_position_risk_level(self, position: PositionMargin) -> str:
        if position.margin_ratio >= 5.0:
            return "LOW"
        elif position.margin_ratio >= 2.5:
            return "MEDIUM"
        elif position.margin_ratio >= 1.5:
            return "HIGH"
        elif position.margin_ratio >= 1.0:
            return "CRITICAL"
        else:
            return "BANKRUPT"

    def get_liquidation_prices(self) -> Dict[str, float]:
        """Get liquidation prices for all positions."""
        return {
            symbol: pos.liquidation_price
            for symbol, pos in self.positions.items()
        }

    def _calc_liquidation_price(
        self,
        side: str,
        entry_price: float,
        maintenance_margin: float,
        mark_price: float,
    ) -> float:
        """Calculate estimated liquidation price."""
        if side == "LONG":
            # Liquidation when: equity - (entry - mark) * size <= maint_margin * size
            # mark <= entry - (1/leverage - maint_rate)
            liquidation_price = entry_price * (1 - 1/leverage + 0.005)
        else:
            # SHORT
            # Liquidation when: equity - (mark - entry) * size <= maint_margin * size
            # mark >= entry + (1/leverage - maint_rate)
            liquidation_price = entry_price * (1 + 1/leverage - 0.005)
        
        return max(0.0001, liquidation_price)

    def estimate_liquidation_time(
        self,
        symbol: str,
        daily_volatility: float,
        current_price: float,
    ) -> Optional[float]:
        """
        Estimate time to liquidation assuming random walk.
        
        Returns days until liquidation (None if unlikely).
        """
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        distance = abs(pos.liquidation_price - pos.mark_price)
        if distance <= 0:
            return None
        
        # Daily expected move
        daily_move = pos.mark_price * daily_volatility
        
        if daily_move <= 0:
            return None
        
        # Rough estimate: distance / (daily_move * sqrt(2/pi)) for expected first passage
        days = distance / (daily_move * 0.8)  # Approximate
        return max(0.1, days)


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "MarginMode",
    "PositionMargin",
    "CrossMarginAccount",
    "CrossMarginManager",
]