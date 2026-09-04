"""
Binance USDT-M Futures Accounting — Exchange-faithful simulation.

Models:
- Mark price (fair price for PnL/liquidation)
- Initial / Maintenance margin
- Funding rate accrual (8h intervals)
- Liquidation price calculation
- Isolated / Cross margin modes
- Realized / Unrealized PnL
- Margin usage / Free margin
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import math


class MarginMode(str, Enum):
    ISOLATED = "ISOLATED"
    CROSS = "CROSS"


class PositionSide(str, Enum):
    LONG = "LONG"
    SHORT = "SELL"


class LiquidationReason(str, Enum):
    MAINTENANCE_MARGIN = "MAINTENANCE_MARGIN"
    MARK_PRICE_BREACH = "MARK_PRICE_BREACH"


@dataclass
class FuturesPosition:
    """Binance-style futures position with full margin accounting — P2.1 2.0.
    
    Includes: mark price, initial/maintenance margin, funding, liquidation,
    isolated/cross, realized/unrealized PnL, available margin.
    """
    symbol: str
    side: PositionSide
    quantity: float = 0.0
    entry_price: float = 0.0
    mark_price: float = 0.0
    leverage: float = 1.0
    margin_mode: MarginMode = MarginMode.ISOLATED
    
    # Margin fields — P2.1
    initial_margin: float = 0.0
    maintenance_margin: float = 0.0  # cached maintenance required
    position_margin: float = 0.0  # For isolated: isolatedWallet
    wallet_balance: float = 0.0  # For cross: share of wallet
    available_balance: float = 0.0  # P2.1: available for new positions (wallet+unrealized - used)
    
    # PnL — P2.1
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    # Funding — P2.2 exchange-specific
    funding_rate: float = 0.0
    last_funding_time: float = 0.0  # UTC timestamp seconds
    accrued_funding: float = 0.0
    funding_history: list = field(default_factory=list)  # list of (timestamp, rate, amount)
    
    # Liquidation — P2.1
    liquidation_price: float = 0.0
    is_liquidated: bool = False

    # Binance USDT-M: maintenance margin 0.5% at 10x, tiered 0.65%+ for larger notional
    # Simplified flat 0.5% for <500k notional, 1% above — we use 0.5% base
    MAINTENANCE_MARGIN_RATE: float = 0.005
    # Initial margin rate = 1/leverage
    @property
    def initial_margin_rate(self) -> float:
        return 1.0 / self.leverage if self.leverage > 0 else 1.0
    
    def notional(self) -> float:
        """Position notional value = quantity * mark_price."""
        return self.quantity * self.mark_price
    
    def position_value(self) -> float:
        """Position value for margin calculation."""
        return abs(self.quantity) * self.mark_price
    
    def initial_margin_required(self) -> float:
        """Initial margin required = position_value / leverage."""
        if self.leverage <= 0:
            return float('inf')
        return self.position_value() / self.leverage
    
    def maintenance_margin_required(self) -> float:
        """Maintenance margin required = position_value * maintenance_rate."""
        return self.position_value() * self.MAINTENANCE_MARGIN_RATE
    
    def unrealized_pnl_calc(self) -> float:
        """Calculate unrealized PnL from mark price."""
        if self.side == PositionSide.LONG:
            return (self.mark_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - self.mark_price) * self.quantity
    
    def update_mark_price(self, mark_price: float) -> None:
        """Update mark price and recalculate unrealized PnL — P2.1 mark price drives PnL/liquidation/available."""
        self.mark_price = mark_price
        self.unrealized_pnl = self.unrealized_pnl_calc()
        self._update_liquidation_price()
        # P2.1: update available margin on mark change
        if self.margin_mode == MarginMode.CROSS:
            self.available_balance = max(0.0, self.wallet_balance + self.unrealized_pnl - self.initial_margin)
        else:
            self.available_balance = max(0.0, self.wallet_balance - self.position_margin)
    
    def _update_liquidation_price(self) -> None:
        """Calculate liquidation price based on maintenance margin."""
        if self.quantity == 0:
            self.liquidation_price = 0.0
            return
        
        # Liquidation when margin_balance < maintenance_margin
        # margin_balance = wallet_balance + unrealized_pnl
        # For isolated: margin_balance = position_margin + unrealized_pnl
        # For cross: margin_balance = wallet_balance + unrealized_pnl
        
        if self.margin_mode == MarginMode.ISOLATED:
            margin_balance = self.position_margin + self.unrealized_pnl
        else:
            margin_balance = self.wallet_balance + self.unrealized_pnl
        
        maintenance_margin = self.maintenance_margin_required()
        
        if margin_balance <= 0:
            self.liquidation_price = self.mark_price
            return
        
        # Liquidation price: price where margin_balance = maintenance_margin
        # For LONG: entry - (margin_balance - maintenance) / quantity
        # For SHORT: entry + (margin_balance - maintenance) / quantity
        if self.side == PositionSide.LONG:
            if self.quantity > 0:
                self.liquidation_price = self.entry_price - (margin_balance - maintenance_margin) / self.quantity
        else:
            if self.quantity > 0:
                self.liquidation_price = self.entry_price + (margin_balance - maintenance_margin) / self.quantity
        
        # Safety bounds
        if self.liquidation_price < 0:
            self.liquidation_price = 0.0
    
    def check_liquidation(self, mark_price: float) -> Optional[LiquidationReason]:
        """Check if position should be liquidated at given mark price."""
        self.update_mark_price(mark_price)
        
        if self.margin_mode == MarginMode.ISOLATED:
            margin_balance = self.position_margin + self.unrealized_pnl
        else:
            margin_balance = self.wallet_balance + self.unrealized_pnl
        
        maintenance_margin = self.maintenance_margin_required()
        
        if margin_balance <= maintenance_margin:
            self.is_liquidated = True
            return LiquidationReason.MAINTENANCE_MARGIN
        
        # Also check if mark price crossed liquidation price
        if self.liquidation_price > 0:
            if self.side == PositionSide.LONG and mark_price <= self.liquidation_price:
                self.is_liquidated = True
                return LiquidationReason.MARK_PRICE_BREACH
            elif self.side == PositionSide.SHORT and mark_price >= self.liquidation_price:
                self.is_liquidated = True
                return LiquidationReason.MARK_PRICE_BREACH
        
        return None
    
    def apply_funding(self, funding_rate: float, funding_time: float | None = None, hours_since_last: float = 8.0) -> float:
        """Apply funding payment — P2.2 exchange-specific (Binance 00/08/16 UTC).
        
        If funding_time provided, uses that timestamp for history; otherwise uses hours_since_last scaling.
        Returns funding amount (positive = receive, negative = pay).
        """
        self.funding_rate = funding_rate
        
        # Funding = notional * funding_rate * (hours/8) — P2.2: use actual funding_rate from exchange at event time
        # For Binance, funding_rate is 8h rate; if called at exact funding event, hours_since_last should be ~8
        funding_amount = self.notional() * funding_rate * (hours_since_last / 8.0)
        
        if self.side == PositionSide.LONG:
            funding_pnl = -funding_amount  # Long pays when funding positive
        else:
            funding_pnl = funding_amount   # Short receives
        
        self.accrued_funding += funding_pnl
        self.realized_pnl += funding_pnl
        # P2.2: record with timestamp for audit
        ts = funding_time if funding_time is not None else self.last_funding_time
        self.funding_history.append((ts, funding_rate, funding_pnl))
        self.last_funding_time = ts if ts else 0
        
        # Update wallet/position margin for funding settlement
        if self.margin_mode == MarginMode.CROSS:
            self.wallet_balance += funding_pnl
        else:
            # Isolated: funding deducted from position margin (if negative) or added
            self.position_margin += funding_pnl
            if self.position_margin < 0:
                self.position_margin = 0
        
        return funding_pnl
    
    def update_wallet_balance(self, balance: float) -> None:
        """Update wallet balance (for cross margin) — P2.1 available = wallet + unrealized - used."""
        self.wallet_balance = balance
        # Available for cross: wallet + unrealized - initialMargin (simplified)
        # For isolated: available is wallet - positionMargin (isolated margin is locked)
        if self.margin_mode == MarginMode.CROSS:
            self.available_balance = self.wallet_balance + self.unrealized_pnl - self.initial_margin
        else:
            self.available_balance = self.wallet_balance - self.position_margin

    def available_margin(self) -> float:
        """P2.1: available margin for new positions."""
        if self.margin_mode == MarginMode.CROSS:
            # Cross: wallet + unrealized - initialMarginUsed
            return max(0.0, self.wallet_balance + self.unrealized_pnl - self.initial_margin)
        else:
            # Isolated: wallet - positionMargin (position margin locked)
            return max(0.0, self.wallet_balance - self.position_margin)

    def add_margin(self, amount: float) -> None:
        """Add margin to isolated position — increases position_margin and updates liquidation."""
        if self.margin_mode == MarginMode.ISOLATED:
            self.position_margin += amount
            self.wallet_balance -= amount
            self._update_liquidation_price()
    
    def remove_margin(self, amount: float) -> float:
        """Remove margin from isolated position. Returns actual amount removed — P2.1."""
        if self.margin_mode == MarginMode.ISOLATED:
            # Cannot remove below maintenance margin
            min_margin = self.maintenance_margin_required() * 1.1  # 10% buffer above maintenance
            max_removable = max(0.0, self.position_margin - min_margin)
            removed = min(amount, max_removable)
            self.position_margin -= removed
            self.wallet_balance += removed
            self._update_liquidation_price()
            return removed
        return 0.0
    
    def margin_ratio(self) -> float:
        """Current margin ratio (margin_balance / maintenance_margin)."""
        if self.margin_mode == MarginMode.ISOLATED:
            margin_balance = self.position_margin + self.unrealized_pnl
        else:
            margin_balance = self.wallet_balance + self.unrealized_pnl
        
        maintenance = self.maintenance_margin_required()
        if maintenance <= 0:
            return float('inf')
        return margin_balance / maintenance
    
    def is_near_liquidation(self, threshold: float = 1.2) -> bool:
        """Check if position is near liquidation (margin_ratio < threshold)."""
        return self.margin_ratio() < threshold


class FundingSchedule:
    """P2.2 Exchange-specific funding schedule — Binance USDT-M 00:00/08:00/16:00 UTC.
    
    Not i%2 nor hardcoded 8h from open, but actual exchange funding events with timestamps.
    """

    # Binance funding times: 00:00, 08:00, 16:00 UTC daily
    FUNDING_HOURS_UTC = (0, 8, 16)

    @classmethod
    def is_funding_time(cls, timestamp) -> bool:
        """Check if timestamp is at funding event (within same hour as 00/08/16 UTC)."""
        try:
            import pandas as pd
            ts = pd.Timestamp(timestamp)
            # Convert to UTC if not
            if ts.tz is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            return ts.hour in cls.FUNDING_HOURS_UTC and ts.minute == 0
        except Exception:
            return False

    @classmethod
    def next_funding_time(cls, timestamp):
        """Next funding event after timestamp."""
        import pandas as pd
        ts = pd.Timestamp(timestamp)
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        # Find next 00/08/16
        for h in cls.FUNDING_HOURS_UTC:
            candidate = ts.normalize() + pd.Timedelta(hours=h)
            if candidate > ts:
                return candidate
        # Next day 00:00
        return ts.normalize() + pd.Timedelta(days=1)

    @classmethod
    def funding_events_in_range(cls, start, end, funding_rate: float = 0.0001) -> list:
        """Generate funding events between start and end with timestamps and rates.
        
        Exchange rules: Binance uses 8h rate published beforehand; we use provided rate per event
        (in prod would fetch from /fapi/v1/fundingRate). Returns list of (timestamp, rate).
        """
        import pandas as pd
        events = []
        cur = cls.next_funding_time(start)
        while cur <= pd.Timestamp(end):
            events.append((cur, funding_rate))
            cur += pd.Timedelta(hours=8)
            # Ensure alignment to 00/08/16 (handle drift)
            # Next funding is always +8h, which stays on schedule
        return events

    @classmethod
    def should_apply_funding(cls, prev_candle_ts, curr_candle_ts) -> bool:
        """P2.2: Check if funding event occurred between candles (exchange-specific)."""
        try:
            import pandas as pd
            prev = pd.Timestamp(prev_candle_ts)
            curr = pd.Timestamp(curr_candle_ts)
            # If any funding hour boundary crossed
            events = cls.funding_events_in_range(prev, curr)
            return len(events) > 0
        except Exception:
            return False


class FuturesAccount:
    """Cross-margin futures account managing multiple positions."""
    
    def __init__(self, initial_equity: float = 10000.0):
        self.total_wallet_balance = initial_equity
        self.total_unrealized_pnl = 0.0
        self.total_realized_pnl = 0.0
        self.total_margin_used = 0.0
        self.positions: dict[str, FuturesPosition] = {}
        self.margin_mode = MarginMode.CROSS
    
    def open_position(self, pos: FuturesPosition) -> bool:
        """P2.1: Open with full margin accounting — initial/maintenance, isolated/cross, realized/unrealized."""
        if pos.symbol in self.positions:
            return False
        
        required_margin = pos.initial_margin_required()
        # P2.1: set initial_margin field for audit
        pos.initial_margin = required_margin
        pos.maintenance_margin = pos.maintenance_margin_required()
        if pos.margin_mode == MarginMode.CROSS:
            # Cross: check available margin (wallet + unrealized - used)
            available = self.free_margin()
            if required_margin > available * 0.95:  # Leave 5% buffer
                return False
            pos.wallet_balance = self.total_wallet_balance  # share, for ratio calc
        else:
            pos.position_margin = required_margin
            pos.wallet_balance = self.total_wallet_balance
            if required_margin > self.total_wallet_balance * 0.95:
                return False
        
        self.positions[pos.symbol] = pos
        # Update account used margin
        self.total_margin_used += required_margin
        return True
    
    def close_position(self, symbol: str, exit_price: float) -> Optional[float]:
        """Close position and return realized PnL."""
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        pos.update_mark_price(exit_price)
        realized = pos.unrealized_pnl
        
        self.total_realized_pnl += realized
        self.total_wallet_balance += realized
        
        if pos.margin_mode == MarginMode.ISOLATED:
            self.total_wallet_balance += pos.position_margin
        
        del self.positions[symbol]
        return realized
    
    def update_all_marks(self, mark_prices: dict[str, float]) -> list[tuple[str, Optional[str]]]:
        """Update all positions with current mark prices. Returns list of (symbol, liquidation_reason)."""
        liquidations = []
        self.total_unrealized_pnl = 0.0
        self.total_margin_used = 0.0
        
        for symbol, pos in self.positions.items():
            if symbol in mark_prices:
                reason = pos.check_liquidation(mark_prices[symbol])
                if reason:
                    liquidations.append((symbol, reason.value))
                self.total_unrealized_pnl += pos.unrealized_pnl
                self.total_margin_used += pos.initial_margin_required()
        
        return liquidations
    
    def total_equity(self) -> float:
        """Total account equity = wallet + unrealized PnL."""
        return self.total_wallet_balance + self.total_unrealized_pnl
    
    def margin_ratio(self) -> float:
        """Overall margin ratio for cross margin."""
        if self.margin_mode != MarginMode.CROSS:
            return float('inf')
        
        total_maintenance = sum(p.maintenance_margin_required() for p in self.positions.values())
        total_equity = self.total_equity()
        
        if total_maintenance <= 0:
            return float('inf')
        return total_equity / total_maintenance
    
    def free_margin(self) -> float:
        """P2.1 Free margin available for new positions — includes unrealized PnL."""
        if self.margin_mode == MarginMode.CROSS:
            # Cross: equity (wallet+unrealized) - used initial margin
            return max(0.0, self.total_equity() - self.total_margin_used)
        # Isolated: wallet - used (isolated margin locked per position, unrealized not counted for new)
        return max(0.0, self.total_wallet_balance - self.total_margin_used)


# Convenience factory
def create_futures_position(
    symbol: str,
    side: PositionSide,
    quantity: float,
    entry_price: float,
    leverage: float = 10.0,
    margin_mode: MarginMode = MarginMode.ISOLATED,
    wallet_balance: float = 10000.0,
) -> FuturesPosition:
    """P2.1: Create a new futures position with proper margin setup — mark, initial/maintenance, isolated/cross."""
    pos = FuturesPosition(
        symbol=symbol,
        side=side,
        quantity=quantity,
        entry_price=entry_price,
        mark_price=entry_price,
        leverage=leverage,
        margin_mode=margin_mode,
        wallet_balance=float(wallet_balance),
    )
    # P2.1: set initial/maintenance and available
    req = pos.initial_margin_required()
    pos.initial_margin = req
    pos.maintenance_margin = pos.maintenance_margin_required()
    if margin_mode == MarginMode.ISOLATED:
        pos.position_margin = req
        # wallet remains total, available = wallet - position_margin
        pos.available_balance = max(0.0, pos.wallet_balance - pos.position_margin)
    else:
        # Cross: wallet shared, available = wallet + unreal - initial (unreal 0 at open)
        pos.available_balance = max(0.0, pos.wallet_balance - req)
    
    pos._update_liquidation_price()
    return pos


# Example usage and tests
if __name__ == "__main__":
    # Test isolated margin
    pos = create_futures_position(
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        quantity=1.0,
        entry_price=50000.0,
        leverage=10.0,
        margin_mode=MarginMode.ISOLATED,
        wallet_balance=10000.0,
    )
    print(f"Initial margin: {pos.initial_margin_required():.2f}")
    print(f"Maintenance margin: {pos.maintenance_margin_required():.2f}")
    print(f"Liquidation price: {pos.liquidation_price:.2f}")
    print(f"Margin ratio: {pos.margin_ratio():.2f}")
    
    # Price drops
    pos.update_mark_price(45000.0)
    print(f"At 45000: unrealized={pos.unrealized_pnl:.2f}, liquidation={pos.liquidation_price:.2f}, ratio={pos.margin_ratio():.2f}")
    
    # Liquidation check
    liquidated = pos.check_liquidation(44000.0)
    print(f"Liquidated at 44000: {liquidated}")
    
    # Test funding
    funding = pos.apply_funding(0.0001, 8.0)
    print(f"Funding (0.01%): {funding:.4f}")
    
    # Test cross margin account
    account = FuturesAccount(10000.0)
    pos1 = create_futures_position("BTCUSDT", PositionSide.LONG, 0.5, 50000, 10.0, MarginMode.CROSS, 10000)
    pos2 = create_futures_position("ETHUSDT", PositionSide.SHORT, 2.0, 3000, 5.0, MarginMode.CROSS, 10000)
    account.open_position(pos1)
    account.open_position(pos2)
    print(f"Account equity: {account.total_equity():.2f}")
    print(f"Margin ratio: {account.margin_ratio():.2f}")