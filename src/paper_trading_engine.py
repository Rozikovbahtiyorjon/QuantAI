"""
=========================================================
QuantAI Professional v5
Paper Trading Engine

Virtual trading engine for paper trading.

This module does NOT:
    - connect to Binance
    - execute real orders
    - calculate indicators
    - generate Strategy signals
    - train ML models
    - send Telegram messages

It only simulates virtual trades.
=========================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


# =========================================================
# POSITION
# =========================================================

@dataclass
class PaperPosition:
    """
    Current virtual position.
    """

    side: str
    entry_price: float
    quantity: float
    entry_fee: float


# =========================================================
# CLOSED TRADE
# =========================================================

@dataclass
class PaperTrade:
    """
    Completed virtual trade.
    """

    side: str
    entry_price: float
    exit_price: float
    quantity: float

    gross_profit: float
    fees: float
    net_profit: float


# =========================================================
# ENGINE
# =========================================================

class PaperTradingEngine:
    """
    Virtual trading engine.

    Supports LONG and SHORT positions.
    """

    def __init__(
        self,
        initial_balance: float = 1000.0,
        commission: float = 0.0004,
    ) -> None:

        if initial_balance <= 0:
            raise ValueError(
                "initial_balance must be greater than zero."
            )

        if commission < 0:
            raise ValueError(
                "commission cannot be negative."
            )

        self.initial_balance = float(
            initial_balance
        )

        self.balance = float(
            initial_balance
        )

        self.commission = float(
            commission
        )

        self.position: PaperPosition | None = None

        self.trade_history: List[PaperTrade] = []

    # =====================================================
    # POSITION STATE
    # =====================================================

    @property
    def has_position(self) -> bool:
        """
        Return True when a virtual position is open.
        """

        return self.position is not None

    # =====================================================
    # EQUITY
    # =====================================================

    @property
    def realized_profit(self) -> float:
        """
        Total realized net profit.
        """

        return round(
            sum(
                trade.net_profit
                for trade in self.trade_history
            ),
            8,
        )

    # =====================================================
    # OPEN POSITION
    # =====================================================

    def open_position(
        self,
        side: str,
        price: float,
        quantity: float,
    ) -> PaperPosition:
        """
        Open a virtual LONG or SHORT position.
        """

        side = side.upper()

        if side not in {"LONG", "SHORT"}:
            raise ValueError(
                "side must be LONG or SHORT."
            )

        if price <= 0:
            raise ValueError(
                "price must be greater than zero."
            )

        if quantity <= 0:
            raise ValueError(
                "quantity must be greater than zero."
            )

        if self.has_position:
            raise RuntimeError(
                "A paper position is already open."
            )

        notional = price * quantity

        entry_fee = (
            notional
            * self.commission
        )

        if entry_fee > self.balance:
            raise ValueError(
                "Insufficient balance for entry fee."
            )

        self.balance -= entry_fee

        self.position = PaperPosition(
            side=side,
            entry_price=float(price),
            quantity=float(quantity),
            entry_fee=float(entry_fee),
        )

        return self.position

    # =====================================================
    # CLOSE POSITION
    # =====================================================

    def close_position(
        self,
        price: float,
    ) -> PaperTrade:
        """
        Close the current virtual position.
        """

        if not self.has_position:
            raise RuntimeError(
                "No paper position is open."
            )

        if price <= 0:
            raise ValueError(
                "price must be greater than zero."
            )

        position = self.position

        entry_price = position.entry_price
        quantity = position.quantity

        if position.side == "LONG":

            gross_profit = (
                price - entry_price
            ) * quantity

        else:

            gross_profit = (
                entry_price - price
            ) * quantity

        exit_notional = (
            price * quantity
        )

        exit_fee = (
            exit_notional
            * self.commission
        )

        fees = (
            position.entry_fee
            + exit_fee
        )

        net_profit = (
            gross_profit
            - fees
        )

        self.balance += (
            gross_profit
            + exit_notional
            - exit_notional
        )

        self.balance -= exit_fee

        trade = PaperTrade(
            side=position.side,
            entry_price=entry_price,
            exit_price=float(price),
            quantity=quantity,
            gross_profit=float(gross_profit),
            fees=float(fees),
            net_profit=float(net_profit),
        )

        self.trade_history.append(
            trade
        )

        self.position = None

        return trade

    # =====================================================
    # RESET
    # =====================================================

    def reset(self) -> None:
        """
        Reset the paper account.
        """

        self.balance = (
            self.initial_balance
        )

        self.position = None

        self.trade_history.clear()


# =========================================================
# EXPORTS
# =========================================================

__all__ = [
    "PaperPosition",
    "PaperTrade",
    "PaperTradingEngine",
]