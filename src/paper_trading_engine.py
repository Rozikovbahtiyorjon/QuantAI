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
class PaperAccountState:
    """
    Strict cash/fees/PnL ledger (R2.2).

    Runs in PARALLEL with the legacy balance math. Identity guaranteed:

        cash == initial_cash + realized_gross - fees_paid
        engine.balance == cash   (after every operation)

    Provides the foundation for future leverage / portfolio /
    margin features without changing legacy behavior.
    """

    initial_cash: float

    cash: float = 0.0
    fees_paid: float = 0.0
    realized_gross: float = 0.0

    # open position snapshot (mirrors PaperPosition)
    position_side: str | None = None
    position_qty: float = 0.0
    position_entry_price: float = 0.0
    position_entry_notional: float = 0.0

    def __post_init__(self) -> None:
        if self.cash == 0.0 and self.fees_paid == 0.0 and self.realized_gross == 0.0:
            self.cash = float(self.initial_cash)

    # -------------------------------------------------- ops

    def apply_open(self, side: str, price: float, quantity: float, entry_fee: float) -> None:
        self.position_side = side
        self.position_qty = float(quantity)
        self.position_entry_price = float(price)
        self.position_entry_notional = float(price) * float(quantity)
        self.cash -= float(entry_fee)
        self.fees_paid += float(entry_fee)

    def apply_close(self, gross_profit: float, exit_fee: float, fees_total: float) -> None:
        """
        fees_total = entry_fee + exit_fee of the closed trade.
        Entry fee was already deducted at apply_open.
        """
        self.cash += float(gross_profit)
        self.cash -= float(exit_fee)
        self.fees_paid += float(exit_fee)
        self.realized_gross += float(gross_profit)

        self.position_side = None
        self.position_qty = 0.0
        self.position_entry_price = 0.0
        self.position_entry_notional = 0.0

    def reset(self) -> None:
        self.cash = float(self.initial_cash)
        self.fees_paid = 0.0
        self.realized_gross = 0.0
        self.position_side = None
        self.position_qty = 0.0
        self.position_entry_price = 0.0
        self.position_entry_notional = 0.0

    # -------------------------------------------------- views

    @property
    def identity_gap(self) -> float:
        """|cash - (initial + realized_gross - fees)| ; must be ~0."""
        return abs(
            self.cash - (self.initial_cash + self.realized_gross - self.fees_paid)
        )

    def unrealized(self, last_price: float) -> float:
        """Mark-to-market PnL of the open position (0 when flat)."""
        if self.position_side is None or self.position_qty == 0:
            return 0.0
        if self.position_side == "LONG":
            return (float(last_price) - self.position_entry_price) * self.position_qty
        return (self.position_entry_price - float(last_price)) * self.position_qty

    def equity(self, last_price: float | None = None) -> float:
        """
        Flat: initial + realized net. With an open position and a
        last price supplied: marked-to-market equity.
        """
        equity_flat = self.initial_cash + (self.realized_gross - self.fees_paid)
        if last_price is not None and self.position_side is not None:
            return equity_flat + self.unrealized(last_price)
        return equity_flat


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

        # R2.2: strict parallel ledger (cash/fees/PnL identity).
        self.account_state = PaperAccountState(
            initial_cash=float(initial_balance)
        )

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

        self.account_state.apply_open(side, float(price), float(quantity), float(entry_fee))

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

        self.account_state.apply_close(
            gross_profit=float(gross_profit),
            exit_fee=float(exit_fee),
            fees_total=float(fees),
        )

        # R2.2 invariant: strict ledger mirrors legacy balance.
        assert abs(self.account_state.cash - self.balance) < 1e-6, (
            "PaperAccountState diverged from legacy balance: "
            f"{self.account_state.cash} != {self.balance}"
        )

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

        self.account_state.reset()


# =========================================================
# EXPORTS
# =========================================================

__all__ = [
    "PaperAccountState",
    "PaperPosition",
    "PaperTrade",
    "PaperTradingEngine",
]