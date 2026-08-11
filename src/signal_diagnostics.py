"""
QuantAI Signal Diagnostics
===========================

Standalone diagnostic module for AI + ML trading signals.

Purpose:
- collect AI / ML / Fusion signal information;
- calculate signal statistics;
- track trade approvals and blocking reasons;
- track trade outcomes;
- analyze ML probability distribution;
- analyze AI confidence;
- keep diagnostics independent from trading execution logic.

This module does NOT execute trades.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional


# ============================================================
# DATA MODELS
# ============================================================


@dataclass
class SignalSnapshot:
    """
    One diagnostic snapshot of an AI + ML decision.
    """

    ai_signal: str = "HOLD"
    ai_confidence: float = 0.0

    ml_signal: str = "HOLD"
    ml_probability: float = 0.0

    ml_buy_probability: float = 0.0
    ml_sell_probability: float = 0.0
    ml_hold_probability: float = 0.0

    fusion_signal: str = "HOLD"
    combined_confidence: float = 0.0

    trade_approved: bool = False
    reason: str = ""

    window_id: Optional[int] = None
    timestamp: Optional[Any] = None

    def __post_init__(self) -> None:
        self.ai_signal = _normalize_signal(self.ai_signal)
        self.ml_signal = _normalize_signal(self.ml_signal)
        self.fusion_signal = _normalize_signal(self.fusion_signal)

        self.ai_confidence = _clamp_probability(
            self.ai_confidence
        )

        self.ml_probability = _clamp_probability(
            self.ml_probability
        )

        self.ml_buy_probability = _clamp_probability(
            self.ml_buy_probability
        )

        self.ml_sell_probability = _clamp_probability(
            self.ml_sell_probability
        )

        self.ml_hold_probability = _clamp_probability(
            self.ml_hold_probability
        )

        self.combined_confidence = _clamp_probability(
            self.combined_confidence
        )

        self.trade_approved = bool(
            self.trade_approved
        )

        if self.reason is None:
            self.reason = ""


@dataclass
class TradeOutcome:
    """
    Result of an executed trade.
    """

    signal: str
    pnl: float

    exit_reason: str = ""

    balance_before: Optional[float] = None
    balance_after: Optional[float] = None

    window_id: Optional[int] = None
    timestamp: Optional[Any] = None

    def __post_init__(self) -> None:
        self.signal = _normalize_signal(self.signal)
        self.pnl = float(self.pnl)

        if self.exit_reason is None:
            self.exit_reason = ""


@dataclass
class SignalDiagnosticsSummary:
    """
    Aggregated diagnostic statistics.
    """

    total_snapshots: int = 0

    ai_buy: int = 0
    ai_sell: int = 0
    ai_hold: int = 0

    ml_buy: int = 0
    ml_sell: int = 0
    ml_hold: int = 0

    fusion_buy: int = 0
    fusion_sell: int = 0
    fusion_hold: int = 0

    approved_trades: int = 0
    blocked_trades: int = 0

    ai_hold_ml_buy_blocks: int = 0
    ml_hold_ai_buy_blocks: int = 0
    ai_hold_ml_sell_blocks: int = 0
    ml_hold_ai_sell_blocks: int = 0

    ml_buy_probability_avg: float = 0.0
    ml_sell_probability_avg: float = 0.0
    ml_hold_probability_avg: float = 0.0

    ai_confidence_avg: float = 0.0
    combined_confidence_avg: float = 0.0

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    flat_trades: int = 0

    total_pnl: float = 0.0
    average_trade_pnl: float = 0.0

    stop_loss_count: int = 0
    take_profit_count: int = 0

    @property
    def approval_rate(self) -> float:
        if self.total_snapshots == 0:
            return 0.0

        return (
            self.approved_trades
            / self.total_snapshots
            * 100.0
        )

    @property
    def ml_hold_rate(self) -> float:
        if self.total_snapshots == 0:
            return 0.0

        return (
            self.ml_hold
            / self.total_snapshots
            * 100.0
        )

    @property
    def ai_hold_rate(self) -> float:
        if self.total_snapshots == 0:
            return 0.0

        return (
            self.ai_hold
            / self.total_snapshots
            * 100.0
        )

    @property
    def fusion_hold_rate(self) -> float:
        if self.total_snapshots == 0:
            return 0.0

        return (
            self.fusion_hold
            / self.total_snapshots
            * 100.0
        )

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0

        return (
            self.winning_trades
            / self.total_trades
            * 100.0
        )


# ============================================================
# HELPERS
# ============================================================


def _clamp_probability(value: Any) -> float:
    """
    Convert a probability/confidence value into [0, 100].

    The diagnostic module expects percentages.

    Examples:
        0.95 -> 95.0
        95.0 -> 95.0
    """

    if value is None:
        return 0.0

    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0

    if 0.0 <= number <= 1.0:
        number *= 100.0

    return max(0.0, min(100.0, number))


def _normalize_signal(signal: Any) -> str:
    """
    Normalize signal names.
    """

    if signal is None:
        return "HOLD"

    value = str(signal).strip().upper()

    aliases = {
        "LONG": "BUY",
        "SHORT": "SELL",
        "NEUTRAL": "HOLD",
        "WAIT": "HOLD",
        "NONE": "HOLD",
    }

    return aliases.get(value, value)


def _average(values: Iterable[float]) -> float:
    values = list(values)

    if not values:
        return 0.0

    return sum(values) / len(values)


# ============================================================
# SIGNAL DIAGNOSTICS
# ============================================================


class SignalDiagnostics:
    """
    Collects and analyzes AI + ML signal decisions.

    This class is intentionally independent from:
        - Strategy
        - ML Engine
        - Trade Engine
        - WalkForward Engine

    It can therefore be introduced without changing
    existing trading behaviour.
    """

    def __init__(self) -> None:
        self.snapshots: list[SignalSnapshot] = []
        self.trade_outcomes: list[TradeOutcome] = []

    # --------------------------------------------------------
    # RECORD SIGNAL
    # --------------------------------------------------------

    def record_signal(
        self,
        *,
        ai_signal: str = "HOLD",
        ai_confidence: float = 0.0,
        ml_signal: str = "HOLD",
        ml_probability: float = 0.0,
        ml_buy_probability: float = 0.0,
        ml_sell_probability: float = 0.0,
        ml_hold_probability: float = 0.0,
        fusion_signal: str = "HOLD",
        combined_confidence: float = 0.0,
        trade_approved: bool = False,
        reason: str = "",
        window_id: Optional[int] = None,
        timestamp: Optional[Any] = None,
    ) -> SignalSnapshot:
        """
        Record one AI + ML decision.
        """

        snapshot = SignalSnapshot(
            ai_signal=ai_signal,
            ai_confidence=ai_confidence,
            ml_signal=ml_signal,
            ml_probability=ml_probability,
            ml_buy_probability=ml_buy_probability,
            ml_sell_probability=ml_sell_probability,
            ml_hold_probability=ml_hold_probability,
            fusion_signal=fusion_signal,
            combined_confidence=combined_confidence,
            trade_approved=trade_approved,
            reason=reason,
            window_id=window_id,
            timestamp=timestamp,
        )

        self.snapshots.append(snapshot)

        return snapshot

    # --------------------------------------------------------
    # RECORD TRADE
    # --------------------------------------------------------

    def record_trade(
        self,
        *,
        signal: str,
        pnl: float,
        exit_reason: str = "",
        balance_before: Optional[float] = None,
        balance_after: Optional[float] = None,
        window_id: Optional[int] = None,
        timestamp: Optional[Any] = None,
    ) -> TradeOutcome:
        """
        Record an executed trade outcome.
        """

        outcome = TradeOutcome(
            signal=signal,
            pnl=pnl,
            exit_reason=exit_reason,
            balance_before=balance_before,
            balance_after=balance_after,
            window_id=window_id,
            timestamp=timestamp,
        )

        self.trade_outcomes.append(outcome)

        return outcome

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    def summarize(self) -> SignalDiagnosticsSummary:
        """
        Build aggregated diagnostic statistics.
        """

        summary = SignalDiagnosticsSummary()

        summary.total_snapshots = len(
            self.snapshots
        )

        for snapshot in self.snapshots:

            # AI
            if snapshot.ai_signal == "BUY":
                summary.ai_buy += 1

            elif snapshot.ai_signal == "SELL":
                summary.ai_sell += 1

            else:
                summary.ai_hold += 1

            # ML
            if snapshot.ml_signal == "BUY":
                summary.ml_buy += 1

            elif snapshot.ml_signal == "SELL":
                summary.ml_sell += 1

            else:
                summary.ml_hold += 1

            # Fusion
            if snapshot.fusion_signal == "BUY":
                summary.fusion_buy += 1

            elif snapshot.fusion_signal == "SELL":
                summary.fusion_sell += 1

            else:
                summary.fusion_hold += 1

            # Approval
            if snapshot.trade_approved:
                summary.approved_trades += 1
            else:
                summary.blocked_trades += 1

            # Blocking reasons
            reason = snapshot.reason.upper()

            if (
                snapshot.ai_signal == "HOLD"
                and snapshot.ml_signal == "BUY"
            ):
                summary.ai_hold_ml_buy_blocks += 1

            if (
                snapshot.ml_signal == "HOLD"
                and snapshot.ai_signal == "BUY"
            ):
                summary.ml_hold_ai_buy_blocks += 1

            if (
                snapshot.ai_signal == "HOLD"
                and snapshot.ml_signal == "SELL"
            ):
                summary.ai_hold_ml_sell_blocks += 1

            if (
                snapshot.ml_signal == "HOLD"
                and snapshot.ai_signal == "SELL"
            ):
                summary.ml_hold_ai_sell_blocks += 1

        # Probability / confidence statistics
        summary.ml_buy_probability_avg = _average(
            snapshot.ml_buy_probability
            for snapshot in self.snapshots
        )

        summary.ml_sell_probability_avg = _average(
            snapshot.ml_sell_probability
            for snapshot in self.snapshots
        )

        summary.ml_hold_probability_avg = _average(
            snapshot.ml_hold_probability
            for snapshot in self.snapshots
        )

        summary.ai_confidence_avg = _average(
            snapshot.ai_confidence
            for snapshot in self.snapshots
        )

        summary.combined_confidence_avg = _average(
            snapshot.combined_confidence
            for snapshot in self.snapshots
        )

        # Trade statistics
        summary.total_trades = len(
            self.trade_outcomes
        )

        for trade in self.trade_outcomes:

            if trade.pnl > 0:
                summary.winning_trades += 1

            elif trade.pnl < 0:
                summary.losing_trades += 1

            else:
                summary.flat_trades += 1

            summary.total_pnl += trade.pnl

            exit_reason = (
                trade.exit_reason.upper()
            )

            if "STOP" in exit_reason:
                summary.stop_loss_count += 1

            if "TAKE" in exit_reason or "TP" in exit_reason:
                summary.take_profit_count += 1

        if summary.total_trades > 0:
            summary.average_trade_pnl = (
                summary.total_pnl
                / summary.total_trades
            )

        return summary

    # --------------------------------------------------------
    # SIGNAL DISTRIBUTION
    # --------------------------------------------------------

    def signal_distribution(self) -> dict[str, dict[str, float]]:
        """
        Return percentage distribution of AI,
        ML and Fusion signals.
        """

        total = len(self.snapshots)

        if total == 0:
            return {
                "AI": {
                    "BUY": 0.0,
                    "SELL": 0.0,
                    "HOLD": 0.0,
                },
                "ML": {
                    "BUY": 0.0,
                    "SELL": 0.0,
                    "HOLD": 0.0,
                },
                "FUSION": {
                    "BUY": 0.0,
                    "SELL": 0.0,
                    "HOLD": 0.0,
                },
            }

        summary = self.summarize()

        return {
            "AI": {
                "BUY": summary.ai_buy / total * 100.0,
                "SELL": summary.ai_sell / total * 100.0,
                "HOLD": summary.ai_hold / total * 100.0,
            },
            "ML": {
                "BUY": summary.ml_buy / total * 100.0,
                "SELL": summary.ml_sell / total * 100.0,
                "HOLD": summary.ml_hold / total * 100.0,
            },
            "FUSION": {
                "BUY": summary.fusion_buy / total * 100.0,
                "SELL": summary.fusion_sell / total * 100.0,
                "HOLD": summary.fusion_hold / total * 100.0,
            },
        }

    # --------------------------------------------------------
    # ML PROBABILITY DIAGNOSTICS
    # --------------------------------------------------------

    def ml_probability_statistics(
        self,
    ) -> dict[str, float]:
        """
        Analyze ML probability distribution.
        """

        if not self.snapshots:
            return {
                "buy_avg": 0.0,
                "sell_avg": 0.0,
                "hold_avg": 0.0,
                "buy_max": 0.0,
                "sell_max": 0.0,
                "hold_max": 0.0,
                "buy_min": 0.0,
                "sell_min": 0.0,
                "hold_min": 0.0,
            }

        buy = [
            s.ml_buy_probability
            for s in self.snapshots
        ]

        sell = [
            s.ml_sell_probability
            for s in self.snapshots
        ]

        hold = [
            s.ml_hold_probability
            for s in self.snapshots
        ]

        return {
            "buy_avg": _average(buy),
            "sell_avg": _average(sell),
            "hold_avg": _average(hold),
            "buy_max": max(buy),
            "sell_max": max(sell),
            "hold_max": max(hold),
            "buy_min": min(buy),
            "sell_min": min(sell),
            "hold_min": min(hold),
        }

    # --------------------------------------------------------
    # APPROVED SIGNALS
    # --------------------------------------------------------

    def approved_signals(
        self,
    ) -> list[SignalSnapshot]:
        """
        Return only approved trade signals.
        """

        return [
            snapshot
            for snapshot in self.snapshots
            if snapshot.trade_approved
        ]

    # --------------------------------------------------------
    # BLOCKED SIGNALS
    # --------------------------------------------------------

    def blocked_signals(
        self,
    ) -> list[SignalSnapshot]:
        """
        Return signals that were blocked.
        """

        return [
            snapshot
            for snapshot in self.snapshots
            if not snapshot.trade_approved
        ]

    # --------------------------------------------------------
    # SIGNAL PAIRS
    # --------------------------------------------------------

    def signal_pairs(
        self,
    ) -> dict[str, int]:
        """
        Count AI/ML signal combinations.

        Example:
            HOLD + BUY
            BUY + BUY
            BUY + HOLD
        """

        result: dict[str, int] = {}

        for snapshot in self.snapshots:

            key = (
                f"{snapshot.ai_signal}"
                f"+"
                f"{snapshot.ml_signal}"
            )

            result[key] = (
                result.get(key, 0) + 1
            )

        return result

    # --------------------------------------------------------
    # WINDOW STATISTICS
    # --------------------------------------------------------

    def window_statistics(
        self,
    ) -> dict[int, dict[str, float]]:
        """
        Aggregate diagnostics by Walk-Forward window.
        """

        windows: dict[
            int,
            list[SignalSnapshot],
        ] = {}

        for snapshot in self.snapshots:

            if snapshot.window_id is None:
                continue

            windows.setdefault(
                snapshot.window_id,
                [],
            ).append(snapshot)

        result: dict[
            int,
            dict[str, float],
        ] = {}

        for window_id, snapshots in windows.items():

            total = len(snapshots)

            if total == 0:
                continue

            approved = sum(
                1
                for snapshot in snapshots
                if snapshot.trade_approved
            )

            result[window_id] = {
                "snapshots": float(total),
                "approved": float(approved),
                "approval_rate": (
                    approved / total * 100.0
                ),
                "ml_buy": float(
                    sum(
                        1
                        for snapshot in snapshots
                        if snapshot.ml_signal == "BUY"
                    )
                ),
                "ml_sell": float(
                    sum(
                        1
                        for snapshot in snapshots
                        if snapshot.ml_signal == "SELL"
                    )
                ),
                "ml_hold": float(
                    sum(
                        1
                        for snapshot in snapshots
                        if snapshot.ml_signal == "HOLD"
                    )
                ),
                "ai_buy": float(
                    sum(
                        1
                        for snapshot in snapshots
                        if snapshot.ai_signal == "BUY"
                    )
                ),
                "ai_sell": float(
                    sum(
                        1
                        for snapshot in snapshots
                        if snapshot.ai_signal == "SELL"
                    )
                ),
                "ai_hold": float(
                    sum(
                        1
                        for snapshot in snapshots
                        if snapshot.ai_signal == "HOLD"
                    )
                ),
            }

        return result

    # --------------------------------------------------------
    # EXPORT
    # --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """
        Convert diagnostics to a serializable dictionary.
        """

        summary = self.summarize()

        return {
            "summary": {
                "total_snapshots":
                    summary.total_snapshots,

                "approved_trades":
                    summary.approved_trades,

                "blocked_trades":
                    summary.blocked_trades,

                "approval_rate":
                    summary.approval_rate,

                "ai_buy":
                    summary.ai_buy,

                "ai_sell":
                    summary.ai_sell,

                "ai_hold":
                    summary.ai_hold,

                "ml_buy":
                    summary.ml_buy,

                "ml_sell":
                    summary.ml_sell,

                "ml_hold":
                    summary.ml_hold,

                "fusion_buy":
                    summary.fusion_buy,

                "fusion_sell":
                    summary.fusion_sell,

                "fusion_hold":
                    summary.fusion_hold,

                "ml_hold_rate":
                    summary.ml_hold_rate,

                "ai_hold_rate":
                    summary.ai_hold_rate,

                "fusion_hold_rate":
                    summary.fusion_hold_rate,

                "ai_hold_ml_buy_blocks":
                    summary.ai_hold_ml_buy_blocks,

                "ml_hold_ai_buy_blocks":
                    summary.ml_hold_ai_buy_blocks,

                "ai_hold_ml_sell_blocks":
                    summary.ai_hold_ml_sell_blocks,

                "ml_hold_ai_sell_blocks":
                    summary.ml_hold_ai_sell_blocks,

                "ai_confidence_avg":
                    summary.ai_confidence_avg,

                "combined_confidence_avg":
                    summary.combined_confidence_avg,

                "ml_buy_probability_avg":
                    summary.ml_buy_probability_avg,

                "ml_sell_probability_avg":
                    summary.ml_sell_probability_avg,

                "ml_hold_probability_avg":
                    summary.ml_hold_probability_avg,

                "total_trades":
                    summary.total_trades,

                "winning_trades":
                    summary.winning_trades,

                "losing_trades":
                    summary.losing_trades,

                "flat_trades":
                    summary.flat_trades,

                "win_rate":
                    summary.win_rate,

                "total_pnl":
                    summary.total_pnl,

                "average_trade_pnl":
                    summary.average_trade_pnl,

                "stop_loss_count":
                    summary.stop_loss_count,

                "take_profit_count":
                    summary.take_profit_count,
            },

            "signal_distribution":
                self.signal_distribution(),

            "ml_probability_statistics":
                self.ml_probability_statistics(),

            "signal_pairs":
                self.signal_pairs(),

            "window_statistics":
                self.window_statistics(),

            "snapshots": [
                {
                    "ai_signal":
                        snapshot.ai_signal,

                    "ai_confidence":
                        snapshot.ai_confidence,

                    "ml_signal":
                        snapshot.ml_signal,

                    "ml_probability":
                        snapshot.ml_probability,

                    "ml_buy_probability":
                        snapshot.ml_buy_probability,

                    "ml_sell_probability":
                        snapshot.ml_sell_probability,

                    "ml_hold_probability":
                        snapshot.ml_hold_probability,

                    "fusion_signal":
                        snapshot.fusion_signal,

                    "combined_confidence":
                        snapshot.combined_confidence,

                    "trade_approved":
                        snapshot.trade_approved,

                    "reason":
                        snapshot.reason,

                    "window_id":
                        snapshot.window_id,

                    "timestamp":
                        snapshot.timestamp,
                }
                for snapshot in self.snapshots
            ],

            "trade_outcomes": [
                {
                    "signal":
                        trade.signal,

                    "pnl":
                        trade.pnl,

                    "exit_reason":
                        trade.exit_reason,

                    "balance_before":
                        trade.balance_before,

                    "balance_after":
                        trade.balance_after,

                    "window_id":
                        trade.window_id,

                    "timestamp":
                        trade.timestamp,
                }
                for trade in self.trade_outcomes
            ],
        }

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    def reset(self) -> None:
        """
        Clear all diagnostic data.
        """

        self.snapshots.clear()
        self.trade_outcomes.clear()


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================


def create_signal_diagnostics() -> SignalDiagnostics:
    """
    Factory function for creating a diagnostics instance.
    """

    return SignalDiagnostics()
