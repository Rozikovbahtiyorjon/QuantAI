from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.order_flow_intelligence import OrderFlowSignal

if TYPE_CHECKING:
    from src.strategy import SignalResult


@dataclass(frozen=True)
class OrderFlowStrategyDecision:
    signal: str
    approved: bool
    strategy_signal: str
    strategy_approved: bool
    order_flow_context: str
    order_flow_score: float
    reason: str


class OrderFlowStrategyIntegration:
    """
    Deterministic decision gate between Strategy and OrderFlow.

    Strategy remains the primary decision source.

    OrderFlow can:
        - confirm a directional strategy signal;
        - block a conflicting directional signal.

    OrderFlow cannot:
        - create a trade from Strategy HOLD;
        - override a strategy decision that was not approved.

    The integration intentionally avoids importing SignalResult at
    runtime in order to prevent a circular dependency.
    """

    def __init__(
        self,
        conflict_threshold: float = 0.15,
    ) -> None:
        if (
            isinstance(conflict_threshold, bool)
            or not isinstance(
                conflict_threshold,
                (int, float),
            )
        ):
            raise TypeError(
                "conflict_threshold must be a finite number."
            )

        threshold = float(conflict_threshold)

        if threshold <= 0.0:
            raise ValueError(
                "conflict_threshold must be greater than zero."
            )

        if threshold > 1.0:
            raise ValueError(
                "conflict_threshold must be at most 1.0."
            )

        self.conflict_threshold = threshold

    @staticmethod
    def _validate_strategy_result(
        strategy_result: Any,
    ) -> None:
        """
        Validate the Strategy result without importing SignalResult
        at runtime.

        AttributeError is intentionally preserved for compatibility
        with the existing Strategy test contract.
        """

        if strategy_result is None:
            raise AttributeError(
                "strategy_result must provide "
                "signal and trade_approved attributes."
            )

        if not hasattr(
            strategy_result,
            "signal",
        ):
            raise AttributeError(
                "strategy_result must provide "
                "signal and trade_approved attributes."
            )

        if not hasattr(
            strategy_result,
            "trade_approved",
        ):
            raise AttributeError(
                "strategy_result must provide "
                "signal and trade_approved attributes."
            )

    @staticmethod
    def _normalize_signal(
        signal: Any,
    ) -> str:
        if signal is None:
            return "HOLD"

        value = str(
            signal
        ).strip().upper()

        aliases = {
            "LONG": "BUY",
            "SHORT": "SELL",
            "NEUTRAL": "HOLD",
            "WAIT": "HOLD",
            "NONE": "HOLD",
        }

        return aliases.get(
            value,
            value,
        )

    @staticmethod
    def _order_flow_score(
        order_flow_signal: OrderFlowSignal,
    ) -> float:
        """
        Convert OrderFlow pressure from [-1, 1] to [0, 1].

        No rounding is performed here. The exact value must be preserved
        when passed to the Strategy layer.
        """

        return max(
            0.0,
            min(
                1.0,
                0.5
                + 0.5
                * float(
                    order_flow_signal.pressure
                ),
            ),
        )

    def evaluate(
        self,
        strategy_result: SignalResult,
        order_flow_signal: OrderFlowSignal,
    ) -> OrderFlowStrategyDecision:
        self._validate_strategy_result(
            strategy_result
        )

        if not isinstance(
            order_flow_signal,
            OrderFlowSignal,
        ):
            raise TypeError(
                "order_flow_signal must be an OrderFlowSignal instance."
            )

        strategy_signal = self._normalize_signal(
            strategy_result.signal
        )

        strategy_approved = bool(
            strategy_result.trade_approved
        )

        order_flow_context = (
            order_flow_signal.context
        )

        order_flow_score = self._order_flow_score(
            order_flow_signal
        )

        # ----------------------------------------------------
        # STRATEGY PRIMARY GATE
        # ----------------------------------------------------

        if (
            strategy_signal == "HOLD"
            or not strategy_approved
        ):
            return OrderFlowStrategyDecision(
                signal="HOLD",
                approved=False,
                strategy_signal=strategy_signal,
                strategy_approved=strategy_approved,
                order_flow_context=order_flow_context,
                order_flow_score=order_flow_score,
                reason=(
                    "Strategy HOLD or strategy trade "
                    "approval is false."
                ),
            )

        # ----------------------------------------------------
        # BUY
        # ----------------------------------------------------

        if strategy_signal == "BUY":
            if (
                float(
                    order_flow_signal.pressure
                )
                <= -self.conflict_threshold
            ):
                return OrderFlowStrategyDecision(
                    signal="HOLD",
                    approved=False,
                    strategy_signal=strategy_signal,
                    strategy_approved=True,
                    order_flow_context=order_flow_context,
                    order_flow_score=order_flow_score,
                    reason=(
                        "OrderFlow conflicts with BUY "
                        "strategy signal."
                    ),
                )

            return OrderFlowStrategyDecision(
                signal="BUY",
                approved=True,
                strategy_signal=strategy_signal,
                strategy_approved=True,
                order_flow_context=order_flow_context,
                order_flow_score=order_flow_score,
                reason=(
                    "OrderFlow confirms or does not "
                    "conflict with BUY strategy signal."
                ),
            )

        # ----------------------------------------------------
        # SELL
        # ----------------------------------------------------

        if strategy_signal == "SELL":
            if (
                float(
                    order_flow_signal.pressure
                )
                >= self.conflict_threshold
            ):
                return OrderFlowStrategyDecision(
                    signal="HOLD",
                    approved=False,
                    strategy_signal=strategy_signal,
                    strategy_approved=True,
                    order_flow_context=order_flow_context,
                    order_flow_score=order_flow_score,
                    reason=(
                        "OrderFlow conflicts with SELL "
                        "strategy signal."
                    ),
                )

            return OrderFlowStrategyDecision(
                signal="SELL",
                approved=True,
                strategy_signal=strategy_signal,
                strategy_approved=True,
                order_flow_context=order_flow_context,
                order_flow_score=order_flow_score,
                reason=(
                    "OrderFlow confirms or does not "
                    "conflict with SELL strategy signal."
                ),
            )

        # ----------------------------------------------------
        # UNKNOWN SIGNAL
        # ----------------------------------------------------

        return OrderFlowStrategyDecision(
            signal="HOLD",
            approved=False,
            strategy_signal=strategy_signal,
            strategy_approved=strategy_approved,
            order_flow_context=order_flow_context,
            order_flow_score=order_flow_score,
            reason=(
                f"Unsupported strategy signal: "
                f"{strategy_signal}"
            ),
        )


__all__ = [
    "OrderFlowStrategyDecision",
    "OrderFlowStrategyIntegration",
]