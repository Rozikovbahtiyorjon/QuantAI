from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from src.order_flow_intelligence import OrderFlowSignal
from src.unified_market_intelligence import (
    UnifiedMarketIntelligenceLayer,
    UnifiedMarketIntelligenceSignal,
)


@dataclass(frozen=True)
class OrderFlowIntegrationResult:
    order_flow_score: float
    unified_signal: UnifiedMarketIntelligenceSignal


class OrderFlowUnifiedMarketIntegration:
    """
    Adapter that converts OrderFlowIntelligence output into the
    normalized score expected by UnifiedMarketIntelligenceLayer.

    The adapter does not modify either engine and keeps the integration
    deterministic and auditable.
    """

    def __init__(
        self,
        unified_layer: UnifiedMarketIntelligenceLayer | None = None,
    ) -> None:
        self.unified_layer = (
            unified_layer
            if unified_layer is not None
            else UnifiedMarketIntelligenceLayer()
        )

    @staticmethod
    def order_flow_score(
        signal: OrderFlowSignal,
    ) -> float:
        if not isinstance(signal, OrderFlowSignal):
            raise TypeError(
                "signal must be an OrderFlowSignal instance."
            )

        if signal.context == "BALANCED":
            return 0.5

        return max(
            0.0,
            min(
                1.0,
                0.5 + 0.5 * signal.pressure,
            ),
        )

    def evaluate(
        self,
        signal: OrderFlowSignal,
        components: Mapping[str, float] | None = None,
        *,
        regime: str = "UNKNOWN",
        event_risk: float | None = None,
    ) -> OrderFlowIntegrationResult:
        if components is not None and not isinstance(
            components,
            Mapping,
        ):
            raise TypeError(
                "components must be a mapping or None."
            )

        score = self.order_flow_score(signal)

        merged = dict(
            components
            if components is not None
            else {}
        )

        merged["order_flow"] = score

        unified_signal = self.unified_layer.evaluate(
            merged,
            regime=regime,
            event_risk=event_risk,
        )

        return OrderFlowIntegrationResult(
            order_flow_score=score,
            unified_signal=unified_signal,
        )


__all__ = [
    "OrderFlowIntegrationResult",
    "OrderFlowUnifiedMarketIntegration",
]