"""ARCHIVED (Phase 0): legacy monolithic-strategy contour tests.

These tests validated the pre-refactor API of src.strategy
(module-level evaluate_market / predict_ml / AI_MODEL contract).
The current modular pipeline (SignalGenerator) no longer exposes
that contract, so these tests cannot pass against live code.

Preserved verbatim for historical reference. Do not run in CI.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

import src.strategy as strategy
from src.confidence_engine import ConfidenceEngine
from src.order_book_market_data import (
    OrderBookLevel,
    OrderBookSnapshot,
)
from src.order_flow_intelligence import (
    OrderFlowIntelligenceEngine,
)
from src.strategy import (
    MarketEngine,
    generate_signal_result,
)


def make_market_data(
    rows: int = 4,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [
                100.0 + index
                for index in range(rows)
            ],
            "high": [
                101.0 + index
                for index in range(rows)
            ],
            "low": [
                99.0 + index
                for index in range(rows)
            ],
            "close": [
                100.0 + index
                for index in range(rows)
            ],
            "atr": [
                1.0
                for _ in range(rows)
            ],
            "volume": [
                1000.0
                for _ in range(rows)
            ],
        }
    )


def make_confidence_result(
    decision: str = "BUY",
    score: float = 2.0,
):
    engine = ConfidenceEngine()

    engine.add_component(
        "trend",
        score,
    )

    engine.add_component(
        "momentum",
        score,
    )

    engine.add_component(
        "volume",
        score,
    )

    engine.add_component(
        "volatility",
        score,
    )

    engine.add_component(
        "structure",
        score,
    )

    result = engine.evaluate()

    assert result.decision == decision

    return result


def make_order_flow_snapshot(
    bid_amount: float,
    ask_amount: float,
    timestamp: int = 1,
) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        symbol="BTC/USDT",
        timestamp=timestamp,
        bids=(
            OrderBookLevel(
                price=100.0,
                amount=bid_amount,
            ),
        ),
        asks=(
            OrderBookLevel(
                price=101.0,
                amount=ask_amount,
            ),
        ),
    )


@dataclass
class FakeModel:
    classes_: tuple[int, ...] = (
        0,
        1,
        2,
    )

    def predict_proba(
        self,
        X: pd.DataFrame,
    ):
        assert isinstance(
            X,
            pd.DataFrame,
        )

        assert len(X) == 1

        return [
            [
                0.05,
                0.05,
                0.90,
            ]
        ]


def test_data_features_ml_boundary_is_connected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    market_data = make_market_data()

    calls: list[str] = []

    def fake_build_features(
        df: pd.DataFrame,
    ) -> dict[str, float]:
        calls.append("features")

        assert isinstance(
            df,
            pd.DataFrame,
        )

        assert not df.empty

        return {
            "feature_1": 1.0,
            "feature_2": 2.0,
        }

    monkeypatch.setattr(
        strategy,
        "build_features",
        fake_build_features,
    )

    monkeypatch.setattr(
        strategy,
        "AI_MODEL",
        FakeModel(),
    )

    signal, probability, probabilities = (
        strategy.predict_ml(
            market_data,
        )
    )

    assert calls == [
        "features"
    ]

    assert signal == "BUY"

    assert probability == pytest.approx(
        90.0
    )

    assert probabilities[0] == pytest.approx(
        5.0
    )

    assert probabilities[1] == pytest.approx(
        5.0
    )

    assert probabilities[2] == pytest.approx(
        90.0
    )


def test_strategy_risk_order_flow_contour(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    market_data = make_market_data(
        rows=1
    )

    confidence_result = (
        make_confidence_result(
            decision="BUY",
            score=2.0,
        )
    )

    evaluate_calls: list[str] = []

    def fake_evaluate_market(
        df: pd.DataFrame,
    ) -> MarketEngine:
        evaluate_calls.append(
            "confidence"
        )

        return MarketEngine(
            trend_score=2.0,
            momentum_score=2.0,
            volume_score=2.0,
            volatility_score=2.0,
            confidence_result=confidence_result,
            reasons=[
                "Synthetic deterministic "
                "contour validation."
            ],
        )

    ml_calls: list[str] = []

    def fake_predict_ml(
        df: pd.DataFrame,
    ):
        ml_calls.append(
            "ml"
        )

        return (
            "BUY",
            90.0,
            {
                0: 5.0,
                1: 5.0,
                2: 90.0,
            },
        )

    monkeypatch.setattr(
        strategy,
        "evaluate_market",
        fake_evaluate_market,
    )

    monkeypatch.setattr(
        strategy,
        "predict_ml",
        fake_predict_ml,
    )

    order_flow_engine = (
        OrderFlowIntelligenceEngine(
            pressure_threshold=0.15,
        )
    )

    order_flow_signal = (
        order_flow_engine.update(
            make_order_flow_snapshot(
                bid_amount=20.0,
                ask_amount=2.0,
            )
        )
    )

    result = generate_signal_result(
        market_data,
        order_flow_signal=order_flow_signal,
    )

    assert evaluate_calls == [
        "confidence"
    ]

    assert ml_calls == [
        "ml"
    ]

    assert result.signal == "BUY"


def test_conflicting_order_flow_blocks_trade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    market_data = make_market_data(
        rows=1
    )

    confidence_result = (
        make_confidence_result(
            decision="BUY",
            score=2.0,
        )
    )

    monkeypatch.setattr(
        strategy,
        "evaluate_market",
        lambda df: MarketEngine(
            confidence_result=confidence_result,
            reasons=[
                "Contour validation."
            ],
        ),
    )

    monkeypatch.setattr(
        strategy,
        "predict_ml",
        lambda df: (
            "BUY",
            90.0,
            {
                0: 5.0,
                1: 5.0,
                2: 90.0,
            },
        ),
    )

    order_flow_signal = (
        OrderFlowIntelligenceEngine(
            pressure_threshold=0.15,
        ).update(
            make_order_flow_snapshot(
                bid_amount=1.0,
                ask_amount=20.0,
            )
        )
    )

    result = generate_signal_result(
        market_data,
        order_flow_signal=order_flow_signal,
    )

    assert result.signal == "HOLD"

    assert (
        "conflicts with BUY"
        in result.order_flow_reason
    )
