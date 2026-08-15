from __future__ import annotations

from dataclasses import dataclass

from src.signal_quality_analyzer import (
    SignalQualityAnalyzer,
    SignalQualitySnapshot,
    analyze_signal_quality,
)


@dataclass
class Result:
    signal: str
    confidence: float
    ai_signal: str = "HOLD"
    ai_confidence: float = 0.0
    ml_signal: str = "HOLD"
    ml_probability: float = 0.0
    trade_approved: bool = False
    timestamp: str | None = None


def test_empty_analyzer_returns_zero_snapshot():
    snapshot = SignalQualityAnalyzer().analyze()

    assert isinstance(snapshot, SignalQualitySnapshot)
    assert snapshot.total_signals == 0
    assert snapshot.trade_rate == 0.0
    assert snapshot.hold_rate == 0.0


def test_signal_distribution_and_rates():
    analyzer = SignalQualityAnalyzer(
        min_sample_size=1,
    )

    analyzer.add(
        "BUY",
        confidence=80,
    )

    analyzer.add(
        "SELL",
        confidence=70,
    )

    analyzer.add(
        "HOLD",
        confidence=50,
    )

    snapshot = analyzer.analyze()

    assert snapshot.total_signals == 3
    assert snapshot.buy_signals == 1
    assert snapshot.sell_signals == 1
    assert snapshot.hold_signals == 1
    assert snapshot.trade_signals == 2
    assert snapshot.buy_rate == 33.3333
    assert snapshot.sell_rate == 33.3333
    assert snapshot.hold_rate == 33.3333
    assert snapshot.trade_rate == 66.6667


def test_signal_aliases_are_normalized():
    analyzer = SignalQualityAnalyzer()

    analyzer.add(
        "LONG",
        confidence=70,
    )

    analyzer.add(
        "SHORT",
        confidence=70,
    )

    analyzer.add(
        "WAIT",
        confidence=40,
    )

    snapshot = analyzer.analyze()

    assert snapshot.buy_signals == 1
    assert snapshot.sell_signals == 1
    assert snapshot.hold_signals == 1


def test_result_object_is_supported():
    analyzer = SignalQualityAnalyzer()

    analyzer.add_result(
        Result(
            signal="BUY",
            confidence=82,
            ai_signal="BUY",
            ai_confidence=85,
            ml_signal="BUY",
            ml_probability=78,
            trade_approved=True,
        )
    )

    snapshot = analyzer.analyze()

    assert snapshot.total_signals == 1
    assert snapshot.average_confidence == 82.0
    assert snapshot.average_trade_confidence == 82.0
    assert snapshot.approved_trade_rate == 100.0
    assert snapshot.ai_ml_agreement_rate == 100.0
    assert snapshot.ai_ml_conflict_rate == 0.0


def test_high_hold_rate_is_flagged_after_minimum_sample():
    analyzer = SignalQualityAnalyzer(
        hold_warning_rate=70,
        min_trade_rate=30,
        max_trade_rate=80,
        min_sample_size=10,
    )

    for _ in range(8):
        analyzer.add(
            "HOLD",
            confidence=45,
            ml_signal="HOLD",
        )

    analyzer.add(
        "BUY",
        confidence=70,
        ml_signal="BUY",
    )

    analyzer.add(
        "SELL",
        confidence=72,
        ml_signal="SELL",
    )

    snapshot = analyzer.analyze()

    assert snapshot.hold_rate == 80.0
    assert "HIGH_HOLD_RATE" in snapshot.diagnostic_flags
    assert "UNDERTRADING" in snapshot.diagnostic_flags


def test_undertrading_and_overtrading_are_distinguished():
    under = SignalQualityAnalyzer(
        min_trade_rate=30,
        max_trade_rate=70,
        min_sample_size=5,
    )

    for _ in range(4):
        under.add("HOLD")

    under.add("BUY")

    assert "UNDERTRADING" in under.analyze().diagnostic_flags

    over = SignalQualityAnalyzer(
        min_trade_rate=20,
        max_trade_rate=60,
        min_sample_size=5,
    )

    for _ in range(4):
        over.add("BUY")

    over.add("HOLD")

    assert "OVERTRADING" in over.analyze().diagnostic_flags


def test_ai_ml_agreement_and_conflict_are_measured():
    analyzer = SignalQualityAnalyzer()

    analyzer.add(
        "BUY",
        ai_signal="BUY",
        ml_signal="BUY",
    )

    analyzer.add(
        "HOLD",
        ai_signal="BUY",
        ml_signal="SELL",
    )

    analyzer.add(
        "SELL",
        ai_signal="SELL",
        ml_signal="HOLD",
    )

    snapshot = analyzer.analyze()

    assert snapshot.ai_ml_agreement_rate == 33.3333
    assert snapshot.ai_ml_conflict_rate == 66.6667
    assert snapshot.ml_hold_rate == 33.3333


def test_invalid_signal_is_flagged():
    analyzer = SignalQualityAnalyzer(
        min_sample_size=1,
    )

    analyzer.add("BUY")
    analyzer.add("UNKNOWN")

    snapshot = analyzer.analyze()

    assert snapshot.invalid_signal_count == 1
    assert "INVALID_SIGNALS" in snapshot.diagnostic_flags


def test_missing_trade_approval_data_is_reported():
    analyzer = SignalQualityAnalyzer(
        min_sample_size=2,
    )

    analyzer.add(
        "BUY",
        confidence=80,
    )

    analyzer.add(
        "SELL",
        confidence=75,
    )

    snapshot = analyzer.analyze()

    assert (
        "MISSING_TRADE_APPROVAL_DATA"
        in snapshot.diagnostic_flags
    )


def test_grouped_summary_by_timeframe():
    analyzer = SignalQualityAnalyzer(
        min_sample_size=1,
    )

    analyzer.add(
        "BUY",
        timeframe="15m",
    )

    analyzer.add(
        "HOLD",
        timeframe="15m",
    )

    analyzer.add(
        "SELL",
        timeframe="1h",
    )

    grouped = analyzer.grouped_summary(
        "timeframe",
    )

    assert set(grouped) == {
        "15m",
        "1h",
    }

    assert grouped["15m"].total_signals == 2
    assert grouped["15m"].trade_signals == 1
    assert grouped["1h"].sell_signals == 1


def test_add_many_accepts_mappings_and_result_objects():
    analyzer = SignalQualityAnalyzer()

    records = [
        {
            "signal": "BUY",
            "confidence": 80,
            "ai_signal": "BUY",
            "ml_signal": "BUY",
            "trade_approved": True,
        },
        Result(
            signal="SELL",
            confidence=75,
            ai_signal="SELL",
            ml_signal="SELL",
            trade_approved=True,
        ),
    ]

    assert analyzer.add_many(records) == 2
    assert analyzer.analyze().trade_signals == 2


def test_convenience_function():
    snapshot = analyze_signal_quality(
        [
            {
                "signal": "BUY",
                "confidence": 80,
            },
            {
                "signal": "HOLD",
                "confidence": 40,
            },
        ]
    )

    assert snapshot.total_signals == 2
    assert snapshot.trade_rate == 50.0


def test_invalid_configuration_is_rejected():
    configurations = (
        {"min_confidence": -1},
        {"hold_warning_rate": 101},
        {
            "min_trade_rate": 80,
            "max_trade_rate": 20,
        },
        {"min_sample_size": 0},
    )

    for kwargs in configurations:
        try:
            SignalQualityAnalyzer(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(
                "Expected ValueError."
            )


def test_clear_removes_records():
    analyzer = SignalQualityAnalyzer()

    analyzer.add(
        "BUY",
        confidence=80,
    )

    assert analyzer.analyze().total_signals == 1

    analyzer.clear()

    assert analyzer.analyze().total_signals == 0
    assert analyzer.records == ()