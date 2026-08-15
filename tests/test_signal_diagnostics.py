"""
Tests for QuantAI Signal Diagnostics.

Tests:
- AI / ML / Fusion snapshots
- trade approval / blocking
- probability and confidence statistics
- trade outcomes
- signal distributions
- signal pairs
- Walk-Forward window statistics
- serialization
- reset
"""

from __future__ import annotations

from src.signal_diagnostics import (
    SignalDiagnostics,
    SignalSnapshot,
    TradeOutcome,
    SignalDiagnosticsSummary,
    create_signal_diagnostics,
)


# ============================================================
# HELPERS
# ============================================================


def make_diagnostics() -> SignalDiagnostics:
    return create_signal_diagnostics()


def record_buy_buy(
    diagnostics: SignalDiagnostics,
    *,
    approved: bool = True,
    window_id: int | None = None,
) -> SignalSnapshot:
    return diagnostics.record_signal(
        ai_signal="BUY",
        ai_confidence=0.80,
        ml_signal="BUY",
        ml_probability=0.90,
        ml_buy_probability=0.90,
        ml_sell_probability=0.05,
        ml_hold_probability=0.05,
        fusion_signal="BUY",
        combined_confidence=0.85,
        trade_approved=approved,
        reason="ML confirms BUY",
        window_id=window_id,
    )


def record_hold_hold(
    diagnostics: SignalDiagnostics,
    *,
    window_id: int | None = None,
) -> SignalSnapshot:
    return diagnostics.record_signal(
        ai_signal="HOLD",
        ai_confidence=0.55,
        ml_signal="HOLD",
        ml_probability=0.90,
        ml_buy_probability=0.05,
        ml_sell_probability=0.05,
        ml_hold_probability=0.90,
        fusion_signal="HOLD",
        combined_confidence=0.55,
        trade_approved=False,
        reason="AI HOLD + ML HOLD",
        window_id=window_id,
    )


# ============================================================
# DATAMODEL TESTS
# ============================================================


def test_signal_snapshot_defaults():
    snapshot = SignalSnapshot()

    assert snapshot.ai_signal == "HOLD"
    assert snapshot.ml_signal == "HOLD"
    assert snapshot.fusion_signal == "HOLD"

    assert snapshot.ai_confidence == 0.0
    assert snapshot.ml_probability == 0.0

    assert snapshot.trade_approved is False


def test_signal_snapshot_normalizes_signals():
    snapshot = SignalSnapshot(
        ai_signal="long",
        ml_signal="short",
        fusion_signal="neutral",
    )

    assert snapshot.ai_signal == "BUY"
    assert snapshot.ml_signal == "SELL"
    assert snapshot.fusion_signal == "HOLD"


def test_signal_snapshot_probability_conversion():
    snapshot = SignalSnapshot(
        ai_confidence=0.95,
        ml_probability=0.90,
        ml_buy_probability=0.80,
        ml_sell_probability=0.10,
        ml_hold_probability=0.10,
        combined_confidence=0.85,
    )

    assert snapshot.ai_confidence == 95.0
    assert snapshot.ml_probability == 90.0
    assert snapshot.ml_buy_probability == 80.0
    assert snapshot.ml_sell_probability == 10.0
    assert snapshot.ml_hold_probability == 10.0
    assert snapshot.combined_confidence == 85.0


def test_trade_outcome_creation():
    trade = TradeOutcome(
        signal="BUY",
        pnl=25.50,
        exit_reason="TAKE_PROFIT",
    )

    assert trade.signal == "BUY"
    assert trade.pnl == 25.50
    assert trade.exit_reason == "TAKE_PROFIT"


# ============================================================
# RECORD SIGNAL
# ============================================================


def test_record_signal_buy_buy():
    diagnostics = make_diagnostics()

    snapshot = record_buy_buy(diagnostics)

    assert isinstance(snapshot, SignalSnapshot)
    assert len(diagnostics.snapshots) == 1

    assert snapshot.ai_signal == "BUY"
    assert snapshot.ml_signal == "BUY"
    assert snapshot.fusion_signal == "BUY"

    assert snapshot.trade_approved is True


def test_record_signal_hold_hold():
    diagnostics = make_diagnostics()

    snapshot = record_hold_hold(diagnostics)

    assert snapshot.ai_signal == "HOLD"
    assert snapshot.ml_signal == "HOLD"
    assert snapshot.fusion_signal == "HOLD"

    assert snapshot.trade_approved is False


def test_ai_hold_blocks_ml_buy():
    diagnostics = make_diagnostics()

    snapshot = diagnostics.record_signal(
        ai_signal="HOLD",
        ai_confidence=0.60,
        ml_signal="BUY",
        ml_probability=0.90,
        ml_buy_probability=0.90,
        ml_sell_probability=0.05,
        ml_hold_probability=0.05,
        fusion_signal="HOLD",
        combined_confidence=0.60,
        trade_approved=False,
        reason="AI HOLD blocks ML BUY",
    )

    assert snapshot.ai_signal == "HOLD"
    assert snapshot.ml_signal == "BUY"
    assert snapshot.trade_approved is False
    assert "ML BUY" in snapshot.reason


def test_ml_hold_blocks_ai_buy():
    diagnostics = make_diagnostics()

    snapshot = diagnostics.record_signal(
        ai_signal="BUY",
        ai_confidence=0.65,
        ml_signal="HOLD",
        ml_probability=0.94,
        ml_buy_probability=0.04,
        ml_sell_probability=0.02,
        ml_hold_probability=0.94,
        fusion_signal="HOLD",
        combined_confidence=0.65,
        trade_approved=False,
        reason="ML HOLD blocks AI BUY",
    )

    assert snapshot.ai_signal == "BUY"
    assert snapshot.ml_signal == "HOLD"
    assert snapshot.trade_approved is False
    assert "ML HOLD" in snapshot.reason


def test_ai_buy_ml_buy_is_approved():
    diagnostics = make_diagnostics()

    snapshot = diagnostics.record_signal(
        ai_signal="BUY",
        ai_confidence=0.63,
        ml_signal="BUY",
        ml_probability=0.96,
        ml_buy_probability=0.96,
        ml_sell_probability=0.01,
        ml_hold_probability=0.03,
        fusion_signal="BUY",
        combined_confidence=0.76,
        trade_approved=True,
        reason="ML confirms BUY",
    )

    assert snapshot.fusion_signal == "BUY"
    assert snapshot.trade_approved is True


# ============================================================
# MULTIPLE SIGNALS
# ============================================================


def test_record_multiple_signals():
    diagnostics = make_diagnostics()

    record_buy_buy(diagnostics)
    record_hold_hold(diagnostics)
    record_buy_buy(diagnostics)

    assert len(diagnostics.snapshots) == 3


def test_signal_counts():
    diagnostics = make_diagnostics()

    record_buy_buy(diagnostics)
    record_hold_hold(diagnostics)
    record_hold_hold(diagnostics)

    summary = diagnostics.summarize()

    assert summary.total_snapshots == 3

    assert summary.ai_buy == 1
    assert summary.ai_hold == 2

    assert summary.ml_buy == 1
    assert summary.ml_hold == 2

    assert summary.fusion_buy == 1
    assert summary.fusion_hold == 2


# ============================================================
# APPROVAL / BLOCKING
# ============================================================


def test_approval_statistics():
    diagnostics = make_diagnostics()

    record_buy_buy(diagnostics)
    record_hold_hold(diagnostics)
    record_buy_buy(diagnostics)

    summary = diagnostics.summarize()

    assert summary.approved_trades == 2
    assert summary.blocked_trades == 1

    assert summary.approval_rate == 2 / 3 * 100


def test_approved_signals():
    diagnostics = make_diagnostics()

    record_buy_buy(diagnostics)
    record_hold_hold(diagnostics)

    approved = diagnostics.approved_signals()

    assert len(approved) == 1
    assert approved[0].trade_approved is True
    assert approved[0].fusion_signal == "BUY"


def test_blocked_signals():
    diagnostics = make_diagnostics()

    record_buy_buy(diagnostics)
    record_hold_hold(diagnostics)

    blocked = diagnostics.blocked_signals()

    assert len(blocked) == 1
    assert blocked[0].trade_approved is False
    assert blocked[0].fusion_signal == "HOLD"


# ============================================================
# BLOCKING REASONS
# ============================================================


def test_ai_hold_ml_buy_block_count():
    diagnostics = make_diagnostics()

    diagnostics.record_signal(
        ai_signal="HOLD",
        ml_signal="BUY",
        ml_probability=0.80,
        ml_buy_probability=0.80,
        ml_hold_probability=0.15,
        ml_sell_probability=0.05,
        fusion_signal="HOLD",
        trade_approved=False,
        reason="AI HOLD blocks ML BUY",
    )

    summary = diagnostics.summarize()

    assert summary.ai_hold_ml_buy_blocks == 1


def test_ml_hold_ai_buy_block_count():
    diagnostics = make_diagnostics()

    diagnostics.record_signal(
        ai_signal="BUY",
        ml_signal="HOLD",
        ml_probability=0.90,
        ml_buy_probability=0.05,
        ml_hold_probability=0.90,
        ml_sell_probability=0.05,
        fusion_signal="HOLD",
        trade_approved=False,
        reason="ML HOLD blocks AI BUY",
    )

    summary = diagnostics.summarize()

    assert summary.ml_hold_ai_buy_blocks == 1


def test_ai_hold_ml_sell_block_count():
    diagnostics = make_diagnostics()

    diagnostics.record_signal(
        ai_signal="HOLD",
        ml_signal="SELL",
        ml_probability=0.80,
        ml_buy_probability=0.05,
        ml_hold_probability=0.15,
        ml_sell_probability=0.80,
        fusion_signal="HOLD",
        trade_approved=False,
        reason="AI HOLD blocks ML SELL",
    )

    summary = diagnostics.summarize()

    assert summary.ai_hold_ml_sell_blocks == 1


def test_ml_hold_ai_sell_block_count():
    diagnostics = make_diagnostics()

    diagnostics.record_signal(
        ai_signal="SELL",
        ml_signal="HOLD",
        ml_probability=0.90,
        ml_buy_probability=0.05,
        ml_hold_probability=0.90,
        ml_sell_probability=0.05,
        fusion_signal="HOLD",
        trade_approved=False,
        reason="ML HOLD blocks AI SELL",
    )

    summary = diagnostics.summarize()

    assert summary.ml_hold_ai_sell_blocks == 1


# ============================================================
# CONFIDENCE / PROBABILITY
# ============================================================


def test_confidence_values_are_recorded():
    diagnostics = make_diagnostics()

    snapshot = diagnostics.record_signal(
        ai_signal="BUY",
        ai_confidence=0.80,
        ml_signal="BUY",
        ml_probability=0.90,
        ml_buy_probability=0.90,
        ml_sell_probability=0.05,
        ml_hold_probability=0.05,
        fusion_signal="BUY",
        combined_confidence=0.85,
        trade_approved=True,
    )

    assert snapshot.ai_confidence == 80.0
    assert snapshot.ml_probability == 90.0
    assert snapshot.combined_confidence == 85.0


def test_confidence_average():
    diagnostics = make_diagnostics()

    diagnostics.record_signal(
        ai_signal="BUY",
        ai_confidence=0.60,
        ml_signal="BUY",
        ml_probability=0.80,
        ml_buy_probability=0.80,
        ml_sell_probability=0.10,
        ml_hold_probability=0.10,
        fusion_signal="BUY",
        combined_confidence=0.70,
        trade_approved=True,
    )

    diagnostics.record_signal(
        ai_signal="BUY",
        ai_confidence=0.80,
        ml_signal="BUY",
        ml_probability=0.90,
        ml_buy_probability=0.90,
        ml_sell_probability=0.05,
        ml_hold_probability=0.05,
        fusion_signal="BUY",
        combined_confidence=0.85,
        trade_approved=True,
    )

    summary = diagnostics.summarize()

    assert summary.ai_confidence_avg == 70.0
    assert summary.combined_confidence_avg == 77.5


# ============================================================
# SIGNAL DISTRIBUTION
# ============================================================


def test_empty_signal_distribution():
    diagnostics = make_diagnostics()

    distribution = diagnostics.signal_distribution()

    assert distribution["AI"]["BUY"] == 0.0
    assert distribution["AI"]["SELL"] == 0.0
    assert distribution["AI"]["HOLD"] == 0.0

    assert distribution["ML"]["BUY"] == 0.0
    assert distribution["ML"]["SELL"] == 0.0
    assert distribution["ML"]["HOLD"] == 0.0


def test_signal_distribution():
    diagnostics = make_diagnostics()

    record_buy_buy(diagnostics)
    record_hold_hold(diagnostics)
    record_hold_hold(diagnostics)

    distribution = diagnostics.signal_distribution()

    assert distribution["AI"]["BUY"] == 1 / 3 * 100
    assert distribution["AI"]["HOLD"] == 2 / 3 * 100

    assert distribution["ML"]["BUY"] == 1 / 3 * 100
    assert distribution["ML"]["HOLD"] == 2 / 3 * 100

    assert distribution["FUSION"]["BUY"] == 1 / 3 * 100
    assert distribution["FUSION"]["HOLD"] == 2 / 3 * 100


# ============================================================
# ML PROBABILITY STATISTICS
# ============================================================


def test_ml_probability_statistics_empty():
    diagnostics = make_diagnostics()

    stats = diagnostics.ml_probability_statistics()

    assert stats["buy_avg"] == 0.0
    assert stats["sell_avg"] == 0.0
    assert stats["hold_avg"] == 0.0

    assert stats["buy_max"] == 0.0
    assert stats["sell_max"] == 0.0
    assert stats["hold_max"] == 0.0


def test_ml_probability_statistics():
    diagnostics = make_diagnostics()

    diagnostics.record_signal(
        ml_signal="BUY",
        ml_probability=0.90,
        ml_buy_probability=0.90,
        ml_sell_probability=0.05,
        ml_hold_probability=0.05,
    )

    diagnostics.record_signal(
        ml_signal="HOLD",
        ml_probability=0.80,
        ml_buy_probability=0.10,
        ml_sell_probability=0.10,
        ml_hold_probability=0.80,
    )

    stats = diagnostics.ml_probability_statistics()

    assert stats["buy_avg"] == 50.0
    assert stats["sell_avg"] == 7.5
    assert stats["hold_avg"] == 42.5

    assert stats["buy_max"] == 90.0
    assert stats["buy_min"] == 10.0

    assert stats["sell_max"] == 10.0
    assert stats["sell_min"] == 5.0

    assert stats["hold_max"] == 80.0
    assert stats["hold_min"] == 5.0


# ============================================================
# SIGNAL PAIRS
# ============================================================


def test_signal_pairs():
    diagnostics = make_diagnostics()

    diagnostics.record_signal(
        ai_signal="HOLD",
        ml_signal="BUY",
    )

    diagnostics.record_signal(
        ai_signal="HOLD",
        ml_signal="BUY",
    )

    diagnostics.record_signal(
        ai_signal="BUY",
        ml_signal="BUY",
    )

    diagnostics.record_signal(
        ai_signal="BUY",
        ml_signal="HOLD",
    )

    pairs = diagnostics.signal_pairs()

    assert pairs["HOLD+BUY"] == 2
    assert pairs["BUY+BUY"] == 1
    assert pairs["BUY+HOLD"] == 1


# ============================================================
# TRADE OUTCOMES
# ============================================================


def test_record_trade():
    diagnostics = make_diagnostics()

    outcome = diagnostics.record_trade(
        signal="BUY",
        pnl=25.50,
        exit_reason="TAKE_PROFIT",
        balance_before=1000.0,
        balance_after=1025.50,
    )

    assert isinstance(outcome, TradeOutcome)
    assert len(diagnostics.trade_outcomes) == 1

    assert outcome.signal == "BUY"
    assert outcome.pnl == 25.50
    assert outcome.balance_before == 1000.0
    assert outcome.balance_after == 1025.50


def test_trade_statistics():
    diagnostics = make_diagnostics()

    diagnostics.record_trade(
        signal="BUY",
        pnl=25.0,
        exit_reason="TAKE_PROFIT",
    )

    diagnostics.record_trade(
        signal="BUY",
        pnl=-10.0,
        exit_reason="STOP_LOSS",
    )

    diagnostics.record_trade(
        signal="SELL",
        pnl=0.0,
        exit_reason="FLAT",
    )

    summary = diagnostics.summarize()

    assert summary.total_trades == 3
    assert summary.winning_trades == 1
    assert summary.losing_trades == 1
    assert summary.flat_trades == 1

    assert summary.total_pnl == 15.0
    assert summary.average_trade_pnl == 5.0

    assert summary.stop_loss_count == 1
    assert summary.take_profit_count == 1

    assert summary.win_rate == 1 / 3 * 100


def test_trade_statistics_multiple_winners():
    diagnostics = make_diagnostics()

    diagnostics.record_trade(
        signal="BUY",
        pnl=10.0,
        exit_reason="TAKE_PROFIT",
    )

    diagnostics.record_trade(
        signal="BUY",
        pnl=20.0,
        exit_reason="TAKE_PROFIT",
    )

    diagnostics.record_trade(
        signal="SELL",
        pnl=-5.0,
        exit_reason="STOP_LOSS",
    )

    summary = diagnostics.summarize()

    assert summary.total_trades == 3
    assert summary.winning_trades == 2
    assert summary.losing_trades == 1

    assert summary.total_pnl == 25.0
    assert summary.average_trade_pnl == 25.0 / 3


# ============================================================
# WINDOW STATISTICS
# ============================================================


def test_window_statistics():
    diagnostics = make_diagnostics()

    record_buy_buy(
        diagnostics,
        window_id=1,
    )

    record_hold_hold(
        diagnostics,
        window_id=1,
    )

    record_buy_buy(
        diagnostics,
        window_id=2,
    )

    result = diagnostics.window_statistics()

    assert 1 in result
    assert 2 in result

    assert result[1]["snapshots"] == 2.0
    assert result[1]["approved"] == 1.0
    assert result[1]["approval_rate"] == 50.0

    assert result[1]["ml_buy"] == 1.0
    assert result[1]["ml_hold"] == 1.0

    assert result[1]["ai_buy"] == 1.0
    assert result[1]["ai_hold"] == 1.0

    assert result[2]["snapshots"] == 1.0
    assert result[2]["approved"] == 1.0
    assert result[2]["approval_rate"] == 100.0


def test_window_statistics_ignores_missing_window_id():
    diagnostics = make_diagnostics()

    record_buy_buy(diagnostics)

    diagnostics.record_signal(
        ai_signal="HOLD",
        ml_signal="HOLD",
        window_id=None,
    )

    result = diagnostics.window_statistics()

    assert result == {}


# ============================================================
# SUMMARY
# ============================================================


def test_empty_summary():
    diagnostics = make_diagnostics()

    summary = diagnostics.summarize()

    assert isinstance(summary, SignalDiagnosticsSummary)

    assert summary.total_snapshots == 0

    assert summary.approved_trades == 0
    assert summary.blocked_trades == 0

    assert summary.total_trades == 0
    assert summary.winning_trades == 0
    assert summary.losing_trades == 0
    assert summary.flat_trades == 0

    assert summary.total_pnl == 0.0
    assert summary.average_trade_pnl == 0.0

    assert summary.approval_rate == 0.0
    assert summary.ml_hold_rate == 0.0
    assert summary.ai_hold_rate == 0.0
    assert summary.fusion_hold_rate == 0.0
    assert summary.win_rate == 0.0


# ============================================================
# RESET
# ============================================================


def test_reset():
    diagnostics = make_diagnostics()

    record_buy_buy(diagnostics)

    diagnostics.record_trade(
        signal="BUY",
        pnl=10.0,
        exit_reason="TAKE_PROFIT",
    )

    assert len(diagnostics.snapshots) == 1
    assert len(diagnostics.trade_outcomes) == 1

    diagnostics.reset()

    assert len(diagnostics.snapshots) == 0
    assert len(diagnostics.trade_outcomes) == 0

    summary = diagnostics.summarize()

    assert summary.total_snapshots == 0
    assert summary.total_trades == 0


# ============================================================
# SERIALIZATION
# ============================================================


def test_to_dict_empty():
    diagnostics = make_diagnostics()

    data = diagnostics.to_dict()

    assert isinstance(data, dict)

    assert "summary" in data
    assert "signal_distribution" in data
    assert "ml_probability_statistics" in data
    assert "signal_pairs" in data
    assert "window_statistics" in data
    assert "snapshots" in data
    assert "trade_outcomes" in data

    assert data["snapshots"] == []
    assert data["trade_outcomes"] == []


def test_to_dict_contains_snapshot():
    diagnostics = make_diagnostics()

    record_buy_buy(diagnostics)

    data = diagnostics.to_dict()

    assert len(data["snapshots"]) == 1

    snapshot = data["snapshots"][0]

    assert snapshot["ai_signal"] == "BUY"
    assert snapshot["ml_signal"] == "BUY"
    assert snapshot["fusion_signal"] == "BUY"
    assert snapshot["trade_approved"] is True


def test_to_dict_contains_trade():
    diagnostics = make_diagnostics()

    diagnostics.record_trade(
        signal="BUY",
        pnl=15.0,
        exit_reason="TAKE_PROFIT",
    )

    data = diagnostics.to_dict()

    assert len(data["trade_outcomes"]) == 1

    trade = data["trade_outcomes"][0]

    assert trade["signal"] == "BUY"
    assert trade["pnl"] == 15.0
    assert trade["exit_reason"] == "TAKE_PROFIT"


# ============================================================
# CONFIDENCE RANGE
# ============================================================


def test_confidence_zero():
    diagnostics = make_diagnostics()

    snapshot = diagnostics.record_signal(
        ai_confidence=0.0,
        ml_probability=0.0,
        combined_confidence=0.0,
    )

    assert snapshot.ai_confidence == 0.0
    assert snapshot.ml_probability == 0.0
    assert snapshot.combined_confidence == 0.0


def test_confidence_one():
    diagnostics = make_diagnostics()

    snapshot = diagnostics.record_signal(
        ai_confidence=1.0,
        ml_probability=1.0,
        combined_confidence=1.0,
    )

    assert snapshot.ai_confidence == 100.0
    assert snapshot.ml_probability == 100.0
    assert snapshot.combined_confidence == 100.0


def test_confidence_percentage_input():
    diagnostics = make_diagnostics()

    snapshot = diagnostics.record_signal(
        ai_confidence=75.0,
        ml_probability=85.0,
        combined_confidence=90.0,
    )

    assert snapshot.ai_confidence == 75.0
    assert snapshot.ml_probability == 85.0
    assert snapshot.combined_confidence == 90.0


def test_confidence_above_range_is_clamped():
    diagnostics = make_diagnostics()

    snapshot = diagnostics.record_signal(
        ai_confidence=150.0,
        ml_probability=200.0,
        combined_confidence=-10.0,
    )

    assert snapshot.ai_confidence == 100.0
    assert snapshot.ml_probability == 100.0
    assert snapshot.combined_confidence == 0.0


# ============================================================
# LARGE DATASET
# ============================================================


def test_large_number_of_signals():
    diagnostics = make_diagnostics()

    for _ in range(100):
        diagnostics.record_signal(
            ai_signal="HOLD",
            ai_confidence=0.55,
            ml_signal="HOLD",
            ml_probability=0.90,
            ml_buy_probability=0.05,
            ml_sell_probability=0.05,
            ml_hold_probability=0.90,
            fusion_signal="HOLD",
            combined_confidence=0.55,
            trade_approved=False,
            reason="AI HOLD + ML HOLD",
        )

    summary = diagnostics.summarize()

    assert summary.total_snapshots == 100
    assert summary.ai_hold == 100
    assert summary.ml_hold == 100
    assert summary.fusion_hold == 100
    assert summary.blocked_trades == 100
    assert summary.approved_trades == 0

    assert summary.ai_hold_rate == 100.0
    assert summary.ml_hold_rate == 100.0
    assert summary.fusion_hold_rate == 100.0
    assert summary.approval_rate == 0.0