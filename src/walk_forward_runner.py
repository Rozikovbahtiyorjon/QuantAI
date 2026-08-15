"""
QuantAI Walk-Forward Runner

Main entry point for Walk-Forward validation.

Responsibilities:
    - load market data;
    - calculate indicators;
    - run Walk-Forward Engine;
    - collect signal diagnostics;
    - collect trade diagnostics when available;
    - print Walk-Forward performance report;
    - print Signal Diagnostics report.

This module does NOT contain trading execution logic.
"""

from __future__ import annotations

from typing import Any

from config.settings import (
    SYMBOL,
    TIMEFRAME,
    LIMIT,
    INITIAL_BALANCE,
)

from src.data_loader import load_binance_data
from src.indicators import add_indicators

from src.walk_forward_engine import (
    WalkForwardEngine,
)

from src.walk_forward_report import (
    WalkForwardReport,
    create_walk_forward_report,
)

from src.signal_diagnostics import (
    SignalDiagnostics,
    create_signal_diagnostics,
)


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_SIZE = 500
TEST_SIZE = 300
STEP_SIZE = 100


# ============================================================
# HELPERS
# ============================================================

def _safe_get(
    obj: Any,
    attribute: str,
    default: Any = None,
) -> Any:
    """
    Safely read an attribute from an object.

    This allows the runner to work with slightly different
    WalkForwardResult implementations without changing the
    Walk-Forward Engine itself.
    """

    return getattr(obj, attribute, default)


def _record_result_diagnostics(
    diagnostics: SignalDiagnostics,
    result: Any,
) -> None:
    """
    Import diagnostic information from WalkForwardResult
    when the result exposes signal/trade information.

    The function intentionally does not require a specific
    WalkForwardResult implementation.

    Supported optional attributes:

        signal_snapshots
        snapshots
        trade_outcomes
        trades

    If these attributes do not exist, diagnostics simply
    remain empty.
    """

    # --------------------------------------------------------
    # SIGNAL SNAPSHOTS
    # --------------------------------------------------------

    snapshots = _safe_get(
        result,
        "signal_snapshots",
        None,
    )

    if snapshots is None:
        snapshots = _safe_get(
            result,
            "snapshots",
            None,
        )

    if snapshots is not None:

        for snapshot in snapshots:

            if isinstance(snapshot, dict):

                diagnostics.record_signal(
                    ai_signal=snapshot.get(
                        "ai_signal",
                        "HOLD",
                    ),
                    ai_confidence=snapshot.get(
                        "ai_confidence",
                        0.0,
                    ),
                    ml_signal=snapshot.get(
                        "ml_signal",
                        "HOLD",
                    ),
                    ml_probability=snapshot.get(
                        "ml_probability",
                        0.0,
                    ),
                    ml_buy_probability=snapshot.get(
                        "ml_buy_probability",
                        0.0,
                    ),
                    ml_sell_probability=snapshot.get(
                        "ml_sell_probability",
                        0.0,
                    ),
                    ml_hold_probability=snapshot.get(
                        "ml_hold_probability",
                        0.0,
                    ),
                    fusion_signal=snapshot.get(
                        "fusion_signal",
                        "HOLD",
                    ),
                    combined_confidence=snapshot.get(
                        "combined_confidence",
                        0.0,
                    ),
                    trade_approved=snapshot.get(
                        "trade_approved",
                        False,
                    ),
                    reason=snapshot.get(
                        "reason",
                        "",
                    ),
                    window_id=snapshot.get(
                        "window_id",
                        None,
                    ),
                    timestamp=snapshot.get(
                        "timestamp",
                        None,
                    ),
                )

            else:

                diagnostics.record_signal(
                    ai_signal=_safe_get(
                        snapshot,
                        "ai_signal",
                        "HOLD",
                    ),
                    ai_confidence=_safe_get(
                        snapshot,
                        "ai_confidence",
                        0.0,
                    ),
                    ml_signal=_safe_get(
                        snapshot,
                        "ml_signal",
                        "HOLD",
                    ),
                    ml_probability=_safe_get(
                        snapshot,
                        "ml_probability",
                        0.0,
                    ),
                    ml_buy_probability=_safe_get(
                        snapshot,
                        "ml_buy_probability",
                        0.0,
                    ),
                    ml_sell_probability=_safe_get(
                        snapshot,
                        "ml_sell_probability",
                        0.0,
                    ),
                    ml_hold_probability=_safe_get(
                        snapshot,
                        "ml_hold_probability",
                        0.0,
                    ),
                    fusion_signal=_safe_get(
                        snapshot,
                        "fusion_signal",
                        "HOLD",
                    ),
                    combined_confidence=_safe_get(
                        snapshot,
                        "combined_confidence",
                        0.0,
                    ),
                    trade_approved=_safe_get(
                        snapshot,
                        "trade_approved",
                        False,
                    ),
                    reason=_safe_get(
                        snapshot,
                        "reason",
                        "",
                    ),
                    window_id=_safe_get(
                        snapshot,
                        "window_id",
                        None,
                    ),
                    timestamp=_safe_get(
                        snapshot,
                        "timestamp",
                        None,
                    ),
                )

    # --------------------------------------------------------
    # TRADE OUTCOMES
    # --------------------------------------------------------

    trades = _safe_get(
        result,
        "trade_outcomes",
        None,
    )

    if trades is None:
        trades = _safe_get(
            result,
            "trades",
            None,
        )

    if trades is not None:

        for trade in trades:

            if isinstance(trade, dict):

                diagnostics.record_trade(
                    signal=trade.get(
                        "signal",
                        "HOLD",
                    ),
                    pnl=trade.get(
                        "pnl",
                        0.0,
                    ),
                    exit_reason=trade.get(
                        "exit_reason",
                        "",
                    ),
                    balance_before=trade.get(
                        "balance_before",
                        None,
                    ),
                    balance_after=trade.get(
                        "balance_after",
                        None,
                    ),
                    window_id=trade.get(
                        "window_id",
                        None,
                    ),
                    timestamp=trade.get(
                        "timestamp",
                        None,
                    ),
                )

            else:

                diagnostics.record_trade(
                    signal=_safe_get(
                        trade,
                        "signal",
                        "HOLD",
                    ),
                    pnl=_safe_get(
                        trade,
                        "pnl",
                        0.0,
                    ),
                    exit_reason=_safe_get(
                        trade,
                        "exit_reason",
                        "",
                    ),
                    balance_before=_safe_get(
                        trade,
                        "balance_before",
                        None,
                    ),
                    balance_after=_safe_get(
                        trade,
                        "balance_after",
                        None,
                    ),
                    window_id=_safe_get(
                        trade,
                        "window_id",
                        None,
                    ),
                    timestamp=_safe_get(
                        trade,
                        "timestamp",
                        None,
                    ),
                )


def _print_signal_diagnostics(
    diagnostics: SignalDiagnostics,
) -> None:
    """
    Print a compact Signal Diagnostics report.
    """

    summary = diagnostics.summarize()

    print()
    print("=" * 70)
    print("QUANTAI SIGNAL DIAGNOSTICS")
    print("=" * 70)

    print(
        f"Total snapshots       : "
        f"{summary.total_snapshots}"
    )

    print(
        f"Approved trades       : "
        f"{summary.approved_trades}"
    )

    print(
        f"Blocked trades        : "
        f"{summary.blocked_trades}"
    )

    print(
        f"Approval rate         : "
        f"{summary.approval_rate:.2f}%"
    )

    print("-" * 70)
    print("AI SIGNALS")
    print("-" * 70)

    print(
        f"BUY                   : "
        f"{summary.ai_buy}"
    )

    print(
        f"SELL                  : "
        f"{summary.ai_sell}"
    )

    print(
        f"HOLD                  : "
        f"{summary.ai_hold}"
    )

    print(
        f"HOLD rate             : "
        f"{summary.ai_hold_rate:.2f}%"
    )

    print("-" * 70)
    print("ML SIGNALS")
    print("-" * 70)

    print(
        f"BUY                   : "
        f"{summary.ml_buy}"
    )

    print(
        f"SELL                  : "
        f"{summary.ml_sell}"
    )

    print(
        f"HOLD                  : "
        f"{summary.ml_hold}"
    )

    print(
        f"HOLD rate             : "
        f"{summary.ml_hold_rate:.2f}%"
    )

    print("-" * 70)
    print("FUSION SIGNALS")
    print("-" * 70)

    print(
        f"BUY                   : "
        f"{summary.fusion_buy}"
    )

    print(
        f"SELL                  : "
        f"{summary.fusion_sell}"
    )

    print(
        f"HOLD                  : "
        f"{summary.fusion_hold}"
    )

    print(
        f"HOLD rate             : "
        f"{summary.fusion_hold_rate:.2f}%"
    )

    print("-" * 70)
    print("BLOCKING REASONS")
    print("-" * 70)

    print(
        f"AI HOLD -> ML BUY     : "
        f"{summary.ai_hold_ml_buy_blocks}"
    )

    print(
        f"ML HOLD -> AI BUY     : "
        f"{summary.ml_hold_ai_buy_blocks}"
    )

    print(
        f"AI HOLD -> ML SELL    : "
        f"{summary.ai_hold_ml_sell_blocks}"
    )

    print(
        f"ML HOLD -> AI SELL    : "
        f"{summary.ml_hold_ai_sell_blocks}"
    )

    print("-" * 70)
    print("PROBABILITY / CONFIDENCE")
    print("-" * 70)

    print(
        f"ML BUY avg            : "
        f"{summary.ml_buy_probability_avg:.2f}%"
    )

    print(
        f"ML SELL avg           : "
        f"{summary.ml_sell_probability_avg:.2f}%"
    )

    print(
        f"ML HOLD avg           : "
        f"{summary.ml_hold_probability_avg:.2f}%"
    )

    print(
        f"AI confidence avg     : "
        f"{summary.ai_confidence_avg:.2f}%"
    )

    print(
        f"Combined confidence   : "
        f"{summary.combined_confidence_avg:.2f}%"
    )

    print("-" * 70)
    print("TRADE OUTCOMES")
    print("-" * 70)

    print(
        f"Total trades          : "
        f"{summary.total_trades}"
    )

    print(
        f"Winning trades        : "
        f"{summary.winning_trades}"
    )

    print(
        f"Losing trades         : "
        f"{summary.losing_trades}"
    )

    print(
        f"Flat trades           : "
        f"{summary.flat_trades}"
    )

    print(
        f"Win rate              : "
        f"{summary.win_rate:.2f}%"
    )

    print(
        f"Total PnL             : "
        f"{summary.total_pnl:.2f}"
    )

    print(
        f"Average trade PnL     : "
        f"{summary.average_trade_pnl:.2f}"
    )

    print(
        f"Stop Loss             : "
        f"{summary.stop_loss_count}"
    )

    print(
        f"Take Profit           : "
        f"{summary.take_profit_count}"
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print("=" * 70)
    print("QUANTAI WALK-FORWARD VALIDATION")
    print("=" * 70)

    print(
        f"Symbol       : {SYMBOL}"
    )

    print(
        f"Timeframe    : {TIMEFRAME}"
    )

    print(
        f"Candles      : {LIMIT}"
    )

    print(
        f"Train size   : {TRAIN_SIZE}"
    )

    print(
        f"Test size    : {TEST_SIZE}"
    )

    print(
        f"Step size    : {STEP_SIZE}"
    )

    print(
        f"Initial      : {INITIAL_BALANCE:.2f}"
    )

    print("=" * 70)
    print()

    # --------------------------------------------------------
    # MARKET DATA
    # --------------------------------------------------------

    print("Loading Binance data...")

    df = load_binance_data(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        limit=LIMIT,
    )

    print(
        f"Downloaded candles : {len(df)}"
    )

    if df.empty:
        raise ValueError(
            "No market data received."
        )

    print()

    # --------------------------------------------------------
    # INDICATORS
    # --------------------------------------------------------

    print("Calculating indicators...")

    df = add_indicators(df)

    print(
        f"Indicator rows     : {len(df)}"
    )

    print()

    # --------------------------------------------------------
    # SIGNAL DIAGNOSTICS
    # --------------------------------------------------------

    print(
        "Initializing Signal Diagnostics..."
    )

    diagnostics = create_signal_diagnostics()

    print(
        "Signal Diagnostics ready."
    )

    print()

    # --------------------------------------------------------
    # WALK-FORWARD ENGINE
    # --------------------------------------------------------

    print(
        "Starting Walk-Forward engine..."
    )

    engine = WalkForwardEngine(
        train_size=TRAIN_SIZE,
        test_size=TEST_SIZE,
        step_size=STEP_SIZE,
        initial_balance=INITIAL_BALANCE,
    )

    result = engine.run(df)

    # --------------------------------------------------------
    # DIAGNOSTICS COLLECTION
    # --------------------------------------------------------

    _record_result_diagnostics(
        diagnostics,
        result,
    )

    # --------------------------------------------------------
    # RESULT INFORMATION
    # --------------------------------------------------------

    completed_windows = _safe_get(
        result,
        "completed_windows",
        None,
    )

    if completed_windows is None:
        completed_windows = _safe_get(
            result,
            "windows",
            None,
        )

    total_trades = _safe_get(
        result,
        "total_trades",
        None,
    )

    if total_trades is None:
        total_trades = _safe_get(
            result,
            "trades",
            0,
        )

    print()

    if completed_windows is not None:

        print(
            f"Completed windows  : "
            f"{completed_windows}"
        )

    else:

        print(
            f"Completed windows  : "
            f"{total_trades}"
        )

    print()

    # --------------------------------------------------------
    # WALK-FORWARD REPORT
    # --------------------------------------------------------

    summary = create_walk_forward_report(
        result
    )

    WalkForwardReport.print_report(
        summary
    )

    # --------------------------------------------------------
    # SIGNAL DIAGNOSTICS REPORT
    # --------------------------------------------------------

    _print_signal_diagnostics(
        diagnostics
    )

    print()

    print(
        "Walk-Forward validation completed."
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()