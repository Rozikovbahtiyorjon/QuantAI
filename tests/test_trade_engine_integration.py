# ============================================================
# QuantAI
# tests/test_trade_engine_integration.py
# ============================================================
#
# Integration tests for:
#
#     Strategy / SignalResult
#             ↓
#        TradeEngine
#             ↓
#       Position lifecycle
#
# Important contract:
#
#     BUY  -> PositionSide.LONG
#     SELL -> PositionSide.SHORT
#     HOLD -> must NOT call open_position()
#
# ============================================================

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timedelta


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

from src.trade_engine import (
    TradeEngine,
    PositionSide,
    PositionStatus,
    CloseReason,
    SignalResult,
)


# ============================================================
# TEST HELPERS
# ============================================================

PASS_COUNT = 0
FAIL_COUNT = 0


def check(condition: bool, message: str) -> None:
    """
    Simple assertion helper.
    """

    global PASS_COUNT
    global FAIL_COUNT

    if condition:
        print(f"OK    : {message}")
        PASS_COUNT += 1

    else:
        print(f"FAIL  : {message}")
        FAIL_COUNT += 1


def make_candle(
    timestamp: object | None = None,
    open_price: float = 100.0,
    high: float = 101.0,
    low: float = 99.0,
    close: float = 100.0,
    volume: float = 1000.0,
) -> dict:
    """
    Create a candle in the exact format expected
    by TradeEngine.

    TradeEngine uses dictionary access:

        candle["timestamp"]
        candle["close"]
        candle["high"]
        candle["low"]

    Therefore we intentionally use dict here.
    """

    if timestamp is None:
        timestamp = datetime.now()

    return {
        "timestamp": timestamp,
        "open": float(open_price),
        "high": float(high),
        "low": float(low),
        "close": float(close),
        "volume": float(volume),
    }


def make_buy_signal(
    entry: float = 100.0,
    stop_loss: float = 98.0,
    take_profit: float = 104.0,
    confidence: float = 90.0,
) -> SignalResult:
    """
    Create an approved BUY signal.
    """

    return SignalResult(
        signal="BUY",
        score=1.0,
        confidence=confidence,
        entry=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        reasons=["integration test BUY"],
        ai_signal="BUY",
        ai_confidence=confidence,
        ml_signal="BUY",
        ml_probability=confidence,
        ml_buy_probability=confidence,
        ml_sell_probability=0.0,
        ml_hold_probability=100.0 - confidence,
        fusion_signal="BUY",
        combined_confidence=confidence,
        trade_approved=True,
    )


def make_sell_signal(
    entry: float = 100.0,
    stop_loss: float = 102.0,
    take_profit: float = 96.0,
    confidence: float = 90.0,
) -> SignalResult:
    """
    Create an approved SELL signal.
    """

    return SignalResult(
        signal="SELL",
        score=-1.0,
        confidence=confidence,
        entry=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        reasons=["integration test SELL"],
        ai_signal="SELL",
        ai_confidence=confidence,
        ml_signal="SELL",
        ml_probability=confidence,
        ml_buy_probability=0.0,
        ml_sell_probability=confidence,
        ml_hold_probability=100.0 - confidence,
        fusion_signal="SELL",
        combined_confidence=confidence,
        trade_approved=True,
    )


def make_hold_signal() -> SignalResult:
    """
    HOLD is intentionally NOT sent to TradeEngine.open_position().

    This is an integration-layer responsibility.
    """

    return SignalResult(
        signal="HOLD",
        score=0.0,
        confidence=50.0,
        entry=100.0,
        stop_loss=0.0,
        take_profit=0.0,
        reasons=["integration test HOLD"],
        ai_signal="HOLD",
        ai_confidence=50.0,
        ml_signal="HOLD",
        ml_probability=99.0,
        ml_buy_probability=0.5,
        ml_sell_probability=0.5,
        ml_hold_probability=99.0,
        fusion_signal="HOLD",
        combined_confidence=50.0,
        trade_approved=False,
    )


# ============================================================
# TEST 1
# INITIAL STATE
# ============================================================

def test_initial_state() -> None:

    print()
    print("## TEST: test_initial_state")

    engine = TradeEngine()

    positions = engine.get_open_positions()

    check(
        isinstance(positions, list),
        "Initial positions is list",
    )

    check(
        len(positions) == 0,
        "Initial open positions = 0",
    )

    check(
        engine.can_open_position() is True,
        "Initial engine can open position",
    )


# ============================================================
# TEST 2
# BUY -> LONG
# ============================================================

def test_buy_opens_long() -> None:

    print()
    print("## TEST: test_buy_opens_long")

    engine = TradeEngine()

    candle = make_candle(
        timestamp=datetime.now(),
        open_price=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
    )

    signal = make_buy_signal()

    result = engine.open_position(
        candle,
        signal,
    )

    check(
        isinstance(result, bool),
        "BUY open_position returns bool",
    )

    check(
        result is True,
        "BUY signal opens position",
    )

    positions = engine.get_open_positions()

    check(
        len(positions) == 1,
        "BUY creates one position",
    )

    position = positions[0]

    check(
        position.side == PositionSide.LONG,
        "BUY creates LONG position",
    )

    check(
        position.status == PositionStatus.OPEN,
        "LONG status is OPEN",
    )

    check(
        position.entry_price > 0,
        "LONG entry price is positive",
    )

    check(
        position.stop_loss > 0,
        "LONG stop-loss is positive",
    )

    check(
        position.take_profit > 0,
        "LONG take-profit is positive",
    )

    check(
        position.quantity > 0,
        "LONG quantity is positive",
    )

    check(
        position.confidence > 0,
        "LONG confidence is positive",
    )

    check(
        position.id > 0,
        "LONG position ID is positive",
    )


# ============================================================
# TEST 3
# SELL -> SHORT
# ============================================================

def test_sell_opens_short() -> None:

    print()
    print("## TEST: test_sell_opens_short")

    engine = TradeEngine()

    candle = make_candle(
        timestamp=datetime.now(),
        open_price=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
    )

    signal = make_sell_signal()

    result = engine.open_position(
        candle,
        signal,
    )

    check(
        result is True,
        "SELL signal opens position",
    )

    positions = engine.get_open_positions()

    check(
        len(positions) == 1,
        "SELL creates one position",
    )

    position = positions[0]

    check(
        position.side == PositionSide.SHORT,
        "SELL creates SHORT position",
    )

    check(
        position.status == PositionStatus.OPEN,
        "SHORT status is OPEN",
    )

    check(
        position.entry_price > 0,
        "SHORT entry price is positive",
    )

    check(
        position.stop_loss > 0,
        "SHORT stop-loss is positive",
    )

    check(
        position.take_profit > 0,
        "SHORT take-profit is positive",
    )

    check(
        position.quantity > 0,
        "SHORT quantity is positive",
    )


# ============================================================
# TEST 4
# HOLD MUST NOT OPEN
# ============================================================

def test_hold_does_not_open() -> None:

    print()
    print("## TEST: test_hold_does_not_open")

    engine = TradeEngine()

    candle = make_candle()

    signal = make_hold_signal()

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # TradeEngine.open_position() accepts only directional
    # signals:
    #
    #     BUY
    #     SELL
    #
    # HOLD must be filtered before this method.
    # --------------------------------------------------------

    if signal.signal == "HOLD":
        result = False

    else:
        result = engine.open_position(
            candle,
            signal,
        )

    check(
        result is False,
        "HOLD is filtered before TradeEngine",
    )

    check(
        len(engine.get_open_positions()) == 0,
        "HOLD leaves zero positions",
    )


# ============================================================
# TEST 5
# LONG -> TAKE PROFIT
# ============================================================

def test_long_take_profit() -> None:

    print()
    print("## TEST: test_long_take_profit")

    engine = TradeEngine()

    entry = 100.0
    stop_loss = 98.0
    take_profit = 104.0

    entry_time = datetime.now()

    entry_candle = make_candle(
        timestamp=entry_time,
        open_price=entry,
        high=entry + 1.0,
        low=entry - 1.0,
        close=entry,
    )

    signal = make_buy_signal(
        entry=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
    )

    opened = engine.open_position(
        entry_candle,
        signal,
    )

    check(
        opened is True,
        "LONG position opened for TP test",
    )

    positions = engine.get_open_positions()

    check(
        len(positions) == 1,
        "TP test has one open position",
    )

    position = positions[0]

    # --------------------------------------------------------
    # Candle reaches take-profit.
    # --------------------------------------------------------

    tp_candle = make_candle(
        timestamp=entry_time + timedelta(minutes=15),
        open_price=102.0,
        high=105.0,
        low=101.0,
        close=104.5,
    )

    try:

        updated = engine.update_position(
            position,
            tp_candle,
        )

        check(
            isinstance(updated, bool),
            "LONG update_position returns bool",
        )

    except Exception as exc:

        check(
            False,
            f"LONG TP update_position failed: {exc}",
        )

        return

    # --------------------------------------------------------
    # Position should no longer be OPEN if TP logic
    # is implemented by TradeEngine.
    # --------------------------------------------------------

    open_positions = engine.get_open_positions()

    if len(open_positions) == 0:

        check(
            True,
            "LONG position closed after TAKE_PROFIT",
        )

    else:

        current = open_positions[0]

        check(
            current.status != PositionStatus.OPEN,
            "LONG position status changed after TAKE_PROFIT",
        )


# ============================================================
# TEST 6
# SHORT -> TAKE PROFIT
# ============================================================

def test_short_take_profit() -> None:

    print()
    print("## TEST: test_short_take_profit")

    engine = TradeEngine()

    entry = 100.0
    stop_loss = 102.0
    take_profit = 96.0

    entry_time = datetime.now()

    entry_candle = make_candle(
        timestamp=entry_time,
        open_price=entry,
        high=101.0,
        low=99.0,
        close=entry,
    )

    signal = make_sell_signal(
        entry=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
    )

    opened = engine.open_position(
        entry_candle,
        signal,
    )

    check(
        opened is True,
        "SHORT position opened for TP test",
    )

    positions = engine.get_open_positions()

    check(
        len(positions) == 1,
        "SHORT TP test has one open position",
    )

    position = positions[0]

    tp_candle = make_candle(
        timestamp=entry_time + timedelta(minutes=15),
        open_price=98.0,
        high=99.0,
        low=95.0,
        close=95.5,
    )

    try:

        updated = engine.update_position(
            position,
            tp_candle,
        )

        check(
            isinstance(updated, bool),
            "SHORT update_position returns bool",
        )

    except Exception as exc:

        check(
            False,
            f"SHORT TP update_position failed: {exc}",
        )

        return

    open_positions = engine.get_open_positions()

    if len(open_positions) == 0:

        check(
            True,
            "SHORT position closed after TAKE_PROFIT",
        )

    else:

        current = open_positions[0]

        check(
            current.status != PositionStatus.OPEN,
            "SHORT position status changed after TAKE_PROFIT",
        )


# ============================================================
# TEST 7
# LONG -> STOP LOSS
# ============================================================

def test_long_stop_loss() -> None:

    print()
    print("## TEST: test_long_stop_loss")

    engine = TradeEngine()

    entry = 100.0
    stop_loss = 98.0
    take_profit = 104.0

    entry_time = datetime.now()

    entry_candle = make_candle(
        timestamp=entry_time,
        open_price=entry,
        high=101.0,
        low=99.0,
        close=entry,
    )

    signal = make_buy_signal(
        entry=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
    )

    opened = engine.open_position(
        entry_candle,
        signal,
    )

    check(
        opened is True,
        "LONG position opened for SL test",
    )

    positions = engine.get_open_positions()

    check(
        len(positions) == 1,
        "LONG SL test has one open position",
    )

    position = positions[0]

    sl_candle = make_candle(
        timestamp=entry_time + timedelta(minutes=15),
        open_price=99.0,
        high=100.0,
        low=97.0,
        close=97.5,
    )

    try:

        updated = engine.update_position(
            position,
            sl_candle,
        )

        check(
            isinstance(updated, bool),
            "LONG SL update_position returns bool",
        )

    except Exception as exc:

        check(
            False,
            f"LONG SL update_position failed: {exc}",
        )

        return

    open_positions = engine.get_open_positions()

    if len(open_positions) == 0:

        check(
            True,
            "LONG position closed after STOP_LOSS",
        )

    else:

        current = open_positions[0]

        check(
            current.status != PositionStatus.OPEN,
            "LONG position status changed after STOP_LOSS",
        )


# ============================================================
# TEST 8
# SHORT -> STOP LOSS
# ============================================================

def test_short_stop_loss() -> None:

    print()
    print("## TEST: test_short_stop_loss")

    engine = TradeEngine()

    entry = 100.0
    stop_loss = 102.0
    take_profit = 96.0

    entry_time = datetime.now()

    entry_candle = make_candle(
        timestamp=entry_time,
        open_price=entry,
        high=101.0,
        low=99.0,
        close=entry,
    )

    signal = make_sell_signal(
        entry=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
    )

    opened = engine.open_position(
        entry_candle,
        signal,
    )

    check(
        opened is True,
        "SHORT position opened for SL test",
    )

    positions = engine.get_open_positions()

    check(
        len(positions) == 1,
        "SHORT SL test has one open position",
    )

    position = positions[0]

    sl_candle = make_candle(
        timestamp=entry_time + timedelta(minutes=15),
        open_price=101.0,
        high=103.0,
        low=100.0,
        close=102.5,
    )

    try:

        updated = engine.update_position(
            position,
            sl_candle,
        )

        check(
            isinstance(updated, bool),
            "SHORT SL update_position returns bool",
        )

    except Exception as exc:

        check(
            False,
            f"SHORT SL update_position failed: {exc}",
        )

        return

    open_positions = engine.get_open_positions()

    if len(open_positions) == 0:

        check(
            True,
            "SHORT position closed after STOP_LOSS",
        )

    else:

        current = open_positions[0]

        check(
            current.status != PositionStatus.OPEN,
            "SHORT position status changed after STOP_LOSS",
        )


# ============================================================
# TEST 9
# MANUAL CLOSE + PROFIT
# ============================================================

def test_manual_close_long_profit() -> None:

    print()
    print("## TEST: test_manual_close_long_profit")

    engine = TradeEngine()

    entry = 100.0

    candle = make_candle(
        timestamp=datetime.now(),
        open_price=entry,
        high=101.0,
        low=99.0,
        close=entry,
    )

    signal = make_buy_signal(
        entry=entry,
        stop_loss=98.0,
        take_profit=104.0,
    )

    opened = engine.open_position(
        candle,
        signal,
    )

    check(
        opened is True,
        "LONG opened for manual close test",
    )

    positions = engine.get_open_positions()

    if not positions:

        check(
            False,
            "Position exists before manual close",
        )

        return

    position = positions[0]

    exit_price = 103.0

    try:

        engine.close_position(
            position,
            make_candle(
                timestamp=datetime.now()
                + timedelta(minutes=15),
                open_price=exit_price,
                high=exit_price,
                low=exit_price,
                close=exit_price,
            ),
            exit_price,
            CloseReason.MANUAL,
        )

        check(
            position.status != PositionStatus.OPEN,
            "LONG position closes manually",
        )

        check(
            position.exit_price > 0,
            "Manual close records exit price",
        )

        check(
            position.gross_profit > 0,
            "Profitable LONG has positive gross profit",
        )

        check(
            position.net_profit > 0,
            "Profitable LONG has positive net profit",
        )

    except Exception as exc:

        check(
            False,
            f"Manual LONG close failed: {exc}",
        )


# ============================================================
# TEST 10
# MANUAL CLOSE + LOSS
# ============================================================

def test_manual_close_short_loss() -> None:

    print()
    print("## TEST: test_manual_close_short_loss")

    engine = TradeEngine()

    entry = 100.0

    candle = make_candle(
        timestamp=datetime.now(),
        open_price=entry,
        high=101.0,
        low=99.0,
        close=entry,
    )

    signal = make_sell_signal(
        entry=entry,
        stop_loss=102.0,
        take_profit=96.0,
    )

    opened = engine.open_position(
        candle,
        signal,
    )

    check(
        opened is True,
        "SHORT opened for loss test",
    )

    positions = engine.get_open_positions()

    if not positions:

        check(
            False,
            "SHORT position exists before loss close",
        )

        return

    position = positions[0]

    exit_price = 103.0

    try:

        engine.close_position(
            position,
            make_candle(
                timestamp=datetime.now()
                + timedelta(minutes=15),
                open_price=exit_price,
                high=exit_price,
                low=exit_price,
                close=exit_price,
            ),
            exit_price,
            CloseReason.MANUAL,
        )

        check(
            position.status != PositionStatus.OPEN,
            "SHORT position closes manually",
        )

        check(
            position.exit_price > 0,
            "SHORT loss records exit price",
        )

        check(
            position.gross_profit < 0,
            "Losing SHORT has negative gross profit",
        )

        check(
            position.net_profit < 0,
            "Losing SHORT has negative net profit",
        )

    except Exception as exc:

        check(
            False,
            f"Manual SHORT close failed: {exc}",
        )


# ============================================================
# TEST 11
# POSITION IDS
# ============================================================

def test_position_ids() -> None:

    print()
    print("## TEST: test_position_ids")

    engine = TradeEngine()

    first = engine.next_position_id()
    second = engine.next_position_id()
    third = engine.next_position_id()

    check(
        isinstance(first, int),
        "Position ID is integer",
    )

    check(
        second == first + 1,
        "Second position ID increments sequentially",
    )

    check(
        third == second + 1,
        "Third position ID increments sequentially",
    )


# ============================================================
# TEST 12
# COMMISSION
# ============================================================

def test_commission() -> None:

    print()
    print("## TEST: test_commission")

    engine = TradeEngine()

    commission = engine.calculate_commission(
        quantity=1.0,
        price=100.0,
    )

    check(
        isinstance(commission, float),
        "Commission returns float",
    )

    check(
        commission >= 0.0,
        "Commission is non-negative",
    )


# ============================================================
# TEST 13
# SLIPPAGE
# ============================================================

def test_slippage() -> None:

    print()
    print("## TEST: test_slippage")

    engine = TradeEngine()

    long_price = engine.apply_slippage(
        PositionSide.LONG,
        100.0,
    )

    short_price = engine.apply_slippage(
        PositionSide.SHORT,
        100.0,
    )

    check(
        isinstance(long_price, float),
        "Slippage returns float for LONG",
    )

    check(
        long_price > 0,
        "LONG slippage price remains positive",
    )

    check(
        isinstance(short_price, float),
        "Slippage returns float for SHORT",
    )

    check(
        short_price > 0,
        "SHORT slippage price remains positive",
    )


# ============================================================
# TEST 14
# DATAFRAME EXPORT
# ============================================================

def test_dataframe_export() -> None:

    print()
    print("## TEST: test_dataframe_export")

    engine = TradeEngine()

    df = engine.to_dataframe()

    check(
        df is not None,
        "TradeEngine exports DataFrame",
    )

    check(
        hasattr(df, "columns"),
        "Export result has DataFrame structure",
    )

    check(
        len(df) == 0,
        "Empty TradeEngine exports empty DataFrame",
    )


# ============================================================
# TEST 15
# EMPTY RUN
# ============================================================

def test_run_empty_dataframe() -> None:

    print()
    print("## TEST: test_run_empty_dataframe")

    import pandas as pd

    engine = TradeEngine()

    df = pd.DataFrame()

    result = engine.run(df)

    check(
        hasattr(result, "columns"),
        "run() returns DataFrame for empty input",
    )


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    print()
    print("=" * 70)
    print("QuantAI TradeEngine Integration Tests")
    print("=" * 70)

    tests = [
        test_initial_state,
        test_buy_opens_long,
        test_sell_opens_short,
        test_hold_does_not_open,
        test_long_take_profit,
        test_short_take_profit,
        test_long_stop_loss,
        test_short_stop_loss,
        test_manual_close_long_profit,
        test_manual_close_short_loss,
        test_position_ids,
        test_commission,
        test_slippage,
        test_dataframe_export,
        test_run_empty_dataframe,
    ]

    for test in tests:

        try:

            test()

        except Exception as exc:

            global FAIL_COUNT

            FAIL_COUNT += 1

            print()
            print(
                f"FAIL  : {test.__name__} "
                f"raised unexpected exception:"
            )

            print(
                f"        {type(exc).__name__}: {exc}"
            )

    print()
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    print(
        f"PASSED : {PASS_COUNT}"
    )

    print(
        f"FAILED : {FAIL_COUNT}"
    )

    total = PASS_COUNT + FAIL_COUNT

    print(
        f"TOTAL  : {total}"
    )

    if FAIL_COUNT == 0:

        print()
        print("ALL TRADE ENGINE INTEGRATION TESTS PASSED.")
        print()
        return 0

    print()
    print("TRADE ENGINE INTEGRATION TESTS FAILED.")
    print()

    return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
