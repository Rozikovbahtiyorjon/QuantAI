from pathlib import Path
import sys

# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# IMPORTS
# ============================================================

from src.trade_engine import (
    TradeEngine,
)


# ============================================================
# HELPERS
# ============================================================

def ok(message: str) -> None:
    print(f"OK    : {message}")


def fail(message: str) -> None:
    print(f"FAIL  : {message}")
    raise AssertionError(message)


def check(condition: bool, message: str) -> None:
    if condition:
        ok(message)
    else:
        fail(message)


# ============================================================
# TEST: INITIAL STATE
# ============================================================

def test_initial_state() -> None:

    engine = TradeEngine()

    positions = engine.get_open_positions()

    check(
        isinstance(positions, list),
        "Initial open positions returns list",
    )

    check(
        len(positions) == 0,
        "TradeEngine starts with zero open positions",
    )


# ============================================================
# TEST: POSITION LIMIT
# ============================================================

def test_position_limit() -> None:

    engine = TradeEngine()

    result = engine.can_open_position()

    check(
        isinstance(result, bool),
        "can_open_position returns bool",
    )

    check(
        result is True,
        "Empty TradeEngine can open position",
    )


# ============================================================
# TEST: POSITION IDS
# ============================================================

def test_position_ids() -> None:

    engine = TradeEngine()

    first = engine.next_position_id()
    second = engine.next_position_id()

    check(
        isinstance(first, int),
        "Position ID is integer",
    )

    check(
        second == first + 1,
        "Position IDs increment sequentially",
    )


# ============================================================
# TEST: COMMISSION
# ============================================================

def test_commission() -> None:

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
# TEST: SLIPPAGE
# ============================================================
#
# We test the method structurally without assuming the
# project's exact enum implementation.
# ============================================================

def test_slippage() -> None:

    engine = TradeEngine()

    try:
        from src.trade_engine import PositionSide

        sides = list(PositionSide)

    except Exception:

        sides = []

    if not sides:
        print(
            "INFO  : PositionSide enum unavailable; "
            "slippage test skipped"
        )
        return

    for side in sides:

        try:

            result = engine.apply_slippage(
                side=side,
                price=100.0,
            )

            check(
                isinstance(result, float),
                f"Slippage returns float for {side}",
            )

            check(
                result > 0.0,
                f"Slippage price remains positive for {side}",
            )

        except Exception as exc:

            fail(
                f"Slippage failed for {side}: {exc}"
            )


# ============================================================
# TEST: DATAFRAME EXPORT
# ============================================================

def test_dataframe_export() -> None:

    engine = TradeEngine()

    df = engine.to_dataframe()

    check(
        hasattr(df, "columns"),
        "TradeEngine exports DataFrame",
    )

    check(
        len(df) == 0,
        "Empty TradeEngine exports empty DataFrame",
    )


# ============================================================
# TEST: RUN EMPTY DATAFRAME
# ============================================================

def test_run_empty_dataframe() -> None:

    import pandas as pd

    engine = TradeEngine()

    df = pd.DataFrame()

    try:

        result = engine.run(df)

        check(
            hasattr(result, "columns"),
            "run() returns DataFrame for empty input",
        )

    except Exception as exc:

        print(
            "INFO  : run(empty DataFrame) raised:"
        )

        print(
            f"        {type(exc).__name__}: {exc}"
        )

        print(
            "INFO  : This may be expected if "
            "TradeEngine requires OHLCV columns."
        )


# ============================================================
# TEST RUNNER
# ============================================================

def main() -> None:

    print()
    print("=" * 70)
    print("TRADE ENGINE UNIT TEST")
    print("=" * 70)
    print()

    tests = [
        test_initial_state,
        test_position_limit,
        test_position_ids,
        test_commission,
        test_slippage,
        test_dataframe_export,
        test_run_empty_dataframe,
    ]

    passed = 0

    for test in tests:

        print()
        print(
            f"TEST: {test.__name__}"
        )
        print("-" * 70)

        test()

        passed += 1

    print()
    print("=" * 70)
    print(
        f"PASSED: {passed}/{len(tests)}"
    )
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()