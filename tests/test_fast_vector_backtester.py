from __future__ import annotations

import pandas as pd
import pytest

from src.fast_vector_backtester import (
    FastVectorBacktestResult,
    FastVectorBacktester,
)


def test_initialization_defaults() -> None:
    backtester = FastVectorBacktester()

    assert backtester.initial_balance == 1000.0
    assert backtester.commission == 0.0004
    assert backtester.quantity == 1.0
    assert backtester.result is None


def test_initialization_custom_values() -> None:
    backtester = FastVectorBacktester(
        initial_balance=5000.0,
        commission=0.001,
        quantity=2.0,
    )

    assert backtester.initial_balance == 5000.0
    assert backtester.commission == 0.001
    assert backtester.quantity == 2.0


@pytest.mark.parametrize(
    "initial_balance",
    [0.0, -1.0],
)
def test_invalid_initial_balance(
    initial_balance: float,
) -> None:
    with pytest.raises(ValueError):
        FastVectorBacktester(
            initial_balance=initial_balance
        )


def test_negative_commission_rejected() -> None:
    with pytest.raises(ValueError):
        FastVectorBacktester(
            commission=-0.001
        )


@pytest.mark.parametrize(
    "quantity",
    [0.0, -1.0],
)
def test_invalid_quantity(
    quantity: float,
) -> None:
    with pytest.raises(ValueError):
        FastVectorBacktester(
            quantity=quantity
        )


def test_non_dataframe_rejected() -> None:
    backtester = FastVectorBacktester()

    with pytest.raises(TypeError):
        backtester.run(
            [100.0, 101.0]  # type: ignore[arg-type]
        )


def test_empty_dataframe_rejected() -> None:
    backtester = FastVectorBacktester()

    with pytest.raises(ValueError):
        backtester.run(
            pd.DataFrame()
        )


def test_missing_close_column_rejected() -> None:
    backtester = FastVectorBacktester()

    df = pd.DataFrame(
        {
            "signal": ["BUY", "CLOSE"],
        }
    )

    with pytest.raises(ValueError):
        backtester.run(df)


def test_missing_signal_column_rejected() -> None:
    backtester = FastVectorBacktester()

    df = pd.DataFrame(
        {
            "close": [100.0, 110.0],
        }
    )

    with pytest.raises(ValueError):
        backtester.run(df)


def test_invalid_close_values_rejected() -> None:
    backtester = FastVectorBacktester()

    df = pd.DataFrame(
        {
            "close": [100.0, "invalid"],
            "signal": ["BUY", "CLOSE"],
        }
    )

    with pytest.raises(ValueError):
        backtester.run(df)


@pytest.mark.parametrize(
    "price",
    [0.0, -1.0],
)
def test_non_positive_close_rejected(
    price: float,
) -> None:
    backtester = FastVectorBacktester()

    df = pd.DataFrame(
        {
            "close": [100.0, price],
            "signal": ["BUY", "CLOSE"],
        }
    )

    with pytest.raises(ValueError):
        backtester.run(df)


def test_long_trade_with_profit() -> None:
    backtester = FastVectorBacktester(
        initial_balance=1000.0,
        commission=0.0,
        quantity=1.0,
    )

    df = pd.DataFrame(
        {
            "close": [100.0, 110.0],
            "signal": ["BUY", "CLOSE"],
        }
    )

    result = backtester.run(df)

    assert isinstance(
        result,
        FastVectorBacktestResult,
    )
    assert result.initial_balance == 1000.0
    assert result.final_balance == 1010.0
    assert result.total_profit == 10.0
    assert result.total_return == 1.0
    assert result.total_trades == 1
    assert result.winning_trades == 1
    assert result.losing_trades == 0
    assert result.win_rate == 100.0


def test_long_trade_with_loss() -> None:
    backtester = FastVectorBacktester(
        initial_balance=1000.0,
        commission=0.0,
        quantity=1.0,
    )

    df = pd.DataFrame(
        {
            "close": [100.0, 90.0],
            "signal": ["BUY", "CLOSE"],
        }
    )

    result = backtester.run(df)

    assert result.final_balance == 990.0
    assert result.total_profit == -10.0
    assert result.total_return == -1.0
    assert result.total_trades == 1
    assert result.winning_trades == 0
    assert result.losing_trades == 1
    assert result.win_rate == 0.0


def test_short_trade_with_profit() -> None:
    backtester = FastVectorBacktester(
        initial_balance=1000.0,
        commission=0.0,
        quantity=1.0,
    )

    df = pd.DataFrame(
        {
            "close": [100.0, 90.0],
            "signal": ["SELL", "CLOSE"],
        }
    )

    result = backtester.run(df)

    assert result.final_balance == 1010.0
    assert result.total_profit == 10.0
    assert result.total_return == 1.0
    assert result.total_trades == 1
    assert result.winning_trades == 1
    assert result.losing_trades == 0
    assert result.win_rate == 100.0


def test_short_trade_with_loss() -> None:
    backtester = FastVectorBacktester(
        initial_balance=1000.0,
        commission=0.0,
        quantity=1.0,
    )

    df = pd.DataFrame(
        {
            "close": [100.0, 110.0],
            "signal": ["SELL", "CLOSE"],
        }
    )

    result = backtester.run(df)

    assert result.final_balance == 990.0
    assert result.total_profit == -10.0
    assert result.total_return == -1.0
    assert result.total_trades == 1
    assert result.winning_trades == 0
    assert result.losing_trades == 1
    assert result.win_rate == 0.0


def test_long_trade_with_commission() -> None:
    backtester = FastVectorBacktester(
        initial_balance=1000.0,
        commission=0.001,
        quantity=1.0,
    )

    df = pd.DataFrame(
        {
            "close": [100.0, 110.0],
            "signal": ["BUY", "CLOSE"],
        }
    )

    result = backtester.run(df)

    expected_profit = 10.0 - (
        (100.0 + 110.0) * 0.001
    )

    assert result.total_profit == pytest.approx(
        expected_profit
    )
    assert result.final_balance == pytest.approx(
        1000.0 + expected_profit
    )


def test_reverse_long_to_short() -> None:
    backtester = FastVectorBacktester(
        initial_balance=1000.0,
        commission=0.0,
        quantity=1.0,
    )

    df = pd.DataFrame(
        {
            "close": [100.0, 110.0, 100.0],
            "signal": [
                "BUY",
                "SELL",
                "CLOSE",
            ],
        }
    )

    result = backtester.run(df)

    assert result.total_trades == 2
    assert result.winning_trades == 2
    assert result.losing_trades == 0
    assert result.total_profit == 20.0
    assert result.final_balance == 1020.0
    assert result.win_rate == 100.0


def test_multiple_trades() -> None:
    backtester = FastVectorBacktester(
        initial_balance=1000.0,
        commission=0.0,
        quantity=1.0,
    )

    df = pd.DataFrame(
        {
            "close": [
                100.0,
                110.0,
                100.0,
                90.0,
            ],
            "signal": [
                "BUY",
                "CLOSE",
                "SELL",
                "CLOSE",
            ],
        }
    )

    result = backtester.run(df)

    assert result.total_trades == 2
    assert result.winning_trades == 2
    assert result.losing_trades == 0
    assert result.total_profit == 20.0
    assert result.final_balance == 1020.0
    assert result.win_rate == 100.0


def test_open_position_is_closed_at_end() -> None:
    backtester = FastVectorBacktester(
        initial_balance=1000.0,
        commission=0.0,
        quantity=1.0,
    )

    df = pd.DataFrame(
        {
            "close": [100.0, 110.0],
            "signal": ["BUY", "HOLD"],
        }
    )

    result = backtester.run(df)

    assert result.total_trades == 1
    assert result.winning_trades == 1
    assert result.losing_trades == 0
    assert result.final_balance == 1010.0
    assert result.total_profit == 10.0


def test_no_trades() -> None:
    backtester = FastVectorBacktester(
        initial_balance=1000.0,
        commission=0.0,
        quantity=1.0,
    )

    df = pd.DataFrame(
        {
            "close": [100.0, 105.0, 103.0],
            "signal": [
                "HOLD",
                "HOLD",
                "HOLD",
            ],
        }
    )

    result = backtester.run(df)

    assert result.final_balance == 1000.0
    assert result.total_profit == 0.0
    assert result.total_return == 0.0
    assert result.total_trades == 0
    assert result.winning_trades == 0
    assert result.losing_trades == 0
    assert result.win_rate == 0.0


def test_result_property() -> None:
    backtester = FastVectorBacktester(
        initial_balance=1000.0,
        commission=0.0,
        quantity=1.0,
    )

    df = pd.DataFrame(
        {
            "close": [100.0, 110.0],
            "signal": ["BUY", "CLOSE"],
        }
    )

    result = backtester.run(df)

    assert backtester.result is result


def test_reset_clears_result() -> None:
    backtester = FastVectorBacktester(
        initial_balance=1000.0,
        commission=0.0,
        quantity=1.0,
    )

    df = pd.DataFrame(
        {
            "close": [100.0, 110.0],
            "signal": ["BUY", "CLOSE"],
        }
    )

    backtester.run(df)

    assert backtester.result is not None

    backtester.reset()

    assert backtester.result is None


def test_equity_curve_contains_initial_balance() -> None:
    backtester = FastVectorBacktester(
        initial_balance=1000.0,
        commission=0.0,
        quantity=1.0,
    )

    df = pd.DataFrame(
        {
            "close": [100.0, 110.0],
            "signal": ["BUY", "CLOSE"],
        }
    )

    result = backtester.run(df)

    assert result.equity_curve[0] == 1000.0
    assert result.equity_curve[-1] == 1010.0


def test_signal_case_is_ignored() -> None:
    backtester = FastVectorBacktester(
        initial_balance=1000.0,
        commission=0.0,
        quantity=1.0,
    )

    df = pd.DataFrame(
        {
            "close": [100.0, 110.0],
            "signal": ["buy", "close"],
        }
    )

    result = backtester.run(df)

    assert result.total_trades == 1
    assert result.total_profit == 10.0
    assert result.final_balance == 1010.0


def test_quantity_affects_profit() -> None:
    backtester = FastVectorBacktester(
        initial_balance=1000.0,
        commission=0.0,
        quantity=2.0,
    )

    df = pd.DataFrame(
        {
            "close": [100.0, 110.0],
            "signal": ["BUY", "CLOSE"],
        }
    )

    result = backtester.run(df)

    assert result.total_profit == 20.0
    assert result.final_balance == 1020.0


def test_result_is_deterministic() -> None:
    backtester = FastVectorBacktester(
        initial_balance=1000.0,
        commission=0.0004,
        quantity=1.0,
    )

    df = pd.DataFrame(
        {
            "close": [
                100.0,
                105.0,
                102.0,
                108.0,
            ],
            "signal": [
                "BUY",
                "CLOSE",
                "SELL",
                "CLOSE",
            ],
        }
    )

    first = backtester.run(df)

    backtester.reset()

    second = backtester.run(df)

    assert first == second