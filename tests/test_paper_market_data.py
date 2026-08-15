import pandas as pd
import pytest

from src.paper_market_data import PaperMarketData


def create_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1000.0, 1100.0, 1200.0],
        }
    )


def test_initial_state() -> None:
    provider = PaperMarketData(
        create_data()
    )

    assert provider.position == 0
    assert provider.total_rows == 3
    assert provider.finished is False


def test_next_returns_first_row() -> None:
    provider = PaperMarketData(
        create_data()
    )

    row = provider.next()

    assert isinstance(row, pd.Series)
    assert row["close"] == 100.5
    assert provider.position == 1


def test_next_consumes_rows_sequentially() -> None:
    provider = PaperMarketData(
        create_data()
    )

    first = provider.next()
    second = provider.next()
    third = provider.next()

    assert first["close"] == 100.5
    assert second["close"] == 101.5
    assert third["close"] == 102.5

    assert provider.position == 3
    assert provider.finished is True


def test_next_after_finished_raises() -> None:
    provider = PaperMarketData(
        create_data()
    )

    provider.next()
    provider.next()
    provider.next()

    with pytest.raises(StopIteration):
        provider.next()


def test_iterator_consumes_all_rows() -> None:
    provider = PaperMarketData(
        create_data()
    )

    rows = list(provider)

    assert len(rows) == 3
    assert rows[0]["close"] == 100.5
    assert rows[1]["close"] == 101.5
    assert rows[2]["close"] == 102.5
    assert provider.finished is True


def test_reset() -> None:
    provider = PaperMarketData(
        create_data()
    )

    provider.next()
    provider.next()

    assert provider.position == 2

    provider.reset()

    assert provider.position == 0
    assert provider.finished is False

    row = provider.next()

    assert row["close"] == 100.5


def test_data_is_copied() -> None:
    data = create_data()
    provider = PaperMarketData(data)

    data.loc[0, "close"] = 999.0

    assert provider.data.loc[0, "close"] == 100.5


def test_index_is_reset() -> None:
    data = create_data()
    data.index = [10, 20, 30]

    provider = PaperMarketData(data)

    assert list(provider.data.index) == [0, 1, 2]


def test_invalid_data_type() -> None:
    with pytest.raises(TypeError):
        PaperMarketData([1, 2, 3])


def test_empty_dataframe() -> None:
    with pytest.raises(ValueError):
        PaperMarketData(
            pd.DataFrame()
        )